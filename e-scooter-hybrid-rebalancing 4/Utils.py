import numpy as np
import pandas as pd
import gurobipy as gp

def varX(x):
    """
    Safe numeric extract from a Gurobi Var or raw number.
    Returns float value of x.X if it's a Gurobi Var, else float(x).
    """
    try:
        return float(x.X)
    except AttributeError:
        try:
            return float(x)
        except Exception:
            return 0.0
    except Exception:
        return 0.0

def values_dict(d):
    """
    Convert a dict of {key: GurobiVar} -> {key: float_value}.
    """
    return {k: varX(v) for k, v in d.items()}

def sum_over_vars(iterable):
    """
    Sum over an iterable where items can be:
      - (coefficient, variable/value) tuple
      - variable/value
    """
    total = 0.0
    for item in iterable:
        if isinstance(item, tuple) and len(item) == 2:
            coeff, var = item
            total += coeff * varX(var)
        else:
            total += varX(item)
    return total

def nearest_enumerated_state(target_tuple, states_list):
    """
    Find the state in states_list closest to target_tuple (L1 distance).
    states_list: list or array of tuples/arrays.
    target_tuple: tuple or array.
    """
    S = np.asarray(states_list, dtype=int)
    t = np.array(target_tuple, dtype=int)
    # L1 distance
    dist = np.sum(np.abs(S - t), axis=1)
    idx = int(np.argmin(dist))
    return tuple(S[idx])


def build_init_inventory_by_system_shares(
    N, all_states, C_i, total_fleet, shares,
    weights=None
):
    """
    Distribute `total_fleet` e-scooters across stations with system-wide
    shares (Sn, Sl, Sh) on (no-power, low-power, high-power).

    Parameters
    ----------
    N : list of station IDs
    all_states : dict {i: list of feasible (n,l,h) tuples for station i}
    C_i : dict {i: capacity}
    total_fleet : int, system-wide fleet size to allocate
    shares : tuple (Sn, Sl, Sh), must sum to 1.0
    weights : optional dict {i: float}; spatial weight for distributing fleet.
              If None, uniform per station.

    Returns
    -------
    Init_inventory : dict {i: (n,l,h)} snapped to nearest enumerated state
    achieved : dict with realized totals and shares (may differ slightly
               from targets due to capacity and state-enumeration snapping)
    """
    Sn, Sl, Sh = shares
    assert abs(Sn + Sl + Sh - 1.0) < 1e-6, "shares must sum to 1"

    N_total_count = int(round(Sn * total_fleet))
    L_total_count = int(round(Sl * total_fleet))
    H_total_count = total_fleet - N_total_count - L_total_count

    if weights is None:
        w = {i: 1.0 for i in N}
    else:
        w = {i: float(weights.get(i, 0.0)) for i in N}
    w_sum = sum(w.values()) or 1.0
    w_norm = {i: w[i] / w_sum for i in N}

    def _apportion(total, weights_norm, caps_remaining):
        """Largest-remainder apportionment respecting per-station caps."""
        frac = {i: weights_norm[i] * total for i in N}
        floor_v = {i: min(int(np.floor(frac[i])), caps_remaining[i]) for i in N}
        remaining = total - sum(floor_v.values())
        # Order stations by descending fractional remainder
        order = sorted(N, key=lambda i: (frac[i] - np.floor(frac[i])), reverse=True)
        out = dict(floor_v)
        for i in order:
            if remaining <= 0:
                break
            if out[i] < caps_remaining[i]:
                out[i] += 1
                remaining -= 1
        return out

    cap_remaining = dict(C_i)
    n_alloc = _apportion(N_total_count, w_norm, cap_remaining)
    for i in N:
        cap_remaining[i] -= n_alloc[i]
    l_alloc = _apportion(L_total_count, w_norm, cap_remaining)
    for i in N:
        cap_remaining[i] -= l_alloc[i]
    h_alloc = _apportion(H_total_count, w_norm, cap_remaining)

    # Snap each (n,l,h) target to nearest enumerated Markov state
    Init_inventory = {}
    for i in N:
        target = (n_alloc[i], l_alloc[i], h_alloc[i])
        Init_inventory[i] = nearest_enumerated_state(target, all_states[i])

    achieved_n = sum(Init_inventory[i][0] for i in N)
    achieved_l = sum(Init_inventory[i][1] for i in N)
    achieved_h = sum(Init_inventory[i][2] for i in N)
    achieved_total = achieved_n + achieved_l + achieved_h

    achieved = {
        "fleet_target": total_fleet,
        "fleet_achieved": achieved_total,
        "n_share_target": Sn,
        "l_share_target": Sl,
        "h_share_target": Sh,
        "n_share_achieved": achieved_n / max(achieved_total, 1),
        "l_share_achieved": achieved_l / max(achieved_total, 1),
        "h_share_achieved": achieved_h / max(achieved_total, 1),
    }
    return Init_inventory, achieved

def choose_state_with_tolerance(target_tuple, states_list, cap, tol_frac, rng):
    """
    Apply component-wise uniform ±tol to target, clip to capacity, snap to nearest enumerated state.
    """
    n0, l0, h0 = map(float, target_tuple)
    # jitter each component uniformly in [1-tol, 1+tol]
    if tol_frac > 0:
        n0 *= rng.uniform(1 - tol_frac, 1 + tol_frac)
        l0 *= rng.uniform(1 - tol_frac, 1 + tol_frac)
        h0 *= rng.uniform(1 - tol_frac, 1 + tol_frac)
    
    # round & enforce capacity by trimming in n->l->h order (or any policy you prefer)
    n, l, h = [int(max(0, round(x))) for x in (n0, l0, h0)]
    tot = n + l + h
    if tot > cap:
        overflow = tot - cap
        k = min(overflow, n);  n -= k; overflow -= k
        k = min(overflow, l);  l -= k; overflow -= k
        if overflow:           h -= overflow

    # if already feasible enumerated, done
    s = (n, l, h)
    Sset = set(states_list)
    if s in Sset:
        return s

    return nearest_enumerated_state(s, states_list)

def reseed_initial_low_share(
    N,
    Init_inventory,             # dict: i -> (n,l,h)
    all_states,                 # dict: i -> list of (n,l,h) feasible states
    C_i,                        # dict: i -> capacity
    low_share,                  # desired system share in [0,1]
    keep_total_per_station=True,
    freeze_no=True,             # keep 'no-power' counts fixed per station
    weights_mode="totals",      # "totals" or "demand" or a dict {i: weight}
    fn_pickup_rates_by_hour=None,      # needed if weights_mode="demand"
    fn_dropoff_rates_by_hour=None,     # needed if weights_mode="demand"
    verbose=False
):
    """
    Returns: new_Init_inventory (dict), summary (dict with achieved shares)
    Enforces: sum_i l_i ≈ round(low_share * sum_i (n_i + l_i + h_i))
    """
    assert 0.0 <= low_share <= 1.0, "low_share must be in [0,1]"

    # 0) basic vectors
    n0 = {i: int(Init_inventory[i][0]) for i in N}
    l0 = {i: int(Init_inventory[i][1]) for i in N}
    h0 = {i: int(Init_inventory[i][2]) for i in N}
    tot0 = {i: n0[i] + l0[i] + h0[i] for i in N}
    S_tot = int(sum(tot0.values()))
    L_target = int(round(low_share * S_tot))

    # 1) per-station weights for distributing LOWs
    if isinstance(weights_mode, dict):
        w = {i: float(weights_mode.get(i, 0.0)) for i in N}
    elif weights_mode == "totals":
        w = {i: float(tot0[i]) for i in N}
    elif weights_mode == "demand":
        # demand weight ~ daily pickups + dropoffs (total, not split)
        if fn_pickup_rates_by_hour is None or fn_dropoff_rates_by_hour is None:
            raise ValueError("weights_mode='demand' requires fn_*_rates_by_hour")
        w = {}
        for i in N:
            # pickups: {hour: value}
            p = sum(float(v) for v in fn_pickup_rates_by_hour.get(i, {}).values())
            # dropoffs: {hour: np.array([no, low, high])}
            d = sum(float(np.asarray(v, float).sum()) for v in fn_dropoff_rates_by_hour.get(i, {}).values())
            w[i] = p + d
    else:
        raise ValueError("weights_mode must be 'totals', 'demand', or a dict")

    # normalize weights (and guard zeros)
    w_sum = sum(w.values())
    if w_sum <= 0:
        w = {i: 1.0 for i in N}
        w_sum = sum(w.values())
    w_norm = {i: w[i] / w_sum for i in N}

    # 2) per-station maximum LOW feasible if we keep totals (so H >= 0)
    if keep_total_per_station:
        if freeze_no:
            l_max = {i: max(0, tot0[i] - n0[i]) for i in N}
            n_new = n0.copy()
        else:
            # If 'no' can move, cap LOW by capacity & total (we'll keep totals though)
            l_max = {i: min(tot0[i], C_i[i]) for i in N}  # practically tot0[i]
            n_new = n0.copy()  # totals fixed; 'n' will change only when snapping states
        tot_per_station = tot0.copy()
    else:
        # Not typical; allowing totals to change complicates redistribution. Keep simple:
        raise NotImplementedError("Set keep_total_per_station=True for now (recommended).")

    # 3) initial fractional LOW targets and integerize with caps
    l_frac = {i: w_norm[i] * L_target for i in N}
    l_floor = {i: min(int(np.floor(l_frac[i])), l_max[i]) for i in N}
    L_now = sum(l_floor.values())
    remain = L_target - L_now

    # remainders for largest remainder rule
    rema = {i: (l_frac[i] - np.floor(l_frac[i])) if l_max[i] > l_floor[i] else -1.0 for i in N}
    order = sorted(N, key=lambda i: rema[i], reverse=True)

    l_new = l_floor.copy()
    # distribute remaining LOW one-by-one to stations with slack and highest remainder
    for i in order:
        if remain <= 0:
            break
        slack = l_max[i] - l_new[i]
        if slack > 0:
            take = min(1, slack)
            l_new[i] += take
            remain -= take

    # If still remain > 0, we cannot hit the system target due to per-station constraints
    unmet = max(0, remain)

    # 4) derive HIGH so totals stay unchanged, then snap to enumerated states
    new_Init = {}
    for i in N:
        l_i = int(l_new[i])
        # keep totals; compute h so that n + l + h == tot
        h_i = max(0, tot_per_station[i] - n_new[i] - l_i)
        # guard capacity (should hold because totals unchanged)
        if n_new[i] + l_i + h_i > C_i[i]:
            # trim from HIGH first
            overflow = n_new[i] + l_i + h_i - C_i[i]
            h_i = max(0, h_i - overflow)

        cand = (int(n_new[i]), int(l_i), int(h_i))
        if cand in set(all_states[i]):
            new_Init[i] = cand
        else:
            new_Init[i] = nearest_enumerated_state(cand, all_states[i])

    # 5) report achieved share
    L_ach = sum(new_Init[i][1] for i in N)
    H_ach = sum(new_Init[i][2] for i in N)
    N_ach = sum(new_Init[i][0] for i in N)
    S_ach = L_ach + H_ach + N_ach
    low_share_ach = (L_ach / S_ach) if S_ach > 0 else 0.0

    summary = {
        "target_low_share": float(low_share),
        "achieved_low_share": round(low_share_ach, 6),
        "target_L_total": int(L_target),
        "achieved_L_total": int(L_ach),
        "unmet_L_due_to_caps": int(unmet),
        "system_totals": {"N": int(N_ach), "L": int(L_ach), "H": int(H_ach), "S": int(S_ach)},
    }

    if verbose:
        print("[RESEED] target low share:", low_share, 
              "achieved:", low_share_ach, "unmet:", unmet)

    return new_Init, summary

def compute_init_inventory_for_window(
    N, i_states, C_i, EI_total_prev, t_start, t_begin_day,
    prev_slice, sim_end_inv_avg, station_delta_by_slice,
    tol_frac=0.10, seed=None
):
    """
    Return Init_inventory dict for stations N at beginning of current window t_start.
    """
    rng = np.random.RandomState(seed if seed is not None else (t_start * 7919) % 2**31)
    Init_inventory = {}

    # beginning of day handled outside this function (you reseed), but we support it too
    if t_start == t_begin_day:
        raise RuntimeError("Call your reseed block for t_begin_day before this function.")

    prev_t_start, prev_commit_end = prev_slice

    # prefer simulation if present
    sim_prev = sim_end_inv_avg.get(prev_slice, None) if isinstance(sim_end_inv_avg, dict) else None
    deltas   = station_delta_by_slice.get(prev_slice, None)

    for i in N:
        # 1) baseline: expected at end of last committed slice and at new start
        # guard indices
        def _safe_row(df, t):
            if t in df.index: 
                return t
            # clamp to nearest index inside df
            idx_min = df.index.min()
            idx_max = df.index.max()
            return idx_min if t < idx_min else idx_max

        base_end_t   = _safe_row(EI_total_prev[i], prev_commit_end)
        base_start_t = _safe_row(EI_total_prev[i], t_start)

        base_end = EI_total_prev[i].loc[base_end_t, ['EI_n','EI_l','EI_h']].astype(float).to_numpy()
        base_new = EI_total_prev[i].loc[base_start_t, ['EI_n','EI_l','EI_h']].astype(float).to_numpy()

        # 2) apply either simulation or deltas
        if sim_prev is not None and i in sim_prev:
            target = np.array(sim_prev[i], dtype=float)
        elif deltas is not None and i in deltas:
            dn, dl, dh = deltas[i]
            # adjust new-start by realized difference between base_end and (base_end + deltas)
            act_end = base_end + np.array([dn, dl, dh], float)
            adj_new = base_new + (act_end - base_end)
            target  = adj_new
        else:
            target = base_new

        # 3) jitter ±tol and snap to enumerated feasible state
        Init_inventory[i] = choose_state_with_tolerance(
            tuple(target), states_list=i_states[i], cap=C_i[i],
            tol_frac=tol_frac, rng=rng
        )

    return Init_inventory

def record_truck_end_state(y, tt_ij, V, commit_end, default_loc_by_v, depot_index):
    best_arrival = {v: -10**9 for v in V}
    end_loc      = {v: default_loc_by_v.get(v, depot_index) for v in V}
    cross        = {v: None for v in V}
    best_depart_cross = {v: -10**9 for v in V}

    for (i, j, t, v), var in y.items():
        if varX(var) <= 0.5: 
            continue
        travel = tt_ij[i][j]
        arr = t + travel

        if arr <= commit_end and arr > best_arrival[v]:
            best_arrival[v] = arr
            end_loc[v] = j
            cross[v] = None

        if t <= commit_end < arr and t > best_depart_cross[v]:
            best_depart_cross[v] = t
            cross[v] = {"i": i, "j": j, "remaining": arr - commit_end}

    for v in V:
        if best_arrival[v] <= -10**8 and cross[v] is not None:
            end_loc[v] = cross[v]["i"]  # still en route; park at last departed node
    return end_loc, cross

def summarize_station_deltas(n_load, n_un, b_swap, z, N, P_nl, RT_ij, t_lo, t_hi):
    """
    Return per-station net deltas over [t_lo, t_hi] (inclusive).
    Keys of inputs are numeric dicts:
      n_load[(i,t,v,p)], n_un[(i,t,v,p)], b_swap[(i,t,v,p)],
      z[(o,d,i,t)] = flow redirected to i (from trip o->d starting at t).
    """
    dN = {i: 0.0 for i in N}
    dL = {i: 0.0 for i in N}
    dH = {i: 0.0 for i in N}

    # load/unload at stations
    for (i, t, v, p), val in n_load.items():
        if t_lo <= t <= t_hi:
            if p == 'l': dL[i] += val
            elif p == 'h': dH[i] += val
            elif p == 'n': dN[i] += val

    for (i, t, v, p), val in n_un.items():
        if t_lo <= t <= t_hi:
            if p == 'l': dL[i] -= val
            elif p == 'h': dH[i] -= val
            elif p == 'n': dN[i] -= val

    # battery swap
    for (i, t, v, p), val in b_swap.items():
        if t_lo <= t <= t_hi and p in P_nl: # check if p is valid source power
            if p == 'n': dN[i] -= val
            elif p == 'l': dL[i] -= val
            dH[i] += val

    # incentives: origin -> dest becomes origin -> i
    # The variable z[o,d,i,t] means user arrives at i at t + RT_ij[o][i]
    for (o, d, i, t), val in z.items():
        # arriving at i
        t_arr_i = t + RT_ij[o][i]
        if t_lo <= t_arr_i <= t_hi:
            dH[i] += val # dropoff high (assumed return high)
        
        # departing from d (original trip cancelled at d)
        # Wait, if trip redirected, it never arrives at d.
        # So we lose the +1 H at d that WAS expected?
        # Typically the "baseline" simulation assumes all trips happen.
        # If we redirect, we SUBTRACT 1 from d's arrival and ADD 1 to i's arrival.
        t_arr_d = t + RT_ij[o][d]
        if t_lo <= t_arr_d <= t_hi:
            dH[d] -= val

    # Return tuple of dicts
    return {i: (dN[i], dL[i], dH[i]) for i in N}


def write_results_excel(*args, **kwargs):
    """
    Dummy placeholder to prevent ImportError in Main.py.
    Main.py seems to handle Excel writing manually, but imports this function.
    """
    print("[Utils] write_results_excel called (dummy implementation).")