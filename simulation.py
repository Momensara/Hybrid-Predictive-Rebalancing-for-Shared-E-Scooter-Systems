import numpy as np
import pandas as pd
from collections import defaultdict

def build_ops_delta_for_window(N_active, t_start, t_end, P_nl, C_i,
                               n_load, n_un, b_swap, z, RT_ij):
    # n_load, n_un, b_swap, z are plain numbers now
    delta_n = {i: {t: 0.0 for t in range(t_start, t_end+1)} for i in N_active}
    delta_l = {i: {t: 0.0 for t in range(t_start, t_end+1)} for i in N_active}
    delta_h = {i: {t: 0.0 for t in range(t_start, t_end+1)} for i in N_active}

    for (i, t, v, p), val in n_load.items():
        if i in N_active and t_start <= t <= t_end and val > 0:
            if p == 'l': delta_l[i][t] += val
            if p == 'h': delta_h[i][t] += val

    for (i, t, v, p), val in n_un.items():
        if i in N_active and t_start <= t <= t_end and val > 0:
            if p == 'l': delta_l[i][t] -= val
            if p == 'h': delta_h[i][t] -= val

    for (i, t, v, p), val in b_swap.items():
        if i in N_active and t_start <= t <= t_end and val > 0 and p in P_nl:
            if p == 'n': delta_n[i][t] -= val
            if p == 'l': delta_l[i][t] -= val
            delta_h[i][t] += val

    for (o, d, i, t0), qty in z.items():
        if qty <= 0 or i not in N_active or d not in N_active: 
            continue
        t_arr_i = t0 + RT_ij[o][i]
        t_arr_d = t0 + RT_ij[o][d]
        if t_start <= t_arr_i <= t_end:
            delta_h[i][t_arr_i] += qty
        if t_start <= t_arr_d <= t_end:
            delta_h[d][t_arr_d] -= qty

    return delta_n, delta_l, delta_h

def _accept_dropoffs_proportionally(req_drop_vec, free_docks):
    """
    Allocate 'free_docks' across drop-off demand classes proportionally.

    req_drop_vec : array-like of nonnegative numbers (e.g., [no, low, high] dropoffs)
    free_docks   : int >= 0

    Returns: np.ndarray[int] of same length as req_drop_vec,
             with sum <= min(sum(req_drop_vec), free_docks) and per-class <= request.
    """
    # Normalize inputs
    r = np.array(req_drop_vec, dtype=float).clip(min=0.0)
    K = r.size
    F = int(max(0, int(free_docks)))

    total = float(r.sum())
    if F == 0 or total <= 0.0:
        return np.zeros(K, dtype=int)

    # Ideal proportional allocation (float)
    ideal = (F * r / total)

    # Floor and cap by per-class demand
    alloc = np.floor(np.minimum(ideal, r)).astype(int)

    # Distribute leftover by largest fractional parts, but only to classes with slack
    leftover = int(min(F, total) - alloc.sum())
    if leftover > 0:
        frac = ideal - alloc
        # Only candidates where demand not yet met
        slack_mask = (alloc < r - 1e-12)
        # Order indices by fractional part descending
        order = np.argsort(-frac)
        for idx in order:
            if leftover == 0:
                break
            if not slack_mask[idx]:
                continue
            alloc[idx] += 1
            leftover -= 1

    return alloc

def _station_demand_to_arrays(station_demand, N, slots_per_hour):
    S = slots_per_hour
    L = 24 * S
    pL_steps = {i: np.zeros(L, dtype=int) for i in N}
    pH_steps = {i: np.zeros(L, dtype=int) for i in N}
    dN_steps = {i: np.zeros(L, dtype=int) for i in N}
    dL_steps = {i: np.zeros(L, dtype=int) for i in N}
    dH_steps = {i: np.zeros(L, dtype=int) for i in N}

    for i in N:
        if i not in station_demand:
            continue
        for h in range(24):
            if h not in station_demand[i]:
                continue
            start = h * S
            end   = start + S
            rec   = station_demand[i][h]
            pL_steps[i][start:end] = np.asarray(rec['pickup_low'], dtype=int)
            pH_steps[i][start:end] = np.asarray(rec['pickup_high'], dtype=int)
            drec = rec['dropoff']
            dN_steps[i][start:end] = np.asarray(drec['no_power'],  dtype=int)
            dL_steps[i][start:end] = np.asarray(drec['low_power'], dtype=int)
            dH_steps[i][start:end] = np.asarray(drec['high_power'], dtype=int)
    return pL_steps, pH_steps, dN_steps, dL_steps, dH_steps


def simulate_window_service_level(
    t_start, t_end, N, C_i, slots_per_hour,
    fn_pickup_rates_by_hour, fn_dropoff_rates_by_hour,
    Init_inventory, n_load, n_un, b_swap, z, RT_ij,
    P_nl=('n','l'),
    n_runs=100,
    hour_cap=60, peak_hour_cap=90, day_cap=1080,  # <-- added peak_hour_cap (default 90)
    prob_l=0.5, prob_h=0.5,
    allow_L_to_serve_low=True,
    allow_H_to_serve_low=True,
    allow_L_to_serve_high=False,   # usually False
    allow_H_to_serve_high=True,
    sim_seed=None,
    t_begin_day=6,                 # clock-hour at which operating slot 0 begins (default 6 AM)
    enforce_time_patterns=False,   # disable the hardcoded EARLY_MORNING / DROPOFF_PEAK / PICKUP_PEAK throttles
):
    """
    Uses wkD_simulated_demand_corrected() to generate per-day demand,
    then simulates service over the [t_start, t_end] window with MILP ops applied.
    """
    from demand_generator import wkD_simulated_demand_corrected

    # Deterministic ops deltas at integer steps from the MILP solution
    N_active = list(N)
    delta_n, delta_l, delta_h = build_ops_delta_for_window(
        N_active, t_start, t_end, P_nl, C_i, n_load, n_un, b_swap, z, RT_ij
    )

    # base RNG only for run seeding
    base_rng = np.random.default_rng(sim_seed)

    # Accumulators across runs (for means)
    keys = [
        'pickups_generated','dropoffs_generated',
        'missed_pickups','missed_dropoffs',
        'served_pickups','served_dropoffs',
        'requests_total','served_total'
    ]
    acc_station = {i: {k: [] for k in keys} for i in N}
    acc_system  = {k: [] for k in keys}
    system_levels = []
    end_inv_runs = []

    for _run in range(n_runs):
        # --- Generate one full-day demand realization with peak-hour cap logic ---
        run_seed = int(base_rng.integers(0, 2**32 - 1))
        sim_out = wkD_simulated_demand_corrected(
            pk_hourly_rates=fn_pickup_rates_by_hour,
            dr_hourly_rates=fn_dropoff_rates_by_hour,
            slots_per_hour=slots_per_hour,
            hour_cap=hour_cap,
            peak_hour_cap=peak_hour_cap,     # <-- afternoon peak cap
            day_cap=day_cap,
            prob_l=prob_l,
            prob_h=prob_h,
            enforce_time_patterns=enforce_time_patterns,  # default False: let empirical hourly rates drive the pattern
            seed=run_seed
        )
        pL_steps, pH_steps, dN_steps, dL_steps, dH_steps = _station_demand_to_arrays(
            sim_out['station_demand'], N_active, slots_per_hour
        )

        # per-run station tallies
        served_by_station         = defaultdict(int)
        total_by_station          = defaultdict(int)
        served_pickups_by_station = defaultdict(int)
        served_drop_by_station    = defaultdict(int)
        pickups_gen_by_station    = defaultdict(int)
        dropoffs_gen_by_station   = defaultdict(int)
        missed_pick_by_station    = defaultdict(int)
        missed_drop_by_station    = defaultdict(int)

        # simulate sequentially over steps in the window
        inv_n = {i: int(Init_inventory[i][0]) for i in N}
        inv_l = {i: int(Init_inventory[i][1]) for i in N}
        inv_h = {i: int(Init_inventory[i][2]) for i in N}

        for t in range(t_start, t_end + 1):
            for i in N:
                # apply deterministic operations decided by MILP
                inv_n[i] += int(delta_n.get(i, {}).get(t, 0))
                inv_l[i] += int(delta_l.get(i, {}).get(t, 0))
                inv_h[i] += int(delta_h.get(i, {}).get(t, 0))

                # clamp inventories and capacity
                inv_n[i] = max(inv_n[i], 0)
                inv_l[i] = max(inv_l[i], 0)
                inv_h[i] = max(inv_h[i], 0)
                cap = C_i[i]
                tot_inv = inv_n[i] + inv_l[i] + inv_h[i]
                if tot_inv > cap:
                    overflow = tot_inv - cap
                    k = min(overflow, inv_n[i]); inv_n[i] -= k; overflow -= k
                    if overflow:
                        k = min(overflow, inv_l[i]); inv_l[i] -= k; overflow -= k
                    if overflow:
                        inv_h[i] -= overflow

                # Map operating-day slot t to clock hour 0-23
                # (operating slot 0 corresponds to t_begin_day, default 6 AM).
                h = (t // slots_per_hour + t_begin_day) % 24
                s_in_day = h * slots_per_hour + (t % slots_per_hour)

                # requested counts this step (from pre-simulated arrays)
                req_pick_low  = int(pL_steps[i][s_in_day])
                req_pick_high = int(pH_steps[i][s_in_day])
                req_drop_vec  = np.array([
                    int(dN_steps[i][s_in_day]),
                    int(dL_steps[i][s_in_day]),
                    int(dH_steps[i][s_in_day]),
                ], dtype=int)

                pickups_generated_step  = req_pick_low + req_pick_high
                dropoffs_generated_step = int(req_drop_vec.sum())

                # ---------- DROP-OFFS FIRST ----------
                free_docks = cap - (inv_n[i] + inv_l[i] + inv_h[i])
                served_drop_vec = _accept_dropoffs_proportionally(req_drop_vec, free_docks)
                served_dropoffs_step = int(served_drop_vec.sum())
                missed_dropoffs_step = dropoffs_generated_step - served_dropoffs_step
                # add to inventories
                inv_n[i] += int(served_drop_vec[0])
                inv_l[i] += int(served_drop_vec[1])
                inv_h[i] += int(served_drop_vec[2])

                # ---------- PICKUPS ----------
                # High-power requests
                serve_h_h = min(req_pick_high, inv_h[i]) if allow_H_to_serve_high else 0
                inv_h[i] -= serve_h_h
                unserved_high = req_pick_high - serve_h_h

                # Low-power requests
                serve_l_l = min(req_pick_low, inv_l[i]) if allow_L_to_serve_low else 0
                inv_l[i] -= serve_l_l
                remain_low = req_pick_low - serve_l_l

                # Optional: L serving leftover high (usually disabled)
                serve_h_l = 0
                if allow_L_to_serve_high and unserved_high > 0:
                    serve_h_l = min(unserved_high, inv_l[i])
                    inv_l[i] -= serve_h_l
                    unserved_high -= serve_h_l

                # H backstops remaining low (if allowed)
                serve_l_h = min(remain_low, inv_h[i]) if allow_H_to_serve_low else 0
                inv_h[i] -= serve_l_h
                remain_low -= serve_l_h

                served_pickups_step = serve_h_h + serve_h_l + serve_l_l + serve_l_h
                missed_pickups_step = pickups_generated_step - served_pickups_step

                # Tallies
                served_drop_by_station[i]    += served_dropoffs_step
                served_pickups_by_station[i] += served_pickups_step
                dropoffs_gen_by_station[i]   += dropoffs_generated_step
                pickups_gen_by_station[i]    += pickups_generated_step
                missed_drop_by_station[i]    += missed_dropoffs_step
                missed_pick_by_station[i]    += missed_pickups_step

                served_total_step = served_dropoffs_step + served_pickups_step
                total_reqs_step   = dropoffs_generated_step + pickups_generated_step
                served_by_station[i] += served_total_step
                total_by_station[i]  += total_reqs_step

        # end-of-run inventories (AFTER the window) — fixed location
        end_inv_runs.append({i: (inv_n[i], inv_l[i], inv_h[i]) for i in N})

        # end of run: aggregate to system + push to accumulators
        sys_totals = {k: 0 for k in keys}
        for i in N:
            st = {
                'pickups_generated' : pickups_gen_by_station[i],
                'dropoffs_generated': dropoffs_gen_by_station[i],
                'missed_pickups'    : missed_pick_by_station[i],
                'missed_dropoffs'   : missed_drop_by_station[i],
                'served_pickups'    : served_pickups_by_station[i],
                'served_dropoffs'   : served_drop_by_station[i],
                'requests_total'    : total_by_station[i],
                'served_total'      : served_by_station[i],
            }
            for k, v in st.items():
                acc_station[i][k].append(int(v))
                sys_totals[k] += int(v)

        for k, v in sys_totals.items():
            acc_system[k].append(int(v))

        sys_served = sys_totals['served_total']
        sys_total  = sys_totals['requests_total']
        system_levels.append((sys_served / sys_total) if sys_total > 0 else 1.0)

    # === Aggregate means across runs ===
    per_station_window_means = {
        i: {k: float(np.mean(acc_station[i][k])) if len(acc_station[i][k]) > 0 else 0.0
            for k in keys}
        for i in N
    }
    system_window_means = {
        k: float(np.mean(v)) if len(v) > 0 else 0.0
        for k, v in acc_system.items()
    }

    # service level mean + CI
    n_runs_eff = max(len(system_levels), 1)
    sys_mean = float(np.mean(system_levels)) if n_runs_eff > 0 else 1.0
    sys_std  = float(np.std(system_levels, ddof=1)) if n_runs_eff > 1 else 0.0
    ci95 = (
        sys_mean - 1.96 * sys_std / np.sqrt(n_runs_eff),
        sys_mean + 1.96 * sys_std / np.sqrt(n_runs_eff),
    )

    # per-station service mean (request-weighted over the window)
    per_station_service_mean = {
        i: (
            per_station_window_means[i]['served_total'] /
            per_station_window_means[i]['requests_total']
        ) if per_station_window_means[i]['requests_total'] > 0 else 1.0
        for i in N
    }

    # average end-of-window inventory across runs (rounded to ints)
    avg_end_inventory = {}
    for i in N:
        n_mean = float(np.mean([d[i][0] for d in end_inv_runs]))
        l_mean = float(np.mean([d[i][1] for d in end_inv_runs]))
        h_mean = float(np.mean([d[i][2] for d in end_inv_runs]))
        avg_end_inventory[i] = (int(round(n_mean)), int(round(l_mean)), int(round(h_mean)))

    return {
        'per_station_service_mean' : per_station_service_mean,
        'system_service_level_mean': sys_mean,
        'system_service_level_ci95': ci95,
        'per_station_window_means' : per_station_window_means,
        'system_window_means'      : system_window_means,
        'runs'                     : n_runs,
        'avg_end_inventory'        : avg_end_inventory,
    }