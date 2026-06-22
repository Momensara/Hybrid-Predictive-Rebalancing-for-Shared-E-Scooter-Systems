"""
Markov_EDL.py — Three-state Markov chain and expected demand loss
==================================================================

Discretised continuous-time Markov model of the inventory at a single
geo-fenced station. The state at time t is the integer triple

    X(t) = (n, l, h)

with n = inactive scooters (battery below 10 %), l = low-power
(10–25 %), and h = high-power (above 25 %).

Transitions are driven by user pickups, drop-offs, and post-trip battery
discharge, with pickup rates scaled by the discrete-choice probabilities
from ``User_choice_model``. The propagation forecasts expected demand
loss (EDL) over a finite look-ahead horizon, which becomes the EDL
objective term in the supporting-plane MIP built by ``main.py``.

Public API
----------
    enumerate_states                    enumerate the (n, l, h) lattice
    compute_Q                           build hourly generator matrices
    build_transition_matrices           per-slot transition matrices
    propagate_and_compute_edl           forward propagation of pi and EDL
    compute_EDL_from_t_to_end           cumulative EDL over [t, t_end]
    affiliate_function                  per-state affiliate value
    refine_planes_shift                 tighten a collection of planes
    compute_affiliate_table_vectorized  batched Bellman backward pass
"""

import numpy as np
import pandas as pd

# ═══════════════════════════════════════════════════════════════════════
# 1. STATE ENUMERATION
# ═══════════════════════════════════════════════════════════════════════

def enumerate_states(capacity: int) -> list:
    """
    List all feasible (n, l, h) inventory tuples for a station.
    """
    return [
        (n, l, h)
        for n in range(capacity + 1)
        for l in range(capacity + 1)
        for h in range(capacity + 1)
        if n + l + h <= capacity
    ]

# ═══════════════════════════════════════════════════════════════════════
# 2. GENERATOR MATRIX Q (continuous-time Markov chain)
# ═══════════════════════════════════════════════════════════════════════

def compute_Q(
    pickup_rates_by_hour,    # dict: hour → scalar pickup rate µ
    dropoff_rates_by_hour,   # dict: hour → np.array([λ_inactive, λ_low, λ_high])
    phi1,                    # dict: hour → discharge rate high → low
    phi2,                    # dict: hour → discharge rate low → no-power
    capacity,                # int: station capacity C_i
    prob_l: float = 0.18,    # P(user picks low-power | arrival)
    prob_h: float = 0.70,    # P(user picks high-power | arrival)
):
    states = enumerate_states(capacity)
    idx = {s: i for i, s in enumerate(states)}
    S = len(states)

    Q_by_hour = {}

    for hour in sorted(pickup_rates_by_hour.keys()):
        pr = pickup_rates_by_hour[hour]
        dr = dropoff_rates_by_hour[hour]

        low_pickr  = pr * prob_l
        high_pickr = pr * prob_h

        low_dropr  = dr[1]
        high_dropr = dr[2]

        Q = np.zeros((S, S))

        for si, (n, l, h) in enumerate(states):
            total = n + l + h

            if h > 0:
                Q[si, idx[(n, l, h - 1)]] = prob_h * high_pickr

            if l > 0:
                Q[si, idx[(n, l - 1, h)]] = prob_l * low_pickr

            if total < capacity:
                rate_low_drop  = (1 - phi2[hour]) * low_dropr + phi1[hour] * high_dropr
                rate_high_drop = (1 - phi1[hour]) * high_dropr
                rate_no_drop   = phi2[hour] * low_dropr

                Q[si, idx[(n, l + 1, h)]]     = rate_low_drop
                Q[si, idx[(n, l, h + 1)]]     = rate_high_drop
                Q[si, idx[(n + 1, l, h)]]     = rate_no_drop

        np.fill_diagonal(Q, -Q.sum(axis=1))
        Q_by_hour[hour] = Q

    return Q_by_hour, states

# ═══════════════════════════════════════════════════════════════════════
# 3. TRANSITION MATRIX DISCRETISATION
# ═══════════════════════════════════════════════════════════════════════

def build_transition_matrices(
    Q_by_hour: dict,
    states: list,
    dt: float,
    T: int,
    slots_per_hour: int,
    n_taylor_steps: int = 100,
):
    nS = len(states)
    I = np.eye(nS)

    P_hour_cache = {}
    for hr, Q_hr in Q_by_hour.items():
        A = Q_hr * dt
        P_hour_cache[hr] = np.linalg.matrix_power(
            I + A / n_taylor_steps, n_taylor_steps
        )

    max_hr = max(Q_by_hour.keys())
    P_mats_step = {}
    for k in range(T + 1):
        hr = min(k // slots_per_hour + 1, max_hr)
        P_mats_step[k] = P_hour_cache[hr]

    return P_mats_step, P_hour_cache

# ═══════════════════════════════════════════════════════════════════════
# 4. FORWARD PROPAGATION & EDL COMPUTATION
# ═══════════════════════════════════════════════════════════════════════

def propagate_and_compute_edl(
    init_state: tuple,
    all_states: list,
    C_i: int,
    P_mats_step: dict,
    fn_pickup_rates_by_hour: dict,
    fn_dropoff_rates_by_hour: dict,
    slots_per_hour: int,
    dt: float,
    t_start: int,
    t_end: int,
    prob_l: float = 0.18,
    prob_h: float = 0.70,
):
    states_arr = np.asarray(all_states, dtype=float)
    nS = len(all_states)
    idx_of = {s: k for k, s in enumerate(all_states)}

    empty_mask = (states_arr[:, 1] == 0) & (states_arr[:, 2] == 0)
    full_mask  = states_arr.sum(axis=1) == C_i

    pi = np.zeros(nS, dtype=float)
    pi[idx_of[init_state]] = 1.0

    pi_by_time = {t_start: pi.copy()}
    edl_records = []
    ei_records = []

    for u, t in enumerate(range(t_start, t_end + 1)):
        P_mat = P_mats_step[t]
        pi = pi @ P_mat
        pi_by_time[t + 1] = pi.copy() if t < t_end else pi

        ei = pi @ states_arr
        ei_records.append(np.round(ei, 2))

        hr = min(t // slots_per_hour, 23)
        pr = fn_pickup_rates_by_hour[hr]
        dr = fn_dropoff_rates_by_hour[hr]

        p_empty = pi[empty_mask].sum()
        p_full  = pi[full_mask].sum()
        edl_empty = (prob_l * pr + prob_h * pr) * dt * p_empty
        edl_full  = (dr[1] + dr[2]) * dt * p_full

        edl_records.append((p_empty, p_full, edl_empty + edl_full))

    idx = pd.RangeIndex(t_start, t_end + 1, name='time_step')

    ei_df = pd.DataFrame(np.array(ei_records), index=idx,
                         columns=['EI_n', 'EI_l', 'EI_h'])
    ei_df.loc[t_start] = init_state

    edl_df = pd.DataFrame(edl_records, index=idx,
                          columns=['prob_empty', 'prob_full', 'EDL_total'])
    
    return pi_by_time, ei_df, edl_df

# ═══════════════════════════════════════════════════════════════════════
# 5. CUMULATIVE EDL FROM T TO END
# ═══════════════════════════════════════════════════════════════════════

def compute_EDL_from_t_to_end(
    station_id,
    inv_new,
    t_intervention,
    t_end,
    EI_total,
    all_states,
    C_i,
    P_mats_step,
    pi_by_time,
    fn_pickup_rates_by_hour,
    fn_dropoff_rates_by_hour,
    period_length_in_steps,
    dt,
    prob_l=0.18, prob_h=0.70):  # Added default args to match expected signature if needed

    i = station_id
    states = all_states[i]
    
    # Check if inv_new matches expected state at t_intervention
    # If so, use stored distribution. Else, use one-hot.
    # Note: EI_total is a dict of DataFrames.
    row = EI_total[station_id].loc[t_intervention]
    expected_state = tuple(row[['EI_n', 'EI_l', 'EI_h']].values)
    
    # Approximate floating point match
    inv_new_arr = np.array(inv_new)
    exp_arr = np.array(expected_state)
    
    if np.allclose(inv_new_arr, exp_arr, atol=0.01):
        pi = pi_by_time[i][t_intervention].copy()
    else:
        pi = np.zeros(len(states))
        if inv_new in states:
            s_idx = states.index(inv_new)
        else:
            # round
            inv_round = tuple(int(round(x)) for x in inv_new)
            if inv_round in states:
                s_idx = states.index(inv_round)
            else:
                 # Fallback: nearest
                s_arr = np.array(states)
                t_arr = np.array(inv_round)
                dists = np.sum(np.abs(s_arr - t_arr), axis=1)
                s_idx = np.argmin(dists)
                
        pi[s_idx] = 1.0

    states_arr = np.array(states)
    empty_mask = (states_arr[:, 1] == 0) & (states_arr[:, 2] == 0)
    full_mask  = states_arr.sum(axis=1) == C_i[i]

    edl_per_step = np.zeros(t_end)
    
    for t in range(t_intervention, t_end):
        hr = min(t // period_length_in_steps + 1,
                max(fn_pickup_rates_by_hour[i].keys()))

        P_step = P_mats_step[i][t]
        pi = pi @ P_step
        
        p_empty = pi[empty_mask].sum()
        p_full  = pi[full_mask].sum()
        
        pr = fn_pickup_rates_by_hour[i][hr]
        dr = fn_dropoff_rates_by_hour[i][hr]

        edl = dt * (((pr * prob_l + pr * prob_h) *  p_empty) + ((dr[1] + dr[2]) * p_full))
        edl_per_step[t] = edl

    slice_ = edl_per_step[t_intervention:t_end]
    tail_sums = slice_[::-1].cumsum()[::-1]

    cum_edl = {
        t0: round(float(tail_sums[t0 - t_intervention]), 3)
        for t0 in range(t_intervention, t_end)
    }

    return cum_edl

# ═══════════════════════════════════════════════════════════════════════
# 6. AFFILIATE FUNCTION
# ═══════════════════════════════════════════════════════════════════════

def affiliate_function(
    station_id,
    inv_new,
    t_intervention,
    t_end,
    input_from_t_to_end_cumu_edl_t,
    EI_total,
    all_states,
    C_i,
    P_mats_step,
    pi_by_time,
    fn_pickup_rates_by_hour,
    fn_dropoff_rates_by_hour,
    period_length_in_steps,
    dt
):
    aff = {}
    for t in range(t_intervention, t_end):
        cum_edl = compute_EDL_from_t_to_end(
                    station_id=station_id,
                    inv_new=inv_new,
                    t_intervention=t,
                    t_end=t_end,
                    EI_total=EI_total,
                    all_states=all_states,
                    C_i=C_i,
                    P_mats_step=P_mats_step,
                    pi_by_time=pi_by_time,
                    fn_pickup_rates_by_hour=fn_pickup_rates_by_hour,
                    fn_dropoff_rates_by_hour=fn_dropoff_rates_by_hour,
                    period_length_in_steps=period_length_in_steps,
                    dt=dt)
        FD_post = cum_edl[t]
        psi = input_from_t_to_end_cumu_edl_t[station_id].loc[t, 'EDL_total']
        aff[t] = round(FD_post - psi, 2)
    return aff

# ═══════════════════════════════════════════════════════════════════════
# 7. REFINE PLANES SHIFT
# ═══════════════════════════════════════════════════════════════════════

def refine_planes_shift(planes, f_vals, all_states, tol=1e-9, eps=1e-9):
    refined = {i: {} for i in planes}
    for i, t_pl in planes.items():
        S = list(all_states[i])
        for t, plist in t_pl.items():
            adj = []
            for pl in plist:
                mn, ml, mh, b0 = pl['m_n'], pl['m_l'], pl['m_h'], pl['b_intercept']
                worst_over = -float("inf")
                for s in S:
                    if t not in f_vals[i][s]:
                        continue
                    n,l,h = s
                    pred  = mn*n + ml*l + mh*h + b0
                    truth = f_vals[i][s][t]
                    worst_over = max(worst_over, pred - truth)
                pl2 = pl.copy()
                if worst_over > tol:
                    pl2['b_intercept'] = b0 - (worst_over + eps) 
                adj.append(pl2)
            
            if not adj:
                min_true = min(f_vals[i][s][t] for s in S if t in f_vals[i][s])
                adj = [{'m_n':0.0, 'm_l':0.0, 'm_h':0.0, 'b_intercept':min_true}]
            refined[i][t] = adj
    return refined
# ═══════════════════════════════════════════════════════════════════════
# 8. VECTORIZED AFFILIATE TABLE (BELLMAN RECURSION)
# ─────────────────────────────────────────────────────────────────────
# Bellman backward recursion for the affiliate function, batched across
# all (n, l, h) states of a single station.
#
# Math:
#   V(t)[s] = expected tail EDL from t to t_forecast given inventory is
#             reset to state s at time t.
#   V(t_forecast)[s] = 0
#   V(t)            = P_step[t] @ (r(t) + V(t+1))
#   r(t)[s']        = α(t)·1{s' empty} + β(t)·1{s' full}
#   α(t)            = (prob_l + prob_h) · pickup_rate(t) · dt
#   β(t)            = (low_dropoff_rate(t) + high_dropoff_rate(t)) · dt
#
# Per station, complexity drops from O(nS·T²·nS²) to O(T·nS²).
# ═══════════════════════════════════════════════════════════════════════
 
def compute_affiliate_table_vectorized(
    station_id,
    t_start, t_end, t_forecast,
    all_states_i,                       # list of (n,l,h) tuples for this station
    C_i_value,                          # int capacity for this station
    P_mats_step_i,                      # dict {t: (nS, nS) transition matrix}
    fn_pickup_rates_by_hour_i,          # dict {clock_hour: scalar rate}
    fn_dropoff_rates_by_hour_i,         # dict {clock_hour: array [n,l,h]}
    input_from_t_to_end_cumu_edl_t_i,   # DataFrame with column 'EDL_total' (baseline)
    period_length_in_steps,             # int (= slots_per_hour)
    dt,                                 # float (= 1 / slots_per_hour)
    prob_l=0.18,
    prob_h=0.70,
    t_begin_day=6,
):
    """
    Vectorised affiliate-table computation for one station.
 
    Returns
    -------
    f_vals_i : dict
        {(n,l,h): {t: affiliate_value}} for every state in all_states_i
        and every t in [t_start, t_end).
 
    Notes
    -----
    Mathematically equivalent to the per-state loop:
 
        for s in all_states[i]:
            aff = affiliate_function(station_id=i, inv_new=s,
                                     t_intervention=t_start, t_end=t_forecast, ...)
            f_vals[i][s] = {t: v for t, v in aff.items() if t_start <= t < t_end}
 
    but does it as one Bellman recursion over time, batching all states.
    """
    states = list(all_states_i)
    nS = len(states)
    states_arr = np.asarray(states, dtype=int)
 
    # Event masks (length-nS float vectors)
    empty_mask = ((states_arr[:, 1] == 0) & (states_arr[:, 2] == 0)).astype(float)
    full_mask  = (states_arr.sum(axis=1) == C_i_value).astype(float)
 
    avail_hours = list(fn_pickup_rates_by_hour_i.keys())
 
    def _hour_at(t):
        hr = (t // period_length_in_steps + t_begin_day) % 24
        if hr not in fn_pickup_rates_by_hour_i:
            hr = min(avail_hours, key=lambda h: abs(h - hr))
        return hr
 
    # Bellman backward recursion from t_forecast down to t_start.
    # We store V(t) for every t in [t_start, t_forecast); only t in
    # [t_start, t_end) feed the affiliate output.
    V_stored = {}
    V_next = np.zeros(nS, dtype=float)   # V(t_forecast) = 0
 
    for t in range(t_forecast - 1, t_start - 1, -1):
        hr = _hour_at(t)
        pr = float(fn_pickup_rates_by_hour_i[hr])
        dr = fn_dropoff_rates_by_hour_i[hr]
 
        alpha_t = (prob_l + prob_h) * pr * dt           # pickup-loss coefficient
        beta_t  = float(dr[1] + dr[2]) * dt             # dropoff-loss coefficient
 
        reward = alpha_t * empty_mask + beta_t * full_mask        # shape (nS,)
        V_curr = P_mats_step_i[t] @ (reward + V_next)             # shape (nS,)
 
        if t_start <= t < t_end:
            V_stored[t] = V_curr
        V_next = V_curr
 
    # Translate to the dict format expected by compute_supporting_planes.
    # affiliate value = V(t)[s] - baseline_psi(t)
    psi_series = input_from_t_to_end_cumu_edl_t_i['EDL_total']
    
    # Return as 2D NumPy array to prevent thousands of dictionary lookups
    times = sorted(V_stored.keys())
    values = np.empty((len(times), nS), dtype=float)
    for row, t_inv in enumerate(times):
        psi = float(psi_series.loc[t_inv])
        values[row] = np.round(V_stored[t_inv] - psi, 2)

    return {"states": states, "times": times, "values": values}
 
 