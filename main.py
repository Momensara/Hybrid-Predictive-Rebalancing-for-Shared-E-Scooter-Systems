"""
main.py — Rolling-horizon driver for hybrid e-scooter rebalancing
==================================================================

Entry point that orchestrates one full operating-day run of the
Direct Look-ahead Hybrid Rebalancing (DL-HR) framework on a geo-fenced
shared e-scooter system.

The driver performs four steps per planning window:

    1. Observe the current state of the fleet (per-station inventory by
       battery class) and the truck position.
    2. Propagate the system state forward over the look-ahead horizon
       using the three-state Markov chain in ``Markov_EDL``, yielding
       expected demand loss (EDL) at every (station × slot).
    3. Solve a mixed-integer program (Gurobi) that selects truck routes,
       on-station operations (load/unload/swap), and optional user-mediated
       drop-off incentives so as to minimise total operating cost plus
       monetised EDL.
    4. Commit the first portion of the resulting plan and roll forward.

Run ``python main.py --help`` for the full list of command-line options.
"""

import os
import time
import random
import argparse
import numpy as np
import pandas as pd
import gurobipy as gp
from gurobipy import Model, GRB


# ── Project modules ──
from ProjectConfig import Config, TIME, NETWORK, COST, CAPACITY
from data_loader import load_all_data
from Markov_EDL import (
    compute_Q, 
    affiliate_function, refine_planes_shift,
    compute_affiliate_table_vectorized, 
)
from simulation import simulate_window_service_level
from Utils import (
    nearest_enumerated_state, reseed_initial_low_share,
    compute_init_inventory_for_window, record_truck_end_state,
    values_dict, varX, sum_over_vars, summarize_station_deltas
)


# ═══════════════════════════════════════════════════════════════════════
# STRATEGY PRESETS
# ═══════════════════════════════════════════════════════════════════════
#
# Four strategies are exposed:
#
#   DL-HR   Full Direct Look-ahead Hybrid Rebalancing:
#             truck-based relocation + on-street battery swapping
#             + user-mediated drop-off incentives.
#   REL+SW  Operator-only relocation + battery swapping (no user incentives).
#   REL     Operator-only relocation (no battery swap, no user incentives).
#   NR      No rebalancing — baseline that lets the system evolve untouched.
# ═══════════════════════════════════════════════════════════════════════

STRATEGY_PRESETS = {
    "DL-HR": Config(
        use_truck=True, use_ops=True, use_swap=True,
        use_truck_inventory=True, use_battery_inventory=True,
        use_supporting_planes=True, use_incentives=True,
    ),
    "REL+SW": Config(
        use_truck=True, use_ops=True, use_swap=True,
        use_truck_inventory=True, use_battery_inventory=True,
        use_supporting_planes=True, use_incentives=False,
    ),
    "REL": Config(
        use_truck=True, use_ops=True, use_swap=False,
        use_truck_inventory=True, use_battery_inventory=False,
        use_supporting_planes=True, use_incentives=False,
    ),
    "NR": Config(
        use_truck=False, use_ops=False, use_swap=False,
        use_truck_inventory=False, use_battery_inventory=False,
        use_supporting_planes=False, use_incentives=False,
    ),
}

# ═══════════════════════════════════════════════════════════════════════
# PRE-COMPUTATION
# ═══════════════════════════════════════════════════════════════════════

def precompute_markov_and_edl(
    N, C_i, all_states, Init_inventory,
    fn_pickup_rates_by_hour, fn_dropoff_rates_by_hour,
    P_mats_step, t_start, t_forecast, slots_per_hour, dt,
    prob_l, prob_h,
):
    EI_total, pi_by_time, station_edl_separate_t = {}, {}, {}

    for i in N:
        S = np.asarray(all_states[i], float)
        nS = S.shape[0]
        empty_mask = (S[:, 1] == 0) & (S[:, 2] == 0)
        full_mask  = S.sum(1) == C_i[i]

        idx_of = {s: k for k, s in enumerate(all_states[i])}
        pi = np.zeros(nS, float)
        pi[idx_of[Init_inventory[i]]] = 1.0

        Tlen = t_forecast - t_start + 1
        PI = np.empty((Tlen + 1, nS), float)
        PI[0] = pi

        rec = np.empty((Tlen, 3), float)
        EI  = np.empty((Tlen, 3), float)

        for u, t in enumerate(range(t_start, t_forecast + 1)):
            P_mat = P_mats_step[i][t]
            pi = pi @ P_mat
            PI[u + 1] = pi
            EI[u] = pi @ S

            hr_rate = (t // slots_per_hour + TIME.t_begin_day) % 24
            pr = fn_pickup_rates_by_hour[i][hr_rate]
            dr = fn_dropoff_rates_by_hour[i][hr_rate]

            p_empty = pi[empty_mask].sum()
            p_full  = pi[full_mask].sum()

            edl_empty = (prob_l * pr + prob_h * pr) * dt * p_empty
            edl_full  = (dr[1] + dr[2]) * dt * p_full
            rec[u] = (p_empty, p_full, edl_empty + edl_full)

        idx = pd.RangeIndex(t_start, t_forecast + 1, name='time_step')
        dfEI = pd.DataFrame(np.round(EI, 2), index=idx, columns=['EI_n', 'EI_l', 'EI_h'])
        dfEI.loc[t_start] = Init_inventory[i]
        EI_total[i] = dfEI

        dfEDL = pd.DataFrame(np.round(rec, 3), index=idx, columns=['prob_empty', 'prob_full', 'EDL_total'])
        station_edl_separate_t[i] = dfEDL
        pi_by_time[i] = {t_start + u: PI[u].copy() for u in range(Tlen + 1)}

    beg_to_t = {i: df[['EDL_total']].cumsum().round(3) for i, df in station_edl_separate_t.items()}
    t_to_end = {i: df[['EDL_total']].iloc[::-1].cumsum().iloc[::-1].round(3) for i, df in station_edl_separate_t.items()}

    return {
        "EI_total": EI_total, "pi_by_time": pi_by_time,
        "station_edl_separate_t": station_edl_separate_t,
        "begining_to_t_cumu_edl_t": beg_to_t, "from_t_to_end_cumu_edl_t": t_to_end,
    }


def compute_supporting_planes(
    N_tracking, all_states, C_i, fn_pickup_rates_by_hour, fn_dropoff_rates_by_hour,
    EI_total, pi_by_time, P_mats_step, from_t_to_end_cumu_edl_t,
    period_length_in_steps, dt, t_start, t_end, t_forecast,
):
    # fn_kwargs = dict(
    #     input_from_t_to_end_cumu_edl_t=from_t_to_end_cumu_edl_t,
    #     EI_total=EI_total, all_states=all_states, C_i=C_i,
    #     P_mats_step=P_mats_step, pi_by_time=pi_by_time,
    #     fn_pickup_rates_by_hour=fn_pickup_rates_by_hour,
    #     fn_dropoff_rates_by_hour=fn_dropoff_rates_by_hour,
    #     period_length_in_steps=period_length_in_steps, dt=dt,
    # )

    # f_vals = {i: {} for i in N_tracking}
    # for i in N_tracking:
    #     for s in all_states[i]:
    #         aff = affiliate_function(
    #             station_id=i, inv_new=s, t_intervention=t_start, t_end=t_forecast, **fn_kwargs,
    #         )
    #         f_vals[i][s] = {t: v for t, v in aff.items() if t_start <= t < t_end}

    # planes = {i: {} for i in N_tracking}
    # for i in N_tracking:
    #     for t in range(t_start, t_end):
    #         plane_list = []
    #         for (n0, l0, h0) in all_states[i]:
    #             if n0 + l0 + h0 >= C_i[i] - 1: continue
    #             f0  = f_vals[i][(n0, l0, h0)][t]
    #             f_n = f_vals[i][(n0+1, l0, h0)][t]
    #             f_l = f_vals[i][(n0, l0+1, h0)][t]
    #             f_h = f_vals[i][(n0, l0, h0+1)][t]

    #             g_n, g_l, g_h = f_n - f0, f_l - f0, f_h - f0
    #             b0 = f0 - (g_n * n0 + g_l * l0 + g_h * h0)
    #             plane_list.append({'m_n': g_n, 'm_l': g_l, 'm_h': g_h, 'b_intercept': b0})
    #         planes[i][t] = plane_list
    #     planes[i][t_end] = planes[i][t_end - 1].copy()

    # planes_win = {i: {t: planes[i][t] for t in range(t_start, t_end + 1) if t in planes[i]} for i in N_tracking}
    # return refine_planes_shift(planes_win, f_vals, all_states)
    # ─────────────────────────────────────────────────────────────────────
    # Affiliate-table computation
    #
    # OLD (per-state Python loop, kept here for reference):
    #     fn_kwargs = dict(...)
    #     f_vals = {i: {} for i in N_tracking}
    #     for i in N_tracking:
    #         for s in all_states[i]:                       # ~286 states
    #             aff = affiliate_function(                 # nested time loops inside
    #                 station_id=i, inv_new=s,
    #                 t_intervention=t_start, t_end=t_forecast, **fn_kwargs,
    #             )
    #             f_vals[i][s] = {t: v for t, v in aff.items() if t_start <= t < t_end}
    #
    # NEW (one Bellman backward pass per station, all states batched):
    #     compute_affiliate_table_vectorized() returns f_vals[i] in a single call.
    #     Complexity per station drops from O(nS · T_forecast² · nS²) to O(T_forecast · nS²).
    # ─────────────────────────────────────────────────────────────────────
    f_vals = {}
    planes = {i: {} for i in N_tracking}
    
    for i in N_tracking:
        res = compute_affiliate_table_vectorized(
            station_id=i,
            t_start=t_start, t_end=t_end, t_forecast=t_forecast,
            all_states_i=all_states[i],
            C_i_value=C_i[i],
            P_mats_step_i=P_mats_step[i],
            fn_pickup_rates_by_hour_i=fn_pickup_rates_by_hour[i],
            fn_dropoff_rates_by_hour_i=fn_dropoff_rates_by_hour[i],
            input_from_t_to_end_cumu_edl_t_i=from_t_to_end_cumu_edl_t[i],
            period_length_in_steps=period_length_in_steps,
            dt=dt,
            t_begin_day=TIME.t_begin_day,
        )
        
        # Compute planes and refinement using NumPy arrays
        times = res['times']
        values = res['values']
        states = res['states']
        
        idx_of = {s: k for k, s in enumerate(states)}
        valid_idx, n_idx, l_idx, h_idx = [], [], [], []
        n_vals_list, l_vals_list, h_vals_list = [], [], []
        
        for k, (n0, l0, h0) in enumerate(states):
            if n0 + l0 + h0 >= C_i[i] - 1: continue
            valid_idx.append(k)
            n_idx.append(idx_of[(n0+1, l0, h0)])
            l_idx.append(idx_of[(n0, l0+1, h0)])
            h_idx.append(idx_of[(n0, l0, h0+1)])
            n_vals_list.append(n0)
            l_vals_list.append(l0)
            h_vals_list.append(h0)
            
        if not valid_idx:
            # Fallback if no valid points to make planes
            min_true = np.min(values, axis=1)
            for r, t in enumerate(times):
                planes[i][t] = [{'m_n': 0.0, 'm_l': 0.0, 'm_h': 0.0, 'b_intercept': float(min_true[r])}]
            if len(times) > 0:
                planes[i][t_end] = planes[i][times[-1]].copy()
            continue

        v_0 = values[:, valid_idx]
        v_n = values[:, n_idx]
        v_l = values[:, l_idx]
        v_h = values[:, h_idx]
        
        g_n = v_n - v_0
        g_l = v_l - v_0
        g_h = v_h - v_0
        
        n_arr = np.array(n_vals_list)
        l_arr = np.array(l_vals_list)
        h_arr = np.array(h_vals_list)
        
        b0 = v_0 - (g_n * n_arr + g_l * l_arr + g_h * h_arr)
        
        S_arr = np.array(states)
        S_n, S_l, S_h = S_arr[:, 0], S_arr[:, 1], S_arr[:, 2]
        
        for r, t in enumerate(times):
            plist = []
            for p in range(len(valid_idx)):
                mn, ml, mh, b_int = g_n[r, p], g_l[r, p], g_h[r, p], b0[r, p]
                preds = mn * S_n + ml * S_l + mh * S_h + b_int
                truths = values[r, :]
                worst_over = np.max(preds - truths)
                if worst_over > 1e-9:
                    b_int -= (worst_over + 1e-9)
                plist.append({'m_n': float(mn), 'm_l': float(ml), 'm_h': float(mh), 'b_intercept': float(b_int)})
            planes[i][t] = plist
            
        if len(times) > 0:
            planes[i][t_end] = planes[i][times[-1]].copy()

    return planes
 
 

# ═══════════════════════════════════════════════════════════════════════
# BUILD AND SOLVE THE MIP FOR ONE WINDOW
# ═══════════════════════════════════════════════════════════════════════

def build_and_solve_mip(
    cfg, data, N_truck, N0_truck, N_tracking, omega_set,
    t_begin, t_start, t_end, commit_end, T_total,     
    Init_inventory, input_EI_total, input_beg_edl, input_end_edl,
    refine_planes, accept_parameter, omega_h, edl_coef, committed_edl_inactive,
    prev_acquired_truck, prev_incentive_cost, end_loc_dict, prev_inventory,
    window_budget_cap=None,
    minute_RT_ij=None,
    mip_time_limit=600.0,
    mip_gap_tol=None,
):
    N, V, P, P_nl, M = NETWORK.N, NETWORK.V, NETWORK.P, NETWORK.P_nl, NETWORK.M
    depot = 0
    C_i, d_ij, tt_ij, RT_ij = data["C_i"], data["d_km"], data["tt_ij"], data["RT_ij"]
    C_v_es = {v: CAPACITY.C_v_escooter for v in V}
    C_v_bt = {v: CAPACITY.C_v_batt for v in V}
    minute_RT_ij = data["minute_RT_ij"]

    model = Model(f"RH_{t_start}_{t_end}")
    model.Params.OutputFlag = 0

    r    = model.addVars(N_tracking, vtype=GRB.BINARY, name="r_not_served")
    s_it = model.addVars(N_tracking, range(t_start, t_end+1), P, lb=0.0, name="s_inv")

    obj = committed_edl_inactive

    # ── A. Supporting planes ──
    w_sp = {}
    if cfg.use_supporting_planes:
        w_sp = model.addVars(N_tracking, range(t_start, t_end+1), vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, name="w_sp")
        obj += gp.quicksum(edl_coef * w_sp[i, t] for i in N_tracking for t in range(t_start, t_end+1))

    # ── B. Truck routing ──
    y, u = {}, {}
    if cfg.use_truck:
        y = model.addVars(N0_truck, N0_truck, range(t_start, t_end+1), V, vtype=GRB.BINARY, name="y_route")
        u = model.addVars(V, vtype=GRB.BINARY, name="u_truck")
        M_dep = len(N0_truck) * (t_end - t_start + 1)

        model.addConstrs((M_dep * u[v] >= gp.quicksum(y[depot, j, t, v] for j in N_truck for t in range(t_start, t_end+1)) for v in V), name='link_truck_use')
        model.addConstrs((u[v] <= gp.quicksum(y[depot, j, t, v] for j in N_truck for t in range(t_start, t_end+1)) for v in V), name='cap_truck_use')

        obj += gp.quicksum((COST.c_v if not prev_acquired_truck[v] else 0) * u[v] for v in V)
        obj += gp.quicksum(COST.c_ij * d_ij[i, j] * y[i, j, t, v] for i in N0_truck for j in N0_truck for v in V for t in range(t_start, t_end+1))
        obj += (gp.quicksum(edl_coef * input_beg_edl[i].loc[t, 'EDL_total'] * y[i, j, t, v] for i in N_truck for j in N0_truck for v in V for t in range(t_start, t_end+1))
              + gp.quicksum(edl_coef * input_end_edl[i].loc[t, 'EDL_total'] * y[i, j, t, v] for i in N_truck for j in N0_truck for v in V for t in range(t_start, t_end+1)))

    # ── C. Load/unload operations ──
    n_load, n_un = {}, {}
    if cfg.use_ops:
        n_load = model.addVars(N0_truck, range(t_start, t_end+1), V, P, vtype=GRB.INTEGER, lb=0, name="n_load")
        n_un   = model.addVars(N0_truck, range(t_start, t_end+1), V, P, vtype=GRB.INTEGER, lb=0, name="n_un")
        obj += COST.c_load_unload * gp.quicksum(n_load[i, t, v, p] + n_un[i, t, v, p] for i in N0_truck for t in range(t_start, t_end+1) for v in V for p in P)

    # ── D. Battery swap ──
    b_swap = {}
    if cfg.use_swap:
        b_swap = model.addVars(N0_truck, range(t_start, t_end+1), V, P_nl, vtype=GRB.INTEGER, lb=0, name="b_swap")
        obj += gp.quicksum(COST.c_swap * b_swap[i, t, v, p] for i in N0_truck for t in range(t_start, t_end+1) for v in V for p in P_nl)

    # ── E/F. Truck cargo tracking ──
    q = {}
    if cfg.use_truck_inventory or cfg.use_battery_inventory:
        q = model.addVars(N0_truck, N0_truck, range(t_start, t_end+1), V, P, M, lb=0, vtype=GRB.INTEGER, name="q_truck")
    # ─────────────────────────────────────────────────────────────────
    # Time-feasibility pruning of y arcs.
    #
    # An arc (i, j, t, v) is feasible iff:
    #   (a) the truck could physically be at i by time t:
    #         t_start + tt_ij[truck_start[v]][i] <= t
    #   (b) the arc terminates within or one step past the window:
    #         t + tt_ij[i][j] <= t_end + 1
    #
    # Infeasible y vars are fixed via UB=0; Gurobi presolve eliminates
    # them and the q vars they constrain (truck_cap_es enforces q <= 
    # C_v * y, so y=0 ⇒ q=0). We also fix q directly to help presolve.
    # ─────────────────────────────────────────────────────────────────
    # ─────────────────────────────────────────────────────────────────
    # Time & Space feasibility pruning
    # ─────────────────────────────────────────────────────────────────
    if cfg.use_truck and y:
        truck_start = {
            v: (0 if t_start == t_begin else end_loc_dict.get((v, t_start), 0))
            for v in V
        }
        y_pruned = 0
        infeasible_yk = set()
        
        # Spatial Pruning (Max 60 minutes / 4 slots)
        # MAX_DRIVE_SLOTS = 4 
        MAX_DRIVE_MIN = 60   # paper: no drive longer than 60 wall-clock minutes
        MAX_DRIVE_SLOTS = max(1, MAX_DRIVE_MIN // TIME.tau)
        
        for key, var in y.items():
            i, j, t, v = key
            
            # If the truck is already at the start node, it takes 0 time to get there.
            travel_to_i = 0 if i == truck_start[v] else tt_ij[truck_start[v]][i]
            earliest_at_i = t_start + travel_to_i
            
            arrival_at_j  = t + tt_ij[i][j]
            
            # Check if the drive takes longer than 60 minutes (and isn't the depot dispatch)
            is_too_far = (tt_ij[i][j] > MAX_DRIVE_SLOTS) and (i != 0)
            
            if earliest_at_i > t or arrival_at_j > t_end + 1 or is_too_far:
                var.UB = 0
                infeasible_yk.add(key)
                y_pruned += 1

        q_pruned = 0
        if q:
            for key, var in q.items():
                i, j, t, v, p, m = key
                if (i, j, t, v) in infeasible_yk:
                    var.UB = 0
                    q_pruned += 1

        if y_pruned or q_pruned:
            total_y, total_q = len(y), len(q) if q else 0
            print(f"      pruned routing: y {y_pruned}/{total_y} "
                  f"({100*y_pruned/max(total_y,1):.0f}%), "
                  f"q {q_pruned}/{total_q} ({100*q_pruned/max(total_q,1):.0f}%)")
    
    # ── H. User incentives ──
    x, z = {}, {}
    # if cfg.use_incentives:
    #     incentive_trips = [(o, d, i, t) for (o, d, t) in omega_set for i in N_truck if accept_parameter.get((o, d, i), 0) == 1]
    if cfg.use_incentives:
        # ─────────────────────────────────────────────────────────────────
        # Two-stage variable elimination
        #
        # Stage 1 — Zero-flow pruning. For each (o, d, t) origin-destination
        # request: skip if there are no high-power scooters at o OR no
        # historical flow from o to d at this hour. Either way, no MIP
        # decision can produce a meaningful z[o,d,i,t] > 0.
        #
        # Stage 2 — Budget-cap pruning. For each surviving (o, d, i, t):
        # skip if a single redirected trip's ride fee already exceeds the
        # window's incentive budget cap. That trip can never be selected.
        # ─────────────────────────────────────────────────────────────────
        # Stage 1: flow filter on (o, d, t)
        feasible_od = []
        for (o, d, t) in omega_set:
            flow_limit =  float(input_EI_total[o].loc[t]['EI_h'])
            if int(flow_limit) >= 1:
                feasible_od.append((o, d, t))

        # Effective budget ceiling for this window's spending
        if window_budget_cap is not None:
            effective_cap = window_budget_cap                                       # adaptive
        else:
            effective_cap = max(COST.incentive_budget - prev_incentive_cost, 0.0)   # cumulative

        # Stage 2: per-trip cost filter on (o, d, i, t) for surviving (o, d, t)
        raw_trips = [
            (o, d, i, t)
            for (o, d, t) in feasible_od
            for i in N_tracking
            if accept_parameter.get((o, d, i), 0) == 1
        ]

        if effective_cap <= 0:
            incentive_trips = []
        else:
            incentive_trips = [
                (o, d, i, t) for (o, d, i, t) in raw_trips
                if COST.RF * minute_RT_ij[o][i] <= effective_cap
            ]

        # Optional log line — keep during calibration, remove later if noisy
        if omega_set:
            n_od_total = len(omega_set)
            n_od_kept  = len(feasible_od)
            print(f"      incentive trips: {len(incentive_trips):4d} kept "
                f"(od flow {n_od_kept}/{n_od_total}, after-cap "
                f"{len(incentive_trips)}/{len(raw_trips)}, cap=€{effective_cap:.2f})")
            

        if incentive_trips:
            z = model.addVars(incentive_trips, vtype=GRB.INTEGER, name="z")
            x = model.addVars(incentive_trips, vtype=GRB.BINARY, name="x")

            # EXACT MINUTE RIDE COST 
            incentive_cost = gp.quicksum(COST.RF * minute_RT_ij[o][i] * z[o, d, i, t] for (o, d, i, t) in incentive_trips)

            model.addConstrs((
                z[o, d, i, t0] <= (min(input_EI_total[o].loc[t0]['EI_h'], omega_h.get((o, d, t0), 0)) * x[o, d, i, t0])
                for (o, d, i, t0) in incentive_trips), name='max_flow_inc')
            model.addConstrs((z[o, d, i, t] >= x[o, d, i, t] for (o, d, i, t) in incentive_trips), name='link_z_x')
            model.addConstrs((x[o, d, i, t] == 0 for (o, d, i, t) in incentive_trips if t + RT_ij[o][i] > commit_end), name='no_late_inc')

            if window_budget_cap is None:
                model.addConstr(
                    incentive_cost + prev_incentive_cost <= COST.incentive_budget,
                    name='budget_cumulative'
                )
            else:
                model.addConstr(
                    incentive_cost <= window_budget_cap,
                    name='budget_adaptive_window'
                )
            
            obj += gp.quicksum(
                edl_coef * (input_beg_edl[d].loc[t + RT_ij[o][d], 'EDL_total'] + input_beg_edl[i].loc[t + RT_ij[o][i], 'EDL_total'] + 
                            input_end_edl[d].loc[t + RT_ij[o][d], 'EDL_total'] + input_end_edl[i].loc[t + RT_ij[o][i], 'EDL_total']) * x[o, d, i, t]
                for (o, d, i, t) in incentive_trips if t + RT_ij[o][d] <= t_end and t + RT_ij[o][i] <= t_end)
            
    # ── Station-served constraint ──
    # for i in N_truck:
    for i in N_tracking: 
        expr = r[i]
        if cfg.use_truck and i in N_truck: 
            expr += gp.quicksum(y[i, j, t, v] for j in N0_truck for t in range(t_start, t_end+1) for v in V)
        if cfg.use_incentives and z:
            expr += (gp.quicksum(x[o, d, i, t0] for (o, d, _, t0) in incentive_trips if _ == i and t0 + RT_ij[o][i] <= t_end)
                     + gp.quicksum(x[k, i, j, t0] for (k, di, j, t0) in incentive_trips if di == i and t0 + RT_ij[k][i] <= t_end))
        model.addConstr(expr == 1, name=f'served[{i}]')

    obj += gp.quicksum(edl_coef * input_end_edl[i].loc[t_start, 'EDL_total'] * r[i] for i in N_tracking)

    model.setObjective(obj, GRB.MINIMIZE)

    # ── Constraints ──
    if cfg.use_truck:
        if t_start == t_begin:
            model.addConstrs((gp.quicksum(y[depot, j, t_start, v] for j in N0_truck) == 1 for v in V), name='depart_day')
            model.addConstrs((gp.quicksum(y[i, j, t - tt_ij[i][j], v] for i in N0_truck if t - tt_ij[i][j] >= t_start) == gp.quicksum(y[j, k, t, v] for k in N0_truck) for j in N_truck for t in range(t_start, t_end) for v in V), name='routing_flow')
        else:
            model.addConstrs((gp.quicksum(y[end_loc_dict.get((v, t_start), 0), j, t_start, v] for j in N0_truck) == 1 for v in V), name='depart_win')
            model.addConstrs((gp.quicksum(y[i, j, t - tt_ij[i][j], v] for i in N0_truck if t - tt_ij[i][j] >= t_start) == gp.quicksum(y[j, k, t, v] for k in N0_truck) for j in N0_truck for t in range(t_start + 1, t_end + 1) for v in V), name='routing_flow')

        model.addConstrs((gp.quicksum(y[i, j, t, v] for i in N0_truck for j in N0_truck) <= 1 for v in V for t in range(t_start, t_end+1)), name='one_visit')

        if t_end == T_total:
            model.addConstrs((gp.quicksum(y[i, depot, T_total - tt_ij[i][depot], v] for i in N0_truck if T_total - tt_ij[i][depot] >= t_start) == 1 for v in V), name='return_depot')
            model.addConstrs((gp.quicksum(y[i, j, T_total - tt_ij[i][j], v] for i in N0_truck if T_total - tt_ij[i][j] >= t_start) == 0 for j in N0_truck if j != depot for v in V), name='no_arr_non_depot_T')

    if cfg.use_ops or cfg.use_swap:
        model.addConstrs((s_it[i, t, 'n'] == input_EI_total[i].loc[t]['EI_n'] - (gp.quicksum(n_un[i, t, v, 'n'] + (b_swap[i, t, v, 'n'] if cfg.use_swap else 0) for v in V) if i in N_truck else 0) for i in N_tracking for t in range(t_start, t_end+1)), name='flow_n')
        model.addConstrs((s_it[i, t, 'l'] == input_EI_total[i].loc[t]['EI_l'] + (gp.quicksum(n_load[i, t, v, 'l'] - (b_swap[i, t, v, 'l'] if cfg.use_swap else 0) - n_un[i, t, v, 'l'] for v in V) if i in N_truck else 0) for i in N_tracking for t in range(t_start, t_end+1)), name='flow_l')
    else:
        model.addConstrs((s_it[i, t, 'n'] == input_EI_total[i].loc[t]['EI_n'] for i in N_tracking for t in range(t_start, t_end+1)), name='pin_n')
        model.addConstrs((s_it[i, t, 'l'] == input_EI_total[i].loc[t]['EI_l'] for i in N_tracking for t in range(t_start, t_end+1)), name='pin_l')

    for i in N_tracking:
        for t in range(t_start, t_end+1):
            expr = input_EI_total[i].loc[t]['EI_h']
            if i in N_truck:
                if cfg.use_ops: expr += gp.quicksum(n_load[i, t, v, 'h'] - n_un[i, t, v, 'h'] for v in V)
                if cfg.use_swap: expr += gp.quicksum(b_swap[i, t, v, pp] for pp in P_nl for v in V)
            if cfg.use_incentives and z:
                expr += gp.quicksum(z[o, d, i, t0] for (o, d, _, t0) in incentive_trips if _ == i and t0 + RT_ij[o][i] == t)
                expr -= gp.quicksum(z[k, i, l, t0] for (k, di, l, t0) in incentive_trips if di == i and t0 + RT_ij[k][i] == t)
            model.addConstr(s_it[i, t, 'h'] == expr, name=f'flow_h[{i},{t}]')
            model.addConstr(gp.quicksum(s_it[i, t, p] for p in P) <= C_i[i], name=f'cap[{i},{t}]')

    if cfg.use_truck and cfg.use_truck_inventory:
        if t_start == t_begin:
            model.addConstrs((gp.quicksum(q[depot, j, t_start, v, p, M[0]] for j in N0_truck) == 0 for v in V for p in P), name='es_inv_start_day')
        else:
            model.addConstrs((gp.quicksum(q[end_loc_dict.get((v, t_start), 0), j, t_start, v, p, M[0]] for j in N0_truck) == prev_inventory[(v, t_start)][p][M[0]] + (gp.quicksum(n_un[end_loc_dict[v, t_start], t_start, v, p] for _ in [0]) if cfg.use_ops else 0) - (gp.quicksum(n_load[end_loc_dict[v, t_start], t_start, v, p] for _ in [0]) if cfg.use_ops else 0) for v in V for p in P if end_loc_dict[v, t_start] != depot), name='es_inv_start_period')

        model.addConstrs((gp.quicksum(q[j, i, t - tt_ij[j][i], v, p, M[0]] for j in N0_truck if t - tt_ij[j][i] >= t_start) + (n_un[i, t, v, p] if cfg.use_ops else 0) - (n_load[i, t, v, p] if cfg.use_ops else 0) == gp.quicksum(q[i, k, t, v, p, M[0]] for k in N0_truck) for i in N0_truck if i != depot for t in range(t_start + 1, t_end) for v in V for p in P), name='es_truck_inv_nondepot')
        model.addConstrs((gp.quicksum(q[j, depot, t - tt_ij[j][depot], v, p, M[0]] for j in N0_truck if t - tt_ij[j][depot] >= t_start) + (n_un[depot, t, v, p] if cfg.use_ops else 0) - (n_load[depot, t, v, p] if cfg.use_ops else 0) == gp.quicksum(q[depot, k, t, v, p, M[0]] for k in N0_truck) + (b_swap[depot, t, v, p] if cfg.use_swap else 0) for t in range(t_start + 1, t_end) for v in V for p in P_nl), name='es_truck_inv_depot_nl')
        model.addConstrs((gp.quicksum(q[j, depot, t - tt_ij[j][depot], v, 'h', M[0]] for j in N0_truck if t - tt_ij[j][depot] >= t_start) + (n_un[depot, t, v, 'h'] if cfg.use_ops else 0) - (n_load[depot, t, v, 'h'] if cfg.use_ops else 0) + (gp.quicksum(b_swap[depot, t, v, p] for p in P_nl) if cfg.use_swap else 0) == gp.quicksum(q[depot, k, t, v, 'h', M[0]] for k in N0_truck) for t in range(t_start + 1, t_end) for v in V), name='es_truck_inv_depot_h')
        model.addConstrs((gp.quicksum(q[i, j, t, v, p, M[0]] for p in P) <= C_v_es[v] * (y[i, j, t, v] if cfg.use_truck else 0) for i in N0_truck for j in N0_truck for t in range(t_start, t_end+1) for v in V), name='truck_cap_es')
        model.addConstrs((gp.quicksum(q[depot, j, t, v, p, M[0]] for j in N0_truck) == 0 for t in range(t_start, t_end+1) for v in V for p in P_nl), name='depot_no_nl_depart')
        model.addConstrs((gp.quicksum(q[i, depot, t - tt_ij[i][depot], v, p, M[0]] for i in N0_truck if t - tt_ij[i][depot] >= t_start) == (b_swap[depot, t, v, p] if cfg.use_swap else 0) for t in range(t_start, t_end+1) for v in V for p in P_nl), name='depot_convert_nl_to_h')
        model.addConstrs((gp.quicksum(q[depot, j, t, v, 'h', M[0]] for j in N0_truck) == gp.quicksum(q[i, depot, t - tt_ij[i][depot], v, 'h', M[0]] for i in N0_truck if t - tt_ij[i][depot] >= t_start) + (gp.quicksum(b_swap[depot, t, v, p] for p in P_nl) if cfg.use_swap else 0) for t in range(t_start, t_end+1) for v in V), name='depot_h_out_balance')

    if cfg.use_truck and cfg.use_battery_inventory and cfg.use_swap:
        model.addConstrs((q[i, j, t, v, 'h', M[1]] <= C_v_bt[v] * y[i, j, t, v] for i in N0_truck for j in N0_truck for t in range(t_start, t_end+1) for v in V), name='truck_cap_batt')
        if t_start != t_begin:
            model.addConstrs((gp.quicksum(q[end_loc_dict[v, t_start], j, t_start, v, p, M[1]] for j in N0_truck) == prev_inventory[(v, t_start)][p][M[1]] - b_swap[end_loc_dict[v, t_start], t_start, v, p] for v in V for p in P_nl if end_loc_dict[v, t_start] != depot), name='bat_inv_start')
        model.addConstrs((gp.quicksum(q[j, i, t - tt_ij[j][i], v, p, M[1]] for j in N0_truck if t - tt_ij[j][i] >= t_start) - gp.quicksum(b_swap[i, t, v, p] for p in P_nl) == gp.quicksum(q[i, k, t, v, p, M[1]] for k in N0_truck) for i in N0_truck if i != depot for t in range(t_start + 1, t_end + 1) for v in V for p in P if p == 'h'), name='bat_flow_nondepot')
        model.addConstrs((q[depot, j, t, v, 'h', M[1]] == C_v_bt[v] * y[depot, j, t, v] for j in N0_truck for t in range(t_start, t_end+1) for v in V), name='bat_inv_depot_visit')
        model.addConstrs((gp.quicksum(b_swap[i, t, v, p] for p in P_nl) <= C_v_bt[v] * gp.quicksum(y[i, j, t, v] for j in N0_truck) for i in N0_truck for t in range(t_start, t_end+1) for v in V), name='swap_cap_per_visit')
        model.addConstrs((gp.quicksum(b_swap[i, t, v, p] for p in P_nl for t in range(t_start, t_end+1)) <= C_v_bt[v] for i in N0_truck if i != depot for v in V), name='total_swap_cap')

    if cfg.use_ops and cfg.use_truck:
        model.addConstrs((n_load[i, t, v, p] <= C_v_es[v] * gp.quicksum(y[i, j, t, v] for j in N0_truck) for i in N0_truck for t in range(t_start, t_end+1) for v in V for p in P), name='load_cap')
        model.addConstrs((n_un[i, t, v, p] <= C_v_es[v] * gp.quicksum(y[i, j, t, v] for j in N0_truck) for i in N0_truck for t in range(t_start, t_end+1) for v in V for p in P), name='unload_cap')
        model.addConstrs((gp.quicksum(n_load[depot, t, v, p] + n_un[depot, t, v, p] for p in P) == 0 for t in range(t_start, t_end+1) for v in V), name='no_ops_depot')

    if cfg.use_truck and (cfg.use_ops or cfg.use_swap):
        visit   = model.addVars(N0_truck, range(t_start, t_end+1), V, vtype=GRB.BINARY, name="visit")
        do_swap = model.addVars(N0_truck, range(t_start, t_end+1), V, P, vtype=GRB.BINARY, name="do_swap")
        do_load = model.addVars(N0_truck, range(t_start, t_end+1), V, P, vtype=GRB.BINARY, name="do_load")
        do_unld = model.addVars(N0_truck, range(t_start, t_end+1), V, P, vtype=GRB.BINARY, name="do_unld")

        model.addConstrs((visit[i, t, v] == gp.quicksum(y[i, j, t, v] for j in N0_truck) for i in N0_truck for t in range(t_start, t_end+1) for v in V), name='visit_link')
        model.addConstrs((gp.quicksum(do_swap[i, t, v, p] + do_load[i, t, v, p] + do_unld[i, t, v, p] for p in P) >= visit[i, t, v] for i in N0_truck for t in range(t_start, t_end+1) for v in V), name='action_per_visit')
        model.addConstrs((n_load[i, t, v, p] <= C_v_es[v] * do_load[i, t, v, p] for i in N0_truck for t in range(t_start, t_end+1) for v in V for p in P), name='load_needs_do')
        model.addConstrs((n_un[i, t, v, p] <= C_v_es[v] * do_unld[i, t, v, p] for i in N0_truck for t in range(t_start, t_end+1) for v in V for p in P), name='unld_needs_do')
        if cfg.use_swap:
            model.addConstrs((b_swap[i, t, v, p] <= C_v_bt[v] * do_swap[i, t, v, p] for i in N0_truck for t in range(t_start, t_end+1) for v in V for p in P_nl), name='swap_needs_do')

        model.addConstrs((do_swap[i, t, v, p] + do_unld[i, t, v, p] == 1 for i in N_truck for t in range(t_start, t_end+1) for v in V for p in P_nl), name='swap_xor_unload')
        model.addConstrs((do_load[i, t, v, 'n'] + do_unld[i, t, v, 'n'] == 1 for i in N_truck for t in range(t_start, t_end+1) for v in V), name='no_load_and_unld_n')

    if cfg.use_supporting_planes and refine_planes:
        for i in N_tracking:
            for t in range(t_start, t_end+1):
                if i not in refine_planes or t not in refine_planes[i]: continue
                n_bef, l_bef, h_bef = input_EI_total[i].loc[t]['EI_n'], input_EI_total[i].loc[t]['EI_l'], input_EI_total[i].loc[t]['EI_h']
                edl_before = max(pr['m_n']*n_bef + pr['m_l']*l_bef + pr['m_h']*h_bef + pr['b_intercept'] for pr in refine_planes[i][t])

                for pl in refine_planes[i][t]:
                    model.addConstr(w_sp[i, t] >= (pl['m_n'] * s_it[i, t, 'n'] + pl['m_l'] * s_it[i, t, 'l'] + pl['m_h'] * s_it[i, t, 'h'] + pl['b_intercept'] - edl_before), name=f'sp[{i},{t}]')

    model.update()
    model.Params.MIPFocus   = 3
    model.Params.Heuristics = 0.1 #0.4
    model.Params.RINS       = 20
    model.Params.Presolve   = 1
    model.Params.Cuts       = 2
    model.Params.TimeLimit  = mip_time_limit   # seconds per MIP; default 600 (10 min)
    if mip_gap_tol is not None:
        model.Params.MIPGap = mip_gap_tol      # accept this much suboptimality; default tight

    for var in (list(x.values()) + list(z.values()) + list(n_load.values()) + list(n_un.values()) + list(b_swap.values()) + list(u.values()) + list(q.values())): var.Start = 0.0
    for i in N_tracking:
        for t in range(t_start, t_end+1):
            s_it[i, t, 'n'].Start, s_it[i, t, 'l'].Start, s_it[i, t, 'h'].Start = float(input_EI_total[i].loc[t]['EI_n']), float(input_EI_total[i].loc[t]['EI_l']), float(input_EI_total[i].loc[t]['EI_h'])
    if cfg.use_truck:
        for v in V:
            i0 = depot if t_start == t_begin else end_loc_dict.get((v, t_start), depot)
            if (i0, i0, t_start, v) in y: y[i0, i0, t_start, v].Start = 1.0

    model.update()
    model.optimize()

    has_incumbent = (model.SolCount > 0 and model.Status not in [GRB.INFEASIBLE, GRB.UNBOUNDED, GRB.INF_OR_UNBD])

    return {
        "model": model,
        "vars": {"y": y, "u": u, "n_load": n_load, "n_un": n_un, "b_swap": b_swap, "q": q, "s_it": s_it, "x": x, "z": z, "w_sp": w_sp, "r": r},
        "has_incumbent": has_incumbent,
    }


# ═══════════════════════════════════════════════════════════════════════
# MAIN ROLLING-HORIZON LOOP
# ═══════════════════════════════════════════════════════════════════════

def run_rolling_horizon(
    cfg_name="DL-HR",
    dow=1,
    data_dir="",
    t_begin_override=None,
    t_end_override=None,
    init_inventory_override=None,
    low_share_override=None,
    demand_scale=1.0,
    capacity_override=None,
    fleet_total=None,
    fleet_shares=None,
    day_cap=4659,
    hour_cap=100,
    peak_hour_cap=200,
    run_tag=None,
    budget_mode="adaptive",   
    user_class_override=None,
    sim_user_class_override=None,
    mip_time_limit=600.0,
    mip_gap_tol=None,
):
    cfg = STRATEGY_PRESETS[cfg_name]

    t_begin = t_begin_override if t_begin_override is not None else 0
    T_total = t_end_override   if t_end_override   is not None else TIME.T
    low_share = low_share_override if low_share_override is not None else 0.40

    def _slot_to_clock(slot):
        total_min = TIME.t_begin_day * 60 + slot * TIME.tau
        return f"{total_min // 60:02d}:{total_min % 60:02d}"

    print(f"\n{'='*60}\n  Strategy : {cfg_name}\n  Time : slot {t_begin} → slot {T_total}\n{'='*60}\n")

    print("[1/5] Loading data...")
    data = load_all_data(data_dir, demand_scale=demand_scale, capacity_override=capacity_override)
    N, V, P, P_nl, M = NETWORK.N, NETWORK.V, NETWORK.P, NETWORK.P_nl, NETWORK.M
    C_i, d_ij, tt_ij, RT_ij = data["C_i"], data["d_km"], data["tt_ij"], data["RT_ij"]
    
    # EXACT MINUTE RIDE TIMES
    minute_RT_ij = data["minute_RT_ij"]
    

    omega_h_all = data["omega_h"]
    accept_parameter = data["accept_parameter"]

    dow_key = str(dow)
    fn_pickup  = data["pickup_rates_by_hour"][dow]
    fn_dropoff = data["dropoff_rates_by_hour"][dow]
    fn_phi1    = data["phi1"][dow_key]
    fn_phi2    = data["phi2"][dow_key]
    omega_h    = omega_h_all[dow]

    dt = TIME.dt
    slots_per_hour = TIME.slots_per_hour
    period_length  = int(1 / dt)
    edl_coef = COST.edl_coef()
    # Default hardcoded baseline (the "mean" class)
    # Default baseline (the "mean" class)
    prob_l, prob_h = 0.18, 0.70

    # MIP-side override (for non-mean classes only)
    if user_class_override is not None and user_class_override != "mean":
        from User_choice_model import compute_probs_for_class
        from ProjectConfig import USER_CLASSES
        cls_params = USER_CLASSES[user_class_override]
        cls_probs = compute_probs_for_class(
            beta_ride=cls_params["beta_ride"],
            beta_batt=cls_params["beta_batt"],
            pct_high=50.0, pct_low=25.0,
        )
        prob_h = cls_probs["prob_h"]
        prob_l = cls_probs["prob_l"]
        print(f"  user_class (MIP) = {user_class_override}  →  P_h={prob_h:.4f}  P_l={prob_l:.4f}")
    elif user_class_override == "mean":
        print(f"  user_class (MIP) = mean  →  P_h=0.70  P_l=0.18  (baseline)")

    # Simulator-side probs — default same as MIP (no decoupling)
    prob_l_sim, prob_h_sim = prob_l, prob_h

    if sim_user_class_override is not None and sim_user_class_override != "mean":
        from User_choice_model import compute_probs_for_class
        from ProjectConfig import USER_CLASSES
        sim_cls_params = USER_CLASSES[sim_user_class_override]
        sim_cls_probs = compute_probs_for_class(
            beta_ride=sim_cls_params["beta_ride"],
            beta_batt=sim_cls_params["beta_batt"],
            pct_high=50.0, pct_low=25.0,
        )
        prob_h_sim = sim_cls_probs["prob_h"]
        prob_l_sim = sim_cls_probs["prob_l"]
        print(f"  sim_user_class = {sim_user_class_override}  →  P_h_sim={prob_h_sim:.4f}  P_l_sim={prob_l_sim:.4f}  (decoupled)")
    elif sim_user_class_override == "mean":
        prob_l_sim, prob_h_sim = 0.18, 0.70
        print(f"  sim_user_class = mean  →  P_h_sim=0.70  P_l_sim=0.18  (baseline)")


    # ── GLOBAL DROPOFF BALANCE FIX (To prevent exogenous ghost trips) ──
    curr_pick = sum(sum(rates.values()) for rates in fn_pickup.values())
    curr_drop = sum(sum(np.sum(rates) for rates in drop_rates.values()) for drop_rates in fn_dropoff.values())
    
    if curr_drop > 0 and curr_pick > 0:
        balance_ratio = curr_pick / curr_drop
        print(f"  [Correction] Balancing Drop-offs to match Pickups (Ratio: {balance_ratio:.3f})")
        
        for i in N:
            if i in fn_dropoff:
                for hr in fn_dropoff[i]:
                    fn_dropoff[i][hr] = fn_dropoff[i][hr] * balance_ratio

    # ── SYSTEM DEMAND FOR BUDGET ADAPTATION ──
    system_pickup_rate_by_hour = {}
    _all_hours = set()
    for i in N:
        if i in fn_pickup:
            _all_hours.update(fn_pickup[i].keys())
    for hr in _all_hours:
        system_pickup_rate_by_hour[hr] = sum(
            float(fn_pickup[i].get(hr, 0.0)) for i in N if i in fn_pickup
        )
    print(f"  budget_mode = {budget_mode}  (system pickup peak hour: "
          f"{max(system_pickup_rate_by_hour, key=system_pickup_rate_by_hour.get) if system_pickup_rate_by_hour else 'n/a'})")

    print("[2/5] Pre-computing Markov transition matrices...")
    _markov_setup_t0 = time.perf_counter()
    forecast_len = TIME.forecast_len
    all_states, P_mats_step, P_hour_cache = {}, {}, {}

    for i in N:
        Q_by_hour, states = compute_Q(fn_pickup[i], fn_dropoff[i], fn_phi1, fn_phi2, C_i[i], prob_l=prob_l, prob_h=prob_h)
        all_states[i] = states
        nS = len(states)
        for hr, Qhr in Q_by_hour.items():
            key = (i, hr)
            if key not in P_hour_cache:
                P_hour_cache[key] = np.linalg.matrix_power(np.eye(nS) + (Qhr * dt) / 100, 100)

        P_mats_step[i] = {}
        for k in range(T_total + forecast_len + 1):
            hr = (k // slots_per_hour + TIME.t_begin_day) % 24
            P_mats_step[i][k] = P_hour_cache[(i, hr)]
    markov_setup_time = time.perf_counter() - _markov_setup_t0
    print(f"  Markov setup time: {markov_setup_time:.2f}s")

    print("[3/5] Setting initial inventory...")
    if fleet_total is not None and fleet_shares is not None:
        from Utils import build_init_inventory_by_system_shares
        Init_inventory_boot, achieved = build_init_inventory_by_system_shares(
            N=N, all_states=all_states, C_i=C_i,
            total_fleet=fleet_total, shares=tuple(fleet_shares),
        )
        print(f"  fleet target={fleet_total}, achieved={achieved['fleet_achieved']}, "
              f"shares achieved=({achieved['n_share_achieved']:.2f}, "
              f"{achieved['l_share_achieved']:.2f}, {achieved['h_share_achieved']:.2f})")
    elif isinstance(init_inventory_override, dict):
        Init_inventory_boot = {i: nearest_enumerated_state(init_inventory_override.get(i, (0,0,0)), all_states[i]) for i in N}
    elif isinstance(init_inventory_override, str):
        inv_df = pd.read_csv(init_inventory_override)
        Init_inventory_boot = {}
        for _, row in inv_df.iterrows():
            Init_inventory_boot[int(row['station'])] = nearest_enumerated_state((int(row['n']), int(row['l']), int(row['h'])), all_states[int(row['station'])])
        for i in N:
            if i not in Init_inventory_boot: Init_inventory_boot[i] = nearest_enumerated_state((0, 0, 0), all_states[i])
    else:
        random.seed(0)
        Init_inventory_boot = {i: random.choice(all_states[i]) for i in N}

    print(f"\n[4/5] Starting rolling-horizon optimisation...\n")
    random.seed(0)
    np.random.seed(0)

    acquired_truck = {v: False for v in V}
    end_loc_dict, prev_inventory, sim_end_inv_avg, station_delta = {}, {}, {}, {}
    prev_incentive_cost = 0.0
    all_results = []
    
    t_current = t_begin
    prev_slice = None

    while t_current < T_total:
        t_start = t_current
        t_end = min(t_start + TIME.plan_len, T_total)
        t_forecast = t_start + forecast_len
        commit_end = T_total if t_start + TIME.commit_len == T_total else min(t_start + TIME.commit_len - 1, T_total)

        print(f"  Window [{t_start} → {t_end}] ({_slot_to_clock(t_start)}–{_slot_to_clock(t_end)}), commit until {commit_end}")
        wall_start = time.perf_counter()

        if t_start == t_begin:
            if (fleet_total is not None and fleet_shares is not None) or isinstance(init_inventory_override, (dict, str)):
                Init_inventory = Init_inventory_boot.copy()
            else:
                Init_inventory, _ = reseed_initial_low_share(
                    N, Init_inventory_boot, all_states, C_i,
                    low_share=low_share, weights_mode="demand",
                    fn_pickup_rates_by_hour=fn_pickup, fn_dropoff_rates_by_hour=fn_dropoff, verbose=True
                )
                for i in N: Init_inventory[i] = nearest_enumerated_state(Init_inventory[i], all_states[i])
        else:
            Init_inventory = compute_init_inventory_for_window(
                N=N, i_states=all_states, C_i=C_i,
                EI_total_prev=EI_total, t_start=t_start, t_begin_day=t_begin,
                prev_slice=prev_slice, sim_end_inv_avg=sim_end_inv_avg,
                station_delta_by_slice=station_delta,
                tol_frac=0.10, seed=4242
            )

        _predict_t0 = time.perf_counter()
        forecast = precompute_markov_and_edl(
            N, C_i, all_states, Init_inventory, fn_pickup, fn_dropoff,
            P_mats_step, t_start, t_forecast, slots_per_hour, dt, prob_l, prob_h
        )
        EI_total, pi_by_time = forecast["EI_total"], forecast["pi_by_time"]
        edl_sep, beg_edl, end_edl = forecast["station_edl_separate_t"], forecast["begining_to_t_cumu_edl_t"], forecast["from_t_to_end_cumu_edl_t"]

        scores = {i: float(edl_sep[i]['EDL_total'].sum()) for i in N}
        sorted_scores = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        
        # ── TIME-SPACE TRUCK PRUNING FIX ──
        # max_truck_visits = (commit_end - t_start) * len(V) 
        max_truck_visits = (t_end - t_start) * len(V) 
        # N_truck = [i for i, s in sorted_scores[:max_truck_visits] if s >= 3.0] 
        N_truck = [i for i, s in sorted_scores[:max_truck_visits]] 
        # N_truck = [i for i, s in scores.items() if s >= 3.0]
        
        for v in V:
            start_loc = end_loc_dict.get((v, t_start), 0)
            if start_loc != 0 and start_loc not in N_truck:
                N_truck.append(start_loc)
                
        N0_truck = [0] + N_truck 
        
        shadow_set = set()
        N_incentive_targets = [i for i, s in sorted_scores if s >= 2 and i not in N_truck]
        shadow_set.update(N_incentive_targets)

        for (o, d, i), val in accept_parameter.items():
            if val == 1 and (i in N_truck or i in N_incentive_targets) and d not in N_truck:
                shadow_set.add(d)
        N_shadow = list(shadow_set)
        
        N_tracking = N_truck + N_shadow

        committed_edl_inactive = edl_coef * sum(float(edl_sep[i].loc[t_start:commit_end, 'EDL_total'].sum()) for i in N if i not in N_tracking)
        omega_set = [(o, d, t) for o in N for d in N_tracking for t in range(t_start, commit_end + 1) if omega_h.get((o, d, t), 0) > 0]

        refine_planes = compute_supporting_planes(
            N_tracking, all_states, C_i, fn_pickup, fn_dropoff,
            EI_total, pi_by_time, P_mats_step, end_edl, period_length, dt, t_start, t_end, t_forecast
        ) if cfg.use_supporting_planes else {}

        input_EI  = {i: EI_total[i].copy() for i in N}
        input_beg = {i: beg_edl[i].copy().shift(1, fill_value=0) for i in N_tracking}
        input_end = {i: end_edl[i].copy() for i in N_tracking}
        if t_end == T_total:
            for i in N:
                input_EI[i].loc[T_total] = EI_total[i].iloc[-1]
            for i in N_tracking:
                input_beg[i].loc[T_total] = beg_edl[i].iloc[-1]
                input_end[i].loc[T_total] = 0
        predict_time = time.perf_counter() - _predict_t0

        if budget_mode == "adaptive":
            def _expected_demand_over_slots(s0, s1):
                tot = 0.0
                for tt in range(s0, s1 + 1):
                    hr = (tt // slots_per_hour + TIME.t_begin_day) % 24
                    tot += system_pickup_rate_by_hour.get(hr, 0.0) * dt
                return tot
            _dem_curr = _expected_demand_over_slots(t_start, commit_end)
            _dem_remaining = _expected_demand_over_slots(t_start, T_total - 1)
            _budget_left = max(COST.incentive_budget - prev_incentive_cost, 0.0)

            if _dem_remaining > 0:
                # Floor: small per-window minimum to ensure some budget is always available
                # (covers 1-2 trips at typical ride costs, but doesn't pre-commit much)
                # floor = min(5.0, _budget_left)
                periods_remaining = (T_total - t_start) // TIME.commit_len  # integer count of remaining 1-hour windows
                floor = min(COST.incentive_budget / periods_remaining, _budget_left)
                
                # Demand-weighted with peak amplification: peak windows get more than
                # their proportional share, off-peak windows get less. Total still
                # bounded by _budget_left.
                demand_weighted_cap = _budget_left * (_dem_curr / _dem_remaining)
                peak_boost = 1.5  # tune: higher = more peak concentration
                
                window_budget_cap = min(max(floor, peak_boost * demand_weighted_cap), _budget_left)
            else:
                window_budget_cap = _budget_left
        else:  
            window_budget_cap = None  

        # # ── MIP build + solve ──
        # if cfg_name == "NR":
        #     # FAST-TRACK BYPASS FOR DO NOTHING
        #     has_inc, vs, model = True, {"y": {}, "n_load": {}, "n_un": {}, "b_swap": {}, "z": {}, "q": {}}, Model()
        #     # model.ObjVal, model.MIPGap, mip_solve_time = 0.0, 0.0, 0.0
        #     obj_val, mip_gap, mip_solve_time = 0.0, 0.0, 0.0
        # else:
        #     _mip_t0 = time.perf_counter()
        #     result = build_and_solve_mip(
        #         cfg=cfg, data=data, N_truck=N_truck, N0_truck=N0_truck, N_tracking=N_tracking, omega_set=omega_set,
        #         t_begin=t_begin, t_start=t_start, t_end=t_end, commit_end=commit_end, T_total=T_total,  
        #         Init_inventory=Init_inventory, input_EI_total=input_EI, input_beg_edl=input_beg, input_end_edl=input_end,
        #         refine_planes=refine_planes, accept_parameter=accept_parameter, omega_h=omega_h, edl_coef=edl_coef,
        #         committed_edl_inactive=committed_edl_inactive, prev_acquired_truck=acquired_truck,
        #         prev_incentive_cost=prev_incentive_cost, end_loc_dict=end_loc_dict, prev_inventory=prev_inventory,
        #         window_budget_cap=window_budget_cap,   
        #         minute_RT_ij=minute_RT_ij,             
        #     )
        #     mip_solve_time = time.perf_counter() - _mip_t0
        #     has_inc, vs, model = result["has_incumbent"], result["vars"], result["model"]
        # ── MIP build + solve ──
        if cfg_name == "NR":
            # FAST-TRACK BYPASS FOR DO NOTHING
            has_inc, vs, model = True, {"y": {}, "n_load": {}, "n_un": {}, "b_swap": {}, "z": {}, "q": {}}, Model()
            # DN: no decisions, but objective still reflects monetized demand loss across all tracked zones
            dn_baseline_cost = sum(edl_coef * input_end[i].loc[t_start, 'EDL_total']
                for i in N_tracking)
            obj_val, mip_gap, mip_solve_time = dn_baseline_cost, 0.0, 0.0
        else:
            _mip_t0 = time.perf_counter()
            result = build_and_solve_mip(
                cfg=cfg, data=data, N_truck=N_truck, N0_truck=N0_truck, N_tracking=N_tracking, omega_set=omega_set,
                t_begin=t_begin, t_start=t_start, t_end=t_end, commit_end=commit_end, T_total=T_total,  
                Init_inventory=Init_inventory, input_EI_total=input_EI, input_beg_edl=input_beg, input_end_edl=input_end,
                refine_planes=refine_planes, accept_parameter=accept_parameter, omega_h=omega_h, edl_coef=edl_coef,
                committed_edl_inactive=committed_edl_inactive, prev_acquired_truck=acquired_truck,
                prev_incentive_cost=prev_incentive_cost, end_loc_dict=end_loc_dict, prev_inventory=prev_inventory,
                window_budget_cap=window_budget_cap,   
                minute_RT_ij=minute_RT_ij,
                mip_time_limit=mip_time_limit,
                mip_gap_tol=mip_gap_tol,
            )
            mip_solve_time = time.perf_counter() - _mip_t0
            has_inc, vs, model = result["has_incumbent"], result["vars"], result["model"]
            
            # Extract Gurobi attributes safely here
            if has_inc:
                obj_val = model.ObjVal
                mip_gap = model.MIPGap   
        wall_elapsed = time.perf_counter() - wall_start

        if has_inc:
            # print(f"    Solved: obj={model.ObjVal:.2f}, gap={model.MIPGap:.1%}, time={wall_elapsed:.1f}s")
            # print(f"    Solved: obj={obj_val:.2f}, gap={mip_gap:.1%}, time={wall_elapsed:.1f}s")
            if cfg_name == "NR":
                print(f"    Skipped MIP: baseline_cost={obj_val:.2f}, time={wall_elapsed:.1f}s")
            else:
                print(f"    Solved: obj={model.ObjVal:.2f}, gap={model.MIPGap:.1%}, time={wall_elapsed:.1f}s")
            _extract_t0 = time.perf_counter()
            truck_route_rows, ops_rows, swap_rows, incentive_rows, inv_rows = [], [], [], [], []
            
            n_load_vals = values_dict(vs.get("n_load", {}))
            n_un_vals   = values_dict(vs.get("n_un", {}))
            b_swap_vals = values_dict(vs.get("b_swap", {}))
            z_vals      = values_dict(vs.get("z", {}))
            
            if cfg.use_truck and vs["y"]:
                for (i, j, t, v), var in vs["y"].items():
                    if varX(var) > 0.5 and t_start <= t <= commit_end:
                        arr_time = t + tt_ij[i][j]
                        truck_route_rows.append({"truck": v, "from_node": i, "to_node": j, "depart_slot": t, "arrive_slot": arr_time, "from_type": "depot" if i == 0 else "station", "to_type": "depot" if j == 0 else "station"})
            truck_route_df = pd.DataFrame(truck_route_rows)
            
            if cfg.use_ops:
                for (i, t, v, p), val in n_load_vals.items():
                    if val > 0.5 and t_start <= t <= commit_end and i != 0: ops_rows.append({"station": i, "slot": t, "truck": v, "power": p, "action": "load_to_truck", "quantity": int(round(val))})
                for (i, t, v, p), val in n_un_vals.items():
                    if val > 0.5 and t_start <= t <= commit_end and i != 0: ops_rows.append({"station": i, "slot": t, "truck": v, "power": p, "action": "unload_to_station", "quantity": int(round(val))})
            ops_df = pd.DataFrame(ops_rows)
            
            if cfg.use_swap and vs.get("b_swap"):
                for (i, t, v, p), val in b_swap_vals.items():
                    if val > 0.5 and t_start <= t <= commit_end and p in P_nl: swap_rows.append({"station": i, "slot": t, "truck": v, "from_power": p, "to_power": "h", "quantity": int(round(val))})
            swap_df = pd.DataFrame(swap_rows)
            
            if cfg.use_incentives and vs.get("z"):
                for (o, d, i, t0), val in z_vals.items():
                    if val > 0.5 and t_start <= t0 <= commit_end:
                        # RIDE COST UPDATED
                        ride_cost = COST.RF * minute_RT_ij[o][i] * val
                        incentive_rows.append({"origin": o, "orig_dest": d, "new_dest": i, "depart_slot": t0, "redirected_trips": int(round(val)), "ride_cost_euro": round(ride_cost, 2)})
            incentive_df = pd.DataFrame(incentive_rows)

            # committed_truck_acq = sum(COST.c_v for v in {v for v in V if any(varX(var) > 0.5 for (i, j, t, vv), var in vs["y"].items() if vv == v and i == 0 and j in N_truck and t_start <= t <= commit_end)} if not acquired_truck.setdefault(v, True)) if cfg.use_truck and vs["y"] else 0.0
            # ── CORRECTED TRUCK DEPLOYMENT COST ──
            committed_truck_acq = 0.0
            if cfg.use_truck and vs.get("y"):
                for v in V:
                    # Only check if we haven't already paid for this truck today
                    if not acquired_truck[v]:
                        # Did the truck leave the depot during this committed window?
                        leaves_depot = any(
                            varX(var) > 0.5 
                            for (i, j, t, vv), var in vs["y"].items() 
                            if vv == v and i == 0 and j in N_truck and t_start <= t <= commit_end
                        )
                        if leaves_depot:
                            committed_truck_acq += COST.c_v
                            acquired_truck[v] = True  # Mark as paid for the rest of the day!
                            
            committed_routing = sum_over_vars((COST.c_ij * d_ij[i, j], var) for (i, j, t, v), var in vs["y"].items() if t_start <= t <= commit_end) if cfg.use_truck and vs["y"] else 0.0
            committed_ops_load = sum_over_vars((COST.c_load_unload, var) for (i, t, v, p), var in vs["n_load"].items() if t_start <= t <= commit_end) if vs.get("n_load") else 0.0
            committed_ops_unload = sum_over_vars((COST.c_load_unload, var) for (i, t, v, p), var in vs["n_un"].items() if t_start <= t <= commit_end) if vs.get("n_un") else 0.0
            committed_swap_cost = sum_over_vars((COST.c_swap, var) for (i, t, v, p), var in vs["b_swap"].items() if p in P_nl and t_start <= t <= commit_end) if vs.get("b_swap") else 0.0

            committed_inc_cost = sum(COST.RF * minute_RT_ij[o][i] * varX(var) for (o, d, i, t0), var in vs["z"].items() if t_start <= t0 <= commit_end) if cfg.use_incentives and vs.get("z") else 0.0
            committed_inc_trips = int(round(sum(varX(var) for (o, d, i, t0), var in vs["z"].items() if t_start <= t0 <= commit_end))) if cfg.use_incentives and vs.get("z") else 0

            prev_incentive_cost += committed_inc_cost
            total_truck_cost = round(committed_truck_acq + committed_routing + committed_ops_load + committed_ops_unload + committed_swap_cost, 2)

            if cfg.use_truck:
                default_loc = {v: end_loc_dict.get((v, t_start), 0) for v in V}
                end_loc, _ = record_truck_end_state(vs["y"], tt_ij, V, commit_end, default_loc, 0)
                next_t = min(commit_end + 1, T_total)
                for v, node in end_loc.items(): 
                    end_loc_dict[(v, next_t)] = node
                    prev_inventory[(v, next_t)] = {p: {m: 0 for m in M} for p in P}
                    if vs["y"] and vs.get("q"):
                        deps = [t for (i, j, t, vv), var in vs["y"].items() if vv == v and varX(var) > 0.5 and t <= commit_end]
                        if deps:
                            last_dep = max(deps)
                            for (i, j, t, vv), var in vs["y"].items():
                                if vv == v and t == last_dep and varX(var) > 0.5:
                                    for p in P:
                                        for m in M:
                                            if (i, j, t, v, p, m) in vs["q"]:
                                                prev_inventory[(v, next_t)][p][m] = int(round(varX(vs["q"][i, j, t, v, p, m])))

            curr_slice = (t_start, commit_end)
            station_delta[curr_slice] = summarize_station_deltas(
                n_load=n_load_vals, n_un=n_un_vals, b_swap=b_swap_vals, z=z_vals,
                N=[0] + list(N), P_nl=P_nl, RT_ij=RT_ij, t_lo=t_start, t_hi=commit_end
            )
            extract_time = time.perf_counter() - _extract_t0

            _sim_t0 = time.perf_counter()
            sim = simulate_window_service_level(
                t_start=t_start, t_end=commit_end, N=N, C_i=C_i, slots_per_hour=slots_per_hour,
                fn_pickup_rates_by_hour=fn_pickup, fn_dropoff_rates_by_hour=fn_dropoff, Init_inventory=Init_inventory,
                n_load=n_load_vals, n_un=n_un_vals, b_swap=b_swap_vals, z=z_vals, RT_ij=RT_ij,
                n_runs=30, sim_seed=4242, prob_l=prob_l_sim, prob_h=prob_h_sim,
                hour_cap=hour_cap, peak_hour_cap=peak_hour_cap, day_cap=day_cap,
                t_begin_day=TIME.t_begin_day,
            )
            simulate_time = time.perf_counter() - _sim_t0
            sim_end_inv_avg[curr_slice] = sim.get("avg_end_inventory", {})
            sim_means = sim.get("system_window_means", {})

            pg = sim_means.get("pickups_generated", 0)
            mp = sim_means.get("missed_pickups", 0)
            dg = sim_means.get("dropoffs_generated", 0)
            md = sim_means.get("missed_dropoffs", 0)
            sl_pickup = (pg - mp) / pg if pg > 0 else 1.0
            sl_dropoff = (dg - md) / dg if dg > 0 else 1.0

            committed_operational_cost = round(total_truck_cost + committed_inc_cost, 2)
            total_window_time = predict_time + mip_solve_time + extract_time + simulate_time

            window_result = {
                "t_start": t_start, "commit_end": commit_end,
                # "obj": model.ObjVal, "gap": model.MIPGap,
                # "obj": obj_val, "gap": mip_gap,
                "obj": (obj_val if cfg_name == "NR" else model.ObjVal),
                "gap": (mip_gap if cfg_name == "NR" else model.MIPGap),
                "user_class_mip": user_class_override if user_class_override else "default",
                "truck_acq_cost":    round(committed_truck_acq, 2),
                "truck_routing_cost": round(committed_routing, 2),
                "truck_load_cost":   round(committed_ops_load, 2),
                "truck_unload_cost":  round(committed_ops_unload, 2),
                "truck_swap_cost":    round(committed_swap_cost, 2),
                "total_truck_cost":   total_truck_cost,
                "incentive_cost": committed_inc_cost, "incentive_trips": committed_inc_trips,
                "committed_operational_cost": committed_operational_cost,
                "service_level": sim["system_service_level_mean"],
                "sl_pickup": sl_pickup, "sl_dropoff": sl_dropoff,
                "sim_pickups_generated": sim_means.get("pickups_generated", 0), "sim_pickups_missed": sim_means.get("missed_pickups", 0),
                "sim_dropoffs_generated": sim_means.get("dropoffs_generated", 0), "sim_dropoffs_missed": sim_means.get("missed_dropoffs", 0),
                "predict_time":      round(predict_time, 3),    
                "mip_solve_time":    round(mip_solve_time, 3),   
                "extract_time":      round(extract_time, 3),     
                "simulate_time":     round(simulate_time, 3),    
                "total_window_time": round(total_window_time, 3),
                "wall_time":         wall_elapsed,               
                "markov_setup_time": round(markov_setup_time, 3), 
                "run_tag": run_tag,
                "user_class_sim": sim_user_class_override if sim_user_class_override else "(=mip)",
                "prob_h_mip": prob_h,
                "prob_l_mip": prob_l,
                "prob_h_sim": prob_h_sim,
                "prob_l_sim": prob_l_sim,
                "_truck_route": truck_route_df, "_operations": ops_df, "_swaps": swap_df, "_incentives": incentive_df,
            }
            all_results.append(window_result)
        else:
            print(f"    No feasible solution found (time={wall_elapsed:.1f}s)")
            all_results.append({"t_start": t_start, "commit_end": commit_end, "obj": None, "wall_time": wall_elapsed,
                                "predict_time": round(predict_time, 3), "mip_solve_time": round(mip_solve_time, 3),
                                "markov_setup_time": round(markov_setup_time, 3), "run_tag": run_tag})

        prev_slice = (t_start, commit_end)
        t_current = min(commit_end + 1, T_total)
        if cfg_name != "NR":
            model.dispose()

    print(f"\n[5/5] Saving results...")
    out_dir = os.path.join("results", cfg_name)
    os.makedirs(out_dir, exist_ok=True)

    safe_tag = "" if not run_tag else "_" + "".join(c if c.isalnum() or c in "-_." else "_" for c in str(run_tag))

    summary_cols = [k for k in all_results[0].keys() if not k.startswith("_")] if all_results and all_results[0].get("obj") is not None else []
    summary_df = pd.DataFrame([{k: r.get(k) for k in summary_cols} for r in all_results if r.get("obj") is not None])
    summary_df.to_csv(os.path.join(out_dir, f"window_results_{t_begin}_{t_end}{safe_tag}.csv"), index=False)

    xlsx_path = os.path.join(out_dir, f"detailed_results_{t_begin}_{t_end}{safe_tag}.xlsx")
    with pd.ExcelWriter(xlsx_path, engine="xlsxwriter") as xl:
        if not summary_df.empty: summary_df.to_excel(xl, sheet_name="summary", index=False)

        def dump_sheet(key, name):
            parts = [r.get(key).assign(window=f"{r['t_start']}–{r['commit_end']}") for r in all_results if r.get(key) is not None and not r.get(key).empty]
            if parts: pd.concat(parts, ignore_index=True).to_excel(xl, sheet_name=name, index=False)
        
        dump_sheet("_truck_route", "truck_routes")
        dump_sheet("_operations", "operations")
        dump_sheet("_swaps", "battery_swaps")
        dump_sheet("_incentives", "incentive_plan")

    print(f"  CSV  → {out_dir}/window_results_{t_begin}_{t_end}{safe_tag}.csv")
    print(f"  Excel → {xlsx_path}\n")

    return all_results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rolling-horizon e-scooter fleet rebalancing")
    parser.add_argument("--config", default="DL-HR", choices=list(STRATEGY_PRESETS.keys()), help="Strategy configuration (default: DL-HR)")
    parser.add_argument("--dow", type=int, default=1, help="Day of week: 0=weekday, 1=weekend (default: 1)")
    parser.add_argument("--data-dir", default="data/", help="Path to data files (default: current dir)")
    parser.add_argument("--t-begin", type=int, default=None, help="Start slot of planning day")
    parser.add_argument("--t-end", type=int, default=None, help="End slot of planning day")
    parser.add_argument("--low-share", type=float, default=None, help="System-wide low-power share 0.0–1.0")
    parser.add_argument("--init-csv", type=str, default=None, help="CSV file with columns: station, n, l, h")
    parser.add_argument("--init-uniform", type=str, default=None, help="Uniform per-station inventory as n,l,h (e.g. '1,3,5')")

    parser.add_argument("--demand-scale", type=float, default=1.0, help="Multiplicative scale on demand rates")
    parser.add_argument("--capacity", type=int, default=None, help="Uniform station capacity")
    parser.add_argument("--fleet", type=int, default=None, help="Total fleet size")
    parser.add_argument("--shares", type=str, default=None, help="System-wide (Sn,Sl,Sh) shares")
    parser.add_argument("--trucks", type=int, default=None, help="Number of rebalancing trucks")
    parser.add_argument("--run-tag", type=str, default=None, help="Free-form tag stored on each result row")
    
    parser.add_argument("--budget-mode", type=str, default="adaptive", choices=["adaptive", "cumulative"], help="Incentive budget allocation policy.")
    parser.add_argument("--num-hexagons", type=int, default=None,
                    help="Override NETWORK.num_hexagons (zone count). "
                         "Required when --data-dir points to a coarsened dataset "
                         "(e.g., data_50/ → --num-hexagons 50).")
    parser.add_argument("--tau", type=int, default=None,
                    help="Override TIME.tau (slot length in minutes). "
                         "Wall-clock horizons (plan_horizon_hours, rev_interval_hours) "
                         "are preserved; T, slots_per_hour, dt, plan_len, commit_len, "
                         "forecast_len recompute from the new tau.")
    parser.add_argument("--user-class", type=str, default=None,
                    choices=["mean", "tough", "easy", "range_anxious_only"],
                    help="Override prob_l/prob_h via the choice model. "
                         "'mean' = baseline (keeps existing P_h=0.70/P_l=0.18); "
                         "others compute from CHOICE_MODEL.")
    parser.add_argument("--sim-user-class", type=str, default=None,
                    choices=["mean", "tough", "easy", "range_anxious_only"],
                    help="Simulator-side: choice probabilities. Independent of --user-class. "
                         "Used to evaluate a fixed plan under different behavioral profiles.")
    parser.add_argument("--budget", type=float, default=None,
                        help="Override COST.incentive_budget (€/day). Default: keep config value (100).")
    parser.add_argument("--plan-horizon-hours", type=float, default=None,
                        help="Override TIME.plan_horizon_hours. Shorter horizons solve faster "
                             "but with reduced look-ahead. Default keeps config value (2 h).")
    parser.add_argument("--forecast-horizon-hours", type=float, default=None,
                        help="Override the EDL forecast (look-ahead) horizon in hours. "
                             "Default is 2 x plan_horizon_hours.")
    parser.add_argument("--mip-time-limit", type=float, default=600.0,
                        help="Per-MIP Gurobi time limit (seconds). Default 600 (10 min). "
                             "Lower this for fast demos at the cost of optimality.")
    parser.add_argument("--mip-gap-tol", type=float, default=None,
                        help="Per-MIP target MIPGap (e.g. 0.05 = 5%%). Gurobi stops as soon "
                             "as the bound is within this gap. Default: tight (Gurobi default).")
    parser.add_argument("--quick-demo", action="store_true",
                        help="Demo mode: aggressively-shrunk instance for showing the "
                             "solution process quickly. Sets: --num-hexagons 15, "
                             "--plan-horizon-hours 1, --forecast-horizon-hours 1, "
                             "--t-end 4 (one rolling-horizon window), --budget 300, "
                             "--mip-time-limit 30, --mip-gap-tol 0.05. Results are "
                             "illustrative only and do NOT reproduce paper findings.")
    args = parser.parse_args()

    # ─── --quick-demo: aggressive bundle for fast demonstration ───
    # Shrinks every axis that drives MIP solve time. Each setting can still
    # be overridden by passing the corresponding flag explicitly.
    if args.quick_demo:
        if args.num_hexagons is None:           args.num_hexagons = 20      # ~10% of paper instance
        if args.plan_horizon_hours is None:     args.plan_horizon_hours = 2.0
        if args.forecast_horizon_hours is None: args.forecast_horizon_hours = 4.0
        if args.t_end is None:                  args.t_end = 20              # one commit window (1 h)
        if args.budget is None:                 args.budget = 150.0         
        if args.mip_time_limit == 600.0:        args.mip_time_limit = 30.0  # hard cap: 30 s per MIP
        if args.mip_gap_tol is None:            args.mip_gap_tol = 0.05     # accept 5% gap
        print("  --quick-demo ON: aggressively-shrunk instance for fast demonstration.")
        print(f"                   20 zones, 2-h plan, 4-h look-ahead, 1 window, "
              f"EUR {args.budget:.0f} budget, MIP cap {args.mip_time_limit:.0f}s/{args.mip_gap_tol*100:.0f}%.")
        print("                   Results are illustrative; they do NOT reproduce paper findings.")

    if args.num_hexagons is not None:
        # Allow running on a coarsened dataset
        # without editing ProjectConfig.py. Must run BEFORE load_all_data
        # so the new station count propagates through distance matrices
        # and the rolling-horizon loop.
        NETWORK.num_hexagons = args.num_hexagons
        NETWORK.N0 = list(range(args.num_hexagons + 1))
        NETWORK.N  = list(range(1, args.num_hexagons + 1))
        print(f"  num_hexagons override: {args.num_hexagons} zones")
    
    if args.plan_horizon_hours is not None:
        # Override planning horizon length (wall-clock hours). Recomputes
        # plan_len; forecast_len is also recomputed unless explicitly set.
        TIME.plan_horizon_hours = args.plan_horizon_hours
        TIME.plan_len = int((args.plan_horizon_hours * 60) // TIME.tau)
        # Default forecast = 2 x plan; will be overridden below if user set it
        TIME.forecast_len = 2 * TIME.plan_len
        print(f"  plan_horizon_hours override: {args.plan_horizon_hours} h  "
              f"-> plan_len={TIME.plan_len} slots, forecast_len={TIME.forecast_len} slots")
    
    if args.tau is not None:
        # Time-granularity sensitivity (Reviewer #2 #1).
        # Preserve wall-clock horizons (plan_horizon_hours, rev_interval_hours),
        # recompute slot counts. Must run BEFORE load_all_data so tt_ij and
        # all per-slot data structures use the new resolution.
        TIME.tau               = args.tau
        TIME.T                 = (TIME.total_hours * 60) // args.tau
        TIME.dt                = args.tau / 60.0
        TIME.slots_per_hour    = 60 // args.tau
        TIME.plan_len          = (TIME.plan_horizon_hours * 60) // args.tau
        TIME.commit_len        = (TIME.rev_interval_hours * 60) // args.tau
        TIME.forecast_len      = 2 * TIME.plan_len
        TIME.time_step_to_min  = args.tau
        print(f"  tau override: {args.tau} min  →  T={TIME.T} slots, "
            f"plan_len={TIME.plan_len}, commit_len={TIME.commit_len}, "
            f"forecast_len={TIME.forecast_len}")
    
    # Apply --forecast-horizon-hours AFTER --tau so the explicit value isn't
    # clobbered by the tau block's "forecast_len = 2 * plan_len" default.
    if args.forecast_horizon_hours is not None:
        TIME.forecast_len = int((args.forecast_horizon_hours * 60) // TIME.tau)
        print(f"  forecast_horizon_hours override: {args.forecast_horizon_hours} h  "
              f"-> forecast_len={TIME.forecast_len} slots")
    
    init_inv = None
    fleet_total = None
    fleet_shares = None
    if args.fleet is not None and args.shares is not None:
        fleet_total = args.fleet
        fleet_shares = tuple(float(x.strip()) for x in args.shares.split(","))
        assert abs(sum(fleet_shares) - 1.0) < 1e-6, f"--shares must sum to 1.0; got {sum(fleet_shares)}"
    elif args.init_csv:
        init_inv = args.init_csv
    elif args.init_uniform:
        n_val, l_val, h_val = [int(x.strip()) for x in args.init_uniform.split(",")]
        init_inv = {i: (n_val, l_val, h_val) for i in range(1, NETWORK.num_hexagons + 1)}

    BASE_DAY_CAP = 4659
    BASE_HOUR_CAP = 700        
    BASE_PEAK_HOUR_CAP = 700
    day_cap_arg       = int(round(BASE_DAY_CAP       * args.demand_scale))
    hour_cap_arg      = int(round(BASE_HOUR_CAP      * args.demand_scale))
    peak_hour_cap_arg = int(round(BASE_PEAK_HOUR_CAP * args.demand_scale))

    print(f"  demand_scale={args.demand_scale}  day_cap={day_cap_arg}  hour_cap={hour_cap_arg}  peak_hour_cap={peak_hour_cap_arg}")
    if args.capacity is not None: print(f"  station capacity override: {args.capacity}")
    if args.trucks is not None:
        NETWORK.V = list(range(1, args.trucks + 1))
        print(f"  truck fleet size: {len(NETWORK.V)}")
    print(f"  budget_mode = {args.budget_mode}")

    # Override the daily incentive budget.
    if args.budget is not None:
        COST.incentive_budget = args.budget
        print(f"  incentive_budget override: €{args.budget}")

    if args.user_class is not None and args.user_class != "mean":
        # Sanity: print what the choice model says for this class
        from User_choice_model import compute_probs_for_class
        from ProjectConfig import USER_CLASSES
        _cls = USER_CLASSES[args.user_class]
        _r = compute_probs_for_class(
            beta_ride=_cls["beta_ride"], beta_batt=_cls["beta_batt"],
            pct_high=50.0, pct_low=25.0,
        )
        print(f"  user_class = {args.user_class}  →  P_h={_r['prob_h']:.4f}  "
            f"P_l={_r['prob_l']:.4f}  P_optout={_r['P_0']:.4f}")
    elif args.user_class == "mean":
        print(f"  user_class = mean (baseline)  →  P_h=0.70  P_l=0.18 ")

    results = run_rolling_horizon(
        cfg_name=args.config, dow=args.dow, data_dir=args.data_dir,
        t_begin_override=args.t_begin, t_end_override=args.t_end,
        init_inventory_override=init_inv, low_share_override=args.low_share,
        demand_scale=args.demand_scale,
        capacity_override=args.capacity,
        fleet_total=fleet_total,
        fleet_shares=fleet_shares,
        day_cap=day_cap_arg,
        hour_cap=hour_cap_arg,
        peak_hour_cap=peak_hour_cap_arg,
        run_tag=args.run_tag,
        budget_mode=args.budget_mode,  
        user_class_override=args.user_class,
        sim_user_class_override=args.sim_user_class, 
        mip_time_limit=args.mip_time_limit,
        mip_gap_tol=args.mip_gap_tol,
    )