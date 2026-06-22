"""
data_loader.py — Data loading and preprocessing
================================================

Reads all input data files, computes distance matrices, and structures
demand rates into the nested dictionaries expected by the Markov chain
and the optimisation modules.

Input files (in `data_dir`)
---------------------------
    station_map.csv                       — H3 hex → station ID mapping (+ lat/lng)
    od_flow_high_power.csv                — high-power trip flow rates (omega_h)
    battery_decline_probs.csv             — battery power-state transition probabilities
    battery_decline_high_to_low.csv       — phi1: H → L discharge rate
    battery_decline_low_to_inactive.csv   — phi2: L → N discharge rate
    pickup_rates.csv                      — hourly pickup demand per station
    dropoff_rates.csv                     — hourly drop-off rates by power class
    incentive_accept.pkl                  — binary incentive-acceptance lookup
                                            (geometric criterion, walk ≤ 500 m)

Note
----
Pickup, drop-off, and OD-flow rates in the shipped data files have been
scaled by 1.4× to match the case-study calibration used in the paper.
No further runtime scaling is needed.

Usage
-----
    from data_loader import load_all_data
    data = load_all_data(data_dir="data/")
"""

import pickle
import numpy as np
import pandas as pd
import h3

from ProjectConfig import NETWORK, TIME, COST, CAPACITY, CHOICE_MODEL


# ═══════════════════════════════════════════════════════════════════════
# HAVERSINE DISTANCE (vectorised)
# ═══════════════════════════════════════════════════════════════════════

def haversine_vectorize(lon1, lat1, lon2, lat2):
    """
    Compute great-circle distances (km) between coordinate arrays.

    Parameters
    ----------
    lon1, lat1 : array-like  — longitude/latitude of origin(s), in degrees
    lon2, lat2 : array-like  — longitude/latitude of destination(s), in degrees

    Returns
    -------
    np.ndarray — distances in kilometres (Earth radius ≈ 6 367 km)
    """
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return 6367 * 2 * np.arcsin(np.sqrt(a))


# ═══════════════════════════════════════════════════════════════════════
# DISTANCE & TRAVEL-TIME MATRICES
# ═══════════════════════════════════════════════════════════════════════

def _build_distance_matrix(station_map):
    """
    Build an (N+1) × (N+1) distance matrix: row/col 0 = depot, 1..N = stations.

    If the station_map includes explicit `lat`/`lng` columns, those are used
    directly. Otherwise station coordinates are recovered from H3 hex centroids
    via the `h3_index` column. Depot coordinates come from `NETWORK.depot_lat`
    and `NETWORK.depot_lon`.
    """
    num = len(station_map)

    if 'lat' in station_map.columns and 'lng' in station_map.columns:
        lats = station_map['lat'].to_numpy(dtype=float)
        lons = station_map['lng'].to_numpy(dtype=float)
    else:
        # Original path: H3 hex centroids → (lat, lon)
        centers = np.array([h3.h3_to_geo(h) for h in station_map['h3_index']], dtype=float)
        lats, lons = centers[:, 0], centers[:, 1]

    # Station-to-station distances (N × N)
    ss_km = haversine_vectorize(lons[:, None], lats[:, None],
                                lons[None, :], lats[None, :]).astype(float)
    np.fill_diagonal(ss_km, 0.0)

    # Depot-to-station distances (1 × N), then flatten
    d2s_km = haversine_vectorize(
        np.full((1, 1), NETWORK.depot_lon), np.full((1, 1), NETWORK.depot_lat),
        lons[None, :], lats[None, :]
    ).reshape(-1)

    # Assemble full matrix with depot at index 0
    d_km = np.empty((num + 1, num + 1), dtype=float)
    d_km[0, 0]  = 0.0
    d_km[0, 1:] = d2s_km            # depot → stations
    d_km[1:, 0] = d2s_km            # stations → depot (symmetric)
    d_km[1:, 1:] = ss_km            # station ↔ station

    return d_km


def _build_travel_times(d_km, truck_visit_min=15):
    """
    Truck travel/service time per arc in slot units.

    Wall-clock truck-arc time is fixed at `truck_visit_min` (default 15 min,
    matching the case-study calibration). The slot count is derived from
    `TIME.tau`:

        tt = max(1, round(truck_visit_min / tau))

    Self-loops also take 1 slot (a truck "waiting" still costs one slot).
    """
    tt_value = max(1, round(truck_visit_min / TIME.tau))
    N0 = NETWORK.N0
    tt = {i: {j: 1 if i == j else tt_value for j in N0} for i in N0}
    tt[0][0] = 1
    return tt

def _build_walk_and_ride_times(d_km):
    """
    Compute walking and e-scooter riding times between stations.

    Walking speed : 4.8 km/h (used for incentive walk distance).
    E-scooter speed: 5 km/h  (used for ride-time estimation).
    """
    N0 = NETWORK.N0
    N  = NETWORK.N

    walk_ij = {i: {j: 1 if i == j else round((d_km[i][j] / 4.8) * 60, 1)
                   for j in N0} for i in N0}

    # Ride time in minutes (for revenue calculation)
    minute_RT_ij = {i: {j: 5 if i == j else 3+round((d_km[i][j] / 5) * 60, 2)
                        for j in N} for i in N}

    # Ride time in discrete slots (for the MIP model), e-scooter at ~25 km/h
    RT_ij = {i: {j: 1 if i == j else int(np.ceil((d_km[i][j] / 25) * (60 / TIME.tau)))
                 for j in N} for i in N}

    return walk_ij, minute_RT_ij, RT_ij


# ═══════════════════════════════════════════════════════════════════════
# DEMAND-RATE PREPROCESSING
# ═══════════════════════════════════════════════════════════════════════

def _process_omega_h(df_omega_h, T):
    """
    Convert hourly high-power trip flows into 15-minute slot-level
    dictionaries keyed by (origin, destination, slot).

    Returns
    -------
    omega_h : dict {weekend_flag: {(i, j, t): flow}}
        Only non-zero flows are stored.
    """
    N = NETWORK.N
    tau = TIME.tau

    tmp = df_omega_h.reset_index()
    tmp = tmp[['is_weekend', 'start_station', 'end_station', 'hour', 'omega']].copy()
    for col in ['is_weekend', 'start_station', 'end_station', 'hour']:
        tmp[col] = tmp[col].astype(int)
    tmp['omega'] = tmp['omega'].astype(float)
    tmp = tmp[(tmp['hour'] >= 0) & (tmp['hour'] < TIME.total_hours)]

    # Explode each hour into 15-min sub-slots
    slots = int(60 / tau)
    tmp2 = tmp.loc[tmp.index.repeat(slots)].copy()
    tmp2['q'] = np.tile(np.arange(slots), len(tmp))
    tmp2['t'] = tmp2['hour'] * slots + tmp2['q']
    tmp2 = tmp2[tmp2['t'] <= T]
    tmp2['omega_share'] = tmp2['omega'] / slots

    g = (tmp2.groupby(['is_weekend', 'start_station', 'end_station', 't'], as_index=False)
         ['omega_share'].sum())

    full_idx = pd.MultiIndex.from_product(
        [[0, 1], N, N, range(T + 1)],
        names=['is_weekend', 'start_station', 'end_station', 't'])
    s_full = (g.set_index(['is_weekend', 'start_station', 'end_station', 't'])
              ['omega_share'].reindex(full_idx, fill_value=0.0))

    omega_h = {
        w: {(i, j, t): max(0, int(np.ceil(v)))
            for (i, j, t), v in s_full.loc[w].items() if v > 0.0}
        for w in [0, 1]
    }
    return omega_h


def _process_pickup_rates(df_pickup_rates):
    """
    Structure pickup rates into: {weekend_flag: {station: {hour: rate}}}.
    """
    tmp = df_pickup_rates.reset_index().copy()
    hour_cols = [c for c in tmp.columns if str(c).isdigit() and 0 <= int(c) <= 23]
    tmp = tmp[['is_weekend', 'start_station'] + hour_cols]
    tmp['is_weekend'] = tmp['is_weekend'].astype(int)
    tmp['start_station'] = tmp['start_station'].astype(int)
    tmp.columns = ['is_weekend', 'start_station'] + [int(c) for c in hour_cols]

    tmp_full = tmp.set_index(['is_weekend', 'start_station']).sort_index()
    for h in range(24):
        if h not in tmp_full.columns:
            tmp_full[h] = 0.0
    tmp_full = tmp_full.reindex(columns=range(24)).fillna(0.0)

    return {
        int(w): g.droplevel('is_weekend').astype(float).to_dict(orient='index')
        for w, g in tmp_full.groupby(level='is_weekend')
    }


def _process_dropoff_rates(df_dropoff_rates):
    """
    Structure drop-off rates into:
        {weekend_flag: {station: {hour: np.array([inactive, low, high])}}}.
    """
    N = NETWORK.N
    class_order = ['inactive', 'low', 'high']

    tmp = df_dropoff_rates.reset_index().copy()
    tmp.columns = [int(c) if str(c).isdigit() else c for c in tmp.columns]

    full_idx = pd.MultiIndex.from_product(
        [[0, 1], N, class_order],
        names=['is_weekend', 'end_station', 'end_power_class'])
    tmp_full = tmp.set_index(['is_weekend', 'end_station', 'end_power_class']).reindex(full_idx)
    for h in range(24):
        if h not in tmp_full.columns:
            tmp_full[h] = 0.0
    tmp_full = tmp_full.reindex(columns=range(24)).fillna(0.0)

    return {
        w: {i: {h: tmp_full.loc[(w, i)].reindex(class_order)[h].to_numpy(dtype=float)
                for h in range(24)}
            for i in N}
        for w in [0, 1]
    }


# ═══════════════════════════════════════════════════════════════════════
# PUBLIC API
# ═══════════════════════════════════════════════════════════════════════

def load_all_data(data_dir="", demand_scale=1.0, capacity_override=None):
    """
    Load and preprocess all input data files.

    Parameters
    ----------
    data_dir : str
        Path prefix for data files (e.g., "data/" or "").
    demand_scale : float, default 1.0
        Multiplicative scale factor applied uniformly to pickup rates,
        dropoff rates, and omega trip counts. Used for the demand
        sensitivity analysis (§6.X.2a) — values <1 model lower-demand
        scenarios, >1 model higher-demand peaks. Conservation of demand
        balance is preserved (pickup, dropoff, and omega all scale by
        the same factor).
    capacity_override : int or None, default None
        If set, overrides the default station capacity (CAPACITY.C_i_default)
        uniformly across all stations. Used for the capacity sensitivity
        analysis (§6.X.2c).

    Returns
    -------
    dict with keys:
        station_map, d_km, tt_ij, walk_ij, minute_RT_ij, RT_ij,
        omega_h, pickup_rates_by_hour, dropoff_rates_by_hour,
        phi1, phi2, power_transition, accept_parameter, C_i
    """
    p = lambda fname: f"{data_dir}{fname}"

    # ── Spatial data ──
    station_map = pd.read_csv(p('station_map.csv'))

    # Align NETWORK with the actual station count read from station_map.
    # This lets the same code base run on different zone aggregations
    # without editing ProjectConfig.
    NETWORK.num_hexagons = len(station_map)
    NETWORK.N = list(range(1, NETWORK.num_hexagons + 1))
    NETWORK.N0 = [0] + NETWORK.N

    d_km = _build_distance_matrix(station_map)
    tt_ij = _build_travel_times(d_km)
    walk_ij, minute_RT_ij, RT_ij = _build_walk_and_ride_times(d_km)

    # ── Trip flow rates ──
    df_omega_h = pd.read_csv(p('od_flow_high_power.csv'), index_col=0)
    if demand_scale != 1.0:
        df_omega_h = df_omega_h.copy()
        df_omega_h['omega'] = df_omega_h['omega'] * demand_scale
    omega_h = _process_omega_h(df_omega_h, TIME.T)

    # ── Battery discharge probabilities ──
    power_transition = pd.read_csv(p('battery_decline_probs.csv'))
    phi1 = pd.read_csv(p('battery_decline_high_to_low.csv'), index_col=0)         # high → low
    phi2 = pd.read_csv(p('battery_decline_low_to_inactive.csv'), index_col=0).to_dict()  # low → no-power

    # ── Pickup / drop-off demand rates ──
    df_pickup  = pd.read_csv(p('pickup_rates.csv'),         index_col=[0, 1])
    df_dropoff = pd.read_csv(p('dropoff_rates.csv'),  index_col=0)
    if demand_scale != 1.0:
        df_pickup  = df_pickup * demand_scale
        # df_dropoff has a non-numeric column (end_power_class) so we only scale hour columns
        hour_cols = [c for c in df_dropoff.columns if str(c).isdigit() and 0 <= int(c) <= 23]
        df_dropoff = df_dropoff.copy()
        df_dropoff[hour_cols] = df_dropoff[hour_cols] * demand_scale
    pickup_rates_by_hour  = _process_pickup_rates(df_pickup)
    dropoff_rates_by_hour = _process_dropoff_rates(df_dropoff)

    # ── Incentive acceptance lookup ──
    with open(p("incentive_accept.pkl"), "rb") as f:
        accept_parameter = pickle.load(f)

    # ── Station capacities ──
    cap_val = capacity_override if capacity_override is not None else CAPACITY.C_i_default
    C_i = {i: cap_val for i in NETWORK.N}

    return {
        "station_map": station_map,
        "d_km": d_km,
        "tt_ij": tt_ij,
        "walk_ij": walk_ij,
        "minute_RT_ij": minute_RT_ij,
        "RT_ij": RT_ij,
        "omega_h": omega_h,
        "pickup_rates_by_hour": pickup_rates_by_hour,
        "dropoff_rates_by_hour": dropoff_rates_by_hour,
        "phi1": phi1,
        "phi2": phi2,
        "power_transition": power_transition,
        "accept_parameter": accept_parameter,
        "C_i": C_i,
    }