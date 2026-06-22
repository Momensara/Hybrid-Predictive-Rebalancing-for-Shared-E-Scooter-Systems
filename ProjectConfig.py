"""
ProjectConfig.py — Configuration, constants, and strategy presets
==================================================================

Central place for every tuneable parameter, cost coefficient, and
strategy toggle used by the rolling-horizon e-scooter rebalancing
framework.

Sections
--------
1. Temporal parameters (operational day, slot length, horizons)
2. Network parameters (stations, depot, capacities)
3. Capacity parameters (station and truck capacity limits)
4. Cost coefficients (truck routing, operations, swap, incentives)
5. User choice-model coefficients (from the estimated mixed-logit)
6. Markov-chain discretisation settings
7. Monte Carlo simulation settings
8. Strategy configuration dataclass (Config)
9. Rolling-horizon data bundle (RHData)

Usage
-----
    from ProjectConfig import Config, RHData, TIME, COST, CAPACITY, CHOICE_MODEL
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List


# ═══════════════════════════════════════════════════════════════════════
# 1. TEMPORAL PARAMETERS
# ═══════════════════════════════════════════════════════════════════════

class TIME:
    """All time-related constants for the operational day."""

    t_begin_day: int = 6            # operational day starts at 6:00 AM
    tau: int = 15                    # length of one time slot (minutes)
    total_hours: int = 14            # operational day spans 14 hours (6 AM – 8 PM)
    T: int = (total_hours * 60) // tau   # total number of slots in the day (= 56)

    dt: float = tau / 60             # slot length expressed in hours (= 0.25 h)
    slots_per_hour: int = 60 // tau  # how many slots fit in one hour (= 4)

    # Rolling-horizon window sizes
    plan_horizon_hours: int = 2      # look-ahead horizon (hours)
    rev_interval_hours: int = 1      # commit-and-revise interval (hours)
    plan_len: int = (plan_horizon_hours * 60) // tau   # planning horizon in slots (= 8)
    commit_len: int = (rev_interval_hours * 60) // tau  # committed portion in slots (= 4)
    forecast_len: int = 2 * plan_len  # forecast horizon in slots (used for EDL tails)

    time_step_to_min: int = 15       # for converting slot indices back to minutes


# ═══════════════════════════════════════════════════════════════════════
# 2. NETWORK PARAMETERS
# ═══════════════════════════════════════════════════════════════════════

class NETWORK:
    """Station network and depot configuration."""

    num_hexagons: int = 50           # number of geo-fenced stations (50-zone case study)
    depot_lat: float = 51.9243       # Rotterdam Centraal latitude
    depot_lon: float = 4.4699        # Rotterdam Centraal longitude

    N0 = list(range(num_hexagons + 1))  # all nodes: depot (0) + stations (1..num_hexagons)
    N  = list(range(1, num_hexagons + 1))  # station IDs only
    V  = [1]                            # truck fleet (single homogeneous truck)

    P    = ['n', 'l', 'h']   # power states: no-power, low-power, high-power
    P_nl = ['n', 'l']        # subset of power states eligible for battery swap
    M    = ['es', 'bat']     # truck cargo types: e-scooters ('es') and batteries ('bat')


# ═══════════════════════════════════════════════════════════════════════
# 3. CAPACITY PARAMETERS
# ═══════════════════════════════════════════════════════════════════════

class CAPACITY:
    """Station and truck capacity limits."""

    C_i_default: int = 10            # max e-scooters per station
    C_v_escooter: int = 10           # max e-scooters a truck can carry
    C_v_batt: int = 20               # max spare batteries a truck can carry


# ═══════════════════════════════════════════════════════════════════════
# 4. COST COEFFICIENTS
# ═══════════════════════════════════════════════════════════════════════

class COST:
    """Operational cost coefficients (EUR)."""

    UNF: float = 1.0                # unlock fee per trip start (€)
    RF: float = 0.3                 # per-minute ride revenue (€/min)

    c_ij: float = 1.2               # per-km truck travel cost (€/km)
    c_v: float = 100.0              # fixed cost of deploying a truck per day (€)
    c_load_unload: float = 0.5      # cost per load/unload operation at a station (€)
    c_swap: float = 0.1             # cost per battery swap (€)
    incentive_budget: float = 100.0  # total daily budget for user incentives (€)

    avg_trip_dur: float = 10.0       # average trip duration (minutes) for EDL costing

    @classmethod
    def edl_coef(cls) -> float:
        """Revenue lost per unit of expected demand loss (€).

        Each unserved trip costs the operator an unlock fee plus the
        expected per-minute revenue over an average trip.
        """
        return cls.UNF + cls.RF * cls.avg_trip_dur


# ═══════════════════════════════════════════════════════════════════════
# 5. USER CHOICE-MODEL COEFFICIENTS (Mixed Logit)
# ═══════════════════════════════════════════════════════════════════════

class CHOICE_MODEL:
    """
    Coefficients from the mixed-logit discrete-choice model.

    The model predicts the probability that a user arriving at a station
    picks up a HIGH-power, LOW-power, or OPT-OUT (no ride) alternative.

    Fixed coefficients apply to all user classes. Two coefficients are
    random (uniformly distributed across the population):
        - beta_ride  (sensitivity to per-minute ride fee)
        - beta_batt  (sensitivity to remaining battery range)

    User classes are defined by combinations of the upper/lower bounds
    of these two random coefficients to capture heterogeneity.
    """

    # ── Fixed coefficients (same for all classes) ──
    beta_es: float     = 10.514    # alternative-specific constant (e-scooter)
    beta_walk: float   = -0.342    # disutility per minute of walking to station
    beta_unlock: float = -1.419    # disutility of unlock fee (€)
    beta_type: float   = -1.02     # dummy for 25 km/h vehicle type
    beta_prev: float   = -1.79     # dummy for previous e-scooter use
    beta_bike: float   = -2.21     # dummy for bicycle as competing mode
    beta_income: float = -1.18     # dummy for low-income respondent
    beta_alone: float  =  1.21     # dummy for living alone
    beta_shared: float = -0.86     # dummy for shared housing
    eta_att: float     =  0.82     # latent attitude toward e-scooters
    eta_range: float   = -0.58     # latent range anxiety score
    beta_int: float    =  0.033    # interaction: battery × range anxiety (optional)

    # ── Random coefficient bounds (uniform distribution) ──
    BETA_RIDE_MEAN: float  = -25.147
    BETA_RIDE_LOWER: float = -45.89   # most fee-sensitive users
    BETA_RIDE_UPPER: float = -12.30   # least fee-sensitive users

    BETA_BATT_MEAN: float  = 0.27
    BETA_BATT_LOWER: float = 0.146    # least range-concerned users
    BETA_BATT_UPPER: float = 0.459    # most range-concerned users

    # ── Aggregate pickup probabilities (used in Markov model) ──
    # These are the *average* probabilities across the population:
    prob_h: float   = 0.70    # P(pick high-power scooter)
    prob_l: float   = 0.18    # P(pick low-power scooter)
    prob_out: float = 0.12    # P(opt-out / leave without ride)


# Pre-defined user classes for scenario analysis
USER_CLASSES = {
    "mean": {
        "beta_ride": CHOICE_MODEL.BETA_RIDE_MEAN,
        "beta_batt": CHOICE_MODEL.BETA_BATT_MEAN,
        "desc": "Average user — mean of both random coefficients",
    },
    "tough": {
        "beta_ride": CHOICE_MODEL.BETA_RIDE_LOWER,
        "beta_batt": CHOICE_MODEL.BETA_BATT_UPPER,
        "desc": "Range-anxious AND fee-sensitive (worst-case acceptance)",
    },
    "easy": {
        "beta_ride": CHOICE_MODEL.BETA_RIDE_UPPER,
        "beta_batt": CHOICE_MODEL.BETA_BATT_LOWER,
        "desc": "Neither range-anxious nor fee-sensitive (best-case acceptance)",
    },
    "range_anxious_only": {
        "beta_ride": CHOICE_MODEL.BETA_RIDE_MEAN,
        "beta_batt": CHOICE_MODEL.BETA_BATT_UPPER,
        "desc": "Range-anxious but average fee sensitivity",
    },
}


# ═══════════════════════════════════════════════════════════════════════
# 6. MARKOV-CHAIN DISCRETISATION SETTINGS
# ═══════════════════════════════════════════════════════════════════════

class MARKOV:
    """Parameters for the discretised continuous-time Markov chain."""

    n_taylor_steps: int = 50   # Taylor expansion order for matrix exponential
    # The one-step transition matrix is: P ≈ (I + Q·Δt / n)^n


# ═══════════════════════════════════════════════════════════════════════
# 7. MONTE CARLO SIMULATION SETTINGS
# ═══════════════════════════════════════════════════════════════════════

class SIMULATION:
    """Parameters for the post-optimisation Monte Carlo simulation."""

    seed_for_window: int = 4242
    sim_runs: int = 100          # number of MC realisations per window
    max_req_day: int = 1240      # system-wide daily request cap
    sys_hour_cap: int = 60       # max requests per station per hour

    share_low_power: float = 0.4  # initial fraction of fleet in low-power state
    init_inv_toler: float = 0.1   # tolerance (±10 %) when updating inventory from simulation


# ═══════════════════════════════════════════════════════════════════════
# 8. STRATEGY CONFIGURATION (Config dataclass)
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class Config:
    """
    Toggle which operational levers are active in the MIP.

    Pre-defined strategies (see ``STRATEGY_PRESETS`` in ``main.py``)
    ----------------------------------------------------------------
    DL-HR   : use_truck + use_ops + use_swap + use_incentives  (full hybrid)
    REL+SW  : use_truck + use_ops + use_swap                   (relocation + swap)
    REL     : use_truck + use_ops                              (relocation only)
    NR      : all False                                        (no rebalancing)
    """

    use_truck: bool = True               # enable truck routing variables (y, u)
    use_ops: bool = True                 # enable load/unload at stations (n_load, n_un)
    use_swap: bool = True                # enable battery swap at stations/depot (b_swap)
    use_truck_inventory: bool = True     # track e-scooter cargo on truck (M['es'])
    use_battery_inventory: bool = True   # track battery cargo on truck (M['bat'])
    use_supporting_planes: bool = True   # use affine EDL approximation (w_sp)
    use_incentives: bool = True          # enable user-incentive variables (x, z)


# ═══════════════════════════════════════════════════════════════════════
# 9. ROLLING-HORIZON DATA BUNDLE (RHData dataclass)
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class RHData:
    """
    Bundles all per-window input data that the MIP solver needs.

    This struct is rebuilt at each rolling-horizon iteration because the
    active station set, omega set, and inventory forecasts change.
    """

    # Sets
    N_active: Iterable[int]              # stations with non-negligible EDL this window
    N0_active: Iterable[int]             # N_active ∪ {depot} ∪ {truck start nodes}
    V: Iterable[int]                     # truck IDs
    P: Iterable[str]                     # power states ['n','l','h']
    P_nl: Iterable[str]                  # swap-eligible states ['n','l']
    T: range                             # time slots in this window
    depot_index: int                     # node index for depot (= 0)

    # Distance / travel-time matrices
    d_ij: Dict[Any, Dict[Any, float]]    # Euclidean distance (km) between nodes
    tt_ij: Dict[Any, Dict[Any, int]]     # truck travel time (slots) between nodes

    # Incentive trip candidates
    omega_set: List[tuple]               # [(origin, dest, t), …] with positive trip flow
    RT_ij: Dict[Any, Dict[Any, int]]     # e-scooter ride time (slots) between stations
    C_i: Dict[Any, int]                  # station capacities

    # Cost parameters
    c_v: float
    c_ij: float
    c_load_unload: float
    c_swap: float
    RF: float
    time_step_to_min: float

    # State carried over from previous window
    prev_acquired_truck: Dict[Any, bool]

    # Simulation inputs (optional — only needed when simulating)
    slots_per_hour: int = 4
    fn_pickup_rates_by_hour: Any = None
    fn_dropoff_rates_by_hour: Any = None
    Init_inventory: Dict[Any, Dict[str, float]] = None
