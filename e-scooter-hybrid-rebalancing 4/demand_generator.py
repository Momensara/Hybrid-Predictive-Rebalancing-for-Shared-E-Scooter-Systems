"""
demand_generator.py — Stochastic demand realisations for the simulator
=======================================================================

Generates one Monte-Carlo realisation of a full-day demand profile per
call. Pickups and drop-offs at each station are sampled from
time-heterogeneous Poisson processes calibrated to the hourly rate
tables loaded from disk, with optional global caps to keep extreme
tails of the Poisson distribution from skewing the simulator.

The output is the input format expected by ``simulation.py``:

    station_demand[i][h] = {
        'pickup_low'  : list[int]  length = slots_per_hour
        'pickup_high' : list[int]  length = slots_per_hour
        'dropoff'     : {
            'no_power'   : list[int],
            'low_power'  : list[int],
            'high_power' : list[int],
        }
    }
"""

import numpy as np


def _sample_per_slot(hourly_rate, slots_per_hour, rng):
    """
    Split an hourly Poisson rate into per-slot Poisson samples.

    The rate per slot is hourly_rate / slots_per_hour. Returns a length-
    slots_per_hour vector of non-negative integer arrivals.
    """
    if hourly_rate <= 0:
        return np.zeros(slots_per_hour, dtype=int)
    per_slot = float(hourly_rate) / float(slots_per_hour)
    return rng.poisson(per_slot, size=slots_per_hour).astype(int)


def _apply_hour_cap(arr, cap):
    """If the total arrivals in `arr` exceed `cap`, trim from the largest
    bins down. Preserves the distribution shape as much as possible."""
    if cap is None or cap <= 0:
        return arr
    total = int(arr.sum())
    if total <= cap:
        return arr
    # Remove arrivals greedily from the largest bins until total <= cap
    arr = arr.copy()
    overflow = total - cap
    order = np.argsort(arr)[::-1]
    for idx in order:
        take = min(int(arr[idx]), overflow)
        arr[idx] -= take
        overflow -= take
        if overflow <= 0:
            break
    return arr


def wkD_simulated_demand_corrected(
    pk_hourly_rates,
    dr_hourly_rates,
    slots_per_hour,
    hour_cap=60,
    peak_hour_cap=90,
    day_cap=1080,
    prob_l=0.5,
    prob_h=0.5,
    enforce_time_patterns=False,
    seed=None,
):
    """
    Generate one full-day demand realisation.

    Parameters
    ----------
    pk_hourly_rates : dict
        Pickup rates ``{station_id: {hour: rate}}``.
    dr_hourly_rates : dict
        Drop-off rates by class
        ``{station_id: {hour: np.array([n_rate, l_rate, h_rate])}}``.
        Convention follows ``data_loader._process_dropoff_rates``:
        index 0 = inactive, 1 = low, 2 = high.
    slots_per_hour : int
        Number of slots per clock hour.
    hour_cap : int
        Per-station per-hour cap on total arrivals.
    peak_hour_cap : int
        Per-station per-hour cap during system-wide peak hours.
        (Currently treated the same as ``hour_cap``; kept for API
        compatibility.)
    day_cap : int
        Per-station daily cap on total arrivals.
    prob_l, prob_h : float
        Average pickup choice probabilities, used to split aggregate
        pickup demand into low-power and high-power components.
    enforce_time_patterns : bool
        Unused in this clean release. Kept for API compatibility.
    seed : int or None
        Seed for reproducibility.

    Returns
    -------
    dict with key
        ``'station_demand'`` : nested dict
            ``{station: {hour: {'pickup_low': ..., 'pickup_high': ...,
            'dropoff': {...}}}}``
    """
    rng = np.random.default_rng(seed)
    stations = list(pk_hourly_rates.keys())

    out = {}
    for i in stations:
        out[i] = {}
        day_total_so_far = 0

        # Lookups for this station
        pk_hours = pk_hourly_rates.get(i, {})
        dr_hours = dr_hourly_rates.get(i, {})

        for h in range(24):
            # ── PICKUPS ─────────────────────────────────────────────
            pk_rate = float(pk_hours.get(h, 0.0))
            pk_arrivals = _sample_per_slot(pk_rate, slots_per_hour, rng)
            pk_arrivals = _apply_hour_cap(pk_arrivals, hour_cap)

            # Split aggregate pickup demand into low / high using the
            # average choice probabilities. The remainder (opt-out share)
            # is implicit in the simulator's serving rules.
            p_total = float(prob_l + prob_h)
            if p_total > 0:
                share_l = prob_l / p_total
            else:
                share_l = 0.5
            pickup_low  = rng.binomial(pk_arrivals, share_l)
            pickup_high = pk_arrivals - pickup_low

            # ── DROP-OFFS ───────────────────────────────────────────
            dr_vec = dr_hours.get(h)
            if dr_vec is None:
                dr_vec = np.zeros(3, dtype=float)
            else:
                dr_vec = np.asarray(dr_vec, dtype=float)
            dr_no  = _sample_per_slot(dr_vec[0], slots_per_hour, rng)
            dr_low = _sample_per_slot(dr_vec[1], slots_per_hour, rng)
            dr_hi  = _sample_per_slot(dr_vec[2], slots_per_hour, rng)

            # ── DAILY CAP ───────────────────────────────────────────
            hour_total = int(
                pickup_low.sum() + pickup_high.sum()
                + dr_no.sum() + dr_low.sum() + dr_hi.sum()
            )
            if day_total_so_far + hour_total > day_cap:
                allowed = max(0, day_cap - day_total_so_far)
                if allowed == 0:
                    pickup_low  = np.zeros(slots_per_hour, dtype=int)
                    pickup_high = np.zeros(slots_per_hour, dtype=int)
                    dr_no  = np.zeros(slots_per_hour, dtype=int)
                    dr_low = np.zeros(slots_per_hour, dtype=int)
                    dr_hi  = np.zeros(slots_per_hour, dtype=int)
                    hour_total = 0
                else:
                    # Proportional scaling
                    scale = allowed / hour_total
                    pickup_low  = (pickup_low  * scale).astype(int)
                    pickup_high = (pickup_high * scale).astype(int)
                    dr_no  = (dr_no  * scale).astype(int)
                    dr_low = (dr_low * scale).astype(int)
                    dr_hi  = (dr_hi  * scale).astype(int)
                    hour_total = int(
                        pickup_low.sum() + pickup_high.sum()
                        + dr_no.sum() + dr_low.sum() + dr_hi.sum()
                    )
            day_total_so_far += hour_total

            out[i][h] = {
                'pickup_low'  : pickup_low.tolist(),
                'pickup_high' : pickup_high.tolist(),
                'dropoff': {
                    'no_power'  : dr_no.tolist(),
                    'low_power' : dr_low.tolist(),
                    'high_power': dr_hi.tolist(),
                },
            }

    return {'station_demand': out}
