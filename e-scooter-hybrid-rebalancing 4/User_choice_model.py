"""
user_choice_model.py — Mixed-Logit User Choice Model
=====================================================

Implements the discrete-choice model from Burghardt et al. (2025), used to predict user
behaviour when arriving at a shared e-scooter station.

A user who arrives at a station faces three alternatives:
    1. Pick up a HIGH-power e-scooter   (preferred, full range)
    2. Pick up a LOW-power e-scooter    (shorter range, still usable)
    3. Opt out (leave without riding)   (utility normalised to 0)

The model accounts for taste heterogeneity through two random
coefficients (per-minute fee sensitivity and battery range sensitivity),
which are assumed uniformly distributed. Different user "classes" are
represented by specific draws from those distributions.

This module also provides the `build_accept_parameter_for_class`
function, which pre-computes a binary lookup table indicating whether
a user of a given class would accept an incentive to redirect their
trip from destination d to alternative station i.

Usage
-----
    from user_choice_model import compute_probs_for_class, USER_CLASSES

    results = {}
    for cls_name, cls_params in USER_CLASSES.items():
        results[cls_name] = compute_probs_for_class(
            beta_ride=cls_params["beta_ride"],
            beta_batt=cls_params["beta_batt"],
            pct_high=50.0, pct_low=25.0
        )
"""

import numpy as np
from ProjectConfig import CHOICE_MODEL, USER_CLASSES


# ═══════════════════════════════════════════════════════════════════════
# UTILITY & PROBABILITY COMPUTATION
# ═══════════════════════════════════════════════════════════════════════

def compute_probs_for_class(
    beta_ride: float,
    beta_batt: float,
    pct_high: float = 50.0,
    pct_low: float = 25.0,
    walk_min: float = 2.0,
    unlock_fee: float = 1.00,
    ride_fee_term: float = 0.30,
    vehicle_type_25: int = 1,
    previous_use: int = 1,
    bike: int = 0,
    income_low: int = 0,
    living_alone: int = 0,
    living_shared: int = 0,
    attitude: float = 0.0,
    range_anxiety: float = 0.0,
    range_per_pct_km: float = 0.6,
) -> dict:
    """
    Compute choice probabilities for a specific user class.

    The utility of each alternative is composed of:
        V = ASC + β_walk · walk + β_unlock · fee + β_ride · ride_cost
            + β_type · type25 + β_prev · prev_use + β_bike · bike
            + β_income · low_income + β_alone · alone + β_shared · shared
            + η_att · attitude + η_range · range_anxiety
            + β_batt · remaining_range

    The opt-out alternative is normalised to V₀ = 0.

    Parameters
    ----------
    beta_ride : float
        Class-specific per-minute ride fee coefficient (random, negative).
    beta_batt : float
        Class-specific remaining battery range coefficient (random, positive).
    pct_high, pct_low : float
        Battery percentage for the high-power and low-power alternatives.
    walk_min : float
        Walking time to the station (minutes).
    unlock_fee : float
        Unlock fee (€).
    ride_fee_term : float
        Total ride cost term (€ or €/min × minutes, depending on β_ride units).
    vehicle_type_25 : int
        1 if vehicle is 25 km/h type, else 0.
    previous_use : int
        1 if the user has used e-scooters before, else 0.
    bike : int
        1 if the competing mode is bicycle, else 0.
    income_low : int
        1 if respondent is low-income, else 0.
    living_alone, living_shared : int
        Household composition dummies.
    attitude : float
        Latent attitude score toward e-scooters (0 if centred).
    range_anxiety : float
        Latent range anxiety score (0 if centred).
    range_per_pct_km : float
        Conversion factor: remaining range (km) = pct × this factor.

    Returns
    -------
    dict with keys:
        V_H, V_L, V_0    — raw utilities for high, low, and opt-out
        prob_h, prob_l, P_0 — softmax choice probabilities
    """
    cm = CHOICE_MODEL

    # Convert battery percentage to remaining range (km)
    range_H = range_per_pct_km * pct_high
    range_L = range_per_pct_km * pct_low

    # Shared (non-battery) part of utility — split ASC equally across alternatives
    ASC_share = 0.5 * cm.beta_es

    common = (
        ASC_share
        + cm.beta_walk   * walk_min
        + cm.beta_unlock * unlock_fee
        + beta_ride      * ride_fee_term       # ← class-specific
        + cm.beta_type   * vehicle_type_25
        + cm.beta_prev   * previous_use
        + cm.beta_bike   * bike
        + cm.beta_income * income_low
        + cm.beta_alone  * living_alone
        + cm.beta_shared * living_shared
        + cm.eta_att     * attitude
        + cm.eta_range   * range_anxiety
    )

    # Full utilities (battery component differs between H and L)
    V_H = common + beta_batt * range_H       # ← class-specific
    V_L = common + beta_batt * range_L
    V_0 = 0.0                                 # opt-out (normalised)

    # Softmax (multinomial logit probabilities)
    vals = np.array([V_H, V_L, V_0], dtype=float)
    exp_v = np.exp(vals)
    probs = exp_v / exp_v.sum()

    return {
        "V_H": V_H, "V_L": V_L, "V_0": V_0,
        "prob_h": probs[0], "prob_l": probs[1], "P_0": probs[2],
    }


# ═══════════════════════════════════════════════════════════════════════
# INCENTIVE ACCEPTANCE PARAMETER (per user class)
# ═══════════════════════════════════════════════════════════════════════

def build_accept_parameter_for_class(
    beta_ride: float,
    beta_walk_coef: float,
    RT_ij,
    walk_ij,
    price_per_min: float,
    walk_max: float,
):
    """
    Build a binary lookup: would a user accept a redirect from d → i?

    For each possible trip (origin o → destination d), consider redirecting
    the user to alternative station i. The user compares:

        U_incentive = β_ride · price_per_min · (−ride_time_o→i) + β_walk · walk_d→i
        U_initial   = β_ride · price_per_min · ride_time_o→d

    The user accepts the incentive iff U_incentive ≥ U_initial. This is
    a simplified comparison: the incentive makes the redirected ride free,
    but the user must walk from d to i.

    Parameters
    ----------
    beta_ride : float
        Class-specific ride fee coefficient.
    beta_walk_coef : float
        Walk disutility coefficient (from the choice model, e.g. −0.342).
    RT_ij : DataFrame or nested dict
        Ride time between stations (minutes or slots).
    walk_ij : DataFrame or nested dict
        Walk time from destination d to alternative station i.
    price_per_min : float
        Per-minute ride fee (€/min).
    walk_max : float
        Maximum acceptable walking distance/time (same units as walk_ij).

    Returns
    -------
    dict {(o, d, i): 0 or 1}
        1 if the user accepts the redirect from d to i; 0 otherwise.
    """
    accept = {}

    for o in RT_ij.index:
        for d in RT_ij.index:
            rt_od = RT_ij.loc[o, d]
            if not np.isfinite(rt_od):
                continue

            for i in RT_ij.index:
                if i == d:
                    continue

                w_id = walk_ij.loc[d][i]
                rt_oi = RT_ij.loc[o, i] if i in RT_ij.index else np.nan
                if not (np.isfinite(w_id) and np.isfinite(rt_oi)):
                    continue
                if w_id > walk_max:
                    continue

                # Utility comparison: redirected trip vs. original trip
                U_incentive = beta_ride * price_per_min * (-rt_oi) + beta_walk_coef * w_id
                U_initial   = beta_ride * price_per_min * rt_od

                accept[(o, d, i)] = 1 if U_incentive >= U_initial else 0

    return accept


# ═══════════════════════════════════════════════════════════════════════
# CONVENIENCE: Run for all pre-defined user classes
# ═══════════════════════════════════════════════════════════════════════

def compute_all_class_probs(**kwargs) -> dict:
    """
    Compute choice probabilities for every pre-defined user class.

    Returns
    -------
    dict {class_name: {V_H, V_L, V_0, prob_h, prob_l, P_0}}
    """
    results = {}
    for cls_name, cls_params in USER_CLASSES.items():
        results[cls_name] = compute_probs_for_class(
            beta_ride=cls_params["beta_ride"],
            beta_batt=cls_params["beta_batt"],
            **kwargs,
        )
    return results


if __name__ == "__main__":
    # Quick demo: compute and print probabilities for all user classes
    results = compute_all_class_probs(pct_high=50.0, pct_low=25.0)
    for cls, out in results.items():
        print(f"{cls:>20}: P_high={out['prob_h']:.3f}  "
              f"P_low={out['prob_l']:.3f}  P_optout={out['P_0']:.3f}")