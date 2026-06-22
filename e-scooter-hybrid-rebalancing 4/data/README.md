# Data files

Input data for the 50-zone Rotterdam case study used in the paper.

## Provenance

These files derive from operational data of a European e-scooter operator
in Rotterdam. The original 110-zone H3-resolution-10 grid was aggregated
to 50 zones using adjacency-constrained hierarchical clustering, and
pickup, drop-off, and OD-flow rates have been uniformly scaled by 1.4
to match the calibration of the case study reported in the paper. No
further runtime scaling is needed.

## Files

| File | Description |
|------|-------------|
| `station_map.csv` | Station ID → (latitude, longitude) mapping. The shipped file also contains the H3 hex index, source-station count, and merge count from the 110→50 aggregation. |
| `pickup_rates.csv` | Hourly pickup demand per station, indexed by `(start_station, is_weekend)` with one column per hour 0–23. |
| `dropoff_rates.csv` | Hourly drop-off rates per station, broken down by end-of-trip battery class (high / low / inactive). |
| `od_flow_high_power.csv` | Hourly origin–destination flows for high-power trips, used to build the `omega_h` lookup. |
| `battery_decline_high_to_low.csv` | Per-hour probability that a high-power scooter becomes low-power after a ride (φ₁). |
| `battery_decline_low_to_inactive.csv` | Per-hour probability that a low-power scooter becomes inactive after a ride (φ₂). |
| `battery_decline_probs.csv` | Combined battery-state transition probabilities. |
| `incentive_accept.pkl` | Binary lookup `(origin, original_destination, alternative_destination) → {0,1}` indicating whether the alternative destination is within 500 m walking distance of the original (per Hoobroeckx et al., 2023). |

## Day-of-week convention

`dow = 0` corresponds to weekdays; `dow = 1` corresponds to weekends.
Pass with the `--dow` flag.

## Note for reuse

The depot coordinates (`NETWORK.depot_lat`, `NETWORK.depot_lon` in
`ProjectConfig.py`) point to Rotterdam Centraal. If you re-run the
framework on a different city, regenerate `station_map.csv` and the
companion files for that geography.
