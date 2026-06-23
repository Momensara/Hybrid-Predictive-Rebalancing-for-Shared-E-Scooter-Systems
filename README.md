# Hybrid Predictive Rebalancing for Shared E-Scooter Systems

Reference implementation of the **Direct Look-ahead Hybrid Rebalancing
(DL-HR)** framework introduced in:

> Momen, S., Maknoon, Y., van Arem, B., Sharif Azadeh, S. (2026).
> *Hybrid Predictive Rebalancing Control of Perceived Usability in Shared
> E-Scooter Systems.*

The framework jointly coordinates **truck-based relocation**, **on-street
battery swapping**, and **user-mediated drop-off incentives** to control the
*perceived usability* of a shared e-scooter fleet — that is, the number of
vehicles that users will actually choose to ride, given their location and
battery state — rather than physical inventory alone. Decisions are made in
a rolling-horizon scheme driven by Markovian state forecasts of expected
demand loss (EDL) over a forward-looking window.

This release considers 50 geo-fenced zones to showcase the case study used in the paper (the data are synthetic). The
framework runs three core strategies:

| Strategy | Truck relocation | Battery swap | User incentives |
|----------|:---:|:---:|:---:|
| **DL-HR** | ✓ | ✓ | ✓ |
| **REL+SW** | ✓ | ✓ | — |
| **REL** | ✓ | — | — |

A `NR` (no-rebalancing) baseline is also included.

---

## Requirements

- Python ≥ 3.9
- A working [Gurobi](https://www.gurobi.com/) installation with a license
  (academic licenses are free for non-commercial use)
- Other dependencies: `numpy`, `pandas`, `scipy`, `h3`, `openpyxl`

Install Python dependencies:

```bash
pip install -r requirements.txt
```

---

## Quick start

This repository ships two ways to run the framework:

1. **Demo mode** (`--quick-demo`) — a shrunk synthetic instance (30-second MIP time cap, and a 5% gap tolerance) for showing the *solution process* quickly. Total wall-clock under one minute on a typical Gurobi license.
2. **Synthetic Full case study** — A case similar to the 50-zone shared e-scooter instance reported in the paper.

> ⚠️ **Disclaimer on the demo.**
> `--quick-demo` is synthetic. It shrinks every axis that drives MIP solve time and caps the per-MIP Gurobi time and optimality gap. The resulting service levels, costs, and incentive plans show **only that the pipeline runs end-to-end**; they are *not* the numbers reported in the paper, are not directly comparable across strategies, and should not be cited as case-study results.

### Demo mode (fast, synthetic, illustrative)

Run DL-HR on the small demo instance:

```bash
python main.py --config DL-HR --quick-demo --run-tag demo
```

Or sweep all four strategies for comparison:

```bash
for cfg in NR REL REL+SW DL-HR; do
  python main.py --config $cfg --quick-demo --run-tag demo
done
```

To dial individual knobs (each overrides the demo default), pass them explicitly:

```bash
# Tighter gap, longer per-MIP cap, larger budget
python main.py --config DL-HR --quick-demo \
  --mip-time-limit 60 --mip-gap-tol 0.02 --budget 500 --run-tag demo_tighter
```

### Full case study (reproduces paper)

Run DL-HR on the shipped 50-zone case study (weekday, 6:00–19:00, 10 e-scooters per zone, 15-minute time step). The plan optimizes over the next two hours and is revised hourly under the rolling horizon as new state information arrives:

```bash
python main.py --config DL-HR --run-tag dlhr_baseline
```

Compare against `REL+SW`:

```bash
python main.py --config REL+SW --run-tag rel_sw_baseline
```

Compare against `REL`:

```bash
python main.py --config REL --run-tag rel_baseline
```

Results are written under `results/<config>/`:
- `window_results_<t_begin>_<t_end>_<run-tag>.csv` — per-window summary (objective, service level, cost components)
- `detailed_results_<t_begin>_<t_end>_<run-tag>.xlsx` — per-window decisions (truck routes, on-station operations, battery swaps, incentive plan)

---

## Command-line options

```
--config CONFIG          Strategy: DL-HR | REL+SW | REL | NR  (default: DL-HR)
--dow {0,1}              Day type: 0=weekday, 1=weekend  (default: 1)
--data-dir DIR           Path to input data folder  (default: data/)

--fleet INT              Total fleet size at the start of the operating day
--shares S_n,S_l,S_h     Initial battery composition (e.g. 0.2,0.5,0.3)
--capacity INT           Uniform per-station parking capacity
--trucks INT             Number of rebalancing trucks
--num-hexagons INT       Number of geo-fenced zones (must match data files)

--tau INT                Slot length in minutes (5, 15, or 30; default: 15)
--plan-horizon-hours FLT Planning horizon in wall-clock hours (default: 2)
--forecast-horizon-hours FLT
                         EDL look-ahead horizon in wall-clock hours
                         (default: 2 x plan-horizon-hours)
--budget FLOAT           Daily incentive budget in € (default: 100)
--budget-mode MODE       Incentive budget allocation: adaptive | cumulative
                         (default: adaptive)

--mip-time-limit FLOAT   Per-MIP Gurobi time limit in seconds (default: 600).
                         Lower this for faster solves at the cost of optimality.
--mip-gap-tol FLOAT      Per-MIP target optimality gap (e.g. 0.05 = 5%).
                         Default: tight (Gurobi default ≈ 1e-4).

--quick-demo             Run a small synthetic instance fast: 30 s per-MIP cap, 5% gap tolerance.
                         Illustrative only; does not reproduce paper results.

--run-tag STR            Free-form tag attached to every output row
```

For sensitivity / behavioral analysis, two additional flags are available:

```
--user-class CLASS       User behavioral profile used by the MIP forecast:
                         mean | tough | range_anxious_only | easy
                         (mean = baseline; others alter pickup probabilities)
```

A full reproduction of the paper's headline result (Section 6.3):

```bash
python main.py --config DL-HR --dow 0 \
  --fleet 300 --shares 0.2,0.5,0.3 \
  --capacity 10 --trucks 1 --tau 15 \
  --budget 100 --budget-mode adaptive \
  --run-tag dlhr_headline
```

---

## Repository layout

```
.
├── main.py                  # Entry point and rolling-horizon driver
├── ProjectConfig.py         # Time, network, cost, capacity, choice-model constants
├── data_loader.py           # CSV/PKL reading and preprocessing
├── Markov_EDL.py            # Three-state Markov chain and EDL forecast
├── User_choice_model.py     # User pickup-and-drop-off behavioural model
├── simulation.py            # Service-level simulator (realised metrics)
├── demand_generator.py      # Stochastic demand realisations for the simulator
├── Utils.py                 # State-space utilities (inventory rollover, etc.)
├── data/                    # Input data files (see data/README.md)
├── requirements.txt
├── LICENSE                  # MIT
└── README.md                # This file
```

---

## How DL-HR works (one-paragraph summary)

At every replanning epoch the framework observes the current state of the
fleet — i.e. the number of e-scooters at each geo-fenced zone in each of three
battery classes: **inactive (n)**, **low-power (l)**, **high-power (h)** —
and projects this state forward over a finite look-ahead horizon using a
three-state Markovian prediction model whose pickup rates are scaled by an empirically
estimated discrete-choice user model and is based on historical data. This forecast yields the expected
demand loss (EDL) at every (station × time-slot) pair. A mixed-integer
optimizsation then selects, jointly: which stations a truck should visit,
which scooters to swap or relocate, and which incoming high-power trips to
incentivize (via per-minute ride-fee discount) toward alternative (neighboring) drop-off
zones. Only decisions inside the **commit window** are executed; the
remainder is discarded and re-optimised when the next information arrives.

For full details — including the Markov state transitions, the supporting-
plane linearisation of EDL, and the rolling-horizon mechanics — please
see the paper (Sections 3–5).

---

## Citation

If you use this code or build on the methodology, please cite the paper:

Hybrid Predictive Rebalancing Control of Perceived Usability in Shared E-Scooter Systems (2026). Momen, Sara and Maknoon, Yousef and van Arem, Bart and Sharif Azadeh, Shadi.
https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5964392


---

## License

MIT (see [LICENSE](LICENSE)).

---

## Contact

For questions about the code or the methodology please contact
**sara.momen@tudelft.nl** or open an issue on this repository.
