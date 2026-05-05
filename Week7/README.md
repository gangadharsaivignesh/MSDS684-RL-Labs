# Lab 7 — Dyna-Q on Taxi-v4: Planning, Adaptation, and Prioritized Sweeping

**MSDS 684 — Reinforcement Learning · Regis University**  
**Author:** Saivignesh Gangadhar  
**Textbook:** Sutton & Barto (2018), *Reinforcement Learning: An Introduction*, 2nd ed., Chapter 8

---

## Repository Structure

```
lab7/
├── src/
│   ├── agents.py          # Dyna-Q, Dyna-Q+, Prioritized Sweeping
│   ├── envs.py            # TaxiDynamicWrapper
│   ├── neural.py          # WorldModel, Policy, REINFORCE for CartPole
│   └── utils.py           # Plotting helpers, ci95, welch_t, smooth
├── experiments/
│   ├── 01_planning_multiplier.py    # Core: n∈{0,5,10,50} comparison
│   ├── 02_dyna_q_plus.py            # Core: Dyna-Q+ on non-stationary Taxi
│   ├── 03_prioritized_sweeping.py   # Core: Prioritized vs uniform
│   ├── 04_neural_world_model.py     # Core: Neural model + REINFORCE
│   ├── 05_kappa_sensitivity.py      # EXTRA: κ sweep for Dyna-Q+
│   ├── 06_theta_sensitivity.py      # EXTRA: θ sweep for prioritized sweeping
│   └── 07_prioritized_vs_uniform_budgets.py  # EXTRA: priority n=10 vs uniform n=50
├── tests/
│   └── test_agents.py     # pytest smoke tests
├── figures/               # Auto-created; all PNGs saved here
├── run_all.py             # Run every experiment in sequence
├── requirements.txt
└── environment.yml
```

---

## Setup

### Option A — pip (recommended)

```bash
git clone <your-repo-url>
cd lab7
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Option B — conda

```bash
conda env create -f environment.yml
conda activate lab7-dyna
```

### Python version

Python **3.9–3.11** recommended. Python 3.12 works but requires `gymnasium[classic-control]` for CartPole.

---

## Running Experiments

### Run all experiments (produces all figures)

```bash
python run_all.py
```

Approximate wall time on a laptop CPU (20 seeds each):

| Experiment | Description | ~Time |
|---|---|---|
| 01 | Planning multiplier | 2–4 min |
| 02 | Dyna-Q+ | 2–3 min |
| 03 | Prioritized sweeping | 2–3 min |
| 04 | Neural world model | 5–10 min |
| 05 | κ sensitivity (extra) | 8–12 min |
| 06 | θ sensitivity (extra) | 10–14 min |
| 07 | Budget comparison (extra) | 8–12 min |

### Run a specific experiment

```bash
python run_all.py 1 3          # only experiments 1 and 3
python experiments/03_prioritized_sweeping.py   # direct
```

### Run tests

```bash
python -m pytest tests/ -v
```

---

## Figures Produced

All figures are saved to `figures/` as PNGs.

| File | Content |
|---|---|
| `fig1a_episode_returns.png` | Episode return vs episode for n∈{0,5,10,50} |
| `fig1b_cumreward.png` | Cumulative reward vs real env steps (6k-step grid) |
| `fig1c_efficiency.png` | Bar charts: episodes-to-competence and total steps |
| `fig2_dyna_q_plus.png` | Dyna-Q vs Dyna-Q+, phase decomposition |
| `fig3_prioritized.png` | Prioritized vs uniform, return curve + bar |
| `fig4_cartpole.png` | CartPole: REINFORCE vs + world model |
| `fig5_kappa_sweep.png` | Phase-2 adaptation vs κ ∈ {0…0.5} |
| `fig6_theta_sweep.png` | Episodes-to-competence vs θ ∈ {0.001…5.0} |
| `fig7_budget_comparison.png` | Prioritized n=10 vs uniform n=50 head-to-head |

---

## Key Implementation Notes (v2 patches)

Four correctness bugs from v1 are fixed in this codebase:

1. **`model_keys_set` (Dyna-Q+, `agents.py`)** — A Python `set` replaces the old `list` for tracking model keys. Previously, a freshly-observed `(s,a)` was appended twice when `plus=True` — once in the seeding loop and once in the guard — giving it 2× sampling weight. The set is idempotent.

2. **Stale-pop budget accounting (prioritized sweeping, `agents.py`)** — Stale heap pops (actual |TD error| < θ) now increment `n_done` instead of skipping with a bare `continue`. A hard ceiling of `3 × n_planning` total pops per real step prevents unbounded work on saturated heaps.

3. **Welch's t-test via scipy (`utils.py`)** — Replaced `erfc(|t|/√2)` (normal approximation / z-test) with `scipy.stats.ttest_ind(equal_var=False)` for the exact t-distribution CDF.

4. **Competence threshold = 5.0** — Raised from 0.0. On Taxi-v4 (optimal ≈ +8), a rolling-mean ≥ 0 threshold is too weak; ≥ 5 requires a genuinely useful policy.

---

## Additional Experiments (beyond the PDF)

Three experiments added in this repo that were absent from the original report:

**Experiment 5 — κ sensitivity (`05_kappa_sensitivity.py`)**  
Sweeps κ ∈ {0, 0.005, 0.01, 0.05, 0.1, 0.5}. Shows the trade-off: too small and the bonus has no effect; too large and Phase-1 efficiency degrades from over-exploration. Justifies κ=0.05 empirically.

**Experiment 6 — θ sensitivity (`06_theta_sensitivity.py`)**  
Sweeps θ ∈ {0.001, 0.05, 0.1, 0.5, 1.0, 5.0}. The original PDF noted θ=0.001 performed worse than uniform without showing why. This sweep makes the full picture visible and confirms θ=0.5 is near-optimal across seeds.

**Experiment 7 — Prioritized n=10 vs uniform n=50 (`07_prioritized_vs_uniform_budgets.py`)**  
This is the key missing comparison from the PDF. If the bottleneck is sampling strategy (not budget), prioritized sweeping at n=10 should match or beat uniform at n=50 — using 5× fewer planning updates per real step. The experiment verifies this claim with Welch's t-tests.

---

## References

- Sutton, R.S. (1990). *Integrated Architectures for Learning, Planning, and Reacting.* ICML.
- Sutton, R.S. & Barto, A.G. (2018). *Reinforcement Learning: An Introduction*, 2nd ed. MIT Press.
- Moore, A.W. & Atkeson, C.G. (1993). Prioritized sweeping. *Machine Learning* 13(1), 103–130.
- Janner, M. et al. (2019). When to Trust Your Model (MBPO). *NeurIPS 2019.*
- Hafner, D. et al. (2023). Mastering Diverse Domains through World Models (DreamerV3). *arXiv:2301.04104.*
