# Lab 1 — Multi-Armed Bandits & Gymnasium Environments

**MSDS 684 — Reinforcement Learning | Week 1**  
**Author:** Gangadhar Saivignesh  
**Date:** May 2026

---

## Overview

This lab implements and compares two classic bandit exploration strategies on a
10-armed Gaussian bandit environment, then explores two standard Gymnasium
environments using a random-agent baseline.

**Part 1 — Multi-Armed Bandit**
- Custom `GaussianBanditEnv` following the Gymnasium API (subclasses `gym.Env`)
- `EpsilonGreedyAgent` with ε ∈ {0.01, 0.1, 0.2}
- `UCBAgent` with c ∈ {0.5, 1.0, 2.0}
- 1,000 independent runs × 2,000 steps each
- Results averaged with ±1 SEM error bands

**Part 2 — Gymnasium Environments**
- FrozenLake-v1 and Taxi-v4 space inspection and MDP tuple mapping
- `RandomAgent` baseline evaluated over 1,000 episodes per environment

**Additional Experiments** (see `Lab1_additional_experiments.ipynb`)
- Non-stationary bandit (drifting arm means)
- Greedy baseline (ε = 0) comparison
- UCB vs ε-greedy regret curves
- Sensitivity analysis: varying k (number of arms) from 5 to 20

---

## Results Summary

| Algorithm | Parameter | Final Avg Reward | Final % Optimal |
|-----------|-----------|-----------------|-----------------|
| ε-Greedy  | ε = 0.01  | 1.3887          | 72.4%           |
| ε-Greedy  | ε = 0.10  | 1.3571          | 82.8%           |
| ε-Greedy  | ε = 0.20  | 1.1993          | 75.0%           |
| UCB       | c = 0.5   | 1.4850          | 88.2%           |
| UCB       | c = 1.0   | 1.5058          | 94.3%           |
| UCB       | c = 2.0   | 1.4827          | 90.3%           |

*All values at step 2,000, averaged over 1,000 independent runs.*

---

## Repository Structure

```
Week1/
├── Lab1_updated.ipynb              # Main lab notebook (primary submission)
├── Lab1_additional_experiments.ipynb  # Additional experiments beyond PDF
├── requirements.txt                # pip dependencies
├── environment.yml                 # conda environment
├── README.md                       # This file
└── visualizations/
    ├── epsilon_greedy_results.png  # Figure 1: ε-Greedy learning curves
    ├── ucb_results.png             # Figure 2: UCB learning curves
    ├── combined_comparison.png     # Figure 3: All configurations comparison
    ├── nonstationary_bandit.png    # Additional: non-stationary results
    ├── greedy_baseline.png         # Additional: ε=0 greedy baseline
    └── regret_curves.png           # Additional: cumulative regret comparison
```

---

## Setup Instructions

### Option A — pip (recommended for VS Code)

```bash
# 1. Clone the repository
git clone https://github.com/gangadharsaivignesh/MSDS684-RL-Labs.git
cd MSDS684-RL-Labs/Week1

# 2. Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate        # macOS/Linux
venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch Jupyter
jupyter notebook Lab1_updated.ipynb
```

### Option B — conda

```bash
conda env create -f environment.yml
conda activate msds684-rl
jupyter notebook Lab1_updated.ipynb
```

### Option C — VS Code (no terminal)

1. Open `Lab1_updated.ipynb` in VS Code
2. Select **Python 3.10+** kernel (or your installed version)
3. Run **Kernel → Restart Kernel and Run All Cells**

---

## Reproducibility

All stochastic operations are seeded:
- Environment noise: `seed = run_index` (0 to 999)
- Agent exploration: `seed = run_index + 50_000`

Re-running the notebook top-to-bottom from a fresh kernel always produces
identical results and visualizations.

---

## Gymnasium Version Compatibility

The notebook auto-detects your installed Gymnasium version:

```python
TAXI_ENV = 'Taxi-v4' if 'Taxi-v4' in gym.envs.registry else 'Taxi-v3'
```

- **Gymnasium ≥ 1.0**: uses `Taxi-v4`
- **Gymnasium < 1.0**: uses `Taxi-v3`

No manual changes needed.

---

## Key References

- Sutton, R.S. & Barto, A.G. (2018). *Reinforcement Learning: An Introduction* (2nd ed.).
  MIT Press. Chapters 1–2.
- Auer, P., Cesa-Bianchi, N., & Fischer, P. (2002). Finite-time analysis of the
  multiarmed bandit problem. *Machine Learning*, 47(2–3), 235–256.
- Gymnasium documentation: https://gymnasium.farama.org/
