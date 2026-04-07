# MSDS 684 — Reinforcement Learning
## Lab 1: Multi-Armed Bandits & Gymnasium Environments

### Overview
This lab implements and compares exploration strategies for the multi-armed bandit problem, and explores standard Gymnasium environments using a random policy baseline.

**Topics covered:**
- Custom 10-armed Gaussian bandit environment (Gymnasium API)
- ε-Greedy agent (ε ∈ {0.01, 0.1, 0.2})
- UCB agent (c ∈ {0.5, 1.0, 2.0})
- 2000 time steps × 1000 independent runs
- FrozenLake-v1 and Taxi-v3 environment exploration
- Random policy baseline and MDP tuple mapping

### Setup

**Requirements:** Python 3.9+

Install dependencies:
```bash
pip install -r requirements.txt
```

### Running the Notebook

Open `Lab1.ipynb` in VS Code or JupyterLab and run all cells in order.

Expected runtime: ~60 seconds for the bandit experiments (1000 runs × 2000 steps).

### Files

| File | Description |
|------|-------------|
| `Lab1.ipynb` | Main notebook — all code, results, and analysis |
| `requirements.txt` | Python dependencies |

### Results Summary

| Algorithm | Final Avg Reward | % Optimal Action |
|-----------|-----------------|-----------------|
| ε-Greedy (ε=0.01) | 1.389 | 72.4% |
| ε-Greedy (ε=0.10) | 1.357 | 82.8% |
| ε-Greedy (ε=0.20) | 1.199 | 75.0% |
| UCB (c=0.5) | 1.483 | 87.8% |
| UCB (c=1.0) | 1.507 | 95.7% |
| UCB (c=2.0) | 1.484 | 90.0% |

| Environment | Random Success Rate | Avg Reward |
|-------------|-------------------|------------|
| FrozenLake-v1 | 0.8% | 0.008 |
| Taxi-v3 | 0.0% | -767.01 |

### References
- Sutton, R.S. & Barto, A.G. (2018). *Reinforcement Learning: An Introduction* (2nd ed.). Chapters 1–2.
- Gymnasium documentation: https://gymnasium.farama.org/
