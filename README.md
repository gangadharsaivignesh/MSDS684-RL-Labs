# MSDS 684 — Reinforcement Learning Labs

Lab assignments for **MSDS 684: Reinforcement Learning**, organized by week. Each lab implements a core RL algorithm from Sutton & Barto and explores its behavior on a benchmark environment.

## Repository Structure

```
MSDS684-RL-Labs/
├── README.md              ← this file
├── requirements.txt       ← shared Python dependencies
├── Week1/                 ← Lab 1: Multi-Armed Bandits
├── Week2/                 ← Lab 2: MDPs & Dynamic Programming
└── Week3/                 ← Lab 3: Monte Carlo Control on Blackjack
```

## Setup

**Requirements:** Python 3.9+

Install all dependencies:

```bash
pip install -r requirements.txt
```

## Labs

| Week | Topic | S&B Chapter |
|------|-------|-------------|
| **Week 1** | ε-Greedy and UCB on a 10-armed Gaussian bandit; FrozenLake & Taxi exploration | Ch. 2 |
| **Week 2** | Policy iteration & value iteration on FrozenLake | Ch. 3–4 |
| **Week 3** | First-visit Monte Carlo control on Blackjack-v1 (ε-soft policies) | Ch. 5 |

Each `WeekN/` folder contains its own notebook (`LabN.ipynb`) with the full implementation, experiments, and analysis. Open in VS Code or JupyterLab and run all cells in order.

## References

- Sutton, R.S. & Barto, A.G. (2018). *Reinforcement Learning: An Introduction* (2nd ed.). MIT Press.
- Gymnasium documentation: https://gymnasium.farama.org/
