# MSDS 684 — Week 6: Policy Gradient Methods

**Student:** Saivignesh Gangadhar  
**Course:** MSDS 684 – Reinforcement Learning, Regis University

## Overview

This lab implements REINFORCE (Monte Carlo policy gradient) on CartPole-v1 and online TD(0) Actor-Critic on LunarLanderContinuous-v3, both from scratch in PyTorch. It empirically verifies that subtracting a learned state-value baseline reduces gradient variance without changing the expected gradient (S&B §13.4), and documents why vanilla TD(0) Actor-Critic fails on Pendulum-v1 while succeeding on LunarLander.

## Quickstart

```bash
# 1. Clone the repo
git clone https://github.com/gangadharsaivignesh/MSDS684-RL-Labs.git
cd MSDS684-RL-Labs/Week6

# 2. Install dependencies
pip install -r requirements.txt

# 3. Generate the notebook
python build_notebook.py

# 4. Run the notebook
jupyter notebook Lab_PolicyGradient.ipynb
```

## File layout

```
Week6/
├── build_notebook.py        ← generates the .ipynb
├── Lab_PolicyGradient.ipynb ← the notebook
├── requirements.txt
├── README.md
├── src/
│   ├── models.py            ← CategoricalPolicy, GaussianPolicy, ValueNet
│   ├── utils.py             ← helpers: seeding, smoothing, CI
│   ├── train_reinforce.py   ← standalone REINFORCE sweep
│   └── train_actor_critic.py← standalone AC sweep
├── experiments/
│   └── extra_experiments.py ← bonus: LR sweep, entropy ablation, grad-clip
├── results/                 ← cached .npz sweep data
└── plots/                   ← generated figures
```

## Caching

Results are cached in `results/` by hyperparameter hash. If you change any hyperparameter the hash changes and the sweep re-runs automatically.

## Known limitations

- Pendulum (§5) is a 5-seed diagnostic run only — intentional failure demonstration
- LunarLander (§6) is the primary Part 2 deliverable (30 seeds × 500 episodes)
- LunarLander does not reach the +200 solved threshold — that requires PPO/SAC
