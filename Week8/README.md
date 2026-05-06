# Lab 8 — Comprehensive Algorithm Comparison
**MSDS 684 · Reinforcement Learning**  
Gangadhar, Sai Vignesh

---

## Overview

Three RL algorithm families compared on **LunarLander-v2** (hard-pinned):

| Algorithm | Family | Key Hyperparameter |
|---|---|---|
| Tabular Q-Learning | Value-based (tabular) | `n_bins` — bins per state dimension |
| REINFORCE with Baseline | Policy gradient (neural) | `lr_policy` — policy network learning rate |
| Dyna-Q | Model-based (tabular) | `n_planning` — simulated updates per real step |

**Experimental setup:** 3 random seeds `{42, 123, 777}` · 500 episodes per run · γ = 0.99

---

## Repository Structure

```
Week8/
├── Lab8_Algorithm_Comparison_final.ipynb   # Main experiment notebook (run this)
├── requirements.txt                         # All dependencies
└── README.md                                # This file
```

Running the notebook generates four figures in the same directory:
- `learning_curves.png` — smoothed reward ±1 std across seeds
- `final_performance.png` — final 50-ep mean ±std bar chart
- `timing_comparison.png` — wall-clock training time per algorithm
- `hyperparameter_ablation.png` — 3-value × 3-seed ablation panels

---

## Setup & Reproduction

### 1. Clone the repo
```bash
git clone https://github.com/gangadharsaivignesh/MSDS684-RL-Labs.git
cd MSDS684-RL-Labs/Week8
```

### 2. Create a virtual environment (recommended)
```bash
python -m venv venv
source venv/bin/activate        # macOS / Linux
# venv\Scripts\activate         # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

> **Note on Box2D:** `gymnasium[box2d]` requires `swig`. If the install fails:
> - **macOS:** `brew install swig` then re-run pip install
> - **Ubuntu/Debian:** `sudo apt-get install swig` then re-run pip install
> - **Windows:** `pip install swig` then re-run pip install

### 4. Run the notebook
```bash
jupyter notebook Lab8_Algorithm_Comparison_final.ipynb
```

Run all cells top-to-bottom. Expect **15–45 minutes** depending on hardware (the hyperparameter ablation trains 9 additional algorithm variants across 3 seeds each).

---

## Key Implementation Notes

### Environment pinning
The environment is hard-pinned to `LunarLander-v2`. Change `ENV_NAME` in the imports cell if your gymnasium version only supports `LunarLander-v3`.

### REINFORCE Advantage Computation (corrected)
Subtract the baseline from **raw** discounted returns, then normalize the resulting advantages:

```python
# CORRECT
advantages = returns - values_t.detach()
advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

# WRONG — trains value network on normalized targets, not actual G_t
# advantages = (returns - returns.mean()) / (returns.std() + 1e-8) - values_t.detach()
```

### Dyna-Q Terminal State Handling (corrected)
```python
model[(s, a)] = (r, s_next, done)          # store done flag
# During planning:
pr, ps_next, p_done = model[(ps, pa)]
pb_next = 0.0 if p_done else np.max(Q[ps_next])   # zero-bootstrap at terminals
```

---

## Expected Results (approximate)

| Algorithm | Final 50-ep Mean | Std (3 seeds) | Solved (≥200)? |
|---|---|---|---|
| Tabular Q-Learning | ~−120 to −80 | ~±31 | No |
| REINFORCE w/ Baseline | ~−30 to +40 | ~±29 | Unlikely in 500 ep |
| Dyna-Q (n=5) | ~−90 to −50 | ~±25 | No |

> 3 seeds is insufficient for formal statistical claims — treat as indicative.

---

## References

- Sutton & Barto, *Reinforcement Learning: An Introduction* (2nd ed.)
  - Ch. 6.5 — Q-Learning · Ch. 8.2 — Dyna-Q · Ch. 13.3 — REINFORCE with Baseline
