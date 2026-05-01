# Lab 4 — SARSA vs Q-Learning on CliffWalking-v0

**MSDS 684 — Reinforcement Learning · Regis University**
**Author:** Saivignesh Gangadhar

This lab implements two on-line TD control algorithms — **SARSA** (on-policy) and
**Q-Learning** (off-policy) — on Gymnasium's `CliffWalking-v0` and contrasts the
policies they converge to. It is the canonical Sutton & Barto Example 6.6.

## Files

| File | Purpose |
|------|---------|
| `build_lab4_notebook.py` | Generator script that builds the notebook from a single source of truth. Run this to regenerate the notebook. |
| `Gangadhar_Saivignesh_Lab4.ipynb` | The executed notebook (53 cells). Open in Jupyter to read or in nbviewer to read on the web. |

## Reproducing the results

```bash
# from the repository root
pip install -r requirements.txt          # numpy, matplotlib, gymnasium, torch
cd Week4
python build_lab4_notebook.py            # rebuilds the .ipynb from the .py
jupyter nbconvert --to notebook --execute Gangadhar_Saivignesh_Lab4.ipynb \
  --output Gangadhar_Saivignesh_Lab4.ipynb
```

End-to-end execution time on a 2024 MacBook: ~90 seconds (≈30 s for the
30-seed main sweep, ≈30 s for the α-sweep, ≈30 s for the four ε-decay
schedules).

## Notebook structure

| Section | Content |
|---------|---------|
| 1–3     | Setup, environment inspection, on-/off-policy theory |
| 4–5     | SARSA and Q-Learning implementations (NumPy Q-tables, online updates) |
| 6       | Single-seed sanity demo |
| 7–8     | 30-seed sweep + 95 % CI learning curves |
| 9       | Greedy-policy arrow grids |
| 10      | Value-function heatmaps (greedy and on-policy readings) |
| 11      | Sample greedy trajectories |
| 11.5    | Greedy vs behavior policy evaluation (the sharpest finding) |
| 12      | α (step-size) sweep |
| 13      | ε-decay schedule comparison |
| 14      | Final comparison table |
| 15      | Behavioral analysis & summary |

## Headline findings (30 seeds, α = 0.5, ε = 0.1, 500 episodes)

- SARSA's training-time return (≈ −29) sits ~17 reward-units **above**
  Q-Learning's (≈ −46) — consistent across α ∈ {0.05, …, 0.9}.
- Q-Learning's *greedy* policy posts exactly −13 on every seed (std = 0).
  SARSA's *median* greedy seed posts −17, but **6 of 30 SARSA seeds loop
  forever at greedy evaluation** because SARSA's bootstrap target has no
  `max` and `np.argmax`'s first-index tie-break picks action 0 ("up") =
  no-op in row 0.
- ε-decay closes the gap (linear) or even flips it (exponential, by ≈ 1.4
  reward — exactly the path-length difference between the safe and
  optimal routes once exploration anneals to the 0.01 floor).
