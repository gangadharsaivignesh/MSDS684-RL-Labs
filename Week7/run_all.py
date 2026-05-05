"""
run_all.py — Run every experiment in sequence and save all figures.
Usage:
    python run_all.py            # all experiments
    python run_all.py 1 2 3      # specific experiment numbers
"""

import subprocess
import sys
import time
from pathlib import Path

EXPERIMENTS = [
    (1, "experiments/01_planning_multiplier.py",
     "Planning multiplier — Dyna-Q n∈{0,5,10,50}"),
    (2, "experiments/02_dyna_q_plus.py",
     "Dyna-Q+ on non-stationary Taxi-v4"),
    (3, "experiments/03_prioritized_sweeping.py",
     "Prioritized sweeping vs uniform replay"),
    (4, "experiments/04_neural_world_model.py",
     "Neural world model + REINFORCE on CartPole-v1"),
    (5, "experiments/05_kappa_sensitivity.py",
     "[EXTRA] κ sensitivity sweep for Dyna-Q+"),
    (6, "experiments/06_theta_sensitivity.py",
     "[EXTRA] θ sensitivity sweep for prioritized sweeping"),
    (7, "experiments/07_prioritized_vs_uniform_budgets.py",
     "[EXTRA] Prioritized n=10 vs uniform n=50 — key missing comparison"),
]

ROOT = Path(__file__).parent

def run(num, script, desc):
    print(f"\n{'='*60}")
    print(f"Experiment {num}: {desc}")
    print(f"{'='*60}")
    t0  = time.time()
    ret = subprocess.call([sys.executable, ROOT / script])
    elapsed = time.time() - t0
    status = "✓" if ret == 0 else "✗ FAILED"
    print(f"\n{status}  ({elapsed:.0f}s)")
    return ret == 0


if __name__ == "__main__":
    # Determine which experiments to run
    if len(sys.argv) > 1:
        to_run = {int(x) for x in sys.argv[1:]}
        exps   = [(n, s, d) for n, s, d in EXPERIMENTS if n in to_run]
    else:
        exps = EXPERIMENTS

    passed, failed = [], []
    total_t0 = time.time()
    for num, script, desc in exps:
        ok = run(num, script, desc)
        (passed if ok else failed).append(num)

    print(f"\n{'='*60}")
    print(f"Summary: {len(passed)} passed, {len(failed)} failed "
          f"in {time.time()-total_t0:.0f}s")
    if failed:
        print(f"  Failed: experiments {failed}")
    print(f"  Figures written to: {ROOT / 'figures'}/")
