"""
utils.py — Shared utilities for MSDS 684 Week 6.

Covers: reproducible seeding, smoothing, confidence intervals, and
matplotlib styling helpers used by all training scripts and the notebook.
"""
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from pathlib import Path


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int):
    """Seed torch, numpy, and Python's hash for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)


# ---------------------------------------------------------------------------
# Statistics helpers
# ---------------------------------------------------------------------------

def smooth(x: np.ndarray, k: int = 20) -> np.ndarray:
    """Centred moving average; preserves array length via 'same' convolution."""
    x = np.asarray(x, dtype=float)
    if k <= 1:
        return x
    return np.convolve(x, np.ones(k) / k, mode='same')


def ci95(arr: np.ndarray, axis: int = 0) -> np.ndarray:
    """95% confidence interval half-width (1.96 × SEM)."""
    n  = arr.shape[axis]
    se = arr.std(axis=axis, ddof=1) / np.sqrt(n)
    return 1.96 * se


# ---------------------------------------------------------------------------
# Matplotlib style
# ---------------------------------------------------------------------------

def style():
    """Apply a clean, consistent plot style across all figures."""
    plt.rcParams.update({
        'figure.dpi':        130,
        'axes.spines.top':   False,
        'axes.spines.right': False,
        'axes.grid':         True,
        'grid.alpha':        0.3,
        'font.size':         11,
        'lines.linewidth':   2.0,
    })


def save_fig(fig: plt.Figure, path: Path, tight: bool = True):
    if tight:
        fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches='tight')
    print(f'  saved → {path}')


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def save_checkpoint(state_dict, path: Path):
    torch.save(state_dict, path)
    print(f'  checkpoint → {path}')


def load_checkpoint(model, path: Path):
    model.load_state_dict(torch.load(path, weights_only=True))
    return model
