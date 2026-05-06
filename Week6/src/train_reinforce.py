"""
train_reinforce.py — REINFORCE on CartPole-v1 (standalone script).

Trains REINFORCE with and without a learned state-value baseline across
N_SEEDS seeds × EPISODES episodes each. Results are cached to
results/results_reinforce.npz; re-running loads the cache.

Usage:
    python src/train_reinforce.py
    python src/train_reinforce.py --seeds 5 --episodes 300   # quick smoke test
"""
import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import gymnasium as gym

# ── project imports ──────────────────────────────────────────────────────────
import sys
sys.path.insert(0, str(Path(__file__).parent))
from models import CategoricalPolicy, ValueNet
from utils  import set_seed, smooth, ci95, style, save_fig, save_checkpoint

# ---------------------------------------------------------------------------
# Defaults (overridable via CLI)
# ---------------------------------------------------------------------------
DEFAULTS = dict(
    seeds    = 30,
    episodes = 600,
    gamma    = 0.99,
    lr_pi    = 1e-3,
    lr_v     = 1e-3,
    hidden   = 128,
)

ROOT     = Path(__file__).parent.parent
RESULTS  = ROOT / 'results' / 'results_reinforce.npz'
CKPT_DIR = ROOT / 'checkpoints'
PLOT_DIR = ROOT / 'plots'

DEVICE   = torch.device('cpu')


# ---------------------------------------------------------------------------
# Single-seed training
# ---------------------------------------------------------------------------

def run_reinforce(seed: int, episodes: int, gamma: float,
                  lr_pi: float, lr_v: float, hidden: int,
                  use_baseline: bool):
    """Train one REINFORCE agent; return per-episode return array."""
    set_seed(seed)
    env = gym.make('CartPole-v1')
    env.reset(seed=seed)
    env.action_space.seed(seed)

    policy = CategoricalPolicy(4, 2, hidden=hidden).to(DEVICE)
    opt_pi = optim.Adam(policy.parameters(), lr=lr_pi)

    if use_baseline:
        value = ValueNet(4, hidden=hidden).to(DEVICE)
        opt_v = optim.Adam(value.parameters(), lr=lr_v)

    returns_per_ep = np.zeros(episodes, dtype=np.float32)

    for ep in range(episodes):
        obs, _ = env.reset()
        log_probs, rewards, states = [], [], []
        done = False
        while not done:
            x    = torch.as_tensor(obs, dtype=torch.float32, device=DEVICE)
            dist = policy.dist(x)
            a    = dist.sample()
            log_probs.append(dist.log_prob(a))
            states.append(x)
            obs, r, term, trunc, _ = env.step(int(a.item()))
            rewards.append(float(r))
            done = term or trunc

        # Discounted returns G_t (backwards pass)
        G = 0.0; Gs = []
        for r in reversed(rewards):
            G = r + gamma * G
            Gs.insert(0, G)
        Gs_raw = torch.tensor(Gs, dtype=torch.float32, device=DEVICE)

        if use_baseline:
            S          = torch.stack(states)
            V          = value(S)
            advantages = Gs_raw - V.detach()
            adv        = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            policy_loss = -(torch.stack(log_probs) * adv).sum()
            value_loss  = F.mse_loss(V, Gs_raw)
            opt_v.zero_grad(); value_loss.backward(); opt_v.step()
        else:
            adv = (Gs_raw - Gs_raw.mean()) / (Gs_raw.std() + 1e-8)
            policy_loss = -(torch.stack(log_probs) * adv).sum()

        opt_pi.zero_grad(); policy_loss.backward(); opt_pi.step()
        returns_per_ep[ep] = sum(rewards)

    env.close()
    return returns_per_ep, policy, (value if use_baseline else None)


# ---------------------------------------------------------------------------
# Multi-seed sweep
# ---------------------------------------------------------------------------

def sweep(use_baseline: bool, seeds: int, episodes: int,
          gamma: float, lr_pi: float, lr_v: float, hidden: int):
    label  = 'with baseline' if use_baseline else 'no baseline  '
    out    = np.zeros((seeds, episodes), dtype=np.float32)
    saved  = None
    t0     = time.time()
    for s in range(seeds):
        rets, pol, val = run_reinforce(
            seed=s, episodes=episodes, gamma=gamma,
            lr_pi=lr_pi, lr_v=lr_v, hidden=hidden,
            use_baseline=use_baseline,
        )
        out[s] = rets
        if s == 0:
            saved = (pol.state_dict(), val.state_dict() if val else None)
        if (s + 1) % 5 == 0:
            print(f'  [{label}] seed {s+1:>2}/{seeds}  '
                  f'elapsed {time.time()-t0:5.1f}s  '
                  f'last-20 mean: {out[s, -20:].mean():6.1f}')
    return out, saved


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def plot_results(ret_no: np.ndarray, ret_bl: np.ndarray,
                 episodes: int, seeds: int, plot_dir: Path):
    style()
    xs = np.arange(episodes)

    # Figure 1 — learning curves
    fig, ax = plt.subplots(figsize=(9, 5))
    for ret, label, color in [
        (ret_no, 'REINFORCE (no baseline)', 'C0'),
        (ret_bl, 'REINFORCE w/ baseline',   'C1'),
    ]:
        mean = ret.mean(0); ci = ci95(ret)
        ax.plot(xs, smooth(mean), color=color, label=label)
        ax.fill_between(xs, smooth(mean - ci), smooth(mean + ci),
                        alpha=0.20, color=color)
    ax.axhline(500, color='gray', ls='--', alpha=0.5, lw=1, label='Optimal (500)')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Return (smoothed, 20-ep window)')
    ax.set_title(f'CartPole-v1 · REINFORCE learning curves · {seeds} seeds · 95% CI')
    ax.legend(loc='lower right')
    save_fig(fig, plot_dir / 'fig1_reinforce_learning_curves.png')
    plt.close(fig)

    # Figure 2 — across-seed variance
    fig, ax = plt.subplots(figsize=(9, 4.5))
    var_no = ret_no.var(axis=0)
    var_bl = ret_bl.var(axis=0)
    ax.plot(smooth(var_no, 20), color='C0', label='REINFORCE (no baseline)')
    ax.plot(smooth(var_bl, 20), color='C1', label='REINFORCE w/ baseline')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Across-seed variance of return')
    ax.set_title('Variance across seeds per episode (lower = tighter)')
    ax.legend()
    save_fig(fig, plot_dir / 'fig2_reinforce_variance.png')
    plt.close(fig)

    print(f'\nVariance summary:')
    print(f'  Mean variance (no baseline) : {var_no.mean():.1f}')
    print(f'  Mean variance (w/ baseline) : {var_bl.mean():.1f}')
    print(f'  Reduction factor            : {var_no.mean()/var_bl.mean():.2f}×')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seeds',    type=int,   default=DEFAULTS['seeds'])
    parser.add_argument('--episodes', type=int,   default=DEFAULTS['episodes'])
    parser.add_argument('--gamma',    type=float, default=DEFAULTS['gamma'])
    parser.add_argument('--lr_pi',    type=float, default=DEFAULTS['lr_pi'])
    parser.add_argument('--lr_v',     type=float, default=DEFAULTS['lr_v'])
    parser.add_argument('--hidden',   type=int,   default=DEFAULTS['hidden'])
    parser.add_argument('--force',    action='store_true',
                        help='Re-run even if cache exists')
    args = parser.parse_args()

    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    if RESULTS.exists() and not args.force:
        cache  = np.load(RESULTS)
        ret_no = cache['no_baseline']
        ret_bl = cache['with_baseline']
        print(f'Loaded cache: {RESULTS}  shape={ret_no.shape}')
    else:
        print('=== REINFORCE without baseline ===')
        ret_no, ckpt_no = sweep(False, args.seeds, args.episodes,
                                args.gamma, args.lr_pi, args.lr_v, args.hidden)
        print('=== REINFORCE with baseline ===')
        ret_bl, ckpt_bl = sweep(True, args.seeds, args.episodes,
                                args.gamma, args.lr_pi, args.lr_v, args.hidden)

        np.savez(RESULTS, no_baseline=ret_no, with_baseline=ret_bl)
        save_checkpoint(ckpt_no[0], CKPT_DIR / 'cartpole_reinforce_nobaseline.pt')
        save_checkpoint(ckpt_bl[0], CKPT_DIR / 'cartpole_reinforce_baseline_policy.pt')
        if ckpt_bl[1]:
            save_checkpoint(ckpt_bl[1], CKPT_DIR / 'cartpole_reinforce_baseline_value.pt')

    plot_results(ret_no, ret_bl, args.episodes, args.seeds, PLOT_DIR)
    print('\nDone.')


if __name__ == '__main__':
    main()
