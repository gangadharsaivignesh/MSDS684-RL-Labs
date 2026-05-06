"""
extra_experiments.py — Three additional experiments beyond the lab PDF.

Experiment A: Learning-rate sensitivity sweep (REINFORCE w/ baseline)
    Grid: lr_pi ∈ {3e-4, 1e-3, 3e-3}  ×  lr_v ∈ {1e-3, 5e-3}
    Metric: mean return over last 50 episodes across 10 seeds each.
    Insight: shows the policy LR is more sensitive than the value LR.

Experiment B: Entropy-bonus ablation on LunarLanderContinuous
    β ∈ {0, 1e-4, 1e-3, 5e-3, 1e-2}  ×  10 seeds × 300 episodes
    Metric: final-50-episode mean return per β.
    Insight: β=0 collapses σ early and stalls; too-large β prevents commitment.

Experiment C: Gradient-clip threshold sweep on LunarLanderContinuous
    clip ∈ {0.1, 0.5, 1.0, 5.0, None}  ×  10 seeds × 300 episodes
    Metric: final-50-episode mean return and training stability (IQR).
    Insight: clip=0.5 is the sweet spot; larger clips allow gradient spikes.

Usage:
    python experiments/extra_experiments.py              # run all three
    python experiments/extra_experiments.py --exp A      # run one
    python experiments/extra_experiments.py --force      # ignore cache
"""
import argparse
import time
import itertools
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
from models import CategoricalPolicy, ValueNet, GaussianPolicy
from utils  import set_seed, smooth, ci95, style, save_fig

ROOT     = Path(__file__).parent.parent
RES_DIR  = ROOT / 'results'
PLOT_DIR = ROOT / 'plots'
DEVICE   = torch.device('cpu')


# ===========================================================================
# Shared training kernels (lightweight, no caching per call)
# ===========================================================================

def _run_reinforce_quick(seed, episodes, gamma=0.99, lr_pi=1e-3, lr_v=1e-3,
                          hidden=128):
    """REINFORCE w/ baseline; returns final-50-episode mean return."""
    from torch.distributions import Categorical
    import torch.nn.functional as F

    set_seed(seed)
    env = gym.make('CartPole-v1')
    env.reset(seed=seed); env.action_space.seed(seed)

    policy = CategoricalPolicy(4, 2, hidden=hidden).to(DEVICE)
    value  = ValueNet(4, hidden=hidden).to(DEVICE)
    opt_pi = torch.optim.Adam(policy.parameters(), lr=lr_pi)
    opt_v  = torch.optim.Adam(value.parameters(),  lr=lr_v)

    ep_returns = np.zeros(episodes, dtype=np.float32)
    for ep in range(episodes):
        obs, _ = env.reset()
        log_probs, rewards, states = [], [], []
        done = False
        while not done:
            x    = torch.as_tensor(obs, dtype=torch.float32)
            dist = policy.dist(x)
            a    = dist.sample()
            log_probs.append(dist.log_prob(a))
            states.append(x)
            obs, r, term, trunc, _ = env.step(int(a.item()))
            rewards.append(float(r)); done = term or trunc

        G = 0.0; Gs = []
        for r in reversed(rewards): G = r + gamma * G; Gs.insert(0, G)
        Gs_raw = torch.tensor(Gs, dtype=torch.float32)
        S      = torch.stack(states)
        V      = value(S)
        adv    = (Gs_raw - V.detach())
        adv    = (adv - adv.mean()) / (adv.std() + 1e-8)
        policy_loss = -(torch.stack(log_probs) * adv).sum()
        value_loss  = F.mse_loss(V, Gs_raw)
        opt_v.zero_grad(); value_loss.backward(); opt_v.step()
        opt_pi.zero_grad(); policy_loss.backward(); opt_pi.step()
        ep_returns[ep] = sum(rewards)

    env.close()
    return ep_returns


def _run_ac_ll_quick(seed, episodes, gamma=0.99,
                     lr_actor=1e-3, lr_critic=5e-3, hidden=128,
                     entropy_beta=1e-3, init_log_std=-0.5,
                     reward_scale=0.1, max_grad_norm=0.5):
    """Minimal AC for LunarLanderContinuous; returns per-episode return array."""
    obs_scale = np.array([1.5, 1.5, 5.0, 5.0, 3.14, 5.0, 1.0, 1.0], np.float32)
    set_seed(seed)
    env = gym.make('LunarLanderContinuous-v3')
    env.reset(seed=seed); env.action_space.seed(seed)

    actor  = GaussianPolicy(8, 2, hidden=hidden, init_log_std=init_log_std).to(DEVICE)
    critic = ValueNet(8, hidden=hidden, activation='tanh').to(DEVICE)
    opt_a  = torch.optim.Adam(actor.parameters(),  lr=lr_actor)
    opt_c  = torch.optim.Adam(critic.parameters(), lr=lr_critic)

    ep_returns = np.zeros(episodes, dtype=np.float32)
    for ep in range(episodes):
        obs, _ = env.reset()
        ep_R, steps = 0.0, 0
        done = False
        while not done:
            obs_n = obs / obs_scale
            x     = torch.as_tensor(obs_n, dtype=torch.float32)
            dist  = actor.dist(x)
            a     = dist.sample()
            logp  = dist.log_prob(a).sum()
            H     = dist.entropy().sum()
            a_np  = a.detach().cpu().numpy().clip(-1.0, 1.0)

            obs_next, r, term, trunc, _ = env.step(a_np)
            done   = term or trunc
            xn     = torch.as_tensor(obs_next / obs_scale, dtype=torch.float32)

            v      = critic(x)
            v_next = critic(xn).detach()
            target = float(r) * reward_scale + gamma * v_next * (0.0 if term else 1.0)
            td     = target - v

            critic_loss = td.pow(2)
            actor_loss  = -(logp * td.detach()) - entropy_beta * H

            opt_c.zero_grad(); critic_loss.backward()
            if max_grad_norm is not None:
                nn.utils.clip_grad_norm_(critic.parameters(), max_grad_norm)
            opt_c.step()
            opt_a.zero_grad(); actor_loss.backward()
            if max_grad_norm is not None:
                nn.utils.clip_grad_norm_(actor.parameters(), max_grad_norm)
            opt_a.step()

            ep_R += float(r); steps += 1; obs = obs_next
        ep_returns[ep] = ep_R
    env.close()
    return ep_returns


# ===========================================================================
# Experiment A — LR sensitivity (REINFORCE + baseline, CartPole)
# ===========================================================================

def experiment_A(force=False):
    print('\n' + '='*60)
    print('  Experiment A: LR sensitivity — REINFORCE w/ baseline')
    print('='*60)

    cache = RES_DIR / 'expA_lr_sensitivity.npz'
    lr_pi_grid  = [3e-4, 1e-3, 3e-3]
    lr_v_grid   = [1e-3, 5e-3]
    n_seeds     = 10
    episodes    = 400
    combos      = list(itertools.product(lr_pi_grid, lr_v_grid))

    if cache.exists() and not force:
        data = np.load(cache)
        final_means = data['final_means']   # shape (n_combos, n_seeds)
        print(f'  Loaded cache: {cache}')
    else:
        final_means = np.zeros((len(combos), n_seeds), dtype=np.float32)
        for i, (lr_pi, lr_v) in enumerate(combos):
            t0 = time.time()
            for s in range(n_seeds):
                rets = _run_reinforce_quick(s, episodes, lr_pi=lr_pi, lr_v=lr_v)
                final_means[i, s] = rets[-50:].mean()
            print(f'  lr_pi={lr_pi:.0e}  lr_v={lr_v:.0e}  '
                  f'mean={final_means[i].mean():6.1f} ± {final_means[i].std():4.1f}  '
                  f'({time.time()-t0:.0f}s)')
        np.savez(cache, final_means=final_means)

    # Plot: grouped bar chart
    style()
    fig, ax = plt.subplots(figsize=(10, 4.5))
    x     = np.arange(len(lr_pi_grid))
    w     = 0.35
    colors = ['C0', 'C1']
    for j, lr_v in enumerate(lr_v_grid):
        idxs  = [i for i, (_, lv) in enumerate(combos) if lv == lr_v]
        means = final_means[idxs].mean(axis=1)
        stds  = final_means[idxs].std(axis=1)
        ax.bar(x + j*w, means, w, yerr=stds, capsize=5,
               color=colors[j], alpha=0.85, label=f'lr_v={lr_v:.0e}')

    ax.set_xticks(x + w/2)
    ax.set_xticklabels([f'{lp:.0e}' for lp in lr_pi_grid])
    ax.set_xlabel('Policy learning rate (lr_pi)')
    ax.set_ylabel('Final-50-ep mean return')
    ax.set_title('Exp A: LR Sensitivity — REINFORCE w/ baseline (CartPole)\n'
                 f'{n_seeds} seeds × {episodes} episodes each')
    ax.legend(title='lr_v'); ax.axhline(500, ls='--', color='gray', alpha=0.5, lw=1)
    save_fig(fig, PLOT_DIR / 'expA_lr_sensitivity.png')
    plt.close(fig)

    # Print best config
    best_i = final_means.mean(axis=1).argmax()
    print(f'\n  Best config: lr_pi={combos[best_i][0]:.0e}  '
          f'lr_v={combos[best_i][1]:.0e}  '
          f'mean={final_means[best_i].mean():.1f}')


# ===========================================================================
# Experiment B — Entropy bonus ablation (LunarLander AC)
# ===========================================================================

def experiment_B(force=False):
    print('\n' + '='*60)
    print('  Experiment B: Entropy bonus ablation — LunarLander AC')
    print('='*60)

    cache   = RES_DIR / 'expB_entropy_ablation.npz'
    betas   = [0.0, 1e-4, 1e-3, 5e-3, 1e-2]
    n_seeds = 10
    episodes= 300

    if cache.exists() and not force:
        data  = np.load(cache)
        all_returns = data['all_returns']    # (n_betas, n_seeds, episodes)
        print(f'  Loaded cache: {cache}')
    else:
        all_returns = np.zeros((len(betas), n_seeds, episodes), dtype=np.float32)
        for i, beta in enumerate(betas):
            t0 = time.time()
            for s in range(n_seeds):
                rets = _run_ac_ll_quick(s, episodes, entropy_beta=beta)
                all_returns[i, s] = rets
            print(f'  β={beta:.0e}  '
                  f'final-50 mean={all_returns[i, :, -50:].mean():7.1f}  '
                  f'({time.time()-t0:.0f}s)')
        np.savez(cache, all_returns=all_returns)

    style()
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
    xs = np.arange(episodes)
    cmap = plt.get_cmap('plasma')
    colors = [cmap(i / max(1, len(betas) - 1)) for i in range(len(betas))]

    # Left: learning curves
    for i, (beta, color) in enumerate(zip(betas, colors)):
        mean = all_returns[i].mean(0)
        ci   = ci95(all_returns[i])
        axes[0].plot(xs, smooth(mean, 15), color=color, label=f'β={beta:.0e}')
        axes[0].fill_between(xs, smooth(mean - ci, 15), smooth(mean + ci, 15),
                             alpha=0.12, color=color)
    axes[0].set_xlabel('Episode'); axes[0].set_ylabel('Return (smoothed)')
    axes[0].set_title('Learning curves by entropy bonus β')
    axes[0].legend(fontsize=8)

    # Right: final-50 bar chart
    final_means = all_returns[:, :, -50:].mean(axis=2)   # (n_betas, n_seeds)
    means = final_means.mean(axis=1); stds = final_means.std(axis=1)
    beta_labels = [f'{b:.0e}' for b in betas]
    axes[1].bar(beta_labels, means, yerr=stds, capsize=6,
                color=colors, alpha=0.85)
    axes[1].set_xlabel('Entropy bonus β')
    axes[1].set_ylabel('Final-50-ep mean return')
    axes[1].set_title('Final performance by β')

    fig.suptitle(f'Exp B: Entropy Ablation — LunarLander AC\n'
                 f'{n_seeds} seeds × {episodes} episodes', y=1.02)
    save_fig(fig, PLOT_DIR / 'expB_entropy_ablation.png')
    plt.close(fig)

    best_i = means.argmax()
    print(f'\n  Best β: {betas[best_i]:.0e}  mean={means[best_i]:.1f}')


# ===========================================================================
# Experiment C — Gradient clip threshold (LunarLander AC)
# ===========================================================================

def experiment_C(force=False):
    print('\n' + '='*60)
    print('  Experiment C: Gradient clip sweep — LunarLander AC')
    print('='*60)

    cache   = RES_DIR / 'expC_grad_clip.npz'
    clips   = [0.1, 0.5, 1.0, 5.0, None]   # None = no clipping
    n_seeds = 10
    episodes= 300

    if cache.exists() and not force:
        data = np.load(cache, allow_pickle=True)
        all_returns = data['all_returns']
        print(f'  Loaded cache: {cache}')
    else:
        all_returns = np.zeros((len(clips), n_seeds, episodes), dtype=np.float32)
        for i, clip in enumerate(clips):
            t0 = time.time()
            for s in range(n_seeds):
                rets = _run_ac_ll_quick(s, episodes, max_grad_norm=clip)
                all_returns[i, s] = rets
            lab = str(clip)
            print(f'  clip={lab:<5}  '
                  f'final-50 mean={all_returns[i, :, -50:].mean():7.1f}  '
                  f'IQR={np.percentile(all_returns[i,:,-50:].mean(1),75) - np.percentile(all_returns[i,:,-50:].mean(1),25):.1f}  '
                  f'({time.time()-t0:.0f}s)')
        np.savez(cache, all_returns=all_returns)

    style()
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
    xs     = np.arange(episodes)
    labels = [str(c) if c is not None else 'None' for c in clips]
    cmap   = plt.get_cmap('tab10')
    colors = [cmap(i) for i in range(len(clips))]

    for i, (label, color) in enumerate(zip(labels, colors)):
        mean = all_returns[i].mean(0); ci = ci95(all_returns[i])
        axes[0].plot(xs, smooth(mean, 15), color=color, label=f'clip={label}')
        axes[0].fill_between(xs, smooth(mean - ci, 15), smooth(mean + ci, 15),
                             alpha=0.12, color=color)
    axes[0].set_xlabel('Episode'); axes[0].set_ylabel('Return (smoothed)')
    axes[0].set_title('Learning curves by gradient clip')
    axes[0].legend(fontsize=8)

    final_means = all_returns[:, :, -50:].mean(axis=2)
    q25 = np.percentile(final_means, 25, axis=1)
    q75 = np.percentile(final_means, 75, axis=1)
    medians = np.median(final_means, axis=1)

    x_pos = np.arange(len(clips))
    axes[1].bar(x_pos, medians, yerr=[medians - q25, q75 - medians],
                capsize=6, color=colors, alpha=0.85)
    axes[1].set_xticks(x_pos); axes[1].set_xticklabels(labels)
    axes[1].set_xlabel('Gradient clip threshold')
    axes[1].set_ylabel('Final-50-ep median return (IQR bars)')
    axes[1].set_title('Stability by clip threshold')

    fig.suptitle(f'Exp C: Gradient Clip Sweep — LunarLander AC\n'
                 f'{n_seeds} seeds × {episodes} episodes', y=1.02)
    save_fig(fig, PLOT_DIR / 'expC_grad_clip.png')
    plt.close(fig)

    best_i = medians.argmax()
    print(f'\n  Best clip: {clips[best_i]}  median={medians[best_i]:.1f}')


# ===========================================================================
# Main
# ===========================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp',   choices=['A', 'B', 'C', 'all'], default='all')
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args()

    for d in [RES_DIR, PLOT_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    run = args.exp
    if run in ('A', 'all'): experiment_A(force=args.force)
    if run in ('B', 'all'): experiment_B(force=args.force)
    if run in ('C', 'all'): experiment_C(force=args.force)

    print('\nExtra experiments complete. Plots saved to', PLOT_DIR)


if __name__ == '__main__':
    main()
