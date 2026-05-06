"""
train_actor_critic.py — Online TD(0) Actor-Critic (standalone script).

Runs two sweeps back-to-back:
  1. Pendulum-v1        (30 seeds × 500 episodes) — documents the failure
  2. LunarLanderContinuous-v3 (30 seeds × 500 episodes) — working pivot

Results are cached; re-running loads from cache unless --force is passed.

Usage:
    python src/train_actor_critic.py
    python src/train_actor_critic.py --env lunarlander --seeds 3 --episodes 50
    python src/train_actor_critic.py --force          # ignore all caches
"""
import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, str(Path(__file__).parent))
from models import GaussianPolicy, ValueNet
from utils  import set_seed, smooth, ci95, style, save_fig, save_checkpoint, load_checkpoint

# ---------------------------------------------------------------------------
# Environment configs
# ---------------------------------------------------------------------------

ENV_CFG = {
    'pendulum': dict(
        env_id      = 'Pendulum-v1',
        obs_dim     = 3,
        act_dim     = 1,
        act_low     = -2.0,
        act_high    = 2.0,
        obs_scale   = np.array([1.0, 1.0, 8.0], dtype=np.float32),
        fig_prefix  = 'pendulum',
        label       = 'Pendulum-v1',
        episodes    = 500,
        seeds       = 30,
    ),
    'lunarlander': dict(
        env_id      = 'LunarLanderContinuous-v3',
        obs_dim     = 8,
        act_dim     = 2,
        act_low     = -1.0,
        act_high    = 1.0,
        obs_scale   = np.array([1.5, 1.5, 5.0, 5.0, 3.14, 5.0, 1.0, 1.0], dtype=np.float32),
        fig_prefix  = 'lunarlander',
        label       = 'LunarLanderContinuous-v3',
        episodes    = 500,
        seeds       = 30,
    ),
}

HYPERPARAMS = dict(
    gamma        = 0.99,
    lr_actor     = 1e-3,
    lr_critic    = 5e-3,
    hidden       = 128,
    entropy_beta = 1e-3,
    init_log_std = -0.5,
    reward_scale = 0.1,
    max_grad_norm= 0.5,
)

ROOT     = Path(__file__).parent.parent
CKPT_DIR = ROOT / 'checkpoints'
PLOT_DIR = ROOT / 'plots'
RES_DIR  = ROOT / 'results'

DEVICE   = torch.device('cpu')


# ---------------------------------------------------------------------------
# Single-seed training
# ---------------------------------------------------------------------------

def run_actor_critic(seed: int, cfg: dict, hp: dict,
                     log_trajectory_every=None):
    set_seed(seed)
    env = gym.make(cfg['env_id'])
    env.reset(seed=seed)
    env.action_space.seed(seed)

    actor  = GaussianPolicy(cfg['obs_dim'], cfg['act_dim'],
                             hidden=hp['hidden'],
                             init_log_std=hp['init_log_std']).to(DEVICE)
    critic = ValueNet(cfg['obs_dim'], hidden=hp['hidden'],
                      activation='tanh').to(DEVICE)
    opt_a  = optim.Adam(actor.parameters(),  lr=hp['lr_actor'])
    opt_c  = optim.Adam(critic.parameters(), lr=hp['lr_critic'])

    obs_scale = cfg['obs_scale']
    episodes  = cfg['episodes']

    ep_returns = np.zeros(episodes, dtype=np.float32)
    ep_entropy = np.zeros(episodes, dtype=np.float32)
    ep_td_abs  = np.zeros(episodes, dtype=np.float32)
    trajectories = {}

    for ep in range(episodes):
        obs, _ = env.reset()
        ep_R, ep_H, ep_td_sum, steps = 0.0, 0.0, 0.0, 0
        log_this = (log_trajectory_every is not None
                    and ep % log_trajectory_every == 0)
        traj = {'obs': [], 'action': [], 'reward': []} if log_this else None
        done = False

        while not done:
            obs_n = obs / obs_scale
            x     = torch.as_tensor(obs_n, dtype=torch.float32, device=DEVICE)
            dist  = actor.dist(x)
            a     = dist.sample()
            logp  = dist.log_prob(a).sum()
            H     = dist.entropy().sum()
            a_np  = a.detach().cpu().numpy().clip(cfg['act_low'], cfg['act_high'])

            obs_next, r, term, trunc, _ = env.step(a_np)
            done   = term or trunc
            obs_n2 = obs_next / obs_scale
            xn     = torch.as_tensor(obs_n2, dtype=torch.float32, device=DEVICE)

            v      = critic(x)
            v_next = critic(xn).detach()
            # Bootstrap through truncation; zero v_next only on true termination
            not_terminal = 0.0 if term else 1.0
            target = float(r) * hp['reward_scale'] + hp['gamma'] * v_next * not_terminal
            td     = target - v

            critic_loss = td.pow(2)
            actor_loss  = -(logp * td.detach()) - hp['entropy_beta'] * H

            opt_c.zero_grad(); critic_loss.backward()
            nn.utils.clip_grad_norm_(critic.parameters(), hp['max_grad_norm'])
            opt_c.step()

            opt_a.zero_grad(); actor_loss.backward()
            nn.utils.clip_grad_norm_(actor.parameters(),  hp['max_grad_norm'])
            opt_a.step()

            if traj is not None:
                traj['obs'].append(obs.copy())
                traj['action'].append(a_np.copy())
                traj['reward'].append(float(r))

            ep_R       += float(r)
            ep_H       += H.item()
            ep_td_sum  += abs(td.item())
            steps      += 1
            obs         = obs_next

        ep_returns[ep] = ep_R
        ep_entropy[ep] = ep_H / max(steps, 1)
        ep_td_abs[ep]  = ep_td_sum / max(steps, 1)
        if traj is not None:
            traj['obs']    = np.array(traj['obs'])
            traj['action'] = np.array(traj['action'])
            trajectories[ep] = traj

    env.close()
    return dict(returns=ep_returns, entropy=ep_entropy, td_abs=ep_td_abs,
                actor=actor, critic=critic, trajectories=trajectories)


# ---------------------------------------------------------------------------
# Multi-seed sweep
# ---------------------------------------------------------------------------

def run_sweep(cfg: dict, hp: dict, force: bool = False):
    prefix   = cfg['fig_prefix']
    res_path = RES_DIR / f'results_ac_{prefix}.npz'
    traj_path= RES_DIR / f'results_ac_{prefix}_seed0_traj.npz'
    actor_pt = CKPT_DIR / f'{prefix}_ac_actor.pt'
    critic_pt= CKPT_DIR / f'{prefix}_ac_critic.pt'
    seeds    = cfg['seeds']
    episodes = cfg['episodes']

    if res_path.exists() and traj_path.exists() and actor_pt.exists() and not force:
        cache      = np.load(res_path)
        returns    = cache['returns']
        entropy    = cache['entropy']
        td_abs     = cache['td_abs']
        print(f'[{prefix}] Loaded cache: {res_path}  shape={returns.shape}')

        actor0  = GaussianPolicy(cfg['obs_dim'], cfg['act_dim'],
                                  hidden=hp['hidden'],
                                  init_log_std=hp['init_log_std']).to(DEVICE)
        critic0 = ValueNet(cfg['obs_dim'], hidden=hp['hidden'],
                           activation='tanh').to(DEVICE)
        load_checkpoint(actor0, actor_pt)
        load_checkpoint(critic0, critic_pt)

        tcache = np.load(traj_path, allow_pickle=True)
        traj0  = {}
        for n in tcache.files:
            if n.endswith('_obs'):
                ep = int(n.split('_')[1])
                traj0[ep] = dict(obs=tcache[f'ep_{ep}_obs'],
                                 action=tcache[f'ep_{ep}_action'],
                                 reward=tcache[f'ep_{ep}_reward'])
        return returns, entropy, td_abs, actor0, critic0, traj0

    # Run the sweep
    returns = np.zeros((seeds, episodes), dtype=np.float32)
    entropy = np.zeros((seeds, episodes), dtype=np.float32)
    td_abs  = np.zeros((seeds, episodes), dtype=np.float32)
    actor0 = critic0 = traj0 = None
    t0 = time.time()

    for s in range(seeds):
        out = run_actor_critic(
            seed=s, cfg=cfg, hp=hp,
            log_trajectory_every=(50 if s == 0 else None),
        )
        returns[s] = out['returns']
        entropy[s] = out['entropy']
        td_abs[s]  = out['td_abs']
        if s == 0:
            actor0  = out['actor']
            critic0 = out['critic']
            traj0   = out['trajectories']
        if (s + 1) % 5 == 0:
            print(f'  [{prefix}] seed {s+1:>2}/{seeds}  '
                  f'elapsed {time.time()-t0:6.1f}s  '
                  f'last-50 mean: {returns[s, -50:].mean():7.1f}')

    # Save results
    np.savez(res_path, returns=returns, entropy=entropy, td_abs=td_abs)
    save_checkpoint(actor0.state_dict(),  actor_pt)
    save_checkpoint(critic0.state_dict(), critic_pt)
    flat = {}
    for ep, t in traj0.items():
        flat[f'ep_{ep}_obs']    = t['obs']
        flat[f'ep_{ep}_action'] = t['action']
        flat[f'ep_{ep}_reward'] = np.array(t['reward'])
    np.savez(traj_path, **flat)
    print(f'  [{prefix}] Saved results, trajectories, checkpoints.')
    return returns, entropy, td_abs, actor0, critic0, traj0


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def plot_ac(returns, entropy, td_abs, actor0, critic0, traj0,
            cfg: dict, plot_dir: Path, fig_offset: int = 3):
    style()
    prefix   = cfg['fig_prefix']
    label    = cfg['label']
    seeds    = cfg['seeds']
    episodes = cfg['episodes']
    xs       = np.arange(episodes)
    k        = 10 if episodes <= 200 else 20

    # Return curve
    fig, ax = plt.subplots(figsize=(9, 4.5))
    mean_R = returns.mean(0); ci_R = ci95(returns)
    ax.plot(xs, smooth(mean_R, k), color='C2', label='Actor-Critic (mean)')
    ax.fill_between(xs, smooth(mean_R - ci_R, k), smooth(mean_R + ci_R, k),
                    alpha=0.20, color='C2', label='95% CI')
    ax.set_xlabel('Episode'); ax.set_ylabel('Return (smoothed)')
    ax.set_title(f'{label} · Actor-Critic return · {seeds} seeds · 95% CI')
    ax.legend(loc='lower right')
    save_fig(fig, plot_dir / f'fig{fig_offset}_{prefix}_ac_return.png')
    plt.close(fig)

    # Entropy + |δ|
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    for ax_, data, ylabel, title, color in [
        (axes[0], entropy, 'H(π) per step (nats)', 'Policy entropy', 'C3'),
        (axes[1], td_abs,  '|TD error| per step',  '|δ| over training', 'C4'),
    ]:
        mean = data.mean(0); ci = ci95(data)
        ax_.plot(xs, smooth(mean, k), color=color)
        ax_.fill_between(xs, smooth(mean - ci, k), smooth(mean + ci, k),
                         alpha=0.20, color=color)
        ax_.set_xlabel('Episode'); ax_.set_ylabel(ylabel)
        ax_.set_title(title)
    fig.suptitle(f'{label} — diagnostics (mean ± 95% CI, {seeds} seeds)', y=1.02)
    save_fig(fig, plot_dir / f'fig{fig_offset+1}_{prefix}_ac_diagnostics.png')
    plt.close(fig)

    # Sample trajectories (seed 0)
    if traj0:
        ep_ids = sorted(traj0.keys())
        cmap   = plt.get_cmap('viridis')
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
        for i, ep in enumerate(ep_ids):
            t     = traj0[ep]
            color = cmap(i / max(1, len(ep_ids) - 1))
            if cfg['obs_dim'] == 3:   # Pendulum: recover θ from cosθ/sinθ
                o     = t['obs']
                theta = np.arctan2(o[:, 1], o[:, 0])
                axes[0].plot(theta, color=color, label=f'ep {ep}', lw=1.4)
                axes[0].axhline(0, color='k', ls=':', alpha=0.4)
                axes[0].set_ylabel('θ (rad)'); axes[0].set_title('Pole angle')
            else:                     # LunarLander: show altitude (y channel)
                o = t['obs']
                axes[0].plot(o[:, 1], color=color, label=f'ep {ep}', lw=1.4)
                axes[0].set_ylabel('Altitude (y)'); axes[0].set_title('Altitude over episode')
            axes[1].plot(np.cumsum(t['reward']), color=color, lw=1.4)
        for ax_ in axes:
            ax_.set_xlabel('Step'); ax_.grid(alpha=0.3)
        axes[1].set_ylabel('Cumulative reward')
        axes[1].set_title('Cumulative reward across training')
        axes[0].legend(fontsize=7, loc='upper right')
        fig.suptitle(f'Seed-0 trajectories across training — {label}', y=1.02)
        save_fig(fig, plot_dir / f'fig{fig_offset+2}_{prefix}_ac_trajectories.png')
        plt.close(fig)

    # Policy/value map (seed 0)
    _plot_policy_value_map(actor0, critic0, cfg, plot_dir, fig_offset + 3)


def _plot_policy_value_map(actor0, critic0, cfg: dict, plot_dir: Path, fig_num: int):
    """2-D slice of learned μ(s) and V(s) for the seed-0 policy."""
    n      = 80
    prefix = cfg['fig_prefix']

    if cfg['obs_dim'] == 3:   # Pendulum: sweep (θ, θ̇)
        theta_grid  = np.linspace(-np.pi, np.pi, n)
        thetad_grid = np.linspace(-8.0, 8.0, n)
        T, Td = np.meshgrid(theta_grid, thetad_grid)
        obs_grid = np.stack([np.cos(T), np.sin(T), Td], axis=-1).reshape(-1, 3)
        obs_n    = obs_grid / cfg['obs_scale']
        xlabel, ylabel = 'θ (rad)', 'θ̇ (rad/s)'
        ext_x, ext_y   = theta_grid, thetad_grid

        with torch.no_grad():
            obs_t    = torch.as_tensor(obs_n, dtype=torch.float32)
            mu, _    = actor0(obs_t)
            V        = critic0(obs_t)
        mu_grid = mu[:, 0].cpu().numpy().reshape(n, n)
        V_grid  = V.cpu().numpy().reshape(n, n)

        fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
        im0 = axes[0].imshow(mu_grid, origin='lower',
                             extent=[ext_x[0], ext_x[-1], ext_y[0], ext_y[-1]],
                             aspect='auto', cmap='RdBu_r', vmin=-2, vmax=2)
        axes[0].set_title('Learned μ(s) — torque\n(red=+, blue=−)')
        axes[0].set_xlabel(xlabel); axes[0].set_ylabel(ylabel)
        plt.colorbar(im0, ax=axes[0], label='torque')

        im1 = axes[1].imshow(V_grid, origin='lower',
                             extent=[ext_x[0], ext_x[-1], ext_y[0], ext_y[-1]],
                             aspect='auto', cmap='viridis')
        axes[1].set_title('Learned V(s)')
        axes[1].set_xlabel(xlabel); axes[1].set_ylabel(ylabel)
        plt.colorbar(im1, ax=axes[1], label='V(s)')

    else:   # LunarLander: sweep (x, y) with all other dims = 0
        x_grid = np.linspace(-1.5, 1.5, n)
        y_grid = np.linspace(0.0,  1.5, n)
        X, Y   = np.meshgrid(x_grid, y_grid)
        flat   = np.zeros((n * n, 8), dtype=np.float32)
        flat[:, 0] = X.ravel()
        flat[:, 1] = Y.ravel()
        obs_n  = flat / cfg['obs_scale']
        xlabel, ylabel = 'x position', 'y (altitude)'

        with torch.no_grad():
            obs_t = torch.as_tensor(obs_n, dtype=torch.float32)
            mu, _ = actor0(obs_t)
            V     = critic0(obs_t)
        mu_main    = mu[:, 0].cpu().numpy().reshape(n, n)
        mu_lateral = mu[:, 1].cpu().numpy().reshape(n, n)

        fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
        im0 = axes[0].imshow(mu_main, origin='lower',
                             extent=[x_grid[0], x_grid[-1], y_grid[0], y_grid[-1]],
                             aspect='auto', cmap='RdBu_r', vmin=-1, vmax=1)
        axes[0].set_title('μ_main(s) — main engine\n(red=fire, blue=cut)')
        axes[0].set_xlabel(xlabel); axes[0].set_ylabel(ylabel)
        plt.colorbar(im0, ax=axes[0], label='main throttle')

        im1 = axes[1].imshow(mu_lateral, origin='lower',
                             extent=[x_grid[0], x_grid[-1], y_grid[0], y_grid[-1]],
                             aspect='auto', cmap='RdBu_r', vmin=-1, vmax=1)
        axes[1].set_title('μ_lateral(s) — lateral engine\n(red=right, blue=left)')
        axes[1].set_xlabel(xlabel); axes[1].set_ylabel(ylabel)
        plt.colorbar(im1, ax=axes[1], label='lateral throttle')

    fig.suptitle(f'Seed-0 learned policy/value map', y=1.02)
    save_fig(fig, plot_dir / f'fig{fig_num}_{prefix}_policy_value_map.png')
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env',      choices=['pendulum', 'lunarlander', 'both'],
                        default='both')
    parser.add_argument('--seeds',    type=int,   default=None,
                        help='Override seeds (default from ENV_CFG)')
    parser.add_argument('--episodes', type=int,   default=None,
                        help='Override episodes (default from ENV_CFG)')
    parser.add_argument('--force',    action='store_true')
    args = parser.parse_args()

    for d in [CKPT_DIR, PLOT_DIR, RES_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    envs_to_run = ['pendulum', 'lunarlander'] if args.env == 'both' else [args.env]
    fig_offsets = {'pendulum': 3, 'lunarlander': 8}

    for env_key in envs_to_run:
        cfg = dict(ENV_CFG[env_key])
        hp  = dict(HYPERPARAMS)
        if args.seeds    is not None: cfg['seeds']    = args.seeds
        if args.episodes is not None: cfg['episodes'] = args.episodes

        print(f'\n{"="*60}')
        print(f'  {cfg["label"]}  ({cfg["seeds"]} seeds × {cfg["episodes"]} episodes)')
        print(f'{"="*60}')
        returns, entropy, td_abs, actor0, critic0, traj0 = run_sweep(
            cfg, hp, force=args.force
        )
        plot_ac(returns, entropy, td_abs, actor0, critic0, traj0,
                cfg, PLOT_DIR, fig_offset=fig_offsets[env_key])

    print('\nAll done.')


if __name__ == '__main__':
    main()
