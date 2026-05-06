"""
models.py — Policy and value network definitions for MSDS 684 Week 6.

All networks are 2-hidden-layer MLPs. Architecture is held constant across
all experiments so comparisons are about algorithms, not network capacity.
"""
import torch
import torch.nn as nn
from torch.distributions import Categorical, Normal


class CategoricalPolicy(nn.Module):
    """MLP → action logits for discrete action spaces (CartPole).

    Outputs raw logits; Categorical(logits=...) handles softmax internally
    for numerical stability.
    """
    def __init__(self, obs_dim: int, n_actions: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),  nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)

    def dist(self, obs: torch.Tensor) -> Categorical:
        return Categorical(logits=self.forward(obs))


class GaussianPolicy(nn.Module):
    """MLP → μ(s) with state-independent learned log σ.

    Serves both Pendulum (act_dim=1) and LunarLanderContinuous (act_dim=2).
    State-independent log_std is the S&B §13.7 formulation — avoids the
    common failure mode where a state-conditioned std head collapses before
    the mean has learned anything.
    """
    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 64,
                 init_log_std: float = 0.0):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden),  nn.Tanh(),
        )
        self.mu_head  = nn.Linear(hidden, act_dim)
        self.log_std  = nn.Parameter(torch.full((act_dim,), float(init_log_std)))

    def forward(self, obs: torch.Tensor):
        h        = self.body(obs)
        mu       = self.mu_head(h)
        log_std  = self.log_std.clamp(-2.0, 1.0)   # prevent collapse / explosion
        return mu, log_std.exp().expand_as(mu)

    def dist(self, obs: torch.Tensor) -> Normal:
        mu, sigma = self.forward(obs)
        return Normal(mu, sigma)


class ValueNet(nn.Module):
    """State-value function V(s).

    Used as:
      • REINFORCE baseline — trained by MSE against Monte-Carlo returns G_t.
      • AC critic          — trained by MSE against bootstrap target r + γV(s').
    """
    def __init__(self, obs_dim: int, hidden: int = 128, activation: str = 'relu'):
        super().__init__()
        Act = nn.ReLU if activation == 'relu' else nn.Tanh
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), Act(),
            nn.Linear(hidden, hidden),  Act(),
            nn.Linear(hidden, 1),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs).squeeze(-1)
