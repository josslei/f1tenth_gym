"""Generic continuous-action Gaussian actor-critic policy."""

from __future__ import annotations

from typing import Any, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from .base import Policy


class GaussianActorCritic(Policy):
    """Continuous-action policy with state-value head.

    Common names: ``obs`` = state/observation, ``action`` = sampled action,
    ``log_prob`` = log policy density, ``value`` = state-value estimate.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int = 2,
        hidden_size: int = 128,
    ) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim

        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
        )
        self.action_mean = nn.Linear(hidden_size, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))
        self.value_head = nn.Linear(hidden_size, 1)

    def forward(self, obs: Tensor) -> Tuple[Tensor, Tensor]:
        features = self.net(obs)
        mean = torch.tanh(self.action_mean(features))
        value = self.value_head(features).squeeze(-1)
        return mean, value

    def _distribution(self, obs: Tensor) -> Tuple[Any, Tensor]:
        mean, value = self(obs)
        std = torch.exp(self.log_std).expand_as(mean)
        dist = torch.distributions.Normal(mean, std)
        return dist, value

    def act(
        self, obs: Tensor, deterministic: bool = False
    ) -> Tuple[Tensor, Tensor, Tensor]:
        dist, value = self._distribution(obs)
        raw_action = dist.mean if deterministic else dist.rsample()
        log_prob = dist.log_prob(raw_action).sum(dim=-1)
        return raw_action, log_prob, value

    def evaluate_actions(
        self, obs: Tensor, actions: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor]:
        dist, value = self._distribution(obs)
        log_prob = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return log_prob, entropy, value


__all__ = ["GaussianActorCritic"]
