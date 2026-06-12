from __future__ import annotations

from typing import Any, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

from .base import Policy


class GaussianMLPPolicy(Policy):
    def __init__(
        self,
        obs_dim: int,
        action_dim: int = 2,
        hidden_size: int = 64,
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
        self.log_std = nn.Parameter(torch.full((action_dim,), float(np.log(0.5))))
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

    def _tanh_correction(self, action: Tensor) -> Tensor:
        return torch.log(1.0 - action.pow(2) + 1e-6).sum(dim=-1)

    def act(
        self, obs: Tensor, deterministic: bool = False
    ) -> Tuple[Tensor, Tensor, Tensor]:
        dist, value = self._distribution(obs)
        if deterministic:
            action = dist.mean
            log_prob = dist.log_prob(action).sum(dim=-1)
            return action, log_prob, value

        raw_action = dist.rsample()
        action = torch.tanh(raw_action)
        log_prob = dist.log_prob(raw_action).sum(dim=-1) - self._tanh_correction(action)
        return action, log_prob, value

    def evaluate_actions(
        self, obs: Tensor, actions: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor]:
        dist, value = self._distribution(obs)
        raw_actions = torch.atanh(torch.clamp(actions, -1.0 + 1e-6, 1.0 - 1e-6))
        log_prob = dist.log_prob(raw_actions).sum(dim=-1) - self._tanh_correction(
            actions
        )
        entropy = dist.entropy().sum(dim=-1)
        return log_prob, entropy, value


__all__ = ["GaussianMLPPolicy"]
