"""Base policy interface for RL algorithms."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Tuple

import torch.nn as nn
from torch import Tensor


class Policy(nn.Module, ABC):
    """Abstract policy that ``models.ppo.LightningPPO`` can optimize.

    Subclasses must provide ``act()`` for rollout sampling and
    ``evaluate_actions()`` for the PPO update.
    """

    @abstractmethod
    def act(
        self, obs: Tensor, deterministic: bool = False
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Sample ``(action, log_prob, value)`` from the policy."""

    @abstractmethod
    def evaluate_actions(
        self, obs: Tensor, actions: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Return ``(log_prob, entropy, value)`` for the given actions."""


__all__ = ["Policy"]
