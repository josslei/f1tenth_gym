"""Residual MLP policy with separate LiDAR encoder — for 22-map F1TENTH."""

from __future__ import annotations

from typing import Any, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from .base import Policy


class ResidualMLPPolicy(Policy):
    """Residual MLP actor-critic with LiDAR encoder, state fusion, and LayerNorm.

    Designed for multi-track generalization: the LiDAR encoder learns scan
    features independently of the ego-state path, and the residual trunk
    keeps parameters moderate (~350k at default width).

    Architecture
    ------------
    LiDAR encoder:
        ``Linear(N → W) → LayerNorm → ReLU → Linear(W → W) → LayerNorm → ReLU``

    Fusion:
        ``Concat(lidar_features, state) → Linear(W+S → W) → LayerNorm → ReLU``

    Residual blocks (N blocks):
        ``Linear(W → W) → ReLU → Linear(W → W) → +skip → LayerNorm → ReLU``

    Policy head:
        ``Linear(W → H) → ReLU → Linear(H → A) → tanh``

    Value head:
        ``Linear(W → H) → ReLU → Linear(H → 1)``

    Parameters
    ----------
    obs_dim:
        Total flattened observation dimension (kept for interface compatibility).
    action_dim:
        Number of continuous action dimensions (default 2: steering + speed).
    scan_size:
        Number of LiDAR beams after downsampling (default 108).
    state_dim:
        Dimension of the ego-state vector concatenated after the scan
        (default 6: ``[v_x, v_y, omega, sin(theta), cos(theta), collision]``).
    width:
        Hidden width for LiDAR encoder and trunk (default 192).
    head_width:
        Hidden width for policy/value heads (default 96).
    num_residual_blocks:
        Number of residual blocks after fusion (default 1).
    activation:
        Activation function — ``"relu"`` or ``"silu"`` (default ``"relu"``).
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int = 2,
        scan_size: int = 108,
        state_dim: int = 6,
        width: int = 192,
        head_width: int = 96,
        num_residual_blocks: int = 1,
        activation: str = "relu",
    ) -> None:
        super().__init__()
        self.scan_size = scan_size
        self.state_dim = state_dim
        self.action_dim = action_dim

        act_fn = nn.ReLU if activation == "relu" else nn.SiLU

        # ── LiDAR encoder ─────────────────────────────────────────────────
        self.lidar_encoder = nn.Sequential(
            nn.Linear(scan_size, width),
            nn.LayerNorm(width),
            act_fn(),
            nn.Linear(width, width),
            nn.LayerNorm(width),
            act_fn(),
        )

        # ── Fusion trunk ────────────────────────────────────────────────────
        self.trunk = nn.Sequential(
            nn.Linear(width + state_dim, width),
            nn.LayerNorm(width),
            act_fn(),
        )

        # ── Residual blocks (skip connections in forward pass) ──────────────
        self.residual_blocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(width, width),
                    act_fn(),
                    nn.Linear(width, width),
                )
                for _ in range(num_residual_blocks)
            ]
        )
        self.residual_norm = nn.LayerNorm(width)
        self.residual_act = act_fn()

        # ── Policy head ─────────────────────────────────────────────────────
        self.action_mean = nn.Sequential(
            nn.Linear(width, head_width),
            act_fn(),
            nn.Linear(head_width, action_dim),
        )
        self.log_std = nn.Parameter(torch.zeros(action_dim))

        # ── Value head ──────────────────────────────────────────────────────
        self.value_head = nn.Sequential(
            nn.Linear(width, head_width),
            act_fn(),
            nn.Linear(head_width, 1),
        )

    def forward(self, obs: Tensor) -> Tuple[Tensor, Tensor]:
        scan = obs[..., : self.scan_size]
        state = obs[..., self.scan_size :]

        lidar_features = self.lidar_encoder(scan)
        x = torch.cat([lidar_features, state], dim=-1)
        x = self.trunk(x)

        for block in self.residual_blocks:
            residual = x
            x = block(x)
            x = self.residual_norm(x + residual)
            x = self.residual_act(x)

        mean = torch.tanh(self.action_mean(x))
        value = self.value_head(x).squeeze(-1)
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


__all__ = ["ResidualMLPPolicy"]
