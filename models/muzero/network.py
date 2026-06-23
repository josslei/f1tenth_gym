from __future__ import annotations

# pyright: reportAttributeAccessIssue=none

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class F110MuZeroNet(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        action_count: int,
        hidden_size: int,
        trunk_size: int = 256,
    ) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.action_count = action_count
        self.hidden_size = hidden_size

        self.representation = nn.Sequential(
            nn.Linear(obs_dim, trunk_size),
            nn.ReLU(),
            nn.Linear(trunk_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.Tanh(),
        )
        self.action_embedding = nn.Embedding(action_count, hidden_size)
        self.dynamics = nn.Sequential(
            nn.Linear(hidden_size * 2, trunk_size),
            nn.ReLU(),
            nn.Linear(trunk_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.Tanh(),
        )
        self.reward_head = nn.Sequential(
            nn.Linear(hidden_size, trunk_size), nn.ReLU(), nn.Linear(trunk_size, 1)
        )
        self.value_head = nn.Sequential(
            nn.Linear(hidden_size, trunk_size), nn.ReLU(), nn.Linear(trunk_size, 1)
        )
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_size, trunk_size),
            nn.ReLU(),
            nn.Linear(trunk_size, action_count),
        )
        self.discount_head = nn.Sequential(
            nn.Linear(hidden_size, trunk_size), nn.ReLU(), nn.Linear(trunk_size, 1)
        )

    def predict_logits(self, hidden: Tensor) -> tuple[Tensor, Tensor]:
        return self.policy_head(hidden), self.value_head(hidden).squeeze(-1)

    def initial_training(self, obs: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        hidden = self.representation(obs)
        policy_logits, value = self.predict_logits(hidden)
        return hidden, policy_logits, value

    def recurrent_training(
        self, hidden: Tensor, action: Tensor
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        action_embed = self.action_embedding(action.long())
        next_hidden = self.dynamics(torch.cat([hidden, action_embed], dim=-1))  # type: ignore[attr-defined]
        reward = self.reward_head(next_hidden).squeeze(-1)
        value = self.value_head(next_hidden).squeeze(-1)
        discount = torch.sigmoid(self.discount_head(next_hidden).squeeze(-1))  # type: ignore[attr-defined]
        policy_logits = self.policy_head(next_hidden)
        return next_hidden, reward, value, discount, policy_logits

    @torch.jit.export
    def initial_inference(self, obs: Tensor) -> tuple[Tensor, Tensor]:
        hidden = self.representation(obs)
        policy_logits, value = self.predict_logits(hidden)
        policy = F.softmax(policy_logits, dim=-1)
        payload = torch.cat([value.unsqueeze(-1), policy], dim=-1)  # type: ignore[attr-defined]
        return hidden, payload

    @torch.jit.export
    def recurrent_inference(
        self, hidden: Tensor, action: Tensor
    ) -> tuple[Tensor, Tensor]:
        next_hidden, reward, value, discount, policy_logits = self.recurrent_training(
            hidden, action
        )
        policy = F.softmax(policy_logits, dim=-1)
        payload = torch.cat(  # type: ignore[attr-defined]
            [
                reward.unsqueeze(-1),
                value.unsqueeze(-1),
                discount.unsqueeze(-1),
                policy,
            ],
            dim=-1,
        )
        return next_hidden, payload


__all__ = ["F110MuZeroNet"]
