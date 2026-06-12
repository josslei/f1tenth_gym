"""Reusable policy network modules."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .base import Policy
from .gaussian_actor_critic import GaussianActorCritic
from .gaussian_mlp import GaussianMLPPolicy

if TYPE_CHECKING:
    from models.ppo.config import PolicyConfig


POLICY_REGISTRY: dict[str, type[Policy]] = {
    "GaussianActorCritic": GaussianActorCritic,
    "GaussianMLPPolicy": GaussianMLPPolicy,
}


def make_policy(config: PolicyConfig, obs_dim: int, action_dim: int) -> Policy:
    policy_type = POLICY_REGISTRY[config.name]
    return policy_type(obs_dim=obs_dim, action_dim=action_dim, **config.kwargs)


__all__ = [
    "GaussianActorCritic",
    "GaussianMLPPolicy",
    "POLICY_REGISTRY",
    "Policy",
    "make_policy",
]
