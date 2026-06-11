"""Configuration object for PPO algorithm training."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Self

import yaml


@dataclass(frozen=True)
class PPOTrainingConfig:
    """PPO training hyperparameters from the ``training`` config section."""

    k_epochs: int
    mini_batch_size: int
    learning_rate: float
    epsilon: float
    c1: float
    c2: float
    normalize_advantages: bool
    max_grad_norm: float


@dataclass(frozen=True)
class PolicyConfig:
    name: str
    kwargs: dict[str, Any]


@dataclass(frozen=True)
class RuntimeConfig:
    seed: int
    ppo_iterations: int
    rollout_steps: int
    progress_bar: bool
    num_envs: int


@dataclass(frozen=True)
class PPOConfig:
    """Structured PPO config loaded from ``configs/ppo/*.yaml``."""

    env: dict[str, Any]
    observation: dict[str, Any]
    action: dict[str, Any]
    reward: dict[str, Any]
    tracks: dict[str, Any]
    policy: PolicyConfig
    training: PPOTrainingConfig
    runtime: RuntimeConfig
    output: dict[str, Any]

    @classmethod
    def from_yaml(cls, path: str | Path) -> Self:
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        return cls(
            env=data["env"],
            observation=data["observation"],
            action=data["action"],
            reward=data["reward"],
            tracks=data.get("tracks", {}),
            policy=PolicyConfig(**data["policy"]),
            training=PPOTrainingConfig(**data["training"]),
            runtime=RuntimeConfig(**data["runtime"]),
            output=data["output"],
        )

    def with_policy_kwargs(self, **kwargs: Any) -> PPOConfig:
        return replace(
            self,
            policy=replace(self.policy, kwargs={**self.policy.kwargs, **kwargs}),
        )


def load_ppo_config(path: str | Path = "configs/ppo/default.yaml") -> PPOConfig:
    """Load a :class:`PPOConfig` from a YAML file."""
    return PPOConfig.from_yaml(path)


__all__ = [
    "PPOConfig",
    "PolicyConfig",
    "PPOTrainingConfig",
    "RuntimeConfig",
    "load_ppo_config",
]
