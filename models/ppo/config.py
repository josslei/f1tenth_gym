"""Configuration object for PPO algorithm training."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from math import ceil
from pathlib import Path
from typing import Any, Self

import numpy as np
import yaml


@dataclass(frozen=True)
class MapConfig:
    """One candidate map for multi-map training.

    Each map is a fully independent environment with its own occupancy
    grid, map image, and centerline.  ``map`` and ``map_ext`` are passed
    to ``gym.make(..., map=, map_ext=)`` (via ``F110Env.update_map``
    at epoch boundaries); ``centerline_csv`` supplies the forward-progress
    waypoints and the reset pose.
    """

    name: str
    map: str
    map_ext: str = ".png"
    centerline_csv: str = ""


@dataclass(frozen=True)
class MapSplitConfig:
    """Deterministic 90/10 train/validation map split."""

    candidates: list[MapConfig]
    exclude: list[str] = field(default_factory=list)
    validation_ratio: float = 0.1
    split_seed: int | None = None


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
    checkpoint_every_n_epochs: int = 50


@dataclass(frozen=True)
class ValidationConfig:
    map: str
    map_ext: str = ".png"
    episodes: int = 2
    centerline_csv: str = ""


@dataclass(frozen=True)
class PPOConfig:
    """Structured PPO config loaded from ``configs/ppo/*.yaml``."""

    env: dict[str, Any]
    observation: dict[str, Any]
    action: dict[str, Any]
    reward: dict[str, Any]
    policy: PolicyConfig
    training: PPOTrainingConfig
    runtime: RuntimeConfig
    output: dict[str, Any]
    validation: ValidationConfig | None = None
    maps: MapSplitConfig | None = None

    @classmethod
    def from_yaml(cls, path: str | Path) -> Self:
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        maps_data = data.get("maps")
        maps: MapSplitConfig | None = None
        if maps_data is not None:
            maps = MapSplitConfig(
                candidates=[MapConfig(**c) for c in maps_data["candidates"]],
                exclude=maps_data.get("exclude", []),
                validation_ratio=maps_data.get("validation_ratio", 0.1),
                split_seed=maps_data.get("split_seed"),
            )
        return cls(
            env=data["env"],
            observation=data["observation"],
            action=data["action"],
            reward=data["reward"],
            policy=PolicyConfig(**data["policy"]),
            training=PPOTrainingConfig(**data["training"]),
            runtime=RuntimeConfig(**data["runtime"]),
            output=data["output"],
            validation=ValidationConfig(**data["validation"])
            if "validation" in data
            else None,
            maps=maps,
        )

    def with_policy_kwargs(self, **kwargs: Any) -> PPOConfig:
        return replace(
            self,
            policy=replace(self.policy, kwargs={**self.policy.kwargs, **kwargs}),
        )


def split_maps(
    candidates: Sequence[MapConfig],
    seed: int,
    validation_ratio: float = 0.1,
    exclude: Sequence[str] = (),
) -> tuple[list[MapConfig], list[MapConfig]]:
    """Deterministic 90/10 train/validation split over *candidates*.

    Maps whose name, map, or centerline_csv contain any string in
    *exclude* (case-insensitive substring match) are filtered out
    *before* the split.  The split itself is seeded so the same seed
    always produces the same partition.

    Returns
    -------
    (train_maps, val_maps)
        Disjoint partitions of the filtered candidate list.
    """
    exclusions = frozenset(s.lower() for s in exclude)

    def _excluded(m: MapConfig) -> bool:
        return any(
            e in m.name.lower() or e in m.map.lower() or e in m.centerline_csv.lower()
            for e in exclusions
        )

    eligible = sorted(
        [m for m in candidates if not _excluded(m)],
        key=lambda m: m.name,
    )
    rng = np.random.default_rng(seed)
    permutation = rng.permutation(len(eligible))
    val_count = max(1, ceil(len(eligible) * validation_ratio))
    val_idx = set(permutation[:val_count].tolist())
    val: list[MapConfig] = []
    train: list[MapConfig] = []
    for i, m in enumerate(eligible):
        (val if i in val_idx else train).append(m)
    return train, val


def build_epoch_map_schedule(
    train_maps: Sequence[MapConfig],
    seed: int,
    epochs: int,
) -> list[MapConfig]:
    """Deterministic round-robin schedule over *train_maps*.

    Maps are randomly permuted using *seed*, then cycled repeatedly
    until *epochs* entries are produced.  The result length always
    equals *epochs* so that ``schedule[epoch]`` is defined for every
    PPO iteration.
    """
    rng = np.random.default_rng(seed)
    perm = sorted(train_maps, key=lambda m: m.name)
    rng.shuffle(perm)
    schedule: list[MapConfig] = []
    while len(schedule) < epochs:
        schedule.extend(perm)
    return schedule[:epochs]


def load_ppo_config(path: str | Path = "configs/ppo/default.yaml") -> PPOConfig:
    """Load a :class:`PPOConfig` from a YAML file."""
    return PPOConfig.from_yaml(path)


__all__ = [
    "PPOConfig",
    "PolicyConfig",
    "PPOTrainingConfig",
    "RuntimeConfig",
    "ValidationConfig",
    "load_ppo_config",
]
