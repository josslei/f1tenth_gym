from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DiscreteActionConfig:
    steering_bins: int
    velocity_bins: int
    velocity_min: float = 1.0
    velocity_max: float = 8.0


class DiscreteActionSpace:
    def __init__(self, config: DiscreteActionConfig) -> None:
        self.config = config
        steering = np.linspace(-1.0, 1.0, config.steering_bins, dtype=np.float32)
        velocity = np.linspace(-1.0, 1.0, config.velocity_bins, dtype=np.float32)
        self.normalized_actions = np.array(
            [[s, v] for s in steering for v in velocity], dtype=np.float32
        )

    @property
    def action_count(self) -> int:
        return int(self.normalized_actions.shape[0])

    def normalized(self, action_index: int) -> np.ndarray:
        return self.normalized_actions[action_index].copy()

    def normalized_batch(self, action_indices: np.ndarray) -> np.ndarray:
        return self.normalized_actions[action_indices.astype(np.int64)]


__all__ = ["DiscreteActionConfig", "DiscreteActionSpace"]
