from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class DiscreteActionConfig:
    steering_bins: NDArray[np.float32]
    velocity_bins: NDArray[np.float32]


class DiscreteActionSpace:
    def __init__(self, config: DiscreteActionConfig) -> None:
        self.config = config
        self.normalized_actions = np.array(
            [[s, v] for s in config.steering_bins for v in config.velocity_bins],
            dtype=np.float32,
        )

    @property
    def action_count(self) -> int:
        return int(self.normalized_actions.shape[0])

    def normalized(self, action_index: int) -> np.ndarray:
        return self.normalized_actions[action_index].copy()

    def normalized_batch(self, action_indices: np.ndarray) -> np.ndarray:
        return self.normalized_actions[action_indices.astype(np.int64)]


__all__ = ["DiscreteActionConfig", "DiscreteActionSpace"]
