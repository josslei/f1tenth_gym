from __future__ import annotations

# pyright: reportAttributeAccessIssue=none

from collections import deque
from dataclasses import dataclass
import random
from typing import Any

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset


@dataclass(frozen=True)
class MuZeroTransition:
    obs: np.ndarray
    action: int
    reward: float
    done: bool
    root_policy: np.ndarray
    root_value: float


class MuZeroReplayBuffer(
    Dataset[tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]]
):
    def __init__(
        self,
        max_size: int,
        unroll_steps: int,
        td_steps: int,
        discount: float,
    ) -> None:
        self.trajectories: deque[list[MuZeroTransition]] = deque(maxlen=max_size)
        self.unroll_steps = unroll_steps
        self.td_steps = td_steps
        self.discount = discount

    def push(self, trajectory: Any) -> None:
        transitions = getattr(trajectory, "transitions", trajectory)
        converted: list[MuZeroTransition] = []
        for transition in transitions:
            obs = transition.obs
            root_policy = transition.root_policy
            if hasattr(obs, "detach"):
                obs = obs.detach().cpu().numpy()
            if hasattr(root_policy, "detach"):
                root_policy = root_policy.detach().cpu().numpy()
            converted.append(
                MuZeroTransition(
                    obs=np.asarray(obs, dtype=np.float32),
                    action=int(transition.action),
                    reward=float(transition.reward),
                    done=bool(transition.done),
                    root_policy=np.asarray(root_policy, dtype=np.float32),
                    root_value=float(getattr(transition, "root_value", 0.0)),
                )
            )
        if converted:
            self.trajectories.append(converted)

    def __len__(self) -> int:
        return sum(len(t) for t in self.trajectories)

    def __getitem__(
        self, _: int
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        trajectory = random.choice(tuple(self.trajectories))
        start = random.randrange(len(trajectory))
        obs = torch.as_tensor(trajectory[start].obs, dtype=torch.float32)  # type: ignore[attr-defined]

        actions: list[int] = []
        target_rewards: list[float] = []
        target_values: list[float] = []
        target_discounts: list[float] = []
        target_policies: list[np.ndarray] = []

        for offset in range(self.unroll_steps + 1):
            idx = start + offset
            if idx < len(trajectory):
                transition = trajectory[idx]
                target_policies.append(transition.root_policy.astype(np.float32))
                target_values.append(self._n_step_return(trajectory, idx))
                target_discounts.append(0.0 if transition.done else self.discount)

                if offset < self.unroll_steps:
                    actions.append(transition.action)
                    target_rewards.append(transition.reward)
            else:
                target_policies.append(
                    np.zeros_like(trajectory[0].root_policy, dtype=np.float32)
                )
                target_values.append(0.0)
                target_discounts.append(0.0)

                if offset < self.unroll_steps:
                    actions.append(0)
                    target_rewards.append(0.0)

        return (
            obs,
            torch.as_tensor(actions, dtype=torch.long),  # type: ignore[attr-defined]
            torch.as_tensor(target_rewards, dtype=torch.float32),  # type: ignore[attr-defined]
            torch.as_tensor(np.asarray(target_values), dtype=torch.float32),  # type: ignore[attr-defined]
            torch.as_tensor(np.asarray(target_discounts), dtype=torch.float32),  # type: ignore[attr-defined]
            torch.as_tensor(np.asarray(target_policies), dtype=torch.float32),  # type: ignore[attr-defined]
        )

    def _n_step_return(self, trajectory: list[MuZeroTransition], start: int) -> float:
        value = 0.0
        discount = 1.0
        end = min(start + self.td_steps, len(trajectory))
        for idx in range(start, end):
            value += discount * trajectory[idx].reward
            if trajectory[idx].done:
                break
            discount *= self.discount
        else:
            if end < len(trajectory):
                value += discount * trajectory[end].root_value
        return float(value)


__all__ = ["MuZeroReplayBuffer", "MuZeroTransition"]
