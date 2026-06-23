from __future__ import annotations

from collections.abc import Callable
from typing import Any, cast

import gymnasium as gym
import numpy as np
import torch

from utils.f110_env import (
    F1TenthActionConfig,
    F1TenthObservationConfig,
    SubprocVecEnv,
    build_observation,
)

from .self_play import StepBatchResult


class GymF110Backend:
    def __init__(
        self,
        env_fns: list[Callable[[], gym.Env]],
        observation_config: F1TenthObservationConfig,
        action_config: F1TenthActionConfig,
        reward_fns: list[Callable[[dict[str, Any], bool], float]],
        reset_fn: Callable[[], dict[str, Any]],
        max_episode_steps: int,
    ) -> None:
        self.sve = SubprocVecEnv(env_fns, action_config, max_episode_steps)
        self.observation_config = observation_config
        self.reward_fns = reward_fns
        self.reset_fn = reset_fn
        self.current_obs: list[dict[str, Any] | None] = [None] * len(env_fns)
        self.prev_actions = np.zeros((len(env_fns), 2), dtype=np.float32)

    def _build_obs_batch(self, obs_list: list[dict[str, Any]]) -> torch.Tensor:
        for i, obs in enumerate(obs_list):
            obs["prev_action"] = self.prev_actions[i]
        obs_np = [build_observation(obs, self.observation_config) for obs in obs_list]
        return getattr(torch, "as_tensor")(
            np.asarray(obs_np), dtype=getattr(torch, "float32")
        )

    def reset_batch(self) -> torch.Tensor:
        if self.current_obs[0] is None:
            self.current_obs = cast(
                list[dict[str, Any] | None],
                self.sve.reset_all([self.reset_fn] * self.sve.n_envs),
            )
        self.prev_actions.fill(0.0)
        return self._build_obs_batch(cast(list[dict[str, Any]], self.current_obs))

    def step_batch(self, normalized_actions: torch.Tensor) -> StepBatchResult:
        actions_np = normalized_actions.detach().cpu().numpy()
        next_obs, terminated, truncated, reset_obs = self.sve.step(actions_np)

        reset_obs_filled = cast(
            list[dict[str, Any]],
            [
                reset_obs[i] if reset_obs[i] is not None else next_obs[i]
                for i in range(self.sve.n_envs)
            ],
        )

        reward = np.zeros((self.sve.n_envs,), dtype=np.float32)
        lap_count = np.zeros((self.sve.n_envs,), dtype=np.int32)
        lap_time = np.zeros((self.sve.n_envs,), dtype=np.float32)
        collision = np.zeros((self.sve.n_envs,), dtype=np.uint8)
        for i in range(self.sve.n_envs):
            done = bool(terminated[i] or truncated[i])
            reward[i] = self.reward_fns[i](next_obs[i], done)
            ego = int(next_obs[i]["ego_idx"])
            lap_count[i] = int(next_obs[i]["lap_counts"][ego])
            lap_time[i] = float(next_obs[i]["lap_times"][ego])
            collision[i] = 1 if next_obs[i]["collisions"][ego] else 0

        self.current_obs = cast(
            list[dict[str, Any] | None],
            [
                reset_obs_filled[i] if (terminated[i] or truncated[i]) else next_obs[i]
                for i in range(self.sve.n_envs)
            ],
        )
        self.prev_actions = actions_np.astype(np.float32)

        as_tensor = getattr(torch, "as_tensor")
        return StepBatchResult(
            obs=self._build_obs_batch(next_obs),
            reward=as_tensor(reward, dtype=getattr(torch, "float32")),
            terminated=as_tensor(terminated, dtype=getattr(torch, "uint8")),
            truncated=as_tensor(truncated, dtype=getattr(torch, "uint8")),
            lap_count=as_tensor(lap_count, dtype=getattr(torch, "int32")),
            lap_time=as_tensor(lap_time, dtype=getattr(torch, "float32")),
            collision=as_tensor(collision, dtype=getattr(torch, "uint8")),
            reset_obs=self._build_obs_batch(reset_obs_filled),
        )

    def close(self) -> None:
        self.sve.close()


__all__ = ["GymF110Backend"]
