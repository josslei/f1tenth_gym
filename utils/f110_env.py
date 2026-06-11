"""F1TENTH environment helpers for RL orchestration scripts.

Observation / action conversion, config dataclasses, and replay utilities
that translate between the Gym ``f110-v0`` interface and tensor-based RL
algorithms.  The ``rollout`` collector is parameterized with callbacks so
it can be reused across different tasks (controller, planner, …) — only
the callbacks need to know about the domain.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import math
from typing import Any

import gymnasium as gym
import lightning as pl
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, IterableDataset

from models.policies import Policy
from models.ppo import compute_gae


DEFAULT_MAP = "Spielberg"
DEFAULT_POSE = np.array([[0.0, 0.0, 0.0]], dtype=np.float64)
DEFAULT_DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)


def initial_pose() -> np.ndarray:
    return DEFAULT_POSE.copy()


# ── Config dataclasses ────────────────────────────────────────────────────────


@dataclass(frozen=True)
class F1TenthObservationConfig:
    scan_size: int = 108
    scan_max_m: float = 30.0
    include_ego_state: bool = True
    speed_scale: float = 8.0
    yaw_rate_scale: float = 10.0


@dataclass(frozen=True)
class F1TenthActionConfig:
    velocity_min: float = 0.0
    velocity_max: float = 8.0


# ── Observation helpers ───────────────────────────────────────────────────────


def observation_dim(config: F1TenthObservationConfig) -> int:
    ego_state_dim = 6 if config.include_ego_state else 0
    return config.scan_size + ego_state_dim


def build_observation(
    obs: dict[str, Any], config: F1TenthObservationConfig
) -> np.ndarray:
    ego = int(obs["ego_idx"])
    scans = np.asarray(obs["scans"], dtype=np.float64)
    ego_scan = scans[ego] if scans.ndim > 1 else scans
    scan_indices = np.linspace(0, ego_scan.shape[0] - 1, config.scan_size).astype(
        np.int64
    )
    scan = np.nan_to_num(
        ego_scan[scan_indices],
        nan=config.scan_max_m,
        posinf=config.scan_max_m,
        neginf=0.0,
    )
    scan = np.clip(scan, 0.0, config.scan_max_m) / config.scan_max_m
    features = [scan]

    if config.include_ego_state:
        linear_vels_x = np.asarray(obs["linear_vels_x"], dtype=np.float64)[ego]
        linear_vels_y = np.asarray(obs["linear_vels_y"], dtype=np.float64)[ego]
        ang_vels_z = np.asarray(obs["ang_vels_z"], dtype=np.float64)[ego]
        theta = np.asarray(obs["poses_theta"], dtype=np.float64)[ego]
        collision = np.asarray(obs["collisions"], dtype=np.float64)[ego]
        features.append(
            np.array(
                [
                    np.clip(linear_vels_x / config.speed_scale, -1.0, 1.0),
                    np.clip(linear_vels_y / config.speed_scale, -1.0, 1.0),
                    np.clip(ang_vels_z / config.yaw_rate_scale, -1.0, 1.0),
                    np.sin(theta),
                    np.cos(theta),
                    np.clip(collision, 0.0, 1.0),
                ],
                dtype=np.float64,
            )
        )

    return np.nan_to_num(
        np.concatenate(features).astype(np.float32),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )


def obs_tensor(obs: dict[str, Any], config: F1TenthObservationConfig) -> Tensor:
    observation = build_observation(obs, config)
    return torch.as_tensor(observation, dtype=torch.float32).unsqueeze(0)


# ── Action helpers ────────────────────────────────────────────────────────────


def scale_action(
    action: np.ndarray,
    env: gym.Env,
    config: F1TenthActionConfig,
) -> np.ndarray:
    normalized_action = np.clip(
        np.asarray(action, dtype=np.float64).reshape(2), -1.0, 1.0
    )
    f110_env: Any = env.unwrapped
    params = f110_env.params
    steering = np.interp(
        normalized_action[0], [-1.0, 1.0], [params["s_min"], params["s_max"]]
    )
    velocity = np.interp(
        normalized_action[1],
        [-1.0, 1.0],
        [config.velocity_min, config.velocity_max],
    )
    return np.array(
        [
            [
                float(np.clip(steering, params["s_min"], params["s_max"])),
                float(np.clip(velocity, params["v_min"], params["v_max"])),
            ]
        ],
        dtype=np.float64,
    )


# ── PPO rollout ────────────────────────────────────────────────────────────────


def rollout(
    env: gym.Env,
    policy: Policy,
    rollout_steps: int,
    *,
    obs_fn: Callable[[dict[str, Any]], Tensor],
    action_fn: Callable[[Tensor, gym.Env], np.ndarray],
    reward_fn: Callable[[dict[str, Any], bool], float],
    reset_fn: Callable[[gym.Env], dict[str, Any]],
    device: torch.device = DEFAULT_DEVICE,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    obs, _info = env.reset(options=reset_fn(env))

    s_values: list[Tensor] = []
    a_values: list[Tensor] = []
    log_p_cur_values: list[Tensor] = []
    V_phi_values: list[Tensor] = []
    r_values: list[float] = []
    terminal_values: list[float] = []
    episode_return = 0.0

    for _t in range(rollout_steps):
        s_t = obs_fn(obs).to(device)
        with torch.no_grad():
            a_t, log_p_cur_t, V_phi_t = policy.act(s_t)

        env_action = action_fn(a_t, env)
        next_obs, _env_reward, terminated, truncated, _info = env.step(env_action)
        terminal = bool(terminated or truncated)
        r_t = reward_fn(next_obs, terminal)

        s_values.append(s_t.squeeze(0))
        a_values.append(a_t.squeeze(0).detach())
        log_p_cur_values.append(log_p_cur_t.squeeze(0).detach())
        V_phi_values.append(V_phi_t.squeeze(0).detach())
        r_values.append(r_t)
        terminal_values.append(float(terminal))
        episode_return += r_t

        obs = next_obs
        if terminal:
            obs, _info = env.reset(options=reset_fn(env))

    final_s = obs_fn(obs).to(device)
    with torch.no_grad():
        _action, _log_p, final_V_phi = policy.act(final_s, deterministic=True)
    V_phi_values.append(final_V_phi.squeeze(0).detach())

    s = torch.stack(s_values).unsqueeze(0)
    a = torch.stack(a_values).unsqueeze(0)
    log_p_cur = torch.stack(log_p_cur_values).unsqueeze(0)
    r = torch.as_tensor(r_values, dtype=torch.float32, device=device).unsqueeze(0)
    terminal = torch.as_tensor(
        terminal_values, dtype=torch.float32, device=device
    ).unsqueeze(0)
    V_phi = torch.stack(V_phi_values).unsqueeze(0)

    A_hat, R_hat = compute_gae(r=r, V_phi=V_phi, terminal=terminal)
    return (
        s.reshape(-1, s.shape[-1]),
        a.reshape(-1, a.shape[-1]),
        log_p_cur.reshape(-1),
        A_hat.reshape(-1),
        R_hat.reshape(-1),
        torch.as_tensor(episode_return, dtype=torch.float32, device=device),
    )


class RolloutDataset(
    IterableDataset[tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]]
):
    def __init__(
        self,
        envs: list[gym.Env],
        policy: Policy,
        rollout_steps: int,
        obs_fn: Callable[[dict[str, Any]], Tensor],
        action_fn: Callable[[Tensor, gym.Env], np.ndarray],
        reward_fn: Callable[[dict[str, Any], bool], float],
        reset_fn: Callable[[gym.Env], dict[str, Any]],
        episode_returns: list[float],
        k_epochs: int,
        mini_batch_size: int,
        normalize_advantages: bool,
        device: torch.device = DEFAULT_DEVICE,
        track_scheduler: Any | None = None,
    ) -> None:
        self.envs = envs
        self.policy = policy
        self.rollout_steps = rollout_steps
        self.obs_fn = obs_fn
        self.action_fn = action_fn
        self.reward_fn = reward_fn
        self.reset_fn = reset_fn
        self.episode_returns = episode_returns
        self.k_epochs = k_epochs
        self.mini_batch_size = mini_batch_size
        self.normalize_advantages = normalize_advantages
        self.device = device
        self.track_scheduler = track_scheduler

    def __iter__(
        self,
    ) -> Iterator[tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]]:
        if self.track_scheduler is not None:
            self.track_scheduler.step_iteration()

        all_s: list[Tensor] = []
        all_a: list[Tensor] = []
        all_log_p_cur: list[Tensor] = []
        all_A_hat: list[Tensor] = []
        all_R_hat: list[Tensor] = []
        total_episode_return = 0.0

        with ThreadPoolExecutor(max_workers=len(self.envs)) as executor:
            futures = [
                executor.submit(
                    rollout,
                    env=env,
                    policy=self.policy,
                    rollout_steps=self.rollout_steps,
                    obs_fn=self.obs_fn,
                    action_fn=self.action_fn,
                    reward_fn=self.reward_fn,
                    reset_fn=self.reset_fn,
                    device=self.device,
                )
                for env in self.envs
            ]
            for future in futures:
                s, a, log_p_cur, A_hat, R_hat, episode_return = future.result()
                all_s.append(s)
                all_a.append(a)
                all_log_p_cur.append(log_p_cur)
                all_A_hat.append(A_hat)
                all_R_hat.append(R_hat)
                total_episode_return += episode_return.item()

        avg_episode_return = total_episode_return / len(self.envs)
        self.episode_returns.append(avg_episode_return)

        s = torch.cat(all_s)
        a = torch.cat(all_a)
        log_p_cur = torch.cat(all_log_p_cur)
        A_hat = torch.cat(all_A_hat)
        R_hat = torch.cat(all_R_hat)

        if self.normalize_advantages:
            A_hat = (A_hat - A_hat.mean()) / (A_hat.std() + 1e-8)

        n = s.shape[0]
        for _k_epoch in range(self.k_epochs):
            index_set_I = torch.randperm(n, device=s.device)
            for start in range(0, n, self.mini_batch_size):
                B = index_set_I[start : start + self.mini_batch_size]
                yield (
                    s[B],
                    a[B],
                    log_p_cur[B],
                    A_hat[B],
                    R_hat[B],
                    torch.as_tensor(
                        avg_episode_return, dtype=torch.float32, device=s.device
                    ),
                )

    def __len__(self) -> int:
        total_samples = len(self.envs) * self.rollout_steps
        return self.k_epochs * math.ceil(total_samples / self.mini_batch_size)


class RolloutDataModule(pl.LightningDataModule):
    def __init__(self, dataset: RolloutDataset) -> None:
        super().__init__()
        self.dataset = dataset

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.dataset, batch_size=None, num_workers=0)


__all__ = [
    "DEFAULT_MAP",
    "DEFAULT_POSE",
    "F1TenthActionConfig",
    "F1TenthObservationConfig",
    "RolloutDataModule",
    "RolloutDataset",
    "build_observation",
    "initial_pose",
    "observation_dim",
    "obs_tensor",
    "rollout",
    "scale_action",
]
