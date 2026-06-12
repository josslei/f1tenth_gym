from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass
import math
import multiprocessing as mp
from multiprocessing.connection import Connection
from typing import Any, cast

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


# ── SubprocVecEnv – multi-process environment runner ──────────────────────────


def _env_worker(
    env_fn: Callable[[], gym.Env],
    pipe: Connection,
    action_config: F1TenthActionConfig,
) -> None:
    env = env_fn()
    f110_env: Any = env.unwrapped
    params = f110_env.params
    current_reset_options: dict[str, Any] | None = None
    while True:
        try:
            cmd, data = pipe.recv()
        except (EOFError, BrokenPipeError):
            break

        if cmd == "reset":
            current_reset_options = data
            obs_dict, info = (
                env.reset(options=data) if data is not None else env.reset()
            )
            pipe.send(obs_dict)
        elif cmd == "step":
            norm = np.asarray(data, dtype=np.float64).reshape(2)
            steering = np.interp(
                norm[0], [-1.0, 1.0], [params["s_min"], params["s_max"]]
            )
            velocity = np.interp(
                norm[1],
                [-1.0, 1.0],
                [action_config.velocity_min, action_config.velocity_max],
            )
            env_action = np.array(
                [
                    [
                        float(np.clip(steering, params["s_min"], params["s_max"])),
                        float(np.clip(velocity, params["v_min"], params["v_max"])),
                    ]
                ],
                dtype=np.float64,
            )
            obs_dict, _, terminated, truncated, info = env.step(env_action)
            terminal = bool(terminated or truncated)
            if terminal:
                reset_obs, _ = env.reset(options=current_reset_options)
                pipe.send((obs_dict, terminal, reset_obs))
            else:
                pipe.send((obs_dict, terminal, None))
        elif cmd == "close":
            break

    env.close()


class SubprocVecEnv:
    """Multi-process environment runner for lockstep parallel rollout.

    Mirrors Stable-Baselines3 ``SubprocVecEnv``: child processes step envs
    in true parallelism (separate GILs), main process handles policy inference
    and reward computation.
    """

    def __init__(
        self,
        env_fns: list[Callable[[], gym.Env]],
        action_config: F1TenthActionConfig,
    ) -> None:
        self.n_envs = len(env_fns)
        self.pipes: list[Connection] = []
        self.processes: list[mp.Process] = []

        for env_fn in env_fns:
            parent_pipe, child_pipe = mp.Pipe()
            proc = mp.Process(
                target=_env_worker,
                args=(env_fn, child_pipe, action_config),
                daemon=True,
            )
            proc.start()
            child_pipe.close()
            self.pipes.append(parent_pipe)
            self.processes.append(proc)

    def reset_all(self, reset_fns: list[Callable[[], dict]]) -> list[dict]:
        for pipe, fn in zip(self.pipes, reset_fns):
            pipe.send(("reset", fn()))
        return [pipe.recv() for pipe in self.pipes]

    def step(
        self, normalized_actions: np.ndarray
    ) -> tuple[list[dict], list[bool], list[dict | None]]:
        for i in range(self.n_envs):
            self.pipes[i].send(("step", normalized_actions[i]))
        obs_list: list[dict] = []
        terminal_list: list[bool] = []
        reset_obs_list: list[dict | None] = []
        for i in range(self.n_envs):
            result = self.pipes[i].recv()
            obs_list.append(result[0])
            terminal_list.append(result[1])
            reset_obs_list.append(result[2])
        return obs_list, terminal_list, reset_obs_list

    def close(self) -> None:
        for p in self.pipes:
            try:
                p.send(("close", None))
            except (BrokenPipeError, EOFError):
                pass
        for proc in self.processes:
            proc.join(timeout=5)


# ── Lockstep PPO rollout ──────────────────────────────────────────────────────


class RolloutDataset(
    IterableDataset[tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]]
):
    """Collects ``rollout_steps`` from N parallel envs in lockstep.

    Parameters
    ----------
    env_fns:
        Factory callables — one per parallel environment.
    policy:
        Policy used for action selection during rollout.
    rollout_steps:
        Number of steps collected from each environment per iteration.
    reward_fns:
        One reward functor per environment (each maintains its own state).
    obs_fn:
        Converts a raw gym observation dict into a ``(1, obs_dim)`` tensor.
    episode_returns:
        External list to which completed episode returns are appended.
    k_epochs:
        Number of PPO update epochs on the collected batch.
    mini_batch_size:
        Minibatch size for each gradient step.
    normalize_advantages:
        Whether to normalise advantage estimates across the batch.
    device:
        Torch device for policy inference and tensor storage.
    """

    def __init__(
        self,
        env_fns: list[Callable[[], gym.Env]],
        policy: Policy,
        rollout_steps: int,
        reward_fns: list[Callable[[dict[str, Any], bool], float]],
        obs_fn: Callable[[dict[str, Any]], Tensor],
        reset_fn: Callable[[], dict[str, Any]],
        episode_returns: list[float],
        k_epochs: int,
        mini_batch_size: int,
        normalize_advantages: bool,
        action_config: F1TenthActionConfig,
        device: torch.device = DEFAULT_DEVICE,
    ) -> None:
        self.sve = SubprocVecEnv(env_fns, action_config)
        self.policy = policy
        self.rollout_steps = rollout_steps
        self.reward_fns = reward_fns
        self.obs_fn = obs_fn
        self.reset_fn = reset_fn
        self.episode_returns = episode_returns
        self.k_epochs = k_epochs
        self.mini_batch_size = mini_batch_size
        self.normalize_advantages = normalize_advantages
        self.device = device
        self.current_obs: list[dict[str, Any] | None] = [None] * self.sve.n_envs

    def __iter__(
        self,
    ) -> Iterator[tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]]:
        n = self.sve.n_envs

        if self.current_obs[0] is None:
            obs_dicts = self.sve.reset_all([self.reset_fn] * self.sve.n_envs)
            self.current_obs = list(obs_dicts)

        # Per-env transition buffers.
        s_buf: list[list[Tensor]] = [[] for _ in range(n)]
        a_buf: list[list[Tensor]] = [[] for _ in range(n)]
        log_p_buf: list[list[Tensor]] = [[] for _ in range(n)]
        V_buf: list[list[Tensor]] = [[] for _ in range(n)]
        r_buf: list[list[float]] = [[] for _ in range(n)]
        terminal_buf: list[list[float]] = [[] for _ in range(n)]
        completed_episode_returns: list[float] = []

        for _step in range(self.rollout_steps):
            obs_list: list[dict[str, Any]] = cast(
                list[dict[str, Any]], self.current_obs
            )
            obs_tensors = [self.obs_fn(d).to(self.device) for d in obs_list]
            obs_batch = torch.cat(obs_tensors)  # (n, obs_dim)

            # Batched policy inference (single no_grad context).
            with torch.no_grad():
                action_batch, log_prob_batch, value_batch = self.policy.act(obs_batch)

            # Step all envs in parallel (child processes, separate GILs).
            actions_np = action_batch.cpu().numpy()
            next_obs_list, terminal_list, reset_obs_list = self.sve.step(actions_np)

            # Store transitions, compute rewards.
            for i in range(n):
                r_t = self.reward_fns[i](next_obs_list[i], terminal_list[i])
                s_buf[i].append(obs_batch[i])
                a_buf[i].append(action_batch[i])
                log_p_buf[i].append(log_prob_batch[i])
                V_buf[i].append(value_batch[i])
                r_buf[i].append(r_t)
                terminal_buf[i].append(float(terminal_list[i]))

                if terminal_list[i]:
                    completed_episode_returns.append(r_t)
                    self.current_obs[i] = reset_obs_list[i]
                else:
                    self.current_obs[i] = next_obs_list[i]

        final_obs_list: list[dict[str, Any]] = cast(
            list[dict[str, Any]], self.current_obs
        )
        final_obs_batch = torch.cat(
            [self.obs_fn(d).to(self.device) for d in final_obs_list]
        )
        with torch.no_grad():
            _, _, final_values = self.policy.act(final_obs_batch, deterministic=True)

        # Compute GAE per env and concatenate.
        all_s, all_a, all_log_p, all_A, all_R = [], [], [], [], []
        for i in range(n):
            V_buf[i].append(final_values[i])
            s = torch.stack(s_buf[i]).unsqueeze(0)
            a = torch.stack(a_buf[i]).unsqueeze(0)
            log_p = torch.stack(log_p_buf[i]).unsqueeze(0)
            r = torch.as_tensor(r_buf[i], device=self.device).unsqueeze(0)
            terminal = torch.as_tensor(terminal_buf[i], device=self.device).unsqueeze(0)
            V = torch.stack(V_buf[i]).unsqueeze(0)

            A, R = compute_gae(r=r, V_phi=V, terminal=terminal)

            all_s.append(s.reshape(-1, s.shape[-1]))
            all_a.append(a.reshape(-1, a.shape[-1]))
            all_log_p.append(log_p.reshape(-1))
            all_A.append(A.reshape(-1))
            all_R.append(R.reshape(-1))

        s = torch.cat(all_s)
        a = torch.cat(all_a)
        log_p_cur = torch.cat(all_log_p)
        A_hat = torch.cat(all_A)
        R_hat = torch.cat(all_R)

        if self.normalize_advantages:
            A_hat = (A_hat - A_hat.mean()) / (A_hat.std() + 1e-8)

        self.episode_returns.extend(completed_episode_returns)

        n_total = s.shape[0]
        for _ in range(self.k_epochs):
            index_set = torch.randperm(n_total, device=s.device)
            for start in range(0, n_total, self.mini_batch_size):
                B = index_set[start : start + self.mini_batch_size]
                yield (
                    s[B],
                    a[B],
                    log_p_cur[B],
                    A_hat[B],
                    R_hat[B],
                    torch.as_tensor(0.0, device=s.device),
                )

    def __len__(self) -> int:
        total_samples = self.sve.n_envs * self.rollout_steps
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
    "SubprocVecEnv",
    "build_observation",
    "initial_pose",
    "observation_dim",
    "obs_tensor",
    "scale_action",
]
