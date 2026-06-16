from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass, field, replace
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
from models.ppo.config import MapConfig
from utils.waypoint_utils import nearest_waypoint_index, resample_path


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


def add_control_state(obs: dict[str, Any], env: Any) -> dict[str, Any]:
    augmented = dict(obs)
    f110_env = env.unwrapped
    augmented["steer_angle"] = np.asarray(
        [agent.state[2] for agent in f110_env.sim.agents], dtype=np.float64
    )
    return augmented


def with_resampled_waypoints(
    config: F1TenthObservationConfig, waypoints_xy: np.ndarray
) -> F1TenthObservationConfig:
    return replace(
        config,
        _waypoints=resample_path(waypoints_xy, config.waypoint_resample_spacing),
    )


# ── Config dataclasses ────────────────────────────────────────────────────────


@dataclass(frozen=True)
class F1TenthObservationConfig:
    scan_size: int = 108
    scan_max_m: float = 30.0
    include_ego_state: bool = True
    speed_scale: float = 8.0
    yaw_rate_scale: float = 10.0
    steer_scale: float = 1.066

    # ── waypoint features ──────────────────────────────────────────────────
    include_waypoints: bool = False
    lookahead_distances: tuple[float, ...] = (
        0.5,
        1.0,
        2.0,
        3.5,
        5.5,
        8.0,
        11.0,
        14.5,
        18.5,
        23.0,
        28.0,
        33.0,
    )
    waypoint_scale: float = 30.0
    waypoint_resample_spacing: float = 0.5

    # Internal — set at runtime after loading the map's centerline.
    _waypoints: np.ndarray | None = field(default=None, repr=False, compare=False)

    # TODO(Fix 6): Add temporal context support for PPO convergence.
    #   A future `frame_stack: int = 1` field should make observation_dim()
    #   multiply the single-frame dimension and make the rollout path maintain
    #   a per-environment ring buffer of the last N observations. Recurrent
    #   policies would instead need hidden-state plumbing through Policy.act(),
    #   Policy.evaluate_actions(), and RolloutDataset reset boundaries.


@dataclass(frozen=True)
class F1TenthActionConfig:
    velocity_min: float = 0.0
    velocity_max: float = 8.0


# ── Observation helpers ───────────────────────────────────────────────────────


def observation_dim(config: F1TenthObservationConfig) -> int:
    ego_state_dim = 7 if config.include_ego_state else 0
    waypoint_dim = (
        len(config.lookahead_distances) * 2 if config.include_waypoints else 0
    )
    return config.scan_size + ego_state_dim + waypoint_dim + 2


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
        steer_angle = np.asarray(obs["steer_angle"], dtype=np.float64)[ego]
        theta = np.asarray(obs["poses_theta"], dtype=np.float64)[ego]
        collision = np.asarray(obs["collisions"], dtype=np.float64)[ego]
        features.append(
            np.array(
                [
                    np.clip(linear_vels_x / config.speed_scale, -1.0, 1.0),
                    np.clip(linear_vels_y / config.speed_scale, -1.0, 1.0),
                    np.clip(ang_vels_z / config.yaw_rate_scale, -1.0, 1.0),
                    np.clip(steer_angle / config.steer_scale, -1.0, 1.0),
                    np.sin(theta),
                    np.cos(theta),
                    np.clip(collision, 0.0, 1.0),
                ],
                dtype=np.float64,
            )
        )

    if config.include_waypoints and len(config.lookahead_distances) > 0:
        if config._waypoints is None:
            features.append(
                np.zeros(len(config.lookahead_distances) * 2, dtype=np.float64)
            )
        else:
            px = float(np.nan_to_num(obs["poses_x"][ego], nan=0.0))
            py = float(np.nan_to_num(obs["poses_y"][ego], nan=0.0))
            theta = float(np.nan_to_num(obs["poses_theta"][ego], nan=0.0))
            position = np.array([px, py], dtype=np.float64)
            wp = config._waypoints
            nearest_idx = nearest_waypoint_index(wp, position)
            n_wp = wp.shape[0]
            cos_t = np.cos(theta)
            sin_t = np.sin(theta)
            waypoint_vals: list[float] = []
            spacing = config.waypoint_resample_spacing
            for d in config.lookahead_distances:
                offset = max(1, int(round(d / spacing)))
                idx = (nearest_idx + offset) % n_wp
                dx = wp[idx, 0] - px
                dy = wp[idx, 1] - py
                x_rel = cos_t * dx + sin_t * dy
                y_rel = -sin_t * dx + cos_t * dy
                waypoint_vals.append(
                    float(np.clip(x_rel / config.waypoint_scale, -1.0, 1.0))
                )
                waypoint_vals.append(
                    float(np.clip(y_rel / config.waypoint_scale, -1.0, 1.0))
                )
            features.append(np.array(waypoint_vals, dtype=np.float64))

    if "prev_action" in obs:
        prev_action = np.clip(
            np.asarray(obs["prev_action"], dtype=np.float64), -1.0, 1.0
        )
        features.append(prev_action)
    else:
        features.append(np.zeros(2, dtype=np.float64))

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
    max_episode_steps: int,
) -> None:
    env = env_fn()
    f110_env: Any = env.unwrapped
    params = f110_env.params
    current_reset_options: dict[str, Any] | None = None
    episode_steps = 0
    while True:
        try:
            cmd, data = pipe.recv()
        except (EOFError, BrokenPipeError):
            break

        if cmd == "reset":
            current_reset_options = data
            episode_steps = 0
            obs_dict, info = (
                env.reset(options=data) if data is not None else env.reset()
            )
            obs_dict = add_control_state(obs_dict, env)
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
            obs_dict = add_control_state(obs_dict, env)
            episode_steps += 1
            if episode_steps >= max_episode_steps:
                truncated = True
            if terminated or truncated:
                episode_steps = 0
                reset_obs, _ = env.reset(options=current_reset_options)
                reset_obs = add_control_state(reset_obs, env)
                pipe.send((obs_dict, terminated, truncated, reset_obs))
            else:
                pipe.send((obs_dict, terminated, truncated, None))
        elif cmd == "update_map":
            map_path, map_ext = data
            env.unwrapped.update_map(map_path, map_ext)  # type: ignore[union-attr]
            pipe.send(True)
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
        max_episode_steps: int,
    ) -> None:
        self.n_envs = len(env_fns)
        self.pipes: list[Connection] = []
        self.processes: list[mp.Process] = []

        for env_fn in env_fns:
            parent_pipe, child_pipe = mp.Pipe()
            proc = mp.Process(
                target=_env_worker,
                args=(env_fn, child_pipe, action_config, max_episode_steps),
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
    ) -> tuple[list[dict], list[bool], list[bool], list[dict | None]]:
        for i in range(self.n_envs):
            self.pipes[i].send(("step", normalized_actions[i]))
        obs_list: list[dict] = []
        terminated_list: list[bool] = []
        truncated_list: list[bool] = []
        reset_obs_list: list[dict | None] = []
        for i in range(self.n_envs):
            result = self.pipes[i].recv()
            obs_list.append(result[0])
            terminated_list.append(result[1])
            truncated_list.append(result[2])
            reset_obs_list.append(result[3])
        return obs_list, terminated_list, truncated_list, reset_obs_list

    def set_map_all(self, map_path: str, map_ext: str) -> None:
        """Update the simulator map in every worker process."""
        for pipe in self.pipes:
            pipe.send(("update_map", (map_path, map_ext)))
        for pipe in self.pipes:
            pipe.recv()  # wait for all acknowledgements

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
    IterableDataset[
        tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]
    ]
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
    observation_config:
        Observation configuration — used to build observation tensors
        directly (replaces the older ``obs_fn`` callable).
    reset_fn:
        Callable returning a ``{"poses": ...}`` dict for environment reset.
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
    map_schedule:
        Per-epoch map schedule for multi-map training.  When set, each
        call to ``__iter__`` (one PPO iteration) switches the simulator
        map, waypoints, reset pose, and reward functions accordingly.
    map_waypoints:
        Pre-loaded waypoints keyed by map name.
    map_poses:
        Pre-loaded reset poses keyed by map name.
    """

    def __init__(
        self,
        env_fns: list[Callable[[], gym.Env]],
        policy: Policy,
        rollout_steps: int,
        reward_fns: list[Callable[[dict[str, Any], bool], float]],
        observation_config: F1TenthObservationConfig,
        reset_fn: Callable[[], dict[str, Any]],
        episode_returns: list[float],
        k_epochs: int,
        mini_batch_size: int,
        normalize_advantages: bool,
        action_config: F1TenthActionConfig,
        max_episode_steps: int,
        device: torch.device = DEFAULT_DEVICE,
        map_schedule: list[MapConfig] | None = None,
        map_waypoints: dict[str, np.ndarray] | None = None,
        map_poses: dict[str, np.ndarray] | None = None,
    ) -> None:
        self.sve = SubprocVecEnv(env_fns, action_config, max_episode_steps)
        self.policy = policy
        self.rollout_steps = rollout_steps
        self.reward_fns = reward_fns
        self.observation_config = observation_config
        self.reset_fn = reset_fn
        self.episode_returns = episode_returns
        self.k_epochs = k_epochs
        self.mini_batch_size = mini_batch_size
        self.normalize_advantages = normalize_advantages
        self.device = device
        self.current_obs: list[dict[str, Any] | None] = [None] * self.sve.n_envs
        self.ep_return: list[float] = []
        self.map_schedule = map_schedule or []
        self.map_waypoints = map_waypoints or {}
        self.map_poses = map_poses or {}
        self._iteration_count = 0
        self._force_reset = True

    def __iter__(
        self,
    ) -> Iterator[
        tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]
    ]:
        n = self.sve.n_envs

        # ── Per-epoch map switching ──────────────────────────────────────
        if self.map_schedule:
            current_map = self.map_schedule[
                self._iteration_count % len(self.map_schedule)
            ]
            self._iteration_count += 1

            self.sve.set_map_all(current_map.map, current_map.map_ext)

            wp_xy = self.map_waypoints.get(current_map.name)
            if wp_xy is not None:
                self.observation_config = with_resampled_waypoints(
                    self.observation_config, wp_xy
                )

            pose = self.map_poses.get(current_map.name)
            if pose is not None:
                self._current_reset_pose = pose.copy()
                self.reset_fn = lambda: {"poses": self._current_reset_pose.copy()}

            if wp_xy is not None:
                for rf in self.reward_fns:
                    rf.set_waypoints(wp_xy)  # type: ignore[union-attr]

            self._force_reset = True

        if self.current_obs[0] is None or self._force_reset:
            obs_dicts = self.sve.reset_all([self.reset_fn] * n)
            self.current_obs = list(obs_dicts)
            self._force_reset = False

        # Per-env transition buffers.
        s_buf: list[list[Tensor]] = [[] for _ in range(n)]
        a_buf: list[list[Tensor]] = [[] for _ in range(n)]
        log_p_buf: list[list[Tensor]] = [[] for _ in range(n)]
        V_buf: list[list[Tensor]] = [[] for _ in range(n)]
        r_buf: list[list[float]] = [[] for _ in range(n)]
        terminal_buf: list[list[float]] = [[] for _ in range(n)]
        completed_episode_returns: list[float] = []
        completed_episode_lap_numbers: list[int] = []
        completed_episode_lap_times: list[float] = []
        if len(self.ep_return) != n:
            self.ep_return = [0.0] * n

        # Previous action buffer (for prev_action observation feature).
        prev_action_batch = np.zeros((n, 2), dtype=np.float64)

        for _step in range(self.rollout_steps):
            obs_list: list[dict[str, Any]] = cast(
                list[dict[str, Any]], self.current_obs
            )
            for i in range(n):
                obs_list[i]["prev_action"] = prev_action_batch[i]
            obs_tensors = [obs_tensor(d, self.observation_config) for d in obs_list]
            obs_batch = torch.cat(obs_tensors).to(self.device)  # (n, obs_dim)

            # Batched policy inference (single no_grad context).
            with torch.no_grad():
                action_batch, log_prob_batch, value_batch = self.policy.act(obs_batch)

            # Step all envs in parallel (child processes, separate GILs).
            actions_np = action_batch.cpu().numpy()
            prev_action_batch = actions_np
            (
                next_obs_list,
                terminated_list,
                truncated_list,
                reset_obs_list,
            ) = self.sve.step(actions_np)

            # Store transitions, compute rewards.
            for i in range(n):
                any_terminal = terminated_list[i] or truncated_list[i]
                r_t = self.reward_fns[i](next_obs_list[i], any_terminal)
                self.ep_return[i] += r_t
                s_buf[i].append(obs_batch[i])
                a_buf[i].append(action_batch[i])
                log_p_buf[i].append(log_prob_batch[i])
                V_buf[i].append(value_batch[i])
                r_buf[i].append(r_t)
                terminal_buf[i].append(float(terminated_list[i]))

                if any_terminal:
                    ego = int(next_obs_list[i]["ego_idx"])
                    completed_episode_lap_numbers.append(
                        int(next_obs_list[i]["lap_counts"][ego])
                    )
                    completed_episode_lap_times.append(
                        float(next_obs_list[i]["lap_times"][ego])
                    )
                    completed_episode_returns.append(self.ep_return[i])
                    self.ep_return[i] = 0.0
                    self.current_obs[i] = reset_obs_list[i]
                else:
                    self.current_obs[i] = next_obs_list[i]

        final_obs_list: list[dict[str, Any]] = cast(
            list[dict[str, Any]], self.current_obs
        )
        for i in range(n):
            final_obs_list[i]["prev_action"] = prev_action_batch[i]
        final_obs_batch = torch.cat(
            [obs_tensor(d, self.observation_config) for d in final_obs_list]
        ).to(self.device)
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
        mean_ep_return = (
            float(np.mean(completed_episode_returns))
            if completed_episode_returns
            else 0.0
        )
        mean_lap_number = (
            float(np.mean(completed_episode_lap_numbers))
            if completed_episode_lap_numbers
            else 0.0
        )
        mean_lap_time = (
            float(np.mean(completed_episode_lap_times))
            if completed_episode_lap_times
            else 0.0
        )
        completed_episode_count = float(len(completed_episode_returns))

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
                    torch.as_tensor(mean_ep_return, device=s.device),
                    torch.as_tensor(mean_lap_number, device=s.device),
                    torch.as_tensor(mean_lap_time, device=s.device),
                    torch.as_tensor(completed_episode_count, device=s.device),
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
