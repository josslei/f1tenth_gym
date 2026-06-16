"""F1TENTH controller PPO training entry point.

This script orchestrates the PPO training loop for an F1TENTH controller:
environment setup, rollout collection, PPO updates, and checkpointing.
Task-specific code (reward function) lives here; reusable environment
helpers live in ``utils/f110_env.py``.

TODO(Fix 6): Add temporal context for PPO convergence. Frame stacking would
require per-environment observation history in ``RolloutDataset``; recurrent
policies would additionally require hidden-state storage and reset handling.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from functools import partial
from pathlib import Path
from typing import Any, cast

import gymnasium as gym
import lightning as pl
import numpy as np
import torch
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import Callback

import f110_gym  # noqa: F401 - registers f110-v0
from models.policies import Policy, make_policy
from models.ppo import (
    LightningPPO,
    MapConfig,
    PolicyConfig,
    build_epoch_map_schedule,
    load_ppo_config,
    split_maps,
)
from utils.f110_env import (
    F1TenthActionConfig,
    F1TenthObservationConfig,
    RolloutDataModule,
    RolloutDataset,
    add_control_state,
    observation_dim,
    obs_tensor,
    scale_action,
    with_resampled_waypoints,
)
from utils.waypoint_view import initial_pose_from_waypoints

DEFAULT_DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/ppo/default.yaml")
    return parser.parse_args()


class F1TenthPPOReward:
    """Reward based on forward progress along the centerline.

    Computes per-step arc-length progress along the resampled waypoint
    path, plus speed bonus and steering smoothness penalty.  Replaces
    the older waypoint-hit-bonus design with a continuous progress
    signal that generalises across tracks.
    """

    def __init__(
        self,
        *,
        waypoints_path: str | Path | None = None,
        speed_reward_weight: float = 0.1,
        progress_weight: float = 1.0,
        steer_smoothness_weight: float = 0.5,
        collision_penalty: float = 50.0,
        spin_threshold: float = 100.0,
        delimiter: str = ";",
        usecols: tuple[int, int] = (1, 2),
    ) -> None:
        self.speed_reward_weight = speed_reward_weight
        self.progress_weight = progress_weight
        self.steer_smoothness_weight = steer_smoothness_weight
        self.collision_penalty = collision_penalty
        self.spin_threshold = spin_threshold

        if waypoints_path is not None:
            self.waypoints = np.genfromtxt(
                str(waypoints_path),
                delimiter=delimiter,
                comments="#",
                usecols=usecols,
            ).reshape(-1, 2)
            self._build_arc()
        else:
            self.waypoints = np.empty((0, 2), dtype=np.float64)
            self.cum_arc_lengths = np.array([0.0])
            self.total_length = 0.0

        self.prev_arc_length: float = 0.0
        self.prev_steer: float = 0.0

    def _build_arc(self) -> None:
        diffs = np.diff(self.waypoints, axis=0)
        seg_lengths = np.sqrt((diffs**2).sum(axis=1))
        self.cum_arc_lengths = np.concatenate([[0.0], np.cumsum(seg_lengths)])
        self.total_length = float(self.cum_arc_lengths[-1])

    def set_waypoints(self, waypoints_xy: np.ndarray) -> None:
        """Hot-swap the path used for forward-progress computation.

        Resets internal progress state and arc-length lookup so the
        next call starts fresh from the new path.
        """
        self.waypoints = np.asarray(waypoints_xy, dtype=np.float64).reshape(-1, 2)
        self._build_arc()
        self.prev_arc_length = 0.0
        self.prev_steer = 0.0

    def __call__(self, obs: dict[str, Any], terminated: bool) -> float:
        ego = int(obs["ego_idx"])
        vx = float(np.nan_to_num(obs["linear_vels_x"][ego], nan=0.0))
        vy = float(np.nan_to_num(obs["linear_vels_y"][ego], nan=0.0))
        collision = bool(obs["collisions"][ego])
        theta = float(np.nan_to_num(obs["poses_theta"][ego], nan=0.0))
        px = float(np.nan_to_num(obs["poses_x"][ego], nan=0.0))
        py = float(np.nan_to_num(obs["poses_y"][ego], nan=0.0))

        # Steering from augment (added by add_control_state).
        steer = float(np.nan_to_num(obs.get("steer_angle", [0.0])[ego], nan=0.0))

        if collision or abs(theta) > self.spin_threshold:
            if terminated:
                self.prev_arc_length = 0.0
                self.prev_steer = 0.0
            return -float(self.collision_penalty)

        if terminated:
            self.prev_arc_length = 0.0
            self.prev_steer = 0.0

        # ── speed reward ────────────────────────────────────────────────────
        vel_magnitude = np.sqrt(vx * vx + vy * vy)
        reward = self.speed_reward_weight * float(vel_magnitude)

        # ── forward progress along centreline ───────────────────────────────
        wx, wy = self.waypoints[:, 0], self.waypoints[:, 1]
        dist_sq = (wx - px) ** 2 + (wy - py) ** 2
        nearest_idx = int(np.argmin(dist_sq))
        current_arc = float(self.cum_arc_lengths[nearest_idx])

        progress = current_arc - self.prev_arc_length
        if progress < 0.0:
            # Wrapped to a new lap.
            progress = (self.total_length - self.prev_arc_length) + current_arc
        reward += self.progress_weight * progress
        self.prev_arc_length = current_arc

        # ── steering smoothness penalty ─────────────────────────────────────
        steer_delta = abs(steer - self.prev_steer)
        reward -= self.steer_smoothness_weight * steer_delta
        self.prev_steer = steer

        return float(np.clip(reward, -5.0, 8.0))


# ── Checkpointing ─────────────────────────────────────────────────────────────


class DeployableCheckpoint(Callback):
    """Save deployable policy ``.pt`` checkpoints every N epochs.

    Each checkpoint is a flat dict loadable by ``PPOController.from_checkpoint()``,
    identical to what ``save_policy()`` produces at the end of training.
    """

    def __init__(
        self,
        dirpath: Path,
        every_n_epochs: int,
        policy_config: PolicyConfig,
        obs_dim: int,
        action_dim: int,
        observation_config: F1TenthObservationConfig | None = None,
        action_config: F1TenthActionConfig | None = None,
    ) -> None:
        self.dirpath = Path(dirpath)
        self.every_n_epochs = every_n_epochs
        self.policy_config = policy_config
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.observation_config = observation_config
        self.action_config = action_config

    def on_train_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        epoch = trainer.current_epoch + 1
        if epoch % self.every_n_epochs == 0:
            self.dirpath.mkdir(parents=True, exist_ok=True)
            save_policy(
                self.dirpath,
                cast(Policy, pl_module.policy),
                policy_config=self.policy_config,
                obs_dim=self.obs_dim,
                action_dim=self.action_dim,
                observation_config=self.observation_config,
                action_config=self.action_config,
                filename=f"policy-epoch-{epoch:04d}.pt",
            )


class ValidationCallback(Callback):
    def __init__(
        self,
        val_maps: list[MapConfig],
        val_episodes: int,
        observation_config: F1TenthObservationConfig,
        action_config: F1TenthActionConfig,
        max_episode_steps: int,
        device: torch.device,
        map_waypoints: dict[str, np.ndarray],
        map_poses: dict[str, np.ndarray],
        laps_to_complete: int,
    ) -> None:
        self.val_maps = val_maps
        self.val_episodes = val_episodes
        self.base_observation_config = observation_config
        self.action_config = action_config
        self.max_episode_steps = max_episode_steps
        self.device = device
        self.map_waypoints = map_waypoints
        self.map_poses = map_poses
        self.laps_to_complete = laps_to_complete

    def on_train_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        policy: Policy = cast(Policy, pl_module.policy)
        policy.eval()

        for val_map in self.val_maps:
            env = gym.make(
                "f110-v0",
                map=val_map.map,
                map_ext=val_map.map_ext,
                num_agents=1,
                laps_to_complete=self.laps_to_complete,
            )

            obs_config = self.base_observation_config
            pose = self.map_poses.get(val_map.name)
            wp_xy = self.map_waypoints.get(val_map.name)
            if obs_config.include_waypoints and wp_xy is not None:
                obs_config = with_resampled_waypoints(obs_config, wp_xy)
            if pose is None:
                pose = np.array([[0.0, 0.0, 0.0]], dtype=np.float64)

            val_returns: list[float] = []
            for _ in range(self.val_episodes):
                obs, _ = env.reset(options={"poses": pose.copy()})
                obs = add_control_state(obs, env)
                prev_action = np.zeros(2, dtype=np.float64)
                obs["prev_action"] = prev_action
                ep_steps = 0

                for _ in range(self.max_episode_steps):
                    obs_t = obs_tensor(obs, obs_config).to(self.device)
                    with torch.no_grad():
                        action, _, _ = policy.act(obs_t, deterministic=True)
                    env_action = scale_action(
                        action.squeeze(0).cpu().numpy(),
                        env,
                        self.action_config,
                    )
                    prev_action = action.squeeze(0).cpu().numpy()
                    obs, _, terminated, truncated, _ = env.step(env_action)
                    obs = add_control_state(obs, env)
                    obs["prev_action"] = prev_action
                    ep_steps += 1
                    if terminated or truncated:
                        break

                val_returns.append(float(ep_steps))

            env.close()

            mean_return = float(np.mean(val_returns)) if val_returns else 0.0
            if trainer.logger is not None:
                trainer.logger.log_metrics(
                    {
                        f"val/{val_map.name}/episode_return": mean_return,
                    },
                    step=trainer.current_epoch,
                )

        policy.train()


def save_policy(
    output_dir: Path,
    policy: Policy,
    policy_config: PolicyConfig,
    obs_dim: int,
    action_dim: int,
    observation_config: F1TenthObservationConfig | None = None,
    action_config: F1TenthActionConfig | None = None,
    filename: str = "final_model.pt",
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "hyper_parameters": {
            "obs_dim": obs_dim,
            "action_dim": action_dim,
            "policy": asdict(policy_config),
        },
        "obs_dim": obs_dim,
        "action_dim": action_dim,
        "policy": asdict(policy_config),
        "policy_state_dict": policy.state_dict(),
    }
    if observation_config is not None:
        obs_cfg_dict = {
            k: v for k, v in asdict(observation_config).items() if not k.startswith("_")
        }
        payload["observation_config"] = obs_cfg_dict
    if action_config is not None:
        payload["action_config"] = asdict(action_config)
    torch.save(payload, output_dir / filename)


def log_map_split(
    output_dir: Path,
    train_maps: list[MapConfig],
    val_maps: list[MapConfig],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "train": [asdict(m) for m in train_maps],
        "validation": [asdict(m) for m in val_maps],
    }
    (output_dir / "map_split.json").write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"Training maps ({len(train_maps)}):")
    for m in train_maps:
        print(f"  - {m.name}")
    print(f"Validation maps ({len(val_maps)}):")
    for m in val_maps:
        print(f"  - {m.name}")


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    args = parse_args()
    config = load_ppo_config(args.config)

    torch.manual_seed(config.runtime.seed)
    np.random.seed(config.runtime.seed)

    output_dir = Path(config.output["dir"])
    observation_config = F1TenthObservationConfig(**config.observation)
    action_config = F1TenthActionConfig(**config.action)
    obs_dim = observation_dim(observation_config)
    action_dim = 2
    policy = make_policy(config.policy, obs_dim=obs_dim, action_dim=action_dim)
    module = LightningPPO(policy=policy, config=config)

    ppo_iterations = config.runtime.ppo_iterations
    rollout_steps = config.runtime.rollout_steps
    num_envs = config.runtime.num_envs

    # ── Multi-map setup ────────────────────────────────────────────────
    has_maps = config.maps is not None and len(config.maps.candidates) > 0
    val_maps: list[MapConfig] = []

    if has_maps:
        assert config.maps is not None  # narrow for pyright
        split_seed = config.maps.split_seed or config.runtime.seed
        train_maps, val_maps = split_maps(
            config.maps.candidates,
            seed=split_seed,
            validation_ratio=config.maps.validation_ratio,
            exclude=config.maps.exclude,
        )
        log_map_split(output_dir, train_maps, val_maps)
        epoch_schedule = build_epoch_map_schedule(
            train_maps,
            seed=config.runtime.seed,
            epochs=ppo_iterations,
        )
        map_waypoints: dict[str, np.ndarray] = {}
        map_poses: dict[str, np.ndarray] = {}
        for m in [*train_maps, *val_maps]:
            cl = np.loadtxt(m.centerline_csv, delimiter=",", skiprows=1)[:, :2]
            map_waypoints[m.name] = cl
            map_poses[m.name] = initial_pose_from_waypoints(cl)
        first_map = epoch_schedule[0]
        first_wp = map_waypoints[first_map.name]
        reset_pose = map_poses[first_map.name]
    else:
        # Fallback for single-map configs (no maps section).
        env_config = dict(config.env)
        centerline_csv = env_config.pop("centerline_csv", None)
        centerline_data: np.ndarray | None = None
        if centerline_csv:
            centerline_data = np.loadtxt(centerline_csv, delimiter=",", skiprows=1)
            reset_pose = initial_pose_from_waypoints(centerline_data[:, :2])
        else:
            reset_pose = np.asarray(env_config.pop("initial_pose"), dtype=np.float64)
        epoch_schedule = []
        map_waypoints = {}
        map_poses = {}
        first_wp = centerline_data[:, :2] if centerline_data is not None else None

    # Wire first map's waypoints into the observation config.
    if observation_config.include_waypoints and first_wp is not None:
        observation_config = with_resampled_waypoints(observation_config, first_wp)

    # Build env factories (no map baked in — map is switched dynamically
    # by RolloutDataset when map_schedule is provided).
    env_config = dict(config.env)
    env_fn = partial(gym.make, "f110-v0", **env_config)
    env_fns = [env_fn] * num_envs

    # Reward setup.
    reward_params = dict(config.reward)
    reward_params.pop("waypoints_path", None)
    reward_fns = [F1TenthPPOReward(**reward_params) for _ in range(num_envs)]
    if first_wp is not None:
        for rf in reward_fns:
            rf.set_waypoints(first_wp)

    episode_returns: list[float] = []
    dataset = RolloutDataset(
        env_fns=env_fns,
        policy=policy,
        rollout_steps=rollout_steps,
        reward_fns=reward_fns,
        observation_config=observation_config,
        reset_fn=lambda: {"poses": reset_pose.copy()},
        episode_returns=episode_returns,
        k_epochs=config.training.k_epochs,
        mini_batch_size=config.training.mini_batch_size,
        normalize_advantages=config.training.normalize_advantages,
        action_config=action_config,
        max_episode_steps=rollout_steps,
        device=DEFAULT_DEVICE,
        map_schedule=epoch_schedule if has_maps else None,
        map_waypoints=map_waypoints if has_maps else None,
        map_poses=map_poses if has_maps else None,
    )
    datamodule = RolloutDataModule(dataset)
    logger = TensorBoardLogger(save_dir=output_dir, name="tensorboard")
    checkpoint_callback = DeployableCheckpoint(
        dirpath=output_dir / "checkpoints",
        every_n_epochs=config.runtime.checkpoint_every_n_epochs,
        policy_config=config.policy,
        obs_dim=obs_dim,
        action_dim=action_dim,
        observation_config=observation_config,
        action_config=action_config,
    )
    callbacks: list[Callback] = [checkpoint_callback]

    if has_maps and val_maps:
        val_cb = ValidationCallback(
            val_maps=val_maps,
            val_episodes=2,
            observation_config=observation_config,
            action_config=action_config,
            max_episode_steps=rollout_steps,
            device=DEFAULT_DEVICE,
            map_waypoints=map_waypoints,
            map_poses=map_poses,
            laps_to_complete=int(config.env.get("laps_to_complete", 2)),
        )
        callbacks.append(val_cb)
    elif config.validation is not None:
        val_cb = ValidationCallback(
            val_maps=[
                MapConfig(
                    name=Path(config.validation.map).stem,
                    map=config.validation.map,
                    map_ext=config.validation.map_ext,
                    centerline_csv=config.validation.centerline_csv,
                )
            ],
            val_episodes=config.validation.episodes,
            observation_config=observation_config,
            action_config=action_config,
            max_episode_steps=rollout_steps,
            device=DEFAULT_DEVICE,
            map_waypoints=(
                {
                    Path(config.validation.map).stem: np.loadtxt(
                        config.validation.centerline_csv, delimiter=",", skiprows=1
                    )[:, :2]
                }
                if config.validation.centerline_csv
                else {}
            ),
            map_poses={},
            laps_to_complete=int(config.env.get("laps_to_complete", 2)),
        )
        callbacks.append(val_cb)

    trainer = pl.Trainer(
        max_epochs=ppo_iterations,
        enable_progress_bar=config.runtime.progress_bar,
        logger=logger,
        callbacks=callbacks,
    )
    trainer.fit(module, datamodule=datamodule)

    save_policy(
        output_dir,
        policy,
        policy_config=config.policy,
        obs_dim=obs_dim,
        action_dim=action_dim,
        observation_config=observation_config,
        action_config=action_config,
    )
    dataset.sve.close()


if __name__ == "__main__":
    main()
