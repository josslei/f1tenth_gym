"""F1TENTH controller PPO training entry point.

This script orchestrates the PPO training loop for an F1TENTH controller:
environment setup, rollout collection, PPO updates, and checkpointing.
Task-specific code (reward function) lives here; reusable environment
helpers live in ``utils/f110_env.py``.
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
from models.ppo import LightningPPO, PolicyConfig, load_ppo_config
from utils.f110_env import (
    F1TenthActionConfig,
    F1TenthObservationConfig,
    RolloutDataModule,
    RolloutDataset,
    observation_dim,
    obs_tensor,
    scale_action,
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
    def __init__(
        self,
        *,
        waypoints_path: str | Path,
        waypoint_proximity: float = 2.0,
        waypoint_bonus_weight: float = 0.5,
        collision_penalty: float = 1.0,
        spin_threshold: float = 100.0,
    ) -> None:
        self.waypoint_proximity = waypoint_proximity
        self.waypoint_bonus_weight = waypoint_bonus_weight
        self.collision_penalty = collision_penalty
        self.spin_threshold = spin_threshold
        self.waypoints = np.genfromtxt(
            str(waypoints_path), delimiter=";", comments="#", usecols=(1, 2)
        ).reshape(-1, 2)
        self.idx = 0

    def __call__(self, obs: dict[str, Any], terminated: bool) -> float:
        ego = int(obs["ego_idx"])
        vx = float(obs["linear_vels_x"][ego])
        vy = float(obs["linear_vels_y"][ego])
        collision = bool(obs["collisions"][ego])
        theta = float(obs["poses_theta"][ego])

        if terminated:
            self.idx = 0

        if collision or abs(theta) > self.spin_threshold:
            return -self.collision_penalty

        vel_magnitude = np.sqrt(vx * vx + vy * vy)
        reward = float(vel_magnitude)

        if self.idx < len(self.waypoints):
            wx, wy = self.waypoints[self.idx, :2]
            px = float(obs["poses_x"][ego])
            py = float(obs["poses_y"][ego])
            dist = np.sqrt((px - wx) ** 2 + (py - wy) ** 2)
            if dist < self.waypoint_proximity:
                self.idx += 1
                if self.idx <= len(self.waypoints):
                    lap_frac = self.idx / len(self.waypoints)
                    reward += float(lap_frac * self.waypoint_bonus_weight)

        return reward


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
    ) -> None:
        self.dirpath = Path(dirpath)
        self.every_n_epochs = every_n_epochs
        self.policy_config = policy_config
        self.obs_dim = obs_dim
        self.action_dim = action_dim

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
                filename=f"policy-epoch-{epoch:04d}.pt",
            )


class ValidationCallback(Callback):
    def __init__(
        self,
        val_map: str,
        val_map_ext: str,
        val_episodes: int,
        centerline_csv: str,
        observation_config: F1TenthObservationConfig,
        action_config: F1TenthActionConfig,
        max_episode_steps: int,
        device: torch.device,
    ) -> None:
        self.val_map = val_map
        self.val_map_ext = val_map_ext
        self.val_episodes = val_episodes
        self.observation_config = observation_config
        self.action_config = action_config
        self.max_episode_steps = max_episode_steps
        self.device = device

        waypoints = np.loadtxt(centerline_csv, delimiter=",", skiprows=1)
        self.val_pose = initial_pose_from_waypoints(waypoints[:, :2])

    def on_train_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        policy: Policy = cast(Policy, pl_module.policy)
        policy.eval()

        env = gym.make(
            "f110-v0",
            map=self.val_map,
            map_ext=self.val_map_ext,
            num_agents=1,
        )
        val_returns: list[float] = []

        for _ in range(self.val_episodes):
            obs, _ = env.reset(options={"poses": self.val_pose.copy()})
            ep_steps = 0

            for _ in range(self.max_episode_steps):
                obs_t = obs_tensor(obs, self.observation_config).to(self.device)
                with torch.no_grad():
                    action, _, _ = policy.act(obs_t, deterministic=True)
                env_action = scale_action(
                    action.squeeze(0).cpu().numpy(),
                    env,
                    self.action_config,
                )
                obs, _, terminated, truncated, _ = env.step(env_action)
                ep_steps += 1
                if terminated or truncated:
                    break

            val_returns.append(float(ep_steps))

        env.close()
        policy.train()

        mean_return = float(np.mean(val_returns))
        if trainer.logger is not None:
            trainer.logger.log_metrics(
                {"val/episode_return": mean_return},
                step=trainer.current_epoch,
            )


def save_policy(
    output_dir: Path,
    policy: Policy,
    policy_config: PolicyConfig,
    obs_dim: int,
    action_dim: int,
    filename: str = "final_model.pt",
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "hyper_parameters": {
                "obs_dim": obs_dim,
                "action_dim": action_dim,
                "policy": asdict(policy_config),
            },
            "obs_dim": obs_dim,
            "action_dim": action_dim,
            "policy": asdict(policy_config),
            "policy_state_dict": policy.state_dict(),
        },
        output_dir / filename,
    )


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record) + "\n")


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    args = parse_args()
    config = load_ppo_config(args.config)

    torch.manual_seed(config.runtime.seed)
    np.random.seed(config.runtime.seed)

    env_config = dict(config.env)
    centerline_csv = env_config.pop("centerline_csv", None)
    if centerline_csv:
        centerline_data = np.loadtxt(centerline_csv, delimiter=",", skiprows=1)
        reset_pose = initial_pose_from_waypoints(centerline_data[:, :2])
    else:
        reset_pose = np.asarray(env_config.pop("initial_pose"), dtype=np.float64)
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
    env_fn = partial(gym.make, "f110-v0", **env_config)
    env_fns = [env_fn] * num_envs

    reward_params = dict(config.reward)
    episode_returns: list[float] = []
    dataset = RolloutDataset(
        env_fns=env_fns,
        policy=policy,
        rollout_steps=rollout_steps,
        reward_fns=[F1TenthPPOReward(**reward_params) for _ in range(num_envs)],
        obs_fn=lambda obs: obs_tensor(obs, observation_config),
        reset_fn=lambda: {"poses": reset_pose.copy()},
        episode_returns=episode_returns,
        k_epochs=config.training.k_epochs,
        mini_batch_size=config.training.mini_batch_size,
        normalize_advantages=config.training.normalize_advantages,
        action_config=action_config,
        max_episode_steps=rollout_steps,
        device=DEFAULT_DEVICE,
    )
    datamodule = RolloutDataModule(dataset)
    logger = TensorBoardLogger(save_dir=output_dir, name="tensorboard")
    checkpoint_callback = DeployableCheckpoint(
        dirpath=output_dir / "checkpoints",
        every_n_epochs=config.runtime.checkpoint_every_n_epochs,
        policy_config=config.policy,
        obs_dim=obs_dim,
        action_dim=action_dim,
    )
    callbacks: list[Callback] = [checkpoint_callback]

    if config.validation is not None:
        val_cb = ValidationCallback(
            val_map=config.validation.map,
            val_map_ext=config.validation.map_ext,
            val_episodes=config.validation.episodes,
            centerline_csv=config.validation.centerline_csv,
            observation_config=observation_config,
            action_config=action_config,
            max_episode_steps=rollout_steps,
            device=DEFAULT_DEVICE,
        )
        callbacks.append(val_cb)

    trainer = pl.Trainer(
        max_epochs=ppo_iterations,
        enable_progress_bar=config.runtime.progress_bar,
        logger=logger,
        callbacks=callbacks,
    )
    trainer.fit(module, datamodule=datamodule)

    for episode_idx, episode_return in enumerate(episode_returns):
        append_jsonl(
            output_dir / "metrics.jsonl",
            {"episode": episode_idx, "episode_return": episode_return},
        )

    save_policy(
        output_dir,
        policy,
        policy_config=config.policy,
        obs_dim=obs_dim,
        action_dim=action_dim,
    )
    dataset.sve.close()


if __name__ == "__main__":
    main()
