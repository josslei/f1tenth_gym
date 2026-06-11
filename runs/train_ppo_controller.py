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
from utils.track_scheduler import TrackScheduler, make_track_reset_fn

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


# ── Task-specific reward ──────────────────────────────────────────────────────


def make_reward_controller(
    reward_cfg: dict[str, Any],
) -> Any:
    speed_coef = float(reward_cfg.get("speed_coef", 2.0))
    clearance_threshold = float(reward_cfg.get("clearance_threshold", 1.5))
    proximity_threshold = float(reward_cfg.get("proximity_threshold", 0.5))
    proximity_coef = float(reward_cfg.get("proximity_coef", 2.0))
    collision_penalty = float(reward_cfg.get("collision_penalty", 1.0))
    lap_bonus = float(reward_cfg.get("lap_bonus", 1.0))

    def _reward(obs: dict[str, Any], terminated: bool) -> float:
        ego = int(obs["ego_idx"])
        speed = float(obs["linear_vels_x"][ego])
        collision = float(obs["collisions"][ego])

        if collision:
            return -collision_penalty
        if terminated:
            return lap_bonus

        min_scan = float(np.min(obs["scans"][ego]))
        clearance_factor = np.clip(min_scan / clearance_threshold, 0.0, 1.0)
        proximity_penalty = 0.0
        if min_scan < proximity_threshold:
            proximity_penalty = (proximity_threshold - min_scan) * proximity_coef

        return speed * speed_coef * clearance_factor - proximity_penalty

    return _reward


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
    output_dir = Path(config.output["dir"])
    observation_config = F1TenthObservationConfig(**config.observation)
    action_config = F1TenthActionConfig(**config.action)
    obs_dim = observation_dim(observation_config)
    action_dim = 2
    policy = make_policy(config.policy, obs_dim=obs_dim, action_dim=action_dim)
    module = LightningPPO(policy=policy, config=config)
    ppo_iterations = config.runtime.ppo_iterations
    rollout_steps = config.runtime.rollout_steps

    # ── Multi-track setup ─────────────────────────────────────────────────
    track_cfg = config.tracks
    cur_cfg = track_cfg["curriculum"]
    scheduler = TrackScheduler(
        root="tracks",
        holdout=track_cfg.get("holdout", []),
        test_ratio=float(track_cfg.get("test_ratio", 0.2)),
        seed=config.runtime.seed,
        initial=int(cur_cfg.get("initial", 4)),
        increment=int(cur_cfg.get("increment", 4)),
        interval_frac=float(cur_cfg.get("interval_frac", 0.1)),
    )
    print(f"TrackScheduler: {scheduler}")
    print(f"  Train: {scheduler.train_tracks}")
    print(f"  Test:  {scheduler.test_tracks}")
    print(f"  Holdout: {scheduler.holdout_tracks}")

    # Envs initialized with dummy map — reset_fn overwrites it before first step
    track_map_ext = env_config.get("map_ext", ".png")
    envs = [gym.make("f110-v0", **env_config) for _ in range(config.runtime.num_envs)]

    episode_returns: list[float] = []
    dataset = RolloutDataset(
        envs=envs,
        policy=policy,
        rollout_steps=rollout_steps,
        obs_fn=lambda obs: obs_tensor(obs, observation_config),
        action_fn=lambda a_t, e: scale_action(
            a_t.squeeze(0).detach().cpu().numpy(), e, action_config
        ),
        reward_fn=make_reward_controller(config.reward),
        reset_fn=make_track_reset_fn(scheduler, ppo_iterations, track_map_ext),
        episode_returns=episode_returns,
        k_epochs=config.training.k_epochs,
        mini_batch_size=config.training.mini_batch_size,
        normalize_advantages=config.training.normalize_advantages,
        device=DEFAULT_DEVICE,
        track_scheduler=scheduler,
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
    trainer = pl.Trainer(
        max_epochs=ppo_iterations,
        enable_progress_bar=config.runtime.progress_bar,
        logger=logger,
        callbacks=[checkpoint_callback],
    )
    trainer.fit(module, datamodule=datamodule)

    for update_idx, episode_return in enumerate(episode_returns):
        append_jsonl(
            output_dir / "metrics.jsonl",
            {"update": update_idx, "episode_return": episode_return},
        )

    save_policy(
        output_dir,
        policy,
        policy_config=config.policy,
        obs_dim=obs_dim,
        action_dim=action_dim,
    )
    for e in envs:
        e.close()


if __name__ == "__main__":
    main()
