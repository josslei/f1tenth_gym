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
from typing import Any

import gymnasium as gym
import lightning as pl
import numpy as np
import torch
from lightning.pytorch.loggers import TensorBoardLogger

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


def reward_controller(obs: dict[str, Any], terminated: bool) -> float:
    ego = int(obs["ego_idx"])
    speed = float(obs["linear_vels_x"][ego])
    collision = float(obs["collisions"][ego])
    return speed - 10.0 * collision + (10.0 if terminated and collision == 0.0 else 0.0)


# ── Checkpointing ─────────────────────────────────────────────────────────────


def save_policy(
    output_dir: Path,
    policy: Policy,
    policy_config: PolicyConfig,
    obs_dim: int,
    action_dim: int,
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
        output_dir / "final_model.pt",
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
    reset_pose = np.asarray(env_config.pop("initial_pose"), dtype=np.float64)
    output_dir = Path(config.output["dir"])
    observation_config = F1TenthObservationConfig(**config.observation)
    action_config = F1TenthActionConfig(**config.action)
    obs_dim = observation_dim(observation_config)
    action_dim = 2
    policy = make_policy(config.policy, obs_dim=obs_dim, action_dim=action_dim)
    module = LightningPPO(policy=policy, config=config)

    envs = [gym.make("f110-v0", **env_config) for _ in range(config.runtime.num_envs)]
    rollout_steps = config.runtime.rollout_steps
    ppo_iterations = config.runtime.ppo_iterations

    episode_returns: list[float] = []
    dataset = RolloutDataset(
        envs=envs,
        policy=policy,
        rollout_steps=rollout_steps,
        obs_fn=lambda obs: obs_tensor(obs, observation_config),
        action_fn=lambda a_t, e: scale_action(
            a_t.squeeze(0).detach().cpu().numpy(), e, action_config
        ),
        reward_fn=reward_controller,
        reset_fn=lambda: {"poses": reset_pose.copy()},
        episode_returns=episode_returns,
        k_epochs=config.training.k_epochs,
        mini_batch_size=config.training.mini_batch_size,
        normalize_advantages=config.training.normalize_advantages,
        device=DEFAULT_DEVICE,
    )
    datamodule = RolloutDataModule(dataset)
    logger = TensorBoardLogger(save_dir=output_dir, name="tensorboard")
    trainer = pl.Trainer(
        max_epochs=ppo_iterations,
        enable_progress_bar=config.runtime.progress_bar,
        logger=logger,
        enable_checkpointing=False,
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
