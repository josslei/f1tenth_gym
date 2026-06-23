from __future__ import annotations

# pyright: reportAttributeAccessIssue=none, reportArgumentType=none, reportCallIssue=none

import argparse
import faulthandler
from functools import partial
from pathlib import Path

import gymnasium as gym
import lightning as pl
import numpy as np
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger

import f110_gym  # noqa: F401 - registers f110-v0
from models.muzero import (
    DiscreteActionConfig,
    DiscreteActionSpace,
    F110MuZeroNet,
    LightningMuZero,
    MuZeroReplayBuffer,
    ProgressReward,
    SelfPlayCallback,
    TorchScriptExportCallback,
    load_muzero_config,
)
from planner.f110_self_play.backend import ActionLattice, MuZeroSearchAdapter
from planner.f110_self_play.gym_backend import GymF110Backend
from planner.f110_self_play.self_play import SelfPlayEngine
from utils.f110_env import (
    F1TenthActionConfig,
    F1TenthObservationConfig,
    observation_dim,
    with_resampled_waypoints,
)
from utils.waypoint_view import initial_pose_from_waypoints


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/muzero/default.yaml")
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Torch device for MuZero model and native TorchScript search.",
    )
    return parser.parse_args()


def resolve_device(device_name: str):
    if device_name != "auto":
        return torch.device(device_name)  # type: ignore[attr-defined]
    return torch.device(  # type: ignore[attr-defined]
        "cuda" if torch.cuda.is_available() else "cpu"
    )


def device_name(device) -> str:
    return str(device).split(":", maxsplit=1)[0]


def main() -> None:
    faulthandler.enable(all_threads=True)

    args = parse_args()
    device = resolve_device(args.device)
    torch.set_default_device(device)  # type: ignore[attr-defined]
    print(f"MuZero device: {device_name(device)}", flush=True)
    config = load_muzero_config(args.config)
    env_section = config["env"]
    observation_section = config["observation"]
    action_section = config["action"]
    reward_section = config["reward"]
    model_section = config["model"]
    search_section = config["search"]
    self_play_section = config["self_play"]
    replay_section = config["replay"]
    training_section = config["training"]
    runtime_section = config["runtime"]

    torch.manual_seed(runtime_section["seed"])
    np.random.seed(runtime_section["seed"])

    output_dir = Path(config["output"]["dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    env_config = dict(env_section)
    centerline_csv = env_config.pop("centerline_csv")
    initial_pose = env_config.pop("initial_pose", None)
    centerline = np.loadtxt(centerline_csv, delimiter=",", skiprows=1)[:, :2]
    reset_pose = (
        np.asarray(initial_pose, dtype=np.float64)
        if initial_pose is not None
        else initial_pose_from_waypoints(centerline)
    )

    observation_config = F1TenthObservationConfig(**observation_section)
    if observation_config.include_waypoints:
        observation_config = with_resampled_waypoints(observation_config, centerline)
    obs_dim = observation_dim(observation_config)

    action_config = F1TenthActionConfig(
        velocity_min=action_section["velocity_min"],
        velocity_max=action_section["velocity_max"],
    )
    discrete_action_config = DiscreteActionConfig(
        steering_bins=action_section["steering_bins"],
        velocity_bins=action_section["velocity_bins"],
        velocity_min=action_section["velocity_min"],
        velocity_max=action_section["velocity_max"],
    )
    action_space = DiscreteActionSpace(discrete_action_config)
    action_lattice = ActionLattice(
        discrete_action_config.steering_bins,
        discrete_action_config.velocity_bins,
        discrete_action_config.velocity_min,
        discrete_action_config.velocity_max,
    )

    model = F110MuZeroNet(
        obs_dim=obs_dim,
        action_count=action_space.action_count,
        hidden_size=model_section["hidden_size"],
        trunk_size=model_section.get("trunk_size", 256),
    ).to(device)
    replay_buffer = MuZeroReplayBuffer(
        max_size=replay_section["capacity"],
        unroll_steps=replay_section["unroll_steps"],
        td_steps=replay_section["td_steps"],
        discount=self_play_section["discount"],
    )
    module = LightningMuZero(
        model=model,
        replay_buffer=replay_buffer,
        training_config=training_section,
        replay_config=replay_section,
    )

    env_fn = partial(gym.make, "f110-v0", **env_config)
    env_fns = [env_fn] * search_section["batch_size"]
    reward_params = dict(reward_section)
    reward_fns = [
        ProgressReward(**reward_params) for _ in range(search_section["batch_size"])
    ]
    for reward_fn in reward_fns:
        reward_fn.set_waypoints(centerline)
    backend = GymF110Backend(
        env_fns=env_fns,
        observation_config=observation_config,
        action_config=action_config,
        reward_fns=reward_fns,
        reset_fn=lambda: {"poses": reset_pose.copy()},
        max_episode_steps=self_play_section["rollout_steps"],
    )

    model_path = output_dir / "current_model.pt"
    scripted = torch.jit.script(model.eval()).to(device)
    scripted.save(str(model_path))
    model.train()
    print(f"MuZero scripted model: {model_path}", flush=True)

    search = MuZeroSearchAdapter(
        str(model_path),
        search_section["num_iters"],
        search_section["temperature"],
        search_section["c_puct"],
        search_section["batch_size"],
        action_lattice.action_count,
        model.hidden_size,
        0,
        device_name(device),
        search_section.get("print_metrics", False),
    )
    print("MuZero native search constructed", flush=True)
    engine = SelfPlayEngine(
        search,
        backend,
        action_lattice,
        self_play_section["discount"],
        True,
        self_play_section.get("print_metrics", False),
    )

    # Seed replay before Lightning asks the module for a dataloader.
    result = engine.generate(self_play_section["rollout_steps"])
    for trajectory in result.trajectories:
        replay_buffer.push(trajectory)

    logger = TensorBoardLogger(save_dir=output_dir, name="tensorboard")
    callbacks = [
        SelfPlayCallback(
            backend=backend,
            action_lattice=action_lattice,
            rollout_steps=self_play_section["rollout_steps"],
            num_iters=search_section["num_iters"],
            c_puct=search_section["c_puct"],
            temperature=search_section["temperature"],
            search_print_metrics=search_section.get("print_metrics", False),
            self_play_print_metrics=self_play_section.get("print_metrics", False),
            discount=self_play_section["discount"],
            device=device,
            model_path=str(output_dir / "current_model.pt"),
        ),
        TorchScriptExportCallback(str(output_dir / "checkpoints")),
        ModelCheckpoint(
            dirpath=output_dir / "checkpoints",
            filename="muzero-{epoch:04d}",
            monitor="train/loss",
            mode="min",
            save_top_k=3,
            every_n_epochs=runtime_section.get("checkpoint_every_n_epochs", 10),
        ),
    ]

    trainer = pl.Trainer(
        accelerator=device_name(device),
        max_epochs=training_section["epochs"],
        enable_progress_bar=runtime_section["progress_bar"],
        logger=logger,
        callbacks=callbacks,
        reload_dataloaders_every_n_epochs=1,
    )
    trainer.fit(module)


if __name__ == "__main__":
    main()
