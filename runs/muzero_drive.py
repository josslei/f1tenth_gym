"""Drive the F1TENTH car with a trained MuZero policy."""

# pyright: reportAttributeAccessIssue=none, reportArgumentType=none, reportCallIssue=none

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import torch

import f110_gym  # noqa: F401 - registers f110-v0
from f110_gym.viewer import F110Viewer
from planner.f110_self_play.backend import ActionLattice, MuZeroSearchAdapter
from utils.f110_env import (
    F1TenthObservationConfig,
    add_control_state,
    obs_tensor as build_obs_tensor,
    observation_dim,
    with_resampled_waypoints,
)
from utils.waypoint_view import WaypointOverlay, initial_pose_from_waypoints

from models.muzero import load_muzero_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/muzero/default.yaml")
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="TorchScript MuZero checkpoint to load.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Torch device for the scripted MuZero model and native search.",
    )
    return parser.parse_args()


def resolve_device(device_name: str) -> Any:
    if device_name != "auto":
        return torch.device(device_name)  # type: ignore[attr-defined]
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")  # type: ignore[attr-defined]


def device_name(device: Any) -> str:
    return str(device).split(":", maxsplit=1)[0]


def action_bins_from_config(
    action_section: dict, env_config: dict
) -> tuple[np.ndarray, np.ndarray]:
    params = dict(env_config.get("params", {}))
    params.setdefault("s_min", -0.4189)
    params.setdefault("s_max", 0.4189)

    steering_config: Any = action_section["steering_bins"]
    velocity_config: Any = action_section["velocity_bins"]

    steering_bins = (
        np.linspace(
            float(params["s_min"]),
            float(params["s_max"]),
            int(steering_config),
            dtype=np.float32,
        )
        if np.isscalar(steering_config)
        else np.asarray(steering_config, dtype=np.float32)
    )
    velocity_bins = (
        np.linspace(
            float(action_section.get("velocity_min", 0.0)),
            float(action_section.get("velocity_max", 8.0)),
            int(velocity_config),
            dtype=np.float32,
        )
        if np.isscalar(velocity_config)
        else np.asarray(velocity_config, dtype=np.float32)
    )
    return steering_bins, velocity_bins


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)

    config = load_muzero_config(args.config)
    env_section = dict(config["env"])
    observation_section = config["observation"]
    action_section = config["action"]
    model_section = config["model"]
    search_section = config["search"]

    env_config = dict(env_section)
    centerline_csv = Path(env_config["centerline_csv"])
    centerline = np.loadtxt(centerline_csv, delimiter=",", skiprows=1)[:, :2]

    observation_config = F1TenthObservationConfig(**observation_section)
    if observation_config.include_waypoints:
        observation_config = with_resampled_waypoints(observation_config, centerline)

    steering_bins, velocity_bins = action_bins_from_config(action_section, env_config)
    action_lattice = ActionLattice(steering_bins, velocity_bins)

    checkpoint = (
        Path(args.checkpoint)
        if args.checkpoint
        else Path(config["output"]["dir"]) / "current_model.pt"
    )

    print(f"MuZero device: {device_name(device)}", flush=True)
    print(f"MuZero checkpoint: {checkpoint}", flush=True)
    print(f"MuZero obs dim: {observation_dim(observation_config)}", flush=True)

    search: Any = MuZeroSearchAdapter(
        str(checkpoint),
        search_section["num_iters"],
        0.0,
        search_section["c_puct"],
        batch_size=1,
        action_count=action_lattice.action_count,
        hidden_size=model_section["hidden_size"],
        device=device_name(device),
        print_metrics=search_section.get("print_metrics", False),
        dirichlet_alpha=0.0,
        dirichlet_epsilon=0.0,
    )

    callbacks = [WaypointOverlay(centerline)]
    env = gym.make("f110-v0", **env_config)
    viewer = F110Viewer.from_env(
        env.unwrapped,
        width=1000,
        height=800,
        target_fps=60.0,
        initial_zoom=1.0,
        callbacks=callbacks,
    )

    initial_pose = initial_pose_from_waypoints(centerline)
    obs, _info = env.reset(options={"poses": initial_pose})
    obs = add_control_state(obs, env)
    prev_action = np.zeros(2, dtype=np.float64)
    obs["prev_action"] = prev_action

    viewer.update(obs)
    viewer.render()

    while True:
        obs["prev_action"] = prev_action
        search_obs = build_obs_tensor(obs, observation_config)
        search_result = search.search_batch(search_obs)
        action_index = int(
            search_result.action_probs.detach().cpu().numpy()[0].argmax()
        )
        action = action_lattice.normalized_action(action_index).astype(np.float64)
        action_batch = np.array([action], dtype=np.float64)

        obs, _reward, terminated, truncated, _info = env.step(action_batch)
        obs = add_control_state(obs, env)
        prev_action = action

        viewer.update(obs)
        viewer.render()

        if terminated or truncated:
            break

    while not viewer.closed:
        viewer.render()

    env.close()


if __name__ == "__main__":
    main()
