"""Drive the F1TENTH car using a trained PPO controller.

Modelled after ``waypoint_drive.py``, but replaces the classical
waypoint-based controller with a neural network policy loaded from a
training checkpoint.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np

import f110_gym  # noqa: F401 - registers f110-v0
from controllers import PPOController
from controllers.controller_base import VehicleState
from f110_gym.viewer import F110Viewer
from utils.waypoint_view import initial_pose_from_waypoints

# ── Configuration ──────────────────────────────────────────────────────────────

MAP = "tracks/Spielberg/Spielberg_map"
CHECKPOINT = "outputs/rl/ppo_spielberg/final_model.pt"
ZOOM = 2.0
WINDOW_WIDTH = 1000
WINDOW_HEIGHT = 800


# ── Helpers ────────────────────────────────────────────────────────────────────


def obs_to_vehicle_state(obs: dict[str, Any]) -> VehicleState:
    ego = int(obs["ego_idx"])
    return VehicleState(
        x=float(obs["poses_x"][ego]),
        y=float(obs["poses_y"][ego]),
        yaw=float(obs["poses_theta"][ego]),
        speed=float(obs["linear_vels_x"][ego]),
    )


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    env = gym.make("f110-v0", map=MAP, num_agents=1)
    controller = PPOController.from_checkpoint(CHECKPOINT)

    viewer = F110Viewer.from_env(
        env.unwrapped,
        width=WINDOW_WIDTH,
        height=WINDOW_HEIGHT,
        target_fps=60.0,
        initial_zoom=ZOOM,
    )

    # Derive initial pose from the track centerline (mirrors waypoint_drive.py)
    map_path = Path(MAP)
    centerline_csv = map_path.parent / f"{map_path.parent.name}_centerline.csv"
    waypoints = np.loadtxt(centerline_csv, delimiter=",", skiprows=1)
    initial_pose = initial_pose_from_waypoints(waypoints[:, :2])

    obs, _info = env.reset(options={"poses": initial_pose})

    viewer.update(obs)
    viewer.render()

    while True:
        state = obs_to_vehicle_state(obs)
        controller.update(state, obs=obs)
        cmd = controller.control()
        action = np.array([[cmd.steering, cmd.velocity]], dtype=np.float64)

        obs, _reward, terminated, truncated, _info = env.step(action)
        viewer.update(obs)
        viewer.render()

        if terminated or truncated:
            break

    while not viewer.closed:
        viewer.render()

    env.close()


if __name__ == "__main__":
    main()
