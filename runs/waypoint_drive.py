"""Drive a car around the track using PurePursuit."""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np

import f110_gym  # noqa: F401 - registers f110-v0
from controllers.controller_base import VehicleState
from controllers.pure_pursuit import PurePursuit
from f110_gym.viewer import F110Viewer
from utils.waypoint_view import WaypointOverlay, initial_pose_from_waypoints

MAP = "berlin"
WAYPOINTS_CSV = "outputs/waypoints/berlin.csv"
LOOKAHEAD = 1.0
WINDOW_WIDTH = 1000
WINDOW_HEIGHT = 800


def obs_to_vehicle_state(obs: dict[str, Any]) -> VehicleState:
    ego = int(obs["ego_idx"])
    return VehicleState(
        x=float(obs["poses_x"][ego]),
        y=float(obs["poses_y"][ego]),
        yaw=float(obs["poses_theta"][ego]),
        speed=float(obs["linear_vels_x"][ego]),
    )


def main() -> None:
    env = gym.make("f110-v0", map=MAP, num_agents=1)
    f110_env: Any = env.unwrapped
    wheelbase = float(f110_env.params["lf"] + f110_env.params["lr"])
    controller = PurePursuit.from_csv(
        WAYPOINTS_CSV,
        lookahead=LOOKAHEAD,
        wheelbase=wheelbase,
    )
    initial_pose = initial_pose_from_waypoints(controller.waypoints)
    waypoint_overlay = WaypointOverlay(controller.waypoints)
    viewer = F110Viewer.from_env(
        env.unwrapped,
        width=WINDOW_WIDTH,
        height=WINDOW_HEIGHT,
        target_fps=60.0,
        callbacks=[waypoint_overlay],
    )

    obs, _info = env.reset(options={"poses": initial_pose})

    viewer.update(obs)
    viewer.render()

    while True:
        state = obs_to_vehicle_state(obs)
        controller.update(state)
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
