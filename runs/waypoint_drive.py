from __future__ import annotations

from typing import Any, Final

import gymnasium as gym
import numpy as np

import f110_gym  # noqa: F401 - registers f110-v0
from controllers.controller_base import VehicleState
from controllers.pure_pursuit import DynamicLookaheadDistance, PurePursuit
from controllers.stanley import Stanley
from f110_gym.viewer import F110Viewer
from utils.waypoint_view import WaypointOverlay, initial_pose_from_waypoints

MAP = "maps/f1tenth_racetracks/Spielberg/Spielberg_map"
WAYPOINTS_CSV = "outputs/waypoints/Spielberg_mintime.csv"
ZOOM = 1.0  # > 1 -> Zoom out; < 1 -> Zoom in
WINDOW_WIDTH = 1000
WINDOW_HEIGHT = 800
CONTROLLER_NAME: Final[str] = "stanley"

# Pure Pursuit hyperparameters
MIN_LOOKAHEAD = 1.0
MAX_LOOKAHEAD = 2.0
LOOKAHEAD_RATIO = 8.0

# Stanley hyperparameters
STANLEY_K = 5.0


def obs_to_vehicle_state(obs: dict[str, Any]) -> VehicleState:
    ego = int(obs["ego_idx"])
    return VehicleState(
        x=float(obs["poses_x"][ego]),
        y=float(obs["poses_y"][ego]),
        yaw=float(obs["poses_theta"][ego]),
        speed=float(obs["linear_vels_x"][ego]),
    )


def build_controller(f110_env: Any):
    if CONTROLLER_NAME == "pure_pursuit":
        lookahead_policy = DynamicLookaheadDistance(
            MIN_LOOKAHEAD, MAX_LOOKAHEAD, LOOKAHEAD_RATIO
        )
        wheelbase = float(f110_env.params["lf"] + f110_env.params["lr"])
        return PurePursuit.from_csv(
            WAYPOINTS_CSV,
            lookahead=lookahead_policy,
            wheelbase=wheelbase,
        )
    if CONTROLLER_NAME == "stanley":
        return Stanley.from_csv(
            WAYPOINTS_CSV,
            lf=float(f110_env.params["lf"]),
            k=STANLEY_K,
        )
    raise ValueError(f"Unknown controller: {CONTROLLER_NAME}")


def main() -> None:
    env = gym.make("f110-v0", map=MAP, num_agents=1)
    f110_env: Any = env.unwrapped
    controller = build_controller(f110_env)
    initial_pose = initial_pose_from_waypoints(controller.waypoints[:, :2])
    waypoint_overlay = WaypointOverlay(controller.waypoints[:, :2])
    viewer = F110Viewer.from_env(
        env.unwrapped,
        width=WINDOW_WIDTH,
        height=WINDOW_HEIGHT,
        target_fps=60.0,
        initial_zoom=ZOOM,
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
