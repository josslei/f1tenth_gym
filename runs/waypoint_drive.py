from __future__ import annotations

from typing import Any, Final

import gymnasium as gym
import numpy as np

import f110_gym  # noqa: F401 - registers f110-v0
from controllers.controller_base import VehicleState
from controllers.lmpc import LMPCController
from controllers.pure_pursuit import DynamicLookaheadDistance, PurePursuit
from controllers.stanley import Stanley
from f110_gym.viewer import F110Viewer
from utils.waypoint_view import (
    DrivenLineOverlay,
    WaypointOverlay,
    initial_pose_from_waypoints,
)

MAP = "maps/custom/f110_gym_10/f110_gym_map.yaml"
# Generate with scripts/generate_lmpc_trajectory.py before running LMPC.
LMPC_TRAJECTORY = "outputs/lmpc_trajectories/f110_gym_centerline.txt"
# Centerline remains useful for display/fallback tooling.
CENTERLINE_CSV = "maps/custom/f110_gym_10/f110_gym_centerline.csv"
# Pure pursuit / Stanley can still use the optimized raceline waypoints.
RACELINE_CSV = "maps/custom/f110_gym_10/example_waypoints.csv"
ZOOM = 2.0  # > 1 -> Zoom out; < 1 -> Zoom in
WINDOW_WIDTH = 1000
WINDOW_HEIGHT = 800
CONTROLLER_NAME: Final[str] = "lmpc"
LAPS_TO_COMPLETE = 5
LMPC_DIAGNOSTIC_INTERVAL_STEPS = 100

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


def load_centerline_xy(csv_path: str) -> np.ndarray:
    centerline = np.loadtxt(csv_path, delimiter=",", skiprows=1, dtype=np.float64)
    centerline = np.atleast_2d(centerline)
    return centerline[:, :2]


def load_lmpc_trajectory_xy(table_path: str) -> np.ndarray:
    trajectory = np.loadtxt(table_path, dtype=np.float64)
    trajectory = np.atleast_2d(trajectory)
    return trajectory[:, :2]


def build_controller(f110_env: Any):
    if CONTROLLER_NAME == "pure_pursuit":
        lookahead_policy = DynamicLookaheadDistance(
            MIN_LOOKAHEAD, MAX_LOOKAHEAD, LOOKAHEAD_RATIO
        )
        wheelbase = float(f110_env.params["lf"] + f110_env.params["lr"])
        return PurePursuit.from_csv(
            RACELINE_CSV,
            lookahead=lookahead_policy,
            wheelbase=wheelbase,
        )
    if CONTROLLER_NAME == "stanley":
        return Stanley.from_csv(
            RACELINE_CSV,
            lf=float(f110_env.params["lf"]),
            k=STANLEY_K,
        )
    if CONTROLLER_NAME == "lmpc":
        return LMPCController.from_trajectory_table(LMPC_TRAJECTORY)
    raise ValueError(f"Unknown controller: {CONTROLLER_NAME}")


def controller_display_points(controller: Any) -> np.ndarray:
    if CONTROLLER_NAME == "lmpc":
        return load_lmpc_trajectory_xy(LMPC_TRAJECTORY)
    return controller.waypoints[:, :2]


def lmpc_diagnostics(controller: Any, lateral_errors: list[float]) -> str:
    if not isinstance(controller, LMPCController):
        return ""
    mean_abs_e_y = float(np.mean(lateral_errors)) if lateral_errors else 0.0
    max_abs_e_y = float(np.max(lateral_errors)) if lateral_errors else 0.0
    return (
        f" | lmpc completed_laps={controller.completed_laps()}"
        f" samples={controller.sample_count()}"
        f" lap_samples={controller.lap_sample_count()}"
        f" ss_points={controller.last_safe_set_points()}"
        f" mean_abs_e_y={mean_abs_e_y:.3f}"
        f" max_abs_e_y={max_abs_e_y:.3f}"
        f" solver_success={controller.solver_success_rate():.1%}"
    )


def main() -> None:
    env = gym.make("f110-v0", map=MAP, num_agents=1, laps_to_complete=LAPS_TO_COMPLETE)
    f110_env: Any = env.unwrapped
    controller = build_controller(f110_env)
    display_points = controller_display_points(controller)
    initial_pose = initial_pose_from_waypoints(display_points)
    waypoint_overlay = WaypointOverlay(display_points)
    driven_line_overlay = DrivenLineOverlay()
    viewer = F110Viewer.from_env(
        env.unwrapped,
        width=WINDOW_WIDTH,
        height=WINDOW_HEIGHT,
        target_fps=60.0,
        initial_zoom=ZOOM,
        callbacks=[waypoint_overlay, driven_line_overlay],
    )

    obs, _info = env.reset(options={"poses": initial_pose})
    previous_lap_count = int(obs["lap_counts"][0])
    previous_lap_time = 0.0
    pending_lap_logs: list[str] = []
    lateral_errors: list[float] = []
    step_count = 0

    viewer.update(obs)
    viewer.render()

    while True:
        if isinstance(controller, LMPCController):
            controller.update_from_observation(obs)
            lateral_errors.append(abs(float(controller.racing_state.e_y)))
        else:
            state = obs_to_vehicle_state(obs)
            controller.update(state)
        cmd = controller.control()
        step_count += 1
        for message in pending_lap_logs:
            print(f"{message}{lmpc_diagnostics(controller, lateral_errors)}")
        pending_lap_logs.clear()
        if (
            isinstance(controller, LMPCController)
            and step_count % LMPC_DIAGNOSTIC_INTERVAL_STEPS == 0
        ):
            print(
                f"LMPC step {step_count}{lmpc_diagnostics(controller, lateral_errors)}"
            )
        action = np.array([[cmd.steering, cmd.velocity]], dtype=np.float64)

        obs, _reward, terminated, truncated, _info = env.step(action)
        lap_count = int(obs["lap_counts"][0])
        lap_time = float(obs["lap_times"][0])
        if lap_count > previous_lap_count:
            for lap_number in range(previous_lap_count + 1, lap_count + 1):
                split_time = lap_time - previous_lap_time
                pending_lap_logs.append(
                    f"Lap {lap_number}/{LAPS_TO_COMPLETE}: "
                    f"{split_time:.3f}s (total {lap_time:.3f}s)"
                )
                previous_lap_time = lap_time
            previous_lap_count = lap_count
        viewer.update(obs)
        viewer.render()

        if terminated or truncated:
            break

    for message in pending_lap_logs:
        print(f"{message}{lmpc_diagnostics(controller, lateral_errors)}")

    while not viewer.closed:
        viewer.render()

    env.close()


if __name__ == "__main__":
    main()
