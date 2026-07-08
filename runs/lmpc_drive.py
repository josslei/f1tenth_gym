from __future__ import annotations


import gymnasium as gym
import numpy as np

import f110_gym  # noqa: F401 - registers f110-v0
from controllers.lmpc import LMPCController
from f110_gym.viewer import F110Viewer
from utils.waypoint_view import (
    DrivenLineOverlay,
    RecedingHorizonOverlay,
    WaypointOverlay,
    initial_pose_from_waypoints,
)

MAP = "maps/custom/f110_gym_10/f110_gym_map.yaml"
# Generate with scripts/generate_lmpc_trajectory.py before running LMPC.
LMPC_TRAJECTORY = "outputs/lmpc_trajectories/f110_gym_centerline.txt"
# Initial safe set (D^0). Generate with scripts/generate_lmpc_seed_lap.py before
# running LMPC; without it the controller has no cost-to-go and will not drive.
LMPC_SEED_LAP = "outputs/lmpc_seed_laps/f110_gym_seed.csv"
# Centerline remains useful for display/fallback tooling.
CENTERLINE_CSV = "maps/custom/f110_gym_10/f110_gym_centerline.csv"
ZOOM = 1.0  # > 1 -> Zoom out; < 1 -> Zoom in
WINDOW_WIDTH = 1000
WINDOW_HEIGHT = 800
LAPS_TO_COMPLETE = 5
DIAGNOSTIC_INTERVAL_STEPS = 100

# Shared physics/control timestep for both the Gym simulator and the LMPC
# controller's own discretization -- must match or the ATV model is linearized
# at a dt different from what the sim actually steps at.
SIM_DT = 0.025

# --- LMPC tuning: read here in Python, passed through to the C++ backend ---
# Halved from 150 for now; lengthen again once a seed lap gives the terminal
# cost-to-go that lets a short horizon stay on the (sub)optimal line.
HORIZON_STEPS = 75
# qrqp active-set iteration cap. Cheap now (convex QP converges fast), but
# load-bearing once the safe set makes the terminal QP degenerate.
SOLVER_MAX_ITER = 100
# qrqp primal/dual feasibility tolerances (constr_viol_tol / dual_inf_tol).
SOLVER_TOLERANCE = 1e-6
# Terminal safe-set size; shrink for speed after a seed lap exists.
REG_MAX_POINTS = 24


def load_lmpc_trajectory_xy(table_path: str) -> np.ndarray:
    trajectory = np.loadtxt(table_path, dtype=np.float64)
    trajectory = np.atleast_2d(trajectory)
    return trajectory[:, :2]


def build_controller() -> LMPCController:
    return LMPCController.from_trajectory_table(
        LMPC_TRAJECTORY,
        horizon=HORIZON_STEPS,
        dt=SIM_DT,
        max_iter=SOLVER_MAX_ITER,
        tolerance=SOLVER_TOLERANCE,
        reg_max_points=REG_MAX_POINTS,
    )


def lmpc_diagnostics(controller: LMPCController, lateral_errors: list[float]) -> str:
    mean_abs_e_y = float(np.mean(lateral_errors)) if lateral_errors else 0.0
    max_abs_e_y = float(np.max(lateral_errors)) if lateral_errors else 0.0
    return (
        f" | lmpc completed_laps={controller.completed_laps()}"
        f" samples={controller.sample_count()}"
        f" lap_samples={controller.lap_sample_count()}"
        f" ss_points={controller.last_safe_set_points()}"
        f" horizon_points={controller.predicted_horizon_xy().shape[0]}"
        f" mean_abs_e_y={mean_abs_e_y:.3f}"
        f" max_abs_e_y={max_abs_e_y:.3f}"
        f" solver_success={controller.solver_success_rate():.1%}"
        f" last_status='{controller.last_solver_status()}'"
    )


def main() -> None:
    env = gym.make(
        "f110-v0",
        map=MAP,
        num_agents=1,
        laps_to_complete=LAPS_TO_COMPLETE,
        timestep=SIM_DT,
    )
    controller = build_controller()
    seed_laps = controller.load_initial_lap(LMPC_SEED_LAP)
    print(f"Loaded {seed_laps} seed lap(s) into the LMPC safe set.")
    display_points = load_lmpc_trajectory_xy(LMPC_TRAJECTORY)
    initial_pose = initial_pose_from_waypoints(display_points)
    waypoint_overlay = WaypointOverlay(display_points)
    driven_line_overlay = DrivenLineOverlay()
    callbacks = [
        waypoint_overlay,
        driven_line_overlay,
        RecedingHorizonOverlay(controller),
    ]
    viewer = F110Viewer.from_env(
        env.unwrapped,
        width=WINDOW_WIDTH,
        height=WINDOW_HEIGHT,
        target_fps=60.0,
        initial_zoom=ZOOM,
        callbacks=callbacks,
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
        controller.update_from_observation(obs)
        lateral_errors.append(abs(float(controller.racing_state.e_y)))
        cmd = controller.control()
        step_count += 1
        for message in pending_lap_logs:
            print(f"{message}{lmpc_diagnostics(controller, lateral_errors)}")
        pending_lap_logs.clear()
        if step_count % DIAGNOSTIC_INTERVAL_STEPS == 0:
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
