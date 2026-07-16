"""Adapt the byte-identical LearningMPC core to this project's controller API."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Callable

import numpy as np
import yaml
from PIL import Image

from controllers.controller_base import ControlCommand, Controller, VehicleState
from utils.waypoint_utils import closed_path_length, cumulative_arc_lengths

_NATIVE_DIR = Path(__file__).resolve().parent
if str(_NATIVE_DIR) not in sys.path:
    sys.path.insert(0, str(_NATIVE_DIR))
import lmpc_native as native  # noqa: E402


def _load_occupancy_grid(
    map_png: Path, map_yaml: Path
) -> tuple[np.ndarray, int, int, float, float, float]:
    with map_yaml.open() as stream:
        metadata = yaml.safe_load(stream)
    source_image = Image.open(map_png).convert("L")
    width, height = source_image.size
    image: np.ndarray = np.array(source_image, dtype=np.float64)
    occupancy = (255.0 - image) / 255.0
    grid = np.full(image.shape, -1, dtype=np.int8)
    grid[occupancy > metadata["occupied_thresh"]] = 100
    grid[occupancy < metadata["free_thresh"]] = 0
    grid = np.flipud(grid)
    origin_x = metadata["origin"][0]
    origin_y = metadata["origin"][1]
    return (
        grid.ravel(),
        width,
        height,
        float(metadata["resolution"]),
        float(origin_x),
        float(origin_y),
    )


def _write_waypoints(centerline_csv: Path, output_csv: Path) -> np.ndarray:
    waypoints = np.loadtxt(centerline_csv, delimiter=",", skiprows=1, dtype=np.float64)[
        :, :2
    ]
    np.savetxt(output_csv, waypoints, delimiter=",", fmt="%.10f")
    return waypoints


def _write_initial_safe_set(
    seed_lap_csv: Path, waypoints: np.ndarray, output_csv: Path
) -> tuple[float, float, float]:
    data = np.genfromtxt(
        seed_lap_csv,
        delimiter=",",
        skip_header=1,
        dtype=np.float64,
        filling_values=np.nan,
    )
    vx, vy, omega, epsi, s, ey = (
        data[:, 0],
        data[:, 1],
        data[:, 2],
        data[:, 3],
        data[:, 4],
        data[:, 5],
    )
    sample_time, acceleration, steering = data[:, 6], data[:, 7], data[:, 8]

    path_s = cumulative_arc_lengths(waypoints)
    track_length = closed_path_length(waypoints)
    closed_s = np.append(path_s, track_length)
    closed_xy = np.vstack([waypoints, waypoints[0]])
    next_xy = np.roll(waypoints, -1, axis=0)
    headings = np.arctan2(
        next_xy[:, 1] - waypoints[:, 1], next_xy[:, 0] - waypoints[:, 0]
    )

    wrapped_s = np.mod(s, track_length)
    path_x = np.interp(wrapped_s, closed_s, closed_xy[:, 0])
    path_y = np.interp(wrapped_s, closed_s, closed_xy[:, 1])
    segment = np.clip(
        np.searchsorted(closed_s, wrapped_s, side="right") - 1,
        0,
        len(headings) - 1,
    )
    path_yaw = headings[segment]
    x = path_x - ey * np.sin(path_yaw)
    y = path_y + ey * np.cos(path_yaw)
    yaw = path_yaw + epsi
    speed = np.hypot(vx, vy)
    slip = np.arctan2(vy, vx)

    acceleration = acceleration.copy()
    steering = steering.copy()
    acceleration[-1] = acceleration[-2]
    steering[-1] = steering[-2]
    rows = np.column_stack(
        [sample_time, x, y, yaw, speed, omega, slip, acceleration, steering, s]
    )

    # The copied controller starts at iteration two and therefore expects two
    # initial trajectories. Repeating D0 provides the same feasible trajectory
    # in both slots without changing controller code.
    np.savetxt(output_csv, np.vstack([rows, rows]), delimiter=",", fmt="%.10f")
    return float(x[0]), float(y[0]), float(yaw[0])


class LMPCController(Controller):
    """Expose LearningMPC through the existing reset/update/control surface.

    ``ControlCommand.velocity`` carries acceleration for this controller.
    Callers must use Gym's ``direct_accel_control=True`` mode.
    """

    def __init__(
        self,
        centerline_csv: str,
        seed_lap_csv: str,
        dt: float,
        horizon_steps: int = 75,
        config_overrides: dict[str, Any] | None = None,
    ) -> None:
        centerline_path = Path(centerline_csv)
        map_name = centerline_path.name.removesuffix("_centerline.csv") + "_map"
        map_stem = centerline_path.with_name(map_name)
        work_dir = Path("outputs/lmpc_adapter") / map_name.removesuffix("_map")
        work_dir.mkdir(parents=True, exist_ok=True)

        grid, width, height, resolution, origin_x, origin_y = _load_occupancy_grid(
            map_stem.with_suffix(".png"), map_stem.with_suffix(".yaml")
        )
        waypoint_csv = work_dir / "waypoints.csv"
        self.waypoints = _write_waypoints(centerline_path, waypoint_csv)
        initial_safe_set_csv = work_dir / "initial_safe_set.csv"
        x0, y0, yaw0 = _write_initial_safe_set(
            Path(seed_lap_csv), self.waypoints, initial_safe_set_csv
        )

        config = native.LmpcConfig()
        config.dt = dt
        config.horizon_steps = horizon_steps
        config.centerline_csv_path = centerline_csv
        config.seed_lap_csv_path = seed_lap_csv
        config.occupancy_grid = grid.tolist()
        config.map_width = width
        config.map_height = height
        config.map_resolution = resolution
        config.map_origin_x = origin_x
        config.map_origin_y = origin_y
        config.reference_waypoint_csv_path = str(waypoint_csv)
        config.reference_seed_lap_csv_path = str(initial_safe_set_csv)
        config.initial_x = x0
        config.initial_y = y0
        config.initial_yaw = yaw0
        aliases = {
            "K_NEAR": "K",
            "ACCELERATION_MAX": "a_max",
            "DECELERATION_MAX": "a_min",
            "SPEED_MAX": "v_max",
            "STEER_MAX": "delta_max",
            "VEL_THRESHOLD": "velocity_threshold",
            "MAP_MARGIN": "map_margin",
            "WAYPOINT_SPACE": "waypoint_space",
            "q_s": "ey_slack_l2",
            "q_s_terminal": "terminal_slack_weight",
        }
        for name, value in (config_overrides or {}).items():
            if name == "vehicle_params":
                for vehicle_name, vehicle_value in value.items():
                    setattr(config.vehicle_params, vehicle_name, vehicle_value)
            else:
                mapped_name = aliases.get(name, name)
                mapped_value = -value if name == "DECELERATION_MAX" else value
                setattr(config, mapped_name, mapped_value)

        self._native = native.NativeLMPCController(config)
        self.track_length = closed_path_length(self.waypoints)
        self.vehicle_state = VehicleState(x0, y0, yaw0, 0.0)
        self.native_state = np.zeros(6, dtype=np.float64)
        self._raw_velocity_state: Callable[[], tuple[float, float, float]] | None = None
        self._last_solve_ok = True
        self._prediction_valid = False
        self._raw_steering_angle: Callable[[], float] | None = None

    def attach_raw_velocity_state(
        self, fn: Callable[[], tuple[float, float, float]]
    ) -> None:
        self._raw_velocity_state = fn

    def attach_raw_steering_angle(self, fn: Callable[[], float]) -> None:
        self._raw_steering_angle = fn

    def reset(self) -> None:
        self._native.reset()

    def begin_next_lap(self) -> None:
        # LMPCCore detects the seam and advances its own iteration in step().
        pass

    def add_lap(
        self, x_lap: np.ndarray, u_lap: np.ndarray, cost_to_go: np.ndarray
    ) -> None:
        # LMPCCore records and appends each completed trajectory internally.
        del x_lap, u_lap, cost_to_go

    def update(self, vehicle_state: VehicleState, t: float | None = None) -> None:
        self.vehicle_state = vehicle_state
        if self._raw_velocity_state is None:
            vx, vy, yaw_rate = vehicle_state.speed, 0.0, 0.0
        else:
            vx, vy, yaw_rate = self._raw_velocity_state()
        speed = float(np.hypot(vx, vy))
        slip = float(np.arctan2(vy, vx))
        self.native_state = np.array(
            [
                vehicle_state.x,
                vehicle_state.y,
                vehicle_state.yaw,
                speed,
                yaw_rate,
                slip,
            ],
            dtype=np.float64,
        )
        self._native.update(
            self.native_state,
            0.0 if t is None else t,
            0.0 if self._raw_steering_angle is None else self._raw_steering_angle(),
        )

    def control(self) -> ControlCommand:
        acceleration, steering = self._native.control()
        self._last_solve_ok = bool(self._native.last_solve_ok())
        self._prediction_valid = self._last_solve_ok
        return ControlCommand(steering=float(steering), velocity=float(acceleration))

    def last_solve_ok(self) -> bool:
        return self._last_solve_ok

    def last_timings(self) -> dict[str, float]:
        timings = self._native.last_timings()
        return {
            "rollout+lin": timings.rollout_lin_ms,
            "knn": timings.knn_ms,
            "set-params": timings.set_params_ms,
            "solver": timings.solver_ms,
            "postcheck": timings.postcheck_ms,
        }

    def last_terminal_slack(self) -> np.ndarray:
        return self._native.last_terminal_slack()

    def predicted_horizon_xy(self) -> np.ndarray:
        if not self._prediction_valid:
            return np.empty((0, 2), dtype=np.float64)
        prediction = np.asarray(self._native.predicted_trajectory(), dtype=np.float64)
        return prediction[:2, :].T
