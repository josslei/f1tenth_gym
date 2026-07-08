from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np

from controllers.controller_base import ControlCommand, Controller, VehicleState
from .adapter import obs_to_gym_vehicle_state
from .binding import (
    CenterlineTrack,
    GymVehicleState,
    LmpcConfig,
    LmpcReference,
    NativeLMPCController,
)


class LMPCController(Controller):
    """Gym-facing wrapper around the native C++ LMPC implementation.

    The wrapper owns the centerline projection used to convert the generic Gym
    vehicle state into the Frenet state order consumed by Racing-LMPC.
    """

    def __init__(
        self,
        centerline_x: Sequence[float] | np.ndarray,
        centerline_y: Sequence[float] | np.ndarray,
        closed: bool = True,
        target_speed: float | None = None,
        dt: float = 0.01,
        wheelbase: float = 0.33,
        speed_s: Sequence[float] | np.ndarray | None = None,
        speed_profile: Sequence[float] | np.ndarray | None = None,
        speed_total_length: float | None = None,
        curvature_profile: Sequence[float] | np.ndarray | None = None,
        left_bound_profile: Sequence[float] | np.ndarray | None = None,
        right_bound_profile: Sequence[float] | np.ndarray | None = None,
        regression_horizon_stride: int = 0,
        horizon: int | None = None,
        max_iter: int | None = None,
        tolerance: float | None = None,
        reg_max_points: int | None = None,
    ) -> None:
        if NativeLMPCController is None:
            raise RuntimeError(
                "NativeLMPCController is not exposed by lmpc_native yet."
            )
        if LmpcConfig is None:
            raise RuntimeError("LmpcConfig is not exposed by lmpc_native yet.")
        if LmpcReference is None:
            raise RuntimeError("LmpcReference is not exposed by lmpc_native yet.")
        self.speed_s = (
            None if speed_s is None else np.asarray(speed_s, dtype=np.float64)
        )
        self.speed_profile = (
            None
            if speed_profile is None
            else np.asarray(speed_profile, dtype=np.float64)
        )
        self.speed_total_length = speed_total_length
        self.curvature_profile = (
            None
            if curvature_profile is None
            else np.asarray(curvature_profile, dtype=np.float64)
        )
        self.left_bound_profile = (
            None
            if left_bound_profile is None
            else np.asarray(left_bound_profile, dtype=np.float64)
        )
        self.right_bound_profile = (
            None
            if right_bound_profile is None
            else np.asarray(right_bound_profile, dtype=np.float64)
        )
        centerline_x_array = np.asarray(centerline_x, dtype=np.float64)
        centerline_y_array = np.asarray(centerline_y, dtype=np.float64)
        self.centerline_x = centerline_x_array
        self.centerline_y = centerline_y_array
        self.closed = closed
        self.track = CenterlineTrack(
            centerline_x_array.tolist(),
            centerline_y_array.tolist(),
            closed,
        )
        native_config = LmpcConfig()
        if target_speed is None:
            if self.speed_profile is not None:
                native_config.target_speed = max(
                    float(np.mean(self.speed_profile)),
                    0.5 * float(np.max(self.speed_profile)),
                )
        else:
            native_config.target_speed = target_speed
        native_config.dt = dt
        if horizon is not None:
            native_config.horizon = horizon
        if max_iter is not None:
            native_config.max_iter = max_iter
        if tolerance is not None:
            native_config.tolerance = tolerance
        if reg_max_points is not None:
            native_config.reg_max_points = reg_max_points
        native_config.wheelbase = wheelbase
        native_config.regression_horizon_stride = regression_horizon_stride
        native_config.track_length = (
            float(speed_total_length)
            if speed_total_length is not None
            else float(self.track.total_length())
        )
        self.native_horizon = int(native_config.horizon)
        self.native_dt = float(native_config.dt)
        self.native_controller = NativeLMPCController(native_config)
        # Give the native controller kappa(s) so it can evaluate per-stage
        # curvature at the plan's predicted s (like the upstream), instead of a
        # uniform-speed-assumed curvature sequence.
        if self.curvature_profile is not None and self.speed_s is not None:
            self.native_controller.set_curvature_profile(
                self.speed_s.tolist(),
                self.curvature_profile.tolist(),
                float(self.speed_total_length)
                if self.speed_total_length is not None
                else 0.0,
            )
        self.vehicle_state = VehicleState(0.0, 0.0, 0.0, 0.0)
        self.racing_state = self.track.to_racing_state(
            self._to_native_gym_state(self.vehicle_state, 0.0, 0.0)
        )
        self._update_native_reference()

    @classmethod
    def from_centerline_csv(
        cls,
        csv_path: str | Path,
        delimiter: str = ";",
        skiprows: int = 1,
        x_col: int = 1,
        y_col: int = 2,
        closed: bool = True,
        target_speed: float | None = None,
        dt: float = 0.01,
        wheelbase: float = 0.33,
        regression_horizon_stride: int = 0,
        horizon: int | None = None,
    ) -> LMPCController:
        centerline = np.loadtxt(
            csv_path, delimiter=delimiter, skiprows=skiprows, dtype=np.float64
        )
        centerline = np.atleast_2d(centerline)
        return cls(
            centerline[:, x_col],
            centerline[:, y_col],
            closed,
            target_speed=target_speed,
            dt=dt,
            wheelbase=wheelbase,
            regression_horizon_stride=regression_horizon_stride,
            horizon=horizon,
        )

    @classmethod
    def from_trajectory_table(
        cls,
        table_path: str | Path,
        closed: bool = True,
        target_speed: float | None = None,
        dt: float = 0.01,
        wheelbase: float = 0.33,
        regression_horizon_stride: int = 0,
        horizon: int | None = None,
        max_iter: int | None = None,
        tolerance: float | None = None,
        reg_max_points: int | None = None,
    ) -> LMPCController:
        table = np.loadtxt(table_path, dtype=np.float64)
        table = np.atleast_2d(table)
        normals = np.column_stack((-np.sin(table[:, 3]), np.cos(table[:, 3])))
        signed_left = np.sum((table[:, 9:11] - table[:, 0:2]) * normals, axis=1)
        signed_right = np.sum((table[:, 11:13] - table[:, 0:2]) * normals, axis=1)
        return cls(
            table[:, 0],
            table[:, 1],
            closed,
            target_speed=target_speed,
            dt=dt,
            wheelbase=wheelbase,
            regression_horizon_stride=regression_horizon_stride,
            speed_s=table[:, 6],
            speed_profile=table[:, 4],
            speed_total_length=float(table[0, 7]),
            curvature_profile=table[:, 5],
            left_bound_profile=np.maximum(signed_left, signed_right),
            right_bound_profile=np.maximum(-signed_left, -signed_right),
            horizon=horizon,
            max_iter=max_iter,
            tolerance=tolerance,
            reg_max_points=reg_max_points,
        )

    def reset(self) -> None:
        self.vehicle_state = VehicleState(0.0, 0.0, 0.0, 0.0)
        self.racing_state = self.track.to_racing_state(
            self._to_native_gym_state(self.vehicle_state, 0.0, 0.0)
        )
        self.native_controller.reset()
        self._update_native_reference()

    def update(
        self,
        vehicle_state: VehicleState,
        lateral_velocity: float = 0.0,
        yaw_rate: float = 0.0,
    ) -> None:
        self.vehicle_state = vehicle_state
        self.racing_state = self.track.to_racing_state(
            self._to_native_gym_state(vehicle_state, lateral_velocity, yaw_rate)
        )
        self._update_native_reference()
        self.native_controller.update(self.racing_state)

    def update_from_observation(self, obs: dict[str, Any]) -> None:
        gym_state = obs_to_gym_vehicle_state(obs)
        self.vehicle_state = VehicleState(
            gym_state.x,
            gym_state.y,
            gym_state.yaw,
            gym_state.v_x,
        )
        self.racing_state = self.track.to_racing_state(gym_state)
        self._update_native_reference()
        self.native_controller.update(self.racing_state)

    def control(self) -> ControlCommand:
        command = self.native_controller.control()
        return ControlCommand(
            steering=float(command.steering),
            velocity=float(command.velocity),
        )

    def load_initial_lap(self, csv_path: str | Path) -> int:
        """Seed the safe set (D^0) from a recorded seed-lap CSV.

        Columns are ``[lap, s, e_y, e_psi, v_x, v_y, omega, lon, delta, k, t]``
        (one row per simulator step). Each distinct ``lap`` value is added to the
        safe set as a separate historical lap. Returns the number of laps loaded.
        """
        data = np.loadtxt(csv_path, delimiter=",", skiprows=1, dtype=np.float64)
        data = np.atleast_2d(data)
        laps_loaded = 0
        for lap_id in np.unique(data[:, 0]):
            lap = data[data[:, 0] == lap_id]
            self.native_controller.add_initial_lap(
                lap[:, 1:7].tolist(),
                lap[:, 7:9].tolist(),
                lap[:, 9].tolist(),
                lap[:, 10].tolist(),
            )
            laps_loaded += 1
        return laps_loaded

    def sample_count(self) -> int:
        return int(self.native_controller.sample_count())

    def completed_laps(self) -> int:
        return int(self.native_controller.completed_laps())

    def lap_sample_count(self) -> int:
        return int(self.native_controller.lap_sample_count())

    def last_safe_set_points(self) -> int:
        return int(self.native_controller.last_safe_set_points())

    def solver_success_rate(self) -> float:
        return float(self.native_controller.solver_success_rate())

    def last_solver_status(self) -> str:
        return str(self.native_controller.last_solver_status())

    def predicted_horizon_xy(self) -> np.ndarray:
        horizon = np.asarray(
            self.native_controller.predicted_horizon(), dtype=np.float64
        )
        if horizon.size == 0:
            return np.empty((0, 2), dtype=np.float64)
        return self._frenet_to_world(horizon[:, 0], horizon[:, 1])

    def _target_speed_at_current_s(self, fallback: float) -> float:
        if self.speed_s is None or self.speed_profile is None:
            return fallback
        return self._interp_closed_profile(self.speed_profile)

    def _update_native_reference(self) -> None:
        if LmpcReference is None:
            raise RuntimeError("LmpcReference is not exposed by lmpc_native yet.")
        reference = LmpcReference()
        reference.target_speed = self._target_speed_at_current_s(reference.target_speed)
        if self.curvature_profile is not None:
            reference.curvature = self._interp_closed_profile(self.curvature_profile)
            reference.curvature_sequence = self._interp_closed_horizon(
                self.curvature_profile,
                max(self.native_horizon - 1, 0),
                reference.target_speed,
            )
        if self.left_bound_profile is not None:
            reference.left_bound = self._interp_closed_profile(self.left_bound_profile)
        if self.right_bound_profile is not None:
            reference.right_bound = self._interp_closed_profile(
                self.right_bound_profile
            )
        self.native_controller.set_reference(reference)

    def _interp_closed_horizon(
        self, profile: np.ndarray, horizon_steps: int, speed: float
    ) -> list[float]:
        assert self.speed_s is not None
        if horizon_steps == 0:
            return []
        s_offsets = self.native_dt * max(float(speed), 0.0) * np.arange(horizon_steps)
        query_s = self.racing_state.s + s_offsets
        if self.speed_total_length is None:
            return np.interp(query_s, self.speed_s, profile).astype(float).tolist()
        speed_s = np.r_[self.speed_s, self.speed_total_length]
        closed_profile = np.r_[profile, profile[0]]
        return (
            np.interp(np.mod(query_s, self.speed_total_length), speed_s, closed_profile)
            .astype(float)
            .tolist()
        )

    def _interp_closed_profile(self, profile: np.ndarray) -> float:
        assert self.speed_s is not None
        if self.speed_total_length is None:
            return float(np.interp(self.racing_state.s, self.speed_s, profile))
        speed_s = np.r_[self.speed_s, self.speed_total_length]
        closed_profile = np.r_[profile, profile[0]]
        return float(
            np.interp(
                np.mod(self.racing_state.s, self.speed_total_length),
                speed_s,
                closed_profile,
            )
        )

    def _frenet_to_world(self, s: np.ndarray, e_y: np.ndarray) -> np.ndarray:
        centerline_s = np.asarray(self.track.s(), dtype=np.float64)
        if self.closed:
            total_length = float(self.track.total_length())
            query_s = np.mod(s, total_length)
            path_s = np.r_[centerline_s, total_length]
            path_x = np.r_[self.centerline_x, self.centerline_x[0]]
            path_y = np.r_[self.centerline_y, self.centerline_y[0]]
        else:
            query_s = np.clip(s, centerline_s[0], centerline_s[-1])
            path_s = centerline_s
            path_x = self.centerline_x
            path_y = self.centerline_y

        segment_indices = np.searchsorted(path_s, query_s, side="right") - 1
        segment_indices = np.clip(segment_indices, 0, path_s.size - 2)
        next_indices = segment_indices + 1
        dx = path_x[next_indices] - path_x[segment_indices]
        dy = path_y[next_indices] - path_y[segment_indices]
        segment_lengths = path_s[next_indices] - path_s[segment_indices]
        t = (query_s - path_s[segment_indices]) / segment_lengths
        base_x = path_x[segment_indices] + t * dx
        base_y = path_y[segment_indices] + t * dy
        normal_x = -dy / segment_lengths
        normal_y = dx / segment_lengths
        return np.column_stack((base_x + e_y * normal_x, base_y + e_y * normal_y))

    def _to_native_gym_state(
        self,
        vehicle_state: VehicleState,
        lateral_velocity: float,
        yaw_rate: float,
    ) -> GymVehicleState:
        return GymVehicleState(
            vehicle_state.x,
            vehicle_state.y,
            vehicle_state.yaw,
            vehicle_state.speed,
            lateral_velocity,
            yaw_rate,
        )
