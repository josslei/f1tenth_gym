"""Python wrapper around the native LMPC controller (lmpc_native).

Mirrors controllers.controller_base.Controller's shape (reset/update/control)
but does two things the native side deliberately does not (DESIGN.md pins the
native class to speak casadi::DM/StateIndex order end to end, no gym-specific
translation inside it):

  1. Projects the gym simulator's global (x, y, yaw) pose onto the track's
     Frenet frame (s, ey, epsi) -- the same convention (and offsets)
     scripts/lmpc_collect_seed_lap.py uses to build D^0, so the native
     controller's own s/ey/epsi state means the same thing here as it did
     when the seed lap was recorded.
  2. Converts the native [a, delta] solution back into a gym ControlCommand
     (steering, velocity) -- DESIGN.md SS4's "no separate integration step":
     the velocity command is the solved trajectory's own one-step-ahead vx,
     not an integrated a*dt.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable

import numpy as np

from controllers.controller_base import ControlCommand, Controller, VehicleState
from utils.waypoint_utils import cumulative_arc_lengths, nearest_waypoint_index

_NATIVE_DIR = Path(__file__).resolve().parent
if str(_NATIVE_DIR) not in sys.path:
    sys.path.insert(0, str(_NATIVE_DIR))
import lmpc_native as native  # noqa: E402


def _wrap_angle(angle: float) -> float:
    return float(np.arctan2(np.sin(angle), np.cos(angle)))


class LMPCController(Controller):
    """Wraps native.NativeLMPCController with the gym <-> Frenet translation."""

    def __init__(
        self,
        centerline_csv: str,
        seed_lap_csv: str,
        dt: float,
        horizon_steps: int = 75,
        v_min: float = -5.0,
    ) -> None:
        config = native.LmpcConfig()
        config.centerline_csv_path = centerline_csv
        config.seed_lap_csv_path = seed_lap_csv
        config.dt = dt
        config.horizon_steps = horizon_steps
        self.config = config
        self._native = native.NativeLMPCController(config)
        # gym's own DEFAULT_PARAMS "v_min" (gym/f110_gym/envs/f110_env.py) --
        # not tracked by LmpcConfig itself (only needed here, to invert
        # gym's braking-side P-controller gain in control() below; the QP
        # has no vx box constraint to enforce it against, DESIGN.md SS3).
        self._v_min = v_min

        # Same centerline convention as
        # scripts/lmpc_collect_seed_lap.py::load_centerline_waypoints: heading
        # is the forward-difference tangent between consecutive (closed-loop,
        # wrapped) points, stored offset by -pi/2 so it matches the raceline
        # CSVs' on-disk convention; project_frenet() below adds the same
        # +pi/2 back.
        xy = np.loadtxt(centerline_csv, delimiter=",", skiprows=1, dtype=np.float64)[
            :, :2
        ]
        next_xy = np.roll(xy, -1, axis=0)
        heading = np.arctan2(next_xy[:, 1] - xy[:, 1], next_xy[:, 0] - xy[:, 0])
        self.waypoints = (
            xy  # (N, 2) -- for WaypointOverlay / initial_pose_from_waypoints
        )
        self._path_xy = xy
        self._path_psi = heading - np.pi / 2
        self._path_s = cumulative_arc_lengths(xy)
        self._last_idx = -1

        # A fixed index-count search window silently becomes physically
        # narrower as waypoint density increases (measured directly:
        # 200 points covered ~20m at a ~0.10m-spaced centerline but only
        # ~4m once regenerated at ~0.02m spacing, losing track entirely --
        # same fix as controllers/stanley.py's search_window, kept
        # consistent with scripts/lmpc_collect_seed_lap.py's own copy).
        search_window_meters = 20.0
        avg_spacing = float(np.linalg.norm(np.diff(xy, axis=0), axis=1).mean())
        self._search_window = max(
            10, int(round(search_window_meters / max(avg_spacing, 1e-6)))
        )

        # gym's public obs dict hardcodes linear_vels_y to 0.0 regardless of
        # actual slip (gym/f110_gym/envs/base_classes.py) -- vy/omega have to
        # come from the simulator's raw internal state instead
        # (scripts/lmpc_collect_seed_lap.py's own comment on this). The
        # runner script wires this in, since only it has direct access to
        # f110_env.sim.agents[...].state.
        self._raw_velocity_state: Callable[[], tuple[float, float, float]] | None = None

        self.vehicle_state = VehicleState(0.0, 0.0, 0.0, 0.0)
        self._t = 0.0
        self._current_speed = 0.0  # sqrt(vx^2+vy^2) -- see control()'s comment

    def attach_raw_velocity_state(
        self, fn: Callable[[], tuple[float, float, float]]
    ) -> None:
        """fn() -> (vx, vy, omega) in the vehicle body frame."""
        self._raw_velocity_state = fn

    def reset(self) -> None:
        self._native.reset()
        self.vehicle_state = VehicleState(0.0, 0.0, 0.0, 0.0)
        self._t = 0.0
        self._last_idx = -1
        self._current_speed = 0.0

    def update(self, vehicle_state: VehicleState, t: float | None = None) -> None:
        self.vehicle_state = vehicle_state
        if t is not None:
            self._t = t

        s, ey, epsi = self._project_frenet(vehicle_state)
        if self._raw_velocity_state is not None:
            vx, vy, omega = self._raw_velocity_state()
        else:
            vx, vy, omega = vehicle_state.speed, 0.0, 0.0
        # vx = v*cos(beta), vy = v*sin(beta) (attach_raw_velocity_state's own
        # convention) -- their magnitude recovers v exactly, the same "current
        # speed" gym/f110_gym/envs/dynamic_models.py::pid() measures against,
        # needed by control() below to invert that same controller.
        self._current_speed = float(np.hypot(vx, vy))

        x_native = np.array([vx, vy, omega, epsi, s, ey], dtype=np.float64)
        self._native.update(x_native, self._t)

    def control(self) -> ControlCommand:
        u = self._native.control()
        a_solved = float(u[0])  # StateIndex::A -- this solve's own [a, delta]

        # gym's own low-level controller (dynamic_models.py::pid) does NOT
        # integrate a velocity command directly -- it treats `velocity` as a
        # SETPOINT its own P-controller chases: accl = kp*(velocity -
        # current_speed), kp = 10*a_max/v_max while accelerating (or
        # 10*a_max/(-v_min) while braking). Sending predicted_next_state()[0]
        # (this solve's own one-step-ahead vx under the LMPC's DIRECT-a
        # dynamics model) as that setpoint massively undershoots what's
        # needed: at dt=0.025 the setpoint gap it implies is only a*dt, but
        # gym's controller needs a gap of a/kp to realize that same a -- a
        # factor of kp*dt (~0.12 at this track's params) too small. Measured
        # directly (2026-07-13): the QP repeatedly solved a~=6-9 m/s^2 near
        # launch, but real speed only reached 0.05 m/s after 4 control steps
        # (0.1s) -- an order of magnitude short of what a~=6-9 m/s^2 implies,
        # and the resulting model/reality gap compounded every step until
        # the QP's own solve broke. Fixed by inverting gym's P-controller:
        # choose velocity so gym's OWN kp*(velocity-current_speed) reproduces
        # a_solved, instead of however small a gap the one-step model
        # integration happens to produce.
        kp = (
            10.0
            * self.config.a_max
            / (self.config.v_max if a_solved >= 0.0 else -self._v_min)
        )
        vx_cmd = self._current_speed + a_solved / kp

        return ControlCommand(steering=float(u[1]), velocity=vx_cmd)

    def predicted_horizon_xy(self) -> np.ndarray:
        """World-frame (x, y) of the last solve's predicted state trajectory.

        For RecedingHorizonOverlay (utils/waypoint_view.py). Converts each
        predicted (s, ey) back to world coordinates via linear interpolation
        against the same centerline project_frenet() below projects onto --
        rendering-only, so the small-angle heading interpolation near a
        possible +-pi wrap is an acceptable approximation (not used for
        control).
        """
        traj = self._native.predicted_trajectory()  # kStateDim x (N+1)
        s = traj[4, :]
        ey = traj[5, :]
        x = np.interp(s, self._path_s, self._path_xy[:, 0])
        y = np.interp(s, self._path_s, self._path_xy[:, 1])
        heading = np.interp(s, self._path_s, self._path_psi) + np.pi / 2
        normal_x = -np.sin(heading)
        normal_y = np.cos(heading)
        return np.column_stack([x + ey * normal_x, y + ey * normal_y])

    def _project_frenet(
        self, vehicle_state: VehicleState
    ) -> tuple[float, float, float]:
        position = np.array([vehicle_state.x, vehicle_state.y])
        idx = nearest_waypoint_index(
            self._path_xy, position, self._last_idx, search_window=self._search_window
        )
        # Same +pi/2 correction as scripts/lmpc_collect_seed_lap.py's
        # project_frenet(): the stored psi is offset -pi/2 from true heading.
        heading = float(self._path_psi[idx]) + np.pi / 2
        tangent = np.array([np.cos(heading), np.sin(heading)])
        normal = np.array([-np.sin(heading), np.cos(heading)])
        delta = position - self._path_xy[idx]
        s = float(self._path_s[idx] + np.dot(delta, tangent))
        ey = float(np.dot(delta, normal))
        epsi = _wrap_angle(vehicle_state.yaw - heading)
        self._last_idx = idx
        return s, ey, epsi
