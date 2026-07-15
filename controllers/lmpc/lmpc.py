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
  2. Converts the native [acceleration, steering_angle] solution into Gym's
     [target_steering_angle, target_velocity] public action by inverting its
     proportional velocity controller.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Callable

import numpy as np

from controllers.controller_base import ControlCommand, Controller, VehicleState
from utils.waypoint_utils import cumulative_arc_lengths, nearest_waypoint_index

_NATIVE_DIR = Path(__file__).resolve().parent
if str(_NATIVE_DIR) not in sys.path:
    sys.path.insert(0, str(_NATIVE_DIR))
import lmpc_native as native  # noqa: E402


def _wrap_angle(angle: float) -> float:
    return float(np.arctan2(np.sin(angle), np.cos(angle)))


class _PeriodicProgress:
    """Unwraps a periodic Frenet s (mod track_length) into a continuous
    coordinate, then rebases each LMPC iteration onto a FIXED multiple of
    track_length -- not onto wherever the current crossing sample actually
    landed. This distinction matters: gym's discrete-time finish detection
    fires slightly past the geometric line by a different amount every
    lap, so if begin_lap() rebased to the exact crossing position (an
    earlier version of this class did: `self._lap_origin =
    self._unwrapped`), each lap's own s=0 would sit at a slightly
    different PHYSICAL point on the track -- s=2 in lap 0, lap 1, and lap
    2 would each mean a different location, even though SafeSet::query()
    treats them as the same coordinate system for its KNN/convex-hull
    match, and Track::curvature(s) would be evaluated at the wrong
    physical position from lap 2 onward (a silently phase-shifted
    curvature profile). Anchoring instead to lap_index * track_length
    keeps every lap on the SAME Frenet gauge: the paper's single fixed
    parameterization tau(s) = tau(s mod L), required for D^{j-1}'s samples
    to be comparable at all. The next lap's first observed s is then
    whatever it actually is (e.g. 0.04, from the real overshoot) -- not
    forced to exactly 0 -- since forcing it would be the same gauge
    distortion in a smaller disguise.
    """

    def __init__(self, track_length: float) -> None:
        if not np.isfinite(track_length) or track_length <= 0.0:
            raise ValueError("track_length must be finite and positive")
        self._length = track_length
        self._last_s_mod = 0.0
        self._unwrapped = 0.0
        self._lap_index = 0
        self._initialized = False

    def reset(self) -> None:
        self._last_s_mod = 0.0
        self._unwrapped = 0.0
        self._lap_index = 0
        self._initialized = False

    def update(self, s_mod: float) -> float:
        """Feed one new periodic observation, return s relative to the
        current lap's fixed origin (lap_index * track_length)."""
        if not np.isfinite(s_mod):
            raise ValueError("s_mod must be finite")
        if not self._initialized:
            self._last_s_mod = s_mod
            self._unwrapped = s_mod
            self._initialized = True
        else:
            ds = s_mod - self._last_s_mod
            half_length = 0.5 * self._length
            if ds < -half_length:
                ds += self._length
            elif ds > half_length:
                ds -= self._length
            self._unwrapped += ds
            self._last_s_mod = s_mod
        return self._unwrapped - self._lap_index * self._length

    def begin_lap(self) -> None:
        """Advance to the next lap's fixed origin (lap_index += 1) -- NOT a
        rebase to the current position. See class docstring for why."""
        if not self._initialized:
            raise RuntimeError("_PeriodicProgress.begin_lap: update() was never called")
        self._lap_index += 1


class LMPCController(Controller):
    """Wraps native.NativeLMPCController with the gym <-> Frenet translation."""

    def __init__(
        self,
        centerline_csv: str,
        seed_lap_csv: str,
        dt: float,
        horizon_steps: int = 75,
        config_overrides: dict[str, Any] | None = None,
    ) -> None:
        config = native.LmpcConfig()
        config.centerline_csv_path = centerline_csv
        config.seed_lap_csv_path = seed_lap_csv
        config.dt = dt
        config.horizon_steps = horizon_steps
        # Any other LmpcConfig field by name -- the cost-term weights are
        # the usual reason (cost_to_go_weight, c_u/c_d_u, ey_slack_l1/l2;
        # the objective they weight is spelled out in
        # controllers/lmpc/include/lmpc_config.hpp). LmpcConfig is a
        # non-dynamic pybind11 class, so a mistyped name raises
        # AttributeError instead of being silently ignored.
        #
        # "vehicle_params" is special-cased to accept a dict of VehicleParams
        # field names (mu, C_Sf, C_Sr, lf, lr, h, m, I) rather than requiring
        # a whole VehicleParams object -- e.g. {"mu": 0.6} to plan against a
        # DERATED friction coefficient without touching gym's own plant
        # params. This is the principled way to compensate for the
        # nominal model overestimating cornering grip (DESIGN.md's SS5/SS6
        # discussion) until the error-dynamics regression is implemented:
        # it directly shrinks what the QP believes is achievable, rather
        # than raising c_u, which isn't part of the paper's formulation
        # for this purpose.
        for name, value in (config_overrides or {}).items():
            if name == "vehicle_params" and isinstance(value, dict):
                for vp_name, vp_value in value.items():
                    setattr(config.vehicle_params, vp_name, vp_value)
            else:
                setattr(config, name, value)
        self.config = config
        self._native = native.NativeLMPCController(config)
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
        # Same convention/algorithm as native Track::length() (cumulative
        # open arclength over the same CSV) -- not an independent second
        # definition, see _project_frenet().
        self.track_length = float(self._path_s[-1])
        self._progress = _PeriodicProgress(self.track_length)

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
        # The plant's actual current steering angle (raw sim state[2]), NOT
        # this controller's last commanded delta -- gym applies steering
        # through a 2-step delay buffer and a rate-limited PID, so the two
        # diverge. Passed to native.update() every step to anchor the FHOCP's
        # stage-0 steering-rate constraint/cost against reality instead of
        # our own last command (NativeLMPCController::update's comment has
        # the full rationale). Defaults to 0.0 -- matches the state actually
        # starting at delta=0 for every standing-start launch.
        self._raw_steering_angle: Callable[[], float] | None = None

        self.vehicle_state = VehicleState(0.0, 0.0, 0.0, 0.0)
        self._t = 0.0
        self._current_speed = 0.0
        # The last update()'s state in the NATIVE convention
        # [vx, vy, omega, epsi, s, ey] -- what the lap-as-iteration runner
        # (runs/lmpc_drive.py) records each step so a completed lap can be
        # handed straight to add_lap() without re-deriving the projection.
        self.native_state = np.zeros(6, dtype=np.float64)
        # False until control() succeeds, and again whenever it fails: the
        # native side keeps the LAST successful solve's trajectory, so
        # while the runner's fallback brake is driving,
        # predicted_horizon_xy() would otherwise render a stale horizon
        # frozen at a fixed spot on the track (measured: a closed
        # triangle-ish ghost line during fallback stretches).
        self._prediction_valid = False

    def attach_raw_velocity_state(
        self, fn: Callable[[], tuple[float, float, float]]
    ) -> None:
        """fn() -> (vx, vy, omega) in the vehicle body frame."""
        self._raw_velocity_state = fn

    def attach_raw_steering_angle(self, fn: Callable[[], float]) -> None:
        """fn() -> the plant's actual current steering angle (rad)."""
        self._raw_steering_angle = fn

    def reset(self) -> None:
        """Full-episode reset. Do NOT call this between LMPC iterations in a
        continuous multi-lap run -- it clears native control memory
        (u_prev, actual_delta, warm starts) and would reintroduce a
        standing-start restart every lap. Use begin_next_lap() instead."""
        self._native.reset()
        self.vehicle_state = VehicleState(0.0, 0.0, 0.0, 0.0)
        self._t = 0.0
        self._last_idx = -1
        self._current_speed = 0.0
        self.native_state = np.zeros(6, dtype=np.float64)
        self._prediction_valid = False
        self._progress.reset()

    def begin_next_lap(self) -> None:
        """Advance Frenet progress to the next LMPC iteration's fixed origin
        (lap_index * track_length), without resetting anything physical.

        Unlike reset(), this touches only the progress tracker: it does not
        call native.reset(), so it does not clear native u_prev,
        actual_delta, or the shifted u_warm trajectory -- the vehicle keeps
        driving through the finish line instead of restarting from rest.
        The caller must immediately call update() again on the same
        post-crossing observation so native_state is rewritten against the
        new origin (the physical state is unchanged, only which origin s is
        measured from). The resulting s is NOT forced to exactly 0 -- the
        crossing sample generally overshoots the geometric line by a small,
        lap-varying amount, and forcing it to 0 would reintroduce the same
        per-lap gauge drift _PeriodicProgress's own docstring explains
        (every lap must share ONE fixed Frenet origin, not one anchored to
        wherever each crossing sample happened to land).
        """
        self._progress.begin_lap()
        self._prediction_valid = False

    def add_lap(
        self, x_lap: np.ndarray, u_lap: np.ndarray, cost_to_go: np.ndarray
    ) -> None:
        """Append a completed closed-loop lap to the native safe set (D^j).

        x_lap is (6, T+1) in native state order, u_lap is (2, T) realized
        [a, delta], cost_to_go is (T+1,) with J_k = T - k -- the same
        conventions scripts/lmpc_collect_seed_lap.py records D^0 in.
        """
        self._native.add_lap(
            np.ascontiguousarray(x_lap, dtype=np.float64),
            np.ascontiguousarray(u_lap, dtype=np.float64),
            np.ascontiguousarray(cost_to_go, dtype=np.float64),
        )

    def update(self, vehicle_state: VehicleState, t: float | None = None) -> None:
        self.vehicle_state = vehicle_state
        if t is not None:
            self._t = t

        s, ey, epsi = self._project_frenet(vehicle_state)
        if self._raw_velocity_state is not None:
            vx, vy, omega = self._raw_velocity_state()
        else:
            vx, vy, omega = vehicle_state.speed, 0.0, 0.0
        self._current_speed = vehicle_state.speed
        x_native = np.array([vx, vy, omega, epsi, s, ey], dtype=np.float64)
        self.native_state = x_native
        actual_delta = (
            self._raw_steering_angle() if self._raw_steering_angle is not None else 0.0
        )
        self._native.update(x_native, self._t, actual_delta)

    def control(self) -> ControlCommand:
        try:
            u = self._native.control()
        except RuntimeError:
            self._prediction_valid = False
            raise
        self._prediction_valid = True
        acceleration = float(u[0])
        gain_scale = 10.0 if self._current_speed > 0.0 else 2.0
        velocity_scale = self.config.v_max if acceleration > 0.0 else -self.config.v_min
        kp = gain_scale * self.config.a_max / velocity_scale
        target_velocity = self._current_speed + acceleration / kp
        # A setpoint outside gym's own velocity range asks its P-controller
        # for an acceleration the inversion above no longer models.
        target_velocity = float(
            np.clip(target_velocity, self.config.v_min, self.config.v_max)
        )
        return ControlCommand(steering=float(u[1]), velocity=target_velocity)

    def predicted_horizon_xy(self) -> np.ndarray:
        """World-frame (x, y) of the last solve's predicted state trajectory.

        For RecedingHorizonOverlay (utils/waypoint_view.py). Converts each
        predicted (s, ey) back to world coordinates via linear interpolation
        against the same centerline project_frenet() below projects onto --
        rendering-only, so the small-angle heading interpolation near a
        possible +-pi wrap is an acceptable approximation (not used for
        control).
        """
        if not self._prediction_valid:
            # No successful solve to show (fallback braking / pre-first-solve):
            # the native buffer still holds the previous solution, and drawing
            # it would freeze a ghost horizon at a fixed spot on the track.
            return np.empty((0, 2), dtype=np.float64)
        traj = self._native.predicted_trajectory()  # kStateDim x (N+1)
        s = traj[4, :]
        ey = traj[5, :]
        # Near the end of an iteration the predicted horizon legitimately
        # runs past track_length (Track::curvature is periodic, same as
        # the native side) -- wrap back into [0, track_length) instead of
        # dropping those stages, so the rendered line actually crosses the
        # seam instead of stopping dead at the last centerline sample.
        # np.mod (not np.interp's own clamping) is what avoids the old
        # pile-up-onto-the-last-waypoint artifact: wrapped points land at
        # their true position near the start of the track, not the end.
        s = np.mod(s, self.track_length)
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
        s_mod = float(self._path_s[idx] + np.dot(delta, tangent))
        ey = float(np.dot(delta, normal))
        epsi = _wrap_angle(vehicle_state.yaw - heading)
        self._last_idx = idx
        # s_mod is the raw geometric projection (periodic on the closed
        # track); unwrap/rebase it into this LMPC iteration's own s so a
        # finish-line crossing doesn't jump the native state backward to 0
        # mid-lap -- begin_next_lap() is what advances the lap origin.
        s_lap = self._progress.update(s_mod)
        return s_lap, ey, epsi
