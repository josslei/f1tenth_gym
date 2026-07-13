import numpy as np
from pathlib import Path
from numba import njit
from typing import Optional

from .controller_base import Controller, VehicleState, ControlCommand
from utils.waypoint_utils import _nearest_waypoint_index

DELTA_MAX: float = 0.4189
DEFAULT_K: float = 1
# A fixed INDEX-count window (the old SEARCH_WINDOW=200 constant) silently
# becomes physically narrower as waypoint density increases -- measured
# directly: 200 points covered ~20m at a ~0.10m-spaced centerline, but only
# ~4m once that same centerline was regenerated at ~0.02m spacing (5x
# denser), causing nearest_waypoint_index's search to lose track and the
# whole closed loop to diverge. Deriving the window from each Stanley
# instance's own actual waypoint spacing (below) keeps the PHYSICAL search
# radius constant regardless of how densely any given track's waypoints
# happen to be sampled.
SEARCH_WINDOW_METERS: float = 20.0
V_SOFT: float = 1.0


class Stanley(Controller):
    def __init__(self, waypoints: np.ndarray, l_f: float, k: float = DEFAULT_K) -> None:
        self.vehicle_state = VehicleState(0, 0, 0, 0)
        self.waypoints = waypoints
        self.l_f = l_f
        self.k = k

        # Index of the goal waypoint returned by the previous control step.
        # Used as a search hint for nearest-waypoint lookup on the next step
        # so we only scan a local window instead of the full track.
        self.last_idx: Optional[int] = None

        self.num_waypoints = waypoints.shape[0]
        if self.num_waypoints == 0:
            raise ValueError("waypoints must not be empty")

        # SEARCH_WINDOW_METERS's comment has the rationale: derived from
        # THIS instance's own waypoint spacing, not a fixed index count, so
        # the actual physical search radius stays ~SEARCH_WINDOW_METERS
        # regardless of how densely these particular waypoints are sampled.
        if self.num_waypoints >= 2:
            xy = waypoints[:, :2]
            avg_spacing = float(np.linalg.norm(np.diff(xy, axis=0), axis=1).mean())
        else:
            avg_spacing = 1.0
        self.search_window = max(
            10, int(round(SEARCH_WINDOW_METERS / max(avg_spacing, 1e-6)))
        )

    @classmethod
    def from_csv(
        cls, csv_path: str | Path, lf: float, k: float = DEFAULT_K
    ) -> "Stanley":
        """Construct a controller from a semicolon-delimited waypoint CSV.

        The CSV is expected to have columns s_m; x_m; y_m; psi_rad;
        kappa_radpm; vx_mps; ax_mps2 (with a header row). Uses x_m (col 1),
        y_m (col 2), psi_rad (col 3), and vx_mps (col 5).
        """
        waypoints = np.loadtxt(csv_path, delimiter=";", skiprows=1, dtype=np.float64)
        waypoints = np.atleast_2d(waypoints)
        return cls(waypoints[:, [1, 2, 3, 5]], lf, k)

    def reset(self) -> None:
        """Reset internal state to origin and clear the cached waypoint index."""
        self.vehicle_state = VehicleState(0, 0, 0, 0)
        self.last_idx = None

    def update(self, vehicle_state: VehicleState) -> None:
        """Receive the latest vehicle state from the environment."""
        self.vehicle_state = vehicle_state

    def control(self) -> ControlCommand:
        steer, vel, self.last_idx = _stanley_control(
            self.waypoints[:, :2],
            self.waypoints[:, 2],
            self.waypoints[:, 3],
            self.vehicle_state.x,
            self.vehicle_state.y,
            self.vehicle_state.yaw,
            self.vehicle_state.speed,
            self.l_f,
            self.k,
            -1 if self.last_idx is None else self.last_idx,
            self.search_window,
        )
        return ControlCommand(steering=steer, velocity=vel)


@njit(cache=True)
def _stanley_control(
    waypoints_xy: np.ndarray,
    waypoints_psi: np.ndarray,
    waypoints_vx: np.ndarray,
    x: float,
    y: float,
    psi: float,
    speed: float,
    l_f: float,
    k: float,
    last_idx: int,
    search_window: int,
) -> tuple[float, float, int]:
    """Full Stanley control law. Returns (steering, velocity, new_index)."""
    # Front axle position
    cos_psi = np.cos(psi)
    sin_psi = np.sin(psi)
    p_f_x = x + l_f * cos_psi
    p_f_y = y + l_f * sin_psi

    p_f = np.array([p_f_x, p_f_y], dtype=np.float64)
    s_star = _nearest_waypoint_index(waypoints_xy, p_f, last_idx, search_window)

    # Signed distance from front axle to path tangent
    # d = p^* - p_f
    d_x = waypoints_xy[s_star, 0] - p_f_x
    d_y = waypoints_xy[s_star, 1] - p_f_y
    psi_path = waypoints_psi[s_star] + np.pi / 2
    e_f = -d_x * np.sin(psi_path) + d_y * np.cos(psi_path)

    # Heading error wrapped in (-pi, pi]
    theta_e = np.arctan2(np.sin(psi_path - psi), np.cos(psi_path - psi))

    v = speed + V_SOFT  # avoid extreme value at low speed
    steer = theta_e + np.arctan2(k * e_f, v)
    steer = max(-DELTA_MAX, min(DELTA_MAX, steer))

    return steer, float(waypoints_vx[s_star]), s_star
