import numpy as np
from pathlib import Path
from numba import njit
from typing import Any, Optional
from abc import ABC, abstractmethod

from .controller_base import Controller, VehicleState, ControlCommand
from utils.waypoint_utils import nearest_waypoint_index


# A fixed INDEX-count window silently becomes physically narrower as
# waypoint density increases (same issue found and fixed in
# controllers/stanley.py -- see that module's comment for the measured
# before/after). Kept as a physical-distance target here too; PurePursuit
# derives its own instance-level index count from actual waypoint spacing
# in __init__ instead of using this as a raw index count directly.
SEARCH_WINDOW_METERS: float = 20.0


class LookaheadPolicy(ABC):
    """Policy for determining the lookahead distance ahead of the vehicle."""

    @abstractmethod
    def get_lookahead_distance(self, *args: Any, **kwargs: Any) -> float: ...

    def __call__(self, *args: Any, **kwargs: Any) -> float:
        return self.get_lookahead_distance(*args, **kwargs)


class FixedLookaheadDistance(LookaheadPolicy):
    """Return a constant lookahead distance regardless of vehicle speed."""

    def __init__(self, lookahead_distance: float = 1.0) -> None:
        self.lookahead_distance = lookahead_distance

    def get_lookahead_distance(self, *args: Any, **kwargs: Any) -> float:
        return self.lookahead_distance


class DynamicLookaheadDistance(LookaheadPolicy):
    """Scale the lookahead distance proportionally to the current vehicle speed."""

    def __init__(
        self,
        min_lookahead: float = 0.5,
        max_lookahead: float = 4.0,
        lookahead_ratio: float = 8.0,
    ) -> None:
        self.min = min_lookahead
        self.max = max_lookahead
        self.ratio = lookahead_ratio

    def get_lookahead_distance(self, cur_speed: float) -> float:
        return min(max(self.min, self.max * cur_speed / self.ratio), self.max)


class PurePursuit(Controller):
    def __init__(
        self,
        waypoints: np.ndarray,
        lookahead: float | LookaheadPolicy,
        wheelbase: float,
    ) -> None:
        self.vehicle_state = VehicleState(0, 0, 0, 0)
        self.waypoints = waypoints
        if isinstance(lookahead, (int, float)):
            self.lookahead_policy: LookaheadPolicy = FixedLookaheadDistance(lookahead)
        else:
            self.lookahead_policy = lookahead
        self.L = wheelbase

        # Index of the goal waypoint returned by the previous control step.
        # Used as a search hint for nearest-waypoint lookup on the next step
        # so we only scan a local window instead of the full track.
        self.last_idx: Optional[int] = None

        self.num_waypoints = waypoints.shape[0]
        if self.num_waypoints == 0:
            raise ValueError("waypoints must not be empty")

        # SEARCH_WINDOW_METERS's comment has the rationale.
        if self.num_waypoints >= 2:
            avg_spacing = float(
                np.linalg.norm(np.diff(waypoints[:, :2], axis=0), axis=1).mean()
            )
        else:
            avg_spacing = 1.0
        self.search_window = max(
            10, int(round(SEARCH_WINDOW_METERS / max(avg_spacing, 1e-6)))
        )

    @classmethod
    def from_csv(
        cls, csv_path: str | Path, lookahead: float | LookaheadPolicy, wheelbase: float
    ) -> "PurePursuit":
        """Construct a controller from a semicolon-delimited waypoint CSV.

        The CSV is expected to have columns s_m; x_m; y_m; psi_rad;
        kappa_radpm; vx_mps; ax_mps2 (with a header row). Only x_m (col 1),
        y_m (col 2), and vx_mps (col 5) are used.
        """
        waypoints = np.loadtxt(csv_path, delimiter=";", skiprows=1, dtype=np.float64)
        waypoints = np.atleast_2d(waypoints)
        return cls(waypoints[:, [1, 2, 5]], lookahead, wheelbase)

    def reset(self) -> None:
        """Reset internal state to origin and clear the cached waypoint index."""
        self.vehicle_state = VehicleState(0, 0, 0, 0)
        self.last_idx = None

    def update(self, vehicle_state: VehicleState) -> None:
        """Receive the latest vehicle state from the environment."""
        self.vehicle_state = vehicle_state

    def control(self) -> ControlCommand:
        """Compute the steering and velocity command for the current state."""
        p_goal, target_speed = self._lookahead_target()
        R = np.dot(p_goal, p_goal) / (2 * p_goal[1])
        kappa = 1 / R

        delta = target_steering(kappa, self.L)
        return ControlCommand(steering=delta, velocity=target_speed)

    def _lookahead_target(self) -> tuple[np.ndarray, float]:
        """Find the goal waypoint ahead of the vehicle and its target speed.

        Uses the previously cached goal index as a hint for nearest-waypoint
        search. On the first call (after reset) the full track is scanned;
        subsequent calls search a local window around the last known position.
        """
        position = np.array(
            (self.vehicle_state.x, self.vehicle_state.y), dtype=np.float64
        )
        start_idx = nearest_waypoint_index(
            self.waypoints[:, :2], position, self.last_idx, self.search_window
        )
        target_speed = float(self.waypoints[start_idx, 2])
        lookahead = self.lookahead_policy(self.vehicle_state.speed)
        p_goal, goal_idx = get_goal_waypoint(
            self.waypoints,
            position,
            self.vehicle_state.yaw,
            start_idx,
            lookahead,
            self.num_waypoints,
        )
        self.last_idx = goal_idx
        return p_goal, target_speed


@njit(cache=True)
def target_steering(curvature: float, L: float) -> float:
    """Steering angle for the given path curvature with the bicycle model."""
    return np.arctan(L * curvature)


@njit(cache=True)
def get_goal_waypoint(
    waypoints: np.ndarray,
    position: np.ndarray,
    yaw: float,
    start_idx: int,
    lookahead: float,
    num_waypoints: int,
) -> tuple[np.ndarray, int]:
    """Walk forward from start_idx, accumulating distance until the
    lookahead threshold is reached. Return the resulting waypoint expressed
    in the vehicle body frame and its index."""
    goal_idx = start_idx
    accumulated_distance = 0.0
    while accumulated_distance < lookahead:
        next_idx = (goal_idx + 1) % num_waypoints
        current_xy = waypoints[goal_idx, :2]
        next_xy = waypoints[next_idx, :2]
        accumulated_distance += np.linalg.norm(next_xy - current_xy)
        goal_idx = next_idx

    goal_world = waypoints[goal_idx, :2]

    dx = goal_world[0] - position[0]
    dy = goal_world[1] - position[1]

    cos_yaw = np.cos(yaw)
    sin_yaw = np.sin(yaw)
    p_goal = np.empty((2,), dtype=np.float64)
    p_goal[0] = cos_yaw * dx + sin_yaw * dy
    p_goal[1] = -sin_yaw * dx + cos_yaw * dy
    return p_goal, goal_idx
