import numpy as np
from pathlib import Path
from numba import njit
from typing import Optional

from .controller_base import Controller, VehicleState, ControlCommand

V_MIN = 0.0
V_MAX = 20.0
A_Y_MAX = 1.0489 * 9.81 / 2


class PurePursuit(Controller):
    def __init__(
        self,
        waypoints: np.ndarray,
        lookahead: float,
        wheelbase: float,
    ) -> None:
        self.vehicle_state = VehicleState(0, 0, 0, 0)
        self.waypoints = waypoints
        self.l_d = lookahead
        self.L = wheelbase

        self.last_idx: Optional[int] = None

        self.num_waypoints = waypoints.shape[0]
        if self.num_waypoints == 0:
            raise ValueError("waypoints must not be empty")

    @classmethod
    def from_csv(
        cls, csv_path: str | Path, lookahead: float, wheelbase: float
    ) -> "PurePursuit":
        waypoints = np.loadtxt(csv_path, delimiter=",", skiprows=1, dtype=np.float64)
        waypoints = np.atleast_2d(waypoints)
        return cls(waypoints[:, :2], lookahead, wheelbase)

    def reset(self) -> None:
        self.vehicle_state = VehicleState(0, 0, 0, 0)
        self.last_idx = None

    def update(self, vehicle_state: VehicleState) -> None:
        self.vehicle_state = vehicle_state

    def control(self) -> ControlCommand:
        p: np.ndarray = self._get_next_waypoint()
        R = np.dot(p, p) / (2 * p[1])  # \frac{(p^x)^2 + (p^y)^2}{2p^y}
        kappa = 1 / R

        delta = target_steering(kappa, self.L)
        speed = target_speed(kappa)
        return ControlCommand(steering=delta, velocity=speed)

    def _get_next_waypoint(self) -> np.ndarray:
        position = np.array(
            (self.vehicle_state.x, self.vehicle_state.y), dtype=np.float64
        )
        last_idx = -1 if self.last_idx is None else self.last_idx
        p_goal, goal_idx = get_goal_waypoint(
            self.waypoints,
            position,
            self.vehicle_state.yaw,
            last_idx,
            self.l_d,
            self.num_waypoints,
        )
        self.last_idx = goal_idx
        return p_goal


@njit(cache=True)
def target_steering(curvature: float, L: float) -> float:
    return np.arctan(L * curvature)


@njit(cache=True)
def target_speed(curvature: float, eps: float = 1e-9) -> float:
    speed = np.sqrt(A_Y_MAX / (np.abs(curvature) + eps))
    return min(max(speed, V_MIN), V_MAX)


@njit(cache=True)
def get_goal_waypoint(
    waypoints: np.ndarray,
    position: np.ndarray,
    yaw: float,
    last_idx: int,
    lookahead: float,
    num_waypoints: int,
) -> tuple[np.ndarray, int]:
    if last_idx < 0:
        start_idx = nearest_waypoint_index(waypoints, position, 0, num_waypoints)
    else:
        start_idx = nearest_waypoint_index(waypoints, position, last_idx)

    goal_idx = start_idx
    accumulated_distance = 0.0
    while accumulated_distance < lookahead:
        next_idx = (goal_idx + 1) % num_waypoints
        current_xy = waypoints[goal_idx, :2]
        next_xy = waypoints[next_idx, :2]
        accumulated_distance += np.linalg.norm(next_xy - current_xy)
        goal_idx = next_idx

    goal_world = waypoints[goal_idx, :2]

    # Transpose to the body frame
    dx = goal_world[0] - position[0]
    dy = goal_world[1] - position[1]

    cos_yaw = np.cos(yaw)
    sin_yaw = np.sin(yaw)
    p_goal = np.empty((2,), dtype=np.float64)
    p_goal[0] = cos_yaw * dx + sin_yaw * dy
    p_goal[1] = -sin_yaw * dx + cos_yaw * dy
    return p_goal, goal_idx


@njit(cache=True)
def nearest_waypoint_index(
    waypoints: np.ndarray,
    position: np.ndarray,
    start_idx: int,
    search_window: int = 200,
) -> int:
    point_count = waypoints.shape[0]
    position_x = position[0]
    position_y = position[1]

    if search_window <= 0 or search_window >= point_count:
        best_idx = 0
        best_distance_sq = np.inf
        for idx in range(point_count):
            dx = waypoints[idx, 0] - position_x
            dy = waypoints[idx, 1] - position_y
            distance_sq = dx * dx + dy * dy
            if distance_sq < best_distance_sq:
                best_distance_sq = distance_sq
                best_idx = idx
        return best_idx

    half_window = search_window // 2
    first_offset = -half_window
    last_offset = search_window - half_window
    best_idx = start_idx % point_count
    best_distance_sq = np.inf
    for offset in range(first_offset, last_offset):
        idx = (start_idx + offset) % point_count
        dx = waypoints[idx, 0] - position_x
        dy = waypoints[idx, 1] - position_y
        distance_sq = dx * dx + dy * dy
        if distance_sq < best_distance_sq:
            best_distance_sq = distance_sq
            best_idx = idx
    return best_idx
