import numpy as np
from numba import njit


def nearest_waypoint_index(
    waypoints: np.ndarray,
    position: np.ndarray,
    start_idx: int | None = None,
    search_window: int = 200,
) -> int:
    """Index of the waypoint closest to the vehicle position.

    Searches a sliding window of size search_window around start_idx.
    When start_idx is None (or -1) the entire track is scanned.
    """
    start_idx = start_idx if start_idx is not None else -1
    return _nearest_waypoint_index(waypoints, position, start_idx, search_window)


@njit(cache=True)
def _nearest_waypoint_index(
    waypoints: np.ndarray,
    position: np.ndarray,
    start_idx: int,
    search_window: int = 200,
) -> int:
    """Index of the waypoint closest to the vehicle position.

    Searches a sliding window of size search_window around start_idx.
    When start_idx is -1 the entire track is scanned.
    """
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
