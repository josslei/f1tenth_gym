import numpy as np
from numba import njit


def cumulative_arc_lengths(waypoints_xy: np.ndarray) -> np.ndarray:
    diffs = np.diff(waypoints_xy, axis=0)
    seg_lengths = np.sqrt((diffs**2).sum(axis=1))
    return np.concatenate([[0.0], np.cumsum(seg_lengths)])


def closed_path_length(waypoints_xy: np.ndarray) -> float:
    path_s = cumulative_arc_lengths(waypoints_xy)
    closing_length = np.linalg.norm(waypoints_xy[0] - waypoints_xy[-1])
    return float(path_s[-1] + closing_length)


def project_to_closed_path(
    waypoints_xy: np.ndarray,
    path_s: np.ndarray,
    position: np.ndarray,
    nearest_idx: int,
) -> tuple[float, float, float]:
    """Project onto either segment adjacent to the nearest closed-path point."""
    best_distance_sq = np.inf
    best_projection = (0.0, 0.0, 0.0)
    point_count = waypoints_xy.shape[0]
    for start_idx in ((nearest_idx - 1) % point_count, nearest_idx):
        end_idx = (start_idx + 1) % point_count
        segment = waypoints_xy[end_idx] - waypoints_xy[start_idx]
        segment_length = float(np.linalg.norm(segment))
        tangent = segment / segment_length
        along = float(
            np.clip(
                np.dot(position - waypoints_xy[start_idx], tangent), 0.0, segment_length
            )
        )
        projected = waypoints_xy[start_idx] + along * tangent
        distance_sq = float(np.sum((position - projected) ** 2))
        if distance_sq < best_distance_sq:
            heading = float(np.arctan2(tangent[1], tangent[0]))
            normal = np.array([-tangent[1], tangent[0]])
            best_distance_sq = distance_sq
            best_projection = (
                float(path_s[start_idx] + along),
                float(np.dot(position - projected, normal)),
                heading,
            )
    return best_projection


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


def resample_path(waypoints_xy: np.ndarray, spacing: float = 0.5) -> np.ndarray:
    """Resample a 2-D path at fixed arc-length spacing.

    The input path is interpolated linearly along cumulative arc length so
    that consecutive output points are separated by *spacing* metres.  The
    total number of points is ``ceil(total_length / spacing)``.

    Args:
        waypoints_xy:  ``(N, 2)`` array of path points.
        spacing:       Desired arc length between consecutive points.

    Returns:
        ``(M, 2)`` array of uniformly spaced points.
    """
    cum_lengths = cumulative_arc_lengths(waypoints_xy)
    total_length = float(cum_lengths[-1])
    if total_length == 0.0:
        return waypoints_xy.astype(np.float64, copy=True)
    new_cum_lengths = np.arange(0.0, total_length, spacing)
    new_cum_lengths = np.append(new_cum_lengths, total_length)
    result = np.empty((len(new_cum_lengths), 2), dtype=np.float64)
    for j in range(2):
        result[:, j] = np.interp(new_cum_lengths, cum_lengths, waypoints_xy[:, j])
    return result


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

    if start_idx < 0 or search_window <= 0 or search_window >= point_count:
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
