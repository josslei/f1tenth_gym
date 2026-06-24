from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from utils.waypoint_utils import cumulative_arc_lengths


class F1TenthProgressReward:
    """Reference progress reward used by the MCTS controllers."""

    def __init__(
        self,
        *,
        track_map: Any | None = None,
        waypoints_path: str | Path | None = None,
        q_s_progress: float = 1.0,
        q_s_alpha: float = 1.0,
        q_s_smooth: float = 0.0,
        terminal_penalty: float = 1000.0,
        alpha_th: float = 0.0,
        slip_terminal_penalty: float = 0.0,
        q_offtrack_grad: float = 0.0,
        delimiter: str = ";",
        usecols: tuple[int, int] = (1, 2),
    ) -> None:
        self.track_map = track_map
        self.q_s_progress = q_s_progress
        self.q_s_alpha = q_s_alpha
        self.q_s_smooth = q_s_smooth
        self.terminal_penalty = terminal_penalty
        self.alpha_th = alpha_th
        self.slip_terminal_penalty = slip_terminal_penalty
        self.q_offtrack_grad = q_offtrack_grad

        if waypoints_path is not None:
            self.waypoints = np.genfromtxt(
                str(waypoints_path),
                delimiter=delimiter,
                comments="#",
                usecols=usecols,
            ).reshape(-1, 2)
            self._build_arc()
        else:
            self.waypoints = np.empty((0, 2), dtype=np.float64)
            self.cum_arc_lengths = np.array([0.0])
            self.total_length = 0.0

        self.prev_arc_length: float = 0.0
        self.last_action_0: float = 0.0
        self.last_action_1: float = 0.0
        self.has_last_action = False

    def _build_arc(self) -> None:
        self.cum_arc_lengths = cumulative_arc_lengths(self.waypoints)
        self.total_length = float(self.cum_arc_lengths[-1])
        n = self.waypoints.shape[0]
        self.headings = np.empty(n, dtype=np.float64)
        for i in range(n):
            prev_i = n - 1 if i == 0 else i - 1
            next_i = (i + 1) % n
            dx = self.waypoints[next_i, 0] - self.waypoints[prev_i, 0]
            dy = self.waypoints[next_i, 1] - self.waypoints[prev_i, 1]
            self.headings[i] = np.arctan2(dy, dx)

    def set_waypoints(self, waypoints_xy: np.ndarray) -> None:
        self.waypoints = np.asarray(waypoints_xy, dtype=np.float64).reshape(-1, 2)
        self._build_arc()
        self.prev_arc_length = 0.0
        self.last_action_0 = 0.0
        self.last_action_1 = 0.0
        self.has_last_action = False

    def set_track_map(self, track_map: Any) -> None:
        self.track_map = track_map

    def reset(self) -> None:
        self.prev_arc_length = 0.0
        self.last_action_0 = 0.0
        self.last_action_1 = 0.0
        self.has_last_action = False

    def _nearest_index(self, px: float, py: float) -> int:
        d2 = (self.waypoints[:, 0] - px) ** 2 + (self.waypoints[:, 1] - py) ** 2
        return int(np.argmin(d2))

    def _arclength_at(self, px: float, py: float) -> float:
        if self.waypoints.shape[0] < 2:
            return 0.0
        idx = self._nearest_index(px, py)
        n = self.waypoints.shape[0]
        prev_i = n - 1 if idx == 0 else idx - 1
        next_i = (idx + 1) % n

        def project(a: int, b: int) -> tuple[float, float]:
            sx, sy = self.waypoints[a]
            ex, ey = self.waypoints[b]
            dx = ex - sx
            dy = ey - sy
            len2 = dx * dx + dy * dy
            if len2 < 1e-12:
                rx = px - sx
                ry = py - sy
                return 0.0, rx * rx + ry * ry
            t = ((px - sx) * dx + (py - sy) * dy) / len2
            t = float(np.clip(t, 0.0, 1.0))
            proj_x = sx + t * dx
            proj_y = sy + t * dy
            rx = px - proj_x
            ry = py - proj_y
            return t * float(np.sqrt(len2)), rx * rx + ry * ry

        p_prev = project(prev_i, idx)
        p_next = project(idx, next_i)
        if p_prev[1] < p_next[1]:
            return float(self.cum_arc_lengths[prev_i] + p_prev[0])
        return float(self.cum_arc_lengths[idx] + p_next[0])

    def _is_backward_terminal(self, px: float, py: float, theta: float) -> bool:
        ref_heading = self.headings[self._nearest_index(px, py)]
        hdiff = abs(theta - ref_heading)
        if hdiff > np.pi:
            hdiff = 2.0 * np.pi - hdiff
        return hdiff > np.pi / 2.0

    def _offtrack_real(self, px: float, py: float, collision: bool) -> float:
        if self.track_map is None:
            return float(collision)
        if hasattr(self.track_map, "query"):
            return float(self.track_map.query(px, py))
        if hasattr(self.track_map, "distance_at"):
            return float(
                np.clip(1.0 - float(self.track_map.distance_at(px, py)), 0.0, 1.0)
            )
        return float(collision)

    def __call__(self, obs: dict[str, Any], terminated: bool) -> float:
        ego = int(obs["ego_idx"])
        vx = float(np.nan_to_num(obs["linear_vels_x"][ego], nan=0.0))
        vy = float(np.nan_to_num(obs["linear_vels_y"][ego], nan=0.0))
        collision = bool(obs["collisions"][ego])
        theta = float(np.nan_to_num(obs["poses_theta"][ego], nan=0.0))
        px = float(np.nan_to_num(obs["poses_x"][ego], nan=0.0))
        py = float(np.nan_to_num(obs["poses_y"][ego], nan=0.0))
        current_action = np.asarray(
            obs.get("prev_action", [0.0, 0.0]), dtype=np.float64
        )
        action_0 = float(np.nan_to_num(current_action[0], nan=0.0))
        action_1 = float(np.nan_to_num(current_action[1], nan=0.0))

        current_arc = self._arclength_at(px, py)
        progress = current_arc - self.prev_arc_length
        if self.total_length > 0.0:
            if progress < -0.5 * self.total_length:
                progress += self.total_length
            elif progress > 0.5 * self.total_length:
                progress -= self.total_length

        beta = float(np.arctan2(vy, vx))
        delta_0 = action_0 - self.last_action_0 if self.has_last_action else 0.0
        delta_1 = action_1 - self.last_action_1 if self.has_last_action else 0.0
        reward = self.q_s_progress * progress
        reward -= self.q_s_alpha * (beta * beta)
        reward -= self.q_s_smooth * (delta_0 * delta_0 + delta_1 * delta_1)
        reward -= self.q_offtrack_grad * self._offtrack_real(px, py, collision)
        if abs(beta) > self.alpha_th:
            reward -= self.slip_terminal_penalty
        if np.hypot(vx, vy) > 12.0:
            reward -= 10000.0
        if self._is_backward_terminal(px, py, theta):
            reward -= self.terminal_penalty

        self.prev_arc_length = current_arc
        self.last_action_0 = action_0
        self.last_action_1 = action_1
        self.has_last_action = True

        if terminated:
            self.reset()

        return float(reward)


__all__ = ["F1TenthProgressReward"]
