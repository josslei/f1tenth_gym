from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


class F1TenthProgressReward:
    """Reward based on forward progress along the centerline.

    Computes per-step arc-length progress along the resampled waypoint path,
    plus speed bonus and steering smoothness penalty.
    """

    def __init__(
        self,
        *,
        waypoints_path: str | Path | None = None,
        speed_reward_weight: float = 0.1,
        progress_weight: float = 1.0,
        steer_smoothness_weight: float = 0.5,
        collision_penalty: float = 50.0,
        spin_threshold: float = 100.0,
        delimiter: str = ";",
        usecols: tuple[int, int] = (1, 2),
    ) -> None:
        self.speed_reward_weight = speed_reward_weight
        self.progress_weight = progress_weight
        self.steer_smoothness_weight = steer_smoothness_weight
        self.collision_penalty = collision_penalty
        self.spin_threshold = spin_threshold

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
        self.prev_steer: float = 0.0

    def _build_arc(self) -> None:
        diffs = np.diff(self.waypoints, axis=0)
        seg_lengths = np.sqrt((diffs**2).sum(axis=1))
        self.cum_arc_lengths = np.concatenate([[0.0], np.cumsum(seg_lengths)])
        self.total_length = float(self.cum_arc_lengths[-1])

    def set_waypoints(self, waypoints_xy: np.ndarray) -> None:
        self.waypoints = np.asarray(waypoints_xy, dtype=np.float64).reshape(-1, 2)
        self._build_arc()
        self.prev_arc_length = 0.0
        self.prev_steer = 0.0

    def __call__(self, obs: dict[str, Any], terminated: bool) -> float:
        ego = int(obs["ego_idx"])
        vx = float(np.nan_to_num(obs["linear_vels_x"][ego], nan=0.0))
        vy = float(np.nan_to_num(obs["linear_vels_y"][ego], nan=0.0))
        collision = bool(obs["collisions"][ego])
        theta = float(np.nan_to_num(obs["poses_theta"][ego], nan=0.0))
        px = float(np.nan_to_num(obs["poses_x"][ego], nan=0.0))
        py = float(np.nan_to_num(obs["poses_y"][ego], nan=0.0))
        steer = float(np.nan_to_num(obs.get("steer_angle", [0.0])[ego], nan=0.0))

        if collision or abs(theta) > self.spin_threshold:
            if terminated:
                self.prev_arc_length = 0.0
                self.prev_steer = 0.0
            return -float(self.collision_penalty)

        if terminated:
            self.prev_arc_length = 0.0
            self.prev_steer = 0.0

        vel_magnitude = np.sqrt(vx * vx + vy * vy)
        reward = self.speed_reward_weight * float(vel_magnitude)

        wx, wy = self.waypoints[:, 0], self.waypoints[:, 1]
        dist_sq = (wx - px) ** 2 + (wy - py) ** 2
        nearest_idx = int(np.argmin(dist_sq))
        current_arc = float(self.cum_arc_lengths[nearest_idx])

        progress = current_arc - self.prev_arc_length
        if progress < 0.0:
            progress = (self.total_length - self.prev_arc_length) + current_arc
        reward += self.progress_weight * progress
        self.prev_arc_length = current_arc

        steer_delta = abs(steer - self.prev_steer)
        reward -= self.steer_smoothness_weight * steer_delta
        self.prev_steer = steer

        return float(np.clip(reward, -5.0, 8.0))


__all__ = ["F1TenthProgressReward"]
