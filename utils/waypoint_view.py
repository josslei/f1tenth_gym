"""Waypoint helpers for shared use across runnable scripts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

WAYPOINT_RENDER_SCALE = 50.0


def load_waypoints(csv_path: str | Path) -> np.ndarray:
    """Load a waypoint CSV and keep the waypoint coordinates only."""
    waypoints = np.loadtxt(csv_path, delimiter=",", skiprows=1, dtype=np.float64)
    waypoints = np.atleast_2d(waypoints)
    return waypoints[:, :2]


def initial_pose_from_waypoints(waypoints_xy: np.ndarray) -> np.ndarray:
    """Build a single-agent initial pose from the first waypoint segment."""
    start_x = float(waypoints_xy[0, 0])
    start_y = float(waypoints_xy[0, 1])

    if waypoints_xy.shape[0] > 1:
        next_x = float(waypoints_xy[1, 0])
        next_y = float(waypoints_xy[1, 1])
        yaw = float(np.arctan2(next_y - start_y, next_x - start_x))
    else:
        yaw = 0.0

    return np.array([[start_x, start_y, yaw]], dtype=np.float64)


def _to_xyz_flat(points_xy: np.ndarray, scale: float) -> list[float]:
    zeros = np.zeros((points_xy.shape[0], 1), dtype=np.float32)
    points_xyz = np.column_stack((scale * points_xy.astype(np.float32), zeros))
    return points_xyz.ravel().tolist()


@dataclass
class WaypointOverlay:
    """Draw the reference waypoint loop as a viewer callback."""

    waypoints_xy: np.ndarray
    color: tuple[int, int, int, int] = (255, 215, 0, 255)
    render_scale: float = WAYPOINT_RENDER_SCALE
    _vertex_list: Any | None = None

    def __call__(self, renderer: Any) -> None:
        if self._vertex_list is not None:
            return

        from pyglet.gl.gl import GL_LINE_LOOP

        positions = _to_xyz_flat(self.waypoints_xy, self.render_scale)
        colors = list(self.color) * self.waypoints_xy.shape[0]
        self._vertex_list = renderer.program.vertex_list(
            self.waypoints_xy.shape[0],
            GL_LINE_LOOP,
            batch=renderer.batch,
            position=("f", positions),
            colors=("Bn", colors),
        )
