"""Waypoint helpers for shared use across runnable scripts."""

from __future__ import annotations

from dataclasses import dataclass, field
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

    def __call__(self, renderer: Any, obs: dict[str, Any] | None = None) -> None:
        del obs
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


@dataclass
class DrivenLineOverlay:
    """Draw the current lap trace and the previous completed lap trace."""

    current_color: tuple[int, int, int, int] = (0, 220, 255, 255)
    previous_color: tuple[int, int, int, int] = (255, 90, 90, 190)
    render_scale: float = WAYPOINT_RENDER_SCALE
    min_point_distance: float = 0.05
    current_points: list[tuple[float, float]] = field(default_factory=list)
    previous_points: list[tuple[float, float]] = field(default_factory=list)
    _last_lap_count: int = 0
    _current_vertex_list: Any | None = None
    _previous_vertex_list: Any | None = None

    def __call__(self, renderer: Any, obs: dict[str, Any] | None = None) -> None:
        if obs is None:
            return

        ego = int(obs["ego_idx"])
        lap_count = int(obs["lap_counts"][ego])
        point = (float(obs["poses_x"][ego]), float(obs["poses_y"][ego]))

        if lap_count > self._last_lap_count:
            self.previous_points = self.current_points
            self.current_points = []
            self._last_lap_count = lap_count
            self._replace_vertex_list(
                "previous", renderer, self.previous_points, self.previous_color
            )

        if self._should_append(point):
            self.current_points.append(point)
            self._replace_vertex_list(
                "current", renderer, self.current_points, self.current_color
            )

    def _should_append(self, point: tuple[float, float]) -> bool:
        if not self.current_points:
            return True
        last_x, last_y = self.current_points[-1]
        dx = point[0] - last_x
        dy = point[1] - last_y
        return dx * dx + dy * dy >= self.min_point_distance * self.min_point_distance

    def _replace_vertex_list(
        self,
        which: str,
        renderer: Any,
        points: list[tuple[float, float]],
        color: tuple[int, int, int, int],
    ) -> None:
        vertex_attr = f"_{which}_vertex_list"
        vertex_list = getattr(self, vertex_attr)
        if vertex_list is not None:
            vertex_list.delete()
            setattr(self, vertex_attr, None)
        if len(points) < 2:
            return

        from pyglet.gl.gl import GL_LINE_STRIP

        points_xy = np.asarray(points, dtype=np.float64)
        positions = _to_xyz_flat(points_xy, self.render_scale)
        colors = list(color) * points_xy.shape[0]
        setattr(
            self,
            vertex_attr,
            renderer.program.vertex_list(
                points_xy.shape[0],
                GL_LINE_STRIP,
                batch=renderer.batch,
                position=("f", positions),
                colors=("Bn", colors),
            ),
        )


@dataclass
class RecedingHorizonOverlay:
    """Draw an LMPC predicted horizon as a viewer callback."""

    controller: Any
    color: tuple[int, int, int, int] = (255, 0, 255, 255)
    point_color: tuple[int, int, int, int] = (255, 255, 255, 255)
    render_scale: float = WAYPOINT_RENDER_SCALE
    _vertex_list: Any | None = None
    _point_vertex_list: Any | None = None

    def __call__(self, renderer: Any, obs: dict[str, Any] | None = None) -> None:
        del obs
        points_xy = self.controller.predicted_horizon_xy()
        if self._vertex_list is not None:
            self._vertex_list.delete()
            self._vertex_list = None
        if self._point_vertex_list is not None:
            self._point_vertex_list.delete()
            self._point_vertex_list = None
        if points_xy.shape[0] < 2:
            return

        from pyglet.gl.gl import GL_LINE_STRIP, GL_POINTS

        positions = _to_xyz_flat(points_xy, self.render_scale)
        colors = list(self.color) * points_xy.shape[0]
        self._vertex_list = renderer.program.vertex_list(
            points_xy.shape[0],
            GL_LINE_STRIP,
            batch=renderer.batch,
            position=("f", positions),
            colors=("Bn", colors),
        )
        self._point_vertex_list = renderer.program.vertex_list(
            points_xy.shape[0],
            GL_POINTS,
            batch=renderer.batch,
            position=("f", positions),
            colors=("Bn", list(self.point_color) * points_xy.shape[0]),
        )
