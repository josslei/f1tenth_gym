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


def _polyline_segment_vertices(
    points_xy: np.ndarray, close: bool = False
) -> np.ndarray:
    """Duplicate interior vertices so a polyline draws as GL_LINES.

    Never batch GL_LINE_STRIP/GL_LINE_LOOP vertex lists here: pyglet's
    Batch packs same-mode lists into one vertex domain and its allocator
    MERGES adjacent allocations into a single glMultiDrawArrays span
    (pyglet/graphics/allocation.py), which fuses separate strips into one
    connected line -- measured in this project as a phantom segment
    joining the receding-horizon line to a fixed vertex of whichever other
    strip sat next to it in the buffer (worst from lap 2 on, once the
    previous-lap trace exists). GL_LINES segments are order-independent,
    so region merging cannot change what they draw.
    """
    if close:
        points_xy = np.vstack([points_xy, points_xy[:1]])
    n = points_xy.shape[0]
    return points_xy[np.repeat(np.arange(n), 2)[1:-1]]


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

        from pyglet.gl.gl import GL_LINES

        vertices = _polyline_segment_vertices(self.waypoints_xy, close=True)
        positions = _to_xyz_flat(vertices, self.render_scale)
        colors = list(self.color) * vertices.shape[0]
        self._vertex_list = renderer.program.vertex_list(
            vertices.shape[0],
            GL_LINES,
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
    # The live trace is stored as fixed-size GL chunks: a full chunk's vertex
    # list is never touched again, and each accepted point rebuilds only the
    # one open (partial) chunk. Rebuilding a single ever-growing vertex list
    # instead -- even only every few points -- is O(points-so-far) per rebuild
    # and therefore O(n^2/stride) accumulated over a lap; measured as an FPS
    # drop that grows with lap progress and is independent of any LMPC solver
    # setting. Each new chunk starts at the previous chunk's last point so the
    # per-chunk polylines join without gaps.
    chunk_capacity: int = 256
    current_points: list[tuple[float, float]] = field(default_factory=list)
    previous_points: list[tuple[float, float]] = field(default_factory=list)
    _last_lap_count: int = 0
    _open_chunk_start: int = 0
    _frozen_vertex_lists: list[Any] = field(default_factory=list)
    _open_vertex_list: Any | None = None
    _previous_vertex_list: Any | None = None

    def __call__(self, renderer: Any, obs: dict[str, Any] | None = None) -> None:
        if obs is None:
            return

        ego = int(obs["ego_idx"])
        lap_count = int(obs["lap_counts"][ego])
        point = (float(obs["poses_x"][ego]), float(obs["poses_y"][ego]))

        # An INCREASE is a completed lap: promote the live trace to the
        # previous-lap trace. A DECREASE is an env.reset() (lap-as-iteration
        # relaunch, runs/lmpc_drive.py): the completed lap already finalized
        # at the crossing on the way up, so only clear the live trace --
        # finalizing again here would overwrite the full previous-lap loop
        # with the two-point crossing-to-reset stub recorded in between.
        if lap_count > self._last_lap_count:
            self.previous_points = self.current_points
            self._reset_current_trace(lap_count)
            if self._previous_vertex_list is not None:
                self._previous_vertex_list.delete()
            # One full build per lap change, not per frame.
            self._previous_vertex_list = self._make_vertex_list(
                renderer, self.previous_points, self.previous_color
            )
        elif lap_count < self._last_lap_count:
            self._reset_current_trace(lap_count)

        if self._should_append(point):
            self.current_points.append(point)
            self._rebuild_open_chunk(renderer)

    def _reset_current_trace(self, lap_count: int) -> None:
        self.current_points = []
        self._last_lap_count = lap_count
        self._open_chunk_start = 0
        for vertex_list in self._frozen_vertex_lists:
            vertex_list.delete()
        self._frozen_vertex_lists = []
        if self._open_vertex_list is not None:
            self._open_vertex_list.delete()
            self._open_vertex_list = None

    def _should_append(self, point: tuple[float, float]) -> bool:
        if not self.current_points:
            return True
        last_x, last_y = self.current_points[-1]
        dx = point[0] - last_x
        dy = point[1] - last_y
        return dx * dx + dy * dy >= self.min_point_distance * self.min_point_distance

    def _rebuild_open_chunk(self, renderer: Any) -> None:
        if self._open_vertex_list is not None:
            self._open_vertex_list.delete()
            self._open_vertex_list = None
        chunk = self.current_points[self._open_chunk_start :]
        self._open_vertex_list = self._make_vertex_list(
            renderer, chunk, self.current_color
        )
        if len(chunk) >= self.chunk_capacity:
            # Freeze this chunk's vertex list permanently; the next chunk
            # shares its last point so the strips stay connected.
            self._frozen_vertex_lists.append(self._open_vertex_list)
            self._open_vertex_list = None
            self._open_chunk_start = len(self.current_points) - 1

    def _make_vertex_list(
        self,
        renderer: Any,
        points: list[tuple[float, float]],
        color: tuple[int, int, int, int],
    ) -> Any | None:
        if len(points) < 2:
            return None

        from pyglet.gl.gl import GL_LINES

        vertices = _polyline_segment_vertices(np.asarray(points, dtype=np.float64))
        positions = _to_xyz_flat(vertices, self.render_scale)
        colors = list(color) * vertices.shape[0]
        return renderer.program.vertex_list(
            vertices.shape[0],
            GL_LINES,
            batch=renderer.batch,
            position=("f", positions),
            colors=("Bn", colors),
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

        from pyglet.gl.gl import GL_LINES, GL_POINTS

        # GL_POINTS is order-independent, so only the line needs the
        # segment conversion (_polyline_segment_vertices' comment).
        segment_vertices = _polyline_segment_vertices(points_xy)
        positions = _to_xyz_flat(segment_vertices, self.render_scale)
        colors = list(self.color) * segment_vertices.shape[0]
        self._vertex_list = renderer.program.vertex_list(
            segment_vertices.shape[0],
            GL_LINES,
            batch=renderer.batch,
            position=("f", positions),
            colors=("Bn", colors),
        )
        point_positions = _to_xyz_flat(points_xy, self.render_scale)
        self._point_vertex_list = renderer.program.vertex_list(
            points_xy.shape[0],
            GL_POINTS,
            batch=renderer.batch,
            position=("f", point_positions),
            colors=("Bn", list(self.point_color) * points_xy.shape[0]),
        )
