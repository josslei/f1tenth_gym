"""Map-to-centerline conversion helpers for raceline generation.

This module turns a ROS-style occupancy-grid map into the reference-track CSV
format consumed by the raceline optimizer:

    x_m, y_m, w_tr_right_m, w_tr_left_m

The algorithm follows the reference ``map_converter.ipynb`` workflow from
``ref/Raceline-Optimization``: threshold free track pixels, compute a Euclidean
distance transform, skeletonize the high-clearance track band, order the
skeleton into a loop, and use the distance transform as the local half-width.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from scipy import ndimage
from scipy.spatial import KDTree
import yaml


@dataclass(frozen=True)
class MapCenterline:
    points_xy: np.ndarray
    widths: np.ndarray
    image: np.ndarray
    resolution: float
    origin: np.ndarray


_NEIGHBOR_OFFSETS = (
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
)


def centerline_from_map(
    map_yaml: Path,
    *,
    min_points: int | None,
    target_spacing: float,
    max_points: int | None,
    free_threshold: float,
    centerline_threshold: float,
    track_width_margin: float = 0.0,
) -> MapCenterline:
    image, resolution, origin = _load_map(map_yaml)
    free_space = _free_space_mask(image, free_threshold)
    distance_px = np.asarray(
        ndimage.distance_transform_edt(free_space),
        dtype=np.float64,
    )
    center_band = distance_px > centerline_threshold * float(distance_px.max())
    skeleton = _thin(center_band)
    skeleton = _largest_component(skeleton)
    skeleton = _prune_open_spurs(skeleton)

    if not skeleton.any():
        points_xy, widths = _centerline_from_boundaries(
            image,
            resolution,
            origin,
            free_threshold,
            min_points,
            target_spacing,
            max_points,
        )
        return MapCenterline(points_xy, widths, image, resolution, origin)

    ordered_pixels_xy = _order_skeleton_loop(skeleton)
    if not _route_spans_free_space(ordered_pixels_xy, free_space):
        points_xy, widths = _centerline_from_boundaries(
            image,
            resolution,
            origin,
            free_threshold,
            min_points,
            target_spacing,
            max_points,
        )
        return MapCenterline(points_xy, widths, image, resolution, origin)

    sampled_pixels_xy = _sample_closed_path(
        ordered_pixels_xy,
        _sample_count(
            ordered_pixels_xy,
            resolution,
            min_points,
            target_spacing,
            max_points,
        ),
    )
    sampled_widths_px = _sample_closed_values(
        _sample_values(distance_px, ordered_pixels_xy),
        ordered_pixels_xy,
        sampled_pixels_xy.shape[0],
    )

    height = image.shape[0]
    points_xy = np.column_stack(
        (
            origin[0] + sampled_pixels_xy[:, 0] * resolution,
            origin[1] + (height - sampled_pixels_xy[:, 1]) * resolution,
        )
    )
    half_widths = np.maximum(
        sampled_widths_px * resolution - track_width_margin,
        0.5 * resolution,
    )
    widths = np.column_stack((half_widths, half_widths))

    return MapCenterline(points_xy, widths, image, resolution, origin)


def _load_map(map_yaml: Path) -> tuple[np.ndarray, float, np.ndarray]:
    meta: dict[str, Any] = yaml.safe_load(map_yaml.read_text(encoding="utf-8"))
    image_path = map_yaml.with_name(meta["image"])
    image = np.asarray(Image.open(image_path).convert("L"), dtype=np.float64)
    resolution = float(meta["resolution"])
    origin = np.asarray(meta["origin"][:2], dtype=np.float64)
    return image, resolution, origin


def _free_space_mask(image: np.ndarray, threshold: float) -> np.ndarray:
    mask = image > threshold
    return _largest_component(mask)


def _largest_component(mask: np.ndarray) -> np.ndarray:
    label_result: Any = ndimage.label(mask)
    labels = np.asarray(label_result[0])
    component_count = int(label_result[1])
    if component_count == 0:
        return mask.astype(bool, copy=True)
    component_sizes = np.asarray(
        ndimage.sum(mask, labels, index=np.arange(1, component_count + 1)),
        dtype=np.float64,
    )
    largest_label = int(np.argmax(component_sizes) + 1)
    return labels == largest_label


def _thin(mask: np.ndarray) -> np.ndarray:
    skeleton = np.pad(mask.astype(bool, copy=True), 1)
    changed = True
    while changed:
        changed = False
        for step in (0, 1):
            remove = []
            ys, xs = np.nonzero(skeleton)
            for y, x in zip(ys, xs, strict=True):
                y_int = int(y)
                x_int = int(x)
                if (
                    y_int == 0
                    or x_int == 0
                    or y_int == skeleton.shape[0] - 1
                    or x_int == skeleton.shape[1] - 1
                ):
                    continue
                n = _clockwise_neighbors(skeleton, y_int, x_int)
                neighbor_count = int(sum(n))
                transitions = sum((not n[i]) and n[(i + 1) % 8] for i in range(8))
                if neighbor_count < 2 or neighbor_count > 6 or transitions != 1:
                    continue
                if step == 0:
                    keep = n[0] and n[2] and n[4] or n[2] and n[4] and n[6]
                else:
                    keep = n[0] and n[2] and n[6] or n[0] and n[4] and n[6]
                if not keep:
                    remove.append((y, x))
            if remove:
                changed = True
                for y, x in remove:
                    skeleton[y, x] = False
    return skeleton[1:-1, 1:-1]


def _clockwise_neighbors(mask: np.ndarray, y: int, x: int) -> tuple[bool, ...]:
    return (
        bool(mask[y - 1, x]),
        bool(mask[y - 1, x + 1]),
        bool(mask[y, x + 1]),
        bool(mask[y + 1, x + 1]),
        bool(mask[y + 1, x]),
        bool(mask[y + 1, x - 1]),
        bool(mask[y, x - 1]),
        bool(mask[y - 1, x - 1]),
    )


def _prune_open_spurs(mask: np.ndarray) -> np.ndarray:
    pruned = mask.copy()
    while True:
        neighbor_counts = ndimage.convolve(
            pruned.astype(np.uint8),
            np.ones((3, 3), dtype=np.uint8),
            mode="constant",
            cval=0,
        ) - pruned.astype(np.uint8)
        endpoints = pruned & (neighbor_counts <= 1)
        if not endpoints.any():
            return pruned
        pruned[endpoints] = False


def _order_skeleton_loop(mask: np.ndarray) -> np.ndarray:
    ordered_pixels_yx = _trace_loop(mask)
    return ordered_pixels_yx[:, ::-1].astype(np.float64)


def _route_spans_free_space(
    route_pixels_xy: np.ndarray, free_space: np.ndarray
) -> bool:
    free_y, free_x = np.nonzero(free_space)
    free_pixels_xy = np.column_stack((free_x, free_y)).astype(np.float64)
    free_span = np.ptp(free_pixels_xy, axis=0)
    route_span = np.ptp(route_pixels_xy, axis=0)
    return bool(np.all(route_span > 0.45 * free_span))


def _sample_count(
    points_xy: np.ndarray,
    resolution: float,
    min_points: int | None,
    target_spacing: float,
    max_points: int | None,
) -> int:
    closed_points = np.vstack([points_xy, points_xy[:1]])
    loop_length_px = float(
        np.sum(np.linalg.norm(np.diff(closed_points, axis=0), axis=1))
    )
    length_based_count = int(np.ceil(loop_length_px * resolution / target_spacing))
    sample_count = max(min_points or 0, length_based_count)
    if max_points is not None:
        sample_count = min(sample_count, max_points)
    return max(sample_count, 3)


def _sample_closed_path(points_xy: np.ndarray, num_points: int) -> np.ndarray:
    closed_points = np.vstack([points_xy, points_xy[:1]])
    segments = np.diff(closed_points, axis=0)
    segment_lengths = np.sqrt((segments**2).sum(axis=1))
    cumulative = np.concatenate([[0.0], np.cumsum(segment_lengths)])
    target_lengths = np.linspace(
        0.0,
        float(cumulative[-1]),
        num_points,
        endpoint=False,
    )
    resampled = np.empty((num_points, points_xy.shape[1]), dtype=np.float64)
    for axis in range(points_xy.shape[1]):
        resampled[:, axis] = np.interp(
            target_lengths,
            cumulative,
            closed_points[:, axis],
        )
    return resampled


def _sample_values(values: np.ndarray, pixels_xy: np.ndarray) -> np.ndarray:
    rounded = np.rint(pixels_xy).astype(np.int64)
    return values[rounded[:, 1], rounded[:, 0]]


def _sample_closed_values(
    values: np.ndarray,
    points_xy: np.ndarray,
    num_points: int,
) -> np.ndarray:
    closed_points = np.vstack([points_xy, points_xy[:1]])
    segments = np.diff(closed_points, axis=0)
    segment_lengths = np.sqrt((segments**2).sum(axis=1))
    cumulative = np.concatenate([[0.0], np.cumsum(segment_lengths)])
    target_lengths = np.linspace(
        0.0,
        float(cumulative[-1]),
        num_points,
        endpoint=False,
    )
    return np.interp(target_lengths, cumulative, np.append(values, values[:1]))


def _centerline_from_boundaries(
    image: np.ndarray,
    resolution: float,
    origin: np.ndarray,
    threshold: float,
    min_points: int | None,
    target_spacing: float,
    max_points: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    free_mask = _largest_component(image >= threshold)
    filled_track = np.asarray(ndimage.binary_fill_holes(free_mask), dtype=bool)
    inner_hole = _largest_component(filled_track & ~free_mask)
    inner_boundary = _boundary(inner_hole)
    outer_boundary = _boundary(filled_track)

    ordered_inner_pixels_yx = _trace_loop(inner_boundary)
    inner_pixels_xy = ordered_inner_pixels_yx[:, ::-1].astype(np.float64)
    outer_pixels_yx = np.column_stack(np.nonzero(outer_boundary))
    outer_pixels_xy = outer_pixels_yx[:, ::-1].astype(np.float64)
    boundary_distances_px, outer_indices = KDTree(outer_pixels_xy).query(
        inner_pixels_xy
    )
    matched_outer_pixels_xy = outer_pixels_xy[outer_indices]
    ordered_pixels_xy = 0.5 * (inner_pixels_xy + matched_outer_pixels_xy)
    sample_count = _sample_count(
        ordered_pixels_xy,
        resolution,
        min_points,
        target_spacing,
        max_points,
    )
    sampled_pixels_xy = _sample_closed_path(ordered_pixels_xy, sample_count)

    height = image.shape[0]
    points_xy = np.column_stack(
        (
            origin[0] + sampled_pixels_xy[:, 0] * resolution,
            origin[1] + (height - sampled_pixels_xy[:, 1]) * resolution,
        )
    )
    sampled_widths_px = _sample_closed_values(
        boundary_distances_px,
        ordered_pixels_xy,
        sample_count,
    )
    half_widths = np.maximum(0.5 * sampled_widths_px * resolution, 0.5 * resolution)
    widths = np.column_stack((half_widths, half_widths))
    return points_xy, widths


def _boundary(mask: np.ndarray) -> np.ndarray:
    eroded = np.asarray(ndimage.binary_erosion(mask), dtype=bool)
    return mask & ~eroded


def _trace_loop(mask: np.ndarray) -> np.ndarray:
    coords = np.column_stack(np.nonzero(mask))
    coord_to_index = {tuple(coord): idx for idx, coord in enumerate(coords)}
    neighbors: list[list[int]] = [[] for _ in range(len(coords))]
    for idx, (y, x) in enumerate(coords):
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                neighbor = coord_to_index.get((y + dy, x + dx))
                if neighbor is not None:
                    neighbors[idx].append(neighbor)

    best_cycle: list[int] | None = None
    best_score = (-1.0, -1.0, -1)
    for start, start_neighbors in enumerate(neighbors):
        for first in start_neighbors:
            cycle = _trace_candidate(coords, neighbors, start, first)
            if cycle is None:
                continue
            cycle_coords = coords[np.asarray(cycle, dtype=np.int64)]
            span = np.ptp(cycle_coords, axis=0)
            closed_cycle = np.vstack([cycle_coords, cycle_coords[:1]])
            length = float(
                np.sum(np.linalg.norm(np.diff(closed_cycle, axis=0), axis=1))
            )
            score = (float(span[0] * span[1]), length, len(cycle))
            if score > best_score:
                best_score = score
                best_cycle = cycle

    if best_cycle is None:
        return coords
    return coords[np.asarray(best_cycle, dtype=np.int64)]


def _trace_candidate(
    coords: np.ndarray,
    neighbors: list[list[int]],
    start: int,
    first: int,
) -> list[int] | None:
    ordered = [start, first]
    visited = {start: 0, first: 1}
    previous = start
    current = first
    for _ in range(len(coords) + 1):
        candidates = [
            neighbor for neighbor in neighbors[current] if neighbor != previous
        ]
        if not candidates:
            return None
        next_idx = _next_boundary_step(coords, previous, current, candidates)
        if next_idx in visited:
            cycle = ordered[visited[next_idx] :]
            return cycle if len(cycle) >= 3 else None
        visited[next_idx] = len(ordered)
        ordered.append(next_idx)
        previous = current
        current = next_idx
    return None


def _next_boundary_step(
    coords: np.ndarray,
    previous: int,
    current: int,
    candidates: list[int],
) -> int:
    if len(candidates) == 1:
        return candidates[0]
    prev_vec = coords[current] - coords[previous]
    prev_norm = float(np.hypot(prev_vec[0], prev_vec[1])) or 1.0

    def score(idx: int) -> tuple[float, float, float]:
        vec = coords[idx] - coords[current]
        vec_norm = float(np.hypot(vec[0], vec[1])) or 1.0
        cos_turn = float(np.dot(prev_vec, vec) / (prev_norm * vec_norm))
        return cos_turn, -float(coords[idx, 0]), -float(coords[idx, 1])

    return max(candidates, key=score)


__all__ = ["MapCenterline", "centerline_from_map"]
