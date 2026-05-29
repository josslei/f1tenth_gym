"""Generate a minimum-curvature racing line from an F1TENTH occupancy map.

The script loads a map YAML and image, extracts the interior free-space
component, estimates a centerline, and measures left/right track width at
each centerline point. It then optimizes lateral offsets to reduce curvature
and writes a CSV with columns:

    x, y, yaw, speed
"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, NoReturn, cast

import numpy as np
import numpy.typing as npt
import yaml
from numba import njit
from PIL import Image
from scipy import ndimage as ndi
from scipy.optimize import minimize


@dataclass(frozen=True)
class MapInfo:
    resolution: float
    origin_x: float
    origin_y: float
    origin_yaw: float


@dataclass(frozen=True)
class TrackGeometry:
    centerline: npt.NDArray[np.float64]
    left_widths: npt.NDArray[np.float64]
    right_widths: npt.NDArray[np.float64]
    total_widths: npt.NDArray[np.float64]
    closed: bool


class HelpOnErrorParser(argparse.ArgumentParser):
    """Argument parser that prints help text when parsing fails."""

    def error(self, message: str) -> NoReturn:
        self.print_help(sys.stderr)
        self.exit(2, f"\nerror: {message}\n")


def _time_block(name: str, start: float, enabled: bool) -> float:
    end = perf_counter()
    if enabled:
        print(f"{name}: {end - start:.3f} s")
    return perf_counter()


def _load_map(
    map_path: Path, map_ext: str | None
) -> tuple[npt.NDArray[np.bool_], MapInfo]:
    """Load a binary free-space mask and its world-frame metadata."""
    with open(map_path, "r", encoding="utf-8") as yaml_stream:
        metadata = cast(dict[str, Any], yaml.safe_load(yaml_stream))

    resolution = float(metadata["resolution"])
    origin = cast(tuple[float, float, float], tuple(metadata["origin"]))
    origin_x, origin_y, origin_yaw = origin

    if map_ext is None:
        image_path = map_path.with_name(str(metadata["image"]))
    else:
        image_path = map_path.with_suffix(map_ext)
    if not image_path.exists() and map_ext is None:
        for fallback_ext in (".png", ".pgm", ".jpg", ".jpeg"):
            fallback_path = map_path.with_suffix(fallback_ext)
            if fallback_path.exists():
                image_path = fallback_path
                break

    image = np.array(
        Image.open(image_path).convert("L").transpose(Image.Transpose.FLIP_TOP_BOTTOM)
    )
    negate = int(metadata.get("negate", 0))
    free_thresh = float(metadata.get("free_thresh", 0.196))
    image_values = image.astype(np.float64) / 255.0
    if negate:
        occupancy = image_values
    else:
        occupancy = 1.0 - image_values
    free = occupancy < free_thresh
    info = MapInfo(
        resolution=resolution,
        origin_x=float(origin_x),
        origin_y=float(origin_y),
        origin_yaw=float(origin_yaw),
    )
    return free, info


def _select_interior_component(free: npt.NDArray[np.bool_]) -> npt.NDArray[np.bool_]:
    """Keep the largest free-space component that does not touch the border."""
    labels, count = cast(
        tuple[npt.NDArray[np.intp], int],
        ndi.label(free, structure=ndi.generate_binary_structure(2, 2)),
    )
    if count == 0:
        raise ValueError("Map does not contain any free-space component.")

    border_labels = set(
        np.unique(
            np.concatenate([labels[0, :], labels[-1, :], labels[:, 0], labels[:, -1]])
        ).tolist()
    )
    border_labels.discard(0)

    best_label = 0
    best_size = -1
    for label in range(1, count + 1):
        if label in border_labels:
            continue
        size = int(np.count_nonzero(labels == label))
        if size > best_size:
            best_label = label
            best_size = size

    if best_label == 0:
        for label in range(1, count + 1):
            size = int(np.count_nonzero(labels == label))
            if size > best_size:
                best_label = label
                best_size = size

    return labels == best_label


def _thin_binary(mask: npt.NDArray[np.bool_]) -> npt.NDArray[np.bool_]:
    """Skeletonize a binary mask with Zhang-Suen thinning."""
    skeleton = mask.copy()
    changed = True

    while changed:
        changed = False
        for step in (0, 1):
            padded = np.pad(skeleton, 1, mode="constant", constant_values=False)
            p2 = padded[:-2, 1:-1]
            p3 = padded[:-2, 2:]
            p4 = padded[1:-1, 2:]
            p5 = padded[2:, 2:]
            p6 = padded[2:, 1:-1]
            p7 = padded[2:, :-2]
            p8 = padded[1:-1, :-2]
            p9 = padded[:-2, :-2]

            neighbors = (p2, p3, p4, p5, p6, p7, p8, p9)
            neighbor_count = sum(neighbor.astype(np.uint8) for neighbor in neighbors)
            transitions = np.zeros_like(neighbor_count)
            ordered = neighbors + (p2,)
            for current, following in zip(ordered[:-1], ordered[1:], strict=True):
                transitions += (~current & following).astype(np.uint8)

            if step == 0:
                connectivity = ~(p2 & p4 & p6) & ~(p4 & p6 & p8)
            else:
                connectivity = ~(p2 & p4 & p8) & ~(p2 & p6 & p8)

            removable = (
                skeleton
                & (neighbor_count >= 2)
                & (neighbor_count <= 6)
                & (transitions == 1)
                & connectivity
            )
            if np.any(removable):
                skeleton[removable] = False
                changed = True

    return skeleton


def _neighbor_count(mask: npt.NDArray[np.bool_]) -> npt.NDArray[np.intp]:
    kernel = np.array(
        [
            [1, 1, 1],
            [1, 0, 1],
            [1, 1, 1],
        ],
        dtype=np.intp,
    )
    return cast(npt.NDArray[np.intp], ndi.convolve(mask.astype(np.intp), kernel))


def _prune_skeleton_endpoints(
    skeleton: npt.NDArray[np.bool_], iterations: int
) -> npt.NDArray[np.bool_]:
    """Remove dangling skeleton branches while preserving closed loops."""
    pruned = skeleton.copy()
    for _ in range(iterations):
        endpoints = pruned & (_neighbor_count(pruned) == 1)
        if not np.any(endpoints):
            break
        pruned[endpoints] = False
    return pruned


def _largest_component(mask: npt.NDArray[np.bool_]) -> npt.NDArray[np.bool_]:
    labels, count = cast(
        tuple[npt.NDArray[np.intp], int],
        ndi.label(mask, structure=ndi.generate_binary_structure(2, 2)),
    )
    if count == 0:
        raise ValueError("Centerline skeleton is empty.")

    best_label = 1
    best_size = 0
    for label in range(1, count + 1):
        size = int(np.count_nonzero(labels == label))
        if size > best_size:
            best_label = label
            best_size = size
    return labels == best_label


def _trace_skeleton(skeleton: npt.NDArray[np.bool_]) -> npt.NDArray[np.float64]:
    """Return ordered pixel-space points along the largest skeleton component."""
    ys, xs = np.nonzero(skeleton)
    if xs.size < 3:
        raise ValueError("Centerline skeleton must contain at least three points.")

    pixels = [(int(y), int(x)) for y, x in zip(ys, xs, strict=True)]
    index_by_pixel = {pixel: index for index, pixel in enumerate(pixels)}
    neighbors: list[list[int]] = [[] for _ in pixels]

    offsets = (
        (-1, -1),
        (-1, 0),
        (-1, 1),
        (0, -1),
        (0, 1),
        (1, -1),
        (1, 0),
        (1, 1),
    )
    for index, (y, x) in enumerate(pixels):
        for dy, dx in offsets:
            neighbor = index_by_pixel.get((y + dy, x + dx))
            if neighbor is not None:
                neighbors[index].append(neighbor)

    degrees = np.array([len(node_neighbors) for node_neighbors in neighbors])
    endpoint_indices = np.flatnonzero(degrees == 1)
    if endpoint_indices.size >= 2:
        ordered_indices = _skeleton_diameter_path(neighbors, endpoint_indices)
        ordered_pixels = np.array(
            [pixels[index] for index in ordered_indices], dtype=np.float64
        )
        return np.column_stack((ordered_pixels[:, 1], ordered_pixels[:, 0]))

    current = int(np.argmin(xs))
    ordered_indices = [current]
    visited = {current}
    previous: int | None = None

    while True:
        candidates = [node for node in neighbors[current] if node not in visited]
        if not candidates:
            break

        if previous is None or len(candidates) == 1:
            next_index = candidates[0]
        else:
            previous_vector = np.array(pixels[current]) - np.array(pixels[previous])
            best_score = -np.inf
            next_index = candidates[0]
            for candidate in candidates:
                candidate_vector = np.array(pixels[candidate]) - np.array(
                    pixels[current]
                )
                score = float(np.dot(previous_vector, candidate_vector))
                if score > best_score:
                    best_score = score
                    next_index = candidate

        previous = current
        current = next_index
        ordered_indices.append(current)
        visited.add(current)

    ordered_pixels = np.array(
        [pixels[index] for index in ordered_indices], dtype=np.float64
    )
    return np.column_stack((ordered_pixels[:, 1], ordered_pixels[:, 0]))


def _skeleton_diameter_path(
    neighbors: list[list[int]], endpoint_indices: npt.NDArray[np.intp]
) -> list[int]:
    """Return the longest endpoint-to-endpoint path in a skeleton graph."""

    def farthest_endpoint(start: int) -> tuple[int, list[int | None]]:
        parents: list[int | None] = [None] * len(neighbors)
        distances = np.full(len(neighbors), -1, dtype=np.intp)
        queue = [start]
        distances[start] = 0
        head = 0
        while head < len(queue):
            current = queue[head]
            head += 1
            for neighbor in neighbors[current]:
                if distances[neighbor] == -1:
                    distances[neighbor] = distances[current] + 1
                    parents[neighbor] = current
                    queue.append(neighbor)

        endpoint_distances = distances[endpoint_indices]
        farthest_local = int(np.argmax(endpoint_distances))
        return int(endpoint_indices[farthest_local]), parents

    first, _ = farthest_endpoint(int(endpoint_indices[0]))
    second, parents = farthest_endpoint(first)

    path = [second]
    while path[-1] != first:
        parent = parents[path[-1]]
        if parent is None:
            break
        path.append(parent)
    path.reverse()
    return path


def _extract_centerline(
    interior: npt.NDArray[np.bool_], prune_iterations: int
) -> npt.NDArray[np.float64]:
    """Extract an ordered centerline from the free-space skeleton."""
    skeleton = _thin_binary(interior)
    pruned = _prune_skeleton_endpoints(skeleton, prune_iterations)
    if np.count_nonzero(pruned) < max(3, int(0.25 * np.count_nonzero(skeleton))):
        pruned = skeleton
    return _trace_skeleton(_largest_component(pruned))


def _pixels_to_world(
    points: npt.NDArray[np.float64], info: MapInfo
) -> npt.NDArray[np.float64]:
    world = np.empty_like(points, dtype=np.float64)
    scaled_x = points[:, 0] * info.resolution
    scaled_y = points[:, 1] * info.resolution
    cos_yaw = float(np.cos(info.origin_yaw))
    sin_yaw = float(np.sin(info.origin_yaw))
    world[:, 0] = info.origin_x + cos_yaw * scaled_x - sin_yaw * scaled_y
    world[:, 1] = info.origin_y + sin_yaw * scaled_x + cos_yaw * scaled_y
    return world


def _world_to_pixels(
    points: npt.NDArray[np.float64], info: MapInfo
) -> npt.NDArray[np.float64]:
    dx = points[:, 0] - info.origin_x
    dy = points[:, 1] - info.origin_y
    cos_yaw = float(np.cos(info.origin_yaw))
    sin_yaw = float(np.sin(info.origin_yaw))
    pixels = np.empty_like(points, dtype=np.float64)
    pixels[:, 0] = (cos_yaw * dx + sin_yaw * dy) / info.resolution
    pixels[:, 1] = (-sin_yaw * dx + cos_yaw * dy) / info.resolution
    return pixels


def _resample_closed_path(
    points: npt.NDArray[np.float64],
    spacing: float,
    closed: bool = True,
) -> npt.NDArray[np.float64]:
    if points.shape[0] < 2:
        return points

    path = np.vstack((points, points[0])) if closed else points
    segment_lengths = np.linalg.norm(np.diff(path, axis=0), axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(segment_lengths)))
    total_length = float(cumulative[-1])
    if total_length <= 0.0:
        return points

    sample_points = np.arange(0.0, total_length, spacing, dtype=np.float64)
    if not closed:
        sample_points = np.append(sample_points, total_length)
    x = np.interp(sample_points, cumulative, path[:, 0])
    y = np.interp(sample_points, cumulative, path[:, 1])
    return np.column_stack((x, y))


def _smooth_closed_path(
    points: npt.NDArray[np.float64], window: int = 5, closed: bool = True
) -> npt.NDArray[np.float64]:
    if points.shape[0] < window:
        return points

    half_window = window // 2
    if closed:
        padded = np.vstack((points[-half_window:], points, points[:half_window]))
    else:
        padded = np.vstack(
            (
                np.repeat(points[:1], half_window, axis=0),
                points,
                np.repeat(points[-1:], half_window, axis=0),
            )
        )
    kernel = np.ones(window, dtype=np.float64) / float(window)
    x = np.convolve(padded[:, 0], kernel, mode="valid")
    y = np.convolve(padded[:, 1], kernel, mode="valid")
    return np.column_stack((x, y))


def _neighboring_points(
    points: npt.NDArray[np.float64], closed: bool
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    if closed:
        return np.roll(points, 1, axis=0), np.roll(points, -1, axis=0)
    prev_points = np.vstack((points[:1], points[:-1]))
    next_points = np.vstack((points[1:], points[-1:]))
    return prev_points, next_points


def _compute_headings(
    points: npt.NDArray[np.float64], closed: bool = True
) -> npt.NDArray[np.float64]:
    _, next_points = _neighboring_points(points, closed)
    deltas = next_points - points
    if not closed and points.shape[0] > 1:
        deltas[-1] = deltas[-2]
    return np.arctan2(deltas[:, 1], deltas[:, 0])


@njit(cache=True, fastmath=True)
def _point_in_mask_numba(
    x: float,
    y: float,
    mask: np.ndarray,
    resolution: float,
    origin_x: float,
    origin_y: float,
    cos_yaw: float,
    sin_yaw: float,
) -> bool:
    dx = x - origin_x
    dy = y - origin_y
    pixel_x = int(np.rint((cos_yaw * dx + sin_yaw * dy) / resolution))
    pixel_y = int(np.rint((-sin_yaw * dx + cos_yaw * dy) / resolution))

    if pixel_x < 0 or pixel_x >= mask.shape[1]:
        return False
    if pixel_y < 0 or pixel_y >= mask.shape[0]:
        return False
    return bool(mask[pixel_y, pixel_x])


@njit(cache=True, fastmath=True)
def _raycast_to_boundary_numba(
    point_x: float,
    point_y: float,
    direction_x: float,
    direction_y: float,
    mask: np.ndarray,
    resolution: float,
    origin_x: float,
    origin_y: float,
    cos_yaw: float,
    sin_yaw: float,
    max_distance: float,
    num_samples: int,
) -> float:
    previous_distance = 0.0

    for sample_index in range(num_samples):
        distance = max_distance * sample_index / (num_samples - 1)
        x = point_x + distance * direction_x
        y = point_y + distance * direction_y
        inside = _point_in_mask_numba(
            x,
            y,
            mask,
            resolution,
            origin_x,
            origin_y,
            cos_yaw,
            sin_yaw,
        )

        if not inside:
            if sample_index == 0:
                return 0.0

            lower = previous_distance
            upper = distance
            for _ in range(10):
                middle = 0.5 * (lower + upper)
                middle_x = point_x + middle * direction_x
                middle_y = point_y + middle * direction_y
                middle_inside = _point_in_mask_numba(
                    middle_x,
                    middle_y,
                    mask,
                    resolution,
                    origin_x,
                    origin_y,
                    cos_yaw,
                    sin_yaw,
                )
                if middle_inside:
                    lower = middle
                else:
                    upper = middle
            return lower

        previous_distance = distance

    return max_distance


@njit(cache=True, fastmath=True)
def _compute_track_widths_numba(
    centerline: np.ndarray,
    mask: np.ndarray,
    resolution: float,
    origin_x: float,
    origin_y: float,
    origin_yaw: float,
    max_distance: float,
    num_samples: int,
) -> tuple[np.ndarray, np.ndarray]:
    point_count = centerline.shape[0]
    left_widths = np.empty(point_count, dtype=np.float64)
    right_widths = np.empty(point_count, dtype=np.float64)
    cos_yaw = np.cos(origin_yaw)
    sin_yaw = np.sin(origin_yaw)

    for index in range(point_count):
        previous_index = (index - 1) % point_count
        next_index = (index + 1) % point_count

        tangent_x = centerline[next_index, 0] - centerline[previous_index, 0]
        tangent_y = centerline[next_index, 1] - centerline[previous_index, 1]
        tangent_norm = np.sqrt(tangent_x * tangent_x + tangent_y * tangent_y)
        if tangent_norm < 1e-9:
            tangent_norm = 1e-9
        tangent_x /= tangent_norm
        tangent_y /= tangent_norm

        left_x = -tangent_y
        left_y = tangent_x
        right_x = tangent_y
        right_y = -tangent_x
        point_x = centerline[index, 0]
        point_y = centerline[index, 1]

        left_widths[index] = _raycast_to_boundary_numba(
            point_x,
            point_y,
            left_x,
            left_y,
            mask,
            resolution,
            origin_x,
            origin_y,
            cos_yaw,
            sin_yaw,
            max_distance,
            num_samples,
        )
        right_widths[index] = _raycast_to_boundary_numba(
            point_x,
            point_y,
            right_x,
            right_y,
            mask,
            resolution,
            origin_x,
            origin_y,
            cos_yaw,
            sin_yaw,
            max_distance,
            num_samples,
        )

    return left_widths, right_widths


def _compute_track_widths(
    centerline: npt.NDArray[np.float64],
    mask: npt.NDArray[np.bool_],
    info: MapInfo,
    closed: bool,
) -> TrackGeometry:
    """Measure left/right track widths at each centerline point."""
    if not closed:
        raise ValueError("Track width computation currently expects a closed path.")

    max_distance = float(np.hypot(mask.shape[0], mask.shape[1])) * info.resolution
    left_widths, right_widths = _compute_track_widths_numba(
        np.ascontiguousarray(centerline, dtype=np.float64),
        np.ascontiguousarray(mask, dtype=np.bool_),
        float(info.resolution),
        float(info.origin_x),
        float(info.origin_y),
        float(info.origin_yaw),
        max_distance,
        256,
    )

    total_widths = left_widths + right_widths
    return TrackGeometry(
        centerline=centerline,
        left_widths=left_widths,
        right_widths=right_widths,
        total_widths=total_widths,
        closed=closed,
    )


def _compute_normals(
    centerline: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    previous_points = np.roll(centerline, 1, axis=0)
    next_points = np.roll(centerline, -1, axis=0)
    tangents = next_points - previous_points
    tangent_norm = np.linalg.norm(tangents, axis=1, keepdims=True)
    tangent_norm = np.maximum(tangent_norm, 1e-9)
    tangents = tangents / tangent_norm
    return np.column_stack((-tangents[:, 1], tangents[:, 0]))


def _racing_line_from_offsets(
    centerline: npt.NDArray[np.float64],
    normals: npt.NDArray[np.float64],
    offsets: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    return centerline + offsets[:, None] * normals


def _segment_lengths(points: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    return np.linalg.norm(np.roll(points, -1, axis=0) - points, axis=1)


def _curvature_three_points(points: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    previous_points = np.roll(points, 1, axis=0)
    next_points = np.roll(points, -1, axis=0)
    a = points - previous_points
    b = next_points - points
    c = next_points - previous_points

    cross = a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]
    denom = np.linalg.norm(a, axis=1)
    denom = denom * np.linalg.norm(b, axis=1) * np.linalg.norm(c, axis=1)
    return 2.0 * cross / np.maximum(denom, 1e-8)


def _minimum_curvature_objective(
    offsets: npt.NDArray[np.float64],
    centerline: npt.NDArray[np.float64],
    normals: npt.NDArray[np.float64],
    lambda_length: float,
    lambda_smooth: float,
) -> float:
    race_line = _racing_line_from_offsets(centerline, normals, offsets)
    curvature = _curvature_three_points(race_line)
    segment_lengths = _segment_lengths(race_line)
    offset_second_difference = (
        np.roll(offsets, -1) - 2.0 * offsets + np.roll(offsets, 1)
    )

    cost_curvature = float(np.sum(curvature * curvature))
    cost_length = float(np.sum(segment_lengths))
    cost_smooth = float(np.sum(offset_second_difference * offset_second_difference))
    return cost_curvature + lambda_length * cost_length + lambda_smooth * cost_smooth


@njit(cache=True, fastmath=True)
def _minimum_curvature_objective_numba(
    offsets: np.ndarray,
    centerline: np.ndarray,
    normals: np.ndarray,
    lambda_length: float,
    lambda_smooth: float,
) -> float:
    point_count = offsets.shape[0]
    cost_curvature = 0.0
    cost_length = 0.0
    cost_smooth = 0.0

    for index in range(point_count):
        previous_index = (index - 1) % point_count
        next_index = (index + 1) % point_count

        previous_x = (
            centerline[previous_index, 0]
            + offsets[previous_index] * normals[previous_index, 0]
        )
        previous_y = (
            centerline[previous_index, 1]
            + offsets[previous_index] * normals[previous_index, 1]
        )
        current_x = centerline[index, 0] + offsets[index] * normals[index, 0]
        current_y = centerline[index, 1] + offsets[index] * normals[index, 1]
        next_x = (
            centerline[next_index, 0] + offsets[next_index] * normals[next_index, 0]
        )
        next_y = (
            centerline[next_index, 1] + offsets[next_index] * normals[next_index, 1]
        )

        ax = current_x - previous_x
        ay = current_y - previous_y
        bx = next_x - current_x
        by = next_y - current_y
        cx = next_x - previous_x
        cy = next_y - previous_y
        cross = ax * by - ay * bx

        norm_a = np.sqrt(ax * ax + ay * ay)
        norm_b = np.sqrt(bx * bx + by * by)
        norm_c = np.sqrt(cx * cx + cy * cy)
        denom = norm_a * norm_b * norm_c
        if denom < 1e-8:
            denom = 1e-8

        curvature = 2.0 * cross / denom
        cost_curvature += curvature * curvature
        cost_length += norm_b

        second_difference = (
            offsets[next_index] - 2.0 * offsets[index] + offsets[previous_index]
        )
        cost_smooth += second_difference * second_difference

    return cost_curvature + lambda_length * cost_length + lambda_smooth * cost_smooth


def _minimum_curvature_objective_fast(
    offsets: npt.NDArray[np.float64],
    centerline: npt.NDArray[np.float64],
    normals: npt.NDArray[np.float64],
    lambda_length: float,
    lambda_smooth: float,
) -> float:
    return float(
        _minimum_curvature_objective_numba(
            np.ascontiguousarray(offsets, dtype=np.float64),
            centerline,
            normals,
            lambda_length,
            lambda_smooth,
        )
    )


def _expand_control_offsets(
    control_offsets: npt.NDArray[np.float64],
    point_count: int,
) -> npt.NDArray[np.float64]:
    control_count = control_offsets.shape[0]
    control_positions = np.linspace(0.0, point_count, control_count + 1)
    full_positions = np.arange(point_count, dtype=np.float64)
    periodic_offsets = np.append(control_offsets, control_offsets[0])
    return np.interp(full_positions, control_positions, periodic_offsets)


def _minimum_curvature_control_objective(
    control_offsets: npt.NDArray[np.float64],
    centerline: npt.NDArray[np.float64],
    normals: npt.NDArray[np.float64],
    lower_offsets: npt.NDArray[np.float64],
    upper_offsets: npt.NDArray[np.float64],
    lambda_length: float,
    lambda_smooth: float,
) -> float:
    offsets = _expand_control_offsets(control_offsets, centerline.shape[0])
    offsets = np.clip(offsets, lower_offsets, upper_offsets)
    return _minimum_curvature_objective_fast(
        offsets,
        centerline,
        normals,
        lambda_length,
        lambda_smooth,
    )


def _optimize_minimum_curvature_line(
    track_geometry: TrackGeometry,
    margin: float,
    lambda_length: float,
    lambda_smooth: float,
    max_iterations: int,
    control_stride: int,
) -> npt.NDArray[np.float64]:
    """Optimize lateral offsets with box constraints from centerline width."""
    centerline = track_geometry.centerline
    normals = _compute_normals(centerline)
    lower_offsets = -track_geometry.right_widths + margin
    upper_offsets = track_geometry.left_widths - margin

    infeasible = lower_offsets > upper_offsets
    if np.any(infeasible):
        bad_count = int(np.count_nonzero(infeasible))
        raise ValueError(
            f"{bad_count} centerline points are infeasible with the current "
            "margin. Reduce car_width/safety_margin_extra or improve the "
            "centerline."
        )

    variable_stride = max(1, int(control_stride))
    control_indices = np.arange(0, centerline.shape[0], variable_stride, dtype=np.intp)
    use_control_offsets = control_indices.shape[0] < centerline.shape[0]

    if use_control_offsets:
        lower_bounds = lower_offsets[control_indices]
        upper_bounds = upper_offsets[control_indices]
        initial_offsets = np.clip(
            np.zeros(control_indices.shape[0], dtype=np.float64),
            lower_bounds,
            upper_bounds,
        )
        objective = _minimum_curvature_control_objective
        objective_args = (
            centerline,
            normals,
            lower_offsets,
            upper_offsets,
            lambda_length,
            lambda_smooth,
        )
    else:
        lower_bounds = lower_offsets
        upper_bounds = upper_offsets
        initial_offsets = np.clip(
            np.zeros(centerline.shape[0], dtype=np.float64),
            lower_bounds,
            upper_bounds,
        )
        objective = _minimum_curvature_objective_fast
        objective_args = (centerline, normals, lambda_length, lambda_smooth)

    bounds = [
        (float(lower), float(upper))
        for lower, upper in zip(lower_bounds, upper_bounds, strict=True)
    ]
    variable_count = initial_offsets.shape[0]

    centerline = np.ascontiguousarray(centerline, dtype=np.float64)
    normals = np.ascontiguousarray(normals, dtype=np.float64)
    lower_offsets = np.ascontiguousarray(lower_offsets, dtype=np.float64)
    upper_offsets = np.ascontiguousarray(upper_offsets, dtype=np.float64)
    if use_control_offsets:
        objective_args = (
            centerline,
            normals,
            lower_offsets,
            upper_offsets,
            lambda_length,
            lambda_smooth,
        )
    else:
        objective_args = (centerline, normals, lambda_length, lambda_smooth)

    result = minimize(
        objective,
        initial_offsets,
        args=objective_args,
        method="L-BFGS-B",
        bounds=bounds,
        options={
            "maxiter": max_iterations,
            "maxfun": max_iterations * max(1000, 20 * variable_count),
        },
    )
    if not result.success and "ITERATIONS REACHED LIMIT" not in str(result.message):
        raise RuntimeError(f"Optimization failed: {result.message}")
    if not result.success:
        print(
            f"warning: optimization stopped at the iteration limit; using the "
            f"best candidate found so far ({result.message})",
            file=sys.stderr,
        )

    result_offsets = cast(npt.NDArray[np.float64], result.x)
    if use_control_offsets:
        offsets = _expand_control_offsets(result_offsets, centerline.shape[0])
        offsets = np.clip(offsets, lower_offsets, upper_offsets)
    else:
        offsets = result_offsets
    return _racing_line_from_offsets(centerline, normals, offsets)


def _points_in_mask(
    points: npt.NDArray[np.float64],
    mask: npt.NDArray[np.bool_],
    info: MapInfo,
) -> npt.NDArray[np.bool_]:
    pixels = np.rint(_world_to_pixels(points, info)).astype(np.intp)
    valid = (
        (pixels[:, 0] >= 0)
        & (pixels[:, 0] < mask.shape[1])
        & (pixels[:, 1] >= 0)
        & (pixels[:, 1] < mask.shape[0])
    )
    inside = np.zeros(points.shape[0], dtype=np.bool_)
    inside[valid] = mask[pixels[valid, 1], pixels[valid, 0]]
    return inside


def _sample_polyline_segments(
    points: npt.NDArray[np.float64],
    samples_per_segment: int = 8,
) -> npt.NDArray[np.float64]:
    sampled_points: list[npt.NDArray[np.float64]] = []
    point_count = points.shape[0]
    for index in range(point_count):
        start = points[index]
        end = points[(index + 1) % point_count]
        alphas = np.linspace(0.0, 1.0, samples_per_segment, endpoint=False)
        for alpha in alphas:
            sampled_points.append((1.0 - alpha) * start + alpha * end)
    return np.asarray(sampled_points, dtype=np.float64)


def _validate_polyline_in_mask(
    points: npt.NDArray[np.float64],
    mask: npt.NDArray[np.bool_],
    info: MapInfo,
) -> None:
    segment_lengths = _segment_lengths(points)
    samples_per_segment = max(
        4,
        int(np.ceil(float(np.max(segment_lengths)) / (0.5 * info.resolution))),
    )
    samples = _sample_polyline_segments(points, samples_per_segment)
    if not np.all(_points_in_mask(samples, mask, info)):
        raise ValueError(
            "Generated race line leaves the drivable mask. Increase the "
            "safety margin, improve width bounds, or add collision penalties."
        )


def _project_points_to_mask(
    reference: npt.NDArray[np.float64],
    points: npt.NDArray[np.float64],
    mask: npt.NDArray[np.bool_],
    info: MapInfo,
) -> npt.NDArray[np.float64]:
    """Shrink invalid offsets toward a known-valid reference path."""
    projected = points.copy()
    inside = _points_in_mask(projected, mask, info)
    if np.all(inside):
        return projected

    for index in np.flatnonzero(~inside):
        for alpha in (0.75, 0.5, 0.25, 0.0):
            candidate = reference[index] + alpha * (points[index] - reference[index])
            candidate_points = np.asarray([candidate], dtype=np.float64)
            if bool(_points_in_mask(candidate_points, mask, info)[0]):
                projected[index] = candidate
                break
    return projected


@njit(cache=True, fastmath=True)
def _compute_speed_profile_numba(
    points: np.ndarray,
    max_speed: float,
    lateral_accel_limit: float,
    min_speed: float,
    acceleration_limit: float,
    braking_limit: float,
) -> np.ndarray:
    point_count = points.shape[0]
    segment_lengths = np.empty(point_count, dtype=np.float64)
    speed = np.empty(point_count, dtype=np.float64)

    for index in range(point_count):
        previous_index = (index - 1) % point_count
        next_index = (index + 1) % point_count

        previous_x = points[previous_index, 0]
        previous_y = points[previous_index, 1]
        current_x = points[index, 0]
        current_y = points[index, 1]
        next_x = points[next_index, 0]
        next_y = points[next_index, 1]

        ax = current_x - previous_x
        ay = current_y - previous_y
        bx = next_x - current_x
        by = next_y - current_y
        cx = next_x - previous_x
        cy = next_y - previous_y
        cross = ax * by - ay * bx

        norm_a = np.sqrt(ax * ax + ay * ay)
        norm_b = np.sqrt(bx * bx + by * by)
        norm_c = np.sqrt(cx * cx + cy * cy)
        denom = norm_a * norm_b * norm_c
        if denom < 1e-8:
            denom = 1e-8

        curvature = abs(2.0 * cross / denom)
        if curvature < 1e-6:
            curvature = 1e-6

        raw_speed = np.sqrt(lateral_accel_limit / curvature)
        if raw_speed < min_speed:
            raw_speed = min_speed
        elif raw_speed > max_speed:
            raw_speed = max_speed

        speed[index] = raw_speed
        segment_lengths[index] = norm_b

    for _ in range(100):
        max_change = 0.0

        for index in range(point_count):
            next_index = (index + 1) % point_count
            reachable_speed = np.sqrt(
                speed[index] * speed[index]
                + 2.0 * acceleration_limit * segment_lengths[index]
            )
            if reachable_speed < speed[next_index]:
                change = speed[next_index] - reachable_speed
                speed[next_index] = reachable_speed
                if change > max_change:
                    max_change = change

        for index in range(point_count - 1, -1, -1):
            previous_index = (index - 1) % point_count
            reachable_speed = np.sqrt(
                speed[index] * speed[index]
                + 2.0 * braking_limit * segment_lengths[previous_index]
            )
            if reachable_speed < speed[previous_index]:
                change = speed[previous_index] - reachable_speed
                speed[previous_index] = reachable_speed
                if change > max_change:
                    max_change = change

        if max_change < 1e-3:
            break

    return speed


def _compute_speed_profile(
    points: npt.NDArray[np.float64],
    max_speed: float,
    lateral_accel_limit: float,
    min_speed: float,
    acceleration_limit: float,
    braking_limit: float,
) -> npt.NDArray[np.float64]:
    return cast(
        npt.NDArray[np.float64],
        _compute_speed_profile_numba(
            np.ascontiguousarray(points, dtype=np.float64),
            max_speed,
            lateral_accel_limit,
            min_speed,
            acceleration_limit,
            braking_limit,
        ),
    )


def _write_waypoints(
    output_path: Path,
    points: npt.NDArray[np.float64],
    headings: npt.NDArray[np.float64],
    speeds: npt.NDArray[np.float64],
) -> None:
    with open(output_path, "w", encoding="utf-8", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["x", "y", "yaw", "speed"])
        for point, heading, speed in zip(points, headings, speeds, strict=True):
            writer.writerow(
                [
                    f"{point[0]:.6f}",
                    f"{point[1]:.6f}",
                    f"{heading:.6f}",
                    f"{speed:.6f}",
                ]
            )


def _plot(
    info: MapInfo,
    free: npt.NDArray[np.bool_],
    centerline: npt.NDArray[np.float64],
    race_line: npt.NDArray[np.float64],
    output_path: Path,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    fig, ax = plt.subplots(figsize=(7, 7))
    height, width = free.shape
    x_min = info.origin_x
    x_max = info.origin_x + float(width) * info.resolution
    y_min = info.origin_y
    y_max = info.origin_y + float(height) * info.resolution
    ax.imshow(
        free,
        cmap="gray",
        origin="lower",
        extent=(x_min, x_max, y_min, y_max),
    )
    ax.plot(centerline[:, 0], centerline[:, 1], linewidth=1.2, label="centerline")
    ax.plot(race_line[:, 0], race_line[:, 1], linewidth=1.5, label="generated")
    ax.set_title(output_path.name)
    ax.set_aspect("equal")
    ax.legend(loc="best")
    fig.tight_layout()
    plt.show()


def parse_args() -> argparse.Namespace:
    parser = HelpOnErrorParser(description=__doc__)
    parser.add_argument("--map_path", type=Path, required=True)
    parser.add_argument(
        "--map_ext",
        type=str,
        default=None,
        help=(
            "Optional image extension override. If omitted, the image path "
            "from the YAML is used."
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--spacing", type=float, default=0.2)
    parser.add_argument(
        "--prune_iterations",
        type=int,
        default=40,
        help="Number of endpoint-pruning passes for small skeleton branches.",
    )
    parser.add_argument("--max_speed", type=float, default=20.0)
    parser.add_argument("--min_speed", type=float, default=0.0)
    parser.add_argument("--lateral_accel_limit", type=float, default=10.0)
    parser.add_argument("--acceleration_limit", type=float, default=5.0)
    parser.add_argument("--braking_limit", type=float, default=8.0)
    parser.add_argument("--car_width", type=float, default=0.31)
    parser.add_argument("--safety_margin_extra", type=float, default=0.05)
    parser.add_argument("--lambda_length", type=float, default=0.01)
    parser.add_argument("--lambda_smooth", type=float, default=1.0)
    parser.add_argument("--max_iterations", type=int, default=1000)
    parser.add_argument(
        "--control_stride",
        type=int,
        default=4,
        help=(
            "Optimize every Nth lateral offset and interpolate between controls. "
            "Use 1 to optimize every waypoint."
        ),
    )
    parser.add_argument(
        "--timing",
        action="store_true",
        help="Print elapsed time for each major generation step.",
    )
    parser.add_argument(
        "--visualize",
        "--plot",
        dest="visualize",
        action="store_true",
        help="Show the map, centerline, and generated line after processing.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    timer = perf_counter()

    free, info = _load_map(args.map_path, args.map_ext)
    timer = _time_block("load map", timer, bool(args.timing))
    interior = _select_interior_component(free)
    timer = _time_block("select interior", timer, bool(args.timing))

    centerline_pixels = _extract_centerline(interior, int(args.prune_iterations))
    timer = _time_block("extract centerline", timer, bool(args.timing))
    endpoint_distance = np.linalg.norm(centerline_pixels[0] - centerline_pixels[-1])
    closed_path = bool(endpoint_distance <= 2.0)
    if not closed_path:
        raise ValueError("This racing-line optimizer currently expects a closed track.")

    centerline = _pixels_to_world(centerline_pixels, info)
    centerline = _resample_closed_path(centerline, args.spacing, closed=closed_path)
    centerline_reference = centerline.copy()
    centerline = _smooth_closed_path(centerline, closed=closed_path)
    centerline = _project_points_to_mask(
        centerline_reference, centerline, interior, info
    )

    track_geometry = _compute_track_widths(centerline, interior, info, closed_path)
    timer = _time_block("compute widths", timer, bool(args.timing))
    margin = 0.5 * float(args.car_width) + float(args.safety_margin_extra)
    race_line = _optimize_minimum_curvature_line(
        track_geometry=track_geometry,
        margin=margin,
        lambda_length=float(args.lambda_length),
        lambda_smooth=float(args.lambda_smooth),
        max_iterations=int(args.max_iterations),
        control_stride=int(args.control_stride),
    )
    timer = _time_block("optimize racing line", timer, bool(args.timing))
    _validate_polyline_in_mask(race_line, interior, info)
    timer = _time_block("validate line", timer, bool(args.timing))

    headings = _compute_headings(race_line, closed=closed_path)
    speeds = _compute_speed_profile(
        race_line,
        max_speed=float(args.max_speed),
        lateral_accel_limit=float(args.lateral_accel_limit),
        min_speed=float(args.min_speed),
        acceleration_limit=float(args.acceleration_limit),
        braking_limit=float(args.braking_limit),
    )
    timer = _time_block("compute speed profile", timer, bool(args.timing))

    print(
        f"Computed widths for {track_geometry.centerline.shape[0]} points: "
        f"min={float(np.min(track_geometry.total_widths)):.3f} m, "
        f"mean={float(np.mean(track_geometry.total_widths)):.3f} m, "
        f"max={float(np.max(track_geometry.total_widths)):.3f} m"
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    _write_waypoints(args.output, race_line, headings, speeds)

    if args.visualize:
        _plot(info, free, track_geometry.centerline, race_line, args.output)


if __name__ == "__main__":
    main()
