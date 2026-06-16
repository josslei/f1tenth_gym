"""Generate a closed-loop centerline CSV from an occupancy-grid map.

The script extracts the medial axis of the white track region in a map image
referenced by a ROS-style YAML file, then writes a standard 4-column
centerline CSV and can optionally save a plot:

    x_m, y_m, w_tr_right_m, w_tr_left_m
"""

from __future__ import annotations

import argparse
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
from scipy import ndimage
import yaml
from PIL import Image


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--map",
        required=True,
        type=Path,
        help="Path to a map YAML file (e.g. maps/f1tenth_maps/maps/circle.yaml)",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Output centerline CSV path",
    )
    parser.add_argument(
        "--num-points",
        type=int,
        default=16,
        help="Minimum number of centerline samples around the loop",
    )
    parser.add_argument(
        "--white-threshold",
        type=float,
        default=240.0,
        help="Grayscale threshold used to identify the white track region",
    )
    parser.add_argument(
        "--save_plot",
        action="store_true",
        help="Save a PNG visualization next to the output CSV",
    )
    return parser.parse_args()


def _load_map(map_yaml: Path) -> tuple[np.ndarray, float, np.ndarray]:
    meta = yaml.safe_load(map_yaml.read_text(encoding="utf-8"))
    image_path = map_yaml.with_name(meta["image"])
    image = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.float64)
    resolution = float(meta["resolution"])
    origin = np.asarray(meta["origin"][:2], dtype=np.float64)
    return image, resolution, origin


def _largest_component(mask: np.ndarray) -> np.ndarray:
    label_result: Any = ndimage.label(mask)
    labels = np.asarray(label_result[0])
    component_count = int(label_result[1])
    if component_count == 0:
        raise ValueError("Could not find the white track region in the map image")

    component_sizes = np.asarray(
        ndimage.sum(mask, labels, index=np.arange(1, component_count + 1)),
        dtype=np.float64,
    )
    largest_label = int(np.argmax(component_sizes) + 1)
    return labels == largest_label


def _binary_skeleton(mask: np.ndarray) -> np.ndarray:
    distance = np.asarray(ndimage.distance_transform_edt(mask), dtype=np.float64)
    ridge = distance == np.asarray(
        ndimage.maximum_filter(distance, size=3, mode="constant"), dtype=np.float64
    )
    ridge &= mask
    ridge &= distance > 0.0
    return ridge


def _prune_skeleton(mask: np.ndarray) -> np.ndarray:
    coords = np.column_stack(np.nonzero(mask))
    if coords.size == 0:
        raise ValueError("Could not extract a centerline from the map image")

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

    degrees = np.array([len(item) for item in neighbors], dtype=np.int32)
    active = np.ones(len(coords), dtype=bool)
    queue: deque[int] = deque(np.flatnonzero(degrees <= 1).tolist())
    while queue:
        idx = queue.popleft()
        if not active[idx] or degrees[idx] > 1:
            continue
        active[idx] = False
        for neighbor in neighbors[idx]:
            if active[neighbor]:
                degrees[neighbor] -= 1
                if degrees[neighbor] == 1:
                    queue.append(neighbor)

    pruned = np.zeros_like(mask)
    pruned[coords[active, 0], coords[active, 1]] = True
    return _largest_component(pruned)


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

    start = int(np.lexsort((coords[:, 1], coords[:, 0]))[0])
    ordered: list[int] = [start]
    visited = {start}
    previous = -1
    current = start
    max_steps = len(coords) + 1

    for _ in range(max_steps):
        candidates = [
            neighbor for neighbor in neighbors[current] if neighbor != previous
        ]
        if not candidates:
            break

        if previous == -1:
            next_idx = min(candidates, key=lambda idx: (coords[idx, 0], coords[idx, 1]))
        elif len(candidates) == 1:
            next_idx = candidates[0]
        else:
            prev_vec = coords[current] - coords[previous]
            prev_norm = float(np.hypot(prev_vec[0], prev_vec[1])) or 1.0

            def score(idx: int) -> tuple[float, float, float]:
                vec = coords[idx] - coords[current]
                vec_norm = float(np.hypot(vec[0], vec[1])) or 1.0
                cos_turn = float(np.dot(prev_vec, vec) / (prev_norm * vec_norm))
                return cos_turn, -float(coords[idx, 0]), -float(coords[idx, 1])

            next_idx = max(candidates, key=score)

        if next_idx == start and len(ordered) > 2:
            break
        if next_idx in visited:
            break

        ordered.append(next_idx)
        visited.add(next_idx)
        previous = current
        current = next_idx

    if len(ordered) < 3:
        raise ValueError("Could not order the extracted centerline loop")

    return coords[np.asarray(ordered, dtype=np.int64)]


def _sample_closed_path(points_xy: np.ndarray, num_points: int) -> np.ndarray:
    closed_points = np.vstack([points_xy, points_xy[:1]])
    segments = np.diff(closed_points, axis=0)
    segment_lengths = np.sqrt((segments**2).sum(axis=1))
    cumulative = np.concatenate([[0.0], np.cumsum(segment_lengths)])
    total_length = float(cumulative[-1])
    if total_length == 0.0:
        return points_xy.astype(np.float64, copy=True)

    target_lengths = np.linspace(0.0, total_length, num_points, endpoint=False)
    resampled = np.empty((num_points, 2), dtype=np.float64)
    for axis in range(2):
        resampled[:, axis] = np.interp(
            target_lengths, cumulative, closed_points[:, axis]
        )
    return resampled


def _estimate_centerline(
    image: np.ndarray,
    resolution: float,
    origin: np.ndarray,
    threshold: float,
    min_points: int,
) -> tuple[np.ndarray, np.ndarray]:
    gray = image.mean(axis=2)
    white_mask = gray >= threshold
    white_mask = _largest_component(white_mask)

    distance = ndimage.distance_transform_edt(white_mask)
    skeleton = _binary_skeleton(white_mask)
    skeleton = _prune_skeleton(skeleton)
    ordered_pixels_yx = _trace_loop(skeleton)

    ordered_pixels_xy = ordered_pixels_yx[:, ::-1].astype(np.float64)
    closed_ordered_pixels_xy = np.vstack([ordered_pixels_xy, ordered_pixels_xy[:1]])
    loop_length_px = float(
        np.sum(np.linalg.norm(np.diff(closed_ordered_pixels_xy, axis=0), axis=1))
    )
    target_spacing_m = 0.5
    sample_count = max(
        min_points, int(np.ceil((loop_length_px * resolution) / target_spacing_m))
    )
    sampled_pixels_xy = _sample_closed_path(ordered_pixels_xy, sample_count)

    height = image.shape[0]
    world_xy = np.column_stack(
        (
            origin[0] + sampled_pixels_xy[:, 0] * resolution,
            origin[1] + (height - sampled_pixels_xy[:, 1]) * resolution,
        )
    )

    sampled_half_widths = ndimage.map_coordinates(
        distance,
        [sampled_pixels_xy[:, 1], sampled_pixels_xy[:, 0]],
        order=1,
        mode="nearest",
    )
    sampled_half_widths = np.maximum(sampled_half_widths * resolution, resolution * 0.5)

    if world_xy.shape[0] == 0:
        raise ValueError("Could not find the white track region in the map image")

    widths = np.column_stack((sampled_half_widths, sampled_half_widths))
    return world_xy, widths


def _save_plot(
    image: np.ndarray,
    resolution: float,
    origin: np.ndarray,
    points: np.ndarray,
    save_path: Path,
) -> None:
    import matplotlib.pyplot as plt

    gray = image.mean(axis=2)
    height, width = gray.shape
    extent = (
        float(origin[0]),
        float(origin[0] + width * resolution),
        float(origin[1]),
        float(origin[1] + height * resolution),
    )

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(gray[::-1], cmap="gray", origin="lower", extent=extent)
    ax.plot(points[:, 0], points[:, 1], color="tab:red", linewidth=1.5)
    ax.set_aspect("equal", adjustable="box")
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = _parse_args()
    image, resolution, origin = _load_map(args.map)
    centerline_xy, widths = _estimate_centerline(
        image,
        resolution,
        origin,
        args.white_threshold,
        args.num_points,
    )

    points = np.column_stack((centerline_xy, widths))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(
        args.output,
        points,
        delimiter=", ",
        header="x_m, y_m, w_tr_right_m, w_tr_left_m",
        comments="# ",
        fmt="%.7f",
    )

    if args.save_plot:
        _save_plot(
            image=image,
            resolution=resolution,
            origin=origin,
            points=np.vstack([points[:, :2], points[:1, :2]]),
            save_path=args.output.with_suffix(".png"),
        )


if __name__ == "__main__":
    main()
