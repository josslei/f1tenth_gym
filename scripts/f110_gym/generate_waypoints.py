# MIT License

# Copyright (c) 2020 Joseph Auckley, Matthew O'Kelly, Aman Sinha, Hongrui Zheng

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR IMPLIED, INCLUDING BUT
# NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
# PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
# HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
# ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
# WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

"""Generate waypoint CSV files from an F1TENTH occupancy map.

The script extracts the enclosed free-space component of the map, samples many
radial rays from the free-space centroid, and keeps the point of maximum
distance-to-obstacle along each ray. The resulting ordered loop is then
resampled into evenly spaced waypoints and written as:

    x, y, yaw, speed

The speed profile is a curvature-based heuristic, not a guaranteed globally
optimal racing line.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt
import yaml
from PIL import Image
from scipy import ndimage as ndi


@dataclass(frozen=True)
class MapInfo:
    resolution: float
    origin_x: float
    origin_y: float
    origin_yaw: float


def _load_map(map_path: Path, map_ext: str) -> tuple[npt.NDArray[np.bool_], MapInfo]:
    with open(map_path, "r", encoding="utf-8") as yaml_stream:
        metadata = yaml.safe_load(yaml_stream)

    resolution = float(metadata["resolution"])
    origin_x, origin_y, origin_yaw = metadata["origin"]
    image_path = map_path.with_suffix(map_ext)
    image = np.array(Image.open(image_path).transpose(Image.Transpose.FLIP_TOP_BOTTOM))
    if image.ndim == 3:
        image = image[..., 0]

    free = image > 128
    info = MapInfo(
        resolution=resolution,
        origin_x=float(origin_x),
        origin_y=float(origin_y),
        origin_yaw=float(origin_yaw),
    )
    return free, info


def _select_interior_component(free: npt.NDArray[np.bool_]) -> npt.NDArray[np.bool_]:
    labels, count = ndi.label(free, structure=ndi.generate_binary_structure(2, 2))
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


def _sample_centerline(
    interior: npt.NDArray[np.bool_],
    distance: npt.NDArray[np.float64],
    num_angles: int,
) -> npt.NDArray[np.float64]:
    ys, xs = np.nonzero(interior)
    center_x = float(xs.mean())
    center_y = float(ys.mean())

    max_radius = 0.5 * float(np.hypot(*interior.shape))
    radii = np.linspace(0.0, max_radius, num=max(1000, int(max_radius * 4)))

    points = np.empty((num_angles, 2), dtype=np.float64)
    for i, angle in enumerate(
        np.linspace(0.0, 2.0 * np.pi, num=num_angles, endpoint=False)
    ):
        direction_x = float(np.cos(angle))
        direction_y = float(np.sin(angle))
        xs_ray = center_x + radii * direction_x
        ys_ray = center_y + radii * direction_y

        inside = (
            (xs_ray >= 0.0)
            & (ys_ray >= 0.0)
            & (xs_ray < float(interior.shape[1]))
            & (ys_ray < float(interior.shape[0]))
        )
        if not np.any(inside):
            points[i, :] = (center_x, center_y)
            continue

        xs_inside = xs_ray[inside]
        ys_inside = ys_ray[inside]
        distance_samples = ndi.map_coordinates(
            distance, np.vstack((ys_inside, xs_inside)), order=1, mode="nearest"
        )
        best_index = int(np.argmax(distance_samples))
        points[i, 0] = xs_inside[best_index]
        points[i, 1] = ys_inside[best_index]

    return points


def _pixels_to_world(
    points: npt.NDArray[np.float64], info: MapInfo
) -> npt.NDArray[np.float64]:
    world = np.empty_like(points, dtype=np.float64)
    world[:, 0] = points[:, 0] * info.resolution + info.origin_x
    world[:, 1] = points[:, 1] * info.resolution + info.origin_y
    return world


def _resample_closed_path(
    points: npt.NDArray[np.float64],
    spacing: float,
) -> npt.NDArray[np.float64]:
    if points.shape[0] < 2:
        return points

    closed = np.vstack((points, points[0]))
    segment_lengths = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(segment_lengths)))
    total_length = float(cumulative[-1])
    if total_length <= 0.0:
        return points

    sample_points = np.arange(0.0, total_length, spacing, dtype=np.float64)
    x = np.interp(sample_points, cumulative, closed[:, 0])
    y = np.interp(sample_points, cumulative, closed[:, 1])
    return np.column_stack((x, y))


def _smooth_closed_path(
    points: npt.NDArray[np.float64], window: int = 5
) -> npt.NDArray[np.float64]:
    if points.shape[0] < window:
        return points

    half_window = window // 2
    padded = np.vstack((points[-half_window:], points, points[:half_window]))
    kernel = np.ones(window, dtype=np.float64) / float(window)
    x = np.convolve(padded[:, 0], kernel, mode="valid")
    y = np.convolve(padded[:, 1], kernel, mode="valid")
    return np.column_stack((x, y))


def _compute_headings(points: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    next_points = np.roll(points, -1, axis=0)
    deltas = next_points - points
    return np.arctan2(deltas[:, 1], deltas[:, 0])


def _compute_curvature(points: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    prev_points = np.roll(points, 1, axis=0)
    next_points = np.roll(points, -1, axis=0)

    a = np.linalg.norm(points - prev_points, axis=1)
    b = np.linalg.norm(next_points - points, axis=1)
    c = np.linalg.norm(next_points - prev_points, axis=1)

    twice_area = np.abs(
        (points[:, 0] - prev_points[:, 0]) * (next_points[:, 1] - prev_points[:, 1])
        - (points[:, 1] - prev_points[:, 1]) * (next_points[:, 0] - prev_points[:, 0])
    )
    denom = np.maximum(a * b * c, 1e-9)
    return 2.0 * twice_area / denom


def _compute_speed_profile(
    points: npt.NDArray[np.float64],
    max_speed: float,
    lateral_accel_limit: float,
) -> npt.NDArray[np.float64]:
    curvature = _compute_curvature(points)
    speed = np.sqrt(lateral_accel_limit / np.maximum(curvature, 1e-6))
    return np.clip(speed, 0.5, max_speed)


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
    free: npt.NDArray[np.bool_],
    centerline: npt.NDArray[np.float64],
    output_path: Path,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(free, cmap="gray", origin="lower")
    ax.plot(centerline[:, 0], centerline[:, 1], linewidth=1.5)
    ax.set_title(output_path.name)
    ax.set_aspect("equal")
    fig.tight_layout()
    plt.show()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map_path", type=Path, required=True)
    parser.add_argument("--map_ext", type=str, default=".png")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--spacing", type=float, default=0.2)
    parser.add_argument("--num_angles", type=int, default=720)
    parser.add_argument("--max_speed", type=float, default=6.0)
    parser.add_argument("--lateral_accel_limit", type=float, default=3.0)
    parser.add_argument("--plot", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    free, info = _load_map(args.map_path, args.map_ext)
    interior = _select_interior_component(free)
    distance = ndi.distance_transform_edt(interior) * info.resolution

    centerline = _sample_centerline(interior, distance, args.num_angles)
    centerline = _pixels_to_world(centerline, info)
    centerline = _resample_closed_path(centerline, args.spacing)
    centerline = _smooth_closed_path(centerline)

    headings = _compute_headings(centerline)
    speeds = _compute_speed_profile(
        centerline,
        max_speed=args.max_speed,
        lateral_accel_limit=args.lateral_accel_limit,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    _write_waypoints(args.output, centerline, headings, speeds)

    if args.plot:
        _plot(free, centerline, args.output)


if __name__ == "__main__":
    main()
