"""Generate a centerline CSV from a circular occupancy-grid map.

The script estimates the circle center and track thickness from the white
track pixels in a map image referenced by a ROS-style YAML file, then writes a
standard 4-column centerline CSV and can optionally save a plot:

    x_m, y_m, w_tr_right_m, w_tr_left_m
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
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
        help="Number of centerline samples around the loop",
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


def _estimate_circle(
    image: np.ndarray, resolution: float, origin: np.ndarray, threshold: float
) -> tuple[float, float, float, float]:
    gray = image.mean(axis=2)
    white_mask = gray >= threshold
    ys, xs = np.nonzero(white_mask)
    if xs.size == 0:
        raise ValueError("Could not find the white track region in the map image")

    cx_px = float(xs.mean())
    cy_px = float(ys.mean())
    dists = np.hypot(xs - cx_px, ys - cy_px) * resolution
    r_inner = float(np.percentile(dists, 1.0))
    r_outer = float(np.percentile(dists, 99.0))

    height = image.shape[0]
    center_x = float(origin[0] + cx_px * resolution)
    center_y = float(origin[1] + (height - cy_px) * resolution)
    radius = 0.5 * (r_inner + r_outer)
    half_width = 0.5 * (r_outer - r_inner)
    return center_x, center_y, radius, half_width


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
    center_x, center_y, radius, half_width = _estimate_circle(
        image, resolution, origin, args.white_threshold
    )

    theta = np.linspace(0.0, 2.0 * np.pi, args.num_points, endpoint=False)
    points = np.column_stack(
        (
            center_x + radius * np.cos(theta),
            center_y + radius * np.sin(theta),
            np.full(args.num_points, half_width, dtype=np.float64),
            np.full(args.num_points, half_width, dtype=np.float64),
        )
    )

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
