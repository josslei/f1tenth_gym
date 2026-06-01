"""Generate a YAML + PNG occupancy-grid map from a track centerline CSV.

The input CSV must have columns x_m; y_m; w_tr_right_m; w_tr_left_m (with
or without a header).  Outputs <track_stem>.yaml and <track_stem>.png in the
output directory, suitable as a custom map for the gym environment.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import yaml
from PIL import Image, ImageDraw


def _track_boundary(track, side):
    """Compute left or right boundary using the same tangent/normal logic
    as the raceline plot in optimize_mintime.py."""
    pts = track[:, :2]
    tangents = np.diff(pts, axis=0)
    tangents = np.vstack([tangents, tangents[0]])
    normals = np.column_stack([-tangents[:, 1], tangents[:, 0]])
    lengths = np.hypot(normals[:, 0], normals[:, 1])
    lengths = np.where(lengths == 0, 1.0, lengths)
    normals /= lengths[:, np.newaxis]
    if side == "right":
        return pts + track[:, 2:3] * normals
    return pts - track[:, 3:4] * normals


def _to_image_coords(points, x_min, y_min, resolution, height_px):
    """Convert world coords to image pixel coords (x, y with y flipped)."""
    px = (points[:, 0] - x_min) / resolution
    py = height_px - (points[:, 1] - y_min) / resolution
    return list(zip(px, py))


def _rasterize_track(centerline, width_right, width_left, yaml_path, resolution):
    """Create a single-channel PNG where only the track boundaries are black."""
    pts = np.vstack([centerline, centerline[:1]])
    wr = np.append(width_right, width_right[0])
    wl = np.append(width_left, width_left[0])
    track = np.column_stack([pts, wr, wl])
    left = _track_boundary(track, "left")
    right = _track_boundary(track, "right")

    margin = max(width_right.max(), width_left.max()) * 4
    x_min = float(min(left[:, 0].min(), right[:, 0].min()) - margin)
    y_min = float(min(left[:, 1].min(), right[:, 1].min()) - margin)
    x_max = float(max(left[:, 0].max(), right[:, 0].max()) + margin)
    y_max = float(max(left[:, 1].max(), right[:, 1].max()) + margin)

    origin = [float(x_min), float(y_min), 0.0]

    width_px = max(1, int(np.ceil((x_max - x_min) / resolution)))
    height_px = max(1, int(np.ceil((y_max - y_min) / resolution)))

    img = Image.new("L", (width_px, height_px), 255)
    draw = ImageDraw.Draw(img)

    line_width = max(1, int(np.ceil(1.0 / resolution)))
    left_pts = _to_image_coords(left, x_min, y_min, resolution, height_px)
    right_pts = _to_image_coords(right, x_min, y_min, resolution, height_px)
    draw.line(left_pts, fill=0, width=line_width)
    draw.line(right_pts, fill=0, width=line_width)

    png_path = yaml_path.with_suffix(".png")
    img.save(png_path)

    meta = {
        "image": png_path.name,
        "resolution": resolution,
        "origin": origin,
        "negate": 0,
        "occupied_thresh": 0.45,
        "free_thresh": 0.196,
    }
    with open(yaml_path, "w") as f:
        yaml.dump(meta, f, default_flow_style=False)


def main():
    parser = argparse.ArgumentParser(description="Generate a map from a track CSV")
    parser.add_argument("track", type=Path, help="Path to track CSV")
    parser.add_argument(
        "-o", "--output", type=Path, help="Output YAML path (default: track stem)"
    )
    parser.add_argument(
        "-r",
        "--resolution",
        type=float,
        default=0.05,
        help="Meters per pixel (default: 0.05)",
    )
    args = parser.parse_args()

    raw = np.loadtxt(args.track, delimiter=",", skiprows=1, dtype=np.float64)
    raw = np.atleast_2d(raw)

    centerline = raw[:, :2]
    width_right = raw[:, 2]
    width_left = raw[:, 3]

    if args.output:
        yaml_path = args.output.with_suffix(".yaml")
    else:
        yaml_path = args.track.with_suffix(".yaml")

    _rasterize_track(
        centerline,
        width_right,
        width_left,
        yaml_path,
        args.resolution,
    )
    print(f"Map written to {yaml_path} and {yaml_path.with_suffix('.png')}")


if __name__ == "__main__":
    main()
