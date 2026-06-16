"""Generate a closed-loop centerline CSV from an occupancy-grid map."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

_RACELINE_OPT = Path(__file__).resolve().parent / "raceline_opt"
sys.path.insert(0, str(_RACELINE_OPT))

from helper_funcs_glob.src.map_centerline import centerline_from_map  # noqa: E402


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
        default=None,
        help="Optional minimum number of centerline samples around the loop",
    )
    parser.add_argument(
        "--target-spacing",
        type=float,
        default=0.5,
        help="Target spacing between generated centerline samples in meters",
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=None,
        help="Optional maximum number of generated centerline samples",
    )
    parser.add_argument(
        "--white-threshold",
        type=float,
        default=210.0,
        help="Grayscale threshold used to identify free track pixels",
    )
    parser.add_argument(
        "--centerline-threshold",
        type=float,
        default=0.17,
        help="Distance-transform fraction used before skeletonizing the track center",
    )
    parser.add_argument(
        "--track-width-margin",
        type=float,
        default=0.0,
        help="Safety margin subtracted from generated left/right widths in meters",
    )
    parser.add_argument(
        "--save_plot",
        action="store_true",
        help="Save a PNG visualization next to the output CSV",
    )
    return parser.parse_args()


def _save_plot(
    image: np.ndarray,
    resolution: float,
    origin: np.ndarray,
    points: np.ndarray,
    save_path: Path,
) -> None:
    import matplotlib.pyplot as plt

    height, width = image.shape
    extent = (
        float(origin[0]),
        float(origin[0] + width * resolution),
        float(origin[1]),
        float(origin[1] + height * resolution),
    )

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(image[::-1], cmap="gray", origin="lower", extent=extent)
    ax.plot(points[:, 0], points[:, 1], color="tab:red", linewidth=1.5)
    ax.set_aspect("equal", adjustable="box")
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = _parse_args()
    centerline = centerline_from_map(
        args.map,
        min_points=args.num_points,
        target_spacing=args.target_spacing,
        max_points=args.max_points,
        free_threshold=args.white_threshold,
        centerline_threshold=args.centerline_threshold,
        track_width_margin=args.track_width_margin,
    )
    points = np.column_stack((centerline.points_xy, centerline.widths))

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
            image=centerline.image,
            resolution=centerline.resolution,
            origin=centerline.origin,
            points=np.vstack([points[:, :2], points[:1, :2]]),
            save_path=args.output.with_suffix(".png"),
        )


if __name__ == "__main__":
    main()
