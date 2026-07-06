"""Generate a Racing-LMPC trajectory table from a centerline-width CSV.

The output matches the 17-column table consumed by Racing-LMPC-ROS2's
``RacingTrajectory`` class and is written as whitespace-delimited numeric text
without a header so CasADi ``DM::from_file(...).T()`` can load it directly.
"""

from __future__ import annotations

import argparse
import configparser
import json
from pathlib import Path
from typing import Any

import numpy as np


TRAJECTORY_COLUMNS = (
    "PX",
    "PY",
    "PZ",
    "YAW",
    "SPEED",
    "CURVATURE",
    "DIST_TO_SF_BWD",
    "DIST_TO_SF_FWD",
    "REGION",
    "LEFT_BOUND_X",
    "LEFT_BOUND_Y",
    "RIGHT_BOUND_X",
    "RIGHT_BOUND_Y",
    "BANK",
    "LON_ACC",
    "LAT_ACC",
    "TIME",
)


def _default_params_path() -> Path:
    return Path("configs/raceline/f110.ini")


def load_raceline_params(params_path: Path) -> dict[str, Any]:
    cfg = configparser.ConfigParser()
    if not cfg.read(str(params_path)):
        raise FileNotFoundError(f"Could not read parameter config: {params_path}")
    return {
        "stepsize_opts": json.loads(cfg.get("GENERAL_OPTIONS", "stepsize_opts")),
        "veh_params": json.loads(cfg.get("GENERAL_OPTIONS", "veh_params")),
        "optim_opts_mintime": json.loads(
            cfg.get("OPTIMIZATION_OPTIONS", "optim_opts_mintime")
        ),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "track",
        type=Path,
        help="Centerline CSV with x_m, y_m, w_tr_right_m, w_tr_left_m columns",
    )
    parser.add_argument(
        "-o",
        "--output",
        required=True,
        type=Path,
        help="Output Racing-LMPC trajectory table path",
    )
    parser.add_argument(
        "--params",
        type=Path,
        default=_default_params_path(),
        help="Raceline parameter config (default: configs/raceline/f110.ini)",
    )
    parser.add_argument(
        "--spacing",
        type=float,
        default=None,
        help=(
            "Output spacing in meters. Defaults to "
            "GENERAL_OPTIONS.stepsize_opts.stepsize_interp_after_opt."
        ),
    )
    parser.add_argument(
        "--speed-mode",
        choices=("curvature", "constant"),
        default="curvature",
        help="Speed profile mode (default: curvature)",
    )
    parser.add_argument(
        "--constant-speed",
        type=float,
        default=None,
        help="Constant speed for --speed-mode constant. Defaults to veh_params.v_max.",
    )
    parser.add_argument(
        "--curvature-eps",
        type=float,
        default=1.0e-6,
        help="Curvature epsilon used in curvature-limited speed calculation.",
    )
    return parser.parse_args()


def _load_track_csv(track_path: Path) -> np.ndarray:
    raw = np.loadtxt(track_path, delimiter=",", skiprows=1, dtype=np.float64)
    raw = np.atleast_2d(raw)
    if np.allclose(raw[0, :2], raw[-1, :2]):
        raw = raw[:-1]
    return raw


def _closed_segment_lengths(points: np.ndarray) -> np.ndarray:
    next_points = np.roll(points, -1, axis=0)
    return np.linalg.norm(next_points - points, axis=1)


def _resample_closed_track(
    track: np.ndarray, spacing: float
) -> tuple[np.ndarray, np.ndarray]:
    points = track[:, :2]
    widths = track[:, 2:4]
    segment_lengths = _closed_segment_lengths(points)
    total_length = float(np.sum(segment_lengths))
    sample_count = max(4, int(np.ceil(total_length / spacing)))
    s_new = np.linspace(0.0, total_length, sample_count, endpoint=False)
    s_nodes = np.r_[0.0, np.cumsum(segment_lengths)]
    points_closed = np.vstack((points, points[:1]))
    widths_closed = np.vstack((widths, widths[:1]))

    columns = [
        np.interp(s_new, s_nodes, points_closed[:, 0]),
        np.interp(s_new, s_nodes, points_closed[:, 1]),
        np.interp(s_new, s_nodes, widths_closed[:, 0]),
        np.interp(s_new, s_nodes, widths_closed[:, 1]),
    ]
    return np.column_stack(columns), s_new


def _heading(points: np.ndarray) -> np.ndarray:
    prev_points = np.roll(points, 1, axis=0)
    next_points = np.roll(points, -1, axis=0)
    tangents = next_points - prev_points
    return np.arctan2(tangents[:, 1], tangents[:, 0])


def _curvature(yaw: np.ndarray, s: np.ndarray, total_length: float) -> np.ndarray:
    yaw_ext = np.r_[yaw[-1], yaw, yaw[0]]
    yaw_ext = np.unwrap(yaw_ext)
    ds = float(np.mean(np.diff(np.r_[s, total_length])))
    return (yaw_ext[2:] - yaw_ext[:-2]) / (2.0 * ds)


def _boundaries(points: np.ndarray, width_right: np.ndarray, width_left: np.ndarray):
    next_points = np.roll(points, -1, axis=0)
    tangents = next_points - points
    normals = np.column_stack((-tangents[:, 1], tangents[:, 0]))
    lengths = np.linalg.norm(normals, axis=1)
    normals /= lengths[:, np.newaxis]
    right = points + width_right[:, np.newaxis] * normals
    left = points - width_left[:, np.newaxis] * normals
    return left, right


def _speed_profile(
    curvature: np.ndarray,
    segment_lengths: np.ndarray,
    params: dict[str, Any],
    speed_mode: str,
    constant_speed: float | None,
    curvature_eps: float,
) -> np.ndarray:
    veh_params = params["veh_params"]
    optim_opts = params["optim_opts_mintime"]
    v_max = float(veh_params["v_max"])
    if speed_mode == "constant":
        speed = v_max if constant_speed is None else constant_speed
        return np.full(curvature.shape, min(float(speed), v_max), dtype=np.float64)

    ay_limit = optim_opts["ay_safe"]
    if ay_limit is None:
        ay_limit = float(optim_opts["mue"]) * float(veh_params["g"])
    speed = np.sqrt(float(ay_limit) / (np.abs(curvature) + curvature_eps))
    speed = np.minimum(speed, v_max)

    ax_pos_limit = optim_opts["ax_pos_safe"]
    if ax_pos_limit is None:
        ax_pos_limit = float(optim_opts["mue"]) * float(veh_params["g"])
    ax_neg_limit = optim_opts["ax_neg_safe"]
    if ax_neg_limit is None:
        ax_neg_limit = float(optim_opts["mue"]) * float(veh_params["g"])
    return _apply_acceleration_limits(
        speed,
        segment_lengths,
        ax_pos_limit=float(ax_pos_limit),
        ax_neg_limit=abs(float(ax_neg_limit)),
    )


def _apply_acceleration_limits(
    speed: np.ndarray,
    segment_lengths: np.ndarray,
    ax_pos_limit: float,
    ax_neg_limit: float,
) -> np.ndarray:
    limited = speed.copy()
    for _ in range(limited.size):
        for i in range(limited.size):
            j = (i + 1) % limited.size
            max_next = np.sqrt(
                limited[i] * limited[i] + 2.0 * ax_pos_limit * segment_lengths[i]
            )
            limited[j] = min(limited[j], max_next)
        for i in range(limited.size - 1, -1, -1):
            j = (i + 1) % limited.size
            max_current = np.sqrt(
                limited[j] * limited[j] + 2.0 * ax_neg_limit * segment_lengths[i]
            )
            limited[i] = min(limited[i], max_current)
    return limited


def build_lmpc_trajectory_table(
    track: np.ndarray,
    params: dict[str, Any],
    spacing: float | None = None,
    speed_mode: str = "curvature",
    constant_speed: float | None = None,
    curvature_eps: float = 1.0e-6,
) -> np.ndarray:
    if spacing is None:
        spacing = float(params["stepsize_opts"]["stepsize_interp_after_opt"])
    resampled, s = _resample_closed_track(track, spacing)
    points = resampled[:, :2]
    width_right = resampled[:, 2]
    width_left = resampled[:, 3]
    segment_lengths = _closed_segment_lengths(points)
    total_length = float(np.sum(segment_lengths))

    yaw = _heading(points)
    kappa = _curvature(yaw, s, total_length)
    speed = _speed_profile(
        kappa, segment_lengths, params, speed_mode, constant_speed, curvature_eps
    )
    left_bound, right_bound = _boundaries(points, width_right, width_left)

    ds_di = np.r_[np.diff(s), total_length - s[-1]]
    dv_ds = np.gradient(speed, s, edge_order=2)
    lon_acc = speed * dv_ds
    lat_acc = speed * speed * kappa
    dt = ds_di / np.maximum(speed, 1.0e-6)
    time = np.r_[0.0, np.cumsum(dt[:-1])]

    return np.column_stack(
        (
            points[:, 0],
            points[:, 1],
            np.zeros_like(s),
            yaw,
            speed,
            kappa,
            s,
            total_length - s,
            np.zeros_like(s),
            left_bound[:, 0],
            left_bound[:, 1],
            right_bound[:, 0],
            right_bound[:, 1],
            np.zeros_like(s),
            lon_acc,
            lat_acc,
            time,
        )
    )


def main() -> None:
    args = _parse_args()
    params = load_raceline_params(args.params)
    track = _load_track_csv(args.track)
    table = build_lmpc_trajectory_table(
        track,
        params,
        spacing=args.spacing,
        speed_mode=args.speed_mode,
        constant_speed=args.constant_speed,
        curvature_eps=args.curvature_eps,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(args.output, table, fmt="%.16e")
    print(f"Wrote {table.shape[0]} rows x {table.shape[1]} columns to {args.output}")


if __name__ == "__main__":
    main()
