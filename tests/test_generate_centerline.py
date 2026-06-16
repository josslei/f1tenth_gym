from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

from scripts.generate_centerline import main as generate_centerline_main
from utils.f110_env import (
    F1TenthObservationConfig,
    build_observation,
    with_resampled_waypoints,
)
from utils.waypoint_view import initial_pose_from_waypoints


MAP_YAML = Path("maps/custom/f110_gym_10/f110_gym_map.yaml")
F1_AUT_WIDE_YAML = Path("maps/f1tenth_maps/maps/f1_aut_wide.yaml")


def _reset_obs() -> dict[str, np.ndarray]:
    return {
        "ego_idx": np.array(0),
        "scans": np.full((1, 1080), 30.0, dtype=np.float64),
        "poses_x": np.zeros(1, dtype=np.float64),
        "poses_y": np.zeros(1, dtype=np.float64),
        "poses_theta": np.zeros(1, dtype=np.float64),
        "linear_vels_x": np.zeros(1, dtype=np.float64),
        "linear_vels_y": np.zeros(1, dtype=np.float64),
        "ang_vels_z": np.zeros(1, dtype=np.float64),
        "steer_angle": np.zeros(1, dtype=np.float64),
        "collisions": np.zeros(1, dtype=np.float64),
    }


def test_generate_centerline_produces_closed_loop_for_ppo(tmp_path: Path, monkeypatch):
    output = tmp_path / "centerline.csv"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "generate_centerline.py",
            "--map",
            str(MAP_YAML),
            "--output",
            str(output),
        ],
    )

    generate_centerline_main()

    centerline = np.loadtxt(output, delimiter=",", skiprows=1, dtype=np.float64)
    centerline = np.atleast_2d(centerline)

    assert centerline.shape[1] == 4
    assert centerline.shape[0] > 20

    points_xy = centerline[:, :2]
    segment_lengths = np.linalg.norm(
        np.diff(np.vstack([points_xy, points_xy[:1]]), axis=0), axis=1
    )
    closure_gap = float(np.linalg.norm(points_xy[0] - points_xy[-1]))

    assert closure_gap > 0.0
    assert closure_gap < 3.0 * float(np.median(segment_lengths))

    meta = yaml.safe_load(MAP_YAML.read_text(encoding="utf-8"))
    image = np.asarray(Image.open(MAP_YAML.with_name(meta["image"])).convert("RGB"))
    white_y, white_x = np.nonzero(image.mean(axis=2) >= 240.0)
    white_span = np.array(
        [white_x.max() - white_x.min(), white_y.max() - white_y.min()],
        dtype=np.float64,
    )
    centerline_span = np.ptp(points_xy, axis=0)

    assert centerline_span[0] > 0.6 * white_span[0] * float(meta["resolution"])
    assert centerline_span[1] > 0.6 * white_span[1] * float(meta["resolution"])

    pose = initial_pose_from_waypoints(points_xy)[0]
    obs_config = with_resampled_waypoints(
        F1TenthObservationConfig(
            include_waypoints=True,
            lookahead_distances=(0.5, 1.0, 2.0),
        ),
        points_xy,
    )
    obs = _reset_obs()
    obs["poses_x"][0] = pose[0]
    obs["poses_y"][0] = pose[1]
    obs["poses_theta"][0] = pose[2]

    observation = build_observation(obs, obs_config)

    assert observation.shape[0] == 108 + 7 + 6 + 2
    assert np.isfinite(observation).all()


def test_generate_centerline_handles_reference_notebook_map_conditions(
    tmp_path: Path, monkeypatch
):
    map_image = np.full((90, 90), 205, dtype=np.uint8)
    map_image[10:80, 10:80] = 0
    map_image[16:74, 16:74] = 255
    map_image[34:56, 34:56] = 0

    image_path = tmp_path / "synthetic.png"
    Image.fromarray(map_image).save(image_path)
    map_yaml = tmp_path / "synthetic.yaml"
    map_yaml.write_text(
        yaml.safe_dump(
            {
                "image": image_path.name,
                "resolution": 0.1,
                "origin": [-4.5, -4.5, 0.0],
                "negate": 0,
                "occupied_thresh": 0.45,
                "free_thresh": 0.196,
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "centerline.csv"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "generate_centerline.py",
            "--map",
            str(map_yaml),
            "--output",
            str(output),
            "--num-points",
            "120",
        ],
    )

    generate_centerline_main()

    centerline = np.loadtxt(output, delimiter=",", skiprows=1, dtype=np.float64)
    centerline = np.atleast_2d(centerline)

    assert centerline.shape == (120, 4)
    assert np.isfinite(centerline).all()
    assert np.all(centerline[:, 2:] > 0.0)
    assert np.linalg.norm(centerline[0, :2] - centerline[-1, :2]) < 1.0


def test_generate_centerline_spans_f1_aut_wide_track(tmp_path: Path, monkeypatch):
    output = tmp_path / "f1_aut_wide_centerline.csv"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "generate_centerline.py",
            "--map",
            str(F1_AUT_WIDE_YAML),
            "--output",
            str(output),
        ],
    )

    generate_centerline_main()

    centerline = np.loadtxt(output, delimiter=",", skiprows=1, dtype=np.float64)
    centerline = np.atleast_2d(centerline)

    meta = yaml.safe_load(F1_AUT_WIDE_YAML.read_text(encoding="utf-8"))
    image = np.asarray(
        Image.open(F1_AUT_WIDE_YAML.with_name(meta["image"])).convert("L")
    )
    white_y, white_x = np.nonzero(image >= 210)
    white_span = np.array(
        [white_x.max() - white_x.min(), white_y.max() - white_y.min()],
        dtype=np.float64,
    )
    centerline_span = np.ptp(centerline[:, :2], axis=0) / float(meta["resolution"])

    assert centerline.shape[1] == 4
    assert centerline.shape[0] > 100
    assert np.all(centerline[:, 2:] > 0.0)
    assert centerline_span[0] > 0.55 * white_span[0]
    assert centerline_span[1] > 0.55 * white_span[1]
