from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from scripts.generate_centerline import main as generate_centerline_main
from utils.f110_env import (
    F1TenthObservationConfig,
    build_observation,
    with_resampled_waypoints,
)
from utils.waypoint_view import initial_pose_from_waypoints


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
            "maps/custom/f110_gym_10/f110_gym_map.yaml",
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
