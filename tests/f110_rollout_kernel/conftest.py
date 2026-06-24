"""Shared fixtures for rollout kernel parity tests."""

from importlib import import_module
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]

G = ROOT / "gym"
if str(G) not in sys.path:
    sys.path.insert(0, str(G))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

sys.modules.pop("f110_gym", None)
sys.modules.pop("f110_gym.envs", None)
import_module("f110_gym")


@pytest.fixture(scope="session")
def rollout_kernel():
    from f110_gym.rollout_kernel.natives import _f110_rollout_kernel as C

    return C


@pytest.fixture(scope="session")
def track_map(rollout_kernel):
    from f110_gym.envs.laser_models import ScanSimulator2D

    from utils.track_map import load_track_map

    map_path = str(ROOT / "maps" / "custom" / "f110_gym_10" / "f110_gym_map.yaml")
    map_ext = ".png"

    py_sim = ScanSimulator2D(1080, 4.7)
    py_sim.set_map(map_path, map_ext)

    backend_track, _, _ = load_track_map(map_path, map_ext)

    C = rollout_kernel
    track = C.TrackMap()
    track.height = backend_track.height
    track.width = backend_track.width
    track.resolution = backend_track.resolution
    track.orig_x = backend_track.orig_x
    track.orig_y = backend_track.orig_y
    track.orig_c = backend_track.orig_c
    track.orig_s = backend_track.orig_s
    track.dt = backend_track.dt
    track.theta_dis = backend_track.theta_dis
    track.num_beams = backend_track.num_beams
    track.fov = backend_track.fov
    track.max_range = backend_track.max_range
    track.eps = backend_track.eps
    track.ttc_thresh = backend_track.ttc_thresh
    track.side_distances = backend_track.side_distances
    track.compute_scan_tables()

    return track, py_sim


@pytest.fixture(scope="session")
def waypoints():
    csv_path = ROOT / "maps" / "custom" / "f110_gym_10" / "f110_gym_centerline.csv"
    wp = np.loadtxt(str(csv_path), delimiter=",", skiprows=1)[:, :2]
    return wp


@pytest.fixture(scope="session")
def default_params(rollout_kernel):
    return rollout_kernel.F110Params()
