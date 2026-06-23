"""Shared fixtures for rollout kernel parity tests."""

from importlib import import_module
import math
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

    map_path = str(ROOT / "maps" / "custom" / "f110_gym_10" / "f110_gym_map.yaml")
    map_ext = ".png"

    py_sim = ScanSimulator2D(1080, 4.7)
    py_sim.set_map(map_path, map_ext)

    C = rollout_kernel
    track = C.TrackMap()
    track.height = py_sim.map_height
    track.width = py_sim.map_width
    track.resolution = py_sim.map_resolution
    track.orig_x = py_sim.orig_x
    track.orig_y = py_sim.orig_y
    track.orig_c = py_sim.orig_c
    track.orig_s = py_sim.orig_s
    track.dt = py_sim.dt.ravel().tolist()
    track.theta_dis = py_sim.theta_dis
    track.num_beams = py_sim.num_beams
    track.fov = py_sim.fov
    track.max_range = py_sim.max_range
    track.eps = py_sim.eps
    track.compute_scan_tables()

    car_length = 0.58
    car_width = 0.31
    side_dist = 0.5 * math.sqrt(car_length**2 + car_width**2)
    track.side_distances = [float(side_dist)] * track.num_beams

    return track, py_sim


@pytest.fixture(scope="session")
def waypoints():
    csv_path = ROOT / "maps" / "custom" / "f110_gym_10" / "f110_gym_centerline.csv"
    wp = np.loadtxt(str(csv_path), delimiter=",", skiprows=1)[:, :2]
    return wp


@pytest.fixture(scope="session")
def default_params(rollout_kernel):
    return rollout_kernel.F110Params()
