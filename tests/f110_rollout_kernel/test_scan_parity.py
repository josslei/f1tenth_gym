"""Parity test: C++ get_scan vs Python ScanSimulator2D.get_scan."""

import numpy as np


class TestScanParity:
    def test_single_scan_matches(self, rollout_kernel, track_map):
        cpp_map, py_sim = track_map
        poses = [
            (0.0, 0.0, 0.0),
            (1.0, 0.5, 0.3),
            (2.0, -1.0, 1.57),
            (-0.5, 1.5, -1.0),
        ]
        for px, py, theta in poses:
            cpp_scan = rollout_kernel.get_scan(px, py, theta, cpp_map)
            py_pose = np.array([px, py, theta], dtype=np.float64)
            py_scan = py_sim.scan(py_pose, rng=None)
            np.testing.assert_allclose(
                cpp_scan,
                py_scan,
                atol=1e-4,
                err_msg=f"Mismatch at pose ({px},{py},{theta})",
            )

    def test_batch_scan_matches(self, rollout_kernel, track_map):
        cpp_map, py_sim = track_map
        poses_np = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.5, 0.3],
                [2.0, -1.0, 1.57],
                [-0.5, 1.5, -1.0],
            ],
            dtype=np.float64,
        )

        cpp_scans = rollout_kernel.get_scan_batch(poses_np, cpp_map)

        for i in range(len(poses_np)):
            py_pose = poses_np[i]
            py_scan = py_sim.scan(py_pose, rng=None)
            np.testing.assert_allclose(
                cpp_scans[i], py_scan, atol=1e-4, err_msg=f"Batch mismatch at pose {i}"
            )

    def test_check_ttc(self, rollout_kernel, track_map):
        cpp_map, py_sim = track_map
        scan = rollout_kernel.get_scan(0.0, 0.0, 0.0, cpp_map)
        vel = 5.0
        result = rollout_kernel.check_ttc(scan, vel, cpp_map)
        assert isinstance(result, bool)

    def test_trace_ray_edge_cases(self, rollout_kernel, track_map):
        cpp_map, py_sim = track_map
        scan = rollout_kernel.get_scan(-999.0, -999.0, 0.0, cpp_map)
        assert np.all(scan > 0)

    def test_scan_zero_pose(self, rollout_kernel, track_map):
        cpp_map, py_sim = track_map
        cpp_scan = rollout_kernel.get_scan(0.0, 0.0, 0.0, cpp_map)
        py_scan = py_sim.scan(np.array([0.0, 0.0, 0.0]), rng=None)
        np.testing.assert_allclose(
            cpp_scan, py_scan, atol=1e-4, err_msg="Zero-pose scan mismatch"
        )
