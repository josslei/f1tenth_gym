"""Parity test: C++ build_observation vs Python build_observation."""

import numpy as np


class TestObservationParity:
    def test_observation_dim_matches(self, rollout_kernel):
        C = rollout_kernel
        config = C.ObservationConfig()
        config.scan_size = 108
        config.include_ego_state = True
        config.include_waypoints = True
        config.lookahead_distances = [
            0.5,
            1.0,
            2.0,
            3.5,
            5.5,
            8.0,
            11.0,
            14.5,
            18.5,
            23.0,
            28.0,
            33.0,
        ]
        config.waypoint_scale = 30.0

        from utils.f110_env import (
            F1TenthObservationConfig,
            observation_dim as py_obs_dim,
        )

        py_config = F1TenthObservationConfig(
            scan_size=108,
            include_ego_state=True,
            include_waypoints=True,
            lookahead_distances=(
                0.5,
                1.0,
                2.0,
                3.5,
                5.5,
                8.0,
                11.0,
                14.5,
                18.5,
                23.0,
                28.0,
                33.0,
            ),
        )
        cpp_dim = C.observation_dim(config)
        py_dim = py_obs_dim(py_config)
        assert (
            cpp_dim == py_dim
        ), f"Observation dim mismatch: C++={cpp_dim}, Python={py_dim}"

    def test_build_observation_matches_python(
        self, rollout_kernel, track_map, waypoints
    ):
        C = rollout_kernel
        config = C.ObservationConfig()
        config.scan_size = 108
        config.scan_max_m = 30.0
        config.include_ego_state = True
        config.speed_scale = 8.0
        config.yaw_rate_scale = 10.0
        config.steer_scale = 1.066
        config.include_waypoints = True
        config.lookahead_distances = [
            0.5,
            1.0,
            2.0,
            3.5,
            5.5,
            8.0,
            11.0,
            14.5,
            18.5,
            23.0,
            28.0,
            33.0,
        ]
        config.waypoint_scale = 30.0
        config.waypoint_resample_spacing = 0.5

        from utils.waypoint_utils import resample_path

        waypoints = resample_path(waypoints, config.waypoint_resample_spacing)

        diffs = np.diff(waypoints, axis=0)
        seg_lengths = np.sqrt((diffs**2).sum(axis=1))
        cum_arc = np.concatenate([[0.0], np.cumsum(seg_lengths)])

        cpp_map = track_map[0]
        scan = C.get_scan(0.5, 0.0, 0.0, cpp_map)
        prev_action = np.array([0.0, 0.0], dtype=np.float32)

        cpp_obs = C.build_observation(
            scan,
            0.5,
            0.0,
            0.0,
            5.0,
            0.0,
            0.0,
            0.1,
            False,
            prev_action,
            np.ascontiguousarray(waypoints[:, 0], dtype=np.float64),
            np.ascontiguousarray(waypoints[:, 1], dtype=np.float64),
            np.ascontiguousarray(cum_arc, dtype=np.float64),
            config,
        )

        from utils.f110_env import (
            F1TenthObservationConfig,
            build_observation as py_build_obs,
        )

        py_config = F1TenthObservationConfig(
            scan_size=108,
            scan_max_m=30.0,
            include_ego_state=True,
            speed_scale=8.0,
            yaw_rate_scale=10.0,
            steer_scale=1.066,
            include_waypoints=True,
            lookahead_distances=(
                0.5,
                1.0,
                2.0,
                3.5,
                5.5,
                8.0,
                11.0,
                14.5,
                18.5,
                23.0,
                28.0,
                33.0,
            ),
            waypoint_scale=30.0,
            waypoint_resample_spacing=0.5,
            _waypoints=waypoints,
        )

        obs_dict = {
            "ego_idx": 0,
            "scans": np.array([scan]),
            "poses_x": np.array([0.5]),
            "poses_y": np.array([0.0]),
            "poses_theta": np.array([0.0]),
            "linear_vels_x": np.array([5.0]),
            "linear_vels_y": np.array([0.0]),
            "ang_vels_z": np.array([0.0]),
            "collisions": np.array([False]),
            "steer_angle": np.array([0.1]),
            "prev_action": np.array([0.0, 0.0]),
        }

        py_obs = py_build_obs(obs_dict, py_config)
        np.testing.assert_allclose(
            np.array(cpp_obs),
            py_obs,
            atol=1e-5,
            err_msg="Observation vector mismatch",
        )
