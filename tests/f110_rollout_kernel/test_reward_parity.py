"""Parity test: C++ F110ProgressReward vs Python F1TenthProgressReward."""

import numpy as np


class TestRewardParity:
    def test_reward_matches_python(self, rollout_kernel, waypoints):
        C = rollout_kernel

        wx = waypoints[:, 0].astype(np.float64)
        wy = waypoints[:, 1].astype(np.float64)

        cpp_reward = C.F110ProgressReward(wx, wy, 0.1, 2.0, 0.5, 50.0, 100.0)

        from utils.f110_reward import F1TenthProgressReward

        py_reward = F1TenthProgressReward(
            speed_reward_weight=0.1,
            progress_weight=2.0,
            steer_smoothness_weight=0.5,
            collision_penalty=50.0,
            spin_threshold=100.0,
        )
        py_reward.set_waypoints(waypoints)

        test_cases = [
            (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, False, False),
            (1.0, 0.5, 0.1, 5.0, 0.0, 0.2, False, False),
            (2.0, -1.0, 1.57, 8.0, 0.0, -0.3, False, True),
            (-0.5, 1.5, -1.0, 3.0, 0.0, 0.1, True, False),
            (0.0, 0.0, 3.2, 4.0, 0.0, 0.0, False, False),
        ]

        for px, py, theta, vx, vy, steer, coll, term in test_cases:
            cpp_val = cpp_reward(px, py, theta, vx, vy, steer, coll, term)

            obs = {
                "ego_idx": 0,
                "poses_x": np.array([px]),
                "poses_y": np.array([py]),
                "poses_theta": np.array([theta]),
                "linear_vels_x": np.array([vx]),
                "linear_vels_y": np.array([vy]),
                "ang_vels_z": np.array([0.0]),
                "collisions": np.array([coll]),
                "steer_angle": np.array([steer]),
            }
            py_val = py_reward(obs, term)

            assert (
                abs(cpp_val - py_val) < 1e-6
            ), f"Reward mismatch at ({px},{py}): C++={cpp_val}, Python={py_val}"

    def test_reward_reset(self, rollout_kernel, waypoints):
        C = rollout_kernel

        wx = waypoints[:, 0].astype(np.float64)
        wy = waypoints[:, 1].astype(np.float64)

        cpp_reward = C.F110ProgressReward(wx, wy, 0.1, 2.0, 0.5, 50.0, 100.0)

        val1 = cpp_reward(1.0, 0.5, 0.1, 5.0, 0.0, 0.2, False, False)
        cpp_reward.reset()
        val2 = cpp_reward(1.0, 0.5, 0.1, 5.0, 0.0, 0.2, False, False)
        assert abs(val1 - val2) < 1e-6, "Reset should produce identical rewards"

    def test_collision_penalty(self, rollout_kernel):
        C = rollout_kernel

        wx = np.array([0.0, 1.0], dtype=np.float64)
        wy = np.array([0.0, 0.0], dtype=np.float64)

        cpp_reward = C.F110ProgressReward(wx, wy, 0.1, 2.0, 0.5, 50.0, 100.0)

        val = cpp_reward(0.5, 0.0, 0.0, 5.0, 0.0, 0.2, True, True)
        assert val == -50.0, f"Collision should give -50.0, got {val}"
