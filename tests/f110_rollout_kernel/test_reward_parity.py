"""Parity test: native reward vs `utils.f110_reward`."""

from __future__ import annotations

import numpy as np

from utils.f110_reward import F1TenthProgressReward


def _obs(
    *,
    vx: float = 0.0,
    vy: float = 0.0,
    collision: bool = False,
    theta: float = 0.0,
    action_0: float = 0.0,
    action_1: float = 0.0,
    px: float = 0.0,
    py: float = 0.0,
):
    return {
        "ego_idx": 0,
        "linear_vels_x": np.array([vx], dtype=np.float64),
        "linear_vels_y": np.array([vy], dtype=np.float64),
        "collisions": np.array([collision], dtype=bool),
        "poses_theta": np.array([theta], dtype=np.float64),
        "poses_x": np.array([px], dtype=np.float64),
        "poses_y": np.array([py], dtype=np.float64),
        "prev_action": np.array([action_0, action_1], dtype=np.float64),
    }


class TestRewardParity:
    def test_reward_matches_python(self, rollout_kernel, track_map, waypoints):
        C = rollout_kernel
        track, _ = track_map

        wx = waypoints[:, 0].astype(np.float64)
        wy = waypoints[:, 1].astype(np.float64)

        cpp_reward = C.F110ProgressReward(
            track,
            wx.tolist(),
            wy.tolist(),
            1.0,
            1.0,
            0.0,
            1000000.0,
            0.475,
            10000.0,
            0.0,
        )

        py_reward = F1TenthProgressReward(
            track_map=track,
            q_s_progress=1.0,
            q_s_alpha=1.0,
            q_s_smooth=0.0,
            terminal_penalty=1000000.0,
            alpha_th=0.475,
            slip_terminal_penalty=10000.0,
            q_offtrack_grad=0.0,
        )
        py_reward.set_waypoints(waypoints)

        test_cases = [
            (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, False, False),
            (1.0, 0.5, 0.1, 5.0, 0.1, 0.2, 0.5, False, False),
            (2.0, -1.0, 1.57, 8.0, 0.0, -0.3, 0.8, False, True),
            (-0.5, 1.5, -1.0, 3.0, 0.1, 0.1, -0.2, True, False),
            (0.0, 0.0, 3.2, 4.0, 0.0, 0.0, 0.0, False, False),
        ]

        for px, py, theta, vx, vy, a0, a1, coll, term in test_cases:
            obs = _obs(
                px=px,
                py=py,
                theta=theta,
                vx=vx,
                vy=vy,
                action_0=a0,
                action_1=a1,
                collision=coll,
            )
            cpp_val = cpp_reward(px, py, theta, vx, vy, a0, a1, coll, term)
            py_val = py_reward(obs, term)

            assert np.isclose(
                cpp_val, py_val
            ), f"Reward mismatch at ({px},{py}): C++={cpp_val}, Python={py_val}"

    def test_reward_reset(self, rollout_kernel, track_map, waypoints):
        C = rollout_kernel
        track, _ = track_map

        wx = waypoints[:, 0].astype(np.float64)
        wy = waypoints[:, 1].astype(np.float64)

        cpp_reward = C.F110ProgressReward(track, wx.tolist(), wy.tolist())
        py_reward = F1TenthProgressReward(track_map=track)
        py_reward.set_waypoints(waypoints)

        obs = _obs(px=1.0, py=0.5, theta=0.1, vx=5.0, vy=0.0, action_0=0.2)

        val1 = cpp_reward(1.0, 0.5, 0.1, 5.0, 0.0, 0.2, 0.3, False, False)
        py_val1 = py_reward(obs, False)

        cpp_reward.reset()
        py_reward.reset()

        val2 = cpp_reward(1.0, 0.5, 0.1, 5.0, 0.0, 0.2, 0.3, False, False)
        py_val2 = py_reward(obs, False)

        assert np.isclose(val1, py_val1)
        assert np.isclose(val2, py_val2)

    def test_backward_terminal_penalty(self, rollout_kernel, track_map, waypoints):
        C = rollout_kernel
        track, _ = track_map

        wx = waypoints[:, 0].astype(np.float64)
        wy = waypoints[:, 1].astype(np.float64)

        cpp_reward = C.F110ProgressReward(track, wx.tolist(), wy.tolist())
        py_reward = F1TenthProgressReward(track_map=track)
        py_reward.set_waypoints(waypoints)

        obs = _obs(px=0.5, py=0.0, theta=3.2, vx=4.0, vy=0.0)
        cpp_val = cpp_reward(0.5, 0.0, 3.2, 4.0, 0.0, 0.0, 0.0, False, False)
        py_val = py_reward(obs, False)

        assert np.isclose(cpp_val, py_val)
