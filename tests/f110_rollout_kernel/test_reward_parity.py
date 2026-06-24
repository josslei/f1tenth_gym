"""Parity test: native F110ProgressReward vs Python reference implementation."""

from __future__ import annotations

import math

import numpy as np


class _PythonReward:
    def __init__(
        self,
        track_map,
        waypoints_xy: np.ndarray,
        q_progress: float = 1.0,
        q_alpha: float = 1.0,
        q_smooth: float = 0.0,
        terminal_penalty: float = 1000.0,
        alpha_th: float = 0.0,
        slip_terminal_penalty: float = 0.0,
        q_offtrack_grad: float = 0.0,
    ) -> None:
        self.track_map = track_map
        self.waypoints = np.asarray(waypoints_xy, dtype=np.float64).reshape(-1, 2)
        self.q_progress = q_progress
        self.q_alpha = q_alpha
        self.q_smooth = q_smooth
        self.terminal_penalty = terminal_penalty
        self.alpha_th = alpha_th
        self.slip_terminal_penalty = slip_terminal_penalty
        self.q_offtrack_grad = q_offtrack_grad
        self._build_arc()
        self.reset()

    def _build_arc(self) -> None:
        diffs = np.diff(self.waypoints, axis=0)
        seg_lengths = np.sqrt((diffs**2).sum(axis=1))
        self.cum_arc_lengths = np.concatenate([[0.0], np.cumsum(seg_lengths)])
        self.total_length = (
            float(self.cum_arc_lengths[-1]) if len(self.cum_arc_lengths) else 0.0
        )
        n = self.waypoints.shape[0]
        self.headings = np.empty(n, dtype=np.float64)
        for i in range(n):
            prev_i = n - 1 if i == 0 else i - 1
            next_i = (i + 1) % n
            dx = self.waypoints[next_i, 0] - self.waypoints[prev_i, 0]
            dy = self.waypoints[next_i, 1] - self.waypoints[prev_i, 1]
            self.headings[i] = math.atan2(dy, dx)

    def reset(self) -> None:
        self.prev_arc_length = 0.0
        self.last_action_0 = 0.0
        self.last_action_1 = 0.0
        self.has_last_action = False

    def _nearest_index(self, px: float, py: float) -> int:
        d2 = (self.waypoints[:, 0] - px) ** 2 + (self.waypoints[:, 1] - py) ** 2
        return int(np.argmin(d2))

    def _arclength_at(self, px: float, py: float) -> float:
        if self.waypoints.shape[0] < 2:
            return 0.0
        idx = self._nearest_index(px, py)
        n = self.waypoints.shape[0]
        prev_i = n - 1 if idx == 0 else idx - 1
        next_i = (idx + 1) % n

        def project(a: int, b: int) -> tuple[float, float]:
            sx, sy = self.waypoints[a]
            ex, ey = self.waypoints[b]
            dx = ex - sx
            dy = ey - sy
            len2 = dx * dx + dy * dy
            if len2 < 1e-12:
                rx = px - sx
                ry = py - sy
                return 0.0, rx * rx + ry * ry
            t = ((px - sx) * dx + (py - sy) * dy) / len2
            t = float(np.clip(t, 0.0, 1.0))
            proj_x = sx + t * dx
            proj_y = sy + t * dy
            rx = px - proj_x
            ry = py - proj_y
            return t * math.sqrt(len2), rx * rx + ry * ry

        p_prev = project(prev_i, idx)
        p_next = project(idx, next_i)
        if p_prev[1] < p_next[1]:
            return float(self.cum_arc_lengths[prev_i] + p_prev[0])
        return float(self.cum_arc_lengths[idx] + p_next[0])

    def _is_backward_terminal(self, px: float, py: float, theta: float) -> bool:
        ref_heading = self.headings[self._nearest_index(px, py)]
        hdiff = abs(theta - ref_heading)
        if hdiff > math.pi:
            hdiff = 2.0 * math.pi - hdiff
        return hdiff > math.pi / 2.0

    def __call__(
        self,
        px: float,
        py: float,
        theta: float,
        vx: float,
        vy: float,
        action_0: float,
        action_1: float,
        collision: bool,
        terminated: bool,
    ) -> float:
        del terminated

        if collision:
            return -50.0

        current_arc = self._arclength_at(px, py)
        delta_s = current_arc - self.prev_arc_length
        if self.total_length > 0.0:
            if delta_s < -0.5 * self.total_length:
                delta_s += self.total_length
            elif delta_s > 0.5 * self.total_length:
                delta_s -= self.total_length

        beta = math.atan2(vy, vx)
        speed = math.hypot(vx, vy)
        delta_0 = action_0 - self.last_action_0 if self.has_last_action else 0.0
        delta_1 = action_1 - self.last_action_1 if self.has_last_action else 0.0

        reward = self.q_progress * delta_s
        reward -= self.q_alpha * (beta * beta)
        reward -= self.q_smooth * (delta_0 * delta_0 + delta_1 * delta_1)

        distance = float(self.track_map.distance_at(px, py))
        off_real = float(np.clip(1.0 - distance, 0.0, 1.0))
        reward -= self.q_offtrack_grad * off_real

        if abs(beta) > self.alpha_th:
            reward -= self.slip_terminal_penalty

        if speed > 12.0:
            reward -= 10000.0

        if self._is_backward_terminal(px, py, theta):
            reward -= self.terminal_penalty

        self.prev_arc_length = current_arc
        self.last_action_0 = action_0
        self.last_action_1 = action_1
        self.has_last_action = True

        return reward


class TestRewardParity:
    def test_reward_matches_python(self, rollout_kernel, track_map, waypoints):
        C = rollout_kernel
        track, _ = track_map

        wx = waypoints[:, 0].astype(np.float64)
        wy = waypoints[:, 1].astype(np.float64)

        cpp_reward = C.F110ProgressReward(
            track,
            wx,
            wy,
            1.0,
            1.0,
            0.0,
            1000000.0,
            0.475,
            10000.0,
            1000000.0,
        )

        py_reward = _PythonReward(
            track,
            waypoints,
            q_progress=1.0,
            q_alpha=1.0,
            q_smooth=0.0,
            terminal_penalty=1000000.0,
            alpha_th=0.475,
            slip_terminal_penalty=10000.0,
            q_offtrack_grad=1000000.0,
        )

        test_cases = [
            (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, False, False),
            (1.0, 0.5, 0.1, 5.0, 0.1, 0.2, 0.5, False, False),
            (2.0, -1.0, 1.57, 8.0, 0.0, -0.3, 0.8, False, True),
            (-0.5, 1.5, -1.0, 3.0, 0.1, 0.1, -0.2, True, False),
            (0.0, 0.0, 3.2, 4.0, 0.0, 0.0, 0.0, False, False),
        ]

        for px, py, theta, vx, vy, a0, a1, coll, term in test_cases:
            cpp_val = cpp_reward(px, py, theta, vx, vy, a0, a1, coll, term)

            py_val = py_reward(px, py, theta, vx, vy, a0, a1, coll, term)

            assert np.isclose(
                cpp_val, py_val
            ), f"Reward mismatch at ({px},{py}): C++={cpp_val}, Python={py_val}"

    def test_reward_reset(self, rollout_kernel, track_map, waypoints):
        C = rollout_kernel
        track, _ = track_map

        wx = waypoints[:, 0].astype(np.float64)
        wy = waypoints[:, 1].astype(np.float64)

        cpp_reward = C.F110ProgressReward(track, wx, wy)

        val1 = cpp_reward(1.0, 0.5, 0.1, 5.0, 0.0, 0.2, 0.3, False, False)
        cpp_reward.reset()
        val2 = cpp_reward(1.0, 0.5, 0.1, 5.0, 0.0, 0.2, 0.3, False, False)
        assert abs(val1 - val2) < 1e-6

    def test_collision_penalty(self, rollout_kernel, track_map):
        C = rollout_kernel
        track, _ = track_map

        wx = np.array([0.0, 1.0], dtype=np.float64)
        wy = np.array([0.0, 0.0], dtype=np.float64)

        cpp_reward = C.F110ProgressReward(track, wx, wy)

        val = cpp_reward(0.5, 0.0, 0.0, 5.0, 0.0, 0.2, 0.3, True, True)
        assert val == -50.0
