"""Parity test: C++ GJK collision vs Python collision."""

import numpy as np


class TestCollisionParity:
    def test_get_vertices_matches_python(self, rollout_kernel):
        C = rollout_kernel

        from gym.f110_gym.envs.collision_models import get_vertices as py_get_vertices

        cases = [
            (0.0, 0.0, 0.0, 0.58, 0.31),
            (1.0, 0.5, 0.3, 0.58, 0.31),
            (-2.0, 3.0, 1.57, 0.58, 0.31),
        ]
        for x, y, theta, length, width in cases:
            cpp_v = C.get_vertices(x, y, theta, length, width)
            py_v = py_get_vertices(np.array([x, y, theta]), length, width)
            np.testing.assert_allclose(
                cpp_v, py_v, atol=1e-8, err_msg=f"Vertices mismatch at ({x},{y})"
            )

    def test_gjk_overlap(self, rollout_kernel):
        C = rollout_kernel
        length = 0.58
        width = 0.31

        v1 = C.get_vertices(0.0, 0.0, 0.0, length, width)
        v2 = C.get_vertices(0.3, 0.0, 0.0, length, width)
        assert C.gjk_collision(v1, v2), "Overlapping cars should collide"

    def test_gjk_no_overlap(self, rollout_kernel):
        C = rollout_kernel
        length = 0.58
        width = 0.31

        v1 = C.get_vertices(0.0, 0.0, 0.0, length, width)
        v2 = C.get_vertices(10.0, 10.0, 0.0, length, width)
        assert not C.gjk_collision(v1, v2), "Far apart cars should not collide"

    def test_gjk_touching(self, rollout_kernel):
        C = rollout_kernel
        length = 0.58
        width = 0.31

        v1 = C.get_vertices(0.0, 0.0, 0.0, length, width)
        v2 = C.get_vertices(0.0, width, 0.0, length, width)
        assert not C.gjk_collision(
            v1, v2
        ), "Touching edge case: GJK should match Python (touching != collision)"

    def test_gjk_matches_python(self, rollout_kernel):
        C = rollout_kernel

        from gym.f110_gym.envs.collision_models import (
            collision as py_collision,
            get_vertices as py_get_vertices,
        )

        cases = [
            (0.0, 0.0, 0.0, 0.3, 0.0, 0.0, True),
            (0.0, 0.0, 0.0, 10.0, 10.0, 0.0, False),
            (0.0, 0.0, 0.0, 0.0, 0.31, 0.0, True),
            (1.0, 1.0, 0.5, -1.0, -1.0, 0.5, False),
        ]
        for x1, y1, t1, x2, y2, t2, expected in cases:
            v1 = C.get_vertices(x1, y1, t1, 0.58, 0.31)
            v2 = C.get_vertices(x2, y2, t2, 0.58, 0.31)
            cpp_result = C.gjk_collision(v1, v2)

            py_v1 = py_get_vertices(np.array([x1, y1, t1]), 0.58, 0.31)
            py_v2 = py_get_vertices(np.array([x2, y2, t2]), 0.58, 0.31)
            py_result = py_collision(py_v1, py_v2)
            assert cpp_result == py_result, (
                f"GJK mismatch at ({x1},{y1}) vs ({x2},{y2}): "
                f"C++={cpp_result}, Python={py_result}"
            )
