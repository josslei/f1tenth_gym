"""Parity test: C++ dynamics vs Python (numba) dynamics."""

import numpy as np

GRAVITY = 9.81


class TestDynamicsParity:
    def test_single_step_vs_python(self, rollout_kernel, default_params):
        C = rollout_kernel
        params = default_params

        init_state = C.F110State()
        init_state.x = 0.0
        init_state.y = 0.0
        init_state.steer_angle = 0.0
        init_state.velocity = 5.0
        init_state.yaw_angle = 0.0
        init_state.yaw_rate = 0.0
        init_state.slip_angle = 0.0
        init_state.steer_buffer_0 = 0.0
        init_state.steer_buffer_1 = 0.0
        init_state.steer_buffer_len = 0
        init_state.in_collision = False

        action = C.F110Action(0.5, 5.0)

        result = C.step(init_state, action, params, C.Integrator.RK4)
        cpp_state = np.array(
            [
                result.state.x,
                result.state.y,
                result.state.steer_angle,
                result.state.velocity,
                result.state.yaw_angle,
                result.state.yaw_rate,
                result.state.slip_angle,
            ]
        )

        py_state = self._python_step(init_state, action, params)
        np.testing.assert_allclose(
            cpp_state, py_state, atol=1e-8, err_msg="Single step mismatch"
        )

    def test_multi_step_vs_python(self, rollout_kernel, default_params):
        C = rollout_kernel
        params = default_params

        state = C.F110State()
        state.velocity = 3.0

        for i in range(100):
            steer = 0.3 * np.sin(i * 0.1)
            vel = 3.0 + 0.5 * np.sin(i * 0.05)
            action = C.F110Action(float(steer), float(vel))
            prev_state = state
            result = C.step(state, action, params, C.Integrator.RK4)
            state = result.state

            py_state = self._python_step(prev_state, action, params)
            cpp_arr = np.array(
                [
                    state.x,
                    state.y,
                    state.steer_angle,
                    state.velocity,
                    state.yaw_angle,
                    state.yaw_rate,
                    state.slip_angle,
                ]
            )
            np.testing.assert_allclose(
                cpp_arr, py_state, atol=1e-6, err_msg=f"Mismatch at step {i}"
            )

    def _python_step(self, cpp_state, action, params):
        s = cpp_state
        x = np.array(
            [
                s.x,
                s.y,
                s.steer_angle,
                s.velocity,
                s.yaw_angle,
                s.yaw_rate,
                s.slip_angle,
            ]
        )
        steer = 0.0 if s.steer_buffer_len < 2 else s.steer_buffer_1
        vel = action.velocity

        from gym.f110_gym.envs.dynamic_models import pid, vehicle_dynamics_st

        accl, sv = pid(
            vel,
            steer,
            s.velocity,
            s.steer_angle,
            params.sv_max,
            params.a_max,
            params.v_max,
            params.v_min,
        )
        u = np.array([sv, accl])

        def dyn_f(x):
            return vehicle_dynamics_st(
                x,
                u,
                params.mu,
                params.c_sf,
                params.c_sr,
                params.lf,
                params.lr,
                params.h,
                params.m,
                params.inertia,
                params.s_min,
                params.s_max,
                params.sv_min,
                params.sv_max,
                params.v_switch,
                params.a_max,
                params.v_min,
                params.v_max,
            )

        dt = params.timestep
        k1 = dyn_f(x)
        k2 = dyn_f(x + dt * (k1 / 2))
        k3 = dyn_f(x + dt * (k2 / 2))
        k4 = dyn_f(x + dt * k3)
        next_state = x + dt * (1 / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
        if next_state[4] > 2 * np.pi:
            next_state[4] -= 2 * np.pi
        elif next_state[4] < 0.0:
            next_state[4] += 2 * np.pi
        return next_state
