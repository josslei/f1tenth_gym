import numpy as np
import pytest
from f110_gym.rollout_kernel.natives import _f110_rollout_kernel as C

from f110_gym.envs.dynamic_models import pid, vehicle_dynamics_st


def _python_step(state, action, params, integrator):
    steer, velocity = action
    accl, sv = pid(
        velocity,
        steer,
        state[3],
        state[2],
        params.sv_max,
        params.a_max,
        params.v_max,
        params.v_min,
    )
    control = np.array([sv, accl], dtype=np.float64)

    if integrator == "rk4":
        dt = params.timestep
        k1 = vehicle_dynamics_st(
            state,
            control,
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
        k2 = vehicle_dynamics_st(
            state + dt * (k1 / 2.0),
            control,
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
        k3 = vehicle_dynamics_st(
            state + dt * (k2 / 2.0),
            control,
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
        k4 = vehicle_dynamics_st(
            state + dt * k3,
            control,
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
        next_state = state + dt * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
    else:
        next_state = state + params.timestep * vehicle_dynamics_st(
            state,
            control,
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

    if next_state[4] > 2.0 * np.pi:
        next_state[4] -= 2.0 * np.pi
    elif next_state[4] < 0.0:
        next_state[4] += 2.0 * np.pi

    return next_state


@pytest.mark.parametrize("integrator", ["euler", "rk4"])
def test_rollout_kernel_matches_python_simulator_step(integrator):
    default_params = C.F110Params()
    state = C.F110State()
    state.x = 0.5
    state.y = 0.0
    state.steer_angle = 0.03
    state.velocity = 4.5
    state.yaw_angle = 0.2
    state.yaw_rate = 0.0
    state.slip_angle = 0.0

    action = C.F110Action(0.12, 5.5)

    cpp_result = C.step(
        state,
        action,
        default_params,
        C.Integrator.RK4 if integrator == "rk4" else C.Integrator.Euler,
    )

    py_state = np.array([0.5, 0.0, 0.03, 4.5, 0.2, 0.0, 0.0], dtype=np.float64)
    py_next = _python_step(py_state, np.array([0.12, 5.5]), default_params, integrator)

    np.testing.assert_allclose(
        np.array(
            [
                cpp_result.state.x,
                cpp_result.state.y,
                cpp_result.state.steer_angle,
                cpp_result.state.velocity,
                cpp_result.state.yaw_angle,
                cpp_result.state.yaw_rate,
                cpp_result.state.slip_angle,
            ],
            dtype=np.float64,
        ),
        py_next,
        atol=1e-10,
        err_msg="Native step state mismatch",
    )
    assert cpp_result.reward == pytest.approx(default_params.timestep)
    assert cpp_result.discount == pytest.approx(1.0)
    assert cpp_result.terminal is False
