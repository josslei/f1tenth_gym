import numpy as np
import pytest
from f110_gym.rollout_kernel.natives import _f110_rollout_kernel as C

from f110_gym.envs.dynamic_models import integrate_rk4, pid, vehicle_dynamics_st


def _python_step(state, action, params, integrator, direct_accel_control=False):
    steer, velocity = action
    accl, sv = pid(
        state[3] if direct_accel_control else velocity,
        0.0,  # first command is held by RaceCar's two-step steering delay
        state[3],
        state[2],
        params.sv_max,
        params.a_max,
        params.v_max,
        params.v_min,
    )
    if direct_accel_control:
        accl = velocity
    control = np.array([sv, accl], dtype=np.float64)

    if integrator == "rk4":
        next_state = integrate_rk4(
            state,
            control,
            params.timestep,
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
@pytest.mark.parametrize("direct_accel_control", [False, True])
@pytest.mark.parametrize("timestep", [0.01, 0.025])
def test_rollout_kernel_matches_python_simulator_step(
    integrator, direct_accel_control, timestep
):
    default_params = C.F110Params()
    default_params.timestep = timestep
    state = C.F110State()
    state.x = 0.5
    state.y = 0.0
    state.steer_angle = 0.03
    state.velocity = 4.5
    state.yaw_angle = 0.2
    state.yaw_rate = 0.0
    state.slip_angle = 0.0

    longitudinal_command = 2.0 if direct_accel_control else 5.5
    action = C.F110Action(0.12, longitudinal_command)

    cpp_result = C.step(
        state,
        action,
        default_params,
        C.Integrator.RK4 if integrator == "rk4" else C.Integrator.Euler,
        direct_accel_control,
    )

    py_state = np.array([0.5, 0.0, 0.03, 4.5, 0.2, 0.0, 0.0], dtype=np.float64)
    py_next = _python_step(
        py_state,
        np.array([0.12, longitudinal_command]),
        default_params,
        integrator,
        direct_accel_control,
    )

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


def test_rk4_remains_stable_while_accelerating_through_low_speed():
    params = C.F110Params()
    params.timestep = 0.025
    state = C.F110State()
    state.steer_angle = 0.2
    state.velocity = 0.45

    for _ in range(80):
        result = C.step(
            state,
            C.F110Action(0.2, 0.5),
            params,
            C.Integrator.RK4,
            True,
        )
        state = result.state

        values = np.array(
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
        assert np.all(np.isfinite(values))
        assert abs(state.yaw_rate) < 5.0
        assert abs(state.slip_angle) < 1.0

    assert state.velocity == pytest.approx(1.45)
