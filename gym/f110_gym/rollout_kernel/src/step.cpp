#include "step.hpp"

#include "dynamics.hpp"

#include <cmath>

namespace f110_rollout_kernel {
namespace {

constexpr double kTwoPi = 6.28318530717958647692;

StateVector to_vector(const F110State &state) {
  return StateVector{state.x,         state.y,         state.steer_angle,
                     state.velocity,  state.yaw_angle, state.yaw_rate,
                     state.slip_angle};
}

void from_vector(F110State &state, const StateVector &values) {
  state.x = values[0];
  state.y = values[1];
  state.steer_angle = values[2];
  state.velocity = values[3];
  state.yaw_angle = values[4];
  state.yaw_rate = values[5];
  state.slip_angle = values[6];
}

StateVector add_scaled(const StateVector &lhs, const StateVector &rhs,
                       double scale) {
  StateVector out{};
  for (std::size_t i = 0; i < out.size(); ++i) {
    out[i] = lhs[i] + rhs[i] * scale;
  }
  return out;
}

StateVector rk4_step(const StateVector &state, const ControlVector &control,
                     const F110Params &params) {
  const double dt = params.timestep;
  const StateVector k1 = vehicle_dynamics_st(state, control, params);
  const StateVector k2 =
      vehicle_dynamics_st(add_scaled(state, k1, dt / 2.0), control, params);
  const StateVector k3 =
      vehicle_dynamics_st(add_scaled(state, k2, dt / 2.0), control, params);
  const StateVector k4 =
      vehicle_dynamics_st(add_scaled(state, k3, dt), control, params);

  StateVector out{};
  for (std::size_t i = 0; i < out.size(); ++i) {
    out[i] = state[i] + dt * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]) / 6.0;
  }
  return out;
}

StateVector euler_step(const StateVector &state, const ControlVector &control,
                       const F110Params &params) {
  return add_scaled(state, vehicle_dynamics_st(state, control, params),
                    params.timestep);
}

double apply_steering_delay(F110State &state, double raw_steer) {
  double delayed_steer = 0.0;
  if (state.steer_buffer_len < 1) {
    state.steer_buffer_0 = raw_steer;
    state.steer_buffer_len = 1;
  } else if (state.steer_buffer_len < 2) {
    state.steer_buffer_1 = state.steer_buffer_0;
    state.steer_buffer_0 = raw_steer;
    state.steer_buffer_len = 2;
  } else {
    delayed_steer = state.steer_buffer_1;
    state.steer_buffer_1 = state.steer_buffer_0;
    state.steer_buffer_0 = raw_steer;
  }
  return delayed_steer;
}

void wrap_yaw(F110State &state) {
  if (state.yaw_angle > kTwoPi) {
    state.yaw_angle -= kTwoPi;
  } else if (state.yaw_angle < 0.0) {
    state.yaw_angle += kTwoPi;
  }
}

} // namespace

F110StepResult step(const F110State &state, const F110Action &action,
                    const F110Params &params, Integrator integrator) {
  F110State next = state;
  const ControlVector pid_out =
      pid(action.velocity, action.steer, next.velocity, next.steer_angle,
          params.sv_max, params.a_max, params.v_max, params.v_min);
  const ControlVector control{pid_out[1], pid_out[0]};
  const StateVector state_vec = to_vector(next);
  const StateVector next_vec = integrator == Integrator::RK4
                                   ? rk4_step(state_vec, control, params)
                                   : euler_step(state_vec, control, params);
  from_vector(next, next_vec);
  wrap_yaw(next);

  return F110StepResult{next, params.timestep, next.in_collision ? 0.0 : 1.0,
                        next.in_collision};
}

} // namespace f110_rollout_kernel
