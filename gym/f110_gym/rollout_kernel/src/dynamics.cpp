#include "dynamics.hpp"

#include <cmath>

namespace f110_rollout_kernel {
namespace {

constexpr double kGravity = 9.81;

} // namespace

double accl_constraints(double vel, double accl, double v_switch, double a_max,
                        double v_min, double v_max) {
  const double pos_limit = vel > v_switch ? a_max * v_switch / vel : a_max;

  if ((vel <= v_min && accl <= 0.0) || (vel >= v_max && accl >= 0.0)) {
    return 0.0;
  }
  if (accl <= -a_max) {
    return -a_max;
  }
  if (accl >= pos_limit) {
    return pos_limit;
  }
  return accl;
}

double steering_constraint(double steering_angle, double steering_velocity,
                           double s_min, double s_max, double sv_min,
                           double sv_max) {
  if ((steering_angle <= s_min && steering_velocity <= 0.0) ||
      (steering_angle >= s_max && steering_velocity >= 0.0)) {
    return 0.0;
  }
  if (steering_velocity <= sv_min) {
    return sv_min;
  }
  if (steering_velocity >= sv_max) {
    return sv_max;
  }
  return steering_velocity;
}

ControlVector pid(double speed, double steer, double current_speed,
                  double current_steer, double max_sv, double max_a,
                  double max_v, double min_v) {
  double sv = 0.0;
  const double steer_diff = steer - current_steer;
  if (std::fabs(steer_diff) > 1e-4) {
    sv = steer_diff / std::fabs(steer_diff) * max_sv;
  }

  const double vel_diff = speed - current_speed;
  double kp = 0.0;
  if (current_speed > 0.0) {
    kp = vel_diff > 0.0 ? 10.0 * max_a / max_v : 10.0 * max_a / (-min_v);
  } else {
    kp = vel_diff > 0.0 ? 2.0 * max_a / max_v : 2.0 * max_a / (-min_v);
  }

  return ControlVector{kp * vel_diff, sv};
}

StateVector vehicle_dynamics_ks(const StateVector &x,
                                const ControlVector &u_init,
                                const F110Params &params) {
  const ControlVector u{
      steering_constraint(x[2], u_init[0], params.s_min, params.s_max,
                          params.sv_min, params.sv_max),
      accl_constraints(x[3], u_init[1], params.v_switch, params.a_max,
                       params.v_min, params.v_max),
  };
  const double wheelbase = params.lf + params.lr;

  return StateVector{x[3] * std::cos(x[4]),
                     x[3] * std::sin(x[4]),
                     u[0],
                     u[1],
                     x[3] / wheelbase * std::tan(x[2]),
                     0.0,
                     0.0};
}

StateVector vehicle_dynamics_st(const StateVector &x,
                                const ControlVector &u_init,
                                const F110Params &params) {
  const ControlVector u{
      steering_constraint(x[2], u_init[0], params.s_min, params.s_max,
                          params.sv_min, params.sv_max),
      accl_constraints(x[3], u_init[1], params.v_switch, params.a_max,
                       params.v_min, params.v_max),
  };

  if (std::fabs(x[3]) < 0.5) {
    StateVector f = vehicle_dynamics_ks(x, u, params);
    const double wheelbase = params.lf + params.lr;
    f[5] = u[1] / wheelbase * std::tan(x[2]) +
           x[3] / (wheelbase * std::pow(std::cos(x[2]), 2.0)) * u[0];
    f[6] = 0.0;
    return f;
  }

  const double lf = params.lf;
  const double lr = params.lr;
  const double c_sf = params.c_sf;
  const double c_sr = params.c_sr;
  const double mu = params.mu;
  const double mass = params.m;
  const double inertia = params.inertia;
  const double wheelbase = lr + lf;
  const double accel = u[1];
  const double gravity_lr = kGravity * lr - accel * params.h;
  const double gravity_lf = kGravity * lf + accel * params.h;

  return StateVector{
      x[3] * std::cos(x[6] + x[4]),
      x[3] * std::sin(x[6] + x[4]),
      u[0],
      u[1],
      x[5],
      -mu * mass / (x[3] * inertia * wheelbase) *
              (lf * lf * c_sf * gravity_lr + lr * lr * c_sr * gravity_lf) *
              x[5] +
          mu * mass / (inertia * wheelbase) *
              (lr * c_sr * gravity_lf - lf * c_sf * gravity_lr) * x[6] +
          mu * mass / (inertia * wheelbase) * lf * c_sf * gravity_lr * x[2],
      (mu / (x[3] * x[3] * wheelbase) *
           (c_sr * gravity_lf * lr - c_sf * gravity_lr * lf) -
       1.0) * x[5] -
          mu / (x[3] * wheelbase) * (c_sr * gravity_lf + c_sf * gravity_lr) *
              x[6] +
          mu / (x[3] * wheelbase) * c_sf * gravity_lr * x[2],
  };
}

} // namespace f110_rollout_kernel
