#ifndef LMPC__DYNAMICS__EXTENDED_KINEMATIC_DYNAMICS_HPP_
#define LMPC__DYNAMICS__EXTENDED_KINEMATIC_DYNAMICS_HPP_

#include "dynamics/common.hpp"

namespace lmpc {
namespace dynamics {

// Kong et al.'s extended kinematic bicycle model (their ref [20], cited by
// Xue et al. for "the planar dynamic bicycle model"; neither the paper nor
// ref/Racing-LMPC-ROS2 implements this specific variant, so this is a
// from-scratch transcription, not a port):
//
//   v_dot_x = (1/m) F_{r,x}
//   v_dot_y = (lr / (lf + lr)) * (delta_dot * vx + delta * v_dot_x)
//   omega_dot = (1 / (lf + lr)) * (delta_dot * vx + delta * v_dot_x)
//
// F_{r,x}/m is exactly our acceleration control, so v_dot_x = a directly.
//
// delta_dot needs delta and its rate, but our pinned state/control
// convention (DESIGN.md SS1) keeps delta a control, not a state -- adding a
// 7th state just for this model would break the uniform state dimension
// every other model, the safe set, and the regression assume. Instead,
// delta_dot is computed as (delta_t - delta_{t-1}) / dt from CONSECUTIVE
// controls the FHOCP already carries at every stage (eq. 4b's u_{t-1},
// which is what makes the control-rate cost possible in the first place) --
// this needs dt, hence the constructor argument that
// GymDynamics doesn't need.
class ExtendedKinematicDynamics final : public DynamicsModel {
public:
  ExtendedKinematicDynamics(VehicleParams params, double dt)
      : params(params), dt(dt) {}

  casadi::SX operator()(const casadi::SX &x_vel, const casadi::SX &u,
                        const casadi::SX &u_prev) const override {
    using casadi::SX;

    const SX &vx = x_vel(VX);
    const SX &a = u(ControlIndex::A);
    const SX &delta = u(DELTA);
    const SX &delta_prev = u_prev(DELTA);

    const double wheelbase = params.lf + params.lr;

    const SX vx_dot = a;
    const SX delta_dot = (delta - delta_prev) / dt;

    // Shared term: d/dt[delta * vx], product rule.
    const SX rate_term = delta_dot * vx + delta * vx_dot;

    const SX vy_dot = (params.lr / wheelbase) * rate_term;
    const SX omega_dot = rate_term / wheelbase;

    return casadi::SX::vertcat({vx_dot, vy_dot, omega_dot});
  }

  const char *name() const override { return "extended_kinematic"; }

private:
  VehicleParams params;
  double dt;
};

} // namespace dynamics
} // namespace lmpc

#endif // LMPC__DYNAMICS__EXTENDED_KINEMATIC_DYNAMICS_HPP_
