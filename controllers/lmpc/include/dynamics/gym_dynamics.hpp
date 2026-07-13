#ifndef LMPC__DYNAMICS__GYM_DYNAMICS_HPP_
#define LMPC__DYNAMICS__GYM_DYNAMICS_HPP_

#include "dynamics/common.hpp"

namespace lmpc
{
namespace dynamics
{

// Reparametrizes f110 Gym's own single-track dynamic model
// (gym/f110_gym/envs/dynamic_models.py::vehicle_dynamics_st, the |v| >= 0.5
// branch) from its native (v, beta) state into the paper's (vx, vy) state --
// same physics, same parameters (VehicleParams mirrors f110_env.params
// exactly), just a coordinate change, so this is the closest available
// nominal model to the actual simulator.
//
// Gym's own low-speed kinematic/dynamic switch is deliberately NOT
// replicated here -- that switch is a simulator-specific numerical patch,
// not something a smooth nominal model for a linearization-based QP should
// mirror. A small speed floor is used instead, purely to keep the 1/v terms
// in gym's own equations well-defined at vx = vy = 0 (every lap launches
// from rest).
class GymDynamics final : public DynamicsModel
{
public:
  explicit GymDynamics(VehicleParams params)
  : params(params)
  {
  }

  casadi::SX operator()(
    const casadi::SX & x_vel,
    const casadi::SX & u,
    const casadi::SX & /*u_prev*/) const override
  {
    using casadi::SX;

    const SX & vx = x_vel(VX);
    const SX & vy = x_vel(VY);
    const SX & omega = x_vel(OMEGA);
    const SX & a = u(A);
    const SX & delta = u(DELTA);

    const double mu = params.mu;
    const double C_Sf = params.C_Sf;
    const double C_Sr = params.C_Sr;
    const double lf = params.lf;
    const double lr = params.lr;
    const double h = params.h;
    const double m = params.m;
    const double I = params.I;

    // Reparametrize (vx, vy) -> (v, beta), gym's own state variables, so
    // the rest of this function is a direct transcription of
    // gym/f110_gym/envs/dynamic_models.py::vehicle_dynamics_st's dynamic
    // branch (x[3]=v, x[6]=beta, x[5]=omega, x[2]=delta, u[1]=a).
    const SX v = sqrt(vx * vx + vy * vy + kSpeedFloor * kSpeedFloor);
    const SX beta = atan2(vy, vx);

    // Load-transfer-adjusted front/rear normal-force terms (gym bundles
    // mass into the mu*m/... prefactors below rather than naming
    // Fz_f/Fz_r separately).
    const SX fzf_term = kGravity * lr - a * h;
    const SX fzr_term = kGravity * lf + a * h;

    const SX omega_dot =
      -mu * m / (v * I * (lr + lf)) *
      (lf * lf * C_Sf * fzf_term + lr * lr * C_Sr * fzr_term) * omega
      + mu * m / (I * (lr + lf)) *
      (lr * C_Sr * fzr_term - lf * C_Sf * fzf_term) * beta
      + mu * m / (I * (lr + lf)) * lf * C_Sf * fzf_term * delta;

    const SX beta_dot =
      mu / (v * v * (lr + lf)) *
      (C_Sr * fzr_term * lr - C_Sf * fzf_term * lf) * omega
      - mu / (v * (lr + lf)) * (C_Sr * fzr_term + C_Sf * fzf_term) * beta
      + mu / (v * (lr + lf)) * C_Sf * fzf_term * delta;

    // gym's f[3] = u[1] directly: v_dot = a, no tire coupling on speed
    // magnitude. Converting back to (vx_dot, vy_dot) via the product rule
    // on vx = v*cos(beta), vy = v*sin(beta).
    const SX vx_dot = a * cos(beta) - v * sin(beta) * beta_dot;
    const SX vy_dot = a * sin(beta) + v * cos(beta) * beta_dot;

    return casadi::SX::vertcat({vx_dot, vy_dot, omega_dot});
  }

  const char * name() const override {return "gym";}

private:
  // Purely a numerical-safety floor for the 1/v, 1/v^2 terms above -- NOT
  // an attempt to replicate gym's own |v| < 0.5 kinematic/dynamic switch
  // (see class comment above). Every lap launches from vx = vy = 0, so
  // this is hit unconditionally at the first evaluation, not an edge case.
  static constexpr double kSpeedFloor = 1e-2;
  static constexpr double kGravity = 9.81;

  VehicleParams params;
};

}  // namespace dynamics
}  // namespace lmpc

#endif  // LMPC__DYNAMICS__GYM_DYNAMICS_HPP_
