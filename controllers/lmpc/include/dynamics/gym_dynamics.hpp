#ifndef LMPC__DYNAMICS__GYM_DYNAMICS_HPP_
#define LMPC__DYNAMICS__GYM_DYNAMICS_HPP_

#include "dynamics/common.hpp"

namespace lmpc {
namespace dynamics {

// Reparametrizes f110 Gym's own single-track model
// (gym/f110_gym/envs/dynamic_models.py::vehicle_dynamics_st) from its
// native (v, beta) state into the paper's (vx, vy) state -- same physics,
// just a coordinate change. Its control is the LMPC's direct
// [longitudinal acceleration, steering angle] input.
//
// BOTH of gym's regimes are modeled, blended smoothly instead of gym's own
// hard |v| < 0.5 switch. An earlier revision modeled only the dynamic
// (tire-force) branch, on the theory that the switch was a
// simulator-specific numerical patch -- measured directly (2026-07-13)
// that this is wrong in a way that breaks the whole controller at launch:
// the dynamic branch's lateral terms scale as 1/v..1/v^2, so at
// vx ~= 0.1 m/s their eigenvalues are O(10^3) and the Euler-discretized
// warm-start rollout at dt = 0.025 is violently unstable under ANY nonzero
// steering (one recorded rollout: omega jumped 0 -> 19 -> -190 rad/s in
// two stages, epsi wound up at -8.4 rad), which fed physically nonsensical
// linearization references to the QP and made qrqp fail with "Failed to
// calculate search direction" on the second closed-loop solve. Gym's
// kinematic low-speed branch exists precisely because the tire model is
// invalid (not merely stiff) at low slip velocities; the nominal model has
// to respect the same regime boundary as the plant it predicts. The blend
// is a C^1 smoothstep over [kBlendStart, kBlendEnd] rather than gym's step
// at 0.5 so the Jacobians the linearizer extracts stay continuous.
class GymDynamics final : public DynamicsModel {
public:
  explicit GymDynamics(VehicleParams params) : params(params) {}

  casadi::SX operator()(const casadi::SX &x_vel, const casadi::SX &u,
                        const casadi::SX & /*u_prev*/) const override {
    using casadi::SX;

    const SX &vx = x_vel(VX);
    const SX &vy = x_vel(VY);
    const SX &omega = x_vel(OMEGA);
    const SX &a = u(ControlIndex::A);
    const SX &delta = u(DELTA);

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
    const SX vx_safe =
        if_else(fabs(vx) < kSpeedFloor,
                if_else(vx >= 0.0, kSpeedFloor, -kSpeedFloor), vx);
    const SX beta = atan2(vy, vx_safe);

    // Load-transfer-adjusted front/rear normal-force terms (gym bundles
    // mass into the mu*m/... prefactors below rather than naming
    // Fz_f/Fz_r separately).
    const SX fzf_term = kGravity * lr - a * h;
    const SX fzr_term = kGravity * lf + a * h;

    // Dynamic (tire-force) branch: gym's |v| >= 0.5 equations.
    const SX omega_dot_dyn =
        -mu * m / (v * I * (lr + lf)) *
            (lf * lf * C_Sf * fzf_term + lr * lr * C_Sr * fzr_term) * omega +
        mu * m / (I * (lr + lf)) *
            (lr * C_Sr * fzr_term - lf * C_Sf * fzf_term) * beta +
        mu * m / (I * (lr + lf)) * lf * C_Sf * fzf_term * delta;

    const SX beta_dot =
        mu / (v * v * (lr + lf)) *
            (C_Sr * fzr_term * lr - C_Sf * fzf_term * lf) * omega -
        mu / (v * (lr + lf)) * (C_Sr * fzr_term + C_Sf * fzf_term) * beta +
        mu / (v * (lr + lf)) * C_Sf * fzf_term * delta;

    // gym's f[3] = u[1] directly: v_dot = a, no tire coupling on speed
    // magnitude. Converting back to (vx_dot, vy_dot) via the product rule
    // on vx = v*cos(beta), vy = v*sin(beta).
    const SX vx_dot_dyn = a * cos(beta) - v * sin(beta) * beta_dot;
    const SX vy_dot_dyn = a * sin(beta) + v * cos(beta) * beta_dot;

    // Kinematic branch: gym's |v| < 0.5 equations, same reparametrization.
    // Gym holds beta constant there (its f[6] = 0), so the product rule
    // reduces to pure a-along-beta; its omega_dot is a/lwb*tan(delta)
    // plus a steering-rate term that has no analogue here (delta is a
    // direct input, not a state with its own rate).
    const double lwb = lf + lr;
    const SX vx_dot_kin = a * cos(beta);
    const SX vy_dot_kin = a * sin(beta);
    const SX omega_dot_kin = a / lwb * tan(delta);

    // C^1 smoothstep from pure-kinematic (v <= kBlendStart, gym's own
    // switch speed -- the tire branch's 1/v terms must carry weight
    // EXACTLY zero there, not merely a small factor, since they are
    // O(10^2..10^4) in that regime) to pure-dynamic (v >= kBlendEnd,
    // where the tire branch's eigenvalues are small enough for the
    // dt=0.025 Euler rollout to integrate stably).
    const SX ramp = fmin(
        fmax((v - kBlendStart) / (kBlendEnd - kBlendStart), SX(0.0)), SX(1.0));
    const SX w = ramp * ramp * (3.0 - 2.0 * ramp);

    const SX vx_dot = (1.0 - w) * vx_dot_kin + w * vx_dot_dyn;
    const SX vy_dot = (1.0 - w) * vy_dot_kin + w * vy_dot_dyn;
    const SX omega_dot = (1.0 - w) * omega_dot_kin + w * omega_dot_dyn;

    return casadi::SX::vertcat({vx_dot, vy_dot, omega_dot});
  }

  const char *name() const override { return "gym"; }

private:
  // Purely a numerical-safety floor for the 1/v, 1/v^2 terms above (kept
  // well-defined at vx = vy = 0 even though the blend zeroes their weight
  // there -- 0 * NaN is still NaN). The REGIME handling is the
  // kinematic/dynamic blend below, not this floor.
  static constexpr double kSpeedFloor = 1e-2;
  static constexpr double kGravity = 9.81;

  // Blend window for the kinematic->dynamic transition. kBlendStart is
  // gym's own switch speed (dynamic_models.py, |v| < 0.5); kBlendEnd is
  // where the tire branch's fastest lateral eigenvalue (~O(50) 1/s at
  // 1 m/s for this vehicle's params) is comfortably inside the dt=0.025
  // Euler stability region the warm-start rollout integrates at.
  static constexpr double kBlendStart = 0.5;
  static constexpr double kBlendEnd = 1.0;

  VehicleParams params;
};

} // namespace dynamics
} // namespace lmpc

#endif // LMPC__DYNAMICS__GYM_DYNAMICS_HPP_
