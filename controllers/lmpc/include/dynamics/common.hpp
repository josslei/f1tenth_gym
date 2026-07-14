#ifndef LMPC__DYNAMICS__COMMON_HPP_
#define LMPC__DYNAMICS__COMMON_HPP_

#include <casadi/casadi.hpp>

namespace lmpc {
namespace dynamics {

// State/control index conventions -- pinned in controllers/lmpc/DESIGN.md SS1.
// x = [vx, vy, omega, epsi, s, ey], u = [a, delta]. Lives here (rather than
// on LMPCController) because the dynamics functors need it to slice/assemble
// state vectors; lmpc_controller.hpp re-exports these into the lmpc namespace.
enum StateIndex : casadi_int {
  VX = 0,
  VY = 1,
  OMEGA = 2,
  EPSI = 3,
  S = 4,
  EY = 5,
};

enum ControlIndex : casadi_int {
  A = 0,
  DELTA = 1,
};

constexpr casadi_int kStateDim = 6;
constexpr casadi_int kControlDim = 2;

// Mirrors f110_env.params's keys (gym/f110_gym/envs/f110_env.py) so values
// can be read straight from the simulator's own config with no
// re-derivation. Not every model uses every field -- e.g.
// ExtendedKinematicDynamics only needs m, lf, lr; GymDynamics needs all of
// them.
// Default member initializers mirror gym/f110_gym/envs/f110_env.py's
// DEFAULT_PARAMS exactly, so VehicleParams{} is a ready-to-use default
// rather than requiring every caller to look those numbers up.
struct VehicleParams {
  double mu = 1.0489;
  double C_Sf = 4.718;
  double C_Sr = 5.4562;
  double lf = 0.15875;
  double lr = 0.17145;
  double h = 0.074;
  double m = 3.74;
  double I = 0.04712;
};

// Produces the body-frame velocity derivatives [vx_dot, vy_dot, omega_dot]
// for one nominal dynamics model. The Frenet pose kinematics (s, ey, epsi
// derivatives) are generic across every model -- see
// frenet_pose_kinematics() below -- so they are deliberately NOT part of
// this interface. Every concrete model is a functor: state is fixed at
// construction (VehicleParams, and whatever else that specific model
// needs), operator() is a pure function of its arguments.
class DynamicsModel {
public:
  virtual ~DynamicsModel() = default;

  // x_vel = [vx, vy, omega] (StateIndex::VX/VY/OMEGA slice of the full
  // state, controllers/lmpc/DESIGN.md SS1).
  // u = [a, delta] at the current stage; u_prev has the same ordering at the
  // previous stage. It is carried because the FHOCP already threads u_{t-1}
  // through every stage for the control-rate cost.
  // No curvature parameter: that only matters for the Frenet pose
  // kinematics (frenet_pose_kinematics() below), which is generic and
  // already separate from this interface -- add it here only if a future
  // model's body-frame dynamics genuinely need it.
  // Returns [vx_dot, vy_dot, omega_dot].
  virtual casadi::SX operator()(const casadi::SX &x_vel, const casadi::SX &u,
                                const casadi::SX &u_prev) const = 0;

  virtual const char *name() const = 0;
};

// Frenet-frame pose kinematics (DESIGN.md SS1 state order: epsi, s, ey),
// identical for every dynamics model -- this is generic Frenet-frame math,
// not specific to any particular tire/force model.
// x_vel = [vx, vy, omega]. Returns [epsi_dot, s_dot, ey_dot].
inline casadi::SX frenet_pose_kinematics(const casadi::SX &x_vel,
                                         const casadi::SX &epsi,
                                         const casadi::SX &ey,
                                         const casadi::SX &kappa) {
  const casadi::SX &vx = x_vel(VX);
  const casadi::SX &vy = x_vel(VY);
  const casadi::SX &omega = x_vel(OMEGA);

  // Standard Frenet-frame kinematics (e.g. ref/Racing-LMPC-ROS2's
  // single_track_planar_model.cpp uses the same relations): s_dot picks up
  // the (1 - ey*kappa) arc-length distortion factor from tracking a path
  // that curves through a nonzero lateral offset; epsi_dot is yaw rate
  // relative to the path's own turning rate.
  const casadi::SX s_dot =
      (vx * cos(epsi) - vy * sin(epsi)) / (1.0 - ey * kappa);
  const casadi::SX ey_dot = vx * sin(epsi) + vy * cos(epsi);
  const casadi::SX epsi_dot = omega - kappa * s_dot;

  return casadi::SX::vertcat({epsi_dot, s_dot, ey_dot});
}

// Assembles the full 6-state x_dot = [vx_dot, vy_dot, omega_dot, epsi_dot,
// s_dot, ey_dot] (DESIGN.md SS1 order) from a DynamicsModel's velocity
// derivatives plus the shared Frenet pose kinematics.
inline casadi::SX full_state_dynamics(const DynamicsModel &model,
                                      const casadi::SX &x, const casadi::SX &u,
                                      const casadi::SX &u_prev,
                                      const casadi::SX &kappa) {
  const casadi::SX x_vel = x(casadi::Slice(VX, EPSI));
  const casadi::SX &epsi = x(EPSI);
  const casadi::SX &ey = x(EY);

  const casadi::SX vel_dot = model(x_vel, u, u_prev);
  const casadi::SX pose_dot = frenet_pose_kinematics(x_vel, epsi, ey, kappa);

  return casadi::SX::vertcat({vel_dot, pose_dot});
}

} // namespace dynamics
} // namespace lmpc

#endif // LMPC__DYNAMICS__COMMON_HPP_
