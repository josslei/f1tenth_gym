#ifndef LMPC__LMPC_CONTROLLER_HPP_
#define LMPC__LMPC_CONTROLLER_HPP_

#include <casadi/casadi.hpp>

#include "dynamics/common.hpp"
#include "dynamics/gym_dynamics.hpp"
#include "integrators/euler.hpp"
#include "linearization.hpp"
#include "lmpc_config.hpp"
#include "qp_builder.hpp"
#include "safe_set.hpp"
#include "track.hpp"

namespace lmpc {

// State/control index conventions live in dynamics/common.hpp (the dynamics
// functors need them too); re-exported here so lmpc::StateIndex etc. keep
// working unqualified.
using dynamics::ControlIndex;
using dynamics::kControlDim;
using dynamics::kStateDim;
using dynamics::StateIndex;

// Mirrors controllers/controller_base.py's Controller shape
// (reset/update/control), but speaks native casadi::DM end to end -- no
// array/DM conversion happens inside this class. The pybind11 boundary
// (src/bindings.cpp) is where numpy <-> DM conversion happens, once, at the
// language boundary.
//
// Implements DESIGN.md SS8's "first implementation pass": the FHOCP is
// solved every control step with A_t = A^f_t, B_t = B^f_t, C_t = C^f_t --
// the nominal model's own discretized Jacobian, no learned error
// correction (SS5/SS6's regression is not implemented yet). This tests
// whether the base MPC mechanism (the QP, the receding horizon, the
// terminal cost-to-go pulling the car forward) works at all before the
// learning layer is added on top.
//
// Nominal model is hardcoded to GymDynamics + Euler for this first pass
// (both interchangeable per DESIGN.md SS8, but a single fixed pair keeps
// this class's surface simple until there's a reason to expose the
// choice).
class LMPCController {
public:
  explicit LMPCController(const LmpcConfig &config);

  void reset();

  // x is the native kStateDim x 1 state vector (StateIndex order).
  // t is the current simulation time in seconds.
  void update(const casadi::DM &x, double t);

  // Runs one FHOCP solve (DESIGN.md SS8 steps 2-6) and returns the
  // kControlDim x 1 control vector (ControlIndex order): [a, delta].
  casadi::DM control();

  // DESIGN.md SS4's "no separate integration step": X[:, 1] of the just-
  // solved trajectory, i.e. the model-consistent one-step-ahead state --
  // this IS x_warm's column 0 after control()'s shift_warm_start() runs
  // (DESIGN.md SS8 step 6's velocity-command source, e.g.
  // predicted_next_state()(StateIndex::VX) as env.step's velocity command,
  // vs. hand-rolling a separate integration). Only meaningful after
  // control() has been called at least once; all-zero before that (reset()
  // state).
  casadi::DM predicted_next_state() const { return x_warm(casadi::Slice(), 0); }

  // The full predicted state trajectory (kStateDim x (N+1), StateIndex
  // order per column) behind predicted_next_state() above -- e.g. for
  // drawing the receding horizon. Same "only meaningful after control()"
  // caveat.
  casadi::DM predicted_trajectory() const { return x_warm; }

private:
  LmpcConfig config;

  dynamics::GymDynamics dynamics_model;
  integrators::Euler integrator;
  Linearizer linearizer;
  Track track;
  SafeSet safe_set;
  QpBuilder qp_builder;

  casadi::DM x;
  double t;
  bool has_state;

  casadi::DM u_prev; // u_{k-1}, ControlIndex order

  // The linearization sequence z_bar_{k:k+N} (DESIGN.md SS8 step 2): the
  // previous solve's own trajectory, shifted one step for the receding
  // horizon, or a naive rollout before the first solve.
  casadi::DM x_warm; // kStateDim x (N+1)
  casadi::DM u_warm; // kControlDim x N
  // Lambda's own warm start (QpBuilder::solve()'s header comment has the
  // rationale: unset, Lambda starts at CasADi's default of all zeros,
  // which isn't even feasible w.r.t. sum(lambda)==1). Not part of z_bar --
  // Lambda has no physical/pose meaning -- but carried the same way:
  // reused across control() calls, seeded uniformly (1/q, the simplex
  // centroid: feasible, no a-priori reason to favor one neighbor) before
  // the first solve.
  casadi::DM lambda_warm; // safe_set_size x 1
  bool has_warm_start;

  // Naive forward rollout under the nominal model, holding u constant at
  // u_prev -- seeds x_warm/u_warm/lambda_warm before the very first solve
  // (DESIGN.md SS8 step 2).
  void seed_warm_start();

  // Shifts the just-solved trajectory by one stage to become the next
  // control step's linearization sequence (receding horizon).
  void shift_warm_start(const QpSolution &solution);
};

} // namespace lmpc

#endif // LMPC__LMPC_CONTROLLER_HPP_
