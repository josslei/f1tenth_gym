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

  // DESIGN.md SS8 step 8: append the just-driven closed-loop lap to the
  // safe set (lap-as-iteration -- the caller detects the crossing, records
  // the lap, resets the sim and this controller, and relaunches iteration
  // j+1 against the grown D^j). x_lap is kStateDim x (T+1), u_lap is
  // kControlDim x T (the REALIZED controls, same convention as the seed
  // collector's CSV), J_lap is (T+1) x 1 with J_k = T - k. Only complete,
  // collision-free laps belong here -- the safe set's guarantee rests on
  // every stored trajectory actually reaching the finish.
  void add_lap(const casadi::DM &x_lap, const casadi::DM &u_lap,
               const casadi::DM &J_lap);

  // X[:, 1] of the just-solved trajectory, i.e. the model-consistent
  // one-step-ahead state. Only meaningful after control() has been called
  // once. (Stored separately from x_warm: x_warm is re-derived from the
  // measured state every step and no longer carries the solved
  // prediction.)
  casadi::DM predicted_next_state() const { return x_pred(casadi::Slice(), 1); }

  // The full just-solved state trajectory (kStateDim x (N+1), StateIndex
  // order per column) behind predicted_next_state() above -- e.g. for
  // drawing the receding horizon. Same "only meaningful after control()"
  // caveat.
  casadi::DM predicted_trajectory() const { return x_pred; }

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

  // The last solve's own predicted state trajectory (kStateDim x (N+1)) --
  // backs predicted_next_state()/predicted_trajectory() above.
  casadi::DM x_pred;

  // The normalized-distance scale for LOCATING safe-set data (query /
  // trajectory_segment) -- deliberately different from the QP's
  // variable-conditioning scale; see the definition's comment for the
  // measured failure that conflating the two caused.
  casadi::DM safe_set_query_scale() const;

  // Seeds u_warm from the D^0 trajectory segment nearest the current state
  // (SafeSet::trajectory_segment) before the very first solve -- NOT a
  // zero-control naive rollout: from rest, holding u = u_prev = 0 parks the
  // whole horizon at the start line, which locks the terminal safe-set
  // query onto D^0's own launch samples and leaves the FHOCP with no
  // forward pull at all (measured directly, 2026-07-13: every cost term
  // ~0 at that operating point, the car never moved, and qrqp eventually
  // lost its search direction on the resulting near-flat KKT system).
  void seed_warm_start_from_safe_set();

  // Rebuilds x_warm as a nominal-model rollout from the MEASURED current
  // state under u_warm -- called every control(), not just the first.
  // Shifting the previous solution's predicted states and only patching
  // column 0 with the measurement (the previous scheme) leaves columns
  // 1..N as stale predictions; whenever realized dynamics diverge from
  // what was predicted (gym's actuator layer guarantees some divergence),
  // the linearization sequence drifts from reality with nothing pulling
  // it back, and the accumulated inconsistency is what ultimately broke
  // the solver. Re-rolling out from the measurement keeps the whole
  // sequence dynamically consistent with where the car actually is.
  void rollout_warm_states_from_current();

  // Shifts the just-solved CONTROL trajectory by one stage for the next
  // control step (receding horizon). Only u_warm -- x_warm is re-derived
  // from the next measured state by rollout_warm_states_from_current().
  void shift_warm_start(const QpSolution &solution);

  // One full FHOCP pass (rollout -> linearize -> terminal query -> QP
  // solve) against the current x/u_warm -- factored out so control() can
  // retry it once with a freshly reseeded warm start after a failure.
  QpSolution solve_once();
};

} // namespace lmpc

#endif // LMPC__LMPC_CONTROLLER_HPP_
