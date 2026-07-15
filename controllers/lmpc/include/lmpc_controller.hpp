#ifndef LMPC__LMPC_CONTROLLER_HPP_
#define LMPC__LMPC_CONTROLLER_HPP_

#include <casadi/casadi.hpp>
#include <memory>

#include "dynamics/common.hpp"
#include "dynamics/gym_dynamics.hpp"
#include "integrators/rk4.hpp"
#include "linearization.hpp"
#include "lmpc_config.hpp"
#include "qp_builder.hpp"
#include "safe_set.hpp"
#include "track.hpp"

namespace lmpc {

struct LMPCControllerTestAccess;

// Per-phase wall-clock cost of the last solve_once() call, in milliseconds
// (recom.md's requested profiling breakdown). rollout_lin_ms covers
// solve_once()'s single combined rollout+linearize loop (recom.md item 1:
// this used to be TWO passes -- a rollout-only pass calling Linearizer::
// step(), then a second pass re-linearizing the same states -- merged into
// one so each stage's Jacobians are computed exactly once). knn_ms is the
// terminal safe_set.query_local_segments() call.
// set_params_ms/solver_ms/postcheck_ms are copied through from the QpBuilder
// call's own QpSolveTimings (qp_builder.hpp) unchanged. Populated on every
// solve_once() call regardless of whether the QP itself succeeded --
// control() only throws AFTER solve_once() already recorded these, so a
// failed step's timings are still visible to callers (e.g. for perf
// reporting around a fallback-braking step, not just successful ones).
struct ControllerTimings {
  double rollout_lin_ms = 0.0;
  double knn_ms = 0.0;
  double set_params_ms = 0.0;
  double solver_ms = 0.0;
  double postcheck_ms = 0.0;
};

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
// Nominal model is hardcoded to GymDynamics + RK4 for this first pass (both
// interchangeable per DESIGN.md SS8, but a single fixed pair keeps this
// class's surface simple until there's a reason to expose the choice). RK4,
// not Euler: F110Env defaults to Integrator.RK4 (gym/f110_gym/envs/
// f110_env.py) and runs/lmpc_drive.py never overrides it, so a nominal
// model discretized with Euler was integrating a different one-step map
// than the actual plant at the same dt -- most visible in corners, where
// (beta, omega) move fastest relative to dt=0.025.
class LMPCController {
public:
  explicit LMPCController(const LmpcConfig &config);

  void reset();

  // x is the native kStateDim x 1 state vector (StateIndex order).
  // t is the current simulation time in seconds. actual_delta is the
  // PLANT's own current steering angle (e.g. gym's raw sim state[2]) --
  // NOT the last angle this controller commanded. Used only to anchor the
  // stage-0 steering-rate constraint/cost in solve_once() (QpBounds::
  // ddelta_max's comment): gym applies steering through a 2-step delay
  // buffer and a rate-limited PID, so the commanded delta and the plant's
  // actual delta diverge, and constraining stage 0's rate against the
  // COMMAND (what an earlier revision did, via u_prev alone) anchors the
  // plan to a steering angle the tires were never actually at.
  void update(const casadi::DM &x, double t, double actual_delta);

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

  // Per-phase timing of the last solve_once() call (struct comment above
  // has the full field-by-field breakdown). Only meaningful after control()
  // has been called at least once (default-constructed/all-zero otherwise).
  const ControllerTimings &last_timings() const { return timings; }

  // Normalized signed terminal error from the last accepted solve, in
  // StateIndex order. Reset to zero with the controller.
  const casadi::DM &last_terminal_slack_value() const {
    return last_terminal_slack;
  }

private:
  friend struct LMPCControllerTestAccess;

  LmpcConfig config;

  dynamics::GymDynamics dynamics_model;
  integrators::Rk4 integrator;
  Linearizer linearizer;
  Track track;
  SafeSet safe_set;
  // Captured from D^0 and retained even after older-lap eviction.
  double cost_to_go_scale;
  std::unique_ptr<QpBuilder> qp_builder;

  casadi::DM x;
  double t;
  bool has_state;

  casadi::DM u_prev; // u_{k-1}, ControlIndex order -- this controller's own
                     // last COMMAND, not necessarily what the plant reached.

  // The plant's actual current steering angle, set by update() -- see its
  // header comment. Used in solve_once() to override just the DELTA
  // component of the u_prev handed to QpBuilder, so the stage-0
  // steering-rate anchor reflects reality instead of the last command.
  double actual_delta;

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
  int consecutive_solve_failures;

  // The last solve's own predicted state trajectory (kStateDim x (N+1)) --
  // backs predicted_next_state()/predicted_trajectory() above.
  casadi::DM x_pred;
  casadi::DM last_terminal_slack;

  // Backs last_timings() above; updated at the end of every solve_once().
  ControllerTimings timings;

  // The normalized-distance scale for LOCATING safe-set data (query /
  // trajectory_segment) -- deliberately different from the QP's
  // variable-conditioning scale; see the definition's comment for the
  // measured failure that conflating the two caused.
  casadi::DM safe_set_query_scale() const;

  casadi_int terminal_set_size() const;
  void rebuild_qp_builder();
  void reset_lambda_warm_start();

  // Seeds u_warm from the D^0 trajectory segment nearest the current state
  // (SafeSet::trajectory_segment) before the very first solve -- NOT a
  // zero-control naive rollout: from rest, holding u = u_prev = 0 parks the
  // whole horizon at the start line, which locks the terminal safe-set
  // query onto D^0's own launch samples and leaves the FHOCP with no
  // forward pull at all (measured directly, 2026-07-13: every cost term
  // ~0 at that operating point, the car never moved, and qrqp eventually
  // lost its search direction on the resulting near-flat KKT system).
  void seed_warm_start_from_safe_set();

  // Shifts the just-solved CONTROL trajectory by one stage for the next
  // control step (receding horizon). Only u_warm -- x_warm is re-derived
  // from the next measured state by solve_once()'s own rollout+linearize
  // loop.
  void shift_warm_start(const QpSolution &solution);
  void record_solve_failure();
  void record_solve_success();

  // One full FHOCP pass (rollout+linearize -> terminal query -> QP solve)
  // against the current x/u_warm -- factored out of control() for
  // testability. Rebuilds x_warm as a nominal-model rollout from the
  // MEASURED current state under u_warm on every call, not just the first:
  // shifting the previous solution's predicted states and only patching
  // column 0 with the measurement (an earlier scheme) leaves columns 1..N
  // as stale predictions, and whenever realized dynamics diverge from what
  // was predicted (gym's actuator layer guarantees some divergence), the
  // linearization sequence drifts from reality with nothing pulling it
  // back -- re-rolling out from the measurement every time keeps the whole
  // sequence dynamically consistent with where the car actually is. The
  // rollout and the per-stage linearization are done in ONE pass (recom.md):
  // Linearizer::operator() already computes x_next alongside (A_t, B_t, C_t)
  // at the same evaluation, so stage stg+1's x_ref is exactly stage stg's
  // x_next -- no separate rollout-only pass re-deriving the same states
  // before a second pass re-linearizes them.
  QpSolution solve_once();
};

} // namespace lmpc

#endif // LMPC__LMPC_CONTROLLER_HPP_
