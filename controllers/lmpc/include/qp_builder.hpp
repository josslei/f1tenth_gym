#ifndef LMPC__QP_BUILDER_HPP_
#define LMPC__QP_BUILDER_HPP_

#include <casadi/casadi.hpp>
#include <string>
#include <vector>

namespace lmpc {

// Box constraints U = {u | u_l <= u <= u_u} and the ey half of
// X = {x | -W/2 <= ey <= W/2} (DESIGN.md SS3). The rest of X (vx, vy,
// omega, epsi, s unconstrained) is deliberately not bounded here -- SS3
// only pins the ey box, and adding more would be guarding against
// conditions the design doc doesn't call for.
struct QpBounds {
  double a_min;
  double a_max;
  double delta_min;
  double delta_max;
  double ey_max;
  // Max |delta_t - delta_{t-1}| per stage: the plant's steering-rate limit
  // integrated over one control period (LmpcConfig::sv_max * dt) -- keeps
  // every planned steering step physically executable by gym's
  // rate-limited actuator (LmpcConfig::sv_max's comment has the measured
  // failure this prevents).
  double ddelta_max;
};

// Scalar cost weights, applied uniformly to the scaled U decision
// variable (the paper's own c_u/c_d_u -- a plain L2 norm on the control
// vector, not a per-component-weighted Q-norm/R matrix; LmpcConfig::c_u's
// comment has the full rationale, including why a per-component weight
// isn't needed once U is already normalized to O(1)).
struct QpWeights {
  // Multiplier on the normalized terminal cost-to-go J^T lambda
  // (LmpcConfig::cost_to_go_weight's comment).
  double cost_to_go;
  double terminal_slack;
  double control;
  double control_rate;
  // Exact-plus-quadratic penalty on the per-stage ey slack (LmpcConfig::
  // ey_slack_l1's comment has the rationale for softening ey at all).
  double ey_slack_l1;
  double ey_slack_l2;
};

// Diagonal variable-scaling factors (LmpcConfig's comment on scale_x_vy
// etc. has the full rationale): the QP's decision variables are declared
// scaled (O(1)) and every constraint/cost is written against the physical
// expression scale*scaled_var, so qrqp's own KKT solve operates on a
// well-conditioned problem regardless of the underlying physical units'
// spread. StateIndex/ControlIndex order.
struct QpScaling {
  casadi::DM x; // kStateDim x 1
  casadi::DM u; // kControlDim x 1
  double j;     // seed-lap cost-to-go scale
};

// One horizon stage's affine dynamics parameters (DESIGN.md SS4):
// x_{t+1} = A x_t + B u_t + C.
struct QpStage {
  casadi::DM A; // kStateDim x kStateDim
  casadi::DM B; // kStateDim x kControlDim
  casadi::DM C; // kStateDim x 1
};

// Per-phase wall-clock cost of one QpBuilder::solve() call, in milliseconds
// (recom.md's t_set-params/t_solver/t_postcheck). Populated on BOTH the
// success and failure paths -- on failure, whatever ran after
// set_params_ms is attributed to solver_ms (postcheck never got a chance to
// run; see solve()'s own comment for the exact attribution rule), so these
// three always sum to the call's total wall time either way.
struct QpSolveTimings {
  double set_params_ms = 0.0;
  double solver_ms = 0.0;
  double postcheck_ms = 0.0;
};

struct QpSolution {
  casadi::DM x_traj;         // kStateDim x (N+1)
  casadi::DM u_traj;         // kControlDim x N
  casadi::DM lambda;         // safe_set_size x 1
  casadi::DM terminal_slack; // normalized kStateDim x 1 signed error
  bool success;
  std::string message; // populated with the solver's own error text on failure
  QpSolveTimings timings;
};

// Builds the multi-shooting QP graph (DESIGN.md SS4) ONCE for a fixed
// horizon length N and terminal safe-set size q. QpBuilder is reconstructed
// when a completed lap changes q; each control step only re-parameterizes the
// existing graph.
//
// solve() re-parametrizes (A_t, B_t, C_t, x_k, u_{k-1}, the safe-set
// matrices) and re-solves the SAME graph every control step -- this is the
// whole point of keeping A_t/B_t/C_t as Opti parameters rather than
// decision variables (DESIGN.md SS4): building the graph is done once, the
// per-step cost is just a conic solve.
class QpBuilder {
public:
  QpBuilder(casadi_int horizon_steps, casadi_int safe_set_size,
            const QpBounds &bounds, const QpWeights &weights,
            const QpScaling &scaling, const std::string &solver_name = "qrqp");

  // stages.size() must equal N. x_warm/u_warm seed the receding-horizon
  // trajectory; lambda_warm supplies a q-dimensional simplex point.
  QpSolution solve(const casadi::DM &x_k, const casadi::DM &u_prev,
                   const std::vector<QpStage> &stages, const casadi::DM &X_ss,
                   const casadi::DM &J_ss, const casadi::DM &x_warm,
                   const casadi::DM &u_warm, const casadi::DM &lambda_warm);

  casadi_int safe_set_size() const { return q; }

  // A failed solve may leave IPOPT multipliers tied to a stale plant state.
  // Keep the primal trajectory warm start, but discard those duals.
  void clear_dual_warm_start();

private:
  casadi_int N;
  casadi_int q;
  QpBounds bounds;
  std::string solver_name;

  QpScaling scaling;

  casadi::Opti opti;
  casadi::MX X;         // kStateDim x (N+1), SCALED (O(1)) decision variable
  casadi::MX U;         // kControlDim x N, SCALED (O(1)) decision variable
  casadi::MX EySlack;   // 1 x (N+1), per-stage ey corridor slack (scaled ey
                        // units, >= 0) -- keeps the QP feasible when the
                        // measured/predicted state is pushed outside the ey
                        // box (QpWeights::ey_slack_l1's comment)
  casadi::MX Lambda;    // q x 1 decision variable
  casadi::MX ETerminal; // normalized kStateDim x 1 signed terminal error
  casadi::MX X_phys;    // scaling.x * X -- physical-unit state, used in every
                        // constraint/cost and extracted at solve time
  casadi::MX U_phys;    // scaling.u * U -- physical-unit control, likewise

  casadi::MX x0_param;
  casadi::MX u_prev_param;
  std::vector<casadi::MX> A_params;
  std::vector<casadi::MX> B_params;
  std::vector<casadi::MX> C_params;
  casadi::MX Xss_param;
  casadi::MX Jss_param;

  // Dual (constraint multiplier) warm start (recom.md item 2) -- IPOPT-only:
  // scoped to solver_name == "ipopt" in solve()'s own body, since dual warm
  // starting is specifically an interior-point-method concept (paired with
  // "warm_start_init_point"=yes below) that qrqp/qpoases's active-set
  // methods don't share the same warm-start contract for. Reset implicitly
  // whenever a new QpBuilder is constructed (q/N changed, e.g. a completed
  // lap resized the safe set) since lam_g's own dimension depends on the
  // constraint count baked into THIS instance's graph.
  casadi::DM lam_g_warm;
  bool has_dual_warm_start = false;
};

} // namespace lmpc

#endif // LMPC__QP_BUILDER_HPP_
