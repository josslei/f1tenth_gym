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
};

// Cost weights from DESIGN.md SS3's Phi(w): c_u on ||u_t||^2, c_du on
// ||u_t - u_{t-1}||^2. Not pinned by the paper or upstream (DESIGN.md's
// open items) -- exposed here so they're a config knob, not a magic
// number, until a tuned value is settled on. terminal_slack penalizes the
// normalized mismatch between the terminal state and safe-set convex hull.
struct QpWeights {
  double c_u;
  double c_du;
  double terminal_slack;
  casadi::DM terminal_slack_state;
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

struct QpSolution {
  casadi::DM x_traj;         // kStateDim x (N+1)
  casadi::DM u_traj;         // kControlDim x N
  casadi::DM lambda;         // safe_set_size x 1
  casadi::DM terminal_slack; // normalized kStateDim x 1
  bool success;
  std::string message; // populated with the solver's own error text on failure
};

// Builds the multi-shooting QP graph (DESIGN.md SS4) ONCE for a fixed
// horizon length N and terminal simplex size. SafeSet searches K candidates
// per lap but reduces them to a two-point local segment before this graph is
// parameterized, so the QP never carries K redundant lambda variables.
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
  // trajectory; lambda_warm supplies the initial scalar segment coordinate.
  QpSolution solve(const casadi::DM &x_k, const casadi::DM &u_prev,
                   const std::vector<QpStage> &stages, const casadi::DM &X_ss,
                   const casadi::DM &J_ss, const casadi::DM &x_warm,
                   const casadi::DM &u_warm, const casadi::DM &lambda_warm);

private:
  casadi_int N;
  casadi_int q;
  QpBounds bounds;
  std::string solver_name;

  QpScaling scaling;

  casadi::Opti opti;
  casadi::MX X;      // kStateDim x (N+1), SCALED (O(1)) decision variable
  casadi::MX U;      // kControlDim x N, SCALED (O(1)) decision variable
  casadi::MX Alpha;  // scalar barycentric coordinate for the terminal segment
  casadi::MX Lambda; // [1-Alpha, Alpha]
  casadi::MX TerminalSlack; // normalized terminal mismatch expression
  casadi::MX X_phys; // scaling.x * X -- physical-unit state, used in every
                     // constraint/cost and extracted at solve time
  casadi::MX U_phys; // scaling.u * U -- physical-unit control, likewise

  casadi::MX x0_param;
  casadi::MX u_prev_param;
  std::vector<casadi::MX> A_params;
  std::vector<casadi::MX> B_params;
  std::vector<casadi::MX> C_params;
  casadi::MX Xss_param;
  casadi::MX Jss_param;
};

} // namespace lmpc

#endif // LMPC__QP_BUILDER_HPP_
