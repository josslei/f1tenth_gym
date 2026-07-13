#ifndef LMPC__QP_BUILDER_HPP_
#define LMPC__QP_BUILDER_HPP_

#include <casadi/casadi.hpp>
#include <string>
#include <vector>

namespace lmpc
{

// Box constraints U = {u | u_l <= u <= u_u} and the ey half of
// X = {x | -W/2 <= ey <= W/2} (DESIGN.md SS3). The rest of X (vx, vy,
// omega, epsi, s unconstrained) is deliberately not bounded here -- SS3
// only pins the ey box, and adding more would be guarding against
// conditions the design doc doesn't call for.
struct QpBounds
{
  double a_min;
  double a_max;
  double delta_min;
  double delta_max;
  double ey_max;
};

// Cost weights from DESIGN.md SS3's Phi(w): c_u on ||u_t||^2, c_du on
// ||u_t - u_{t-1}||^2. Not pinned by the paper or upstream (DESIGN.md's
// open items) -- exposed here so they're a config knob, not a magic
// number, until a tuned value is settled on.
struct QpWeights
{
  double c_u;
  double c_du;
};

// One horizon stage's affine dynamics parameters (DESIGN.md SS4):
// x_{t+1} = A x_t + B u_t + C.
struct QpStage
{
  casadi::DM A;  // kStateDim x kStateDim
  casadi::DM B;  // kStateDim x kControlDim
  casadi::DM C;  // kStateDim x 1
};

struct QpSolution
{
  casadi::DM x_traj;   // kStateDim x (N+1)
  casadi::DM u_traj;   // kControlDim x N
  casadi::DM lambda;   // safe_set_size x 1
  bool success;
  std::string message;  // populated with the solver's own error text on failure
};

// Builds the multi-shooting QP graph (DESIGN.md SS4) ONCE for a fixed
// horizon length N and safe-set size q = K * (number of laps loaded when
// the controller was constructed) -- q is fixed at construction because
// Opti's decision-variable dimensions can't change afterward. Growing the
// safe set mid-run (P increasing as more laps are recorded, DESIGN.md SS8
// step 8) would need a new QpBuilder; out of scope for the first
// dummy-A/B/C pass, which only ever has D^0 loaded.
//
// solve() re-parametrizes (A_t, B_t, C_t, x_k, u_{k-1}, the safe-set
// matrices) and re-solves the SAME graph every control step -- this is the
// whole point of keeping A_t/B_t/C_t as Opti parameters rather than
// decision variables (DESIGN.md SS4): building the graph is done once, the
// per-step cost is just a conic solve.
class QpBuilder
{
public:
  QpBuilder(
    casadi_int horizon_steps, casadi_int safe_set_size, const QpBounds & bounds,
    const QpWeights & weights, const std::string & solver_name = "qrqp");

  // stages.size() must equal N. warm-start values (x_warm, u_warm) seed
  // opti.set_initial() -- DESIGN.md SS8 step 2's linearization sequence
  // doubles as the solver's warm start.
  QpSolution solve(
    const casadi::DM & x_k, const casadi::DM & u_prev, const std::vector<QpStage> & stages,
    const casadi::DM & X_ss, const casadi::DM & J_ss, const casadi::DM & x_warm,
    const casadi::DM & u_warm);

private:
  casadi_int N;
  casadi_int q;
  QpBounds bounds;

  casadi::Opti opti;
  casadi::MX X;       // kStateDim x (N+1)
  casadi::MX U;       // kControlDim x N
  casadi::MX Lambda;  // q x 1

  casadi::MX x0_param;
  casadi::MX u_prev_param;
  std::vector<casadi::MX> A_params;
  std::vector<casadi::MX> B_params;
  std::vector<casadi::MX> C_params;
  casadi::MX Xss_param;
  casadi::MX Jss_param;
};

}  // namespace lmpc

#endif  // LMPC__QP_BUILDER_HPP_
