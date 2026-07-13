#include "qp_builder.hpp"

#include <stdexcept>

#include "dynamics/common.hpp"

namespace lmpc
{

QpBuilder::QpBuilder(
  casadi_int horizon_steps, casadi_int safe_set_size, const QpBounds & bounds,
  const QpWeights & weights, const std::string & solver_name)
: N(horizon_steps), q(safe_set_size), bounds(bounds), opti("conic")
{
  using casadi::MX;
  using casadi::Slice;
  using dynamics::A;
  using dynamics::DELTA;
  using dynamics::EY;
  using dynamics::kControlDim;
  using dynamics::kStateDim;

  X = opti.variable(kStateDim, N + 1);
  U = opti.variable(kControlDim, N);
  Lambda = opti.variable(q, 1);

  x0_param = opti.parameter(kStateDim, 1);
  u_prev_param = opti.parameter(kControlDim, 1);
  Xss_param = opti.parameter(kStateDim, q);
  Jss_param = opti.parameter(q, 1);

  A_params.reserve(N);
  B_params.reserve(N);
  C_params.reserve(N);
  for (casadi_int t = 0; t < N; ++t) {
    A_params.push_back(opti.parameter(kStateDim, kStateDim));
    B_params.push_back(opti.parameter(kStateDim, kControlDim));
    C_params.push_back(opti.parameter(kStateDim, 1));
  }

  // DESIGN.md SS4: g_eq(w) = 0.
  opti.subject_to(X(Slice(), 0) == x0_param);
  for (casadi_int t = 0; t < N; ++t) {
    const MX x_next = MX::mtimes(A_params[t], X(Slice(), t)) +
      MX::mtimes(B_params[t], U(Slice(), t)) + C_params[t];
    opti.subject_to(X(Slice(), t + 1) == x_next);
  }
  opti.subject_to(MX::mtimes(Xss_param, Lambda) == X(Slice(), N));
  opti.subject_to(MX::sum1(Lambda) == 1);

  // DESIGN.md SS4: g_ineq(w) <= 0 -- U box and the ey half of X (class
  // comment in qp_builder.hpp on why only ey is bounded here).
  opti.subject_to(bounds.a_min <= U(A, Slice()) <= bounds.a_max);
  opti.subject_to(bounds.delta_min <= U(DELTA, Slice()) <= bounds.delta_max);
  opti.subject_to(-bounds.ey_max <= X(EY, Slice()) <= bounds.ey_max);
  opti.subject_to(0 <= Lambda <= 1);

  // DESIGN.md SS3's Phi(w). The min-time indicator 1_F(x_t) is a constant
  // +1 per stage in practice (SS3: the horizon never actually crosses the
  // finish line mid-plan) -- a constant additive N doesn't change the QP's
  // argmin, so it is omitted rather than adding dead cost terms to the
  // graph.
  MX cost = MX::mtimes(Jss_param.T(), Lambda);
  for (casadi_int t = 0; t < N; ++t) {
    const MX u_t = U(Slice(), t);
    const MX u_prev_t = (t == 0) ? u_prev_param : MX(U(Slice(), t - 1));
    cost += weights.c_u * MX::sumsqr(u_t) + weights.c_du * MX::sumsqr(u_t - u_prev_t);
  }
  opti.minimize(cost);

  // Every control step re-solves this same graph (class comment above), so
  // per-iteration solver logging would spam stdout every ~dt seconds in
  // closed-loop use -- silenced here rather than left to each caller.
  //
  // Passed as the SECOND (plugin_options) argument, not the third
  // (solver_options): Opti::solver's third arg gets nested under a key
  // named after the solver itself (OptiNode::solver in
  // optistack_internal.cpp: solver_options_[solver_name] = solver_options)
  // before being handed to casadi::conic() -- which this vendored CasADi
  // version does not actually unpack for the qrqp plugin, producing
  // "Unknown option: qrqp" (the literal nesting key itself rejected as an
  // unrecognized option). Flat options in the second argument avoid that
  // nesting entirely.
  // error_on_fail=false: on a failed solve, qrqp otherwise dumps every
  // input matrix to stdout before throwing (a wall of text per failure,
  // every control step in closed-loop use) -- opti.solve() still throws
  // its own concise status error regardless of this flag, which
  // QpBuilder::solve()'s try/catch below handles either way.
  opti.solver(
    solver_name, casadi::Dict{
      {"print_time", false}, {"print_iter", false}, {"print_header", false},
      {"print_info", false}, {"error_on_fail", false}});
}

QpSolution QpBuilder::solve(
  const casadi::DM & x_k, const casadi::DM & u_prev, const std::vector<QpStage> & stages,
  const casadi::DM & X_ss, const casadi::DM & J_ss, const casadi::DM & x_warm,
  const casadi::DM & u_warm)
{
  using casadi::Slice;

  if (static_cast<casadi_int>(stages.size()) != N) {
    throw std::invalid_argument("QpBuilder::solve: stages.size() must equal the horizon length");
  }

  QpSolution result;
  try {
    // set_value/set_initial calls are INSIDE this try block deliberately,
    // not just opti.solve() -- a non-finite A_t/B_t/C_t (e.g. from a
    // degenerate linearization) makes CasADi's set_value itself assert
    // ("v.is_regular() failed"), which is a plain C++ exception like any
    // solver failure. Narrowing the try to only wrap solve() would let
    // that assertion escape uncaught.
    opti.set_value(x0_param, x_k);
    opti.set_value(u_prev_param, u_prev);
    opti.set_value(Xss_param, X_ss);
    opti.set_value(Jss_param, J_ss);
    for (casadi_int t = 0; t < N; ++t) {
      opti.set_value(A_params[t], stages[t].A);
      opti.set_value(B_params[t], stages[t].B);
      opti.set_value(C_params[t], stages[t].C);
    }
    opti.set_initial(X, x_warm);
    opti.set_initial(U, u_warm);

    const casadi::OptiSol sol = opti.solve();
    casadi::DM x_traj = sol.value(X);
    casadi::DM u_traj = sol.value(U);
    casadi::DM lambda = sol.value(Lambda);

    // error_on_fail=false (constructor) stops qrqp from throwing on a
    // non-converged solve, but that also means a solve qrqp itself
    // considers "successful" can still return a finite-but-garbage
    // iterate that violates the very box constraints it was given --
    // a documented qrqp behavior on ill-conditioned QPs (DESIGN.md's open
    // item: no variable scaling yet). is_regular() alone only catches
    // NaN/Inf, not this. Bounds are checked with a small numerical
    // tolerance, not exact equality, since the solver's own constraint
    // tolerance already permits tiny (sub-tolerance) overshoot.
    constexpr double kBoundsTolerance = 1e-3;
    const bool regular = x_traj.is_regular() && u_traj.is_regular() && lambda.is_regular();
    const bool within_bounds = regular &&
      static_cast<double>(casadi::DM::mmax(u_traj(dynamics::A, Slice()))) <=
      bounds.a_max + kBoundsTolerance &&
      static_cast<double>(casadi::DM::mmin(u_traj(dynamics::A, Slice()))) >=
      bounds.a_min - kBoundsTolerance &&
      static_cast<double>(casadi::DM::mmax(u_traj(dynamics::DELTA, Slice()))) <=
      bounds.delta_max + kBoundsTolerance &&
      static_cast<double>(casadi::DM::mmin(u_traj(dynamics::DELTA, Slice()))) >=
      bounds.delta_min - kBoundsTolerance;

    if (!within_bounds) {
      throw std::runtime_error(
              regular ?
              "qrqp reported success but returned a control trajectory violating its own "
              "box constraints (known conditioning symptom on an unscaled QP)" :
              "qrqp reported success but returned a non-finite (NaN/Inf) solution");
    }

    result.x_traj = x_traj;
    result.u_traj = u_traj;
    result.lambda = lambda;
    result.success = true;
  } catch (const std::exception & e) {
    // Leave the previous warm-start values as the reported trajectory so
    // the caller still has something plausible for the next step's
    // linearization sequence (DESIGN.md SS8 step 2) -- but flag failure so
    // it doesn't apply a stale control as if it were a fresh solve.
    result.x_traj = x_warm;
    result.u_traj = u_warm;
    result.lambda = casadi::DM::zeros(q, 1);
    result.success = false;
    result.message = e.what();
  }
  return result;
}

}  // namespace lmpc
