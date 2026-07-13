#include "qp_builder.hpp"

#include <sstream>
#include <stdexcept>

#include "dynamics/common.hpp"

namespace lmpc {

QpBuilder::QpBuilder(casadi_int horizon_steps, casadi_int safe_set_size,
                     const QpBounds &bounds, const QpWeights &weights,
                     const QpScaling &scaling, const std::string &solver_name)
    : N(horizon_steps), q(safe_set_size), bounds(bounds),
      solver_name(solver_name), scaling(scaling), opti("conic") {
  using casadi::MX;
  using casadi::Slice;
  using dynamics::A;
  using dynamics::DELTA;
  using dynamics::EY;
  using dynamics::kControlDim;
  using dynamics::kStateDim;

  // X/U are the actual decision variables qrqp solves for -- kept O(1) by
  // construction (this->scaling's factors, LmpcConfig's comment on
  // scale_x_vy et al. has the full rationale) so the QP's KKT system stays
  // well-conditioned regardless of the underlying physical units' spread.
  // X_phys/U_phys are the physical-unit expressions used in every
  // constraint/cost below and extracted at solve time -- CasADi's autodiff
  // treats this as ordinary symbolic substitution, so nothing downstream
  // needs to know scaling exists except the scale factors themselves.
  X = opti.variable(kStateDim, N + 1);
  U = opti.variable(kControlDim, N);
  Alpha = opti.variable(1, 1);
  Lambda = MX::vertcat({1.0 - Alpha, Alpha});
  X_phys = this->scaling.x * X;
  U_phys = this->scaling.u * U;

  x0_param = opti.parameter(kStateDim, 1);
  u_prev_param = opti.parameter(kControlDim, 1);
  Xss_param = opti.parameter(kStateDim, q);
  Jss_param = opti.parameter(q, 1);
  TerminalSlack =
      (X_phys(Slice(), N) - MX::mtimes(Xss_param, Lambda)) / this->scaling.x;

  A_params.reserve(N);
  B_params.reserve(N);
  C_params.reserve(N);
  for (casadi_int t = 0; t < N; ++t) {
    A_params.push_back(opti.parameter(kStateDim, kStateDim));
    B_params.push_back(opti.parameter(kStateDim, kControlDim));
    C_params.push_back(opti.parameter(kStateDim, 1));
  }

  // DESIGN.md SS4: g_eq(w) = 0. Written entirely in physical units
  // (X_phys/U_phys) -- A_t/B_t/C_t come out of Linearizer in physical
  // units too (that class has no reason to know about QP-specific
  // scaling), so the affine dynamics equality is literally the physical
  // relation x_{t+1} = A_t x_t + B_t u_t + C_t; CasADi's graph handles the
  // implied division back through scaling.x for the actual scaled X.
  opti.subject_to(X_phys(Slice(), 0) == x0_param);
  for (casadi_int t = 0; t < N; ++t) {
    const MX x_next = MX::mtimes(A_params[t], X_phys(Slice(), t)) +
                      MX::mtimes(B_params[t], U_phys(Slice(), t)) + C_params[t];
    opti.subject_to(X_phys(Slice(), t + 1) == x_next);
  }
  // DESIGN.md SS4: g_ineq(w) <= 0 -- U box and the ey half of X (class
  // comment in qp_builder.hpp on why only ey is bounded here).
  // Do NOT write these as C++ chained comparisons (`lo <= expr <= hi`).
  // Unlike mathematical notation, C++ parses that as `(lo <= expr) <= hi`,
  // so CasADi never receives the intended lower+upper box constraints. That
  // exact bug let qrqp return steering trajectories with delta around +/-2 rad
  // even though the intended bound is +/-0.4189. Encode every side explicitly.
  const double scale_a = static_cast<double>(this->scaling.u(A));
  const double scale_delta = static_cast<double>(this->scaling.u(DELTA));
  const double scale_ey = static_cast<double>(this->scaling.x(EY));
  opti.subject_to(bounds.a_min / scale_a <= U(A, Slice()));
  opti.subject_to(U(A, Slice()) <= bounds.a_max / scale_a);
  opti.subject_to(bounds.delta_min / scale_delta <= U(DELTA, Slice()));
  opti.subject_to(U(DELTA, Slice()) <= bounds.delta_max / scale_delta);
  opti.subject_to(-bounds.ey_max / scale_ey <= X(EY, Slice()));
  opti.subject_to(X(EY, Slice()) <= bounds.ey_max / scale_ey);
  // Minimal barycentric parameterization of the two-point terminal simplex.
  opti.subject_to(0 <= Alpha);
  opti.subject_to(Alpha <= 1.0);

  // DESIGN.md SS3's Phi(w). The min-time indicator 1_F(x_t) is a constant
  // +1 per stage in practice (SS3: the horizon never actually crosses the
  // finish line mid-plan) -- a constant additive N doesn't change the QP's
  // argmin, so it is omitted rather than adding dead cost terms to the
  // graph.
  MX cost = MX::mtimes(Jss_param.T(), Lambda);
  for (casadi_int t = 0; t < N; ++t) {
    const MX u_t = U_phys(Slice(), t);
    const MX u_prev_t = (t == 0) ? u_prev_param : MX(U_phys(Slice(), t - 1));
    cost += (weights.c_u / this->scaling.j) * MX::sumsqr(u_t) +
            (weights.c_du / this->scaling.j) * MX::sumsqr(u_t - u_prev_t);
  }
  cost += weights.terminal_slack *
          MX::sumsqr(weights.terminal_slack_state * TerminalSlack);
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
  opti.solver(solver_name, casadi::Dict{{"print_time", false},
                                        {"print_iter", false},
                                        {"print_header", false},
                                        {"print_info", false},
                                        {"error_on_fail", false}});
}

QpSolution QpBuilder::solve(const casadi::DM &x_k, const casadi::DM &u_prev,
                            const std::vector<QpStage> &stages,
                            const casadi::DM &X_ss, const casadi::DM &J_ss,
                            const casadi::DM &x_warm, const casadi::DM &u_warm,
                            const casadi::DM &lambda_warm) {
  using casadi::Slice;

  if (static_cast<casadi_int>(stages.size()) != N) {
    throw std::invalid_argument(
        "QpBuilder::solve: stages.size() must equal the horizon length");
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
    opti.set_value(Jss_param, J_ss / scaling.j);
    for (casadi_int t = 0; t < N; ++t) {
      opti.set_value(A_params[t], stages[t].A);
      opti.set_value(B_params[t], stages[t].B);
      opti.set_value(C_params[t], stages[t].C);
    }
    // x_warm/u_warm are physical (callers, e.g. LMPCController, only ever
    // deal in physical units) -- divide through by the scale factors to
    // seed the actual (scaled) decision variables. CasADi broadcasts a
    // column-vector divisor across a matrix's columns elementwise (verified
    // directly against this vendored CasADi build, not assumed).
    opti.set_initial(X, x_warm / scaling.x);
    opti.set_initial(U, u_warm / scaling.u);
    opti.set_initial(Alpha, lambda_warm(1));

    const casadi::OptiSol sol = opti.solve();
    const casadi::DM x_traj = sol.value(X_phys);
    const casadi::DM u_traj = sol.value(U_phys);
    const casadi::DM lambda = sol.value(Lambda);
    const casadi::DM terminal_slack = sol.value(TerminalSlack);

    const bool regular = x_traj.is_regular() && u_traj.is_regular() &&
                         lambda.is_regular() && terminal_slack.is_regular();

    const double a_hi =
        static_cast<double>(casadi::DM::mmax(u_traj(dynamics::A, Slice())));
    const double a_lo =
        static_cast<double>(casadi::DM::mmin(u_traj(dynamics::A, Slice())));
    const double delta_hi =
        static_cast<double>(casadi::DM::mmax(u_traj(dynamics::DELTA, Slice())));
    const double delta_lo =
        static_cast<double>(casadi::DM::mmin(u_traj(dynamics::DELTA, Slice())));
    const double lambda_hi_over_bound =
        static_cast<double>(casadi::DM::mmax(lambda - 1.0));
    const double lambda_lo = static_cast<double>(casadi::DM::mmin(lambda));

    constexpr double kBoundsTolerance = 1e-3;
    const bool within_bounds =
        regular && a_hi <= bounds.a_max + kBoundsTolerance &&
        a_lo >= bounds.a_min - kBoundsTolerance &&
        delta_hi <= bounds.delta_max + kBoundsTolerance &&
        delta_lo >= bounds.delta_min - kBoundsTolerance &&
        lambda_hi_over_bound <= kBoundsTolerance &&
        lambda_lo >= -kBoundsTolerance;

    if (!within_bounds) {
      std::ostringstream diag;
      diag << "solver (" << solver_name << ") reported success but returned a ";
      if (!regular) {
        diag << "non-finite (NaN/Inf) solution";
      } else {
        diag << "solution violating its own box constraints. "
             << "a: [" << a_lo << ", " << a_hi << "] vs bound [" << bounds.a_min
             << ", " << bounds.a_max << "]; "
             << "delta: [" << delta_lo << ", " << delta_hi << "] vs bound ["
             << bounds.delta_min << ", " << bounds.delta_max << "]; "
             << "lambda: min=" << lambda_lo
             << " max-over-bound=" << lambda_hi_over_bound;
      }
      throw std::runtime_error(diag.str());
    }

    result.x_traj = x_traj;
    result.u_traj = u_traj;
    result.lambda = lambda;
    result.terminal_slack = terminal_slack;
    result.success = true;
  } catch (const std::exception &e) {
    result.x_traj = x_warm;
    result.u_traj = u_warm;
    result.lambda = lambda_warm;
    result.terminal_slack = casadi::DM::zeros(dynamics::kStateDim, 1);
    result.success = false;
    result.message = e.what();
  }
  return result;
}

} // namespace lmpc
