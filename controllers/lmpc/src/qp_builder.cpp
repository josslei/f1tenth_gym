#include "qp_builder.hpp"

#include <chrono>
#include <cmath>
#include <limits>
#include <sstream>
#include <stdexcept>

#include "dynamics/common.hpp"
#include "log.hpp"

namespace lmpc {

namespace {

void require_shape(const casadi::DM &value, casadi_int rows, casadi_int cols,
                   const char *name) {
  if (value.size1() != rows || value.size2() != cols) {
    std::ostringstream message;
    message << "QpBuilder::solve: " << name << " must be " << rows << "x"
            << cols << ", got " << value.size1() << "x" << value.size2();
    throw std::invalid_argument(message.str());
  }
}

} // namespace

QpBuilder::QpBuilder(casadi_int horizon_steps, casadi_int safe_set_size,
                     const QpBounds &bounds, const QpWeights &weights,
                     const QpScaling &scaling, const std::string &solver_name)
    : N(horizon_steps), q(safe_set_size), bounds(bounds),
      solver_name(solver_name), scaling(scaling),
      // "conic" mode restricts Opti to CasADi's conic-solver plugin
      // interface (qrqp, osqp, ...); ipopt is registered under the
      // separate nlpsol interface instead, so it needs plain (default,
      // NLP-mode) Opti. Same MX graph either way -- this QP's affine
      // dynamics/convex quadratic objective satisfy "conic" mode's extra
      // structural requirements, but nothing about how the graph itself is
      // built below depends on which mode constructed it.
      opti(solver_name == "ipopt" ? casadi::Opti() : casadi::Opti("conic")) {
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
  EySlack = opti.variable(1, N + 1);
  Lambda = opti.variable(q, 1);
  ETerminal = opti.variable(kStateDim, 1);
  X_phys = this->scaling.x * X;
  U_phys = this->scaling.u * U;

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
  // The ey corridor is SOFT (per-stage slack, exact-plus-quadratic penalty
  // in the cost below), not a hard box: x_0 is pinned to the measurement by
  // equality, so one disturbance pushing the car past ey_max -- or a
  // high-speed stage where the linearized reachable tube can't stay inside
  // the corridor under the steering-rate limit -- would otherwise make the
  // whole QP instantly infeasible ("Failed to calculate search direction")
  // exactly when the controller most needs a recovery plan. The l1 term
  // keeps the penalty exact (slack stays identically 0 whenever the hard
  // corridor is achievable), so this changes nothing on the nominal path.
  opti.subject_to(0 <= EySlack);
  opti.subject_to(-bounds.ey_max / scale_ey - EySlack <= X(EY, Slice()));
  opti.subject_to(X(EY, Slice()) <= bounds.ey_max / scale_ey + EySlack);
  // Steering-rate constraint (QpBounds::ddelta_max's comment): every
  // planned per-stage steering change must be executable by gym's
  // rate-limited actuator. Anchored at u_prev_param for stage 0, the same
  // anchor the control-rate cost below uses. Written in physical units;
  // both sides encoded explicitly (no C++ chained comparisons -- see the
  // U box constraints above).
  for (casadi_int t = 0; t < N; ++t) {
    const MX delta_prev =
        (t == 0) ? MX(u_prev_param(DELTA)) : MX(U_phys(DELTA, t - 1));
    const MX ddelta = U_phys(DELTA, t) - delta_prev;
    opti.subject_to(-bounds.ddelta_max <= ddelta);
    opti.subject_to(ddelta <= bounds.ddelta_max);
  }
  opti.subject_to(X_phys(Slice(), N) ==
                  MX::mtimes(Xss_param, Lambda) + this->scaling.x * ETerminal);
  opti.subject_to(0 <= Lambda);
  opti.subject_to(Lambda <= 1.0);
  opti.subject_to(MX::sum1(Lambda) == 1.0);

  // DESIGN.md SS3's Phi(w). The min-time indicator 1_F(x_t) is a constant
  // +1 per stage in practice (SS3: the horizon never actually crosses the
  // finish line mid-plan) -- a constant additive N doesn't change the QP's
  // argmin, so it is omitted rather than adding dead cost terms to the
  // graph.
  MX cost = weights.cost_to_go * MX::mtimes(Jss_param.T(), Lambda);
  cost += weights.terminal_slack * MX::sumsqr(ETerminal);
  for (casadi_int t = 0; t < N; ++t) {
    const MX u_t = U(Slice(), t);
    const MX u_prev_t =
        (t == 0) ? u_prev_param / this->scaling.u : MX(U(Slice(), t - 1));
    const MX du_t = u_t - u_prev_t;
    cost += weights.control * MX::sumsqr(u_t) +
            weights.control_rate * MX::sumsqr(du_t);
  }
  // Soft-ey penalty (constraint block above): l1 keeps it exact, l2 keeps
  // the recovery direction well-scaled once a violation does occur.
  cost += weights.ey_slack_l1 * MX::sum2(EySlack) +
          weights.ey_slack_l2 * MX::sumsqr(EySlack);
  // Regularize otherwise unpenalized scaled state directions.
  constexpr double kRegularization = 1e-6;
  cost += kRegularization * MX::sumsqr(X);
  opti.minimize(cost);

  // Every control step re-solves this same graph (class comment above), so
  // per-iteration solver logging would spam stdout every ~dt seconds in
  // closed-loop use -- silenced here rather than left to each caller.
  if (solver_name == "ipopt") {
    // Opti::solver's third arg is ALREADY auto-wrapped as
    // {solver_name: third_arg} before being handed to casadi::nlpsol()
    // (OptiNode::solver in optistack_internal.cpp: solver_options_[
    // solver_name] = solver_options) -- so ipopt's OWN options
    // (max_iter, print_level, ...) go FLAT here, not nested under an
    // extra "ipopt" key of our own: that would double-nest into
    // {"ipopt": {"ipopt": {...}}}, which is exactly what produced
    // "No such IPOPT option: ipopt" (IpoptInterface tried to set a
    // literal option NAMED "ipopt" from the surviving outer key) when
    // first tried (2026-07-14). print_time/expand are generic nlpsol-level
    // options, so they go in the SECOND (plugin_options) arg instead.
    // sb="yes" silences ipopt's startup banner (no print_header/
    // print_info equivalent for it). error_on_fail is qrqp/conic-only;
    // ipopt failures are instead read off OptiSol the same way
    // solve_limited() already handles a qrqp non-convergence.
    //
    // recom.md item 2: expand=true converts the fixed MX graph to SX at
    // solver-build time -- a real win here specifically because the SAME
    // graph is re-solved every control step (SX ops are scalar-unrolled,
    // no MX heap-allocated node overhead per re-evaluation), unlike a
    // build-once/solve-once NLP where expansion cost wouldn't be amortized.
    // warm_start_init_point=yes is what makes IPOPT actually USE the
    // primal (X/U/Lambda, already set via set_initial below) and dual
    // (lam_g_warm, this class's own member) initial guesses instead of
    // reprojecting to its own default starting point. tol/acceptable_tol
    // relaxed from IPOPT's own stricter defaults: this project's terminal
    // residual check already accepts 1e-4 (qp_builder.cpp's
    // kTerminalEqualityTolerance), so demanding tighter KKT convergence
    // from the solver itself buys nothing this controller can use.
    // max_iter deliberately left at 2000, NOT dropped to something small --
    // recom.md's own caution: a successful solve doesn't usually approach
    // 2000 anyway, so this isn't the lever that matters; shrinking it
    // without first recording actual iteration counts risks turning a
    // slow-but-recoverable solve into a spurious failure instead.
    opti.solver(solver_name,
                casadi::Dict{{"print_time", false}, {"expand", true}},
                casadi::Dict{{"max_iter", 2000},
                             {"print_level", 0},
                             {"sb", "yes"},
                             {"tol", 1e-6},
                             {"acceptable_tol", 1e-4},
                             {"acceptable_iter", 2},
                             {"warm_start_init_point", "yes"}});
  } else {
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
    // input matrix to stdout. solve_limited() accepts iteration-limited
    // results but still throws for statuses such as infeasibility; the
    // try/catch below handles those while the post-solve checks reject
    // malformed iterates.
    opti.solver(solver_name, casadi::Dict{{"max_iter", 2000},
                                          {"print_time", false},
                                          {"print_iter", false},
                                          {"print_header", false},
                                          {"print_info", false},
                                          {"error_on_fail", false}});
  }
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
  require_shape(x_k, dynamics::kStateDim, 1, "x_k");
  require_shape(u_prev, dynamics::kControlDim, 1, "u_prev");
  require_shape(X_ss, dynamics::kStateDim, q, "X_ss");
  require_shape(J_ss, q, 1, "J_ss");
  require_shape(x_warm, dynamics::kStateDim, N + 1, "x_warm");
  require_shape(u_warm, dynamics::kControlDim, N, "u_warm");
  require_shape(lambda_warm, q, 1, "lambda_warm");
  if (!lambda_warm.is_regular()) {
    throw std::invalid_argument(
        "QpBuilder::solve: lambda_warm must contain only finite values");
  }

  // recom.md's t_set-params/t_solver/t_postcheck: checkpoints declared
  // outside the try block, each pre-initialized to t_start, so a failure
  // partway through still yields non-negative, correctly-attributed
  // durations (the catch block below documents the exact attribution rule
  // on failure) instead of the negative spans a naive "diff two timestamps
  // taken inside try" scheme would produce once an earlier phase never
  // reaches its own checkpoint.
  using Clock = std::chrono::steady_clock;
  const auto elapsed_ms = [](Clock::time_point from, Clock::time_point to) {
    return std::chrono::duration<double, std::milli>(to - from).count();
  };
  const Clock::time_point t_start = Clock::now();
  Clock::time_point t_params_done = t_start;
  Clock::time_point t_solve_done = t_start;

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
    opti.set_initial(Lambda, lambda_warm);
    const casadi::DM projection_warm = casadi::DM::mtimes(X_ss, lambda_warm);
    const casadi::DM terminal_slack_warm =
        (x_warm(Slice(), N) - projection_warm) / scaling.x;
    opti.set_initial(ETerminal, terminal_slack_warm);
    // recom.md item 2: dual warm start, IPOPT-only (this class's own member
    // comment has the scoping rationale). Skipped on a QpBuilder's first
    // solve (has_dual_warm_start starts false) or right after this instance
    // was rebuilt for a resized safe set -- lam_g_warm's dimension would no
    // longer match this graph's constraint count.
    if (solver_name == "ipopt" && has_dual_warm_start) {
      opti.set_initial(opti.lam_g(), lam_g_warm);
    }
    t_params_done = Clock::now();

    const casadi::OptiSol sol = opti.solve_limited();
    t_solve_done = Clock::now();
    const casadi::DM x_traj = sol.value(X_phys);
    const casadi::DM u_traj = sol.value(U_phys);
    const casadi::DM lambda = sol.value(Lambda);
    const casadi::DM terminal_slack = sol.value(ETerminal);
    const casadi::DM terminal_projection = casadi::DM::mtimes(X_ss, lambda);
    const casadi::DM terminal_residual =
        (x_traj(Slice(), N) - terminal_projection -
         scaling.x * terminal_slack) /
        scaling.x;
    const double lambda_sum = static_cast<double>(casadi::DM::sum1(lambda));

    SPDLOG_LOGGER_DEBUG(
        log(),
        "q={} X_ss={}x{} J_ss={}x{}\nx0={}\nX_ss={}\nJ_ss={} scale.j={}\n"
        "lambda sum={} min={} max={}\nx_N={}\nX_ss*lambda={}\n"
        "normalized terminal slack={} norm_inf={}\n"
        "normalized terminal equality residual={}",
        q, X_ss.size1(), X_ss.size2(), J_ss.size1(), J_ss.size2(), x_k.T(),
        X_ss, J_ss.T(), scaling.j, lambda_sum, casadi::DM::mmin(lambda),
        casadi::DM::mmax(lambda), x_traj(casadi::Slice(), N).T(),
        terminal_projection.T(), terminal_slack.T(),
        casadi::DM::mmax(fabs(terminal_slack)), terminal_residual.T());

    const bool regular = x_traj.is_regular() && u_traj.is_regular() &&
                         lambda.is_regular() && terminal_slack.is_regular() &&
                         terminal_residual.is_regular();

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
    const double terminal_residual_max =
        terminal_residual.is_regular()
            ? static_cast<double>(casadi::DM::mmax(fabs(terminal_residual)))
            : std::numeric_limits<double>::infinity();

    // qrqp's convergence check declares success once no further active-set
    // flip is attempted, which is NOT a strict primal-feasibility
    // guarantee -- it can return solutions a fraction of a percent past a
    // box bound (root-caused by reading casadi_qrqp.hpp's
    // casadi_qrqp_prepare, this project's lmpc-solver-degeneracy memory).
    // Accept violations up to kViolationFraction of each bound's own range
    // WITHOUT mutating the solution: gym's own actuator layer
    // (steering_constraint/accl_constraints in dynamic_models.py) clamps
    // the realized input at the true physical limit regardless, so
    // applying e.g. a = 9.58 vs the 9.51 bound is physically identical to
    // applying 9.51 -- while post-hoc clamping the returned trajectory was
    // measured (2026-07-13) to corrupt the warm start (x_traj is solved
    // self-consistently against the UNCLAMPED u_traj). Anything past the
    // fraction is still a genuine solver failure and throws.
    constexpr double kViolationFraction = 0.02;
    const double a_tol = kViolationFraction * (bounds.a_max - bounds.a_min);
    const double delta_tol =
        kViolationFraction * (bounds.delta_max - bounds.delta_min);
    const double lambda_tol = kViolationFraction; // lambda's range is [0, 1]
    const bool within_bounds = regular && a_hi <= bounds.a_max + a_tol &&
                               a_lo >= bounds.a_min - a_tol &&
                               delta_hi <= bounds.delta_max + delta_tol &&
                               delta_lo >= bounds.delta_min - delta_tol &&
                               lambda_hi_over_bound <= lambda_tol &&
                               lambda_lo >= -lambda_tol;

    constexpr double kTerminalEqualityTolerance = 1e-4;
    constexpr double kLambdaSumTolerance = 1e-6;
    if (regular && (terminal_residual_max > kTerminalEqualityTolerance ||
                    std::abs(lambda_sum - 1.0) > kLambdaSumTolerance)) {
      std::ostringstream diag;
      diag << "solver (" << solver_name
           << ") returned a solution violating the terminal simplex equality. "
           << "normalized residual max=" << terminal_residual_max
           << "; lambda sum=" << lambda_sum;
      throw std::runtime_error(diag.str());
    }

    if (!within_bounds) {
      std::ostringstream diag;
      diag << "solver (" << solver_name << ") returned a ";
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

    // recom.md item 2: captured only on a genuinely accepted solve (past
    // every check above), same IPOPT-only scope as the read side.
    if (solver_name == "ipopt") {
      lam_g_warm = sol.value(opti.lam_g());
      has_dual_warm_start = lam_g_warm.is_regular();
      SPDLOG_LOGGER_DEBUG(log(), "ipopt iter_count={} (dual warm start {})",
                          opti.stats().at("iter_count"),
                          has_dual_warm_start
                              ? "primed for next solve"
                              : "unavailable (non-finite lam_g)");
    }
  } catch (const std::exception &e) {
    result.x_traj = x_warm;
    result.u_traj = u_warm;
    result.lambda = lambda_warm;
    result.terminal_slack = casadi::DM::zeros(dynamics::kStateDim, 1);
    result.success = false;
    result.message = e.what();
    // Attribution on failure: t_solve_done never advanced past whatever it
    // was when the exception fired (still == t_params_done if params
    // succeeded and solve_limited()/postcheck is what threw), so folding
    // "now" into it here means the elapsed_ms() calls below attribute all
    // of that time to solver_ms, not postcheck_ms -- postcheck never ran on
    // a failed solve, so it should never show non-zero time.
    t_solve_done = Clock::now();
  }
  result.timings.set_params_ms = elapsed_ms(t_start, t_params_done);
  result.timings.solver_ms = elapsed_ms(t_params_done, t_solve_done);
  result.timings.postcheck_ms = elapsed_ms(t_solve_done, Clock::now());
  return result;
}

void QpBuilder::clear_dual_warm_start() {
  has_dual_warm_start = false;
  lam_g_warm = casadi::DM();
}

} // namespace lmpc
