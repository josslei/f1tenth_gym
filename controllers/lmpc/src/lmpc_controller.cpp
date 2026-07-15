#include "lmpc_controller.hpp"

#include <chrono>
#include <stdexcept>

#include "log.hpp"

namespace lmpc {

LMPCController::LMPCController(const LmpcConfig &config_in)
    : config(config_in), dynamics_model(config.vehicle_params), integrator(),
      linearizer(dynamics_model, integrator, config.dt),
      track(config.centerline_csv_path),
      safe_set(config.seed_lap_csv_path, track.length()),
      cost_to_go_scale(safe_set.cost_scale()), qp_builder(nullptr),
      x(casadi::DM::zeros(kStateDim, 1)), t(0.0), has_state(false),
      u_prev(casadi::DM::zeros(kControlDim, 1)), actual_delta(0.0),
      x_warm(casadi::DM::zeros(kStateDim, config.horizon_steps + 1)),
      u_warm(casadi::DM::zeros(kControlDim, config.horizon_steps)),
      lambda_warm(casadi::DM::zeros(0, 1)), has_warm_start(false),
      x_pred(casadi::DM::zeros(kStateDim, config.horizon_steps + 1)) {
  rebuild_qp_builder();
}

casadi_int LMPCController::terminal_set_size() const {
  return safe_set.terminal_point_count(config.K);
}

void LMPCController::reset_lambda_warm_start() {
  const casadi_int q = terminal_set_size();
  lambda_warm = casadi::DM::ones(q, 1) / q;
}

void LMPCController::rebuild_qp_builder() {
  qp_builder = std::make_unique<QpBuilder>(
      config.horizon_steps, terminal_set_size(),
      QpBounds{config.a_min, config.a_max, config.delta_min, config.delta_max,
               config.ey_max, config.sv_max * config.dt},
      QpWeights{config.cost_to_go_weight, config.terminal_slack_weight,
                config.c_u, config.c_d_u, config.ey_slack_l1,
                config.ey_slack_l2},
      QpScaling{
          casadi::DM({config.v_max, config.scale_x_vy, config.scale_x_omega,
                      config.scale_x_epsi, track.length(), config.ey_max}),
          casadi::DM({config.a_max, config.delta_max}), cost_to_go_scale},
      config.solver_name);
  reset_lambda_warm_start();
}

void LMPCController::reset() {
  x = casadi::DM::zeros(kStateDim, 1);
  t = 0.0;
  has_state = false;
  u_prev = casadi::DM::zeros(kControlDim, 1);
  actual_delta = 0.0;
  x_warm = casadi::DM::zeros(kStateDim, config.horizon_steps + 1);
  u_warm = casadi::DM::zeros(kControlDim, config.horizon_steps);
  reset_lambda_warm_start();
  has_warm_start = false;
  x_pred = casadi::DM::zeros(kStateDim, config.horizon_steps + 1);
}

void LMPCController::update(const casadi::DM &x_in, double t_in,
                            double actual_delta_in) {
  if (x_in.size1() != kStateDim || x_in.size2() != 1) {
    throw std::invalid_argument("LMPCController::update: x must be a 6x1 "
                                "vector [vx, vy, omega, epsi, s, ey]");
  }
  x = x_in;
  t = t_in;
  actual_delta = actual_delta_in;
  has_state = true;
}

casadi::DM LMPCController::safe_set_query_scale() const {
  // The metric for locating safe-set data near a state is NOT the QP's
  // variable-conditioning scale, even though both are per-state vectors.
  // The QP scale normalizes s by the whole track's length so the decision
  // variable stays O(1) -- correct for solver conditioning, but as a
  // LOCALITY metric it makes track position nearly free: 2.0m of s counts
  // as 0.012, so "nearest" ends up decided by noise in the other
  // coordinates. Measured directly (2026-07-13): the terminal query
  // returned samples 2m BEHIND the terminal reference (J higher than the
  // data available right at it), the cost-to-go stopped pulling forward,
  // and the car decelerated from 3.8 to 1.5 m/s until the QP failed.
  // DESIGN.md SS2's pinned metric is D = diag(0,0,0,0,1,1) -- s and ey in
  // raw METERS, position-dominated -- so s's entry here is 1.0 (meters),
  // not track.length(). The remaining entries keep the conditioning-style
  // normalization: at these scales (vx/20 etc.) they act as mild
  // tie-breakers rather than competing with position, which is as close
  // to the pinned metric as the shared normalized_distance_sq interface
  // allows.
  return casadi::DM({config.v_max, config.scale_x_vy, config.scale_x_omega,
                     config.scale_x_epsi, 1.0, config.ey_max});
}

void LMPCController::seed_warm_start_from_safe_set() {
  // First-solve warm start: the recorded D^0 controls nearest the current
  // state (header comment has the full rationale for why NOT a
  // zero-control naive rollout). Only u_warm is taken from the recording;
  // x_warm is derived from the measured state by solve_once()'s own
  // rollout+linearize loop so the linearization sequence stays dynamically
  // consistent with the nominal model regardless of how the recorded lap's
  // own dynamics differed.
  const SafeSet::TrajectorySegment segment = safe_set.trajectory_segment(
      x, config.horizon_steps, safe_set_query_scale());
  u_warm = segment.u_traj;
  has_warm_start = true;
}

void LMPCController::shift_warm_start(const QpSolution &solution) {
  // Only the CONTROL trajectory shifts (receding horizon); the freed final
  // slot holds the last control constant. x_warm is deliberately NOT
  // shifted from solution.x_traj -- solve_once()'s own rollout+linearize
  // loop rebuilds it from the next measured state instead (its header
  // comment has the rationale).
  using casadi::Slice;

  const casadi_int N = config.horizon_steps;
  if (N > 1) {
    u_warm(Slice(), Slice(0, N - 1)) = solution.u_traj(Slice(), Slice(1, N));
  }
  u_warm(Slice(), N - 1) = solution.u_traj(Slice(), N - 1);
  has_warm_start = true;
}

QpSolution LMPCController::solve_once() {
  // recom.md's t_rollout+lin/t_knn checkpoints -- ControllerTimings'
  // comment (lmpc_controller.hpp) has the full rationale for what each
  // bucket covers.
  using Clock = std::chrono::steady_clock;
  const auto elapsed_ms = [](Clock::time_point from, Clock::time_point to) {
    return std::chrono::duration<double, std::milli>(to - from).count();
  };
  const Clock::time_point t_start = Clock::now();

  // x_warm is rebuilt as a nominal-model rollout from the MEASURED current
  // state under u_warm on EVERY call, not just the first (solve_once()'s
  // header comment has the full rationale) -- fused with the per-stage
  // linearization below into a single pass (recom.md item 1):
  // Linearizer::operator() already evaluates x_next alongside (A_t, B_t,
  // C_t) at the same call, so stage stg+1's x_ref is exactly stage stg's
  // x_next, and no state is ever linearized twice.
  using casadi::Slice;
  x_warm(Slice(), 0) = x;

  // DESIGN.md SS8 step 3 (dummy-A/B/C pass: steps 3a/3b skipped, so
  // A_t = A^f_t, B_t = B^f_t, C_t = C^f_t -- no learned error correction).
  const casadi_int N = config.horizon_steps;
  std::vector<QpStage> stages;
  stages.reserve(static_cast<std::size_t>(N));
  for (casadi_int stg = 0; stg < N; ++stg) {
    const casadi::DM x_ref = x_warm(Slice(), stg);
    const casadi::DM u_ref = u_warm(Slice(), stg);
    const casadi::DM u_prev_ref =
        (stg == 0) ? u_prev : casadi::DM(u_warm(Slice(), stg - 1));
    const double kappa_ref =
        track.curvature(static_cast<double>(x_ref(dynamics::S)));

    const LinearizedDynamics lin =
        linearizer(x_ref, u_ref, u_prev_ref, kappa_ref);
    x_warm(Slice(), stg + 1) = lin.x_next;
    stages.push_back(QpStage{lin.A, lin.B, lin.C});

    SPDLOG_LOGGER_TRACE(log(),
                        "stage {}: x_ref={} u_ref={} |A|max={} |B|max={}", stg,
                        x_ref.T(), u_ref.T(), casadi::DM::mmax(fabs(lin.A)),
                        casadi::DM::mmax(fabs(lin.B)));
  }

  const Clock::time_point t_rollout_lin_done = Clock::now();

  // Select a contiguous local trajectory segment from each lap using only
  // periodic track progress. Terminal slack handles dynamic-state mismatch.
  const casadi::DM x_terminal_ref = x_warm(casadi::Slice(), N);
  SafeSet::QueryResult safe_set_result = safe_set.query_local_segments(
      static_cast<double>(x_terminal_ref(dynamics::S)), config.K);
  const Clock::time_point t_knn_done = Clock::now();
  const casadi_int expected_q = terminal_set_size();
  if (safe_set_result.X_ss.size1() != kStateDim ||
      safe_set_result.X_ss.size2() != expected_q ||
      safe_set_result.J_ss.size1() != expected_q ||
      safe_set_result.J_ss.size2() != 1) {
    throw std::runtime_error("LMPCController::solve_once: safe-set query "
                             "returned invalid dimensions");
  }
  // Safe-set vertices are reselected every control step, so the previous
  // lambda entries no longer refer to the same columns. Seed the new simplex
  // at its feasible centroid instead of carrying incompatible coordinates.
  lambda_warm = casadi::DM::ones(expected_q, 1) / expected_q;

  // Anchor the QP's stage-0 steering-rate cost/constraint (QpBounds::
  // ddelta_max) against the PLANT's actual current steering angle, not this
  // controller's own last command -- see actual_delta's declaration
  // comment. Acceleration has no equivalent hard rate constraint and no
  // comparably direct raw measurement, so u_prev(A) is left as the last
  // commanded value.
  casadi::DM u_prev_anchor = u_prev;
  u_prev_anchor(dynamics::DELTA) = actual_delta;

  // DESIGN.md SS8 step 5.
  QpSolution solution =
      qp_builder->solve(x, u_prev_anchor, stages, safe_set_result.X_ss,
                        safe_set_result.J_ss, x_warm, u_warm, lambda_warm);

  // Populated regardless of solution.success -- see this function's own
  // return path in control(): the failure throw happens AFTER solve_once()
  // returns, so these are already valid by then.
  timings.rollout_lin_ms = elapsed_ms(t_start, t_rollout_lin_done);
  timings.knn_ms = elapsed_ms(t_rollout_lin_done, t_knn_done);
  timings.set_params_ms = solution.timings.set_params_ms;
  timings.solver_ms = solution.timings.solver_ms;
  timings.postcheck_ms = solution.timings.postcheck_ms;

  return solution;
}

casadi::DM LMPCController::control() {
  if (!has_state) {
    throw std::logic_error(
        "LMPCController::control: update() must be called before control()");
  }
  if (!has_warm_start) {
    seed_warm_start_from_safe_set();
  }

  // No retry: exactly one solve per control() call, bounded by the
  // solver's own max_iter, so per-step solve time stays bounded instead of
  // silently doubling on every failure (measured 2026-07-14: a retry here
  // was why solve time -- and therefore viewer FPS -- degraded so sharply
  // once the QP started failing, since every failing step paid for TWO
  // full solves before giving up). Every infeasibility is surfaced via
  // this exception, not masked by a second attempt; the caller's fallback
  // brake (runs/lmpc_drive.py) is what actually handles a real failure
  // (including gym's own low-speed plant divergence), not this layer.
  const QpSolution solution = solve_once();
  if (!solution.success) {
    qp_builder->clear_dual_warm_start();
    throw std::runtime_error("LMPCController::control: QP solve failed: " +
                             solution.message);
  }

  x_pred = solution.x_traj;
  shift_warm_start(solution);

  // DESIGN.md SS8 step 6: apply u_0*.
  u_prev = solution.u_traj(casadi::Slice(), 0);
  return u_prev;
}

void LMPCController::add_lap(const casadi::DM &x_lap, const casadi::DM &u_lap,
                             const casadi::DM &J_lap) {
  const casadi_int num_states = x_lap.size2();
  if (x_lap.size1() != kStateDim || num_states < 2 ||
      u_lap.size1() != kControlDim || u_lap.size2() != num_states - 1 ||
      J_lap.size1() != num_states || J_lap.size2() != 1) {
    throw std::invalid_argument(
        "LMPCController::add_lap: expected x_lap kStateDim x (T+1), "
        "u_lap kControlDim x T, J_lap (T+1) x 1 with T >= 1");
  }

  std::vector<SafeSetSample> lap;
  lap.reserve(static_cast<std::size_t>(num_states));
  for (casadi_int k = 0; k < num_states; ++k) {
    // The final state has no successor, hence no realized control -- same
    // has_control convention the seed-lap CSV loader produces.
    const bool has_control = k < num_states - 1;
    lap.push_back(SafeSetSample{x_lap(casadi::Slice(), k),
                                has_control
                                    ? casadi::DM(u_lap(casadi::Slice(), k))
                                    : casadi::DM::zeros(kControlDim, 1),
                                static_cast<double>(J_lap(k)), has_control});
  }
  // QpBuilder's J normalization (scaling.j) stays pinned to D^0's own cost
  // scale from construction time: later laps are only ever FASTER (smaller
  // J), so the fixed scale keeps J_ss/scaling.j in (0, 1] -- exactly the
  // conditioning it was chosen for.
  safe_set.add_lap(std::move(lap));
  rebuild_qp_builder();
}

} // namespace lmpc
