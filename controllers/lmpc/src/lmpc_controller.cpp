#include "lmpc_controller.hpp"

#include <cstdlib>
#include <iostream>
#include <stdexcept>

namespace lmpc {

LMPCController::LMPCController(const LmpcConfig &config_in)
    : config(config_in), dynamics_model(config.vehicle_params), integrator(),
      linearizer(dynamics_model, integrator, config.dt),
      track(config.centerline_csv_path), safe_set(config.seed_lap_csv_path),
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
      QpWeights{config.cost_to_go_weight,
                casadi::DM({config.c_a, config.c_delta}),
                casadi::DM({config.c_d_a, config.c_d_delta}),
                config.ey_slack_l1, config.ey_slack_l2},
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
  // x_warm is derived from the measured state by
  // rollout_warm_states_from_current() so the linearization sequence stays
  // dynamically consistent with the nominal model regardless of how the
  // recorded lap's own dynamics differed.
  const SafeSet::TrajectorySegment segment = safe_set.trajectory_segment(
      x, config.horizon_steps, safe_set_query_scale());
  u_warm = segment.u_traj;
  has_warm_start = true;
}

void LMPCController::rollout_warm_states_from_current() {
  // Header comment has the rationale: x_warm must be re-derived from the
  // MEASURED state under u_warm every step, not carried over as the
  // previous solve's (increasingly stale) prediction.
  using casadi::Slice;

  x_warm(Slice(), 0) = x;
  for (casadi_int stg = 0; stg < config.horizon_steps; ++stg) {
    const casadi::DM x_stage = x_warm(Slice(), stg);
    const casadi::DM u_stage = u_warm(Slice(), stg);
    const casadi::DM u_prev_stage =
        (stg == 0) ? u_prev : casadi::DM(u_warm(Slice(), stg - 1));
    const double kappa =
        track.curvature(static_cast<double>(x_stage(dynamics::S)));
    x_warm(Slice(), stg + 1) =
        linearizer.step(x_stage, u_stage, u_prev_stage, kappa);
  }
}

void LMPCController::shift_warm_start(const QpSolution &solution) {
  // Only the CONTROL trajectory shifts (receding horizon); the freed final
  // slot holds the last control constant. x_warm is deliberately NOT
  // shifted from solution.x_traj -- rollout_warm_states_from_current()
  // rebuilds it from the next measured state instead (header comment).
  using casadi::Slice;

  const casadi_int N = config.horizon_steps;
  if (N > 1) {
    u_warm(Slice(), Slice(0, N - 1)) = solution.u_traj(Slice(), Slice(1, N));
  }
  u_warm(Slice(), N - 1) = solution.u_traj(Slice(), N - 1);
  has_warm_start = true;
}

QpSolution LMPCController::solve_once() {
  // Every solve linearizes against a fresh nominal rollout from the
  // measured state under u_warm -- never against the previous solve's own
  // stale predicted states (rollout_warm_states_from_current()'s header
  // comment has the failure mode that motivated this).
  rollout_warm_states_from_current();

  // DESIGN.md SS8 step 3 (dummy-A/B/C pass: steps 3a/3b skipped, so
  // A_t = A^f_t, B_t = B^f_t, C_t = C^f_t -- no learned error correction).
  const casadi_int N = config.horizon_steps;
  std::vector<QpStage> stages;
  stages.reserve(static_cast<std::size_t>(N));
  for (casadi_int stg = 0; stg < N; ++stg) {
    casadi::DM x_ref = x_warm(casadi::Slice(), stg);
    const casadi::DM u_ref = u_warm(casadi::Slice(), stg);
    const casadi::DM u_prev_ref =
        (stg == 0) ? u_prev : casadi::DM(u_warm(casadi::Slice(), stg - 1));
    const double kappa_ref =
        track.curvature(static_cast<double>(x_ref(dynamics::S)));

    const LinearizedDynamics lin =
        linearizer(x_ref, u_ref, u_prev_ref, kappa_ref);
    stages.push_back(QpStage{lin.A, lin.B, lin.C});

    if (std::getenv("LMPC_DEBUG_STAGES") != nullptr) {
      std::cerr << "stage " << stg << ": x_ref=" << x_ref.T()
                << " u_ref=" << u_ref.T()
                << " |A|max=" << casadi::DM::mmax(fabs(lin.A))
                << " |B|max=" << casadi::DM::mmax(fabs(lin.B)) << std::endl;
    }
  }

  // DESIGN.md SS8 step 4: terminal safe-set query at x_bar_{k+N} -- a
  // DIFFERENT query than the per-stage ones above: terminal neighbors are
  // selected in normalized [vx, epsi, s, ey] space.
  const casadi::DM x_terminal_ref = x_warm(casadi::Slice(), N);
  SafeSet::QueryResult safe_set_result =
      safe_set.query(x_terminal_ref, config.K, safe_set_query_scale());
  const casadi_int expected_q = terminal_set_size();
  if (safe_set_result.X_ss.size1() != kStateDim ||
      safe_set_result.X_ss.size2() != expected_q ||
      safe_set_result.J_ss.size1() != expected_q ||
      safe_set_result.J_ss.size2() != 1) {
    throw std::runtime_error("LMPCController::solve_once: safe-set query "
                             "returned invalid dimensions");
  }
  // Finish-mode terminal set: within the last few horizons of a lap the
  // terminal reference runs past the end of the recorded data (s is
  // non-periodic, each stored lap ends where gym's finish detection fired).
  // The plain query can then only clamp onto each lap's final samples, whose
  // s sits BEHIND the reference, so the hard terminal equality can pull the
  // prediction backward after J has already bottomed out at ~0 and the QP's
  // optimum is to brake and park exactly on the data's endpoint instead of
  // driving through the line (the measured 3.7 -> 2.3 m/s braking at the seam).
  // Past the data's end the real target is the finish set {x : s >= L}: keep
  // the queried samples' dynamic states as the terminal anchor but free the s
  // row (match it to the reference itself, no backward pull) and zero the
  // cost-to-go -- a forward absorbing extension of the safe set. The lap
  // ends (lap-as-iteration: runs/lmpc_drive.py resets and starts iteration
  // j+1) before the horizon runs meaningfully past the line, so no data is
  // ever fabricated beyond that.
  if (static_cast<double>(x_terminal_ref(dynamics::S)) >
      safe_set.data_end_s()) {
    safe_set_result.X_ss(dynamics::S, casadi::Slice()) =
        x_terminal_ref(dynamics::S);
    safe_set_result.J_ss = casadi::DM::zeros(safe_set_result.J_ss.size1(), 1);
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
  return qp_builder->solve(x, u_prev_anchor, stages, safe_set_result.X_ss,
                           safe_set_result.J_ss, x_warm, u_warm, lambda_warm);
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
