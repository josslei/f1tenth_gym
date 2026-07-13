#include "lmpc_controller.hpp"

#include <cmath>
#include <stdexcept>

namespace lmpc
{

LMPCController::LMPCController(const LmpcConfig & config_in)
: config(config_in),
  dynamics_model(config.vehicle_params),
  integrator(),
  linearizer(dynamics_model, integrator, config.dt),
  track(config.centerline_csv_path),
  safe_set(config.seed_lap_csv_path),
  qp_builder(
    config.horizon_steps, config.K * safe_set.num_laps(),
    QpBounds{config.a_min, config.a_max, config.delta_min, config.delta_max, config.ey_max},
    QpWeights{config.c_u, config.c_du}, config.solver_name),
  x(casadi::DM::zeros(kStateDim, 1)),
  t(0.0),
  has_state(false),
  u_prev(casadi::DM::zeros(kControlDim, 1)),
  x_warm(casadi::DM::zeros(kStateDim, config.horizon_steps + 1)),
  u_warm(casadi::DM::zeros(kControlDim, config.horizon_steps)),
  has_warm_start(false)
{
}

void LMPCController::reset()
{
  x = casadi::DM::zeros(kStateDim, 1);
  t = 0.0;
  has_state = false;
  u_prev = casadi::DM::zeros(kControlDim, 1);
  x_warm = casadi::DM::zeros(kStateDim, config.horizon_steps + 1);
  u_warm = casadi::DM::zeros(kControlDim, config.horizon_steps);
  has_warm_start = false;
}

void LMPCController::update(const casadi::DM & x_in, double t_in)
{
  if (x_in.size1() != kStateDim || x_in.size2() != 1) {
    throw std::invalid_argument(
            "LMPCController::update: x must be a 6x1 vector [vx, vy, omega, epsi, s, ey]");
  }
  x = x_in;
  t = t_in;
  has_state = true;
}

void LMPCController::seed_warm_start()
{
  // Naive rollout under the nominal model, holding u constant at u_prev --
  // DESIGN.md SS8 step 2's fallback for "the very first solve" (no prior
  // trajectory exists yet to shift).
  using casadi::Slice;

  x_warm(Slice(), 0) = x;
  for (casadi_int stg = 0; stg < config.horizon_steps; ++stg) {
    u_warm(Slice(), stg) = u_prev;
    const casadi::DM x_stage = x_warm(Slice(), stg);
    const double kappa = track.curvature(static_cast<double>(x_stage(dynamics::S)));
    x_warm(Slice(), stg + 1) = linearizer.step(x_stage, u_prev, u_prev, kappa);
  }
  has_warm_start = true;
}

void LMPCController::shift_warm_start(const QpSolution & solution)
{
  // The just-solved trajectory shifted by one stage is next step's
  // linearization sequence (receding horizon, DESIGN.md SS8 step 2). The
  // freed final slot is filled by holding the last state/control constant
  // rather than re-rolling out the model -- a cheap approximation that only
  // affects the very last horizon stage's linearization point.
  using casadi::Slice;

  const casadi_int N = config.horizon_steps;
  x_warm(Slice(), Slice(0, N)) = solution.x_traj(Slice(), Slice(1, N + 1));
  x_warm(Slice(), N) = solution.x_traj(Slice(), N);
  if (N > 1) {
    u_warm(Slice(), Slice(0, N - 1)) = solution.u_traj(Slice(), Slice(1, N));
  }
  u_warm(Slice(), N - 1) = solution.u_traj(Slice(), N - 1);
  has_warm_start = true;
}

casadi::DM LMPCController::control()
{
  if (!has_state) {
    throw std::logic_error("LMPCController::control: update() must be called before control()");
  }
  if (!has_warm_start) {
    seed_warm_start();
  }

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
    const double kappa_ref = track.curvature(static_cast<double>(x_ref(dynamics::S)));

    // lmpc_config.hpp's linearization_speed_floor: avoids GymDynamics's
    // atan2(vy, vx) Jacobian singularity at rest. Only this local reference
    // copy is floored -- x_warm itself (and the QP's true x0) keep the real
    // value.
    if (std::abs(static_cast<double>(x_ref(dynamics::VX))) < config.linearization_speed_floor) {
      x_ref(dynamics::VX) = config.linearization_speed_floor;
    }

    const LinearizedDynamics lin = linearizer(x_ref, u_ref, u_prev_ref, kappa_ref);
    stages.push_back(QpStage{lin.A, lin.B, lin.C});
  }

  // DESIGN.md SS8 step 4: terminal safe-set query at x_bar_{k+N} -- a
  // DIFFERENT query than the per-stage ones above (position-only, DESIGN.md
  // SS2's D/K/P, not the regression neighbor search).
  const casadi::DM x_terminal_ref = x_warm(casadi::Slice(), N);
  const SafeSet::QueryResult safe_set_result = safe_set.query(x_terminal_ref, config.K);

  // DESIGN.md SS8 step 5.
  const QpSolution solution = qp_builder.solve(
    x, u_prev, stages, safe_set_result.X_ss, safe_set_result.J_ss, x_warm, u_warm);
  if (!solution.success) {
    throw std::runtime_error("LMPCController::control: QP solve failed: " + solution.message);
  }

  shift_warm_start(solution);

  // DESIGN.md SS8 step 6: apply u_0*.
  u_prev = solution.u_traj(casadi::Slice(), 0);
  return u_prev;
}

}  // namespace lmpc
