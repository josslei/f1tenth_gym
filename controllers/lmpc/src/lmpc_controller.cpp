#include "lmpc_controller.hpp"

#include <cmath>
#include <map>
#include <utility>

// The copied file is deliberately included, rather than modified, so this
// translation unit can adapt its LMPCCore class to lmpc_controller.hpp.
#include "lmpc_core.cpp"

namespace lmpc {

namespace {

std::map<std::string, double> make_params(const LmpcConfig &config) {
  std::map<std::string, double> params = {
      {"N", static_cast<double>(config.horizon_steps)},
      {"Ts", config.dt},
      {"K_NEAR", static_cast<double>(config.K)},
      {"ACCELERATION_MAX", config.a_max},
      {"DECELERATION_MAX", -config.a_min},
      {"SPEED_MAX", config.v_max},
      {"STEER_MAX", config.delta_max},
      {"VEL_THRESHOLD", config.velocity_threshold},
      {"WAYPOINT_SPACE", config.waypoint_space},
      {"r_accel", config.r_accel},
      {"r_steer", config.r_steer},
      {"r_d_accel", config.r_d_accel},
      {"r_d_steer", config.r_d_steer},
      {"q_s", config.ey_slack_l2},
      {"q_s_terminal", config.terminal_slack_weight},
      {"MAP_MARGIN", config.map_margin},
      {"osqp_max_iter", static_cast<double>(config.osqp_max_iter)},
      {"osqp_scaling", static_cast<double>(config.osqp_scaling)},
      {"osqp_eps_prim_inf", config.osqp_eps_prim_inf},
      {"osqp_eps_abs", config.osqp_eps_abs},
      {"osqp_eps_rel", config.osqp_eps_rel},
      {"wheelbase", config.vehicle_params.lf + config.vehicle_params.lr},
      {"friction_coeff", config.vehicle_params.mu},
      {"height_cg", config.vehicle_params.h},
      {"l_cg2rear", config.vehicle_params.lr},
      {"l_cg2front", config.vehicle_params.lf},
      {"C_S_front", config.vehicle_params.C_Sf},
      {"C_S_rear", config.vehicle_params.C_Sr},
      {"mass", config.vehicle_params.m},
      {"moment_inertia", config.vehicle_params.I},
      {"regression_enabled", config.regression_enabled ? 1.0 : 0.0},
      {"regression_num_neighbors",
       static_cast<double>(config.regression_num_neighbors)},
      {"regression_bandwidth", config.regression_bandwidth},
      {"regression_regularization", config.regression_regularization},
  };
  // regression_Q is 8x8 (64 entries); the scalar param map has no matrix
  // slot, so it's flattened row-major -- see LMPCCore::getParameters.
  for (std::size_t i = 0; i < config.regression_Q.size() && i < 64; ++i) {
    params["regression_Q_" + std::to_string(i)] = config.regression_Q[i];
  }
  return params;
}

} // namespace

struct LMPCController::Impl {
  explicit Impl(LmpcConfig config_in)
      : config(std::move(config_in)),
        prediction(casadi::DM::zeros(6, config.horizon_steps + 1)),
        terminal_slack(casadi::DM::zeros(6, 1)) {
    rebuild();
  }

  void rebuild() {
    py::array_t<std::int8_t> grid(config.occupancy_grid.size(),
                                  config.occupancy_grid.data());
    core = std::make_unique<LMPCCore>(
        make_params(config), grid, config.map_width, config.map_height,
        config.map_resolution, config.map_origin_x, config.map_origin_y,
        config.reference_waypoint_csv_path, config.reference_seed_lap_csv_path,
        config.initial_x, config.initial_y, config.initial_yaw);
    prediction = casadi::DM::zeros(6, config.horizon_steps + 1);
    terminal_slack = casadi::DM::zeros(6, 1);
    solved = true;
    timings = {};
  }

  LmpcConfig config;
  std::unique_ptr<LMPCCore> core;
  casadi::DM prediction;
  casadi::DM terminal_slack;
  ControllerTimings timings;
  bool solved = true;
};

LMPCController::LMPCController(const LmpcConfig &config)
    : impl(std::make_unique<Impl>(config)) {}

LMPCController::~LMPCController() = default;
LMPCController::LMPCController(LMPCController &&) noexcept = default;
LMPCController &LMPCController::operator=(LMPCController &&) noexcept = default;

void LMPCController::reset() { impl->rebuild(); }

void LMPCController::update(const casadi::DM &x, double t,
                            double actual_delta) {
  (void)t;
  (void)actual_delta;
  const double speed = static_cast<double>(x(3));
  const double slip = static_cast<double>(x(5));
  impl->core->set_state(static_cast<double>(x(0)), static_cast<double>(x(1)),
                        static_cast<double>(x(2)), speed * std::cos(slip),
                        speed * std::sin(slip), static_cast<double>(x(4)));
}

casadi::DM LMPCController::control() {
  const py::tuple result = impl->core->step();
  impl->timings.rollout_lin_ms = impl->core->last_rollout_lin_ms();
  impl->timings.knn_ms = impl->core->last_knn_ms();
  impl->timings.regression_ms = impl->core->last_regression_ms();
  impl->timings.set_params_ms = impl->core->last_set_params_ms();
  impl->timings.solver_ms = impl->core->last_solver_ms();
  impl->timings.postcheck_ms = impl->core->last_postcheck_ms();
  impl->solved = result[2].cast<bool>();

  const Eigen::Matrix<double, 6, 1> slack = impl->core->last_terminal_slack();
  for (int i = 0; i < 6; ++i) {
    impl->terminal_slack(i) = slack(i);
  }

  const py::array_t<double> predicted = impl->core->predicted_states();
  const auto view = predicted.unchecked<2>();
  for (casadi_int stage = 0; stage < impl->prediction.size2(); ++stage) {
    for (casadi_int state = 0; state < impl->prediction.size1(); ++state) {
      impl->prediction(state, stage) = view(stage, state);
    }
  }
  return casadi::DM({result[0].cast<double>(), result[1].cast<double>()});
}

void LMPCController::add_lap(const casadi::DM &x_lap, const casadi::DM &u_lap,
                             const casadi::DM &J_lap) {
  (void)x_lap;
  (void)u_lap;
  (void)J_lap;
}

casadi::DM LMPCController::predicted_next_state() const {
  return impl->prediction(casadi::Slice(), 1);
}

casadi::DM LMPCController::predicted_trajectory() const {
  return impl->prediction;
}

const ControllerTimings &LMPCController::last_timings() const {
  return impl->timings;
}

const casadi::DM &LMPCController::last_terminal_slack_value() const {
  return impl->terminal_slack;
}

bool LMPCController::last_solve_ok() const { return impl->solved; }

bool LMPCController::using_dynamic_model() const {
  return impl->core->use_dyn();
}

int LMPCController::regression_pool_size() const {
  return impl->core->regression_pool_size();
}

double LMPCController::last_regression_correction_norm() const {
  return impl->core->last_regression_correction_norm();
}

} // namespace lmpc
