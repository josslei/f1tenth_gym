// Copyright 2023 Haoru Xue
// Copyright 2026 Joss Lei
//
// This file adapts the Racing-LMPC-ROS2 CasADi MPC structure for a non-ROS,
// simulator-agnostic controller core. Racing-LMPC-ROS2 is licensed under the
// GNU Lesser General Public License v3 or later.

#include "lmpc/state.hpp"

#include <algorithm>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <vector>

#include <casadi/casadi.hpp>

#include "base_vehicle_model/base_vehicle_model_config.hpp"
#include "kinematic_bicycle_model/kinematic_bicycle_model.hpp"
#include "racing_trajectory/safe_set.hpp"

namespace f110_gym_lmpc {
namespace {

namespace base = ::lmpc::vehicle_model::base_vehicle_model;
namespace kb = ::lmpc::vehicle_model::kinematic_bicycle_model;
namespace rt = ::lmpc::vehicle_model::racing_trajectory;

constexpr double kTerminalSlackPenalty = 1.0e4;
constexpr double kVehicleMass = 3.47;

double clamp(double value, double low, double high) {
  return std::min(std::max(value, low), high);
}

base::BaseVehicleModelConfig::SharedPtr
make_base_config(const LmpcConfig &config) {
  auto front_tyre = std::make_shared<base::TyreConfig>();
  front_tyre->radius = 0.05;
  front_tyre->width = 0.03;
  front_tyre->mass = 0.1;
  front_tyre->moi = 1.0e-4;

  auto rear_tyre = std::make_shared<base::TyreConfig>(*front_tyre);

  auto front_brake = std::make_shared<base::BrakeConfig>();
  front_brake->max_brake = 1000.0;
  front_brake->brake_pad_out_r = 0.03;
  front_brake->brake_pad_in_r = 0.01;
  front_brake->brake_pad_friction_coeff = 0.4;
  front_brake->piston_area = 1.0e-4;
  front_brake->bias = 0.5;

  auto rear_brake = std::make_shared<base::BrakeConfig>(*front_brake);

  auto steer = std::make_shared<base::SteerConfig>();
  steer->max_steer_rate = 3.2;
  steer->max_steer = config.max_steer;
  steer->turn_left_bias = 0.0;

  auto chassis = std::make_shared<base::ChassisConfig>();
  chassis->total_mass = 3.47;
  chassis->sprung_mass = 3.47;
  chassis->unsprung_mass = 0.0;
  chassis->cg_ratio = 0.5;
  chassis->cg_height = 0.074;
  chassis->wheel_base = config.wheelbase;
  chassis->tw_f = 0.25;
  chassis->tw_r = 0.25;
  chassis->moi = 0.04712;
  chassis->b = 0.31;
  chassis->fr = 0.0;

  auto aero = std::make_shared<base::AeroConfig>();
  aero->air_density = 1.225;
  aero->drag_coeff = 0.0;
  aero->frontal_area = 0.1;
  aero->cl_f = 0.0;
  aero->cl_r = 0.0;

  auto powertrain = std::make_shared<base::PowerTrainConfig>();
  powertrain->gear_ratio = {1.0};
  powertrain->final_drive_ratio = 1.0;
  powertrain->kd = 0.5;
  powertrain->mechanical_efficiency = 1.0;

  auto modeling = std::make_shared<base::ModelingConfig>();
  modeling->use_frenet = true;
  modeling->integrator_type = base::IntegratorType::RK4;
  modeling->sample_throttle = 50.0;

  return std::make_shared<base::BaseVehicleModelConfig>(
      base::BaseVehicleModelConfig{front_tyre, rear_tyre, front_brake,
                                   rear_brake, steer, chassis, aero, powertrain,
                                   modeling});
}

kb::KinematicBicycleModelConfig::SharedPtr
make_kinematic_config(const LmpcConfig &config) {
  auto model_config = std::make_shared<kb::KinematicBicycleModelConfig>();
  model_config->Fd_max = config.max_drive_force;
  model_config->Fb_max = config.max_brake_force;
  model_config->Td = 0.1;
  model_config->Tb = 0.1;
  model_config->v_max = std::max(config.target_speed * 2.0, 5.0);
  model_config->P_max = 100.0;
  model_config->mu = 1.0;
  return model_config;
}

} // namespace

class NativeLMPCController::Impl {
public:
  explicit Impl(const LmpcConfig &config)
      : config_(config),
        model_(make_base_config(config), make_kinematic_config(config)),
        safe_set_(std::make_unique<rt::SafeSetManager>(config.max_lap_stored)) {
    build_solver();
    reset();
  }

  void reset() {
    safe_set_ = std::make_unique<rt::SafeSetManager>(config_.max_lap_stored);
    current_state_ = RacingLmpcState{};
    current_reference_ =
        LmpcReference{0.0, config_.target_speed, config_.track_half_width,
                      config_.track_half_width};
    previous_command_ = LmpcControlCommand{};
    previous_native_u_ = casadi::DM::zeros(model_.nu(), 1);
    last_u_ = casadi::DM::zeros(model_.nu(),
                                static_cast<casadi_int>(config_.horizon - 1));
    A_values_.clear();
    B_values_.clear();
    C_values_.clear();
    ss_x_value_ = casadi::DM();
    ss_costs_value_ = casadi::DM();
    lap_x_ = casadi::DM();
    lap_u_ = casadi::DM();
    lap_k_ = casadi::DM();
    lap_t_ = casadi::DM();
    lap_sample_count_ = 0;
    total_sample_count_ = 0;
    completed_laps_ = 0;
    last_safe_set_points_ = 0;
    elapsed_time_ = 0.0;
    last_recorded_s_ = 0.0;
    has_recorded_sample_ = false;
    error_model_ = SparseErrorModel{};
    solved_ = false;
  }

  void update(const RacingLmpcState &state) { current_state_ = state; }

  void set_reference(const LmpcReference &reference) {
    current_reference_ = reference;
  }

  LmpcControlCommand control() {
    using casadi::DM;
    using casadi::Slice;

    const auto N = static_cast<casadi_int>(config_.horizon);
    const DM x0 = DM{current_state_.s, current_state_.e_y, current_state_.e_psi,
                     std::max(current_state_.v_x, 0.05)};

    opti_.set_value(x0_, x0);
    opti_.set_value(dt_, config_.dt);
    opti_.set_value(kappa_, curvature_horizon(N - 1));
    opti_.set_value(target_speed_, current_reference_.target_speed);
    opti_.set_value(left_bound_, current_reference_.left_bound);
    opti_.set_value(right_bound_, current_reference_.right_bound);

    DM x_init = DM::zeros(model_.nx(), N);
    x_init(Slice(), 0) = x0;
    for (casadi_int i = 1; i < N; ++i) {
      x_init(kb::XIndex::PX, i) =
          x_init(kb::XIndex::PX, i - 1) + config_.dt * x0(kb::XIndex::V);
      x_init(kb::XIndex::PY, i) = x0(kb::XIndex::PY);
      x_init(kb::XIndex::YAW, i) = x0(kb::XIndex::YAW);
      x_init(kb::XIndex::V, i) = x0(kb::XIndex::V);
    }
    set_dynamics_parameters(x_init);
    set_safe_set_terminal(x_init(Slice(), N - 1));
    for (casadi_int i = 0; i < N - 1; ++i) {
      opti_.set_value(A_params_[i], A_values_[i]);
      opti_.set_value(B_params_[i], B_values_[i]);
      opti_.set_value(C_params_[i], C_values_[i]);
    }
    opti_.set_value(ss_x_, ss_x_value_);
    opti_.set_value(ss_costs_, ss_costs_value_);
    opti_.set_initial(X_, solved_ ? last_x_ : x_init);
    opti_.set_initial(U_, solved_ ? last_u_ : DM::zeros(model_.nu(), N - 1));

    try {
      sol_ = std::make_shared<casadi::OptiSol>(opti_.solve_limited());
      last_x_ = sol_->value(X_);
      last_u_ = sol_->value(U_);
      solved_ = true;
      previous_native_u_ = last_u_(Slice(), 0);
      previous_command_ = command_from_solution(last_x_, last_u_);
    } catch (const std::exception &) {
      solved_ = false;
      previous_command_ = fallback_command();
      previous_native_u_ = fallback_native_u(previous_command_);
    }
    record_current_sample();
    return previous_command_;
  }

  const SparseErrorModel &error_model() const { return error_model_; }
  std::size_t sample_count() const { return total_sample_count_; }
  std::size_t completed_laps() const { return completed_laps_; }
  std::size_t lap_sample_count() const { return lap_sample_count_; }
  std::size_t last_safe_set_points() const { return last_safe_set_points_; }

private:
  casadi::DM curvature_horizon(casadi_int horizon_steps) const {
    casadi::DM values =
        casadi::DM::ones(1, horizon_steps) * current_reference_.curvature;
    const auto sequence_size =
        static_cast<casadi_int>(current_reference_.curvature_sequence.size());
    const auto count = std::min(horizon_steps, sequence_size);
    for (casadi_int i = 0; i < count; ++i) {
      values(i) =
          current_reference_.curvature_sequence[static_cast<std::size_t>(i)];
    }
    return values;
  }

  double curvature_at(casadi_int horizon_index) const {
    const auto index = static_cast<std::size_t>(horizon_index);
    if (index < current_reference_.curvature_sequence.size()) {
      return current_reference_.curvature_sequence[index];
    }
    return current_reference_.curvature;
  }

  void build_solver() {
    using casadi::DM;
    using casadi::MX;
    using casadi::Slice;

    const auto N = static_cast<casadi_int>(config_.horizon);
    opti_ = casadi::Opti();
    X_ = opti_.variable(model_.nx(), N);
    U_ = opti_.variable(model_.nu(), N - 1);
    x0_ = opti_.parameter(model_.nx(), 1);
    dt_ = opti_.parameter(1, 1);
    kappa_ = opti_.parameter(1, N - 1);
    target_speed_ = opti_.parameter(1, 1);
    left_bound_ = opti_.parameter(1, 1);
    right_bound_ = opti_.parameter(1, 1);
    ss_x_ = opti_.parameter(model_.nx(), config_.reg_max_points);
    ss_costs_ = opti_.parameter(1, config_.reg_max_points);
    lambda_ = opti_.variable(config_.reg_max_points, 1);
    terminal_slack_ = opti_.variable(model_.nx(), 1);

    MX cost = MX::zeros(1);
    opti_.subject_to(X_(Slice(), 0) == x0_);

    for (casadi_int i = 0; i < N - 1; ++i) {
      A_params_.push_back(opti_.parameter(model_.nx(), model_.nx()));
      B_params_.push_back(opti_.parameter(model_.nx(), model_.nu()));
      C_params_.push_back(opti_.parameter(model_.nx(), 1));
      const auto xi = X_(Slice(), i);
      const auto xip1 = X_(Slice(), i + 1);
      const auto ui = U_(Slice(), i);
      // TODO(dynamics-model): kinematic_bicycle_model is the first Gym target.
      // Replace model_ construction plus bounds/config here when switching to
      // single_track_planar_model or another Racing-LMPC vehicle model.
      opti_.subject_to(xip1 == casadi::MX::mtimes({A_params_[i], xi}) +
                                   casadi::MX::mtimes({B_params_[i], ui}) +
                                   C_params_[i]);

      cost += config_.lateral_weight * xi(kb::XIndex::PY) * xi(kb::XIndex::PY);
      cost +=
          config_.heading_weight * xi(kb::XIndex::YAW) * xi(kb::XIndex::YAW);
      const auto dv = xi(kb::XIndex::V) - target_speed_;
      cost += config_.speed_weight * dv * dv;
      cost -=
          config_.progress_weight * (xip1(kb::XIndex::PX) - xi(kb::XIndex::PX));
      cost += 1.0e-3 * ui(kb::UIndex::FD) * ui(kb::UIndex::FD);
      cost += 1.0e-3 * ui(kb::UIndex::FB) * ui(kb::UIndex::FB);
      cost += 0.1 * ui(kb::UIndex::STEER) * ui(kb::UIndex::STEER);

      opti_.subject_to(
          opti_.bounded(0.0, ui(kb::UIndex::FD), config_.max_drive_force));
      opti_.subject_to(
          opti_.bounded(config_.max_brake_force, ui(kb::UIndex::FB), 0.0));
      opti_.subject_to(opti_.bounded(-config_.max_steer, ui(kb::UIndex::STEER),
                                     config_.max_steer));
      opti_.subject_to(xi(kb::XIndex::PY) >= -right_bound_);
      opti_.subject_to(xi(kb::XIndex::PY) <= left_bound_);
      opti_.subject_to(opti_.bounded(
          0.0, xi(kb::XIndex::V), std::max(config_.target_speed * 2.0, 5.0)));
    }

    const auto xN = X_(Slice(), N - 1);
    opti_.subject_to(lambda_ >= 0.0);
    opti_.subject_to(casadi::MX::sum1(lambda_) == 1.0);
    opti_.subject_to(xN ==
                     casadi::MX::mtimes({ss_x_, lambda_}) + terminal_slack_);
    cost +=
        config_.safe_set_cost_weight * casadi::MX::mtimes({ss_costs_, lambda_});
    cost += kTerminalSlackPenalty * casadi::MX::sumsqr(terminal_slack_);
    cost += config_.terminal_lateral_weight * xN(kb::XIndex::PY) *
            xN(kb::XIndex::PY);
    cost += config_.terminal_heading_weight * xN(kb::XIndex::YAW) *
            xN(kb::XIndex::YAW);
    opti_.minimize(cost);

    const auto p_opts = casadi::Dict{{"expand", true}, {"print_time", false}};
    const auto s_opts = casadi::Dict{{"conic", "qrqp"},
                                     {"max_iter", config_.max_iter},
                                     {"print_header", false},
                                     {"print_iteration", false},
                                     {"print_time", false},
                                     {"tol_du", config_.tolerance},
                                     {"tol_pr", config_.tolerance}};
    opti_.solver("sqpmethod", p_opts, s_opts);
  }

  LmpcControlCommand command_from_solution(const casadi::DM &x,
                                           const casadi::DM &u) const {
    const double steer = clamp(static_cast<double>(u(kb::UIndex::STEER, 0)),
                               -config_.max_steer, config_.max_steer);
    const double fd = static_cast<double>(u(kb::UIndex::FD, 0));
    const double fb = static_cast<double>(u(kb::UIndex::FB, 0));
    const double v = static_cast<double>(x(kb::XIndex::V, 0));
    const double acceleration = (fd + fb) / kVehicleMass;
    const double velocity = clamp(v + config_.dt * acceleration, 0.0,
                                  std::max(config_.target_speed * 2.0, 5.0));
    return LmpcControlCommand{steer, velocity};
  }

  LmpcControlCommand fallback_command() const {
    const double feedforward =
        std::atan(config_.wheelbase * current_reference_.curvature);
    const double steer = clamp(feedforward - 0.6 * current_state_.e_y -
                                   1.2 * current_state_.e_psi,
                               -config_.max_steer, config_.max_steer);
    const double velocity = clamp(current_reference_.target_speed, 0.0,
                                  std::max(config_.target_speed, 1.0));
    return LmpcControlCommand{steer, velocity};
  }

  casadi::DM fallback_native_u(const LmpcControlCommand &command) const {
    casadi::DM u = casadi::DM::zeros(model_.nu(), 1);
    u(kb::UIndex::STEER) = command.steering;
    return u;
  }

  casadi::DM current_state_dm() const {
    return casadi::DM{current_state_.s, current_state_.e_y,
                      current_state_.e_psi, std::max(current_state_.v_x, 0.05)};
  }

  void record_current_sample() {
    const casadi::DM x = current_state_dm();
    const casadi::DM u = previous_native_u_;
    const casadi::DM k = casadi::DM{current_reference_.curvature};
    const casadi::DM t = casadi::DM{elapsed_time_};
    const bool wrapped =
        has_recorded_sample_ &&
        last_recorded_s_ - current_state_.s > 0.5 * config_.track_length;

    if (wrapped && lap_sample_count_ > 1) {
      safe_set_->add_lap(lap_x_, lap_u_, lap_k_, lap_t_, config_.track_length);
      completed_laps_++;
      lap_x_ = x;
      lap_u_ = u;
      lap_k_ = k;
      lap_t_ = t;
      lap_sample_count_ = 1;
    } else if (lap_sample_count_ == 0) {
      lap_x_ = x;
      lap_u_ = u;
      lap_k_ = k;
      lap_t_ = t;
      lap_sample_count_ = 1;
    } else {
      lap_x_ = casadi::DM::horzcat({lap_x_, x});
      lap_u_ = casadi::DM::horzcat({lap_u_, u});
      lap_k_ = casadi::DM::horzcat({lap_k_, k});
      lap_t_ = casadi::DM::horzcat({lap_t_, t});
      lap_sample_count_++;
    }

    total_sample_count_++;
    elapsed_time_ += config_.dt;
    last_recorded_s_ = current_state_.s;
    has_recorded_sample_ = true;
  }

  void set_dynamics_parameters(const casadi::DM &x_init) {
    using casadi::Slice;

    A_values_.clear();
    B_values_.clear();
    C_values_.clear();
    A_values_.reserve(config_.horizon - 1);
    B_values_.reserve(config_.horizon - 1);
    C_values_.reserve(config_.horizon - 1);
    for (casadi_int i = 0; i < static_cast<casadi_int>(config_.horizon - 1);
         ++i) {
      casadi::DM dA = casadi::DM::zeros(model_.nx(), model_.nx());
      casadi::DM dB = casadi::DM::zeros(model_.nx(), model_.nu());
      casadi::DM dC = casadi::DM::zeros(model_.nx(), 1);
      const casadi::DM x_ref = x_init(Slice(), i);
      const casadi::DM u_ref =
          solved_ ? last_u_(Slice(), i) : previous_native_u_;
      casadi::DM A;
      casadi::DM B;
      casadi::DM C;
      const double curvature = curvature_at(i);
      compute_nominal_affine_model(x_ref, u_ref, curvature, A, B, C);
      if (completed_laps_ > 0 && should_update_regression(i)) {
        compute_regression_residual(x_ref, u_ref, A, B, C, dA, dB, dC);
      }
      A += dA;
      B += dB;
      C += dC;
      A_values_.push_back(A);
      B_values_.push_back(B);
      C_values_.push_back(C);
      if (i == 0) {
        store_error_model(A, B, C);
      }
    }
  }

  bool should_update_regression(casadi_int horizon_index) const {
    return horizon_index == 0 ||
           (config_.regression_horizon_stride > 0 &&
            horizon_index % static_cast<casadi_int>(
                                config_.regression_horizon_stride) ==
                0);
  }

  void compute_nominal_affine_model(const casadi::DM &x, const casadi::DM &u,
                                    double curvature, casadi::DM &reg_a,
                                    casadi::DM &reg_b, casadi::DM &reg_c) {
    const auto jac = model_.discrete_dynamics_jacobian()(casadi::DMDict{
        {"x", x},
        {"u", u},
        {"k", casadi::DM{curvature}},
        {"dt", casadi::DM{config_.dt}},
    });
    reg_a = jac.at("A");
    reg_b = jac.at("B");
    const auto xip1 = model_
                          .discrete_dynamics()(casadi::DMDict{
                              {"x", x},
                              {"u", u},
                              {"k", casadi::DM{curvature}},
                              {"dt", casadi::DM{config_.dt}},
                          })
                          .at("xip1");
    reg_c = xip1 - casadi::DM::mtimes(reg_a, x) - casadi::DM::mtimes(reg_b, u);
  }

  void compute_regression_residual(const casadi::DM &x, const casadi::DM &u,
                                   const casadi::DM &nominal_a,
                                   const casadi::DM &nominal_b,
                                   const casadi::DM &nominal_c, casadi::DM &dA,
                                   casadi::DM &dB, casadi::DM &dC) {
    const rt::RegQuery query{
        casadi::DM::vertcat({x(kb::XIndex::V), u}),
        nominal_a,
        nominal_b,
        nominal_c,
        model_.discrete_dynamics(),
        config_.reg_dist_max,
        static_cast<casadi_int>(config_.reg_max_points),
        static_cast<casadi_int>(config_.reg_max_points_per_lap),
        rt::RegQuery::Indices{{kb::XIndex::V}},
        rt::RegQuery::Indices{
            {kb::UIndex::FD, kb::UIndex::FB, kb::UIndex::STEER}},
        rt::RegQuery::Indices{{kb::XIndex::V}},
    };
    const auto result = safe_set_->query(query);
    dA = result.A - nominal_a;
    dB = result.B - nominal_b;
    dC = result.C - nominal_c;
  }

  void store_error_model(const casadi::DM &reg_a, const casadi::DM &reg_b,
                         const casadi::DM &reg_c) {
    for (casadi_int row = 0; row < static_cast<casadi_int>(model_.nx());
         ++row) {
      for (casadi_int col = 0; col < static_cast<casadi_int>(model_.nx());
           ++col) {
        error_model_.A[row][col] = static_cast<double>(reg_a(row, col));
      }
      for (casadi_int col = 0; col < static_cast<casadi_int>(model_.nu());
           ++col) {
        error_model_.B[row][col] = static_cast<double>(reg_b(row, col));
      }
      error_model_.C[row] = static_cast<double>(reg_c(row));
    }
  }

  void set_safe_set_terminal(const casadi::DM &terminal_state) {
    using casadi::Slice;

    const auto num_points = static_cast<casadi_int>(config_.reg_max_points);
    casadi::DM ss_x = casadi::DM::repmat(terminal_state, 1, num_points);
    casadi::DM ss_costs = casadi::DM::zeros(1, num_points);

    if (completed_laps_ > 0) {
      const rt::SSQuery query{
          terminal_state,
          config_.reg_dist_max,
          static_cast<casadi_int>(config_.reg_max_points),
          static_cast<casadi_int>(config_.reg_max_points_per_lap),
      };
      const auto result = safe_set_->query(query);
      last_safe_set_points_ = static_cast<std::size_t>(result.x.size2());
      if (result.x.size2() > 0) {
        ss_x = result.x;
        ss_costs = result.J - result.J(0);
        if (ss_x.size2() < num_points) {
          const auto pad_count = num_points - ss_x.size2();
          ss_x = casadi::DM::horzcat(
              {ss_x, casadi::DM::repmat(ss_x(Slice(), -1), 1, pad_count)});
          ss_costs = casadi::DM::horzcat(
              {ss_costs,
               casadi::DM::repmat(ss_costs(Slice(), -1), 1, pad_count)});
        } else if (ss_x.size2() > num_points) {
          ss_x = ss_x(Slice(), Slice(0, num_points));
          ss_costs = ss_costs(Slice(), Slice(0, num_points));
        }
      }
    }

    ss_x_value_ = ss_x;
    ss_costs_value_ = ss_costs;
  }

  LmpcConfig config_;
  kb::KinematicBicycleModel model_;
  std::unique_ptr<rt::SafeSetManager> safe_set_;
  casadi::Opti opti_;
  casadi::MX X_;
  casadi::MX U_;
  casadi::MX x0_;
  casadi::MX dt_;
  casadi::MX kappa_;
  casadi::MX target_speed_;
  casadi::MX left_bound_;
  casadi::MX right_bound_;
  std::vector<casadi::MX> A_params_;
  std::vector<casadi::MX> B_params_;
  std::vector<casadi::MX> C_params_;
  casadi::MX ss_x_;
  casadi::MX ss_costs_;
  casadi::MX lambda_;
  casadi::MX terminal_slack_;
  casadi::DM last_x_;
  casadi::DM last_u_;
  casadi::DM previous_native_u_;
  std::vector<casadi::DM> A_values_;
  std::vector<casadi::DM> B_values_;
  std::vector<casadi::DM> C_values_;
  casadi::DM ss_x_value_;
  casadi::DM ss_costs_value_;
  casadi::DM lap_x_;
  casadi::DM lap_u_;
  casadi::DM lap_k_;
  casadi::DM lap_t_;
  std::shared_ptr<casadi::OptiSol> sol_;
  RacingLmpcState current_state_;
  LmpcReference current_reference_;
  LmpcControlCommand previous_command_;
  SparseErrorModel error_model_;
  std::size_t lap_sample_count_ = 0;
  std::size_t total_sample_count_ = 0;
  std::size_t completed_laps_ = 0;
  std::size_t last_safe_set_points_ = 0;
  double elapsed_time_ = 0.0;
  double last_recorded_s_ = 0.0;
  bool has_recorded_sample_ = false;
  bool solved_ = false;
};

NativeLMPCController::NativeLMPCController()
    : impl_(std::make_unique<Impl>(LmpcConfig{})) {}

NativeLMPCController::NativeLMPCController(const LmpcConfig &config)
    : impl_(std::make_unique<Impl>(config)) {}

NativeLMPCController::~NativeLMPCController() = default;

void NativeLMPCController::reset() { impl_->reset(); }

void NativeLMPCController::update(const RacingLmpcState &state) {
  impl_->update(state);
}

void NativeLMPCController::set_reference(const LmpcReference &reference) {
  impl_->set_reference(reference);
}

LmpcControlCommand NativeLMPCController::control() { return impl_->control(); }

const SparseErrorModel &NativeLMPCController::error_model() const {
  return impl_->error_model();
}

std::size_t NativeLMPCController::sample_count() const {
  return impl_->sample_count();
}

std::size_t NativeLMPCController::completed_laps() const {
  return impl_->completed_laps();
}

std::size_t NativeLMPCController::lap_sample_count() const {
  return impl_->lap_sample_count();
}

std::size_t NativeLMPCController::last_safe_set_points() const {
  return impl_->last_safe_set_points();
}

} // namespace f110_gym_lmpc
