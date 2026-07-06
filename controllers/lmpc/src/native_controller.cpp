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

#include <casadi/casadi.hpp>

#include "base_vehicle_model/base_vehicle_model_config.hpp"
#include "kinematic_bicycle_model/kinematic_bicycle_model.hpp"

namespace f110_gym_lmpc {
namespace {

namespace base = ::lmpc::vehicle_model::base_vehicle_model;
namespace kb = ::lmpc::vehicle_model::kinematic_bicycle_model;

constexpr double kLargeBound = 1.0e6;

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
        model_(make_base_config(config), make_kinematic_config(config)) {
    build_solver();
    reset();
  }

  void reset() {
    current_state_ = RacingLmpcState{};
    current_reference_ =
        LmpcReference{0.0, config_.target_speed, config_.track_half_width,
                      config_.track_half_width};
    previous_command_ = LmpcControlCommand{};
    last_u_ = casadi::DM::zeros(model_.nu(),
                                static_cast<casadi_int>(config_.horizon - 1));
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
    opti_.set_value(kappa_, DM::ones(1, N - 1) * current_reference_.curvature);
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
    opti_.set_initial(X_, solved_ ? last_x_ : x_init);
    opti_.set_initial(U_, solved_ ? last_u_ : DM::zeros(model_.nu(), N - 1));

    try {
      sol_ = std::make_shared<casadi::OptiSol>(opti_.solve_limited());
      last_x_ = sol_->value(X_);
      last_u_ = sol_->value(U_);
      solved_ = true;
      previous_command_ = command_from_solution(last_x_, last_u_);
    } catch (const std::exception &) {
      solved_ = false;
      previous_command_ = fallback_command();
    }
    return previous_command_;
  }

  const SparseErrorModel &error_model() const { return error_model_; }
  std::size_t sample_count() const { return 0; }

private:
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

    MX cost = MX::zeros(1);
    opti_.subject_to(X_(Slice(), 0) == x0_);

    for (casadi_int i = 0; i < N - 1; ++i) {
      const auto xi = X_(Slice(), i);
      const auto xip1 = X_(Slice(), i + 1);
      const auto ui = U_(Slice(), i);
      const auto kappa_i = kappa_(i);

      casadi::MXDict constraint_in = {
          {"x", xi},  {"u", ui},      {"xip1", xip1},
          {"t", dt_}, {"k", kappa_i}, {"track_length", kLargeBound},
      };
      // TODO(dynamics-model): kinematic_bicycle_model is the first Gym target.
      // Replace model_ construction plus bounds/config here when switching to
      // single_track_planar_model or another Racing-LMPC vehicle model.
      model_.add_nlp_constraints(opti_, constraint_in);

      cost += 3.0 * xi(kb::XIndex::PY) * xi(kb::XIndex::PY);
      cost += 1.5 * xi(kb::XIndex::YAW) * xi(kb::XIndex::YAW);
      const auto dv = xi(kb::XIndex::V) - target_speed_;
      cost += 0.5 * dv * dv;
      cost += 1.0e-3 * ui(kb::UIndex::FD) * ui(kb::UIndex::FD);
      cost += 1.0e-3 * ui(kb::UIndex::FB) * ui(kb::UIndex::FB);
      cost += 0.1 * ui(kb::UIndex::STEER) * ui(kb::UIndex::STEER);

      opti_.subject_to(xi(kb::XIndex::PY) >= -right_bound_);
      opti_.subject_to(xi(kb::XIndex::PY) <= left_bound_);
      opti_.subject_to(opti_.bounded(
          0.0, xi(kb::XIndex::V), std::max(config_.target_speed * 2.0, 5.0)));
    }

    const auto xN = X_(Slice(), N - 1);
    cost += 10.0 * xN(kb::XIndex::PY) * xN(kb::XIndex::PY);
    cost += 5.0 * xN(kb::XIndex::YAW) * xN(kb::XIndex::YAW);
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
    (void)x;
    const double velocity = clamp(current_reference_.target_speed, 0.0,
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

  LmpcConfig config_;
  kb::KinematicBicycleModel model_;
  casadi::Opti opti_;
  casadi::MX X_;
  casadi::MX U_;
  casadi::MX x0_;
  casadi::MX dt_;
  casadi::MX kappa_;
  casadi::MX target_speed_;
  casadi::MX left_bound_;
  casadi::MX right_bound_;
  casadi::DM last_x_;
  casadi::DM last_u_;
  std::shared_ptr<casadi::OptiSol> sol_;
  RacingLmpcState current_state_;
  LmpcReference current_reference_;
  LmpcControlCommand previous_command_;
  SparseErrorModel error_model_;
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

} // namespace f110_gym_lmpc
