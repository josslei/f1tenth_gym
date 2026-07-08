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
#include "racing_trajectory/safe_set.hpp"
#include "single_track_planar_model/single_track_planar_model.hpp"

namespace f110_gym_lmpc {
namespace {

namespace base = ::lmpc::vehicle_model::base_vehicle_model;
namespace rt = ::lmpc::vehicle_model::racing_trajectory;
namespace st = ::lmpc::vehicle_model::single_track_planar_model;

constexpr double kTerminalSlackPenalty = 1.0e4;
constexpr double kVehicleMass = 3.47;
// Tiny ridge on the safe-set multipliers. The terminal constraint
// xN == ss_x*lambda + slack is rank-deficient whenever safe-set columns repeat
// (always, before a seed lap exists: ss_x is one point repmat'd). Without this,
// qrqp's active set cannot pin down the under-determined lambda and grinds to
// the iteration cap every solve. The ridge selects the minimum-norm lambda,
// which does not change the physical solution (xN/X/U) when ss_x columns are
// identical, and only mildly biases lambda toward uniform once they differ.
constexpr double kLambdaRidge = 1.0e-3;

double clamp(double value, double low, double high) {
  return std::min(std::max(value, low), high);
}

base::BaseVehicleModelConfig::SharedPtr
make_base_config(const LmpcConfig &config) {
  auto front_tyre = std::make_shared<base::TyreConfig>();
  front_tyre->radius = 0.05;
  front_tyre->width = 0.04;
  front_tyre->mass = 0.05;
  front_tyre->moi = 0.05;
  front_tyre->pacejka_b = 5.0;
  front_tyre->pacejka_c = 2.28;
  front_tyre->pacejka_e = 1.0;
  front_tyre->pacejka_fz0 = 1543.5;
  front_tyre->pacejka_eps = -0.0813;

  auto rear_tyre = std::make_shared<base::TyreConfig>(*front_tyre);
  rear_tyre->pacejka_fz0 = 1886.5;
  rear_tyre->pacejka_eps = -0.1263;

  auto front_brake = std::make_shared<base::BrakeConfig>();
  front_brake->max_brake = 1000.0;
  front_brake->brake_pad_out_r = 0.03;
  front_brake->brake_pad_in_r = 0.01;
  front_brake->brake_pad_friction_coeff = 0.5;
  front_brake->piston_area = 0.00389508401;
  front_brake->bias = 0.5;

  auto rear_brake = std::make_shared<base::BrakeConfig>(*front_brake);
  rear_brake->piston_area = 0.0050543317;

  auto steer = std::make_shared<base::SteerConfig>();
  steer->max_steer_rate = 10.0;
  steer->max_steer = config.max_steer;
  steer->turn_left_bias = 0.0;

  auto chassis = std::make_shared<base::ChassisConfig>();
  chassis->total_mass = 3.47;
  chassis->sprung_mass = 3.47;
  chassis->unsprung_mass = 0.0;
  chassis->cg_ratio = 0.5;
  chassis->cg_height = 0.07;
  chassis->wheel_base = config.wheelbase;
  chassis->tw_f = 0.281;
  chassis->tw_r = 0.281;
  chassis->moi = 0.04712;
  chassis->b = 0.281;
  chassis->fr = 0.012;

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

st::SingleTrackPlanarModelConfig::SharedPtr
make_single_track_config(const LmpcConfig &config) {
  auto model_config = std::make_shared<st::SingleTrackPlanarModelConfig>();
  model_config->Fd_max = config.max_drive_force;
  model_config->Fb_max = config.max_brake_force;
  model_config->Td = 0.1;
  model_config->Tb = 0.1;
  model_config->v_max = std::max(config.target_speed * 2.0, 5.0);
  model_config->P_max = 100.0;
  model_config->mu = 1.0;
  model_config->simplify_lon_control = true;
  return model_config;
}

} // namespace

class NativeLMPCController::Impl {
public:
  explicit Impl(const LmpcConfig &config)
      : config_(config),
        model_(make_base_config(config), make_single_track_config(config)),
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
    last_horizon_.clear();
    lap_x_ = casadi::DM();
    lap_u_ = casadi::DM();
    lap_k_ = casadi::DM();
    lap_t_ = casadi::DM();
    lap_sample_count_ = 0;
    total_sample_count_ = 0;
    completed_laps_ = 0;
    driven_laps_ = 0;
    last_safe_set_points_ = 0;
    solver_attempt_count_ = 0;
    solver_success_count_ = 0;
    last_solver_status_.clear();
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

  // Seed the safe set with a pre-recorded lap (the paper's D^0). x_rows are M
  // states [s, e_y, e_psi, v_x, v_y, omega_z]; u_rows are M controls
  // [longitudinal, delta]; k and t are per-step curvature and timestamp.
  // Bumping completed_laps_ turns on the terminal cost-to-go, so LMPC has a
  // safe set to drive toward from the first control step.
  void add_initial_lap(const std::vector<std::vector<double>> &x_rows,
                       const std::vector<std::vector<double>> &u_rows,
                       const std::vector<double> &k,
                       const std::vector<double> &t) {
    const auto M = static_cast<casadi_int>(x_rows.size());
    casadi::DM x(model_.nx(), M);
    casadi::DM u(model_.nu(), M);
    casadi::DM k_dm(1, M);
    casadi::DM t_dm(1, M);
    for (casadi_int j = 0; j < M; ++j) {
      for (casadi_int i = 0; i < model_.nx(); ++i) {
        x(i, j) = x_rows[j][i];
      }
      for (casadi_int i = 0; i < model_.nu(); ++i) {
        u(i, j) = u_rows[j][i];
      }
      k_dm(0, j) = k[j];
      t_dm(0, j) = t[j];
    }
    safe_set_->add_lap(x, u, k_dm, t_dm, config_.track_length);
    completed_laps_++;
  }

  void set_curvature_profile(const std::vector<double> &s,
                             const std::vector<double> &k,
                             double total_length) {
    curv_s_ = s;
    curv_k_ = k;
    curv_total_length_ = total_length;
    // Append a wrap-around closing sample so interpolation is continuous across
    // the start/finish seam.
    if (!curv_s_.empty() && total_length > curv_s_.back()) {
      curv_s_.push_back(total_length);
      curv_k_.push_back(curv_k_.front());
    }
  }

  LmpcControlCommand control() {
    using casadi::DM;
    using casadi::Slice;

    const auto N = static_cast<casadi_int>(config_.horizon);
    const DM x0 = DM{current_state_.s,     current_state_.e_y,
                     current_state_.e_psi, std::max(current_state_.v_x, 0.05),
                     current_state_.v_y,   current_state_.omega};

    DM x_init = DM::zeros(model_.nx(), N);
    x_init(Slice(), 0) = x0;
    for (casadi_int i = 1; i < N; ++i) {
      x_init(st::XIndex::PX, i) =
          x_init(st::XIndex::PX, i - 1) + config_.dt * x0(st::XIndex::VX);
      x_init(st::XIndex::PY, i) = x0(st::XIndex::PY);
      x_init(st::XIndex::YAW, i) = x0(st::XIndex::YAW);
      x_init(st::XIndex::VX, i) = x0(st::XIndex::VX);
      x_init(st::XIndex::VY, i) = x0(st::XIndex::VY);
      x_init(st::XIndex::VYAW, i) = x0(st::XIndex::VYAW);
    }
    // Reference z_bar for the ATV linearization and the safe-set query. Per the
    // paper (eq. 4) z_bar is the previous FHOCP solution. We apply the standard
    // receding-horizon shift z_bar_t = x*_{t+1} (repeating the terminal) so the
    // reference and warm start stay aligned with the current horizon: the
    // unshifted solution lags one step every solve, drifts off the vehicle, and
    // makes the plan diverge (and the drawn horizon scatter) after a few
    // hundred steps. The first solve falls back to the current-speed
    // roll-forward.
    DM ref;
    DM u_ws;
    if (solved_) {
      // Fill the shifted terminal by rolling the old terminal forward through
      // the discrete dynamics (matching upstream racing_mpc_node.cpp:
      // last_x_(-1) = f(last_x_(-2), last_u_(-1))). Duplicating the old
      // terminal instead leaves a stale zero-progress last column that the
      // solver must re-fix every step, adding terminal jitter to the plan.
      const DM rolled_terminal =
          model_
              .discrete_dynamics()(casadi::DMDict{
                  {"x", last_x_(Slice(), N - 1)},
                  {"u", last_u_(Slice(), N - 2)},
                  {"k", casadi::DM{curvature_at_s(static_cast<double>(
                            last_x_(st::XIndex::PX, N - 1)))}},
                  {"dt", casadi::DM{config_.dt}}})
              .at("xip1");
      ref = DM::horzcat({last_x_(Slice(), Slice(1, N)), rolled_terminal});
      u_ws = DM::horzcat(
          {last_u_(Slice(), Slice(1, N - 1)), last_u_(Slice(), N - 2)});
    } else {
      ref = x_init;
      u_ws = DM::zeros(model_.nu(), N - 1);
    }
    // Anchor the reference start at the measured state so the t=0 linearization
    // is about where the vehicle actually is.
    ref(Slice(), 0) = x0;
    store_predicted_horizon(ref);

    // Everything that can touch the Opti object -- including parameter
    // preparation, not just solve_limited() -- must be inside this try. A
    // genuinely degenerate solve (e.g. "Failed to calculate search direction")
    // can leave last_x_/last_u_ or a regression result non-finite; the NEXT
    // call's opti_.set_value then throws CasADi's "v.is_regular()" assertion,
    // which used to escape uncaught (set_value was outside the try) and kill
    // the process. Widening the try costs nothing on the non-throwing path --
    // C++ exception handling here is zero-cost until something actually
    // throws -- so this is a pure robustness fix, not a perf tradeoff.
    try {
      set_dynamics_parameters(ref, u_ws);
      set_safe_set_terminal(ref(Slice(), N - 1));
      opti_.set_value(x0_, x0);
      opti_.set_value(dt_, config_.dt);
      opti_.set_value(kappa_, curvature_horizon(N - 1));
      opti_.set_value(u_prev_param_, previous_native_u_);
      opti_.set_value(left_bound_, current_reference_.left_bound);
      opti_.set_value(right_bound_, current_reference_.right_bound);
      for (casadi_int i = 0; i < N - 1; ++i) {
        opti_.set_value(A_params_[i], A_values_[i]);
        opti_.set_value(B_params_[i], B_values_[i]);
        opti_.set_value(C_params_[i], C_values_[i]);
      }
      opti_.set_value(ss_x_, ss_x_value_);
      opti_.set_value(ss_costs_, ss_costs_value_);
      opti_.set_initial(X_, ref);
      opti_.set_initial(U_, u_ws);

      sol_ = std::make_shared<casadi::OptiSol>(opti_.solve_limited());
      last_solver_status_ = opti_.return_status();
      const casadi::DM sol_x = sol_->value(X_);
      const casadi::DM sol_u = sol_->value(U_);
      // qrqp can report "success" yet return a garbage iterate on the
      // ill-conditioned dynamic-model QP: either non-finite (NaN/Inf) OR
      // exploding-but-finite (e_y seen reaching 5e10 while the corridor is
      // +-1.5 m -- it VIOLATES its own constraints). Both must be rejected: a
      // bad solution reaching the sim corrupts the vehicle state, and storing
      // it as last_x_ poisons the next warm start so the plan (and its drawn
      // horizon) explodes and "flies" every step. Reject -> fall back, leaving
      // last_x_/solved_ untouched. (Root cause is unscaled QP conditioning;
      // the permanent fix is variable scaling -- task #7.)
      const double kMaxLateral = 2.0 * std::max(current_reference_.left_bound,
                                                current_reference_.right_bound);
      const double kMaxSpeed = 3.0 * std::max(config_.target_speed * 2.0, 5.0);
      const double max_abs_ey = static_cast<double>(
          casadi::DM::mmax(casadi::DM::abs(sol_x(st::XIndex::PY, Slice()))));
      const double max_vx = static_cast<double>(
          casadi::DM::mmax(casadi::DM::abs(sol_x(st::XIndex::VX, Slice()))));
      if (!sol_x.is_regular() || !sol_u.is_regular()) {
        throw std::runtime_error("non-finite solution (" + last_solver_status_ +
                                 ")");
      }
      if (max_abs_ey > kMaxLateral || max_vx > kMaxSpeed) {
        throw std::runtime_error("solution violates physical bounds");
      }
      last_x_ = sol_x;
      last_u_ = sol_u;
      store_predicted_horizon(last_x_);
      solved_ = true;
      previous_command_ = command_from_solution(last_x_, last_u_);
      previous_native_u_ = command_native_u(previous_command_);
      solver_success_count_++;
    } catch (const std::exception &e) {
      solved_ = false;
      last_solver_status_ = e.what();
      previous_command_ = fallback_command();
      previous_native_u_ = command_native_u(previous_command_);
    }
    solver_attempt_count_++;
    record_current_sample();
    return previous_command_;
  }

  const std::vector<std::array<double, 2>> &predicted_horizon() const {
    return last_horizon_;
  }

  const SparseErrorModel &error_model() const { return error_model_; }
  std::size_t sample_count() const { return total_sample_count_; }
  std::size_t completed_laps() const { return completed_laps_; }
  std::size_t lap_sample_count() const { return lap_sample_count_; }
  std::size_t last_safe_set_points() const { return last_safe_set_points_; }
  const std::string &last_solver_status() const { return last_solver_status_; }
  double solver_success_rate() const {
    if (solver_attempt_count_ == 0) {
      return 0.0;
    }
    return static_cast<double>(solver_success_count_) /
           static_cast<double>(solver_attempt_count_);
  }

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

  // Curvature as a function of arc length, evaluated at the PLAN's predicted s
  // per stage. The upstream (racing_mpc_node.cpp) linearizes/evaluates its
  // reference at abscissa = last_x_(PX, :); our old curvature_sequence assumed
  // a uniform target speed, so once the car accelerates the plan's real s
  // (spanning 10+ m) diverges from the assumed s (~3.7 m) and the ATV dynamics
  // get curvature from the wrong section of track, bending the plan off the
  // real track (the "wiggle / not following the track").
  double curvature_at_s(double s) const {
    if (curv_s_.empty()) {
      return current_reference_.curvature;
    }
    double q = s;
    if (curv_total_length_ > 0.0) {
      q = std::fmod(s, curv_total_length_);
      if (q < 0.0) {
        q += curv_total_length_;
      }
    }
    if (q <= curv_s_.front()) {
      return curv_k_.front();
    }
    if (q >= curv_s_.back()) {
      return curv_k_.back();
    }
    const auto it = std::upper_bound(curv_s_.begin(), curv_s_.end(), q);
    const auto hi = static_cast<std::size_t>(it - curv_s_.begin());
    const auto lo = hi - 1;
    const double frac = (q - curv_s_[lo]) / (curv_s_[hi] - curv_s_[lo]);
    return curv_k_[lo] + frac * (curv_k_[hi] - curv_k_[lo]);
  }

  void build_solver() {
    using casadi::DM;
    using casadi::MX;
    using casadi::Slice;

    const auto N = static_cast<casadi_int>(config_.horizon);
    opti_ = casadi::Opti("conic");
    X_ = opti_.variable(model_.nx(), N);
    U_ = opti_.variable(model_.nu(), N - 1);
    x0_ = opti_.parameter(model_.nx(), 1);
    dt_ = opti_.parameter(1, 1);
    kappa_ = opti_.parameter(1, N - 1);
    u_prev_param_ = opti_.parameter(model_.nu(), 1);
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
      opti_.subject_to(xip1 == casadi::MX::mtimes({A_params_[i], xi}) +
                                   casadi::MX::mtimes({B_params_[i], ui}) +
                                   C_params_[i]);

      // Paper eq. (4a) stage cost: the minimum-time indicator 1_F(x_t) (= 1 for
      // every not-yet-finished step) plus input and control-rate penalties. The
      // progress driver is NOT a stage reward but the terminal cost-to-go
      // J_N(x_N)^T*lambda below; with z_bar taken from the previous solution
      // the safe-set query walks forward and pulls x_N toward lower
      // time-to-finish.
      cost += 1.0;
      cost += config_.input_weight_lon * ui(st::UIndexSimple::LON) *
              ui(st::UIndexSimple::LON);
      cost += config_.input_weight_steer * ui(st::UIndexSimple::STEER_SIMPLE) *
              ui(st::UIndexSimple::STEER_SIMPLE);
      casadi::MX u_prev_i;
      if (i == 0) {
        u_prev_i = u_prev_param_;
      } else {
        u_prev_i = U_(Slice(), i - 1);
      }
      const auto du = ui - u_prev_i;
      cost += config_.control_rate_weight * casadi::MX::sumsqr(du);

      opti_.subject_to(opti_.bounded(config_.max_brake_force / 1000.0,
                                     ui(st::UIndexSimple::LON),
                                     config_.max_drive_force / 1000.0));
      opti_.subject_to(opti_.bounded(-config_.max_steer,
                                     ui(st::UIndexSimple::STEER_SIMPLE),
                                     config_.max_steer));
      opti_.subject_to(xi(st::XIndex::PY) >= -right_bound_);
      opti_.subject_to(xi(st::XIndex::PY) <= left_bound_);
      opti_.subject_to(opti_.bounded(
          0.0, xi(st::XIndex::VX), std::max(config_.target_speed * 2.0, 5.0)));
    }

    const auto xN = X_(Slice(), N - 1);
    // The terminal state must also respect the track corridor and speed bounds.
    // The stage loop above only constrains x_0..x_{N-2}; leaving x_N bounded
    // solely by the (slacked) safe-set constraint let its e_y drift off-track.
    opti_.subject_to(xN(st::XIndex::PY) >= -right_bound_);
    opti_.subject_to(xN(st::XIndex::PY) <= left_bound_);
    opti_.subject_to(opti_.bounded(0.0, xN(st::XIndex::VX),
                                   std::max(config_.target_speed * 2.0, 5.0)));
    opti_.subject_to(lambda_ >= 0.0);
    opti_.subject_to(casadi::MX::sum1(lambda_) == 1.0);
    opti_.subject_to(xN ==
                     casadi::MX::mtimes({ss_x_, lambda_}) + terminal_slack_);
    cost +=
        config_.safe_set_cost_weight * casadi::MX::mtimes({ss_costs_, lambda_});
    cost += kLambdaRidge * casadi::MX::sumsqr(lambda_);
    cost += kTerminalSlackPenalty * casadi::MX::sumsqr(terminal_slack_);
    cost += config_.terminal_lateral_weight * xN(st::XIndex::PY) *
            xN(st::XIndex::PY);
    cost += config_.terminal_heading_weight * xN(st::XIndex::YAW) *
            xN(st::XIndex::YAW);
    opti_.minimize(cost);

    // The FHOCP is a convex QP (affine A/B/C dynamics, convex quadratic cost,
    // linear constraints), so it is solved directly as a conic QP with qrqp.
    // Wrapping it in an SQP (sqpmethod) only adds outer-loop machinery around a
    // single QP solve and is ~7x slower for identical accuracy. error_on_fail
    // keeps a non-converged QP from dumping to the console; a max-iteration
    // limit is reported as SOLVER_RET_LIMITED and accepted by solve_limited().
    const auto solver_opts =
        casadi::Dict{{"print_time", false},
                     {"print_iter", false},
                     {"print_header", false},
                     {"print_info", false},
                     {"error_on_fail", false},
                     {"max_iter", config_.max_iter},
                     {"constr_viol_tol", config_.tolerance},
                     {"dual_inf_tol", config_.tolerance}};
    opti_.solver("qrqp", solver_opts);
  }

  LmpcControlCommand command_from_solution(const casadi::DM &x,
                                           const casadi::DM &u) const {
    const double steer =
        clamp(static_cast<double>(u(st::UIndexSimple::STEER_SIMPLE, 0)),
              -config_.max_steer, config_.max_steer);
    // Command the plan's longitudinal progress rate ds/dt a short preview
    // ahead. The Gym action is a forward-speed setpoint (~ds/dt for small
    // e_y/e_psi). Two reasons not to use the velocity state x(V) directly: (1)
    // with no stage progress reward the plan defers acceleration, so the
    // immediate next step is ~0; (2) the linearized model can advance s through
    // its affine offset while leaving x(V) low, so x(V) is an unreliable
    // command. The s-progress rate is what the terminal cost actually drives,
    // so previewing it keeps the car moving.
    const casadi_int preview = std::min(
        static_cast<casadi_int>(config_.command_preview_steps), x.size2() - 2);
    const double ds = static_cast<double>(x(st::XIndex::PX, preview + 1)) -
                      static_cast<double>(x(st::XIndex::PX, preview));
    const double velocity =
        clamp(ds / config_.dt, 0.0, std::max(config_.target_speed * 2.0, 5.0));
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

  casadi::DM command_native_u(const LmpcControlCommand &command) const {
    casadi::DM u = casadi::DM::zeros(model_.nu(), 1);
    const double acceleration =
        (command.velocity - std::max(current_state_.v_x, 0.0)) / config_.dt;
    const double force = kVehicleMass * acceleration;
    if (force >= 0.0) {
      u(st::UIndexSimple::LON) =
          clamp(force / 1000.0, 0.0, config_.max_drive_force / 1000.0);
    } else {
      u(st::UIndexSimple::LON) =
          clamp(force / 1000.0, config_.max_brake_force / 1000.0, 0.0);
    }
    u(st::UIndexSimple::STEER_SIMPLE) = command.steering;
    return u;
  }

  void store_predicted_horizon(const casadi::DM &x) {
    last_horizon_.clear();
    last_horizon_.reserve(static_cast<std::size_t>(x.size2()));
    for (casadi_int i = 0; i < x.size2(); ++i) {
      last_horizon_.push_back({static_cast<double>(x(st::XIndex::PX, i)),
                               static_cast<double>(x(st::XIndex::PY, i))});
    }
  }

  casadi::DM current_state_dm() const {
    return casadi::DM{current_state_.s,     current_state_.e_y,
                      current_state_.e_psi, std::max(current_state_.v_x, 0.05),
                      current_state_.v_y,   current_state_.omega};
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
      driven_laps_++;
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

  void set_dynamics_parameters(const casadi::DM &x_ref_mat,
                               const casadi::DM &u_ref_mat) {
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
      casadi::DM x_ref = x_ref_mat(Slice(), i);
      // Floor the linearization speed: the dynamic single-track Jacobian
      // scales like 1/v_x and blows up (~1e5) near rest, making the QP
      // non-finite. Linearize at max(v_x, floor) so A/B/C stay well
      // conditioned; the true state is still pinned via x0 and the bounds.
      x_ref(st::XIndex::VX) =
          std::max(static_cast<double>(x_ref(st::XIndex::VX)),
                   config_.linearization_speed_floor);
      const casadi::DM u_ref = u_ref_mat(Slice(), i);
      casadi::DM A;
      casadi::DM B;
      casadi::DM C;
      const double curvature =
          curvature_at_s(static_cast<double>(x_ref_mat(st::XIndex::PX, i)));
      compute_nominal_affine_model(x_ref, u_ref, curvature, A, B, C);
      if (driven_laps_ > 0 && should_update_regression(i)) {
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
        casadi::DM::vertcat({x, u}),
        nominal_a,
        nominal_b,
        nominal_c,
        model_.discrete_dynamics(),
        config_.reg_dist_max,
        static_cast<casadi_int>(config_.reg_max_points),
        static_cast<casadi_int>(config_.reg_max_points_per_lap),
        rt::RegQuery::Indices{
            {st::XIndex::PX, st::XIndex::PY, st::XIndex::YAW, st::XIndex::VX,
             st::XIndex::VY, st::XIndex::VYAW},
            {st::XIndex::PX, st::XIndex::PY, st::XIndex::YAW, st::XIndex::VX,
             st::XIndex::VY, st::XIndex::VYAW},
            {st::XIndex::PX, st::XIndex::PY, st::XIndex::YAW, st::XIndex::VX,
             st::XIndex::VY, st::XIndex::VYAW}},
        rt::RegQuery::Indices{
            {st::UIndexSimple::LON, st::UIndexSimple::STEER_SIMPLE},
            {st::UIndexSimple::LON, st::UIndexSimple::STEER_SIMPLE},
            {st::UIndexSimple::LON, st::UIndexSimple::STEER_SIMPLE}},
        rt::RegQuery::Indices{
            {st::XIndex::VX}, {st::XIndex::VY}, {st::XIndex::VYAW}},
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
        ss_costs = result.J;
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
        // NOTE: upstream racing_mpc normalizes here (ss_j - ss_j[0]) to keep
        // the terminal cost-to-go well scaled, but that only works together
        // with its variable scaling (scale_x/scale_u). Without scaling,
        // normalizing shrinks the cost-to-go pull below the input penalties and
        // the car freezes; raw J drives but is large and ill-conditions qrqp.
        // Proper fix is to port the variable scaling (see task #7).
      }
    }

    ss_x_value_ = ss_x;
    ss_costs_value_ = ss_costs;
  }

  LmpcConfig config_;
  st::SingleTrackPlanarModel model_;
  std::unique_ptr<rt::SafeSetManager> safe_set_;
  casadi::Opti opti_;
  casadi::MX X_;
  casadi::MX U_;
  casadi::MX x0_;
  casadi::MX dt_;
  casadi::MX kappa_;
  casadi::MX u_prev_param_;
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
  std::vector<std::array<double, 2>> last_horizon_;
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
  // Laps actually driven closed-loop (excludes seeded D^0 laps). The error
  // dynamics regression must only use real (x,u)->x_next data; the synthesized
  // seed's u is analytic and would corrupt the ATV model, so regression is
  // gated on this while the safe set / cost-to-go still uses the seed.
  std::size_t driven_laps_ = 0;
  std::size_t last_safe_set_points_ = 0;
  std::size_t solver_attempt_count_ = 0;
  std::size_t solver_success_count_ = 0;
  std::string last_solver_status_;
  // Curvature profile kappa(s) for evaluating per-stage curvature at the plan's
  // predicted s (set once via set_curvature_profile).
  std::vector<double> curv_s_;
  std::vector<double> curv_k_;
  double curv_total_length_ = 0.0;
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

void NativeLMPCController::add_initial_lap(
    const std::vector<std::vector<double>> &x,
    const std::vector<std::vector<double>> &u, const std::vector<double> &k,
    const std::vector<double> &t) {
  impl_->add_initial_lap(x, u, k, t);
}

void NativeLMPCController::set_curvature_profile(const std::vector<double> &s,
                                                 const std::vector<double> &k,
                                                 double total_length) {
  impl_->set_curvature_profile(s, k, total_length);
}

LmpcControlCommand NativeLMPCController::control() { return impl_->control(); }

std::vector<std::array<double, 2>>
NativeLMPCController::predicted_horizon() const {
  return impl_->predicted_horizon();
}

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

double NativeLMPCController::solver_success_rate() const {
  return impl_->solver_success_rate();
}

std::string NativeLMPCController::last_solver_status() const {
  return impl_->last_solver_status();
}

} // namespace f110_gym_lmpc
