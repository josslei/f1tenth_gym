#ifndef LMPC__LMPC_CONTROLLER_HPP_
#define LMPC__LMPC_CONTROLLER_HPP_

#include <casadi/casadi.hpp>
#include <memory>

#include "lmpc_config.hpp"

namespace lmpc {

struct ControllerTimings {
  double rollout_lin_ms = 0.0;
  double knn_ms = 0.0;
  double regression_ms = 0.0;
  double set_params_ms = 0.0;
  double solver_ms = 0.0;
  double postcheck_ms = 0.0;
};

// The stable C++ controller API. All LearningMPC-specific behavior is hidden
// behind this adaptation boundary.
class LMPCController {
public:
  explicit LMPCController(const LmpcConfig &config);
  ~LMPCController();
  LMPCController(LMPCController &&) noexcept;
  LMPCController &operator=(LMPCController &&) noexcept;

  LMPCController(const LMPCController &) = delete;
  LMPCController &operator=(const LMPCController &) = delete;

  void reset();

  // Global state order: [x, y, yaw, speed, yaw_rate, slip_angle].
  void update(const casadi::DM &x, double t, double actual_delta);

  // Control order: [acceleration, steering_angle].
  casadi::DM control();

  // LearningMPC owns lap recording internally; retained for API compatibility.
  void add_lap(const casadi::DM &x_lap, const casadi::DM &u_lap,
               const casadi::DM &J_lap);

  casadi::DM predicted_next_state() const;
  casadi::DM predicted_trajectory() const;
  const ControllerTimings &last_timings() const;
  const casadi::DM &last_terminal_slack_value() const;
  bool last_solve_ok() const;
  bool using_dynamic_model() const;
  int regression_pool_size() const;
  double last_regression_correction_norm() const;

private:
  struct Impl;
  std::unique_ptr<Impl> impl;
};

} // namespace lmpc

#endif // LMPC__LMPC_CONTROLLER_HPP_
