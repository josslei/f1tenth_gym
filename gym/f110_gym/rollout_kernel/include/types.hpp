#ifndef F110_ROLLOUT_KERNEL_TYPES_HPP_
#define F110_ROLLOUT_KERNEL_TYPES_HPP_

#include <array>
#include <cstddef>

namespace f110_rollout_kernel {

enum class Integrator {
  Euler = 0,
  RK4 = 1,
};

struct F110Params {
  double mu = 1.0489;
  double c_sf = 4.718;
  double c_sr = 5.4562;
  double lf = 0.15875;
  double lr = 0.17145;
  double h = 0.074;
  double m = 3.74;
  double inertia = 0.04712;
  double s_min = -0.4189;
  double s_max = 0.4189;
  double sv_min = -3.2;
  double sv_max = 3.2;
  double v_switch = 7.319;
  double a_max = 9.51;
  double v_min = -5.0;
  double v_max = 20.0;
  double timestep = 0.01;
};

struct F110State {
  double x = 0.0;
  double y = 0.0;
  double steer_angle = 0.0;
  double velocity = 0.0;
  double yaw_angle = 0.0;
  double yaw_rate = 0.0;
  double slip_angle = 0.0;
  double steer_buffer_0 = 0.0;
  double steer_buffer_1 = 0.0;
  int steer_buffer_len = 0;
  bool in_collision = false;
};

struct F110Action {
  double steer = 0.0;
  double velocity = 0.0;
};

struct F110StepResult {
  F110State state;
  double reward = 0.0;
  double discount = 1.0;
  bool terminal = false;
};

inline constexpr std::size_t kStateSize = 11;
inline constexpr std::size_t kActionSize = 2;
inline constexpr std::size_t kStepResultSize = 14;

using StateVector = std::array<double, 7>;
using ControlVector = std::array<double, 2>;

} // namespace f110_rollout_kernel

#endif // F110_ROLLOUT_KERNEL_TYPES_HPP_
