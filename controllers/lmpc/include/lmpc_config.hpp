#ifndef LMPC__LMPC_CONFIG_HPP_
#define LMPC__LMPC_CONFIG_HPP_

#include <cstdint>
#include <string>
#include <vector>

namespace lmpc {

namespace detail {
inline std::vector<double> DefaultRegressionQ() {
  std::vector<double> q(64, 0.0);
  for (int i = 0; i < 8; ++i)
    q[i * 8 + i] = 1.0;
  return q;
}
} // namespace detail

struct VehicleParams {
  double mu = 1.0489;
  double C_Sf = 4.718;
  double C_Sr = 5.4562;
  double lf = 0.15875;
  double lr = 0.17145;
  double h = 0.074;
  double m = 3.74;
  double I = 0.04712;
};

struct LmpcConfig {
  double dt = 0.025;
  long long horizon_steps = 75;
  std::string centerline_csv_path;
  std::string seed_lap_csv_path;
  VehicleParams vehicle_params{};

  long long K = 16;
  double a_min = -9.51;
  double a_max = 9.51;
  double delta_min = -0.41;
  double delta_max = 0.41;
  double v_max = 10.0;
  double velocity_threshold = 0.8;
  double map_margin = 0.1;
  double waypoint_space = 0.2;
  double r_accel = 1.5;
  double r_steer = 18.0;
  double r_d_accel = 0.1;
  double r_d_steer = 0.1;
  double ey_slack_l2 = 3000.0;
  double terminal_slack_weight = 800.0;
  long long osqp_max_iter = 20000;
  long long osqp_scaling = -1;
  double osqp_eps_prim_inf = 0.0;
  double osqp_eps_abs = 0.0;
  double osqp_eps_rel = 0.0;

  // Filled by the Python adaptation layer from the map and converted CSVs.
  std::vector<std::int8_t> occupancy_grid;
  std::uint32_t map_width = 0;
  std::uint32_t map_height = 0;
  double map_resolution = 0.0;
  double map_origin_x = 0.0;
  double map_origin_y = 0.0;
  std::string reference_waypoint_csv_path;
  std::string reference_seed_lap_csv_path;
  double initial_x = 0.0;
  double initial_y = 0.0;
  double initial_yaw = 0.0;

  // Dynamics-error regression (additive correction to the nominal model's
  // one-step v/omega/beta prediction, learned only in those rows -- see
  // plan.md and ref/lmpc.tex). M, h, lambda have no default: pick them from
  // data before enabling.
  bool regression_enabled = false;
  long long regression_num_neighbors = 0; // M
  double regression_bandwidth = 0.0;      // h
  double regression_regularization = 0.0; // lambda
  std::vector<double> regression_Q =
      detail::DefaultRegressionQ(); // 8x8, row-major
};

} // namespace lmpc

#endif // LMPC__LMPC_CONFIG_HPP_
