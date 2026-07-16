#include <casadi/casadi.hpp>

#include <cmath>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "dynamics/common.hpp"
#include "lmpc_controller.hpp"
#include "qp_builder.hpp"
#include "safe_set.hpp"
#include "track.hpp"

namespace lmpc {

struct LMPCControllerTestAccess {
  static void set_warm_start(LMPCController &controller, bool ready) {
    controller.has_warm_start = ready;
  }

  static bool has_warm_start(const LMPCController &controller) {
    return controller.has_warm_start;
  }

  static int failure_count(const LMPCController &controller) {
    return controller.consecutive_solve_failures;
  }

  static void fail(LMPCController &controller) {
    controller.record_solve_failure();
  }

  static void succeed(LMPCController &controller) {
    controller.record_solve_success();
  }
};

} // namespace lmpc

namespace {

void require(bool condition, const std::string &message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

void require_close(double actual, double expected, double tolerance,
                   const std::string &message) {
  require(std::abs(actual - expected) <= tolerance,
          message + ": expected " + std::to_string(expected) + ", got " +
              std::to_string(actual));
}

void write_lap_csv(const std::filesystem::path &path, int period,
                   double track_length) {
  std::ofstream csv(path);
  csv << "vx,vy,omega,epsi,s,ey,t,a,delta,J\n";
  for (int k = 0; k <= period; ++k) {
    csv << "1,0,0,0," << track_length * k / period << ",0," << k;
    csv << (k < period ? ",0,0," : ",,,");
    csv << period - k << '\n';
  }
}

std::vector<lmpc::SafeSetSample> make_lap(int period, double track_length) {
  std::vector<lmpc::SafeSetSample> lap;
  lap.reserve(static_cast<std::size_t>(period + 1));
  for (int k = 0; k <= period; ++k) {
    casadi::DM x = casadi::DM::zeros(lmpc::dynamics::kStateDim, 1);
    x(lmpc::dynamics::S) = track_length * k / period;
    lap.emplace_back(x, casadi::DM::zeros(lmpc::dynamics::kControlDim, 1),
                     period - k, k < period);
  }
  return lap;
}

void test_contiguous_periodic_segment() {
  const std::filesystem::path csv_path =
      std::filesystem::temp_directory_path() / "lmpc_safe_set_segment.csv";
  write_lap_csv(csv_path, 10, 10.0);

  lmpc::SafeSet safe_set(csv_path.string(), 10.0);
  const lmpc::SafeSet::QueryResult result =
      safe_set.query_local_segments(8.5, 6);
  const std::vector<double> expected_s{6, 7, 8, 9, 10, 11};
  const std::vector<double> expected_J{5, 4, 3, 2, 1, 0};
  const std::vector<long> expected_indices{6, 7, 8, 9, 10, 1};
  require(result.selected.size() == expected_s.size(),
          "safe-set query returned the wrong number of points");
  for (std::size_t i = 0; i < expected_s.size(); ++i) {
    require_close(static_cast<double>(result.X_ss(lmpc::dynamics::S, i)),
                  expected_s[i], 1e-12, "unexpected lifted s");
    require_close(static_cast<double>(result.J_ss(i)), expected_J[i], 1e-12,
                  "unexpected local cost");
    require(result.selected[i].sample_index == expected_indices[i],
            "selected samples are not temporally contiguous");
  }

  const lmpc::SafeSet::QueryResult forward_cycles =
      safe_set.query_local_segments(28.5, 6);
  const lmpc::SafeSet::QueryResult reverse_cycles =
      safe_set.query_local_segments(-11.5, 6);
  for (std::size_t i = 0; i < expected_s.size(); ++i) {
    require_close(
        static_cast<double>(forward_cycles.X_ss(lmpc::dynamics::S, i)),
        expected_s[i] + 20.0, 1e-12, "positive multi-cycle lift is incorrect");
    require_close(
        static_cast<double>(reverse_cycles.X_ss(lmpc::dynamics::S, i)),
        expected_s[i] - 20.0, 1e-12, "negative multi-cycle lift is incorrect");
  }

  safe_set.add_lap(make_lap(13, 10.0));
  const lmpc::SafeSet::QueryResult different_periods =
      safe_set.query_local_segments(8.5, 6);
  for (casadi_int lap_index = 0; lap_index < 2; ++lap_index) {
    for (casadi_int j = 0; j < 6; ++j) {
      require_close(
          static_cast<double>(different_periods.J_ss(lap_index * 6 + j)), 5 - j,
          1e-12, "lap-specific transition period produced the wrong cost");
    }
  }
  std::filesystem::remove(csv_path);
}

void test_closed_track_length_and_curvature_seam() {
  const std::filesystem::path csv_path =
      std::filesystem::temp_directory_path() / "lmpc_closed_track.csv";
  {
    std::ofstream csv(csv_path);
    csv << "# x_m,y_m,w_tr_right_m,w_tr_left_m\n";
    csv << "0,0,1,1\n2,0,1,1\n2,1,1,1\n";
  }

  const lmpc::Track track(csv_path.string());
  const double expected_length = 3.0 + std::sqrt(5.0);
  require_close(track.length(), expected_length, 1e-12,
                "track length omits the closing segment");
  require_close(track.curvature(expected_length - 1e-8), track.curvature(0.0),
                1e-7, "curvature is discontinuous at the closing seam");
  std::filesystem::remove(csv_path);
}

void test_normalized_terminal_slack() {
  using namespace lmpc;
  using namespace lmpc::dynamics;

  const casadi::DM state_scale = casadi::DM({1, 1, 2, 1, 10, 1});
  QpBuilder qp(1, 1, QpBounds{-10, 10, -1, 1, 10, 1},
               QpWeights{0, 800, 0.01, 0.01, 10, 100},
               QpScaling{state_scale, casadi::DM({10, 1}), 10}, "qrqp");

  casadi::DM x0 = casadi::DM::zeros(kStateDim, 1);
  x0(OMEGA) = 1.0;
  const std::vector<QpStage> stages{QpStage{
      casadi::DM::eye(kStateDim), casadi::DM::zeros(kStateDim, kControlDim),
      casadi::DM::zeros(kStateDim, 1)}};
  const QpSolution solution =
      qp.solve(x0, casadi::DM::zeros(kControlDim, 1), stages,
               casadi::DM::zeros(kStateDim, 1), casadi::DM::zeros(1, 1),
               casadi::DM::repmat(x0, 1, 2), casadi::DM::zeros(kControlDim, 1),
               casadi::DM::ones(1, 1));

  require(solution.success,
          "terminal-slack QP did not solve: " + solution.message);
  require_close(static_cast<double>(solution.terminal_slack(OMEGA)), 0.5, 1e-6,
                "omega terminal slack is not normalized");
  const casadi::DM residual = (solution.x_traj(casadi::Slice(), 1) -
                               state_scale * solution.terminal_slack) /
                              state_scale;
  require(static_cast<double>(casadi::DM::mmax(fabs(residual))) < 1e-6,
          "slack-inclusive terminal residual is not zero");
  qp.clear_dual_warm_start();
}

void test_consecutive_failure_recovery() {
  const std::filesystem::path track_path =
      std::filesystem::temp_directory_path() / "lmpc_failure_track.csv";
  const std::filesystem::path lap_path =
      std::filesystem::temp_directory_path() / "lmpc_failure_lap.csv";
  {
    std::ofstream csv(track_path);
    csv << "# x_m,y_m,w_tr_right_m,w_tr_left_m\n";
    csv << "0,0,1,1\n2,0,1,1\n2,1,1,1\n";
  }
  write_lap_csv(lap_path, 10, 3.0 + std::sqrt(5.0));

  lmpc::LmpcConfig config;
  config.centerline_csv_path = track_path.string();
  config.seed_lap_csv_path = lap_path.string();
  config.horizon_steps = 1;
  config.K = 2;
  config.solver_name = "qrqp";
  lmpc::LMPCController controller(config);
  lmpc::LMPCControllerTestAccess::set_warm_start(controller, true);

  lmpc::LMPCControllerTestAccess::fail(controller);
  require(lmpc::LMPCControllerTestAccess::has_warm_start(controller),
          "first failure discarded the primal warm start");
  lmpc::LMPCControllerTestAccess::fail(controller);
  require(lmpc::LMPCControllerTestAccess::has_warm_start(controller),
          "second failure discarded the primal warm start");
  lmpc::LMPCControllerTestAccess::fail(controller);
  require(!lmpc::LMPCControllerTestAccess::has_warm_start(controller),
          "failure threshold did not request a primal reseed");

  lmpc::LMPCControllerTestAccess::succeed(controller);
  require(lmpc::LMPCControllerTestAccess::failure_count(controller) == 0,
          "successful solve did not reset the failure count");
  std::filesystem::remove(track_path);
  std::filesystem::remove(lap_path);
}

} // namespace

int main() {
  test_contiguous_periodic_segment();
  test_closed_track_length_and_curvature_seam();
  test_normalized_terminal_slack();
  test_consecutive_failure_recovery();
  return 0;
}
