#include <casadi/casadi.hpp>

#include <cmath>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "dynamics/common.hpp"
#include "qp_builder.hpp"
#include "safe_set.hpp"

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

void test_contiguous_periodic_segment() {
  const std::filesystem::path csv_path =
      std::filesystem::temp_directory_path() / "lmpc_safe_set_segment.csv";
  {
    std::ofstream csv(csv_path);
    csv << "vx,vy,omega,epsi,s,ey,t,a,delta,J\n";
    for (int k = 0; k <= 10; ++k) {
      csv << "1,0,0,0," << k << ",0," << k;
      csv << (k < 10 ? ",0,0," : ",,,");
      csv << 10 - k << '\n';
    }
  }

  lmpc::SafeSet safe_set(csv_path.string(), 10.0);
  const lmpc::SafeSet::QueryResult result =
      safe_set.query_local_segments(8.5, 6);
  std::filesystem::remove(csv_path);

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

} // namespace

int main() {
  test_contiguous_periodic_segment();
  test_normalized_terminal_slack();
  return 0;
}
