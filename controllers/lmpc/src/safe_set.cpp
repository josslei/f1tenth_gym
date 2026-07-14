#include "safe_set.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <utility>

#include "dynamics/common.hpp"

namespace lmpc {

namespace {

// Parses the header written by
// scripts/lmpc_collect_seed_lap.py::write_seed_lap_csv:
// vx,vy,omega,epsi,s,ey,t,a,delta,J. The a/delta columns are kept (not
// discarded) -- trajectory_segment() returns them as the first solve's
// control warm start. They are blank only on the final row (no successor
// state), which maps to has_control = false.
std::vector<SafeSetSample> load_lap(const std::string &csv_path) {
  std::ifstream file(csv_path);
  if (!file.is_open()) {
    throw std::runtime_error("SafeSet: could not open seed-lap CSV: " +
                             csv_path);
  }

  std::string header;
  if (!std::getline(file, header)) {
    throw std::runtime_error("SafeSet: empty seed-lap CSV: " + csv_path);
  }

  std::vector<SafeSetSample> samples;
  std::string line;
  while (std::getline(file, line)) {
    if (line.empty()) {
      continue;
    }
    std::istringstream iss(line);
    std::string field;
    std::vector<std::string> fields;
    while (std::getline(iss, field, ',')) {
      fields.push_back(field);
    }
    // vx,vy,omega,epsi,s,ey,t,a,delta,J -- 10 columns.
    if (fields.size() != 10) {
      throw std::runtime_error("SafeSet: malformed row in " + csv_path);
    }

    casadi::DM x = casadi::DM::zeros(dynamics::kStateDim, 1);
    x(dynamics::VX) = std::stod(fields[0]);
    x(dynamics::VY) = std::stod(fields[1]);
    x(dynamics::OMEGA) = std::stod(fields[2]);
    x(dynamics::EPSI) = std::stod(fields[3]);
    x(dynamics::S) = std::stod(fields[4]);
    x(dynamics::EY) = std::stod(fields[5]);
    const double J = std::stod(fields[9]);

    casadi::DM u = casadi::DM::zeros(dynamics::kControlDim, 1);
    const bool has_control = !fields[7].empty() && !fields[8].empty();
    if (has_control) {
      u(dynamics::A) = std::stod(fields[7]);
      u(dynamics::DELTA) = std::stod(fields[8]);
    }

    samples.push_back(SafeSetSample{x, u, J, has_control});
  }

  if (samples.empty()) {
    throw std::runtime_error("SafeSet: no data rows in " + csv_path);
  }
  return samples;
}

double normalized_distance_sq(const casadi::DM &a, const casadi::DM &b,
                              const casadi::DM &scale) {
  const double dvx = (static_cast<double>(a(dynamics::VX)) -
                      static_cast<double>(b(dynamics::VX))) /
                     static_cast<double>(scale(dynamics::VX));
  const double depsi = (static_cast<double>(a(dynamics::EPSI)) -
                        static_cast<double>(b(dynamics::EPSI))) /
                       static_cast<double>(scale(dynamics::EPSI));
  const double ds = (static_cast<double>(a(dynamics::S)) -
                     static_cast<double>(b(dynamics::S))) /
                    static_cast<double>(scale(dynamics::S));
  const double dey = (static_cast<double>(a(dynamics::EY)) -
                      static_cast<double>(b(dynamics::EY))) /
                     static_cast<double>(scale(dynamics::EY));
  return dvx * dvx + depsi * depsi + ds * ds + dey * dey;
}

} // namespace

SafeSet::SafeSet(const std::string &seed_lap_csv_path) {
  laps.push_back(load_lap(seed_lap_csv_path));
}

void SafeSet::add_lap(std::vector<SafeSetSample> lap) {
  if (lap.empty()) {
    throw std::invalid_argument("SafeSet::add_lap: empty lap");
  }
  laps.push_back(std::move(lap));
  if (laps.size() > kMaxLaps) {
    laps.erase(laps.begin());
  }
}

void SafeSet::add_lap(const std::string &lap_csv_path) {
  add_lap(load_lap(lap_csv_path));
}

double SafeSet::data_end_s() const {
  double end_s = std::numeric_limits<double>::infinity();
  for (const std::vector<SafeSetSample> &lap : laps) {
    end_s = std::min(end_s, static_cast<double>(lap.back().x(dynamics::S)));
  }
  return end_s;
}

double SafeSet::cost_scale() const {
  double scale = 0.0;
  for (const std::vector<SafeSetSample> &lap : laps) {
    for (const SafeSetSample &sample : lap) {
      scale = std::max(scale, sample.J);
    }
  }
  return scale;
}

SafeSet::QueryResult SafeSet::query(const casadi::DM &x_query, casadi_int K,
                                    const casadi::DM &state_scale) const {
  // K remains the per-lap candidate count. The terminal QP receives one
  // global simplex selected from the combined candidate pool. Each sample's
  // distance is computed exactly once and carried alongside its pointer --
  // recomputing it inside sort comparators is O(n log n) casadi::DM
  // element reads per lap per control step, a real per-frame cost.
  std::vector<std::pair<double, const SafeSetSample *>> candidates;
  for (const std::vector<SafeSetSample> &lap : laps) {
    const std::size_t k =
        std::min<std::size_t>(static_cast<std::size_t>(K), lap.size());
    std::vector<std::pair<double, const SafeSetSample *>> ranked;
    ranked.reserve(lap.size());
    for (const SafeSetSample &sample : lap) {
      ranked.emplace_back(
          normalized_distance_sq(sample.x, x_query, state_scale), &sample);
    }
    std::nth_element(ranked.begin(), ranked.begin() + (k - 1), ranked.end());
    candidates.insert(candidates.end(), ranked.begin(), ranked.begin() + k);
  }

  std::sort(candidates.begin(), candidates.end());

  std::vector<casadi::DM> x_cols;
  std::vector<double> j_vals;
  std::vector<std::array<double, dynamics::kStateDim>> basis;
  x_cols.reserve(kTerminalSimplexSize);
  j_vals.reserve(kTerminalSimplexSize);
  basis.reserve(kTerminalSimplexSize - 1);

  const SafeSetSample *base = candidates.front().second;
  x_cols.push_back(base->x);
  j_vals.push_back(base->J);

  constexpr double kAffineRankTolerance = 1e-3;
  for (std::size_t candidate_idx = 1;
       candidate_idx < candidates.size() &&
       static_cast<casadi_int>(x_cols.size()) < kTerminalSimplexSize;
       ++candidate_idx) {
    const SafeSetSample *candidate = candidates[candidate_idx].second;
    std::array<double, dynamics::kStateDim> residual{};
    for (casadi_int row = 0; row < dynamics::kStateDim; ++row) {
      residual[static_cast<std::size_t>(row)] =
          (static_cast<double>(candidate->x(row)) -
           static_cast<double>(base->x(row))) /
          static_cast<double>(state_scale(row));
    }

    for (const auto &direction : basis) {
      double projection = 0.0;
      for (casadi_int row = 0; row < dynamics::kStateDim; ++row) {
        projection += residual[static_cast<std::size_t>(row)] *
                      direction[static_cast<std::size_t>(row)];
      }
      for (casadi_int row = 0; row < dynamics::kStateDim; ++row) {
        residual[static_cast<std::size_t>(row)] -=
            projection * direction[static_cast<std::size_t>(row)];
      }
    }

    double norm_sq = 0.0;
    for (double value : residual) {
      norm_sq += value * value;
    }
    const double norm = std::sqrt(norm_sq);
    if (norm > kAffineRankTolerance) {
      for (double &value : residual) {
        value /= norm;
      }
      basis.push_back(residual);
      x_cols.push_back(candidate->x);
      j_vals.push_back(candidate->J);
    }
  }

  QueryResult result;
  result.X_ss = casadi::DM::horzcat(x_cols);
  result.J_ss = casadi::DM(j_vals);
  return result;
}

SafeSet::TrajectorySegment
SafeSet::trajectory_segment(const casadi::DM &x_query, casadi_int horizon_steps,
                            const casadi::DM &state_scale) const {
  using casadi::Slice;

  const std::vector<SafeSetSample> &lap = laps.back();
  const std::size_t n = lap.size();

  std::size_t nearest = 0;
  double best = normalized_distance_sq(lap[0].x, x_query, state_scale);
  for (std::size_t i = 1; i < n; ++i) {
    const double d = normalized_distance_sq(lap[i].x, x_query, state_scale);
    if (d < best) {
      best = d;
      nearest = i;
    }
  }

  TrajectorySegment segment;
  segment.x_traj = casadi::DM::zeros(dynamics::kStateDim, horizon_steps + 1);
  segment.u_traj = casadi::DM::zeros(dynamics::kControlDim, horizon_steps);
  for (casadi_int t = 0; t <= horizon_steps; ++t) {
    // Clamp past the lap's end (one open lap, s non-periodic): hold the
    // final sample rather than wrapping to the start line.
    const std::size_t i =
        std::min(nearest + static_cast<std::size_t>(t), n - 1);
    segment.x_traj(Slice(), t) = lap[i].x;
    if (t < horizon_steps) {
      // The final sample has no recorded control (has_control == false);
      // hold the last real one instead of its zero placeholder.
      const std::size_t i_u = lap[i].has_control ? i : i - 1;
      segment.u_traj(Slice(), t) = lap[i_u].u;
    }
  }
  return segment;
}

} // namespace lmpc
