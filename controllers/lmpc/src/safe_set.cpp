#include "safe_set.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <numeric>
#include <sstream>
#include <stdexcept>

#include "dynamics/common.hpp"

namespace lmpc {

namespace {

// Parses the header written by
// scripts/lmpc_collect_seed_lap.py::write_seed_lap_csv:
// vx,vy,omega,epsi,s,ey,t,a,delta,J -- only x = [vx,vy,omega,epsi,s,ey] and
// J are needed for the safe-set query, so a/delta/t (which are blank on the
// CSV's last row anyway -- there is no successor state for it) are read and
// discarded.
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

    samples.push_back(SafeSetSample{x, J});
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

void SafeSet::add_lap(const std::string &lap_csv_path) {
  laps.push_back(load_lap(lap_csv_path));
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
  std::vector<const SafeSetSample *> candidates;

  // K remains the per-lap candidate count. The terminal QP receives one
  // global simplex selected from the combined candidate pool.
  for (const std::vector<SafeSetSample> &lap : laps) {
    const casadi_int k =
        std::min<casadi_int>(K, static_cast<casadi_int>(lap.size()));
    std::vector<std::size_t> idx(lap.size());
    std::iota(idx.begin(), idx.end(), 0);
    std::partial_sort(
        idx.begin(), idx.begin() + k, idx.end(),
        [&](std::size_t i, std::size_t j) {
          return normalized_distance_sq(lap[i].x, x_query, state_scale) <
                 normalized_distance_sq(lap[j].x, x_query, state_scale);
        });
    for (casadi_int n = 0; n < k; ++n) {
      candidates.push_back(&lap[idx[n]]);
    }
  }

  std::sort(candidates.begin(), candidates.end(),
            [&](const auto *a, const auto *b) {
              return normalized_distance_sq(a->x, x_query, state_scale) <
                     normalized_distance_sq(b->x, x_query, state_scale);
            });

  std::vector<casadi::DM> x_cols;
  std::vector<double> j_vals;
  std::vector<std::array<double, dynamics::kStateDim>> basis;
  x_cols.reserve(kTerminalSimplexSize);
  j_vals.reserve(kTerminalSimplexSize);
  basis.reserve(kTerminalSimplexSize - 1);

  const SafeSetSample *base = candidates.front();
  x_cols.push_back(base->x);
  j_vals.push_back(base->J);

  constexpr double kAffineRankTolerance = 1e-3;
  for (std::size_t candidate_idx = 1;
       candidate_idx < candidates.size() &&
       static_cast<casadi_int>(x_cols.size()) < kTerminalSimplexSize;
       ++candidate_idx) {
    const SafeSetSample *candidate = candidates[candidate_idx];
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

} // namespace lmpc
