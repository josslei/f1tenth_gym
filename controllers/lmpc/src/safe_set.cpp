#include "safe_set.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <tuple>
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

SafeSet::SafeSet(const std::string &seed_lap_csv_path, double track_length)
    : track_length(track_length) {
  if (!(track_length > 0.0) || !std::isfinite(track_length)) {
    throw std::invalid_argument(
        "SafeSet::SafeSet: track_length must be finite and positive");
  }
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

double SafeSet::cost_scale() const {
  double scale = 0.0;
  for (const std::vector<SafeSetSample> &lap : laps) {
    for (const SafeSetSample &sample : lap) {
      scale = std::max(scale, sample.J);
    }
  }
  return scale;
}

casadi_int SafeSet::terminal_point_count(casadi_int K) const {
  if (K <= 0) {
    throw std::invalid_argument(
        "SafeSet::terminal_point_count: K must be positive");
  }
  if (laps.empty()) {
    throw std::runtime_error(
        "SafeSet::terminal_point_count: safe set contains no laps");
  }
  for (std::size_t lap_idx = 0; lap_idx < laps.size(); ++lap_idx) {
    if (laps[lap_idx].size() < static_cast<std::size_t>(K)) {
      throw std::runtime_error("SafeSet::terminal_point_count: lap " +
                               std::to_string(lap_idx) +
                               " contains fewer than K samples");
    }
  }
  return K * num_laps();
}

SafeSet::QueryResult SafeSet::query(const casadi::DM &x_query, casadi_int K,
                                    const casadi::DM &state_scale) const {
  const casadi_int q = terminal_point_count(K);
  QueryResult result{casadi::DM::zeros(dynamics::kStateDim, q),
                     casadi::DM::zeros(q, 1)};
  casadi_int output_col = 0;

  for (const std::vector<SafeSetSample> &lap : laps) {
    // T = transitions in this lap (lap.size() samples = T+1 states, per
    // add_lap()'s own convention). Ranks a PERIODIC candidate pool: every
    // real sample considered at THREE shifted positions -- shift=-1
    // (s-track_length, J+T), shift=0 (s, J), shift=+1 (s+track_length,
    // J-T) -- so a terminal reference near/past the seam naturally matches
    // real data from the adjacent lap wraparound instead of being clamped
    // to this lap's own endpoint (class comment has the full rationale;
    // this replaces the removed finish-mode fabrication in
    // LMPCController::solve_once). shift/sample_idx are carried as
    // deterministic tie-breakers for equal distances.
    const double T = static_cast<double>(lap.size()) - 1.0;
    std::vector<std::tuple<double, std::size_t, int>> ranked;
    ranked.reserve(lap.size() * 3);
    for (std::size_t sample_idx = 0; sample_idx < lap.size(); ++sample_idx) {
      for (int shift = -1; shift <= 1; ++shift) {
        casadi::DM x_shifted = lap[sample_idx].x;
        x_shifted(dynamics::S) =
            static_cast<double>(x_shifted(dynamics::S)) + shift * track_length;
        ranked.emplace_back(
            normalized_distance_sq(x_shifted, x_query, state_scale), sample_idx,
            shift);
      }
    }
    std::partial_sort(ranked.begin(), ranked.begin() + K, ranked.end());
    for (casadi_int selected = 0; selected < K; ++selected) {
      const auto &[distance, sample_idx, shift] =
          ranked[static_cast<std::size_t>(selected)];
      (void)distance;
      const SafeSetSample &sample = lap[sample_idx];
      casadi::DM x_shifted = sample.x;
      x_shifted(dynamics::S) =
          static_cast<double>(x_shifted(dynamics::S)) + shift * track_length;
      result.X_ss(casadi::Slice(), output_col) = x_shifted;
      result.J_ss(output_col, 0) = sample.J - shift * T;
      ++output_col;
    }
  }

  if (output_col != q || result.X_ss.size1() != dynamics::kStateDim ||
      result.X_ss.size2() != q || result.J_ss.size1() != q ||
      result.J_ss.size2() != 1 || !result.X_ss.is_regular() ||
      !result.J_ss.is_regular()) {
    throw std::runtime_error("SafeSet::query: constructed terminal matrices "
                             "have invalid dimensions or values");
  }
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
