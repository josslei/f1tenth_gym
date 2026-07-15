#include "safe_set.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <utility>

#include "dynamics/common.hpp"

namespace lmpc {

SafeSetSample::SafeSetSample(casadi::DM x_in, casadi::DM u_in, double J_in,
                             bool has_control_in)
    : x(x_in), u(std::move(u_in)), J(J_in), has_control(has_control_in),
      vx(static_cast<double>(x_in(dynamics::VX))),
      epsi(static_cast<double>(x_in(dynamics::EPSI))),
      s(static_cast<double>(x_in(dynamics::S))),
      ey(static_cast<double>(x_in(dynamics::EY))) {}

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

// The query side of normalized_distance_sq below, precomputed once per
// trajectory_segment() call rather than once per candidate.
// inv_scale_* are 1/scale, so the hot loop multiplies instead of divides.
struct DistanceQuery {
  double vx;
  double epsi;
  double s;
  double ey;
  double inv_scale_vx;
  double inv_scale_epsi;
  double inv_scale_s;
  double inv_scale_ey;
};

DistanceQuery make_distance_query(const casadi::DM &x_query,
                                  const casadi::DM &state_scale) {
  return DistanceQuery{
      static_cast<double>(x_query(dynamics::VX)),
      static_cast<double>(x_query(dynamics::EPSI)),
      static_cast<double>(x_query(dynamics::S)),
      static_cast<double>(x_query(dynamics::EY)),
      1.0 / static_cast<double>(state_scale(dynamics::VX)),
      1.0 / static_cast<double>(state_scale(dynamics::EPSI)),
      1.0 / static_cast<double>(state_scale(dynamics::S)),
      1.0 / static_cast<double>(state_scale(dynamics::EY)),
  };
}

// Plain-double distance from `sample` to a precomputed warm-start query.
double normalized_distance_sq(const SafeSetSample &sample, double s_shift,
                              const DistanceQuery &query) {
  const double dvx = (sample.vx - query.vx) * query.inv_scale_vx;
  const double depsi = (sample.epsi - query.epsi) * query.inv_scale_epsi;
  const double ds = (sample.s + s_shift - query.s) * query.inv_scale_s;
  const double dey = (sample.ey - query.ey) * query.inv_scale_ey;
  return dvx * dvx + depsi * depsi + ds * ds + dey * dey;
}

long positive_mod(long value, long modulus) {
  const long remainder = value % modulus;
  return remainder < 0 ? remainder + modulus : remainder;
}

long floor_div(long value, long divisor) {
  const long quotient = value / divisor;
  const long remainder = value % divisor;
  return remainder < 0 ? quotient - 1 : quotient;
}

double periodic_delta(double value, double reference, double period) {
  const double delta = value - reference;
  return delta - period * std::round(delta / period);
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
    if (laps[lap_idx].size() - 1 < static_cast<std::size_t>(K)) {
      throw std::runtime_error("SafeSet::terminal_point_count: lap " +
                               std::to_string(lap_idx) +
                               " contains fewer than K samples");
    }
  }
  return K * num_laps();
}

SafeSet::QueryResult SafeSet::query_local_segments(double s_query,
                                                   casadi_int K) const {
  const casadi_int q = terminal_point_count(K);
  QueryResult result{
      casadi::DM::zeros(dynamics::kStateDim, q), casadi::DM::zeros(q, 1), {}};
  result.selected.reserve(static_cast<std::size_t>(q));
  casadi_int output_col = 0;

  for (std::size_t lap_index = 0; lap_index < laps.size(); ++lap_index) {
    const std::vector<SafeSetSample> &lap = laps[lap_index];
    // The canonical terminal phase view is x_1..x_T. phase_index therefore
    // maps to stored sample phase_index+1.
    const long period = static_cast<long>(lap.size()) - 1;
    long nearest = 0;
    double best_distance =
        std::abs(periodic_delta(lap[1].s, s_query, track_length));
    for (long phase_index = 1; phase_index < period; ++phase_index) {
      const double distance = std::abs(
          periodic_delta(lap[static_cast<std::size_t>(phase_index + 1)].s,
                         s_query, track_length));
      if (distance < best_distance) {
        best_distance = distance;
        nearest = phase_index;
      }
    }

    const SafeSetSample &nearest_sample =
        lap[static_cast<std::size_t>(nearest + 1)];
    const long base_cycle = static_cast<long>(
        std::llround((s_query - nearest_sample.s) / track_length));
    const long start = nearest - (K - 1) / 2;

    struct SelectedPoint {
      const SafeSetSample *sample;
      long sample_index;
      long cycle;
      double lifted_s;
      double extended_J;
    };
    std::vector<SelectedPoint> selected;
    selected.reserve(static_cast<std::size_t>(K));
    for (casadi_int j = 0; j < K; ++j) {
      const long raw_index = start + j;
      const long phase_index = positive_mod(raw_index, period);
      const long relative_wrap = floor_div(raw_index, period);
      const long sample_index = phase_index + 1;
      const long cycle = base_cycle + relative_wrap;
      const SafeSetSample &sample = lap[static_cast<std::size_t>(sample_index)];
      selected.push_back(
          SelectedPoint{&sample, sample_index, cycle,
                        sample.s + static_cast<double>(cycle) * track_length,
                        sample.J - static_cast<double>(cycle * period)});
    }

    const double endpoint_cost = selected.back().extended_J;
    for (const SelectedPoint &point : selected) {
      const double local_J = point.extended_J - endpoint_cost;
      const SafeSetSample &sample = *point.sample;
      casadi::DM x_shifted = sample.x;
      x_shifted(dynamics::S) = point.lifted_s;
      result.X_ss(casadi::Slice(), output_col) = x_shifted;
      result.J_ss(output_col, 0) = local_J;
      result.selected.push_back(QueryResult::SelectedPointInfo{
          lap_index, point.sample_index, point.cycle, point.lifted_s, local_J});
      ++output_col;
    }
  }

  if (output_col != q || result.X_ss.size1() != dynamics::kStateDim ||
      result.X_ss.size2() != q || result.J_ss.size1() != q ||
      result.J_ss.size2() != 1 || !result.X_ss.is_regular() ||
      !result.J_ss.is_regular()) {
    throw std::runtime_error(
        "SafeSet::query_local_segments: constructed terminal matrices have "
        "invalid dimensions or values");
  }
  return result;
}

SafeSet::TrajectorySegment
SafeSet::trajectory_segment(const casadi::DM &x_query, casadi_int horizon_steps,
                            const casadi::DM &state_scale) const {
  using casadi::Slice;

  const std::vector<SafeSetSample> &lap = laps.back();
  const std::size_t n = lap.size();

  const DistanceQuery dq = make_distance_query(x_query, state_scale);
  std::size_t nearest = 0;
  double best = normalized_distance_sq(lap[0], 0.0, dq);
  for (std::size_t i = 1; i < n; ++i) {
    const double d = normalized_distance_sq(lap[i], 0.0, dq);
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
