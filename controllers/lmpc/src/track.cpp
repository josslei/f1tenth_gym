#include "track.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace lmpc {

namespace {

double wrap_angle(double angle) {
  return std::atan2(std::sin(angle), std::cos(angle));
}

// Parses "x_m, y_m, w_tr_right_m, w_tr_left_m" rows, skipping the leading
// "#"-commented header -- the same format scripts/generate_centerline.py
// writes and scripts/lmpc_collect_seed_lap.py reads with np.loadtxt.
std::vector<std::pair<double, double>> load_xy(const std::string &csv_path) {
  std::ifstream file(csv_path);
  if (!file.is_open()) {
    throw std::runtime_error("Track: could not open centerline CSV: " +
                             csv_path);
  }

  std::vector<std::pair<double, double>> xy;
  std::string line;
  while (std::getline(file, line)) {
    if (line.empty() || line[0] == '#') {
      continue;
    }
    std::replace(line.begin(), line.end(), ',', ' ');
    std::istringstream iss(line);
    double x, y;
    if (!(iss >> x >> y)) {
      continue;
    }
    xy.emplace_back(x, y);
  }
  if (xy.size() < 2) {
    throw std::runtime_error("Track: centerline CSV has fewer than 2 points: " +
                             csv_path);
  }
  return xy;
}

} // namespace

Track::Track(const std::string &centerline_csv_path) {
  const std::vector<std::pair<double, double>> xy =
      load_xy(centerline_csv_path);
  const std::size_t n = xy.size();

  // Open (non-periodic) cumulative arclength -- matches
  // utils/waypoint_utils.cumulative_arc_lengths exactly.
  s_.resize(n);
  s_[0] = 0.0;
  for (std::size_t i = 1; i < n; ++i) {
    const double dx = xy[i].first - xy[i - 1].first;
    const double dy = xy[i].second - xy[i - 1].second;
    s_[i] = s_[i - 1] + std::sqrt(dx * dx + dy * dy);
  }

  // Per-segment length, PERIODIC (size n: seg_len[i] is the distance from
  // point i to point (i+1)%n) -- seg_len[n-1] is the closing segment back to
  // point 0, which s_ itself does not cover (s_ stays open/non-periodic).
  // Needed below to place each segment heading at its true midpoint arc
  // length, including the two segments that straddle the seam at i=0.
  std::vector<double> seg_len(n);
  for (std::size_t i = 0; i + 1 < n; ++i) {
    seg_len[i] = s_[i + 1] - s_[i];
  }
  {
    const double dx = xy[0].first - xy[n - 1].first;
    const double dy = xy[0].second - xy[n - 1].second;
    seg_len[n - 1] = std::sqrt(dx * dx + dy * dy);
  }
  track_length_ = s_.back() + seg_len.back();

  // Forward-difference heading at every sample, wrapping the last sample to
  // the first -- matches load_centerline_waypoints()'s use of np.roll.
  // heading[i] is the tangent of segment i (points i -> (i+1)%n), so it sits
  // at that segment's MIDPOINT arc length, not at s_[i] itself.
  std::vector<double> heading(n);
  for (std::size_t i = 0; i < n; ++i) {
    const std::size_t next = (i + 1) % n;
    const double dx = xy[next].first - xy[i].first;
    const double dy = xy[next].second - xy[i].second;
    heading[i] = std::atan2(dy, dx);
  }

  // Discrete curvature at each sample from the two segment headings
  // straddling it (periodic in the heading/segment indices, even though s_
  // itself is open): kappa_[i] = wrap(heading[i] - heading[i-1]) / ds, where
  // ds is the arc length BETWEEN the two segments' midpoints, i.e.
  // 0.5*(seg_len[i-1] + seg_len[i]) -- NOT seg_len[i-1] + seg_len[i] (a
  // previous version used the full s_[i+1]-s_[i-1] span as the denominator,
  // which is exactly 2x too large and understated every curvature by half;
  // it also hardcoded kappa_[0] = 0 instead of differencing across the seam,
  // flattening the start/finish straight's curvature). i=0 and i=n-1 wrap
  // through the closing segment (heading[n-1]/seg_len[n-1]) so the seam gets
  // a real finite difference like every other sample.
  kappa_.resize(n);
  for (std::size_t i = 0; i < n; ++i) {
    const std::size_t prev = (i == 0) ? (n - 1) : (i - 1);
    const double dtheta = wrap_angle(heading[i] - heading[prev]);
    const double ds = 0.5 * (seg_len[prev] + seg_len[i]);
    kappa_[i] = (ds > 1e-9) ? (dtheta / ds) : 0.0;
  }
}

double Track::curvature(double s) const {
  // Nonnegative modulo: fmod alone can return a negative result for
  // negative s (e.g. a small reverse prediction just before the finish
  // line wrapping to just before s=length, not to a negative value).
  double s_periodic = std::fmod(s, track_length_);
  if (s_periodic < 0.0) {
    s_periodic += track_length_;
  }

  if (s_periodic >= s_.back()) {
    const double span = track_length_ - s_.back();
    const double t = (s_periodic - s_.back()) / span;
    return kappa_.back() + t * (kappa_.front() - kappa_.back());
  }

  // Binary search for the bracketing segment, then linearly interpolate.
  const auto it = std::upper_bound(s_.begin(), s_.end(), s_periodic);
  std::size_t hi = static_cast<std::size_t>(std::distance(s_.begin(), it));
  hi = std::clamp(hi, std::size_t(1), s_.size() - 1);
  const std::size_t lo = hi - 1;

  const double span = s_[hi] - s_[lo];
  const double t = (span > 1e-9) ? (s_periodic - s_[lo]) / span : 0.0;
  return kappa_[lo] + t * (kappa_[hi] - kappa_[lo]);
}

} // namespace lmpc
