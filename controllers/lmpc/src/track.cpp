#include "track.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace lmpc
{

namespace
{

double wrap_angle(double angle)
{
  return std::atan2(std::sin(angle), std::cos(angle));
}

// Parses "x_m, y_m, w_tr_right_m, w_tr_left_m" rows, skipping the leading
// "#"-commented header -- the same format scripts/generate_centerline.py
// writes and scripts/lmpc_collect_seed_lap.py reads with np.loadtxt.
std::vector<std::pair<double, double>> load_xy(const std::string & csv_path)
{
  std::ifstream file(csv_path);
  if (!file.is_open()) {
    throw std::runtime_error("Track: could not open centerline CSV: " + csv_path);
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
    throw std::runtime_error("Track: centerline CSV has fewer than 2 points: " + csv_path);
  }
  return xy;
}

}  // namespace

Track::Track(const std::string & centerline_csv_path)
{
  const std::vector<std::pair<double, double>> xy = load_xy(centerline_csv_path);
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

  // Forward-difference heading at every sample, wrapping the last sample to
  // the first -- matches load_centerline_waypoints()'s use of
  // np.roll(xy, -1): the CSV describes one closed loop, so this final
  // segment is the true closing tangent.
  std::vector<double> heading(n);
  for (std::size_t i = 0; i < n; ++i) {
    const std::size_t next = (i + 1) % n;
    const double dx = xy[next].first - xy[i].first;
    const double dy = xy[next].second - xy[i].second;
    heading[i] = std::atan2(dy, dx);
  }

  // Discrete curvature at each sample: turn rate between the incoming and
  // outgoing heading, divided by the local arclength step. Endpoints reuse
  // their single available neighbor (one-sided difference) since s_ itself
  // is not periodic (class comment in track.hpp).
  kappa_.resize(n);
  for (std::size_t i = 0; i < n; ++i) {
    const std::size_t prev = (i == 0) ? 0 : i - 1;
    const std::size_t next = (i + 1) % n;
    const double dtheta = wrap_angle(heading[i] - heading[prev]);
    const double ds = (i == 0) ? (s_[1] - s_[0]) : (s_[std::min(i + 1, n - 1)] - s_[prev]);
    kappa_[i] = (ds > 1e-9) ? (dtheta / ds) : 0.0;
    (void)next;
  }
}

double Track::curvature(double s) const
{
  s = std::clamp(s, s_.front(), s_.back());

  // Binary search for the bracketing segment, then linearly interpolate.
  const auto it = std::upper_bound(s_.begin(), s_.end(), s);
  std::size_t hi = static_cast<std::size_t>(std::distance(s_.begin(), it));
  hi = std::clamp(hi, std::size_t(1), s_.size() - 1);
  const std::size_t lo = hi - 1;

  const double span = s_[hi] - s_[lo];
  const double t = (span > 1e-9) ? (s - s_[lo]) / span : 0.0;
  return kappa_[lo] + t * (kappa_[hi] - kappa_[lo]);
}

}  // namespace lmpc
