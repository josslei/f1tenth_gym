#include "lmpc/state.hpp"

#include <cmath>
#include <limits>
#include <utility>

namespace f110_gym_lmpc {
namespace {

double squared_distance(double ax, double ay, double bx, double by) {
  const double dx = ax - bx;
  const double dy = ay - by;
  return dx * dx + dy * dy;
}

} // namespace

std::array<double, 6> RacingLmpcState::to_array() const {
  return {s, e_y, e_psi, v_x, v_y, omega};
}

std::array<double, 6> PaperLmpcState::to_array() const {
  return {v_x, v_y, omega, e_psi, s, e_y};
}

double normalize_angle(double angle) {
  constexpr double pi = 3.141592653589793238462643383279502884;
  while (angle > pi) {
    angle -= 2.0 * pi;
  }
  while (angle <= -pi) {
    angle += 2.0 * pi;
  }
  return angle;
}

CenterlineTrack::CenterlineTrack(std::vector<double> x, std::vector<double> y,
                                 bool closed)
    : x_(std::move(x)), y_(std::move(y)), closed_(closed) {
  s_.resize(x_.size());
  for (std::size_t i = 1; i < x_.size(); ++i) {
    s_[i] = s_[i - 1] +
            std::sqrt(squared_distance(x_[i], y_[i], x_[i - 1], y_[i - 1]));
  }
  total_length_ = s_.back();
  if (closed_) {
    total_length_ += std::sqrt(
        squared_distance(x_.front(), y_.front(), x_.back(), y_.back()));
  }
}

FrenetProjection CenterlineTrack::project(double x, double y) const {
  double best_dist2 = std::numeric_limits<double>::infinity();
  FrenetProjection best;
  const std::size_t segment_count = closed_ ? x_.size() : x_.size() - 1;

  for (std::size_t i = 0; i < segment_count; ++i) {
    const std::size_t j = (i + 1) % x_.size();
    const double ax = x_[i];
    const double ay = y_[i];
    const double bx = x_[j];
    const double by = y_[j];
    const double vx = bx - ax;
    const double vy = by - ay;
    const double seg_len2 = vx * vx + vy * vy;
    double t = ((x - ax) * vx + (y - ay) * vy) / seg_len2;
    if (t < 0.0) {
      t = 0.0;
    } else if (t > 1.0) {
      t = 1.0;
    }

    const double proj_x = ax + t * vx;
    const double proj_y = ay + t * vy;
    const double dist2 = squared_distance(x, y, proj_x, proj_y);
    if (dist2 < best_dist2) {
      const double seg_len = std::sqrt(seg_len2);
      const double heading = std::atan2(vy, vx);
      const double normal_x = -vy / seg_len;
      const double normal_y = vx / seg_len;
      best_dist2 = dist2;
      best.s = s_[i] + t * seg_len;
      best.e_y = (x - proj_x) * normal_x + (y - proj_y) * normal_y;
      best.heading = heading;
      best.segment_index = i;
    }
  }

  if (best.s >= total_length_) {
    best.s -= total_length_;
  }
  return best;
}

RacingLmpcState
CenterlineTrack::to_racing_state(const GymVehicleState &state) const {
  const FrenetProjection projection = project(state.x, state.y);
  return RacingLmpcState{
      projection.s,
      projection.e_y,
      normalize_angle(state.yaw - projection.heading),
      state.v_x,
      state.v_y,
      state.omega,
  };
}

PaperLmpcState
CenterlineTrack::to_paper_state(const GymVehicleState &state) const {
  return racing_to_paper(to_racing_state(state));
}

double CenterlineTrack::total_length() const { return total_length_; }

const std::vector<double> &CenterlineTrack::s() const { return s_; }

PaperLmpcState racing_to_paper(const RacingLmpcState &state) {
  return PaperLmpcState{state.v_x,   state.v_y, state.omega,
                        state.e_psi, state.s,   state.e_y};
}

} // namespace f110_gym_lmpc
