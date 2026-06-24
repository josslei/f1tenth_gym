#ifndef F110_ROLLOUT_KERNEL_F110_REWARD_HPP_
#define F110_ROLLOUT_KERNEL_F110_REWARD_HPP_

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <utility>
#include <vector>

#include "f110_track_map.hpp"

namespace f110_gym {

class F110ProgressReward {
public:
  F110ProgressReward() = default;

  F110ProgressReward(const f110_rollout_kernel::TrackMap &track_map,
                     const std::vector<double> &waypoints_x,
                     const std::vector<double> &waypoints_y,
                     double q_s_progress = 1.0, double q_s_alpha = 1.0,
                     double q_s_smooth = 0.0, double terminal_penalty = 1000.0,
                     double alpha_th = 0.0, double slip_terminal_penalty = 0.0,
                     double q_offtrack_grad = 0.0)
      : track_map_(&track_map), waypoints_x_(waypoints_x),
        waypoints_y_(waypoints_y), q_s_progress_(q_s_progress),
        q_s_alpha_(q_s_alpha), q_s_smooth_(q_s_smooth),
        terminal_penalty_(terminal_penalty), alpha_th_(alpha_th),
        slip_terminal_penalty_(slip_terminal_penalty),
        q_offtrack_grad_(q_offtrack_grad) {
    build_arc();
  }

  F110ProgressReward(const std::vector<double> &waypoints_x,
                     const std::vector<double> &waypoints_y,
                     double q_s_progress = 1.0, double q_s_alpha = 1.0,
                     double q_s_smooth = 0.0, double terminal_penalty = 1000.0,
                     double alpha_th = 0.0, double slip_terminal_penalty = 0.0,
                     double q_offtrack_grad = 0.0)
      : waypoints_x_(waypoints_x), waypoints_y_(waypoints_y),
        q_s_progress_(q_s_progress), q_s_alpha_(q_s_alpha),
        q_s_smooth_(q_s_smooth), terminal_penalty_(terminal_penalty),
        alpha_th_(alpha_th), slip_terminal_penalty_(slip_terminal_penalty),
        q_offtrack_grad_(q_offtrack_grad) {
    build_arc();
  }

  void set_waypoints(const std::vector<double> &waypoints_x,
                     const std::vector<double> &waypoints_y) {
    waypoints_x_ = waypoints_x;
    waypoints_y_ = waypoints_y;
    reset();
    build_arc();
  }

  void reset() {
    prev_arc_length_ = 0.0;
    last_action_0_ = 0.0;
    last_action_1_ = 0.0;
    has_last_action_ = false;
  }

  double operator()(double px, double py, double theta, double vx, double vy,
                    double action_0, double action_1, bool collision,
                    bool terminated) {
    (void)terminated;

    const double current_arc = arclength_at(px, py);
    double delta_s = current_arc - prev_arc_length_;
    if (total_length_ > 0.0) {
      if (delta_s < -0.5 * total_length_) {
        delta_s += total_length_;
      } else if (delta_s > 0.5 * total_length_) {
        delta_s -= total_length_;
      }
    }

    const double beta = std::atan2(vy, vx);
    const double d0 = has_last_action_ ? (action_0 - last_action_0_) : 0.0;
    const double d1 = has_last_action_ ? (action_1 - last_action_1_) : 0.0;

    double reward = q_s_progress_ * delta_s;
    reward -= q_s_alpha_ * (beta * beta);
    reward -= q_s_smooth_ * (d0 * d0 + d1 * d1);

    if (track_map_ != nullptr && q_offtrack_grad_ > 0.0) {
      const double distance = track_map_->distance_at(px, py);
      const double off_real = std::clamp(1.0 - distance, 0.0, 1.0);
      reward -= q_offtrack_grad_ * off_real;
    } else if (collision && q_offtrack_grad_ > 0.0) {
      reward -= q_offtrack_grad_;
    }

    if (std::abs(beta) > alpha_th_) {
      reward -= slip_terminal_penalty_;
    }

    if (std::hypot(vx, vy) > 12.0) {
      reward -= 10000.0;
    }

    if (is_backward_terminal(px, py, theta)) {
      reward -= terminal_penalty_;
    }

    prev_arc_length_ = current_arc;
    last_action_0_ = action_0;
    last_action_1_ = action_1;
    has_last_action_ = true;

    if (terminated) {
      reset();
    }

    return reward;
  }

  bool is_terminal(double px, double py, double theta, bool collision) const {
    return collision || is_backward_terminal(px, py, theta);
  }

private:
  void build_arc() {
    const std::size_t n = waypoints_x_.size();
    if (n == 0) {
      cum_arc_lengths_.assign(1, 0.0);
      headings_.clear();
      total_length_ = 0.0;
      return;
    }

    cum_arc_lengths_.resize(n);
    headings_.resize(n);
    cum_arc_lengths_[0] = 0.0;
    for (std::size_t i = 1; i < n; ++i) {
      const double dx = waypoints_x_[i] - waypoints_x_[i - 1];
      const double dy = waypoints_y_[i] - waypoints_y_[i - 1];
      cum_arc_lengths_[i] =
          cum_arc_lengths_[i - 1] + std::sqrt(dx * dx + dy * dy);
    }
    total_length_ = cum_arc_lengths_.back();

    for (std::size_t i = 0; i < n; ++i) {
      const std::size_t prev = (i == 0) ? (n - 1) : (i - 1);
      const std::size_t next = (i + 1) % n;
      const double dx = waypoints_x_[next] - waypoints_x_[prev];
      const double dy = waypoints_y_[next] - waypoints_y_[prev];
      headings_[i] = std::atan2(dy, dx);
    }
  }

  std::size_t nearest_index(double px, double py) const {
    std::size_t nearest_idx = 0;
    double best_dist_sq = std::numeric_limits<double>::max();
    for (std::size_t i = 0; i < waypoints_x_.size(); ++i) {
      const double dx = waypoints_x_[i] - px;
      const double dy = waypoints_y_[i] - py;
      const double dist_sq = dx * dx + dy * dy;
      if (dist_sq < best_dist_sq) {
        best_dist_sq = dist_sq;
        nearest_idx = i;
      }
    }
    return nearest_idx;
  }

  bool is_backward_terminal(double px, double py, double theta) const {
    if (waypoints_x_.empty()) {
      return false;
    }
    const double ref_heading = nearest_heading(px, py);
    double hdiff = std::abs(theta - ref_heading);
    if (hdiff > M_PI) {
      hdiff = TWO_PI - hdiff;
    }
    return hdiff > M_PI / 2.0;
  }

  double nearest_heading(double px, double py) const {
    return headings_[nearest_index(px, py)];
  }

  double arclength_at(double px, double py) const {
    if (waypoints_x_.size() < 2) {
      return 0.0;
    }

    const std::size_t idx = nearest_index(px, py);
    const std::size_t n = waypoints_x_.size();
    const std::size_t prev = (idx == 0) ? (n - 1) : (idx - 1);
    const std::size_t next = (idx + 1) % n;

    auto project = [&](std::size_t a,
                       std::size_t b) -> std::pair<double, double> {
      const double sx = waypoints_x_[a];
      const double sy = waypoints_y_[a];
      const double ex = waypoints_x_[b];
      const double ey = waypoints_y_[b];
      const double dx = ex - sx;
      const double dy = ey - sy;
      const double len2 = dx * dx + dy * dy;
      if (len2 < 1e-12) {
        const double rx = px - sx;
        const double ry = py - sy;
        return {0.0, rx * rx + ry * ry};
      }
      double t = ((px - sx) * dx + (py - sy) * dy) / len2;
      t = std::clamp(t, 0.0, 1.0);
      const double proj_x = sx + t * dx;
      const double proj_y = sy + t * dy;
      const double rx = px - proj_x;
      const double ry = py - proj_y;
      return {t * std::sqrt(len2), rx * rx + ry * ry};
    };

    const std::pair<double, double> p_prev = project(prev, idx);
    const std::pair<double, double> p_next = project(idx, next);
    if (p_prev.second < p_next.second) {
      return cum_arc_lengths_[prev] + p_prev.first;
    }
    return cum_arc_lengths_[idx] + p_next.first;
  }

  std::vector<double> waypoints_x_;
  std::vector<double> waypoints_y_;
  std::vector<double> cum_arc_lengths_;
  std::vector<double> headings_;
  double total_length_ = 0.0;

  const f110_rollout_kernel::TrackMap *track_map_ = nullptr;
  double q_s_progress_ = 1.0;
  double q_s_alpha_ = 1.0;
  double q_s_smooth_ = 0.0;
  double terminal_penalty_ = 1000.0;
  double alpha_th_ = 0.0;
  double slip_terminal_penalty_ = 0.0;
  double q_offtrack_grad_ = 0.0;

  double prev_arc_length_ = 0.0;
  double last_action_0_ = 0.0;
  double last_action_1_ = 0.0;
  bool has_last_action_ = false;

  static constexpr double TWO_PI = 2.0 * M_PI;
};

} // namespace f110_gym

#endif // F110_ROLLOUT_KERNEL_F110_REWARD_HPP_
