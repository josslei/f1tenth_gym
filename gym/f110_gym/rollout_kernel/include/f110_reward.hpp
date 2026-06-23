#ifndef F110_ROLLOUT_KERNEL_F110_REWARD_HPP_
#define F110_ROLLOUT_KERNEL_F110_REWARD_HPP_

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <utility>
#include <vector>

namespace f110_gym {

class F110ProgressReward {
public:
  F110ProgressReward() = default;

  F110ProgressReward(const std::vector<double> &waypoints_x,
                     const std::vector<double> &waypoints_y,
                     double speed_reward_weight = 0.1,
                     double progress_weight = 2.0,
                     double steer_smoothness_weight = 0.5,
                     double collision_penalty = 50.0,
                     double spin_threshold = 100.0)
      : waypoints_x_(waypoints_x), waypoints_y_(waypoints_y),
        speed_reward_weight_(speed_reward_weight),
        progress_weight_(progress_weight),
        steer_smoothness_weight_(steer_smoothness_weight),
        collision_penalty_(collision_penalty), spin_threshold_(spin_threshold),
        prev_arc_length_(0.0), prev_steer_(0.0) {
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
    prev_steer_ = 0.0;
  }

  double operator()(double px, double py, double theta, double vx, double vy,
                    double steer, bool collision, bool terminated) {
    if (collision || std::abs(theta) > spin_threshold_) {
      if (terminated) {
        prev_arc_length_ = 0.0;
        prev_steer_ = 0.0;
      }
      return -collision_penalty_;
    }

    if (terminated) {
      prev_arc_length_ = 0.0;
      prev_steer_ = 0.0;
    }

    double vel_magnitude = std::sqrt(vx * vx + vy * vy);
    double reward = speed_reward_weight_ * vel_magnitude;

    std::size_t nearest_idx = 0;
    double best_dist_sq = std::numeric_limits<double>::max();
    for (std::size_t i = 0; i < waypoints_x_.size(); ++i) {
      double dx = waypoints_x_[i] - px;
      double dy = waypoints_y_[i] - py;
      double dist_sq = dx * dx + dy * dy;
      if (dist_sq < best_dist_sq) {
        best_dist_sq = dist_sq;
        nearest_idx = i;
      }
    }

    double current_arc = cum_arc_lengths_[nearest_idx];
    double progress = current_arc - prev_arc_length_;
    if (progress < 0.0) {
      progress = (total_length_ - prev_arc_length_) + current_arc;
    }
    reward += progress_weight_ * progress;
    prev_arc_length_ = current_arc;

    double steer_delta = std::abs(steer - prev_steer_);
    reward -= steer_smoothness_weight_ * steer_delta;
    prev_steer_ = steer;

    return std::clamp(reward, -5.0, 8.0);
  }

private:
  void build_arc() {
    std::size_t n = waypoints_x_.size();
    cum_arc_lengths_.resize(n);
    cum_arc_lengths_[0] = 0.0;
    for (std::size_t i = 1; i < n; ++i) {
      double dx = waypoints_x_[i] - waypoints_x_[i - 1];
      double dy = waypoints_y_[i] - waypoints_y_[i - 1];
      cum_arc_lengths_[i] =
          cum_arc_lengths_[i - 1] + std::sqrt(dx * dx + dy * dy);
    }
    total_length_ = n > 0 ? cum_arc_lengths_.back() : 0.0;
  }

  std::vector<double> waypoints_x_;
  std::vector<double> waypoints_y_;
  std::vector<double> cum_arc_lengths_;
  double total_length_ = 0.0;

  double speed_reward_weight_ = 0.1;
  double progress_weight_ = 2.0;
  double steer_smoothness_weight_ = 0.5;
  double collision_penalty_ = 50.0;
  double spin_threshold_ = 100.0;

  double prev_arc_length_ = 0.0;
  double prev_steer_ = 0.0;
};

} // namespace f110_gym

#endif // F110_ROLLOUT_KERNEL_F110_REWARD_HPP_
