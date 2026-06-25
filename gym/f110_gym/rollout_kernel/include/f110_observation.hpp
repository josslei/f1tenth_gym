#ifndef F110_ROLLOUT_KERNEL_OBSERVATION_HPP_
#define F110_ROLLOUT_KERNEL_OBSERVATION_HPP_

#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

#include "simd.hpp"

namespace f110_rollout_kernel {

struct ObservationConfig {
  int scan_size = 108;
  float scan_max_m = 30.0f;
  bool include_ego_state = true;
  float speed_scale = 8.0f;
  float yaw_rate_scale = 10.0f;
  float steer_scale = 1.066f;
  bool include_waypoints = false;
  std::vector<float> lookahead_distances;
  float waypoint_scale = 30.0f;
  float waypoint_resample_spacing = 0.5f;
};

inline float clip_float(float value, float lo, float hi) {
  if (value < lo) {
    return lo;
  }
  if (value > hi) {
    return hi;
  }
  return value;
}

inline int observation_dim(const ObservationConfig &config) {
  int dim = config.scan_size;
  if (config.include_ego_state) {
    dim += 7;
  }
  if (config.include_waypoints) {
    dim += static_cast<int>(config.lookahead_distances.size()) * 2;
  }
  dim += 2;
  return dim;
}

inline int positive_mod(int value, int modulus) {
  int result = value % modulus;
  return result < 0 ? result + modulus : result;
}

inline int nearest_waypoint_index_default(const double *waypoints_x,
                                          const double *waypoints_y,
                                          int num_waypoints, double px,
                                          double py) {
  constexpr int search_window = 200;
  constexpr int start_idx = -1;
  int best_idx = 0;
  double best_dist_sq = std::numeric_limits<double>::infinity();

  if (start_idx < 0 || search_window <= 0 || search_window >= num_waypoints) {
    for (int idx = 0; idx < num_waypoints; ++idx) {
      double dx = waypoints_x[idx] - px;
      double dy = waypoints_y[idx] - py;
      double d2 = dx * dx + dy * dy;
      if (d2 < best_dist_sq) {
        best_dist_sq = d2;
        best_idx = idx;
      }
    }
    return best_idx;
  }

  int half_window = search_window / 2;
  int first_offset = -half_window;
  int last_offset = search_window - half_window;
  best_idx = positive_mod(start_idx, num_waypoints);
  for (int offset = first_offset; offset < last_offset; ++offset) {
    int idx = positive_mod(start_idx + offset, num_waypoints);
    double dx = waypoints_x[idx] - px;
    double dy = waypoints_y[idx] - py;
    double d2 = dx * dx + dy * dy;
    if (d2 < best_dist_sq) {
      best_dist_sq = d2;
      best_idx = idx;
    }
  }
  return best_idx;
}

inline void build_observation_one(const float *scan_1080, double px, double py,
                                  double theta, double vx, double vy,
                                  double yaw_rate, double steer,
                                  uint8_t collision, const float *prev_action,
                                  const double *waypoints_x,
                                  const double *waypoints_y, int num_waypoints,
                                  const double *cum_arc_lengths,
                                  const ObservationConfig &config, float *out) {
  int idx = 0;

  for (int i = 0; i < config.scan_size; ++i) {
    int scan_idx = static_cast<int>(static_cast<double>(i) * 1079.0 /
                                    static_cast<double>(config.scan_size - 1));
    float s = scan_1080[static_cast<std::size_t>(scan_idx)];
    if (s < 0.0f) {
      s = 0.0f;
    }
    if (s > config.scan_max_m) {
      s = config.scan_max_m;
    }
    out[idx++] = s / config.scan_max_m;
  }

  if (config.include_ego_state) {
    out[idx++] = clip_float(
        static_cast<float>(vx / static_cast<double>(config.speed_scale)), -1.0f,
        1.0f);
    out[idx++] = clip_float(
        static_cast<float>(vy / static_cast<double>(config.speed_scale)), -1.0f,
        1.0f);
    out[idx++] =
        clip_float(static_cast<float>(
                       yaw_rate / static_cast<double>(config.yaw_rate_scale)),
                   -1.0f, 1.0f);
    out[idx++] = clip_float(
        static_cast<float>(steer / static_cast<double>(config.steer_scale)),
        -1.0f, 1.0f);
    out[idx++] = static_cast<float>(std::sin(theta));
    out[idx++] = static_cast<float>(std::cos(theta));
    out[idx++] = collision ? 1.0f : 0.0f;
  }

  if (config.include_waypoints && num_waypoints > 0) {
    double cos_t = std::cos(theta);
    double sin_t = std::sin(theta);
    int nearest_idx = nearest_waypoint_index_default(waypoints_x, waypoints_y,
                                                     num_waypoints, px, py);

    for (std::size_t di = 0; di < config.lookahead_distances.size(); ++di) {
      float lookahead = config.lookahead_distances[di];

      int offset = static_cast<int>(
          std::round(lookahead / config.waypoint_resample_spacing));
      if (offset < 1) {
        offset = 1;
      }
      int target_idx = (nearest_idx + offset) % num_waypoints;
      double best_target_x = waypoints_x[target_idx];
      double best_target_y = waypoints_y[target_idx];

      double dx = best_target_x - px;
      double dy = best_target_y - py;

      float local_dx = static_cast<float>(cos_t * dx + sin_t * dy);
      float local_dy = static_cast<float>(-sin_t * dx + cos_t * dy);

      out[idx++] = clip_float(local_dx / config.waypoint_scale, -1.0f, 1.0f);
      out[idx++] = clip_float(local_dy / config.waypoint_scale, -1.0f, 1.0f);
    }
  }

  out[idx++] = prev_action[0];
  out[idx++] = prev_action[1];
}

inline void build_observation_batch(
    const float *scans, const double *poses_x, const double *poses_y,
    const double *poses_theta, const double *vx, const double *vy,
    const double *yaw_rate, const double *steer, const uint8_t *collision,
    const float *prev_actions, const double *waypoints_x,
    const double *waypoints_y, int num_waypoints, const double *cum_arc_lengths,
    int B, const ObservationConfig &config, float *out) {
  int obs_dim = observation_dim(config);
  for_batch(B, [&](int start, int count) {
    for (int b = start; b < start + count; ++b) {
      build_observation_one(scans + b * 1080, poses_x[b], poses_y[b],
                            poses_theta[b], vx[b], vy[b], yaw_rate[b], steer[b],
                            collision[b], prev_actions + b * 2, waypoints_x,
                            waypoints_y, num_waypoints, cum_arc_lengths, config,
                            out + b * obs_dim);
    }
  });
}

} // namespace f110_rollout_kernel

#endif // F110_ROLLOUT_KERNEL_OBSERVATION_HPP_
