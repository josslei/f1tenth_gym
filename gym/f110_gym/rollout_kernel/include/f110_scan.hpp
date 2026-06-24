#ifndef F110_ROLLOUT_KERNEL_SCAN_HPP_
#define F110_ROLLOUT_KERNEL_SCAN_HPP_

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

#include "f110_track_map.hpp"
#include "simd.hpp"

namespace f110_rollout_kernel {

inline void xy_to_rc(double x, double y, double orig_x, double orig_y,
                     double orig_c, double orig_s, double resolution,
                     int height, int width, int &r, int &c) {
  double tx = x - orig_x;
  double ty = y - orig_y;
  double rx = tx * orig_c + ty * orig_s;
  double ry = -tx * orig_s + ty * orig_c;
  c = static_cast<int>(std::floor(rx / resolution));
  r = static_cast<int>(std::floor(ry / resolution));
  if (r < 0 || r >= height || c < 0 || c >= width) {
    r = -1;
    c = -1;
  }
}

inline double dt_lookup(double x, double y, const TrackMap &map) {
  int r = 0, c = 0;
  xy_to_rc(static_cast<double>(x), static_cast<double>(y),
           static_cast<double>(map.orig_x), static_cast<double>(map.orig_y),
           static_cast<double>(map.orig_c), static_cast<double>(map.orig_s),
           static_cast<double>(map.resolution), map.height, map.width, r, c);
  if (r < 0 || c < 0) {
    return static_cast<double>(map.max_range);
  }
  return map
      .dt[static_cast<std::size_t>(r) * static_cast<std::size_t>(map.width) +
          static_cast<std::size_t>(c)];
}

inline float trace_ray(double x, double y, double theta_index,
                       const TrackMap &map) {
  int ti = static_cast<int>(std::floor(theta_index)) % map.theta_dis;
  if (ti < 0) {
    ti += map.theta_dis;
  }
  double dx = map.cosines[static_cast<std::size_t>(ti)];
  double dy = map.sines[static_cast<std::size_t>(ti)];

  double dist = dt_lookup(x, y, map);
  double total_dist = dist;

  while (dist > map.eps && total_dist <= map.max_range) {
    x += dx * dist;
    y += dy * dist;
    dist = dt_lookup(x, y, map);
    total_dist += dist;
  }

  if (total_dist > map.max_range) {
    total_dist = map.max_range;
  }

  return static_cast<float>(total_dist);
}

inline void get_scan_one(double px, double py, double theta,
                         const TrackMap &map, float *out) {
  double theta_to_index =
      static_cast<double>(map.theta_dis) / 6.283185307179586;
  double theta_start = theta * theta_to_index;
  double beam_span = theta_to_index * static_cast<double>(map.fov);
  double theta_i = theta_start - beam_span * 0.5;
  theta_i = std::fmod(theta_i, static_cast<double>(map.theta_dis));
  while (theta_i < 0.0) {
    theta_i += static_cast<double>(map.theta_dis);
  }

  for (int i = 0; i < map.num_beams; ++i) {
    out[i] = trace_ray(px, py, theta_i, map);
    theta_i += map.theta_index_increment;
    while (theta_i >= static_cast<double>(map.theta_dis)) {
      theta_i -= static_cast<double>(map.theta_dis);
    }
  }
}

inline void get_scan_batch(const double *poses, int B, const TrackMap &map,
                           float *out) {
  for_batch(B, [&](int start, int count) {
    for (int b = start; b < start + count; ++b) {
      get_scan_one(poses[b * 3], poses[b * 3 + 1], poses[b * 3 + 2], map,
                   out + b * map.num_beams);
    }
  });
}

inline bool check_ttc_one(const float *scan, double vel, const TrackMap &map) {
  if (std::fabs(vel) < 0.01) {
    return false;
  }
  for (int i = 0; i < map.num_beams; ++i) {
    const double beam_angle =
        -0.5 * map.fov + static_cast<double>(i) * map.angle_increment;
    float proj_vel = static_cast<float>(vel * std::cos(beam_angle));
    if (std::fabs(proj_vel) < 0.01f) {
      continue;
    }
    float ttc =
        (scan[i] - map.side_distances[static_cast<std::size_t>(i)]) / proj_vel;
    if (ttc >= 0.0f && ttc < map.ttc_thresh) {
      return true;
    }
  }
  return false;
}

inline void check_ttc_batch(const float *scans, const double *vels, int B,
                            const TrackMap &map, bool *out) {
  for_batch(B, [&](int start, int count) {
    for (int b = start; b < start + count; ++b) {
      out[b] = check_ttc_one(scans + b * map.num_beams, vels[b], map);
    }
  });
}

} // namespace f110_rollout_kernel

#endif // F110_ROLLOUT_KERNEL_SCAN_HPP_
