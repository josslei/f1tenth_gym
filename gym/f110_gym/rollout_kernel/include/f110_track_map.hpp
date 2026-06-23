#ifndef F110_ROLLOUT_KERNEL_TRACK_MAP_HPP_
#define F110_ROLLOUT_KERNEL_TRACK_MAP_HPP_

#include <cmath>
#include <cstdint>
#include <vector>

namespace f110_rollout_kernel {

struct TrackMap {
  int height = 0;
  int width = 0;
  double resolution = 0.05;
  double orig_x = 0.0;
  double orig_y = 0.0;
  double orig_c = 1.0;
  double orig_s = 0.0;

  std::vector<double> dt;

  int theta_dis = 2000;
  int num_beams = 1080;
  double fov = 4.7;
  double max_range = 30.0;
  double eps = 0.0001;
  double theta_index_increment{};
  double angle_increment{};
  std::vector<double> sines;
  std::vector<double> cosines;

  std::vector<float> side_distances;
  double ttc_thresh = 0.15;

  void compute_scan_tables() {
    angle_increment = fov / static_cast<double>(num_beams - 1);
    theta_index_increment =
        static_cast<double>(theta_dis) * angle_increment / 6.283185307179586;

    sines.resize(static_cast<std::size_t>(theta_dis));
    cosines.resize(static_cast<std::size_t>(theta_dis));
    for (int i = 0; i < theta_dis; ++i) {
      double angle = static_cast<double>(i) * 6.283185307179586 /
                     static_cast<double>(theta_dis - 1);
      sines[static_cast<std::size_t>(i)] = std::sin(angle);
      cosines[static_cast<std::size_t>(i)] = std::cos(angle);
    }

    side_distances.assign(static_cast<std::size_t>(num_beams), 0.0f);
  }

  bool xy_to_rc(float x, float y, int &r, int &c) const {
    float tx = x - orig_x;
    float ty = y - orig_y;
    float rx = tx * orig_c + ty * orig_s;
    float ry = -tx * orig_s + ty * orig_c;
    c = static_cast<int>(std::floor(rx / resolution));
    r = static_cast<int>(std::floor(ry / resolution));
    if (r < 0 || r >= height || c < 0 || c >= width) {
      r = -1;
      c = -1;
      return false;
    }
    return true;
  }

  float distance_at(float x, float y) const {
    int r = 0, c = 0;
    if (!xy_to_rc(x, y, r, c)) {
      return 0.0f;
    }
    return static_cast<float>(
        dt[static_cast<std::size_t>(r) * static_cast<std::size_t>(width) +
           static_cast<std::size_t>(c)]);
  }
};

} // namespace f110_rollout_kernel

#endif // F110_ROLLOUT_KERNEL_TRACK_MAP_HPP_
