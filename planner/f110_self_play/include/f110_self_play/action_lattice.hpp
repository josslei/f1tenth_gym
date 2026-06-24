#ifndef PLANNER_F110_SELF_PLAY_ACTION_LATTICE_HPP_
#define PLANNER_F110_SELF_PLAY_ACTION_LATTICE_HPP_

#include <cstdint>
#include <utility>
#include <vector>

#include <torch/torch.h>

namespace planner::f110_self_play {

struct ActionLattice {
  std::vector<float> steering_bins;
  std::vector<float> velocity_bins;
  float velocity_min;
  float velocity_max;

  inline ActionLattice(int32_t steering_bins, int32_t velocity_bins,
                       float velocity_min = 1.0f, float velocity_max = 8.0f)
      : steering_bins(linspace(-1.0f, 1.0f, steering_bins)),
        velocity_bins(linspace(-1.0f, 1.0f, velocity_bins)),
        velocity_min(velocity_min), velocity_max(velocity_max) {}

  inline ActionLattice(std::vector<float> steering_bins,
                       std::vector<float> velocity_bins)
      : steering_bins(std::move(steering_bins)),
        velocity_bins(std::move(velocity_bins)), velocity_min(0.0f),
        velocity_max(0.0f) {}

  inline int32_t action_count() const {
    return static_cast<int32_t>(steering_bins.size() * velocity_bins.size());
  }

  inline torch::Tensor normalized_action(int32_t action_index) const {
    const int32_t velocity_count = static_cast<int32_t>(velocity_bins.size());
    const int32_t steer_idx = action_index / velocity_count;
    const int32_t vel_idx = action_index % velocity_count;
    const float steer = steering_bins[static_cast<std::size_t>(steer_idx)];
    const float velocity = velocity_bins[static_cast<std::size_t>(vel_idx)];
    auto out = torch::empty({2}, torch::TensorOptions().dtype(torch::kFloat32));
    out[0] = steer;
    out[1] = velocity;
    return out;
  }

  inline torch::Tensor
  normalized_batch(const torch::Tensor &action_indices) const {
    auto indices = action_indices.to(torch::kCPU).contiguous();
    auto out = torch::empty({indices.size(0), 2},
                            torch::TensorOptions().dtype(torch::kFloat32));
    const int64_t *idx_ptr = indices.data_ptr<int64_t>();
    float *out_ptr = out.data_ptr<float>();
    for (int64_t i = 0; i < indices.size(0); ++i) {
      const auto action = normalized_action(static_cast<int32_t>(idx_ptr[i]));
      out_ptr[i * 2 + 0] = action[0].item<float>();
      out_ptr[i * 2 + 1] = action[1].item<float>();
    }
    return out;
  }

private:
  static inline std::vector<float> linspace(float start, float end,
                                            int32_t count) {
    std::vector<float> values(static_cast<std::size_t>(count));
    if (count == 1) {
      values[0] = start;
      return values;
    }
    for (int32_t i = 0; i < count; ++i) {
      values[static_cast<std::size_t>(i)] =
          start +
          (end - start) * static_cast<float>(i) / static_cast<float>(count - 1);
    }
    return values;
  }
};

} // namespace planner::f110_self_play

#endif // PLANNER_F110_SELF_PLAY_ACTION_LATTICE_HPP_
