#ifndef PLANNER_F110_SELF_PLAY_ACTION_LATTICE_HPP_
#define PLANNER_F110_SELF_PLAY_ACTION_LATTICE_HPP_

#include <cstdint>

#include <torch/torch.h>

namespace planner::f110_self_play {

struct ActionLattice {
  int32_t steering_bins;
  int32_t velocity_bins;
  float velocity_min;
  float velocity_max;

  inline ActionLattice(int32_t steering_bins, int32_t velocity_bins,
                       float velocity_min = 1.0f, float velocity_max = 8.0f)
      : steering_bins(steering_bins), velocity_bins(velocity_bins),
        velocity_min(velocity_min), velocity_max(velocity_max) {}

  inline int32_t action_count() const { return steering_bins * velocity_bins; }

  inline torch::Tensor normalized_action(int32_t action_index) const {
    const int32_t steer_idx = action_index / velocity_bins;
    const int32_t vel_idx = action_index % velocity_bins;
    const float steer = steering_bins > 1
                            ? -1.0f + 2.0f * static_cast<float>(steer_idx) /
                                          static_cast<float>(steering_bins - 1)
                            : 0.0f;
    const float velocity =
        velocity_bins > 1 ? -1.0f + 2.0f * static_cast<float>(vel_idx) /
                                        static_cast<float>(velocity_bins - 1)
                          : 0.0f;
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
};

} // namespace planner::f110_self_play

#endif // PLANNER_F110_SELF_PLAY_ACTION_LATTICE_HPP_
