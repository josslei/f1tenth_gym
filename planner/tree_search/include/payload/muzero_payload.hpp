#ifndef PLANNER_TREE_SEARCH_PAYLOAD_MUZERO_PAYLOAD_HPP_
#define PLANNER_TREE_SEARCH_PAYLOAD_MUZERO_PAYLOAD_HPP_

#include <algorithm>
#include <vector>

#include <torch/torch.h>

#include "tree/batched_tree_base.hpp"

namespace planner::tree_search {

struct MuZeroPayload {
  BatchedTreeShape shape;
  BatchedTreeIndex index;

  std::vector<float> reward;
  std::vector<float> discount;
  torch::Tensor hidden_state;

  explicit inline MuZeroPayload(const BatchedTreeShape &s,
                                torch::Device device = torch::kCPU)
      : shape(s), index(s), reward(index.edge_count(), 0.0f),
        discount(index.edge_count(), 1.0f),
        hidden_state(torch::zeros(
            {shape.B, shape.Nmax, shape.H},
            torch::TensorOptions().dtype(torch::kFloat32).device(device))) {}

  inline void clear() {
    std::fill(reward.begin(), reward.end(), 0.0f);
    std::fill(discount.begin(), discount.end(), 1.0f);
    hidden_state.zero_();
  }

  inline float edge_reward(int b, int n, int a) const {
    return reward[index.edge(b, n, a)];
  }

  inline void set_reward(int b, int n, int a, float r) {
    reward[index.edge(b, n, a)] = r;
  }

  inline float edge_discount(int b, int n, int a) const {
    return discount[index.edge(b, n, a)];
  }

  inline void set_discount(int b, int n, int a, float d) {
    discount[index.edge(b, n, a)] = d;
  }

  inline torch::Tensor hidden(int b, int n) {
    return hidden_state.index({b, n});
  }

  inline torch::Tensor hidden(int b, int n) const {
    return hidden_state.index({b, n});
  }

  inline void set_hidden(int b, int n, const torch::Tensor &h) {
    hidden_state.index_put_({b, n}, h);
  }
};

} // namespace planner::tree_search

#endif // PLANNER_TREE_SEARCH_PAYLOAD_MUZERO_PAYLOAD_HPP_
