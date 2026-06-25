#ifndef PLANNER_F110_SELF_PLAY_MUZERO_SEARCH_ADAPTER_HPP_
#define PLANNER_F110_SELF_PLAY_MUZERO_SEARCH_ADAPTER_HPP_

#include <cstdint>
#include <map>
#include <memory>
#include <string>

#include <torch/script.h>

#include "search/muzero_search.hpp"

namespace planner::f110_self_play {

struct SearchBatchResult {
  torch::Tensor action_probs;
  torch::Tensor root_values;
  std::map<std::string, double> metrics;
};

class MuZeroSearchAdapter final {
public:
  inline MuZeroSearchAdapter(const std::string &model_path, int32_t num_iters,
                             float temperature, float c_puct,
                             float dirichlet_alpha, float dirichlet_epsilon,
                             int32_t batch_size, int32_t action_count,
                             int32_t hidden_size, int32_t max_nodes,
                             torch::Device device, bool print_metrics)
      : search(load_model(model_path, supported_device(device)), num_iters,
               temperature, c_puct, dirichlet_alpha, dirichlet_epsilon,
               planner::tree_search::BatchedTreeShape(
                   batch_size, max_nodes, action_count, hidden_size),
               supported_device(device), print_metrics),
        device(supported_device(device)) {}

  inline SearchBatchResult search_batch(const torch::Tensor &obs_batch) {
    auto action_probs = search.search_batch(obs_batch.to(device));
    return {action_probs, search.root_values(), search.get_metrics()};
  }

private:
  static inline torch::Device supported_device(torch::Device device) {
    TORCH_CHECK(device.type() == torch::kCPU || device.type() == torch::kCUDA,
                "MuZero native backend supports only CPU or CUDA, got ",
                device.str());
    return device;
  }

  static inline torch::jit::Module load_model(const std::string &model_path,
                                              torch::Device device) {
    auto model = torch::jit::load(model_path);
    model.to(device);
    model.eval();
    return model;
  }

  planner::tree_search::MuZeroSearch search;
  torch::Device device;
};

} // namespace planner::f110_self_play

#endif // PLANNER_F110_SELF_PLAY_MUZERO_SEARCH_ADAPTER_HPP_
