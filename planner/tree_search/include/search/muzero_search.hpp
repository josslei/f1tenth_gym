#ifndef PLANNER_TREE_SEARCH_SEARCH_MUZERO_SEARCH_HPP_
#define PLANNER_TREE_SEARCH_SEARCH_MUZERO_SEARCH_HPP_

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <map>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include <c10/core/InferenceMode.h>
#include <torch/script.h>

#include "tree/batched_muzero_tree.hpp"
#include "tree_policies/puct_policy.hpp"

namespace planner::tree_search {

class MuZeroSearch {
  using Clock = std::chrono::steady_clock;

public:
  inline MuZeroSearch(torch::jit::Module model, int32_t num_iters,
                      float temperature, float c_puct, float dirichlet_alpha,
                      float dirichlet_epsilon, const BatchedTreeShape &shape,
                      torch::Device device, bool print_metrics = false)
      : model(std::move(model)), num_iters(num_iters), temperature(temperature),
        c_puct(c_puct), dirichlet_alpha(dirichlet_alpha),
        dirichlet_epsilon(dirichlet_epsilon), shape(shape), index(shape),
        tree(shape, device), tree_policy(c_puct), device(device),
        print_metrics(print_metrics),
        initial_method(this->model.get_method("initial_inference")),
        recurrent_method(this->model.get_method("recurrent_inference")),
        recurrent_hidden(torch::empty(
            {shape.B, shape.H},
            torch::TensorOptions().dtype(torch::kFloat32).device(device))),
        recurrent_action(torch::empty(
            {shape.B},
            torch::TensorOptions().dtype(torch::kInt64).device(device))),
        selected_parent(static_cast<std::size_t>(shape.B), 0),
        selected_action(static_cast<std::size_t>(shape.B), 0),
        selected_child(static_cast<std::size_t>(shape.B), 0),
        selected_terminal(static_cast<std::size_t>(shape.B), 0),
        selected_child_terminal(static_cast<std::size_t>(shape.B), 0),
        path_node(static_cast<std::size_t>(shape.B) * shape.Nmax, 0),
        path_action(static_cast<std::size_t>(shape.B) * shape.Nmax, 0),
        path_length(static_cast<std::size_t>(shape.B), 0),
        initial_payload_cpu(torch::empty(
            {shape.B, initial_payload_width()},
            torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU))),
        recurrent_payload_cpu(torch::empty(
            {shape.B, recurrent_payload_width()},
            torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU))),
        recurrent_action_cpu(torch::empty(
            {shape.B},
            torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU))),
        legal_mask_cpu(torch::ones(
            {shape.B, shape.A},
            torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCPU))) {
    // TODO: If multiple searches run concurrently in the future, consider
    // async GPU evaluation and overlap CPU tree work with pending model copies.
    // Keep the evaluator embedded per search for now; a shared batch evaluator
    // can add synchronization overhead and is unnecessary when B is large.
  }

  inline torch::Tensor operator()(const torch::Tensor &obs_batch) {
    return search_batch(obs_batch);
  }

  inline torch::Tensor search_batch(const torch::Tensor &obs_batch) {
    c10::InferenceMode inference_guard;
    metrics.reset();
    const auto total_start = Clock::now();

    initialize_roots(obs_batch);

    for (int32_t iter = 0; iter < num_iters; ++iter) {
      const auto selection_start = Clock::now();
      select_leaves();
      metrics.selection_time_us += elapsed_us(selection_start, Clock::now());

      gather_model_inputs();

      auto recurrent = call_recurrent_model();

      const auto expand_start = Clock::now();
      expand_selected_edges(recurrent);
      metrics.expand_time_us += elapsed_us(expand_start, Clock::now());

      const auto backup_start = Clock::now();
      backup_selected_paths(recurrent.payload);
      metrics.backup_time_us += elapsed_us(backup_start, Clock::now());
    }

    const auto root_policy_start = Clock::now();
    auto probs = root_action_probabilities();
    last_root_values = tree.root_values_batch(/*root_node=*/0);
    metrics.root_policy_time_us += elapsed_us(root_policy_start, Clock::now());

    metrics.search_total_time_us = elapsed_us(total_start, Clock::now());
    finalize_metrics();
    if (print_metrics) {
      print_metrics_summary();
    }
    return probs;
  }

  inline torch::Tensor search_one(const torch::Tensor &obs) {
    // Placeholder for a future single-instance fast path.
    (void)obs;
    return torch::Tensor();
  }

  inline std::map<std::string, double> get_metrics() const {
    return metrics.to_map();
  }

  inline torch::Tensor root_values() const { return last_root_values; }

private:
  static constexpr int32_t kInitialValueOffset = 0;
  static constexpr int32_t kInitialPolicyOffset = 1;
  static constexpr int32_t kRecurrentRewardOffset = 0;
  static constexpr int32_t kRecurrentValueOffset = 1;
  static constexpr int32_t kRecurrentDiscountOffset = 2;
  static constexpr int32_t kRecurrentPolicyOffset = 3;

  struct InitialOutput {
    // initial_inference returns hidden [B, H] and payload [B, A + 1].
    // payload columns: value, policy[0:A].
    torch::Tensor hidden;
    torch::Tensor payload;
  };

  struct RecurrentOutput {
    // recurrent_inference returns hidden [B, H] and payload [B, A + 3].
    // payload columns: reward, value, discount, policy[0:A].
    torch::Tensor hidden;
    torch::Tensor payload;
  };

  struct MuZeroMetrics {
    long long search_total_time_us = 0;
    long long initial_inference_time_us = 0;
    long long recurrent_inference_time_us = 0;
    long long payload_copy_time_us = 0;
    long long selection_time_us = 0;
    long long expand_time_us = 0;
    long long backup_time_us = 0;
    long long root_policy_time_us = 0;
    long long simulations = 0;
    long long iterations = 0;
    double simulations_per_lane = 0.0;

    long long nodes_allocated_sum = 0;
    long long nodes_allocated_count = 0;
    int32_t nodes_allocated_min = std::numeric_limits<int32_t>::max();
    int32_t nodes_allocated_max = 0;

    long long search_depth_sum = 0;
    long long search_depth_count = 0;
    int32_t search_depth_min = std::numeric_limits<int32_t>::max();
    int32_t search_depth_max = 0;

    long long root_visit_count_sum = 0;
    long long root_visit_count_count = 0;
    int32_t root_visit_count_min = std::numeric_limits<int32_t>::max();
    int32_t root_visit_count_max = 0;

    inline void reset() {
      search_total_time_us = 0;
      initial_inference_time_us = 0;
      recurrent_inference_time_us = 0;
      payload_copy_time_us = 0;
      selection_time_us = 0;
      expand_time_us = 0;
      backup_time_us = 0;
      root_policy_time_us = 0;
      simulations = 0;
      iterations = 0;
      simulations_per_lane = 0.0;

      nodes_allocated_sum = 0;
      nodes_allocated_count = 0;
      nodes_allocated_min = std::numeric_limits<int32_t>::max();
      nodes_allocated_max = 0;

      search_depth_sum = 0;
      search_depth_count = 0;
      search_depth_min = std::numeric_limits<int32_t>::max();
      search_depth_max = 0;

      root_visit_count_sum = 0;
      root_visit_count_count = 0;
      root_visit_count_min = std::numeric_limits<int32_t>::max();
      root_visit_count_max = 0;
    }

    inline double mean(long long sum, long long count) const {
      return count > 0 ? static_cast<double>(sum) / static_cast<double>(count)
                       : 0.0;
    }

    inline double maybe_min(int32_t value, long long count) const {
      return count > 0 ? static_cast<double>(value) : 0.0;
    }

    inline double maybe_max(int32_t value, long long count) const {
      return count > 0 ? static_cast<double>(value) : 0.0;
    }

    inline double simulations_per_second() const {
      if (search_total_time_us <= 0) {
        return 0.0;
      }

      return static_cast<double>(simulations) /
             (static_cast<double>(search_total_time_us) / 1.0e6);
    }

    inline std::map<std::string, double> to_map() const {
      return {
          {"search/total_time_us", static_cast<double>(search_total_time_us)},
          {"search/selection_time_us", static_cast<double>(selection_time_us)},
          {"search/expand_time_us", static_cast<double>(expand_time_us)},
          {"search/backup_time_us", static_cast<double>(backup_time_us)},
          {"search/root_policy_time_us",
           static_cast<double>(root_policy_time_us)},
          {"search/iterations", static_cast<double>(iterations)},
          {"search/simulations_total", static_cast<double>(simulations)},
          {"search/simulations_per_lane", simulations_per_lane},
          {"inference/initial_time_us",
           static_cast<double>(initial_inference_time_us)},
          {"inference/recurrent_time_us",
           static_cast<double>(recurrent_inference_time_us)},
          {"inference/payload_copy_time_us",
           static_cast<double>(payload_copy_time_us)},
          {"search/simulations", static_cast<double>(simulations)},
          {"throughput/simulations_per_second", simulations_per_second()},
          {"tree/nodes_allocated_avg",
           mean(nodes_allocated_sum, nodes_allocated_count)},
          {"tree/nodes_allocated_min",
           maybe_min(nodes_allocated_min, nodes_allocated_count)},
          {"tree/nodes_allocated_max",
           maybe_max(nodes_allocated_max, nodes_allocated_count)},
          {"tree/search_depth_avg", mean(search_depth_sum, search_depth_count)},
          {"tree/search_depth_min",
           maybe_min(search_depth_min, search_depth_count)},
          {"tree/search_depth_max",
           maybe_max(search_depth_max, search_depth_count)},
          {"tree/root_visit_count_avg",
           mean(root_visit_count_sum, root_visit_count_count)},
          {"tree/root_visit_count_min",
           maybe_min(root_visit_count_min, root_visit_count_count)},
          {"tree/root_visit_count_max",
           maybe_max(root_visit_count_max, root_visit_count_count)},
          {"tree/root_visit_count_count",
           static_cast<double>(root_visit_count_count)},
      };
    }
  };

  torch::jit::Module model;
  torch::jit::Method initial_method;
  torch::jit::Method recurrent_method;
  int32_t num_iters;
  float temperature;
  float c_puct;
  float dirichlet_alpha;
  float dirichlet_epsilon;

  BatchedTreeShape shape;
  BatchedTreeIndex index;
  BatchedMuZeroTree tree;
  PUCTPolicy tree_policy;
  torch::Device device;
  bool print_metrics;

  torch::Tensor recurrent_hidden;
  torch::Tensor recurrent_action;

  torch::Tensor initial_payload_cpu;
  torch::Tensor recurrent_payload_cpu;
  torch::Tensor recurrent_action_cpu;
  torch::Tensor legal_mask_cpu;
  torch::Tensor last_root_values;

  std::mt19937 rng{std::random_device{}()};

  std::vector<int32_t> selected_parent;
  std::vector<int32_t> selected_action;
  std::vector<int32_t> selected_child;
  std::vector<uint8_t> selected_terminal;
  std::vector<uint8_t> selected_child_terminal;

  std::vector<int32_t> path_node;
  std::vector<int32_t> path_action;
  std::vector<int32_t> path_length;

  MuZeroMetrics metrics;

  inline int32_t initial_payload_width() const {
    return shape.A + kInitialPolicyOffset;
  }

  inline int32_t recurrent_payload_width() const {
    return shape.A + kRecurrentPolicyOffset;
  }

  inline std::size_t initial_policy(int b, int a) const {
    return index.batch_matrix(b, kInitialPolicyOffset + a,
                              initial_payload_width());
  }

  inline std::size_t recurrent_reward(int b) const {
    return index.batch_matrix(b, kRecurrentRewardOffset,
                              recurrent_payload_width());
  }

  inline std::size_t recurrent_value(int b) const {
    return index.batch_matrix(b, kRecurrentValueOffset,
                              recurrent_payload_width());
  }

  inline std::size_t recurrent_discount(int b) const {
    return index.batch_matrix(b, kRecurrentDiscountOffset,
                              recurrent_payload_width());
  }

  static inline long long elapsed_us(const Clock::time_point &start,
                                     const Clock::time_point &end) {
    return std::chrono::duration_cast<std::chrono::microseconds>(end - start)
        .count();
  }

  inline void initialize_roots(const torch::Tensor &obs_batch) {
    tree.clear();

    InitialOutput root = call_initial_model(obs_batch);

    const auto root_hidden_start = Clock::now();
    tree.init_roots(0);
    tree.set_root_hidden_batch(root.hidden);
    metrics.payload_copy_time_us += elapsed_us(root_hidden_start, Clock::now());

    const auto expand_start = Clock::now();
    expand_root_batch(root.payload, root_legal_mask_batch());
    metrics.expand_time_us += elapsed_us(expand_start, Clock::now());
  }

  inline InitialOutput call_initial_model(const torch::Tensor &obs_batch) {
    // MuZero initial inference:
    //   observation -> hidden, packed payload
    // TorchScript method lookup is cached in the constructor.
    const auto start = Clock::now();
    auto out = initial_method({obs_batch});
    const auto tuple = out.toTuple()->elements();

    InitialOutput result;
    result.hidden = tuple[0].toTensor();
    result.payload = tuple[1].toTensor();

    metrics.initial_inference_time_us += elapsed_us(start, Clock::now());

    const auto copy_start = Clock::now();
    check_supported_tensor(result.payload, "initial_payload");
    if (device.type() == torch::kCPU) {
      initial_payload_cpu = result.payload;
    } else {
      initial_payload_cpu.copy_(result.payload);
    }
    metrics.payload_copy_time_us += elapsed_us(copy_start, Clock::now());

    result.payload = initial_payload_cpu;
    return result;
  }

  inline RecurrentOutput call_recurrent_model() {
    // MuZero recurrent inference:
    //   hidden + action -> next_hidden, packed payload
    const auto start = Clock::now();
    auto out = recurrent_method({recurrent_hidden, recurrent_action});
    const auto tuple = out.toTuple()->elements();

    RecurrentOutput result;
    result.hidden = tuple[0].toTensor();
    result.payload = tuple[1].toTensor();

    metrics.recurrent_inference_time_us += elapsed_us(start, Clock::now());

    const auto copy_start = Clock::now();
    check_supported_tensor(result.payload, "recurrent_payload");
    if (device.type() == torch::kCPU) {
      recurrent_payload_cpu = result.payload;
    } else {
      recurrent_payload_cpu.copy_(result.payload);
    }
    metrics.payload_copy_time_us += elapsed_us(copy_start, Clock::now());

    result.payload = recurrent_payload_cpu;
    return result;
  }

  inline void expand_root_batch(const torch::Tensor &root_payload,
                                const torch::Tensor &root_legal_mask) {
    // Batched root expansion lives here so the top-level loop stays readable.
    const float *payload_ptr = root_payload.data_ptr<float>();
    const uint8_t *legal_ptr = root_legal_mask.data_ptr<uint8_t>();

    for (int32_t b = 0; b < shape.B; ++b) {
      std::vector<float> priors(static_cast<std::size_t>(shape.A), 0.0f);
      std::vector<int32_t> legal_actions;
      legal_actions.reserve(static_cast<std::size_t>(shape.A));

      tree.topology.set_expanded(b, 0, true);
      tree.topology.clear_legal_actions(b, 0);

      for (int32_t a = 0; a < shape.A; ++a) {
        const bool legal = legal_ptr[index.batch_action(b, a)] != 0;
        tree.topology.set_legal(b, 0, a, legal);
        if (!legal) {
          continue;
        }
        priors[static_cast<std::size_t>(a)] = payload_ptr[initial_policy(b, a)];
        legal_actions.push_back(a);
      }

      if (dirichlet_alpha > 0.0f && dirichlet_epsilon > 0.0f &&
          legal_actions.size() > 1) {
        std::gamma_distribution<float> gamma_dist(dirichlet_alpha, 1.0f);
        std::vector<float> noise(legal_actions.size(), 0.0f);
        float noise_sum = 0.0f;
        for (std::size_t i = 0; i < legal_actions.size(); ++i) {
          noise[i] = gamma_dist(rng);
          noise_sum += noise[i];
        }
        const float inv_noise_sum = noise_sum > 0.0f ? 1.0f / noise_sum : 0.0f;
        float prior_sum = 0.0f;
        for (int32_t a : legal_actions) {
          prior_sum += priors[static_cast<std::size_t>(a)];
        }
        const float inv_prior_sum = prior_sum > 0.0f ? 1.0f / prior_sum : 0.0f;
        for (std::size_t i = 0; i < legal_actions.size(); ++i) {
          const int32_t a = legal_actions[i];
          const std::size_t idx_a = static_cast<std::size_t>(a);
          const float prior = priors[idx_a] * inv_prior_sum;
          const float sampled = noise[i] * inv_noise_sum;
          priors[idx_a] =
              (1.0f - dirichlet_epsilon) * prior + dirichlet_epsilon * sampled;
        }
      }

      for (int32_t a = 0; a < shape.A; ++a) {
        tree.edge_stats.set_prior(b, 0, a, priors[static_cast<std::size_t>(a)]);
      }
    }
  }

  inline void select_leaves() {
    // Selection is intentionally kept as a separate phase.
    // The final implementation can use Highway across batch lanes.
    std::fill(selected_action.begin(), selected_action.end(), 0);
    std::fill(selected_terminal.begin(), selected_terminal.end(), 0);
    std::fill(selected_child_terminal.begin(), selected_child_terminal.end(),
              0);
    for (int32_t b = 0; b < shape.B; ++b) {
      int32_t node = 0;
      int32_t depth = 0;
      bool found_leaf = false;

      while (!found_leaf && tree.is_expanded(b, node) &&
             !tree.is_terminal(b, node)) {
        const int32_t action = select_action(b, node);

        path_node[index.path(depth, b)] = node;
        path_action[index.path(depth, b)] = action;
        depth += 1;

        const int32_t child = tree.child(b, node, action);
        if (child == BatchedTreeTopology::kInvalidNode) {
          selected_parent[index.batch(b)] = node;
          selected_action[index.batch(b)] = action;
          path_length[index.batch(b)] = depth;
          found_leaf = true;
          continue;
        }

        node = child;
      }

      if (!found_leaf) {
        selected_parent[index.batch(b)] = node;
        path_length[index.batch(b)] = depth;
        selected_terminal[index.batch(b)] = tree.is_terminal(b, node) ? 1 : 0;
      }
    }
  }

  inline int32_t select_action(int b, int node) const {
    return tree_policy(tree, b, node);
  }

  inline void gather_model_inputs() {
    // Gather selected parent hidden state and actions into device-resident
    // model inputs.
    const auto hidden_start = Clock::now();
    recurrent_hidden = tree.gather_hidden_batch(selected_parent);
    check_supported_tensor(recurrent_hidden, "recurrent_hidden");

    int64_t *action_ptr = recurrent_action_cpu.data_ptr<int64_t>();
    for (int32_t b = 0; b < shape.B; ++b) {
      action_ptr[index.batch(b)] = selected_terminal[index.batch(b)]
                                       ? 0
                                       : selected_action[index.batch(b)];
    }
    metrics.payload_copy_time_us += elapsed_us(hidden_start, Clock::now());

    const auto action_copy_start = Clock::now();
    check_supported_tensor(recurrent_action, "recurrent_action");
    check_supported_tensor(recurrent_action_cpu, "recurrent_action_cpu");
    if (device.type() == torch::kCPU) {
      recurrent_action = recurrent_action_cpu;
    } else {
      recurrent_action.copy_(recurrent_action_cpu);
    }
    metrics.payload_copy_time_us += elapsed_us(action_copy_start, Clock::now());
  }

  static inline void check_supported_tensor(const torch::Tensor &tensor,
                                            const char *name) {
    const auto tensor_device = tensor.device();
    TORCH_CHECK(tensor_device.type() == torch::kCPU ||
                    tensor_device.type() == torch::kCUDA,
                name, " must be on CPU or CUDA, got ", tensor_device.str());
  }

  inline void expand_selected_edges(const RecurrentOutput &recurrent) {
    // Allocate one child per batch lane, store its hidden state, then expand.
    const float *payload_ptr = recurrent.payload.data_ptr<float>();

    for (int32_t b = 0; b < shape.B; ++b) {
      if (selected_terminal[index.batch(b)] != 0) {
        selected_child[index.batch(b)] = selected_parent[index.batch(b)];
        continue;
      }

      const int32_t parent = selected_parent[index.batch(b)];
      const int32_t action = selected_action[index.batch(b)];
      const float reward = payload_ptr[recurrent_reward(b)];
      const float discount = payload_ptr[recurrent_discount(b)];

      const int32_t child = tree.allocate_child(
          b, parent, action, /*next_player=*/0, reward, discount);
      selected_child[index.batch(b)] = child;
      if (child == BatchedTreeTopology::kInvalidNode) {
        continue;
      }
      if (discount < 0.5f) {
        tree.topology.set_terminal(b, child, true);
        selected_child_terminal[index.batch(b)] = 1;
        continue;
      }

      tree.muzero.hidden_state.select(0, b).select(0, child).copy_(
          recurrent.hidden.select(0, b));
      tree.topology.set_expanded(b, child, true);
      tree.topology.clear_legal_actions(b, child);
      for (int32_t a = 0; a < shape.A; ++a) {
        tree.edge_stats.set_prior(
            b, child, a,
            payload_ptr[index.batch_matrix(b, kRecurrentPolicyOffset + a,
                                           recurrent_payload_width())]);
        tree.topology.set_legal(b, child, a, true);
      }
    }
  }

  inline void backup_selected_paths(const torch::Tensor &payload) {
    const float *payload_ptr = payload.data_ptr<float>();

    for (int32_t b = 0; b < shape.B; ++b) {
      float value = selected_terminal[index.batch(b)] != 0
                        ? 0.0f
                        : payload_ptr[recurrent_value(b)];
      if (selected_child_terminal[index.batch(b)] != 0) {
        value = 0.0f;
      }

      for (int32_t d = path_length[index.batch(b)] - 1; d >= 0; --d) {
        const int32_t node = path_node[index.path(d, b)];
        const int32_t action = path_action[index.path(d, b)];

        const float reward = tree.reward(b, node, action);
        const float discount = tree.discount(b, node, action);
        value = reward + discount * value;
        tree.backup_edge(b, node, action, value);
      }
    }
  }

  inline torch::Tensor root_action_probabilities() {
    // Convert root edge visit counts to action probabilities.
    return tree.action_probabilities_batch(/*root_node=*/0, temperature);
  }

  inline torch::Tensor root_legal_mask_batch() {
    // Root legality is driven by the environment/model boundary.
    legal_mask_cpu.fill_(1);
    return legal_mask_cpu;
  }

  inline torch::Tensor latent_legal_mask_batch(const std::vector<int32_t> &) {
    // Latent-space actions are treated as all legal; legality is undefined in
    // latent state unless the action space itself supplies constraints.
    legal_mask_cpu.fill_(1);
    return legal_mask_cpu;
  }

  inline void finalize_metrics() {
    metrics.iterations = num_iters;
    metrics.simulations = static_cast<long long>(num_iters) * shape.B;
    metrics.simulations_per_lane =
        shape.B > 0 ? static_cast<double>(metrics.simulations) /
                          static_cast<double>(shape.B)
                    : 0.0;

    int32_t depth_min = std::numeric_limits<int32_t>::max();
    int32_t depth_max = 0;
    long long depth_sum = 0;
    for (int32_t b = 0; b < shape.B; ++b) {
      const int32_t depth = path_length[index.batch(b)];
      depth_sum += depth;
      depth_min = std::min(depth_min, depth);
      depth_max = std::max(depth_max, depth);
    }
    metrics.search_depth_sum = depth_sum;
    metrics.search_depth_count = shape.B;
    metrics.search_depth_min = depth_min;
    metrics.search_depth_max = depth_max;

    int32_t nodes_min = std::numeric_limits<int32_t>::max();
    int32_t nodes_max = 0;
    long long nodes_sum = 0;
    for (int32_t b = 0; b < shape.B; ++b) {
      const int32_t nodes = tree.topology.allocated_nodes(b);
      nodes_sum += nodes;
      nodes_min = std::min(nodes_min, nodes);
      nodes_max = std::max(nodes_max, nodes);
    }
    metrics.nodes_allocated_sum = nodes_sum;
    metrics.nodes_allocated_count = shape.B;
    metrics.nodes_allocated_min = nodes_min;
    metrics.nodes_allocated_max = nodes_max;

    int32_t root_visit_min = std::numeric_limits<int32_t>::max();
    int32_t root_visit_max = 0;
    long long root_visit_sum = 0;
    long long root_visit_count = 0;
    for (int32_t b = 0; b < shape.B; ++b) {
      for (int32_t a = 0; a < shape.A; ++a) {
        const int32_t visits = tree.edge_visits(b, 0, a);
        if (visits <= 0) {
          continue;
        }

        root_visit_sum += visits;
        root_visit_count += 1;
        root_visit_min = std::min(root_visit_min, visits);
        root_visit_max = std::max(root_visit_max, visits);
      }
    }

    metrics.root_visit_count_sum = root_visit_sum;
    metrics.root_visit_count_count = root_visit_count;
    metrics.root_visit_count_min = root_visit_count > 0 ? root_visit_min : 0;
    metrics.root_visit_count_max = root_visit_count > 0 ? root_visit_max : 0;
  }

  inline void print_metrics_summary() const {
    const auto metrics_map = metrics.to_map();

    std::cout << "\n---------------- Search Metrics ----------------"
              << std::endl;
    std::cout << "[Search]" << std::endl;
    std::cout << "  total:   "
              << metrics_map.at("search/total_time_us") / 1000.0 << " ms"
              << std::endl;
    std::cout << "  select:  "
              << metrics_map.at("search/selection_time_us") / 1000.0
              << " ms | expand: "
              << metrics_map.at("search/expand_time_us") / 1000.0
              << " ms | backup: "
              << metrics_map.at("search/backup_time_us") / 1000.0 << " ms"
              << std::endl;
    std::cout << "  policy:  "
              << metrics_map.at("search/root_policy_time_us") / 1000.0 << " ms"
              << std::endl;
    std::cout << "  iterations: " << metrics_map.at("search/iterations")
              << " | simulations/lane: "
              << metrics_map.at("search/simulations_per_lane")
              << " | simulations/batch: "
              << metrics_map.at("search/simulations_total") << std::endl;

    std::cout << "[Inference]" << std::endl;
    std::cout << "  initial:  "
              << metrics_map.at("inference/initial_time_us") / 1000.0 << " ms"
              << " | recurrent: "
              << metrics_map.at("inference/recurrent_time_us") / 1000.0
              << " ms | copy: "
              << metrics_map.at("inference/payload_copy_time_us") / 1000.0
              << " ms" << std::endl;

    std::cout << "[Tree]" << std::endl;
    std::cout << "  nodes avg/min/max: "
              << metrics_map.at("tree/nodes_allocated_avg") << " / "
              << metrics_map.at("tree/nodes_allocated_min") << " / "
              << metrics_map.at("tree/nodes_allocated_max") << std::endl;
    std::cout << "  depth avg/min/max: "
              << metrics_map.at("tree/search_depth_avg") << " / "
              << metrics_map.at("tree/search_depth_min") << " / "
              << metrics_map.at("tree/search_depth_max") << std::endl;
    std::cout << "  root visits avg/min/max/count: "
              << metrics_map.at("tree/root_visit_count_avg") << " / "
              << metrics_map.at("tree/root_visit_count_min") << " / "
              << metrics_map.at("tree/root_visit_count_max") << " / "
              << metrics_map.at("tree/root_visit_count_count") << std::endl;

    std::cout << "[Throughput]" << std::endl;
    std::cout << "  simulations/sec: "
              << metrics_map.at("throughput/simulations_per_second")
              << std::endl;
    std::cout << "------------------------------------------------\n"
              << std::endl;
  }
};

} // namespace planner::tree_search

#endif // PLANNER_TREE_SEARCH_SEARCH_MUZERO_SEARCH_HPP_
