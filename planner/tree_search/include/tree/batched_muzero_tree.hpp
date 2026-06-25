#ifndef PLANNER_TREE_SEARCH_TREE_BATCHED_MUZERO_TREE_HPP_
#define PLANNER_TREE_SEARCH_TREE_BATCHED_MUZERO_TREE_HPP_

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

#include <torch/torch.h>

#include "payload/edge_stats.hpp"
#include "payload/muzero_payload.hpp"
#include "tree/batched_tree_base.hpp"

namespace planner::tree_search {

struct BatchedMuZeroTree {
  BatchedTreeShape shape;
  BatchedTreeIndex index;

  BatchedTreeTopology topology;
  EdgeStats edge_stats;
  MuZeroPayload muzero;

  explicit inline BatchedMuZeroTree(const BatchedTreeShape &s,
                                    torch::Device device = torch::kCPU)
      : shape(s), index(s), topology(s), edge_stats(s), muzero(s, device) {}

  inline void clear() {
    topology.clear();
    edge_stats.clear();
    muzero.clear();
  }

  inline void set_root_hidden_batch(const torch::Tensor &root_hidden) {
    muzero.hidden_state.select(1, 0).copy_(root_hidden);
  }

  inline torch::Tensor
  gather_hidden_batch(const std::vector<int32_t> &parent_batch) const {
    auto hidden_batch =
        torch::empty({shape.B, shape.H}, muzero.hidden_state.options());

    for (int32_t b = 0; b < shape.B; ++b) {
      hidden_batch.select(0, b).copy_(muzero.hidden_state.select(0, b).select(
          0, parent_batch[index.batch(b)]));
    }

    return hidden_batch;
  }

  inline void set_hidden_batch(const std::vector<int32_t> &node_batch,
                               const torch::Tensor &hidden_batch) {
    for (int32_t b = 0; b < shape.B; ++b) {
      muzero.hidden_state.select(0, b)
          .select(0, node_batch[index.batch(b)])
          .copy_(hidden_batch.select(0, b));
    }
  }

  inline void expand_batch(const std::vector<int32_t> &node_batch,
                           const torch::Tensor &policy_payload,
                           const torch::Tensor &legal_mask,
                           int32_t policy_offset = 0) {
    const float *policy_ptr = policy_payload.data_ptr<float>();
    const uint8_t *legal_ptr = legal_mask.data_ptr<uint8_t>();

    for (int32_t b = 0; b < shape.B; ++b) {
      const int32_t node = node_batch[index.batch(b)];
      topology.set_expanded(b, node, true);
      topology.clear_legal_actions(b, node);

      for (int32_t a = 0; a < shape.A; ++a) {
        edge_stats.set_prior(
            b, node, a,
            policy_ptr[index.batch_matrix(b, policy_offset + a,
                                          shape.A + policy_offset)]);
        topology.set_legal(b, node, a,
                           legal_ptr[index.batch_action(b, a)] != 0);
      }
    }
  }

  inline torch::Tensor action_probabilities_batch(int root_node,
                                                  float temperature) const {
    auto probs = torch::zeros({shape.B, shape.A},
                              torch::TensorOptions().dtype(torch::kFloat32));
    float *prob_ptr = probs.data_ptr<float>();

    for (int32_t b = 0; b < shape.B; ++b) {
      if (temperature <= 1.0e-3f) {
        int32_t best_action = BatchedTreeTopology::kInvalidAction;
        int32_t best_visits = -1;

        for (int32_t a = 0; a < shape.A; ++a) {
          if (!is_legal(b, root_node, a)) {
            continue;
          }

          const int32_t visits = edge_visits(b, root_node, a);
          if (visits > best_visits) {
            best_visits = visits;
            best_action = a;
          }
        }

        if (best_action != BatchedTreeTopology::kInvalidAction) {
          prob_ptr[index.batch_action(b, best_action)] = 1.0f;
        }
        continue;
      }

      float sum = 0.0f;
      for (int32_t a = 0; a < shape.A; ++a) {
        if (!is_legal(b, root_node, a)) {
          continue;
        }

        const float weight =
            std::pow(static_cast<float>(edge_visits(b, root_node, a)),
                     1.0f / temperature);
        prob_ptr[index.batch_action(b, a)] = weight;
        sum += weight;
      }

      if (sum > 0.0f) {
        for (int32_t a = 0; a < shape.A; ++a) {
          prob_ptr[index.batch_action(b, a)] /= sum;
        }
      } else {
        int32_t legal_count = 0;
        for (int32_t a = 0; a < shape.A; ++a) {
          legal_count += is_legal(b, root_node, a) ? 1 : 0;
        }

        const float uniform = legal_count > 0 ? 1.0f / legal_count : 0.0f;
        for (int32_t a = 0; a < shape.A; ++a) {
          prob_ptr[index.batch_action(b, a)] =
              is_legal(b, root_node, a) ? uniform : 0.0f;
        }
      }
    }

    return probs;
  }

  inline torch::Tensor root_values_batch(int root_node) const {
    auto values =
        torch::zeros({shape.B}, torch::TensorOptions().dtype(torch::kFloat32));
    float *value_ptr = values.data_ptr<float>();

    for (int32_t b = 0; b < shape.B; ++b) {
      float weighted_sum = 0.0f;
      int32_t visit_sum = 0;
      for (int32_t a = 0; a < shape.A; ++a) {
        if (!is_legal(b, root_node, a)) {
          continue;
        }

        const int32_t visits = edge_visits(b, root_node, a);
        weighted_sum += static_cast<float>(visits) * q_value(b, root_node, a);
        visit_sum += visits;
      }

      value_ptr[index.batch(b)] =
          visit_sum > 0 ? weighted_sum / static_cast<float>(visit_sum) : 0.0f;
    }

    return values;
  }

  inline void init_roots(int32_t player = 0) {
    for (int b = 0; b < shape.B; ++b) {
      topology.init_root(b, player);
    }
  }

  inline int32_t allocate_child(int b, int parent_n, int action,
                                int32_t next_player, float reward,
                                float discount) {
    const int32_t child_n =
        topology.add_child(b, parent_n, action, next_player);

    if (child_n == BatchedTreeTopology::kInvalidNode) {
      return BatchedTreeTopology::kInvalidNode;
    }

    muzero.set_reward(b, parent_n, action, reward);
    muzero.set_discount(b, parent_n, action, discount);

    return child_n;
  }

  inline bool is_expanded(int b, int n) const {
    return topology.is_expanded(b, n);
  }

  inline bool is_terminal(int b, int n) const {
    return topology.is_terminal(b, n);
  }

  inline int32_t child(int b, int n, int a) const {
    return topology.child(b, n, a);
  }

  inline bool has_child(int b, int n, int a) const {
    return topology.has_child(b, n, a);
  }

  inline bool is_legal(int b, int n, int a) const {
    return topology.is_legal(b, n, a);
  }

  inline float prior(int b, int n, int a) const {
    return edge_stats.prior_prob(b, n, a);
  }

  inline float q_value(int b, int n, int a) const {
    return edge_stats.q_value(b, n, a);
  }

  inline int32_t edge_visits(int b, int n, int a) const {
    return edge_stats.visits(b, n, a);
  }

  inline int32_t node_visits(int b, int n) const {
    return topology.node_visits(b, n);
  }

  inline float reward(int b, int n, int a) const {
    return muzero.edge_reward(b, n, a);
  }

  inline float discount(int b, int n, int a) const {
    return muzero.edge_discount(b, n, a);
  }

  inline torch::Tensor hidden(int b, int n) { return muzero.hidden(b, n); }

  inline torch::Tensor hidden(int b, int n) const {
    return muzero.hidden(b, n);
  }

  inline void backup_edge(int b, int n, int a, float value) {
    edge_stats.add_visit(b, n, a, value);
    topology.increment_node_visit(b, n);
  }
};

} // namespace planner::tree_search

#endif // PLANNER_TREE_SEARCH_TREE_BATCHED_MUZERO_TREE_HPP_
