#ifndef PLANNER_TREE_SEARCH_TREE_POLICIES_PUCT_POLICY_HPP_
#define PLANNER_TREE_SEARCH_TREE_POLICIES_PUCT_POLICY_HPP_

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>

#include "tree/batched_tree_base.hpp"

namespace planner::tree_search {

class PUCTPolicy {
public:
  explicit inline PUCTPolicy(float c_puct) : c_puct(c_puct) {}

  template <typename Tree>
  inline int32_t operator()(const Tree &tree, int b, int node) const {
    const float sqrt_parent_visits = std::sqrt(
        std::max(1.0f, static_cast<float>(tree.node_visits(b, node))));

    int32_t best_action = BatchedTreeTopology::kInvalidAction;
    float best_score = -std::numeric_limits<float>::infinity();

    for (int32_t a = 0; a < tree.shape.A; ++a) {
      if (!tree.is_legal(b, node, a)) {
        continue;
      }

      const float q = tree.q_value(b, node, a);
      const float prior = tree.prior(b, node, a);
      const float visits = static_cast<float>(tree.edge_visits(b, node, a));
      const float u = c_puct * prior * sqrt_parent_visits / (1.0f + visits);
      const float score = q + u;

      if (score > best_score) {
        best_score = score;
        best_action = a;
      }
    }

    return best_action;
  }

private:
  float c_puct;
};

} // namespace planner::tree_search

#endif // PLANNER_TREE_SEARCH_TREE_POLICIES_PUCT_POLICY_HPP_
