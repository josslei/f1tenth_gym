#ifndef PLANNER_TREE_SEARCH_TREE_BATCHED_MCTS_TREE_HPP_
#define PLANNER_TREE_SEARCH_TREE_BATCHED_MCTS_TREE_HPP_

#include <cstdint>

#include "payload/edge_stats.hpp"
#include "tree/batched_tree_base.hpp"

namespace planner::tree_search {

struct BatchedMCTSTree {
  BatchedTreeShape shape;
  BatchedTreeIndex index;

  BatchedTreeTopology topology;
  EdgeStats edge_stats;

  explicit inline BatchedMCTSTree(const BatchedTreeShape &s)
      : shape(s), index(s), topology(s), edge_stats(s) {}

  inline void clear() {
    topology.clear();
    edge_stats.clear();
  }

  inline void init_roots(int32_t player = 0) {
    for (int b = 0; b < shape.B; ++b) {
      topology.init_root(b, player);
    }
  }

  inline int32_t allocate_child(int b, int parent_n, int action,
                                int32_t next_player) {
    return topology.add_child(b, parent_n, action, next_player);
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

  inline void backup_edge(int b, int n, int a, float value) {
    edge_stats.add_visit(b, n, a, value);
    topology.increment_node_visit(b, n);
  }
};

} // namespace planner::tree_search

#endif // PLANNER_TREE_SEARCH_TREE_BATCHED_MCTS_TREE_HPP_
