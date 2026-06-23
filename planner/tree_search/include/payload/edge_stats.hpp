#ifndef PLANNER_TREE_SEARCH_PAYLOAD_EDGE_STATS_HPP_
#define PLANNER_TREE_SEARCH_PAYLOAD_EDGE_STATS_HPP_

#include <algorithm>
#include <cstdint>
#include <vector>

#include "tree/batched_tree_base.hpp"

namespace planner::tree_search {

struct EdgeStats {
  BatchedTreeShape shape;
  BatchedTreeIndex index;

  std::vector<int32_t> visit_count;
  std::vector<float> value_sum;
  std::vector<float> prior;

  explicit inline EdgeStats(const BatchedTreeShape &s)
      : shape(s), index(s), visit_count(index.edge_count(), 0),
        value_sum(index.edge_count(), 0.0f), prior(index.edge_count(), 0.0f) {}

  inline void clear() {
    std::fill(visit_count.begin(), visit_count.end(), 0);
    std::fill(value_sum.begin(), value_sum.end(), 0.0f);
    std::fill(prior.begin(), prior.end(), 0.0f);
  }

  inline int32_t visits(int b, int n, int a) const {
    return visit_count[index.edge(b, n, a)];
  }

  inline float prior_prob(int b, int n, int a) const {
    return prior[index.edge(b, n, a)];
  }

  inline void set_prior(int b, int n, int a, float p) {
    prior[index.edge(b, n, a)] = p;
  }

  inline float q_value(int b, int n, int a) const {
    const std::size_t ei = index.edge(b, n, a);
    return visit_count[ei] == 0 ? 0.0f : value_sum[ei] / visit_count[ei];
  }

  inline void add_visit(int b, int n, int a, float value_delta) {
    const std::size_t ei = index.edge(b, n, a);
    visit_count[ei] += 1;
    value_sum[ei] += value_delta;
  }

  inline void reset_edge(int b, int n, int a) {
    const std::size_t ei = index.edge(b, n, a);
    visit_count[ei] = 0;
    value_sum[ei] = 0.0f;
    prior[ei] = 0.0f;
  }
};

} // namespace planner::tree_search

#endif // PLANNER_TREE_SEARCH_PAYLOAD_EDGE_STATS_HPP_
