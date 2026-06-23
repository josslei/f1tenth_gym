#ifndef PLANNER_TREE_SEARCH_TREE_BATCHED_TREE_BASE_HPP_
#define PLANNER_TREE_SEARCH_TREE_BATCHED_TREE_BASE_HPP_

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace planner::tree_search {

/**
 * Batched tree dimensions used by the shared shape/index helpers.
 *
 * B, Nmax, A, and H are provided at runtime by the caller.
 * They are fixed for the duration of one library call.
 * B is not a compile-time parameter; it must only stay constant within the
 * call because SIMD layout depends on it.
 *
 * B: batch size
 * Nmax: maximum nodes per tree
 * A: maximum number of discrete actions
 * H: flat hidden-state dimension
 */
struct BatchedTreeShape {
  int32_t B = 0;
  int32_t Nmax = 0;
  int32_t A = 0;
  int32_t H = 0;

  inline BatchedTreeShape() = default;

  inline BatchedTreeShape(int32_t batch_size, int32_t max_nodes,
                          int32_t action_count, int32_t hidden_size = 0)
      : B(batch_size), Nmax(max_nodes), A(action_count), H(hidden_size) {}
};

/**
 * Indexer for the batched tree storage layout.
 *
 * CPU tree arrays are batch-contiguous so Highway can vectorize over B:
 * - node-level data: [Nmax, B]
 * - edge-level data: [Nmax, A, B]
 * - hidden helper layout, if used for CPU buffers: [Nmax, B, H]
 *
 * Model-facing buffers keep their natural row-major batch layout:
 * - batch-action data: [B, A]
 * - search paths: [depth, B]
 */
struct BatchedTreeIndex {
  BatchedTreeShape shape;

  explicit inline BatchedTreeIndex(const BatchedTreeShape &s) : shape(s) {}

  inline std::size_t node_count() const {
    return static_cast<std::size_t>(shape.B) *
           static_cast<std::size_t>(shape.Nmax);
  }

  inline std::size_t edge_count() const {
    return node_count() * static_cast<std::size_t>(shape.A);
  }

  inline std::size_t hidden_count() const {
    return node_count() * static_cast<std::size_t>(shape.H);
  }

  inline std::size_t batch(int b) const { return static_cast<std::size_t>(b); }

  inline std::size_t node(int b, int n) const {
    return static_cast<std::size_t>(n) * static_cast<std::size_t>(shape.B) +
           batch(b);
  }

  inline std::size_t edge(int b, int n, int a) const {
    return (static_cast<std::size_t>(n) * static_cast<std::size_t>(shape.A) +
            static_cast<std::size_t>(a)) *
               static_cast<std::size_t>(shape.B) +
           batch(b);
  }

  inline std::size_t hidden(int b, int n, int h) const {
    return node(b, n) * static_cast<std::size_t>(shape.H) +
           static_cast<std::size_t>(h);
  }

  inline std::size_t batch_action(int b, int a) const {
    return batch(b) * static_cast<std::size_t>(shape.A) +
           static_cast<std::size_t>(a);
  }

  inline std::size_t path(int depth, int b) const {
    return static_cast<std::size_t>(depth) * static_cast<std::size_t>(shape.B) +
           batch(b);
  }

  inline std::size_t matrix(int row, int col, int cols) const {
    return static_cast<std::size_t>(row) * static_cast<std::size_t>(cols) +
           static_cast<std::size_t>(col);
  }

  inline std::size_t batch_matrix(int b, int col, int cols) const {
    return matrix(b, col, cols);
  }
};

struct BatchedTreeTopology {
  static constexpr int32_t kInvalidNode = -1;
  static constexpr int32_t kInvalidAction = -1;

  BatchedTreeShape shape;
  BatchedTreeIndex index;

  std::vector<uint8_t> expanded;
  std::vector<uint8_t> terminal;
  std::vector<int32_t> parent_index;
  std::vector<int32_t> parent_action;
  std::vector<int32_t> depth;
  std::vector<int32_t> player_id;
  std::vector<int32_t> node_visit_count;
  std::vector<int32_t> next_node_index;
  std::vector<int32_t> child_index;
  std::vector<uint8_t> legal_action;

  explicit inline BatchedTreeTopology(const BatchedTreeShape &s)
      : shape(s), index(s), expanded(index.node_count(), 0),
        terminal(index.node_count(), 0),
        parent_index(index.node_count(), kInvalidNode),
        parent_action(index.node_count(), kInvalidAction),
        depth(index.node_count(), 0), player_id(index.node_count(), 0),
        node_visit_count(index.node_count(), 0),
        next_node_index(static_cast<std::size_t>(shape.B), 0),
        child_index(index.edge_count(), kInvalidNode),
        legal_action(index.edge_count(), 0) {}

  inline void clear() {
    std::fill(expanded.begin(), expanded.end(), 0);
    std::fill(terminal.begin(), terminal.end(), 0);
    std::fill(parent_index.begin(), parent_index.end(), kInvalidNode);
    std::fill(parent_action.begin(), parent_action.end(), kInvalidAction);
    std::fill(depth.begin(), depth.end(), 0);
    std::fill(player_id.begin(), player_id.end(), 0);
    std::fill(node_visit_count.begin(), node_visit_count.end(), 0);
    std::fill(next_node_index.begin(), next_node_index.end(), 0);
    std::fill(child_index.begin(), child_index.end(), kInvalidNode);
    std::fill(legal_action.begin(), legal_action.end(), 0);
  }

  inline int32_t allocated_nodes(int b) const {
    return next_node_index[index.batch(b)];
  }

  inline bool has_capacity(int b) const {
    return next_node_index[index.batch(b)] < shape.Nmax;
  }

  inline int32_t allocate_node(int b) {
    int32_t n = next_node_index[index.batch(b)];
    if (n >= shape.Nmax) {
      return kInvalidNode;
    }

    next_node_index[index.batch(b)] += 1;
    return n;
  }

  inline void init_root(int b, int32_t player) {
    const int32_t root = allocate_node(b);
    const std::size_t ni = index.node(b, root);
    parent_index[ni] = kInvalidNode;
    parent_action[ni] = kInvalidAction;
    depth[ni] = 0;
    player_id[ni] = player;
    terminal[ni] = 0;
    expanded[ni] = 0;
    node_visit_count[ni] = 0;
  }

  inline int32_t add_child(int b, int parent_n, int action, int32_t player) {
    const int32_t child_n = allocate_node(b);
    const std::size_t child_ni = index.node(b, child_n);

    parent_index[child_ni] = parent_n;
    parent_action[child_ni] = action;
    depth[child_ni] = depth[index.node(b, parent_n)] + 1;
    player_id[child_ni] = player;
    terminal[child_ni] = 0;
    expanded[child_ni] = 0;
    node_visit_count[child_ni] = 0;

    child_index[index.edge(b, parent_n, action)] = child_n;

    return child_n;
  }

  inline bool is_expanded(int b, int n) const {
    return expanded[index.node(b, n)] != 0;
  }

  inline void set_expanded(int b, int n, bool value) {
    expanded[index.node(b, n)] = static_cast<uint8_t>(value);
  }

  inline bool is_terminal(int b, int n) const {
    return terminal[index.node(b, n)] != 0;
  }

  inline void set_terminal(int b, int n, bool value) {
    terminal[index.node(b, n)] = static_cast<uint8_t>(value);
  }

  inline int32_t parent(int b, int n) const {
    return parent_index[index.node(b, n)];
  }

  inline int32_t action_from_parent(int b, int n) const {
    return parent_action[index.node(b, n)];
  }

  inline int32_t child(int b, int n, int a) const {
    return child_index[index.edge(b, n, a)];
  }

  inline void set_child(int b, int n, int a, int32_t child_n) {
    child_index[index.edge(b, n, a)] = child_n;
  }

  inline bool has_child(int b, int n, int a) const {
    return child(b, n, a) != kInvalidNode;
  }

  inline bool is_legal(int b, int n, int a) const {
    return legal_action[index.edge(b, n, a)] != 0;
  }

  inline void set_legal(int b, int n, int a, bool legal) {
    legal_action[index.edge(b, n, a)] = static_cast<uint8_t>(legal);
  }

  inline void clear_legal_actions(int b, int n) {
    for (int a = 0; a < shape.A; ++a) {
      legal_action[index.edge(b, n, a)] = 0;
    }
  }

  inline void increment_node_visit(int b, int n) {
    node_visit_count[index.node(b, n)] += 1;
  }

  inline int32_t node_visits(int b, int n) const {
    return node_visit_count[index.node(b, n)];
  }
};

} // namespace planner::tree_search

#endif // PLANNER_TREE_SEARCH_TREE_BATCHED_TREE_BASE_HPP_
