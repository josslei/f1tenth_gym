#ifndef PLANNER_TREE_SEARCH_SEARCH_MCTS_SEARCH_HPP_
#define PLANNER_TREE_SEARCH_SEARCH_MCTS_SEARCH_HPP_

namespace planner::tree_search {

class MCTSSearch {
public:
  inline MCTSSearch() = default;

  template <typename Tree, typename Evaluator>
  inline void operator()(Tree &tree, Evaluator &evaluator) const {
    search(tree, evaluator);
  }

  template <typename Tree, typename Evaluator>
  inline void search(Tree &tree, Evaluator &evaluator) const {
    (void)tree;
    (void)evaluator;
  }
};

} // namespace planner::tree_search

#endif // PLANNER_TREE_SEARCH_SEARCH_MCTS_SEARCH_HPP_
