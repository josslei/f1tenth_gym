#ifndef LMPC__SAFE_SET_HPP_
#define LMPC__SAFE_SET_HPP_

#include <casadi/casadi.hpp>
#include <string>
#include <vector>

namespace lmpc {

// One driven lap's worth of (x_k, J_k) samples -- DESIGN.md SS2's D^{j-1}.
struct SafeSetSample {
  casadi::DM x; // kStateDim x 1, StateIndex order
  double J;     // cost-to-go: steps remaining to the finish line
};

// Holds up to P previous laps of driven data and answers the SS2
// K-nearest-neighbor query the FHOCP's terminal safe set needs: for a query
// state, the K nearest samples FROM EACH stored lap under the weighted
// normalized distance over [vx, epsi, s, ey]. Stacked across laps this gives
// X^j (kStateDim x KP) and J^j (KP x 1), P = number of laps currently loaded.
class SafeSet {
public:
  // A single recorded lap is locally a one-dimensional trajectory, so its
  // well-conditioned local simplex is the line segment between two states.
  // Terminal slack handles mismatch outside that demonstrated manifold.
  static constexpr casadi_int kTerminalSimplexSize = 2;

  // Loads one lap (DESIGN.md SS8's first pass only ever has D^0, i.e. one
  // lap; add_lap() below is what makes P > 1 possible once later laps are
  // recorded).
  explicit SafeSet(const std::string &seed_lap_csv_path);

  // Appends another lap's worth of samples (e.g. the lap just driven,
  // DESIGN.md SS8 step 8) -- P grows by one each call.
  void add_lap(const std::string &lap_csv_path);

  casadi_int num_laps() const { return static_cast<casadi_int>(laps.size()); }

  double cost_scale() const;

  struct QueryResult {
    casadi::DM X_ss; // kStateDim x kTerminalSimplexSize
    casadi::DM J_ss; // kTerminalSimplexSize x 1
  };

  // Build a candidate pool from the K nearest points in each lap, then
  // greedily retain a numerically affinely independent simplex in normalized
  // six-state space. state_scale follows StateIndex order.
  QueryResult query(const casadi::DM &x_query, casadi_int K,
                    const casadi::DM &state_scale) const;

private:
  std::vector<std::vector<SafeSetSample>> laps;
};

} // namespace lmpc

#endif // LMPC__SAFE_SET_HPP_
