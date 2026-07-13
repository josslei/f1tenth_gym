#ifndef LMPC__SAFE_SET_HPP_
#define LMPC__SAFE_SET_HPP_

#include <casadi/casadi.hpp>
#include <string>
#include <vector>

namespace lmpc
{

// One driven lap's worth of (x_k, J_k) samples -- DESIGN.md SS2's D^{j-1},
// one lap of it. u_k/t_k are not needed for the safe-set query (only for
// the regression, DESIGN.md SS5/SS6, not yet implemented), so they are not
// loaded here.
struct SafeSetSample
{
  casadi::DM x;  // kStateDim x 1, StateIndex order
  double J;      // cost-to-go: steps remaining to the finish line
};

// Holds up to P previous laps of driven data and answers the SS2
// K-nearest-neighbor query the FHOCP's terminal safe set needs: for a query
// state, the K nearest samples FROM EACH stored lap under the weighted
// distance (x^i_k - x)^T D (x^i_k - x), D = diag(0,0,0,0,1,1) (nonzero only
// at IDX_S, IDX_EY -- DESIGN.md SS2's pinned D, i.e. nearest-by-track-
// position). Stacked across laps this gives X^j (kStateDim x KP) and J^j
// (KP x 1), P = number of laps currently loaded.
class SafeSet
{
public:
  // Loads one lap (DESIGN.md SS8's first pass only ever has D^0, i.e. one
  // lap; add_lap() below is what makes P > 1 possible once later laps are
  // recorded).
  explicit SafeSet(const std::string & seed_lap_csv_path);

  // Appends another lap's worth of samples (e.g. the lap just driven,
  // DESIGN.md SS8 step 8) -- P grows by one each call.
  void add_lap(const std::string & lap_csv_path);

  casadi_int num_laps() const {return static_cast<casadi_int>(laps.size());}

  struct QueryResult
  {
    casadi::DM X_ss;  // kStateDim x (K * num_laps())
    casadi::DM J_ss;  // (K * num_laps()) x 1
  };

  // K nearest neighbors (by s, ey) from EACH loaded lap, stacked together.
  // K is clamped to a given lap's sample count if that lap is shorter.
  QueryResult query(const casadi::DM & x_query, casadi_int K) const;

private:
  std::vector<std::vector<SafeSetSample>> laps;
};

}  // namespace lmpc

#endif  // LMPC__SAFE_SET_HPP_
