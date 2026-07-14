#ifndef LMPC__SAFE_SET_HPP_
#define LMPC__SAFE_SET_HPP_

#include <casadi/casadi.hpp>
#include <string>
#include <vector>

namespace lmpc {

// One driven lap's worth of (x_k, u_k, J_k) samples -- DESIGN.md SS2's
// D^{j-1}. u_k is the REALIZED control the collector recorded for the
// k -> k+1 transition (scripts/lmpc_collect_seed_lap.py stores realized
// acceleration, not the commanded velocity setpoint) -- kept, not
// discarded, because trajectory_segment() below hands these controls back
// out as the first solve's warm start, where they must mean the same
// [a, delta] the FHOCP's own u does.
struct SafeSetSample {
  casadi::DM x; // kStateDim x 1, StateIndex order
  casadi::DM u; // kControlDim x 1, ControlIndex order -- zeros if !has_control
  double J;     // cost-to-go: steps remaining to the finish line
  // False only for a lap's final sample (the CSV's last row has no
  // successor state, so its a/delta columns are blank by construction).
  bool has_control;
};

// Holds up to P previous laps of driven data and answers the SS2
// K-nearest-neighbor query the FHOCP's terminal safe set needs: for a query
// state, the K nearest samples FROM EACH stored lap under the weighted
// normalized distance over [vx, epsi, s, ey]. Stacked across laps this gives
// X^j (kStateDim x KP) and J^j (KP x 1), P = number of laps currently loaded.
class SafeSet {
public:
  // DESIGN.md SS2's P: at most this many laps are kept; add_lap() evicts
  // the OLDEST lap once full. Without a cap the per-query work (a KNN pass
  // over every stored lap) grows without bound as laps accumulate --
  // another FPS-degrades-with-progress source, and later laps are faster
  // (lower J) than what they evict anyway, so nothing of value is lost.
  static constexpr std::size_t kMaxLaps = 3;

  // Loads one lap (DESIGN.md SS8's first pass only ever has D^0, i.e. one
  // lap; add_lap() below is what makes P > 1 possible once later laps are
  // recorded).
  explicit SafeSet(const std::string &seed_lap_csv_path);

  // Appends another lap's worth of samples (e.g. the lap just driven,
  // DESIGN.md SS8 step 8), evicting the oldest lap beyond kMaxLaps. The
  // in-memory overload is the closed-loop path (lap-as-iteration:
  // runs/lmpc_drive.py records the driven lap and hands it over at the
  // line, no CSV round-trip); the CSV overload mirrors the constructor.
  void add_lap(std::vector<SafeSetSample> lap);
  void add_lap(const std::string &lap_csv_path);

  casadi_int num_laps() const { return static_cast<casadi_int>(laps.size()); }

  // Number of terminal vertices returned by query(): K from every stored lap.
  casadi_int terminal_point_count(casadi_int K) const;

  double cost_scale() const;

  // Where the recorded data runs out along the track, i.e. the earliest
  // lap-final s over the stored laps (each lap ends where gym's finish
  // detection fired, slightly short of the geometric line). Once a horizon's
  // terminal reference passes this, there is no data ahead of it to query --
  // LMPCController::solve_once switches to its finish-mode terminal set
  // instead of letting the query clamp to (and pull back toward) the last
  // samples.
  double data_end_s() const;

  struct QueryResult {
    casadi::DM X_ss; // kStateDim x (K * num_laps())
    casadi::DM J_ss; // (K * num_laps()) x 1
  };

  // Select exactly K nearest samples independently from each stored lap,
  // sorted by ascending distance within each lap, and concatenate all of
  // them. No affine-rank or global fixed-size reduction is applied.
  // state_scale follows StateIndex order.
  QueryResult query(const casadi::DM &x_query, casadi_int K,
                    const casadi::DM &state_scale) const;

  struct TrajectorySegment {
    casadi::DM x_traj; // kStateDim x (horizon_steps + 1)
    casadi::DM u_traj; // kControlDim x horizon_steps
  };

  // The recorded trajectory segment starting at the stored sample nearest
  // x_query (same normalized metric as query()), horizon_steps transitions
  // long -- the first solve's warm start (LMPCController::
  // seed_warm_start_from_safe_set). A zero-control naive rollout from rest
  // parks the whole horizon at the start line, which locks the terminal
  // query onto D^0's own launch samples and leaves the FHOCP with no
  // forward pull at all (measured directly: every cost term ~0, car never
  // moved); the recorded segment instead hands the QP a reference that
  // actually drives the horizon, exactly what the previous solve's own
  // solution provides on every later step. Indices past the lap's end
  // hold the last sample/control constant (s is non-periodic, one lap).
  // Searches the most recently added lap (the best data available).
  TrajectorySegment trajectory_segment(const casadi::DM &x_query,
                                       casadi_int horizon_steps,
                                       const casadi::DM &state_scale) const;

private:
  std::vector<std::vector<SafeSetSample>> laps;
};

} // namespace lmpc

#endif // LMPC__SAFE_SET_HPP_
