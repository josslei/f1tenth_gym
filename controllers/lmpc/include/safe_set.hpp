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
  // POD mirrors used by trajectory_segment()'s first-solve warm-start search.
  // Defined out-of-line since construction needs dynamics::StateIndex.
  SafeSetSample(casadi::DM x_in, casadi::DM u_in, double J_in,
                bool has_control_in);

  casadi::DM x; // kStateDim x 1, StateIndex order
  casadi::DM u; // kControlDim x 1, ControlIndex order -- zeros if !has_control
  double J;     // cost-to-go: steps remaining to the finish line
  // False only for a lap's final sample (the CSV's last row has no
  // successor state, so its a/delta columns are blank by construction).
  bool has_control;

  double vx;
  double epsi;
  double s;
  double ey;
};

// Holds up to P previous laps of driven data. Terminal queries select one
// contiguous K-sample trajectory window from every lap using periodic Frenet
// progress only. Each stored lap remains T+1 states for regression and warm
// starts, while the terminal phase view is x_1..x_T (exactly T samples), so
// the physically identical seam states x_0/x_T are not both counted.
class SafeSet {
public:
  // DESIGN.md SS2's P: at most this many laps are kept; add_lap() evicts
  // the OLDEST lap once full. Without a cap the per-query work (a segment
  // search over every stored lap) grows without bound as laps accumulate --
  // another FPS-degrades-with-progress source, and later laps are faster
  // (lower J) than what they evict anyway, so nothing of value is lost.
  static constexpr std::size_t kMaxLaps = 3;

  // track_length defines periodic distance and continuous s lifting.
  //
  // Loads one lap (DESIGN.md SS8's first pass only ever has D^0, i.e. one
  // lap; add_lap() below is what makes P > 1 possible once later laps are
  // recorded).
  SafeSet(const std::string &seed_lap_csv_path, double track_length);

  // Appends another lap's worth of samples (e.g. the lap just driven,
  // DESIGN.md SS8 step 8), evicting the oldest lap beyond kMaxLaps. The
  // in-memory overload is the closed-loop path (lap-as-iteration:
  // runs/lmpc_drive.py records the driven lap and hands it over at the
  // line, no CSV round-trip); the CSV overload mirrors the constructor.
  void add_lap(std::vector<SafeSetSample> lap);
  void add_lap(const std::string &lap_csv_path);

  casadi_int num_laps() const { return static_cast<casadi_int>(laps.size()); }

  // Number of terminal vertices returned by query_local_segments().
  casadi_int terminal_point_count(casadi_int K) const;

  double cost_scale() const;

  struct QueryResult {
    casadi::DM X_ss; // kStateDim x (K * num_laps())
    casadi::DM J_ss; // (K * num_laps()) x 1
    struct SelectedPointInfo {
      std::size_t lap_index;
      long sample_index;
      long wrap_count;
      double lifted_s;
      double local_J;
    };
    std::vector<SelectedPointInfo> selected;
  };

  // Select a contiguous K-point phase window per lap around the sample with
  // minimum periodic s distance. Selected s values are lifted onto the
  // continuous branch nearest s_query, and each lap's recorded cost-to-go is
  // offset by the window endpoint before being returned.
  QueryResult query_local_segments(double s_query, casadi_int K) const;

  struct TrajectorySegment {
    casadi::DM x_traj; // kStateDim x (horizon_steps + 1)
    casadi::DM u_traj; // kControlDim x horizon_steps
  };

  // The recorded trajectory segment starting at the stored sample nearest
  // x_query under its normalized warm-start metric, horizon_steps transitions
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
  double track_length;
};

} // namespace lmpc

#endif // LMPC__SAFE_SET_HPP_
