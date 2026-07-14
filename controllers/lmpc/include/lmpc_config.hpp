#ifndef LMPC__LMPC_CONFIG_HPP_
#define LMPC__LMPC_CONFIG_HPP_

#include <casadi/casadi.hpp>
#include <string>
#include <vector>

#include "dynamics/common.hpp"

namespace lmpc {

// Tunables settable from Python at construction time (LMPCController's //
// pybind11 binding). Every symbol here is traceable to a section of
// controllers/lmpc/DESIGN.md -- see the per-field comments.
struct LmpcConfig {
  // Must match the simulator's own control period -- the ATV/discretized
  // dynamics are linearized at this dt, so a mismatch against what the sim
  // actually steps at corrupts the linearization from the first solve.
  double dt = 0.025;

  // Receding-horizon length N (DESIGN.md SS3/SS4).
  casadi_int horizon_steps = 75;

  // Raw centerline CSV (x_m, y_m, w_tr_right_m, w_tr_left_m) this track's s
  // coordinate is defined against -- must be the SAME file
  // scripts/lmpc_collect_seed_lap.py projected onto to produce
  // seed_lap_csv_path below, or curvature(s) and the state's own s
  // disagree about what s means. No default: path is repo-relative and
  // working-directory dependent, so the caller (Python) supplies it rather
  // than this layer guessing a repo root.
  std::string centerline_csv_path;

  // D^0 (or later D^{j-1}) seed-lap CSV in
  // scripts/lmpc_collect_seed_lap.py's format -- DESIGN.md SS2/SS8. Same
  // no-default reasoning as centerline_csv_path.
  std::string seed_lap_csv_path;

  // Vehicle physical parameters -- defaults mirror
  // gym/f110_gym/envs/f110_env.py's DEFAULT_PARAMS (dynamics/common.hpp).
  dynamics::VehicleParams vehicle_params{};

  // Safe-set neighbor count K (DESIGN.md SS2, pinned from upstream's
  // barc_lmpc.param.yaml num_ss_pts_per_lap): neighbors taken PER LAP. P
  // (laps kept in the safe set) is not a separate field here -- it is
  // simply how many laps are loaded into the SafeSet the controller was
  // built with (1 for D^0 alone; grows only if add_lap() is called, not
  // yet wired up for this first pass -- DESIGN.md SS8 step 8).
  casadi_int K = 32;

  // U = {u | u_l <= u <= u_u} (DESIGN.md SS3).
  double a_min = -9.51;
  double a_max = 9.51;
  double delta_min = -0.4189;
  double delta_max = 0.4189;

  // Gym's steering actuator is RATE-limited (dynamic_models.py's
  // steering_constraint, sv in [sv_min, sv_max]): a commanded angle is
  // approached at at most sv_max rad/s, so the realized delta can move at
  // most sv_max*dt per control step. Without the matching per-stage rate
  // constraint in the FHOCP (QpBuilder), the plan treats delta as
  // instantaneous and happily flips full lock (+/-0.4189, ~0.84 rad) in a
  // single 0.025s step -- ~10x beyond what the plant can execute -- which
  // was measured (2026-07-13) to chatter the real simulator into its own
  // documented low-speed steering divergence (omega hit -420 rad/s in the
  // RAW sim state) during launch. Default mirrors gym DEFAULT_PARAMS.
  double sv_max = 3.2;

  // Gym velocity limits used when converting solved acceleration to the
  // public velocity-setpoint action and when scaling vx.
  double v_min = -5.0;
  double v_max = 20.0;

  // The ey half of X = {x | -W/2 <= ey <= W/2} (DESIGN.md SS3). Default is
  // conservative for this track: f110_gym_10's centerline half-width is
  // ~1.44-1.52 m (min over the loaded centerline), the vehicle itself is
  // 0.31 m wide (f110_env DEFAULT_PARAMS), so 1.0 m leaves a comfortable
  // margin without needing this controller to know the local half-width at
  // every s.
  double ey_max = 1.0;

  // ---- Cost-term weights ----------------------------------------------
  // The FHOCP's objective is (all in scaled/normalized coordinates):
  //
  //   cost_to_go_weight * J^T lambda / scaling.j          (min-time pull)
  // + sum_t c_a*a_t^2 + c_delta*delta_t^2                 (control effort)
  // + sum_t c_d_a*da_t^2 + c_d_delta*ddelta_t^2           (control rate)
  // + terminal_slack_weight
  //     * || terminal_slack_state o slack_N ||^2          (safe-set anchor)
  // + ey_slack_l1 * sum sigma + ey_slack_l2 * sum sigma^2 (soft corridor)
  //
  // The RATIOS between these decide how aggressively the controller seeks
  // time over shadowing the demonstrated data: raising cost_to_go_weight
  // (or lowering the terminal anchor / rate weights) trades conservatism
  // for speed. Caveat while SS5/SS6's error regression is unimplemented:
  // the nominal model overestimates cornering grip above the demonstrated
  // speeds, so aggressive settings buy sprints that end in real slides,
  // not lap time.

  // Multiplier on the normalized terminal cost-to-go J^T lambda -- the
  // ONLY term that rewards finishing sooner (the per-stage min-time
  // indicator is constant over the horizon and omitted). At 1.0 the
  // normalized J gradient is small against the effort/anchor terms below,
  // which reads as "not actually seeking the fastest path".
  double cost_to_go_weight = 1.0;

  // Per-control effort/rate weights applied in scaled control coordinates.
  double c_a = 0.0;
  double c_delta = 0.01;
  double c_d_a = 0.1;
  double c_d_delta = 0.1;

  // Penalty on normalized terminal safe-set mismatch. Dynamic-state mismatch
  // is penalized less than epsi/s/ey mismatch (per-state weights below).
  double terminal_slack_weight = 100.0;

  // Per-state weights inside the terminal-slack norm, StateIndex order
  // [vx, vy, omega, epsi, s, ey]. With a 2-point simplex the demonstrated
  // manifold is locally a line segment, so this soft anchor is ALSO the
  // exploration knob: the ey/epsi entries decide how far off the
  // demonstrated line the terminal may drift (corner-cutting), the vx
  // entry how far past demonstrated speed it may aim.
  std::vector<double> terminal_slack_state = {1.0, 0.25, 0.25, 4.0, 4.0, 4.0};

  // Exact-plus-quadratic penalty on the per-stage ey corridor slack
  // (QpBuilder softens the ey box with a >= 0 slack per stage). A HARD ey
  // box makes the QP instantly infeasible whenever the measured x_0 -- or
  // any linearized stage under the steering-rate limit -- can't sit inside
  // the corridor, killing the solve exactly when recovery matters
  // (measured 2026-07-13: iteration 2 outran its data to ~5.6 m/s and
  // qrqp failed with "Failed to calculate search direction" twice). The l1
  // weight is sized so the marginal cost of the first meter of violation
  // (~10) dwarfs anything the cost-to-go can offer (J spans [0, 1] after
  // scaling.j normalization), which is what keeps the penalty exact: slack
  // stays 0 whenever the hard corridor is achievable.
  double ey_slack_l1 = 10.0;
  double ey_slack_l2 = 100.0;

  // DESIGN.md's open item: variable scaling. The QP's decision vector mixes
  // wildly different physical magnitudes (s up to ~164m, vx O(1-20), ey
  // O(1), a O(1-9.5), delta O(0.1)) in one KKT system -- qrqp solves this
  // reliably only once every variable is normalized to O(1). scale_x/scale_u
  // are derived from THIS vehicle/track's own limits (v_max above, ey_max,
  // a_max, delta_max, and the loaded Track's own length -- computed in
  // LMPCController's constructor, not stored here since Track isn't built
  // yet at LmpcConfig construction time), NOT copied from the prior port's
  // BARC-specific constants (that mismatch was flagged there as a real
  // bug). scale_x's vy/omega/epsi entries (2.0/2.0/0.5, StateIndex order)
  // are the one exception: reused as-is from that prior, empirically-
  // validated port, since they were reasonable order-of-magnitude defaults
  // rather than vehicle-specific derivations in the first place, and
  // deriving fresh values for them would be guessing, not improving.
  double scale_x_vy = 2.0;
  double scale_x_omega = 2.0;
  double scale_x_epsi = 0.5;

  // DESIGN.md SS7: qrqp for correctness-first bring-up.
  std::string solver_name = "qrqp";
};

} // namespace lmpc

#endif // LMPC__LMPC_CONFIG_HPP_
