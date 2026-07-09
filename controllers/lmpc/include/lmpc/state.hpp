#pragma once

#include <array>
#include <cstddef>
#include <memory>
#include <string>
#include <vector>

namespace f110_gym_lmpc {

struct GymVehicleState {
  double x = 0.0;
  double y = 0.0;
  double yaw = 0.0;
  double v_x = 0.0;
  double v_y = 0.0;
  double omega = 0.0;
};

struct RacingLmpcState {
  double s = 0.0;
  double e_y = 0.0;
  double e_psi = 0.0;
  double v_x = 0.0;
  double v_y = 0.0;
  double omega = 0.0;

  std::array<double, 6> to_array() const;
};

struct PaperLmpcState {
  double v_x = 0.0;
  double v_y = 0.0;
  double omega = 0.0;
  double e_psi = 0.0;
  double s = 0.0;
  double e_y = 0.0;

  std::array<double, 6> to_array() const;
};

struct LmpcControlCommand {
  double steering = 0.0;
  double velocity = 0.0;
};

struct LmpcReference {
  double curvature = 0.0;
  double target_speed = 3.0;
  double left_bound = 1.0;
  double right_bound = 1.0;
  std::vector<double> curvature_sequence;
};

struct LmpcConfig {
  std::size_t horizon = 15;
  double dt = 0.01;
  double target_speed = 3.0;
  double max_cpu_time = 0.08;
  int max_iter = 100;
  double tolerance = 1e-3;
  double track_half_width = 1.0;
  double max_drive_force = 5.0;
  double max_brake_force = -10.0;
  double max_steer = 0.42;
  // Hard rate limit on the OUTPUT steering command (not the QP's own
  // control_rate_weight, which is only a soft cost -- the primary solve
  // could still choose a large single-step swing if the cost tradeoff
  // favored it, and fallback_command() had no rate limiting at all).
  // Diagnosed via direct instrumentation: fallback_command()'s steering,
  // driven by a noisy e_psi, swung the full +-max_steer range within a
  // single 0.025s step (implying ~34 rad/s -- no real actuator can do
  // this); the real vehicle couldn't track it, yawed violently, producing
  // MORE e_psi noise next step, commanding another full swing -- a
  // classic actuator-rate-limit instability (measured: vehicle omega grew
  // from 0 to -1.4 million rad/s over ~10 steps and never recovered).
  // 10.0 matches the max_steer_rate already declared (but never enforced)
  // in make_base_config's SteerConfig.
  double max_steer_rate = 10.0;
  // Steering is HELD AT EXACTLY 0 while v_x < low_speed_steer_zero_below,
  // then ramps linearly back to full authority by v_x ==
  // low_speed_steer_restore_at. Works around a confirmed bug in the F110
  // Gym simulator itself (gym/f110_gym/envs/dynamic_models.py's
  // vehicle_dynamics_st): it hard-switches from a kinematic to a dynamic
  // (slip-angle) vehicle model at |v| < 0.5 m/s, and the dynamic model
  // diverges catastrophically (omega runs away to 1e5+ rad/s within ~10
  // steps) if ANY non-zero steering angle is applied at that exact
  // crossing -- verified directly: sending a raw constant [steer=0.0,
  // v=1.0] command through env.step() (bypassing our controller entirely)
  // crosses v=0.5 cleanly with omega staying exactly 0; the same test
  // with steer=0.04 diverges. A soft taper from v=0 (e.g. scale=v/0.7)
  // was measured INSUFFICIENT -- at the critical v=0.5 crossing it only
  // reduced steer by ~29%, which still triggered the divergence (delayed,
  // not prevented). Needs a genuine zero-steering floor through the
  // danger zone, not a gradual taper starting from rest. Not a bug in our
  // LMPC, QP, or fallback logic -- gym/f110_gym/envs/ is vendored/
  // authoritative simulator code (see CLAUDE.md), so this compensates at
  // the command-output boundary rather than patching vendored physics.
  double low_speed_steer_zero_below = 2.0;
  double low_speed_steer_restore_at = 3.0;
  double wheelbase = 0.33;
  double track_length = 1.0e6;
  // Single FLAT hard cap on v_x, used for every stage and the terminal
  // state alike -- NOT curvature-varying inside the FHOCP. This mirrors how
  // upstream racing_mpc.cpp actually limits speed in LMPC/learning mode: its
  // vel_ref_/max_vel_ref_diff mechanism only exists in the (unused by us)
  // tracking-MPC cost path (build_tracking_cost); build_lmpc_cost never
  // references vel_ref_ at all. In true LMPC mode the ONLY explicit speed
  // defense is this flat box bound (upstream's BARC config hardcodes a
  // conservative 3.0 m/s); everything beyond that comes from the learned
  // safe set over successive laps, exactly as the paper describes. Should
  // be set (by the caller, e.g. controller.py from the curvature-limited
  // speed profile's minimum) conservatively enough that the tightest corner
  // on the actual track is tire-safe at this speed -- NOT derived from
  // target_speed, which is sized for average-case operating envelope/QP
  // scaling and is routinely much faster than the tightest corner allows.
  double max_speed = 5.0;
  // Per-stage v_x bound = clamp(sqrt(lateral_accel_limit / (|kappa(s)| +
  // eps)), min_corner_speed, max_speed), evaluated at each stage's predicted
  // s via the existing curvature_at_s() lookup. This is the SAME tire
  // lateral-acceleration formula scripts/generate_lmpc_trajectory.py uses to
  // build the reference/seed speed profile (ay_safe, falling back to
  // mue*g -- f110.ini defaults to mue=0.5, g=9.81 -> 4.905), not an
  // independently-invented limit. Necessary because empirically neither
  // extreme of a FLAT v_max worked on this track: capping at the profile's
  // min starved the terminal safe-set constraint (xN(VX) below nearly every
  // safe-set neighbor's recorded speed, forcing terminal_slack_ to absorb
  // the gap every solve, solver_success collapsed to ~19%); capping at the
  // profile's max let the plan cruise at ~9-10 m/s into a corner the tires
  // can't hold (measured: steering pinned at max for 6+ consecutive steps,
  // e_y climbing straight into the wall) -- the terminal cost-to-go alone
  // did not suppress this. A flat cap cannot fit a track with this much
  // curvature variation; max_speed above remains the absolute ceiling used
  // for scaling/model envelope/command clamps.
  double lateral_accel_limit = 4.905;
  // Multiplies lateral_accel_limit ONLY for the in-QP hard bound above
  // (corner_speed_bound), NOT the reference/seed generation. Tried 0.85
  // (real racing lines conventionally target ~85-90% of theoretical grip)
  // and measured it WORSE, not better: distance 38.43m -> 16.18m,
  // solver_success 47% -> 23%. Root cause: scripts/generate_lmpc_trajectory.py
  // (and therefore the loaded seed lap D^0's recorded v_x) uses the FULL,
  // unscaled lateral_accel_limit -- applying a factor here only makes the
  // QP's hard bound systematically slower (sqrt(0.85)=92%) than the safe
  // set's own recorded speed at the same location, reintroducing the
  // terminal-constraint-vs-hard-bound conflict from the min(profile) flat-cap
  // experiment (see native_controller.cpp / lmpc-performance-and-robustness
  // memory finding 11), just at smaller magnitude. Left at 1.0 (no-op) by
  // default; if a safety margin is wanted, it MUST also be applied when
  // generating/loading the reference trajectory and seed lap so the QP
  // bound and the safe-set data stay consistent -- do not just tighten this
  // in isolation.
  double corner_speed_safety_factor = 1.0;
  double min_corner_speed = 1.0;
  // fallback_command() gains -- see native_controller.cpp for why these
  // replaced a fixed-gain (-0.6*e_y - 1.2*e_psi) law: diagnosed via direct
  // instrumentation that a run of consecutive solver failures (which makes
  // fallback_command() the ONLY thing driving, for many consecutive steps)
  // at v~9.5 m/s produced a real vehicle spin (omega up to +13 rad/s,
  // e_psi wrapping past pi/2). fallback_lateral_gain is the standard
  // Stanley control law's k (Thrun et al.); it enters through
  // atan2(k*e_y, v), which naturally tapers the lateral correction at high
  // speed instead of applying a fixed gain regardless of speed.
  // fallback_heading_gain replaces the old flat 1.2 with a gentler value.
  double fallback_lateral_gain = 2.0;
  double fallback_heading_gain = 0.6;
  // Speed floor (m/s) used ONLY for the ATV linearization reference of the
  // dynamic single-track model. Its lateral/yaw Jacobian scales like 1/v_x
  // (slip-angle derivatives), so near v_x=0 the A matrix blows up to ~1e5 and
  // the QP goes non-finite. Linearizing at max(v_x, this) keeps the model well
  // conditioned while the actual state/command still use the true low speed, so
  // the car can launch from rest. Does not affect the model above this speed.
  double linearization_speed_floor = 2.0;
  std::size_t max_lap_stored = 3;
  double reg_dist_max = 2.0;
  std::size_t reg_max_points = 96;
  std::size_t reg_max_points_per_lap = 32;
  std::size_t regression_horizon_stride = 0;
  double lateral_weight = 0.0;
  double heading_weight = 0.0;
  double terminal_lateral_weight = 0.0;
  double terminal_heading_weight = 0.0;
  // Base weight for the terminal convex-hull slack's e_y component
  // (weight_i = terminal_slack_position_weight / scale_x_i^2, further
  // scaled down by kTerminalSlackHeadingRatio for heading/yaw-rate states
  // -- see native_controller.cpp for the full derivation). Since
  // scale_x_[e_y] is always 1.0 (see the ctor), this value alone
  // determines e_y's terminal-slack weight directly; every other state's
  // weight is derived from it via scale_x_. Default 1e4 matches the prior
  // uniform kTerminalSlackPenalty's magnitude, so e_y's own penalization
  // is unchanged -- only the OTHER states get properly reweighted
  // relative to their own physical scale.
  double terminal_slack_position_weight = 1.0e4;
  // Corridor (e_y) bound is SOFT, not hard: shrink the raw track half-widths
  // by boundary_margin (a safety pad, independent of vehicle geometry -- the
  // vehicle half-width itself is added separately from the model's chassis
  // config) and allow a single shared slack variable to expand that shrunk
  // corridor back out at quadratic cost boundary_slack_weight. A HARD e_y
  // bound makes the QP genuinely infeasible (not just numerically difficult)
  // the moment any stage's plan needs to cross it -- which happens
  // constantly for a car actually using the track width -- so every such
  // moment was a guaranteed solver failure with no relation to iteration
  // count. Values match upstream racing_mpc.cpp's BARC config
  // (margin=0.1, q_boundary=1000).
  double boundary_margin = 0.1;
  double boundary_slack_weight = 1000.0;
  // Was 1e-3 -- on ui(LON), bounded to +-0.01 (kN, see scale_u_ in the
  // ctor), that gave a max cost contribution of ~1e-7: effectively
  // unregularized, leaving the QP with near-singular directions in the
  // longitudinal-control dimension (could pick a different minimizer each
  // step from near-identical inputs -- part of the frame-to-frame plan
  // "wiggle"). Re-derived for parity with input_weight_steer's own
  // contribution at full input scale, normalized by our own scale_u_ (not
  // upstream's -- same reasoning as scale_x_/terminal_slack_weight_):
  // input_weight_steer * max_steer^2 == input_weight_lon *
  // (max(max_drive_force,-max_brake_force)/1000)^2, i.e. 0.1*0.42^2 ==
  // input_weight_lon*0.01^2 -> input_weight_lon = 176.4 at these defaults.
  double input_weight_lon = 176.4;
  double input_weight_steer = 0.1;
  double control_rate_weight = 0.1;
  double safe_set_cost_weight = 1.0;
  // The velocity command tracks the plan's speed this many steps ahead instead
  // of the immediate next step. The FHOCP has no stage-level progress reward,
  // so its optimal plan defers acceleration to late in the horizon; commanding
  // the next-step velocity would keep the car crawling. Previewing ~0.2 s ahead
  // lets the sim's velocity PID chase the plan's intended speed. Clamped to
  // N-1.
  std::size_t command_preview_steps = 20;
};

struct SparseErrorModel {
  std::array<std::array<double, 6>, 6> A{};
  std::array<std::array<double, 2>, 6> B{};
  std::array<double, 6> C{};
};

struct LmpcSample {
  PaperLmpcState x;
  std::array<double, 2> u{};
  PaperLmpcState x_next;
};

struct FrenetProjection {
  double s = 0.0;
  double e_y = 0.0;
  double heading = 0.0;
  std::size_t segment_index = 0;
};

class CenterlineTrack {
public:
  CenterlineTrack(std::vector<double> x, std::vector<double> y,
                  bool closed = true);

  FrenetProjection project(double x, double y) const;
  RacingLmpcState to_racing_state(const GymVehicleState &state) const;
  PaperLmpcState to_paper_state(const GymVehicleState &state) const;

  double total_length() const;
  const std::vector<double> &s() const;

private:
  std::vector<double> x_;
  std::vector<double> y_;
  std::vector<double> s_;
  bool closed_ = true;
  double total_length_ = 0.0;
};

double normalize_angle(double angle);
PaperLmpcState racing_to_paper(const RacingLmpcState &state);

class NativeLMPCController {
public:
  NativeLMPCController();
  explicit NativeLMPCController(const LmpcConfig &config);
  ~NativeLMPCController();

  void reset();
  void update(const RacingLmpcState &state);
  void set_reference(const LmpcReference &reference);
  void add_initial_lap(const std::vector<std::vector<double>> &x,
                       const std::vector<std::vector<double>> &u,
                       const std::vector<double> &k,
                       const std::vector<double> &t);
  void set_curvature_profile(const std::vector<double> &s,
                             const std::vector<double> &k, double total_length);
  LmpcControlCommand control();

  std::vector<std::array<double, 2>> predicted_horizon() const;
  const SparseErrorModel &error_model() const;
  std::size_t sample_count() const;
  std::size_t completed_laps() const;
  std::size_t lap_sample_count() const;
  std::size_t last_safe_set_points() const;
  double solver_success_rate() const;
  std::string last_solver_status() const;

private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace f110_gym_lmpc
