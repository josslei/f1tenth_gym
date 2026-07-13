#ifndef LMPC__LMPC_CONFIG_HPP_
#define LMPC__LMPC_CONFIG_HPP_

#include <casadi/casadi.hpp>
#include <string>

#include "dynamics/common.hpp"

namespace lmpc
{

// Tunables settable from Python at construction time (LMPCController's
// pybind11 binding). Every symbol here is traceable to a section of
// controllers/lmpc/DESIGN.md -- see the per-field comments.
struct LmpcConfig
{
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

  // U = {u | u_l <= u <= u_u} (DESIGN.md SS3). Defaults mirror
  // gym/f110_gym/envs/f110_env.py's DEFAULT_PARAMS: a in [-a_max, a_max],
  // delta in [s_min, s_max] (the simulator's own steering-angle limits).
  double a_min = -9.51;
  double a_max = 9.51;
  double delta_min = -0.4189;
  double delta_max = 0.4189;

  // The ey half of X = {x | -W/2 <= ey <= W/2} (DESIGN.md SS3). Default is
  // conservative for this track: f110_gym_10's centerline half-width is
  // ~1.44-1.52 m (min over the loaded centerline), the vehicle itself is
  // 0.31 m wide (f110_env DEFAULT_PARAMS), so 1.0 m leaves a comfortable
  // margin without needing this controller to know the local half-width at
  // every s.
  double ey_max = 1.0;

  // Phi(w)'s c_u (control effort) and c_du (control-rate) weights
  // (DESIGN.md SS3). Not pinned by the paper or upstream (DESIGN.md's open
  // items) -- placeholder magnitudes for the first dummy-A/B/C pass, to be
  // tuned once the base mechanism is verified to work at all.
  double c_u = 0.01;
  double c_du = 0.1;

  // DESIGN.md SS7: qrqp for correctness-first bring-up.
  std::string solver_name = "qrqp";

  // Floors |vx| used ONLY when evaluating a linearization reference point
  // (never the true state x0, never a warm-start value stored/returned
  // elsewhere) -- GymDynamics reparametrizes to (v, beta) via
  // beta = atan2(vy, vx), whose JACOBIAN is singular at vx = vy = 0 (the
  // VALUE is fine -- atan2(0,0) = 0 -- only its derivative blows up).
  // Every lap launches from rest, so stage 0's linearization point hits
  // this exactly on the very first solve. Flooring vx alone is sufficient:
  // it keeps vx^2+vy^2 bounded away from zero regardless of vy, and the
  // true rest state is still reached in the solved trajectory since x0 is
  // pinned to the real (unfloored) x_k via the QP's equality constraint --
  // this only accepts a small linearization-accuracy tradeoff at that one
  // stage, not a change to what the car is allowed to do.
  double linearization_speed_floor = 1.0;
};

}  // namespace lmpc

#endif  // LMPC__LMPC_CONFIG_HPP_
