"""Drive the barc_oval map with the native LMPC controller.

Mirrors runs/waypoint_drive.py's shape (env/viewer setup, obs->state->cmd
loop) but for controllers.lmpc.LMPCController instead of Pure
Pursuit/Stanley, and adds two LMPC-specific overlays: the car's own driven
path (DrivenLineOverlay) and the solver's receding-horizon prediction
(RecedingHorizonOverlay) alongside the reference centerline.

Runs the paper's iteration scheme (lap-as-iteration) as ONE continuous
physical episode: env.reset()/controller.reset() fire exactly once, before
iteration 0. Each lap is one LMPC iteration j -- drive until gym's finish
detection fires, append the driven trajectory to the safe set (D^j), then
REBASE the Frenet progress coordinate (LMPCController.begin_next_lap()) so
the same crossing state becomes s=0 for iteration j+1, without touching the
simulator or any native control memory (u_prev, actual_delta, the shifted
u_warm trajectory) -- the vehicle physically drives through the finish line
instead of restarting from rest. Only iteration 0 launches from rest; every
later iteration begins flying, at whatever speed the vehicle crossed the
finish line with. s within an iteration still stays non-periodic (0 at the
iteration start, >= L at its finish) and the cost-to-go J_k = T - k keeps
its single-task meaning per iteration, exactly matching how D^0 was
recorded -- D^0 itself remains a standing-start recording here (not
regenerated as a flying lap), so the safe set's only s~0 data is still at
~zero speed even though live iterations after the first reach s~0 flying;
that is a known quality/feasibility caveat at the lap seam, not addressed
by this change. A crashed or truncated lap is never added -- the safe set's
guarantee rests on every stored trajectory actually finishing.
"""

from __future__ import annotations

import time
from typing import Any

import gymnasium as gym
import numpy as np

import f110_gym  # noqa: F401 - registers f110-v0
from controllers.controller_base import VehicleState
from controllers.lmpc.lap_data import LapSample, build_lap_arrays
from controllers.lmpc.lmpc import LMPCController
from f110_gym.viewer import F110Viewer
from utils.waypoint_view import (
    DrivenLineOverlay,
    RecedingHorizonOverlay,
    WaypointOverlay,
    initial_pose_from_waypoints,
)

MAP = "maps/custom/barc_oval/barc_oval_map"
# Raw geometric centerline, not a mintime raceline -- must be the SAME file
# the seed lap below was recorded against (controllers/lmpc/DESIGN.md SS1:
# the native controller's own s/ey/epsi are meaningless if the two disagree
# about what s means). Converted from ref/Racing-LMPC-ROS2's own BARC oval
# (the track their actual LMPC demo drives, not just their tracking-MPC
# demos) -- a much smaller, tighter track than f110_gym_10 (~17m lap vs
# ~164m, median turn radius ~1.57m vs ~41.7m).
# scripts/lmpc_collect_seed_lap.py has the full rationale for this track's
# speed/lookahead tuning.
CENTERLINE_CSV = "maps/custom/barc_oval/barc_oval_centerline.csv"
SEED_LAP_CSV = "outputs/lmpc_seed_laps/barc_oval_seed_lap.csv"
HORIZON_STEPS = 40
# gym/f110_gym/envs/f110_env.py's F110Env defaults timestep to 0.01 unless a
# caller overrides it -- explicit here rather than editing that vendored
# default (CLAUDE.md: treat gym/ as a black box). 0.025 is the value
# controllers/lmpc/DESIGN.md's N=75 horizon was originally pinned alongside
# (SS8/config discussion) -- must match the seed lap's own dt too
# (scripts/lmpc_collect_seed_lap.py's SIM_TIMESTEP), or D^0's recorded
# states/costs-to-go don't correspond to this controller's own step size.
SIM_TIMESTEP = 0.025
# LMPC iterations (laps) to drive before parking the viewer. Each completed
# lap grows the safe set, so later iterations should be at least as fast as
# earlier ones (the paper's core property).
MAX_ITERATIONS = 20
# Perf-metrics reporting cadence (PerfMonitor below): print one aggregated
# p50/p95/p99 block every this many control steps, instead of a line per
# step -- recom.md asks for exactly this profiling breakdown but explicitly
# wants percentiles, not just an average, so a single periodic block is both
# the more useful view and the one that keeps stdout readable.
PERF_REPORT_INTERVAL_STEPS = 200
# ---------------------------------------------------------------------
# LmpcConfig overrides. Every constant below maps 1:1 to a
# controllers/lmpc/include/lmpc_config.hpp field (see that file for the
# exact objective/constraint each one scales) and is applied via
# CONFIG_OVERRIDES at the bottom of this section -- exposed individually
# here, rather than left to LmpcConfig's own C++ defaults, so tuning this
# (or any other) track doesn't require editing C++ or guessing field
# names blind. Values below match LmpcConfig's own defaults except where
# a comment says otherwise for this track.
#
# This track's safe operating window is unusually narrow (measured
# 2026-07-14, barc_oval): below ~2.6-3.0 m/s gym's own dynamic
# single-track model diverges (documented plant defect, see
# LOW_SPEED_STEER_ZERO_BELOW/RESTORE_AT below), above ~3.16 m/s the
# tightest corner (min radius ~0.97m) exceeds available grip -- both ends
# fail fast (~2s). Traced directly: with every weight below at its
# LmpcConfig default, the min-time pull accelerated the car to 6.3 m/s --
# 2x the seed lap's 3.0 m/s and 2x this track's grip ceiling -- and it
# crashed into the wall at 2.0s (confirmed a real collision, not a solver
# failure: the QP solved successfully every step). DESIGN.md's SS5/SS6
# discussion already diagnoses this exact pattern ("sprint now, brake at
# horizon end") as CORRECT min-time behavior IF THE MODEL IS RIGHT -- the
# actual defect is the uncorrected nominal model overestimating cornering
# grip, which is what the (unimplemented) error-dynamics regression is
# meant to fix. Raising C_U to discourage acceleration papers over that,
# rather than fixing it -- MU below is the more principled lever: derating
# the PLANNER's own friction assumption (not gym's real plant) directly
# shrinks what the QP believes is achievable, which is a reasonable proxy
# for what the regression would learn.

# Safe-set neighbor count K (DESIGN.md SS2) -- neighbors taken PER LAP.
SAFE_SET_K = 16

# Vehicle physical parameters used ONLY by the LMPC's own nominal
# planning model (dynamics/gym_dynamics.hpp), applied via the
# "vehicle_params" dict special-case in LMPCController.__init__ -- NOT
# gym's real simulator params, which stay at DEFAULT_PARAMS regardless
# (the plant should model the actual car; only the planner's belief about
# it is being derated here). mu defaults to 1.0489, matching gym's own
# DEFAULT_PARAMS["mu"] exactly -- i.e. today the planner and the plant
# agree on friction. Lowering MU makes the planner conservative relative
# to the real car: it directly reduces the lateral tire force the nominal
# model predicts is available (GymDynamics's mu*C_Sf/mu*C_Sr terms), so
# min-time solves stop planning cornering speeds the real tires can't
# deliver -- see the block comment above for why this is preferred over
# raising C_U.
MU = 1.0489

# Control bounds: U = {u | u_l <= u <= u_u}.
A_MIN = -9.51
A_MAX = 9.51
DELTA_MIN = -0.4189
DELTA_MAX = 0.4189

# Gym's steering-rate actuator limit (rad/s) -- per-stage
# |delta_t - delta_{t-1}| <= SV_MAX*dt in the FHOCP.
SV_MAX = 3.2

# Gym velocity limits used when converting solved acceleration to the
# public velocity-setpoint action and when scaling vx.
V_MIN = -5.0
V_MAX = 20.0

# ey corridor half-width (X = {x | -EY_MAX <= ey <= EY_MAX}). Overridden
# down from LmpcConfig's default (1.0, sized for f110_gym_10's ~1.5m
# centerline half-width): this track's centerline half-width is only
# ~0.49m minimum, and the vehicle itself is 0.31m wide (f110_env
# DEFAULT_PARAMS) -- 0.25 leaves ~0.08m clearance at the tightest point.
EY_MAX = 0.25

# ---- Cost-term weights (controllers/lmpc/include/lmpc_config.hpp spells
# out the full objective each one scales) ----
# Multiplier on the raw local terminal cost-to-go J^T lambda -- the min-time
# pull. The safe-set window already offsets J to [K-1, ..., 0].
COST_TO_GO_WEIGHT = 1.0
# Reference Hessian weight for the physical six-state terminal error;
# QpBuilder applies the corresponding 0.5 factor explicitly.
TERMINAL_SLACK_WEIGHT = 800.0
# Control effort/rate weights, applied uniformly to the scaled control
# vector (the paper's own c_u/c_d_u -- a plain L2 norm, not a
# per-component-weighted Q-norm; lmpc_config.hpp's own comment has the
# rationale for why a single scalar is correct here, not separate
# accel/steering weights).
C_U = 0.01
C_D_U = 0.5

# Soft ey-corridor slack penalty (exact L1 + quadratic L2 on violation).
EY_SLACK_L1 = 10.0
EY_SLACK_L2 = 100.0

# QP variable-conditioning scale for the vy/omega/epsi state entries
# (StateIndex order) -- reused as-is from the prior BARC-validated port,
# not derived from this track specifically (lmpc_config.hpp's own comment
# has the rationale for why these three are the exception).
SCALE_X_VY = 2.0
SCALE_X_OMEGA = 2.0
SCALE_X_EPSI = 0.5

# "qrqp" or "ipopt" -- lmpc_config.hpp's own comment has the tradeoffs.
SOLVER_NAME = "ipopt"

CONFIG_OVERRIDES: dict[str, Any] = {
    "K": SAFE_SET_K,
    "vehicle_params": {"mu": MU},
    "a_min": A_MIN,
    "a_max": A_MAX,
    "delta_min": DELTA_MIN,
    "delta_max": DELTA_MAX,
    "sv_max": SV_MAX,
    "v_min": V_MIN,
    "v_max": V_MAX,
    "ey_max": EY_MAX,
    "cost_to_go_weight": COST_TO_GO_WEIGHT,
    "terminal_slack_weight": TERMINAL_SLACK_WEIGHT,
    "c_u": C_U,
    "c_d_u": C_D_U,
    "ey_slack_l1": EY_SLACK_L1,
    "ey_slack_l2": EY_SLACK_L2,
    "scale_x_vy": SCALE_X_VY,
    "scale_x_omega": SCALE_X_OMEGA,
    "scale_x_epsi": SCALE_X_EPSI,
    "solver_name": SOLVER_NAME,
}
# Controlled-brake fallback for solve failures. Without the SS5/SS6 error
# regression the nominal model overestimates cornering grip a little above
# the demonstrated speeds, so the min-time QP's (individually rational)
# sprint-and-brake plans can carry the car beyond where its own
# linearization stays solvable; braking back toward demonstrated territory
# while holding the last steering recovers the solver (measured 2026-07-13:
# extended iteration 2 from 16.7s to 37.6s of a ~44s lap). The setpoint
# decrement maps through gym's braking-branch P gain (10*a_max/-v_min ~ 19)
# to a firm but not full-lock ~4.8 m/s^2. If the solver stays down through
# a full window of consecutive fallback steps, the iteration is abandoned.
FALLBACK_BRAKE_DELTA_V = 0.25
MAX_CONSECUTIVE_FALLBACK_STEPS = 80

ZOOM = 1.0  # > 1 -> Zoom out; < 1 -> Zoom in
WINDOW_WIDTH = 1000
WINDOW_HEIGHT = 800

# F110 Gym's own single-track model diverges when handed nonzero steering
# at low speed (measured this project: raw sim omega reached -26 rad/s at
# 0.9 m/s under small, rate-limited steering commands -- a plant-side
# defect no controller-side fix can compensate for; confirmed again
# 2026-07-14 mid-lap, not just at launch: braking through a real hairpin
# under the corrected nominal model, gym's own raw sim omega diverged to
# ~1e42 within a handful of steps as v crossed zero while steering was
# still nonzero). Same guard, same measured-safe thresholds, as
# scripts/lmpc_collect_seed_lap.py. Applied UNCONDITIONALLY, any time speed
# is low, not just through the one-time launch crossing -- this is a
# plant-level speed regime, not a launch-specific one. This is an
# actuator-level safety mask, not a second controller -- the LMPC still
# plans/solves every step, only what reaches gym's action is suppressed.
LOW_SPEED_STEER_ZERO_BELOW = 1.0
LOW_SPEED_STEER_RESTORE_AT = 3.0


def apply_low_speed_steering_guard(steer: float, speed: float) -> float:
    if speed >= LOW_SPEED_STEER_RESTORE_AT:
        return steer
    if speed <= LOW_SPEED_STEER_ZERO_BELOW:
        return 0.0
    ramp = (speed - LOW_SPEED_STEER_ZERO_BELOW) / (
        LOW_SPEED_STEER_RESTORE_AT - LOW_SPEED_STEER_ZERO_BELOW
    )
    return steer * ramp


def obs_to_vehicle_state(obs: dict[str, Any]) -> VehicleState:
    ego = int(obs["ego_idx"])
    return VehicleState(
        x=float(obs["poses_x"][ego]),
        y=float(obs["poses_y"][ego]),
        yaw=float(obs["poses_theta"][ego]),
        speed=float(obs["linear_vels_x"][ego]),
    )


def raw_velocity_state(f110_env: Any, ego_idx: int) -> tuple[float, float, float]:
    """(vx, vy, omega) in the body frame, from the simulator's raw state.

    gym's public obs dict hardcodes linear_vels_y to 0.0 regardless of
    actual slip -- vy has to be reconstructed from the raw slip angle
    instead: vx = v*cos(beta), vy = v*sin(beta) (same as
    scripts/lmpc_collect_seed_lap.py).
    """
    raw_state = f110_env.sim.agents[ego_idx].state  # [x,y,delta,v,psi,yaw_rate,beta]
    v = float(raw_state[3])
    beta = float(raw_state[6])
    return v * np.cos(beta), v * np.sin(beta), float(raw_state[5])


def raw_steer_and_speed(f110_env: Any, ego_idx: int) -> tuple[float, float]:
    """(actual steering angle, raw scalar speed v) from the simulator's raw state.

    Both are gym's own raw_state[2]/[3] -- neither is what this controller
    last commanded. Steering: gym applies steer through a 2-step delay
    buffer and a rate-limited PID (base_classes.py::update_pose), so the
    commanded and actual angles diverge. Speed: gym's ST model integrates
    dv/dt = a directly with NO other coupling (vehicle_dynamics_st's f[3] =
    u[1] exactly) -- unlike vx = v*cos(beta), which also picks up a
    -v*sin(beta)*beta_dot term under slip, finite-differencing v alone
    recovers the realized acceleration gym's PID actually applied, exactly
    (RaceCar.accel exists but is dead -- update_pose never writes the accl
    it computes back into it, only resets it to 0 on collision -- so it
    can't be read directly and must be reconstructed this way instead).
    """
    raw_state = f110_env.sim.agents[ego_idx].state  # [x,y,delta,v,psi,yaw_rate,beta]
    return float(raw_state[2]), float(raw_state[3])


class PerfMonitor:
    """Accumulates per-step phase timings (ms) and prints ONE compact
    p50/p95/p99 summary block every `interval_steps` steps, instead of a
    line per step -- recom.md's requested profiling
    (t_rollout+lin/t_knn/t_set-params/t_solver/t_postcheck/t_env/t_render),
    without the messy stdout a per-step print would produce. Tumbling
    window: each report covers exactly the steps since the previous one,
    then the buffers clear -- so a stretch of the run that gets slower (or
    recovers) shows up as a new block, not smeared into a running average
    since program start.
    """

    METRICS = (
        "rollout+lin",
        "knn",
        "set-params",
        "solver",
        "postcheck",
        "env",
        "render",
    )
    SLACK_METRICS = (
        "slack_vx",
        "slack_vy",
        "slack_omega",
        "slack_epsi",
        "slack_s",
        "slack_ey",
    )

    def __init__(self, interval_steps: int = PERF_REPORT_INTERVAL_STEPS) -> None:
        self._interval_steps = interval_steps
        self._samples: dict[str, list[float]] = {name: [] for name in self.METRICS}
        self._terminal_slack_samples: list[np.ndarray] = []

    def record(
        self, *, terminal_slack: np.ndarray | None = None, **metrics_ms: float
    ) -> None:
        for name in self.METRICS:
            self._samples[name].append(metrics_ms[name])
        if terminal_slack is not None:
            self._terminal_slack_samples.append(terminal_slack)
        if len(self._samples["env"]) >= self._interval_steps:
            self.report()

    def report(self) -> None:
        n = len(self._samples["env"])
        if n == 0:
            return
        rows = [
            f"{'metric':<12}{'n':>5}{'mean':>8}{'p50':>8}{'p95':>8}{'p99':>8}{'max':>8}"
        ]
        total = np.zeros(n)
        for name in self.METRICS:
            values = np.asarray(self._samples[name])
            total += values
            rows.append(
                f"{name:<12}{n:>5}{values.mean():>8.2f}"
                f"{np.percentile(values, 50):>8.2f}"
                f"{np.percentile(values, 95):>8.2f}"
                f"{np.percentile(values, 99):>8.2f}"
                f"{values.max():>8.2f}"
            )
        rows.append(
            f"{'total':<12}{n:>5}{total.mean():>8.2f}"
            f"{np.percentile(total, 50):>8.2f}"
            f"{np.percentile(total, 95):>8.2f}"
            f"{np.percentile(total, 99):>8.2f}"
            f"{total.max():>8.2f}"
        )
        print(f"--- perf (last {n} steps, ms) ---")
        print("\n".join(rows))
        if self._terminal_slack_samples:
            slack = np.abs(np.asarray(self._terminal_slack_samples))
            slack_rows = [f"{'terminal slack':<16}{'p50':>10}{'p95':>10}{'max':>10}"]
            for idx, name in enumerate(self.SLACK_METRICS):
                values = slack[:, idx]
                slack_rows.append(
                    f"{name:<16}{np.percentile(values, 50):>10.4f}"
                    f"{np.percentile(values, 95):>10.4f}{values.max():>10.4f}"
                )
            norm_inf = slack.max(axis=1)
            slack_rows.append(
                f"{'slack_norm_inf':<16}{np.percentile(norm_inf, 50):>10.4f}"
                f"{np.percentile(norm_inf, 95):>10.4f}{norm_inf.max():>10.4f}"
            )
            print("\n".join(slack_rows))
            self._terminal_slack_samples.clear()
        for name in self.METRICS:
            self._samples[name].clear()


def main() -> None:
    env = gym.make("f110-v0", map=MAP, num_agents=1, timestep=SIM_TIMESTEP)
    f110_env: Any = env.unwrapped
    ego_idx = 0
    dt = float(f110_env.timestep)

    controller = LMPCController(
        centerline_csv=CENTERLINE_CSV,
        seed_lap_csv=SEED_LAP_CSV,
        dt=dt,
        horizon_steps=HORIZON_STEPS,
        config_overrides=CONFIG_OVERRIDES,
    )
    controller.attach_raw_velocity_state(lambda: raw_velocity_state(f110_env, ego_idx))
    controller.attach_raw_steering_angle(
        lambda: raw_steer_and_speed(f110_env, ego_idx)[0]
    )

    initial_pose = initial_pose_from_waypoints(controller.waypoints)
    waypoint_overlay = WaypointOverlay(controller.waypoints)
    driven_line_overlay = DrivenLineOverlay()
    horizon_overlay = RecedingHorizonOverlay(controller)
    viewer = F110Viewer.from_env(
        f110_env,
        width=WINDOW_WIDTH,
        height=WINDOW_HEIGHT,
        target_fps=60.0,
        initial_zoom=ZOOM,
        callbacks=[waypoint_overlay, driven_line_overlay, horizon_overlay],
    )

    # Single full-episode reset -- everything after this point drives
    # continuously across lap boundaries (module docstring has the
    # rationale). Only iteration 0 launches from rest.
    obs, _info = env.reset(options={"poses": initial_pose})
    controller.reset()
    sim_t = 0.0

    state = obs_to_vehicle_state(obs)
    controller.update(state, sim_t)
    actual_delta, raw_speed = raw_steer_and_speed(f110_env, ego_idx)
    samples = [LapSample(controller.native_state.copy(), actual_delta, raw_speed)]

    viewer.update(obs)
    viewer.render()

    crashed = False
    last_steer = 0.0
    fallback_steps = 0
    perf = PerfMonitor()

    for iteration in range(MAX_ITERATIONS):
        if crashed or viewer.closed:
            break
        lap_start_t = sim_t
        # No env.reset() happens between iterations, so gym's cumulative
        # lap_counts only ever increases by 1 per finish crossing -- the
        # target for iteration j is simply j+1 (0-indexed), unlike the old
        # per-iteration-reset loop where the reset obs's stale lap_counts
        # made ">= 1.0" ambiguous after the first lap.
        target_lap_count = float(iteration + 1)
        lap_done = False

        while not (lap_done or crashed or viewer.closed):
            terminal_slack = None
            try:
                cmd = controller.control()
                terminal_slack = controller.last_terminal_slack()
                fallback_steps = 0
                steer = apply_low_speed_steering_guard(cmd.steering, state.speed)
                velocity = cmd.velocity
            except RuntimeError as e:
                # FALLBACK_BRAKE_DELTA_V's comment has the rationale; this is
                # an actuator-level safety net like the steering guard above,
                # not a second controller -- the LMPC is re-attempted every step.
                # Every failure is printed, not just the one that exhausts the
                # fallback budget: LMPCController::control() no longer retries
                # internally (single solve per step, bounded by the solver's
                # own max_iter), so each RuntimeError here is one genuine,
                # unmasked solver report -- surfacing all of them is what lets
                # an infeasibility be diagnosed from where it STARTS, not just
                # from the final "gave up" message.
                fallback_steps += 1
                print(f"iteration {iteration} step fallback({fallback_steps}): {e}")
                if fallback_steps > MAX_CONSECUTIVE_FALLBACK_STEPS:
                    print(
                        f"iteration {iteration}: solver never recovered under "
                        f"the fallback brake"
                    )
                    crashed = True
                    break
                steer = last_steer
                velocity = max(state.speed - FALLBACK_BRAKE_DELTA_V, 0.0)
            last_steer = steer
            action = np.array([[steer, velocity]], dtype=np.float64)

            t_env0 = time.perf_counter()
            obs, _reward, _terminated, truncated, _info = env.step(action)
            t_env1 = time.perf_counter()
            sim_t += dt
            viewer.update(obs)
            viewer.render()
            t_render1 = time.perf_counter()
            perf.record(
                **controller.last_timings(),
                env=(t_env1 - t_env0) * 1000.0,
                render=(t_render1 - t_env1) * 1000.0,
                terminal_slack=terminal_slack,
            )

            crashed = bool(f110_env.collisions[ego_idx]) or truncated
            if not crashed:
                state = obs_to_vehicle_state(obs)
                controller.update(state, sim_t)
                actual_delta, raw_speed = raw_steer_and_speed(f110_env, ego_idx)
                samples.append(
                    LapSample(controller.native_state.copy(), actual_delta, raw_speed)
                )
            lap_done = (
                not crashed and float(obs["lap_counts"][ego_idx]) >= target_lap_count
            )

        if viewer.closed:
            break
        if crashed:
            print(
                f"iteration {iteration}: crashed after {sim_t - lap_start_t:.2f}s "
                "-- lap NOT added"
            )
            break

        x_lap, u_lap, J_lap = build_lap_arrays(samples, dt)
        controller.add_lap(x_lap, u_lap, J_lap)
        print(
            f"iteration {iteration}: lap completed in {sim_t - lap_start_t:.2f}s "
            f"(transitions={u_lap.shape[1]}, states={x_lap.shape[1]})"
        )

        if iteration + 1 < MAX_ITERATIONS:
            # Same physical instant, rebased to s=0 for the next iteration --
            # no env.step() happens between x_T^j (the sample just appended
            # above) and this rebased x_0^(j+1).
            controller.begin_next_lap()
            controller.update(state, sim_t)
            actual_delta, raw_speed = raw_steer_and_speed(f110_env, ego_idx)
            samples = [
                LapSample(controller.native_state.copy(), actual_delta, raw_speed)
            ]

    perf.report()  # flush whatever partial window hasn't hit the interval yet

    while not viewer.closed:
        viewer.render()

    env.close()


if __name__ == "__main__":
    main()
