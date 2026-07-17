"""Drive barc_oval with the byte-identical LearningMPC implementation."""

from __future__ import annotations

import time
from typing import Any

import gymnasium as gym
import numpy as np

import f110_gym  # noqa: F401 - registers f110-v0
from controllers.controller_base import VehicleState
from controllers.lmpc.lmpc import LMPCController
from f110_gym.envs.base_classes import Integrator
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
HORIZON_STEPS = 50
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
PERF_REPORT_INTERVAL_STEPS = 1000
SAFE_SET_K = 16
MU = 1.0489
C_SF = 4.718
C_SR = 5.4562
LF = 0.15875
LR = 0.17145
CG_HEIGHT = 0.074
MASS = 3.74
YAW_INERTIA = 0.04712
A_MIN = -9.51
A_MAX = 9.51
DELTA_MIN = -0.4189
DELTA_MAX = 0.4189
V_MAX = 20.0
VELOCITY_THRESHOLD = 0.8
# The QP constrains the vehicle center, so inflate walls beyond the 0.155 m
# half-width. The BARC oval's minimum half-width is 0.488 m; 0.22 m retains a
# feasible center corridor while allowing for modest heading error.
MAP_MARGIN = 0.22
WAYPOINT_SPACE = 0.2
TERMINAL_SLACK_WEIGHT = 800.0
R_ACCEL = 18.0
R_STEER = 5.0
R_D_ACCEL = 0.1
R_D_STEER = 0.1
EY_SLACK_L2 = 3000.0
OSQP_MAX_ITER = 20000
# Negative/zero values preserve the corresponding OSQP library defaults.
OSQP_SCALING = -1
OSQP_EPS_PRIM_INF = 0.0
OSQP_EPS_ABS = 0.0
OSQP_EPS_REL = 0.0
# Dynamics-error regression (plan.md / ref/lmpc.tex): local weighted ridge
# regression correcting the nominal model's one-step v/omega/beta
# prediction. Off by default -- M/h/lambda need to be picked from data
# before this is worth turning on, not guessed, so they're left at 0 here;
# the native validator raises if regression_enabled=True and any of them
# is left non-positive.
REGRESSION_ENABLED = False
REGRESSION_NUM_NEIGHBORS = 32
REGRESSION_BANDWIDTH = 0.6
REGRESSION_REGULARIZATION = 1.0
# Neighbor-distance metric over z=(x,y,psi,v,omega,beta,a,delta) in R^8 --
# Q must be symmetric PSD (validated natively). Identity is the paper's own
# choice (plan.md: "I will not introduce arbitrary feature scales into the
# mathematical specification"); edit only with a deliberate reason to weight
# some state/control dimensions over others in neighbor selection.
REGRESSION_Q = np.eye(8)

CONFIG_OVERRIDES: dict[str, Any] = {
    "K": SAFE_SET_K,
    "vehicle_params": {
        "mu": MU,
        "C_Sf": C_SF,
        "C_Sr": C_SR,
        "lf": LF,
        "lr": LR,
        "h": CG_HEIGHT,
        "m": MASS,
        "I": YAW_INERTIA,
    },
    "a_min": A_MIN,
    "a_max": A_MAX,
    "delta_min": DELTA_MIN,
    "delta_max": DELTA_MAX,
    "v_max": V_MAX,
    "velocity_threshold": VELOCITY_THRESHOLD,
    "map_margin": MAP_MARGIN,
    "waypoint_space": WAYPOINT_SPACE,
    "terminal_slack_weight": TERMINAL_SLACK_WEIGHT,
    "r_accel": R_ACCEL,
    "r_steer": R_STEER,
    "r_d_accel": R_D_ACCEL,
    "r_d_steer": R_D_STEER,
    "ey_slack_l2": EY_SLACK_L2,
    "osqp_max_iter": OSQP_MAX_ITER,
    "osqp_scaling": OSQP_SCALING,
    "osqp_eps_prim_inf": OSQP_EPS_PRIM_INF,
    "osqp_eps_abs": OSQP_EPS_ABS,
    "osqp_eps_rel": OSQP_EPS_REL,
    "regression_enabled": REGRESSION_ENABLED,
    "regression_num_neighbors": REGRESSION_NUM_NEIGHBORS,
    "regression_bandwidth": REGRESSION_BANDWIDTH,
    "regression_regularization": REGRESSION_REGULARIZATION,
    "regression_Q": REGRESSION_Q,
}
ZOOM = 1.0  # > 1 -> Zoom out; < 1 -> Zoom in
WINDOW_WIDTH = 1000
WINDOW_HEIGHT = 800


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
        "regression",
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
        self._regression_correction_samples: list[float] = []
        self._regression_pool_size = 0

    def record(
        self,
        *,
        terminal_slack: np.ndarray | None = None,
        regression_pool_size: int = 0,
        regression_correction_norm: float = 0.0,
        **metrics_ms: float,
    ) -> None:
        for name in self.METRICS:
            self._samples[name].append(metrics_ms[name])
        if terminal_slack is not None:
            self._terminal_slack_samples.append(terminal_slack)
        self._regression_pool_size = regression_pool_size
        if regression_pool_size > 0:
            self._regression_correction_samples.append(regression_correction_norm)
        if len(self._samples["env"]) >= self._interval_steps:
            self.report()

    def report(self) -> None:
        n = len(self._samples["env"])
        if n == 0:
            return
        rows = [
            f"{'metric':<12}{'n':>5}{'mean':>9}{'p50':>9}{'p95':>9}{'p99':>9}{'max':>9}"
        ]
        total = np.zeros(n)
        for name in self.METRICS:
            values = np.asarray(self._samples[name])
            total += values
            rows.append(
                f"{name:<12}{n:>5}{values.mean():>9.4f}"
                f"{np.percentile(values, 50):>9.4f}"
                f"{np.percentile(values, 95):>9.4f}"
                f"{np.percentile(values, 99):>9.4f}"
                f"{values.max():>9.4f}"
            )
        rows.append(
            f"{'total':<12}{n:>5}{total.mean():>9.4f}"
            f"{np.percentile(total, 50):>9.4f}"
            f"{np.percentile(total, 95):>9.4f}"
            f"{np.percentile(total, 99):>9.4f}"
            f"{total.max():>9.4f}"
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
        if self._regression_correction_samples:
            norms = np.asarray(self._regression_correction_samples)
            print(
                f"regression: pool_size={self._regression_pool_size} "
                f"active_fraction={np.mean(norms > 1e-9):.2f} "
                f"correction_norm p50={np.percentile(norms, 50):.4f} "
                f"p95={np.percentile(norms, 95):.4f} max={norms.max():.4f}"
            )
            self._regression_correction_samples.clear()
        for name in self.METRICS:
            self._samples[name].clear()


def main() -> None:
    env = gym.make(
        "f110-v0",
        map=MAP,
        num_agents=1,
        timestep=SIM_TIMESTEP,
        integrator=Integrator.RK4,
        direct_accel_control=True,
    )
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
    viewer.update(obs)
    viewer.render()

    crashed = False
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
            # control() never raises for a failed SOLVE (ported 2026-07-16
            # to match ref/LearningMPC's own failure handling exactly): on
            # a solve failure it reapplies the previous control unchanged,
            # no fallback brake, no abandon-the-iteration threshold -- the
            # reference has neither, relying on gym's own episode bounds
            # (truncation) as the only eventual exit if a solve never
            # recovers. last_solve_ok() distinguishes a fresh solve from a
            # reapplied stale one, purely for visibility here.
            cmd = controller.control()
            if not controller.last_solve_ok():
                print(
                    f"iteration {iteration} step: QP solve failed, "
                    "reapplying previous control"
                )
            terminal_slack = controller.last_terminal_slack()
            acceleration = cmd.velocity
            action = np.array([[cmd.steering, acceleration]], dtype=np.float64)

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
                regression_pool_size=controller.regression_pool_size(),
                regression_correction_norm=controller.last_regression_correction_norm(),
            )

            crashed = bool(f110_env.collisions[ego_idx]) or truncated
            if not crashed:
                state = obs_to_vehicle_state(obs)
                controller.update(state, sim_t)
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

        print(
            f"iteration {iteration}: lap completed in {sim_t - lap_start_t:.2f}s "
            "(safe set updated internally)"
        )

    perf.report()  # flush whatever partial window hasn't hit the interval yet

    while not viewer.closed:
        viewer.render()

    env.close()


if __name__ == "__main__":
    main()
