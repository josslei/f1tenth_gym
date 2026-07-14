"""Drive the f110_gym_10 map with the native LMPC controller.

Mirrors runs/waypoint_drive.py's shape (env/viewer setup, obs->state->cmd
loop) but for controllers.lmpc.LMPCController instead of Pure
Pursuit/Stanley, and adds two LMPC-specific overlays: the car's own driven
path (DrivenLineOverlay) and the solver's receding-horizon prediction
(RecedingHorizonOverlay) alongside the reference centerline.

Runs the paper's actual iteration scheme (lap-as-iteration): each lap is one
LMPC iteration j -- launch from rest at the common initial state, drive until
gym's finish detection fires, append the driven trajectory to the safe set
(D^j), reset the sim and controller, relaunch. s stays non-periodic and the
cost-to-go J_k = T - k keeps its single-task meaning, exactly matching how
D^0 was recorded; continuous multi-lap driving would instead need flying-lap
data and an unwrapped s/J redefinition (controllers/lmpc/DESIGN.md's seam
discussion). A crashed or truncated lap is never added -- the safe set's
guarantee rests on every stored trajectory actually finishing.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np

import f110_gym  # noqa: F401 - registers f110-v0
from controllers.controller_base import VehicleState
from controllers.lmpc.lmpc import LMPCController
from f110_gym.viewer import F110Viewer
from utils.waypoint_view import (
    DrivenLineOverlay,
    RecedingHorizonOverlay,
    WaypointOverlay,
    initial_pose_from_waypoints,
)

MAP = "maps/custom/f110_gym_10/f110_gym_map"
# Raw geometric centerline, not a mintime raceline -- must be the SAME file
# the seed lap below was recorded against (controllers/lmpc/DESIGN.md SS1:
# the native controller's own s/ey/epsi are meaningless if the two disagree
# about what s means).
CENTERLINE_CSV = "maps/custom/f110_gym_10/f110_gym_centerline.csv"
SEED_LAP_CSV = "outputs/lmpc_seed_laps/f110_gym_10_seed_lap.csv"
HORIZON_STEPS = 30
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
MAX_ITERATIONS = 10
# Cost-term weight overrides, applied onto LmpcConfig by field name --
# controllers/lmpc/include/lmpc_config.hpp spells out the objective each
# weight scales. The RATIOS set how hard the controller chases lap time vs.
# shadowing the demonstrated laps: raise "cost_to_go_weight" (min-time pull)
# or lower "terminal_slack_weight"/"terminal_slack_state" entries (safe-set
# anchor = the exploration leash) to be more aggressive. While DESIGN.md
# SS5/SS6's error regression is unimplemented, aggressive settings buy
# sprints that end in real slides, not lap time -- tune in small steps.
# cost_to_go_weight = 5.0 was measured (2026-07-14, headless) to iterate
# faster (44.52s -> 40.90s -> 39.15s) before the unmodeled grip limit ends
# iteration 2 -- but that sweep was run against a nominal model with a
# curvature bug (understated by 2x, and hardcoded to 0 at the start/finish
# seam), a missing -omega term in beta_dot, an Euler/RK4 discretization
# mismatch against the plant, and a stage-0 steering-rate anchor pinned to
# the last COMMAND instead of the plant's actual angle -- all fixed
# 2026-07-14 (recom.md). 5.0 was tuned to compensate for exactly those
# errors' extra optimism, so it is not validated against the corrected
# model yet -- re-sweep before trusting it as more than a starting point.
CONFIG_OVERRIDES: dict[str, Any] = {"cost_to_go_weight": 5.0}
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
LOW_SPEED_STEER_ZERO_BELOW = 2.0
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


def finalize_lap(
    samples: list[tuple[np.ndarray, float, float]], dt: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build LMPCController.add_lap arrays from one lap's recorded samples.

    samples holds (native_state, actual_delta, raw_speed) per control step,
    all three sampled BEFORE stepping -- so the last sample is the state one
    step short of the crossing and gets no control, exactly the convention
    scripts/lmpc_collect_seed_lap.py::write_seed_lap_csv pins for D^0
    (T = len(samples) - 1 transitions; J_k = T - k).

    Both control columns are REALIZED plant values, not commanded ones
    (raw_steer_and_speed's comment has the full rationale for why each
    needs to be reconstructed from the raw state rather than read off the
    action this loop sent to gym). Storing the commanded versions instead
    corrupts exactly the newest, fastest-driving lap the very next
    iteration's warm start is seeded from.
    """
    x_lap = np.column_stack([x for x, _, _ in samples])  # (6, T+1)
    total_steps = len(samples) - 1
    raw_speed = np.array([v for _, _, v in samples], dtype=np.float64)
    accel = (raw_speed[1:] - raw_speed[:-1]) / dt
    steer = np.array([delta for _, delta, _ in samples[:-1]], dtype=np.float64)
    u_lap = np.vstack([accel, steer])  # (2, T)
    cost_to_go = np.arange(total_steps, -1, -1, dtype=np.float64)
    return x_lap, u_lap, cost_to_go


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

    for iteration in range(MAX_ITERATIONS):
        obs, _info = env.reset(options={"poses": initial_pose})
        controller.reset()
        samples: list[tuple[np.ndarray, float, float]] = []
        t = 0.0

        viewer.update(obs)
        viewer.render()

        crashed = False
        lap_done = False
        last_steer = 0.0
        fallback_steps = 0
        while not (lap_done or crashed or viewer.closed):
            state = obs_to_vehicle_state(obs)
            controller.update(state, t)
            try:
                cmd = controller.control()
                fallback_steps = 0
                steer = apply_low_speed_steering_guard(cmd.steering, state.speed)
                velocity = cmd.velocity
            except RuntimeError as e:
                # FALLBACK_BRAKE_DELTA_V's comment has the rationale; this is
                # an actuator-level safety net like the steering guard above,
                # not a second controller -- the LMPC is re-attempted every step.
                fallback_steps += 1
                if fallback_steps > MAX_CONSECUTIVE_FALLBACK_STEPS:
                    print(
                        f"iteration {iteration}: solver never recovered under "
                        f"the fallback brake: {e}"
                    )
                    crashed = True
                    break
                steer = last_steer
                velocity = max(state.speed - FALLBACK_BRAKE_DELTA_V, 0.0)
            last_steer = steer
            # actual_delta/raw_speed, not the command above -- finalize_lap's
            # comment has the rationale; sampled now (before stepping) to
            # align 1:1 with the native_state also captured before this step.
            actual_delta, raw_speed = raw_steer_and_speed(f110_env, ego_idx)
            samples.append((controller.native_state.copy(), actual_delta, raw_speed))
            action = np.array([[steer, velocity]], dtype=np.float64)

            obs, _reward, _terminated, truncated, _info = env.step(action)
            t += dt
            viewer.update(obs)
            viewer.render()

            # lap_counts is recomputed from the toggle list every step, so
            # the post-step value is trustworthy even right after a reset
            # (the reset obs itself carries the previous episode's stale
            # count -- only ever read it here, post-step).
            crashed = bool(f110_env.collisions[ego_idx]) or truncated
            lap_done = not crashed and float(obs["lap_counts"][ego_idx]) >= 1.0

        if viewer.closed:
            break
        if crashed:
            print(f"iteration {iteration}: crashed after {t:.2f}s -- lap NOT added")
            break

        controller.add_lap(*finalize_lap(samples, dt))
        print(
            f"iteration {iteration}: lap completed in {t:.2f}s ({len(samples)} steps)"
        )

    while not viewer.closed:
        viewer.render()

    env.close()


if __name__ == "__main__":
    main()
