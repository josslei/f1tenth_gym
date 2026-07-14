"""Drive one lap with the Pure Pursuit controller and record it as an LMPC D^0 seed.

Headless by default -- this only needs to run once per track to produce a CSV
of (x_k, u_k, J_k) samples in the native LMPC state/control convention pinned
in controllers/lmpc/DESIGN.md SS1: x = [vx, vy, omega, epsi, s, ey], u = [a,
delta]. Pass --visualize to watch the drive in the pyglet viewer.

Pure Pursuit, not Stanley: Stanley's heading-error feedback term reacts
sharply to instantaneous heading error, and at this map's SIM_TIMESTEP
(0.025s, 2.5x coarser than this project's earlier 0.01s work) that reaction
had time to develop into a genuine tire-slip spin-out before the next
correction arrived -- confirmed directly against raw simulator pose (yaw
diverging ~45 degrees from the path heading within ~0.3s while steering sat
pinned at its bound), not fixed by retuning Stanley's own gain in either
direction (lower gain made cross-track error worse, matching a classic
"too sluggish to keep up with curvature" failure, not a stability fix).
Pure Pursuit's geometric "aim at a point ahead" law has no heading-error
term to react sharply in the first place. The recorded lap starts from rest
so D^0 contains the LMPC run's actual initial condition.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np

import f110_gym  # noqa: F401 - registers f110-v0
from controllers.controller_base import VehicleState
from controllers.lmpc.lap_data import LapSample, build_lap_arrays
from controllers.pure_pursuit import DynamicLookaheadDistance, PurePursuit
from utils.waypoint_utils import cumulative_arc_lengths, nearest_waypoint_index
from utils.waypoint_view import initial_pose_from_waypoints

MAP = "maps/custom/barc_oval/barc_oval_map"
# Raw geometric centerline (scripts/generate_centerline.py's output format:
# x_m, y_m, w_tr_right_m, w_tr_left_m -- no heading/speed columns), not a
# mintime-optimized raceline. This matches the paper's own D^0 recipe
# (Section V-C / III: "closed-loop trajectories from a simple low-speed
# center-line tracking controller"), not an optimized trajectory.
WAYPOINTS_CSV = "maps/custom/barc_oval/barc_oval_centerline.csv"
OUTPUT_CSV = "outputs/lmpc_seed_laps/barc_oval_seed_lap.csv"

# gym/f110_gym/envs/f110_env.py's F110Env defaults timestep to 0.01 unless a
# caller overrides it -- explicit here rather than editing that vendored
# default (CLAUDE.md: treat gym/ as a black box). 0.025 is the value
# controllers/lmpc/DESIGN.md's N=75 horizon was originally pinned alongside;
# must match runs/lmpc_drive.py's own SIM_TIMESTEP, or this seed lap's own
# recorded states/costs-to-go don't correspond to what the controller runs
# at.
SIM_TIMESTEP = 0.025

# NOT a conservative "low speed" pass here, unlike f110_gym_10's 3.5 m/s:
# this track (converted from ref/Racing-LMPC-ROS2's BARC oval) has a median
# turn radius of ~1.57m and a minimum of ~0.97m (vs f110_gym_10's ~41.7m
# median) -- essentially continuous tight cornering, not occasional
# hairpins. Measured directly (2026-07-14): gym's dynamic single-track
# model diverges (raw sim omega -> 1e29+ within ~1s) at v~1.0 m/s on this
# track's curvature REGARDLESS of steering magnitude or smoothness (tested
# with both Pure Pursuit and open-loop curvature-derived steering) -- this
# is gym's own low-speed dynamic-branch instability (see the guard below),
# not a controller tuning problem, and this track's tightness keeps the
# vehicle inside that band far more persistently than f110_gym_10 ever
# does. 3.0 m/s sits just above the instability band and just below the
# grip limit at the tightest corner (mu*g*r ~= 3.16 m/s at r=0.97m) --
# verified to complete a full lap cleanly (omega stayed under ~3 rad/s
# throughout) but with a much thinner safety margin than f110_gym_10's 3.5.
SEED_LAP_SPEED = 3.0

ZOOM = 1.0
WINDOW_WIDTH = 1000
WINDOW_HEIGHT = 800

# Pure Pursuit lookahead policy, scaled down from f110_gym_10's (2.0/4.0/8.0)
# to match this track's much smaller scale (~17m lap vs ~164m): a lookahead
# comparable to or larger than the track's own turn radius (~1-1.6m) made
# Pure Pursuit's geometric curvature demand ill-conditioned -- measured
# steering commands over 1.0 rad (>2x delta_max) from a 0.3m lookahead on a
# ~1m-radius corner, which is what first triggered the divergence above,
# not the low speed by itself. 0.6-0.9m (roughly half the median turn
# radius) keeps commanded steering within the physical range naturally.
MIN_LOOKAHEAD = 0.6
MAX_LOOKAHEAD = 0.9
LOOKAHEAD_RATIO = 8.0

# F110 Gym's dynamic single-track model (gym/f110_gym/envs/dynamic_models.py,
# vehicle_dynamics_st) switches to a kinematic model below |v| < 0.5 m/s, and
# diverges if handed nonzero steering while crossing that switch. A soft taper
# from rest was measured insufficient in earlier work on this controller; steer
# must be held at exactly zero through the whole danger zone. Every lap starts
# from rest, so this is unconditionally needed, not situational.
LOW_SPEED_STEER_ZERO_BELOW = 2.0
LOW_SPEED_STEER_RESTORE_AT = 3.0


def obs_to_vehicle_state(obs: dict[str, Any], ego_idx: int) -> VehicleState:
    return VehicleState(
        x=float(obs["poses_x"][ego_idx]),
        y=float(obs["poses_y"][ego_idx]),
        yaw=float(obs["poses_theta"][ego_idx]),
        speed=float(obs["linear_vels_x"][ego_idx]),
    )


class LaunchSteeringGuard:
    """Suppress steering only through the one-time launch-from-rest crossing.

    The gym's divergence risk (see module docstring) is specific to the
    discrete kinematic/dynamic model switch at |v| < 0.5 m/s. SEED_LAP_SPEED
    is constant (the paper's own D^0 recipe), so once Pure Pursuit's P
    controller settles onto it after launch, speed never dips back into the
    danger zone -- measured directly (2026-07-14, this track): speed reaches
    ~3.0 m/s within ~1.5s of launch and stays there for the rest of the lap,
    even through the tightest corner. The guard therefore only needs to
    latch open once, not re-suppress every time speed dips during cornering.
    """

    def __init__(self) -> None:
        self._launched = False

    def apply(self, steer: float, speed: float) -> float:
        if self._launched:
            return steer
        if speed >= LOW_SPEED_STEER_RESTORE_AT:
            self._launched = True
            return steer
        if speed <= LOW_SPEED_STEER_ZERO_BELOW:
            return 0.0
        ramp = (speed - LOW_SPEED_STEER_ZERO_BELOW) / (
            LOW_SPEED_STEER_RESTORE_AT - LOW_SPEED_STEER_ZERO_BELOW
        )
        return steer * ramp


def load_centerline_waypoints(
    csv_path: str, speed: float
) -> tuple[np.ndarray, np.ndarray]:
    """Build PurePursuit waypoints and a project_frenet heading reference.

    Raw centerline CSVs (scripts/generate_centerline.py's format: x_m, y_m,
    w_tr_right_m, w_tr_left_m) have no heading or speed column. Heading is
    the finite-difference tangent between consecutive closed-loop points --
    PurePursuit itself doesn't need heading (its waypoints are [x, y, speed],
    controllers/pure_pursuit.py's PurePursuit.__init__), but project_frenet()
    below still does for epsi, so it's returned separately here rather than
    folded into the waypoints array. Stored offset by -pi/2 to match the same
    on-disk convention the raceline CSVs use -- project_frenet() adds +pi/2
    back, matching the discovered convention from this project's earlier
    Stanley-based version. Speed is constant (SEED_LAP_SPEED), matching the
    paper's own D^0 recipe: a simple low-speed center-line tracking pass,
    not an optimized profile.

    Returns (pursuit_waypoints [x, y, speed], path_psi [stored, -pi/2 offset]).
    """
    xy = np.loadtxt(csv_path, delimiter=",", skiprows=1, dtype=np.float64)[:, :2]
    next_xy = np.roll(xy, -1, axis=0)
    heading = np.arctan2(next_xy[:, 1] - xy[:, 1], next_xy[:, 0] - xy[:, 0])
    psi_stored = heading - np.pi / 2
    vx = np.full(xy.shape[0], speed, dtype=np.float64)
    return np.column_stack([xy, vx]), psi_stored


def wrap_angle(angle: float) -> float:
    return float(np.arctan2(np.sin(angle), np.cos(angle)))


def project_frenet(
    position: np.ndarray,
    path_xy: np.ndarray,
    path_psi: np.ndarray,
    path_s: np.ndarray,
    last_idx: int,
    search_window: int,
) -> tuple[float, float, float, int]:
    """Project a global (x, y) onto the reference path.

    Returns (s, ey, path_heading, nearest_idx). ey is positive to the left of
    the path direction (tangent rotated +90 degrees). search_window should be
    the driving controller's own instance value (derived from this same
    path's actual spacing, controllers/pure_pursuit.py's comment has the
    rationale) -- a fixed index count here would silently become physically
    too narrow on a densely-sampled path and lose track.
    """
    idx = nearest_waypoint_index(
        path_xy, position, last_idx, search_window=search_window
    )
    # The raceline CSV's psi_rad column is offset -90 degrees from the
    # direction of travel -- controllers/stanley.py's own control law adds
    # this same +pi/2 to get its heading reference (psi_path); matching it
    # here so epsi is actually a small heading error, not a constant ~pi/2.
    heading = float(path_psi[idx]) + np.pi / 2
    tangent = np.array([np.cos(heading), np.sin(heading)])
    normal = np.array([-np.sin(heading), np.cos(heading)])
    delta = position - path_xy[idx]
    s = float(path_s[idx] + np.dot(delta, tangent))
    ey = float(np.dot(delta, normal))
    return s, ey, heading, idx


def capture_lap_sample(
    obs: dict[str, Any],
    f110_env: Any,
    ego_idx: int,
    path_xy: np.ndarray,
    path_psi: np.ndarray,
    path_s: np.ndarray,
    last_idx: int,
    search_window: int,
) -> tuple[LapSample, int]:
    """Capture one native LMPC state and the plant values at the same instant."""
    state = obs_to_vehicle_state(obs, ego_idx)
    raw_state = f110_env.sim.agents[ego_idx].state
    raw_speed = float(raw_state[3])
    beta = float(raw_state[6])
    s, ey, path_heading, nearest_idx = project_frenet(
        np.array([state.x, state.y]),
        path_xy,
        path_psi,
        path_s,
        last_idx,
        search_window,
    )
    x = np.array(
        [
            raw_speed * np.cos(beta),
            raw_speed * np.sin(beta),
            float(raw_state[5]),
            wrap_angle(state.yaw - path_heading),
            s,
            ey,
        ],
        dtype=np.float64,
    )
    return LapSample(x, float(raw_state[2]), raw_speed), nearest_idx


def main(visualize: bool = False) -> None:
    env = gym.make("f110-v0", map=MAP, num_agents=1, timestep=SIM_TIMESTEP)
    f110_env: Any = env.unwrapped
    ego_idx = 0
    dt = float(f110_env.timestep)

    pursuit_waypoints, path_psi = load_centerline_waypoints(
        WAYPOINTS_CSV, SEED_LAP_SPEED
    )
    wheelbase = float(f110_env.params["lf"] + f110_env.params["lr"])
    lookahead_policy = DynamicLookaheadDistance(
        MIN_LOOKAHEAD, MAX_LOOKAHEAD, LOOKAHEAD_RATIO
    )
    controller = PurePursuit(
        pursuit_waypoints, lookahead=lookahead_policy, wheelbase=wheelbase
    )
    path_xy = controller.waypoints[:, :2]
    path_s = cumulative_arc_lengths(path_xy)

    initial_pose = initial_pose_from_waypoints(path_xy)
    obs, _info = env.reset(options={"poses": initial_pose})

    viewer = None
    if visualize:
        # Imported lazily so headless runs (the default) don't need pyglet.
        from f110_gym.viewer import F110Viewer
        from utils.waypoint_view import WaypointOverlay

        waypoint_overlay = WaypointOverlay(path_xy)
        viewer = F110Viewer.from_env(
            f110_env,
            width=WINDOW_WIDTH,
            height=WINDOW_HEIGHT,
            target_fps=60.0,
            initial_zoom=ZOOM,
            callbacks=[waypoint_overlay],
        )
        viewer.update(obs)
        viewer.render()

    last_idx = -1
    t = 0.0
    initial_sample, last_idx = capture_lap_sample(
        obs,
        f110_env,
        ego_idx,
        path_xy,
        path_psi,
        path_s,
        last_idx,
        controller.search_window,
    )
    samples = [initial_sample]
    steering_guard = LaunchSteeringGuard()

    while True:
        state = obs_to_vehicle_state(obs, ego_idx)
        controller.update(state)
        cmd = controller.control()
        steer = steering_guard.apply(cmd.steering, state.speed)

        action = np.array([[steer, cmd.velocity]], dtype=np.float64)
        obs, _reward, terminated, truncated, _info = env.step(action)
        t += dt

        post_step_sample, last_idx = capture_lap_sample(
            obs,
            f110_env,
            ego_idx,
            path_xy,
            path_psi,
            path_s,
            last_idx,
            controller.search_window,
        )
        samples.append(post_step_sample)

        if viewer is not None:
            viewer.update(obs)
            viewer.render()

        if float(obs["lap_counts"][ego_idx]) >= 1.0 or terminated or truncated:
            break

    if viewer is not None:
        while not viewer.closed:
            viewer.render()

    env.close()

    lap_completed = (
        float(obs["lap_counts"][ego_idx]) >= 1.0
        and not bool(f110_env.collisions[ego_idx])
        and not truncated
    )
    if not lap_completed:
        raise RuntimeError(
            "Run ended before completing a lap (terminated/truncated early); "
            "no seed lap written."
        )

    x_lap, u_lap, J_lap = build_lap_arrays(samples, dt)
    write_seed_lap_csv(x_lap, u_lap, J_lap, dt, OUTPUT_CSV)
    print(
        f"Wrote {x_lap.shape[1]} states ({u_lap.shape[1]} transitions) to {OUTPUT_CSV}"
    )


def write_seed_lap_csv(
    x_lap: np.ndarray,
    u_lap: np.ndarray,
    J_lap: np.ndarray,
    dt: float,
    output_path: str,
) -> None:
    """Write finalized (x_k, u_k, J_k) arrays to the seed-lap CSV.

    One lap with T transitions has T+1 states including the first post-step
    finish-crossing state, T realized inputs, and J_k = T-k.
    """
    total_steps = u_lap.shape[1]
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "vx",
                "vy",
                "omega",
                "epsi",
                "s",
                "ey",
                "t",
                "a",
                "delta",
                "J",
            ]
        )
        for k in range(x_lap.shape[1]):
            is_last = k == total_steps
            if is_last:
                a_k = ""
                delta_k = ""
            else:
                a_k = u_lap[0, k]
                delta_k = u_lap[1, k]
            writer.writerow(
                [
                    *x_lap[:, k],
                    k * dt,
                    a_k,
                    delta_k,
                    J_lap[k],
                ]
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Drive and record an LMPC D^0 seed lap"
    )
    parser.add_argument(
        "--visualize", action="store_true", help="Show the drive in the pyglet viewer"
    )
    args = parser.parse_args()
    main(visualize=args.visualize)
