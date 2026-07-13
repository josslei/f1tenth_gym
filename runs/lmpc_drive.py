"""Drive the f110_gym_10 map with the native LMPC controller.

Mirrors runs/waypoint_drive.py's shape (env/viewer setup, obs->state->cmd
loop) but for controllers.lmpc.LMPCController instead of Pure
Pursuit/Stanley, and adds two LMPC-specific overlays: the car's own driven
path (DrivenLineOverlay) and the solver's receding-horizon prediction
(RecedingHorizonOverlay) alongside the reference centerline.
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
HORIZON_STEPS = 15
# gym/f110_gym/envs/f110_env.py's F110Env defaults timestep to 0.01 unless a
# caller overrides it -- explicit here rather than editing that vendored
# default (CLAUDE.md: treat gym/ as a black box). 0.025 is the value
# controllers/lmpc/DESIGN.md's N=75 horizon was originally pinned alongside
# (SS8/config discussion) -- must match the seed lap's own dt too
# (scripts/lmpc_collect_seed_lap.py's SIM_TIMESTEP), or D^0's recorded
# states/costs-to-go don't correspond to this controller's own step size.
SIM_TIMESTEP = 0.025

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
    )
    controller.attach_raw_velocity_state(lambda: raw_velocity_state(f110_env, ego_idx))

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

    obs, _info = env.reset(options={"poses": initial_pose})
    controller.reset()

    viewer.update(obs)
    viewer.render()

    t = 0.0
    while True:
        state = obs_to_vehicle_state(obs)
        controller.update(state, t)
        cmd = controller.control()
        action = np.array([[cmd.steering, cmd.velocity]], dtype=np.float64)

        obs, _reward, terminated, truncated, _info = env.step(action)
        t += dt
        viewer.update(obs)
        viewer.render()

        if terminated or truncated:
            break

    while not viewer.closed:
        viewer.render()

    env.close()


if __name__ == "__main__":
    main()
