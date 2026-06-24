# pyright: reportAttributeAccessIssue=none, reportArgumentType=none, reportCallIssue=none
"""Drive the F1TENTH car with a trained MuZero policy via the native kernel."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
import yaml
from PIL import Image

import f110_gym  # noqa: F401 - registers f110-v0
from f110_gym.rollout_kernel.natives import _f110_rollout_kernel as rk
from f110_gym.viewer import F110Viewer, ViewerConfig
from planner.f110_self_play.backend import ActionLattice, MuZeroSearchAdapter
from utils.f110_env import F1TenthObservationConfig, with_resampled_waypoints
from utils.waypoint_view import WaypointOverlay, initial_pose_from_waypoints
from utils.f110_env import obs_tensor as build_obs_tensor

from models.muzero import load_muzero_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/muzero/default.yaml")
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="TorchScript MuZero checkpoint to load.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Torch device for the scripted MuZero model and native search.",
    )
    return parser.parse_args()


def resolve_device(device_name: str) -> Any:
    if device_name != "auto":
        return torch.device(device_name)  # type: ignore[attr-defined]
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")  # type: ignore[attr-defined]


def device_name(device: Any) -> str:
    return str(device).split(":", maxsplit=1)[0]


def action_bins_from_config(
    action_section: dict, env_config: dict
) -> tuple[np.ndarray, np.ndarray]:
    params = dict(env_config.get("params", {}))
    params.setdefault("s_min", -0.4189)
    params.setdefault("s_max", 0.4189)

    steering_config: Any = action_section["steering_bins"]
    velocity_config: Any = action_section["velocity_bins"]

    steering_bins = (
        np.linspace(
            float(params["s_min"]),
            float(params["s_max"]),
            int(cast(Any, steering_config)),
            dtype=np.float32,
        )
        if np.isscalar(steering_config)
        else np.asarray(steering_config, dtype=np.float32)
    )
    velocity_bins = (
        np.linspace(
            float(action_section.get("velocity_min", 0.0)),
            float(action_section.get("velocity_max", 8.0)),
            int(cast(Any, velocity_config)),
            dtype=np.float32,
        )
        if np.isscalar(velocity_config)
        else np.asarray(velocity_config, dtype=np.float32)
    )
    return steering_bins, velocity_bins


def native_observation_config(config: F1TenthObservationConfig) -> rk.ObservationConfig:
    native = rk.ObservationConfig()
    native.scan_size = config.scan_size
    native.scan_max_m = config.scan_max_m
    native.include_ego_state = config.include_ego_state
    native.speed_scale = config.speed_scale
    native.yaw_rate_scale = config.yaw_rate_scale
    native.steer_scale = config.steer_scale
    native.include_waypoints = config.include_waypoints
    native.lookahead_distances = list(config.lookahead_distances)
    native.waypoint_scale = config.waypoint_scale
    native.waypoint_resample_spacing = config.waypoint_resample_spacing
    return native


def native_dynamics_params(env_config: dict) -> rk.F110Params:
    defaults = rk.DEFAULT_PARAMS
    params = {
        "mu": defaults.mu,
        "C_Sf": defaults.c_sf,
        "C_Sr": defaults.c_sr,
        "lf": defaults.lf,
        "lr": defaults.lr,
        "h": defaults.h,
        "m": defaults.m,
        "I": defaults.inertia,
        "s_min": defaults.s_min,
        "s_max": defaults.s_max,
        "sv_min": defaults.sv_min,
        "sv_max": defaults.sv_max,
        "v_switch": defaults.v_switch,
        "a_max": defaults.a_max,
        "v_min": defaults.v_min,
        "v_max": defaults.v_max,
    }
    params.update(env_config.get("params", {}))
    native = rk.F110Params()
    native.mu = params["mu"]
    native.c_sf = params["C_Sf"]
    native.c_sr = params["C_Sr"]
    native.lf = params["lf"]
    native.lr = params["lr"]
    native.h = params["h"]
    native.m = params["m"]
    native.inertia = params["I"]
    native.s_min = params["s_min"]
    native.s_max = params["s_max"]
    native.sv_min = params["sv_min"]
    native.sv_max = params["sv_max"]
    native.v_switch = params["v_switch"]
    native.a_max = params["a_max"]
    native.v_min = params["v_min"]
    native.v_max = params["v_max"]
    native.timestep = env_config.get("timestep", 0.01)
    return native


def load_native_track_map(
    map_yaml: Path,
    map_ext: str,
    num_beams: int = 1080,
    fov: float = 4.7,
    max_range: float = 30.0,
    eps: float = 0.0001,
    theta_dis: int = 2000,
    car_length: float = 0.58,
    car_width: float = 0.31,
) -> tuple[rk.TrackMap, float, float]:
    with map_yaml.open("r", encoding="utf-8") as f:
        meta = yaml.safe_load(f)

    map_img = map_yaml.with_suffix(map_ext)
    resolution = float(meta["resolution"])
    origin = [float(v) for v in meta["origin"]]

    img = Image.open(map_img).transpose(Image.Transpose.FLIP_TOP_BOTTOM)
    bitmap = np.asarray(img, dtype=np.float32)
    bitmap = np.where(bitmap <= 128, 0.0, 255.0)

    import scipy.ndimage

    dt = np.asarray(scipy.ndimage.distance_transform_edt(bitmap), dtype=np.float32)
    dt *= resolution

    track = rk.TrackMap()
    track.height = int(dt.shape[0])
    track.width = int(dt.shape[1])
    track.resolution = float(resolution)
    track.orig_x = float(origin[0])
    track.orig_y = float(origin[1])
    track.orig_c = float(math.cos(origin[2]))
    track.orig_s = float(math.sin(origin[2]))
    track.dt = dt.ravel().astype(np.float32).tolist()
    track.theta_dis = int(theta_dis)
    track.num_beams = int(num_beams)
    track.fov = float(fov)
    track.max_range = float(max_range)
    track.eps = float(eps)
    track.compute_scan_tables()

    dist_sides = car_width / 2.0
    dist_fr = car_length / 2.0
    scan_ang_incr = fov / float(num_beams - 1)
    side_distances = np.empty((num_beams,), dtype=np.float32)
    for i in range(num_beams):
        angle = -fov / 2.0 + float(i) * scan_ang_incr
        if angle > 0.0:
            if angle < math.pi / 2.0:
                to_side = dist_sides / math.sin(angle)
                to_fr = dist_fr / math.cos(angle)
                side_distances[i] = float(min(to_side, to_fr))
            else:
                to_side = dist_sides / math.cos(angle - math.pi / 2.0)
                to_fr = dist_fr / math.sin(angle - math.pi / 2.0)
                side_distances[i] = float(min(to_side, to_fr))
        else:
            if angle > -math.pi / 2.0:
                to_side = dist_sides / math.sin(-angle)
                to_fr = dist_fr / math.cos(-angle)
                side_distances[i] = float(min(to_side, to_fr))
            else:
                to_side = dist_sides / math.cos(-angle - math.pi / 2.0)
                to_fr = dist_fr / math.sin(-angle - math.pi / 2.0)
                side_distances[i] = float(min(to_side, to_fr))

    track.side_distances = side_distances.tolist()
    track.ttc_thresh = 0.005
    return track, car_length, car_width


def initial_state_from_waypoints(
    waypoints_xy: np.ndarray, initial_velocity: float
) -> rk.F110State:
    pose = initial_pose_from_waypoints(waypoints_xy)[0]
    state = rk.F110State()
    state.x = float(pose[0])
    state.y = float(pose[1])
    state.yaw_angle = float(pose[2])
    state.steer_angle = 0.0
    state.velocity = float(initial_velocity)
    state.yaw_rate = 0.0
    state.slip_angle = 0.0
    state.steer_buffer_0 = 0.0
    state.steer_buffer_1 = 0.0
    state.steer_buffer_len = 0
    state.in_collision = False
    return state


def native_obs_dict(
    state: rk.F110State,
    prev_action: np.ndarray,
    collision: bool,
    step_idx: int,
    timestep: float,
) -> dict[str, Any]:
    lap_time = float(step_idx) * float(timestep)
    return {
        "ego_idx": 0,
        "scans": np.zeros((1, 1080), dtype=np.float64),
        "poses_x": np.array([state.x], dtype=np.float64),
        "poses_y": np.array([state.y], dtype=np.float64),
        "poses_theta": np.array([state.yaw_angle], dtype=np.float64),
        "linear_vels_x": np.array(
            [state.velocity * math.cos(state.slip_angle)], dtype=np.float64
        ),
        "linear_vels_y": np.array(
            [state.velocity * math.sin(state.slip_angle)], dtype=np.float64
        ),
        "ang_vels_z": np.array([state.yaw_rate], dtype=np.float64),
        "collisions": np.array([float(collision)], dtype=np.float64),
        "lap_times": np.array([lap_time], dtype=np.float64),
        "lap_counts": np.array([0.0], dtype=np.float64),
        "steer_angle": np.array([state.steer_angle], dtype=np.float64),
        "prev_action": prev_action,
    }


def checkpoint_done(
    state: rk.F110State,
    start_pose: np.ndarray,
    start_rot: np.ndarray,
    toggle_count: int,
    near_start: bool,
    laps_to_complete: int,
) -> tuple[bool, bool, int, bool]:
    delta = start_rot @ np.array([state.x - start_pose[0], state.y - start_pose[1]])
    temp_y = float(delta[1])
    left_t = 2.0
    right_t = 2.0
    if temp_y > left_t:
        temp_y -= left_t
    elif temp_y < -right_t:
        temp_y = -right_t - temp_y
    else:
        temp_y = 0.0

    dist2 = float(delta[0]) * float(delta[0]) + temp_y * temp_y
    close = dist2 <= 0.1
    if close and not near_start:
        near_start = True
        toggle_count += 1
    elif not close and near_start:
        near_start = False
        toggle_count += 1

    checkpoint = toggle_count >= 2 * laps_to_complete
    done = checkpoint
    return done, checkpoint, toggle_count, near_start


def viewer_obs(
    state: rk.F110State,
    step_idx: int,
    timestep: float,
    lap_times: np.ndarray,
    lap_counts: np.ndarray,
) -> dict[str, Any]:
    return {
        "ego_idx": 0,
        "poses_x": np.array([state.x], dtype=np.float64),
        "poses_y": np.array([state.y], dtype=np.float64),
        "poses_theta": np.array([state.yaw_angle], dtype=np.float64),
        "lap_times": lap_times,
        "lap_counts": lap_counts,
    }


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)

    config = load_muzero_config(args.config)
    env_section = dict(config["env"])
    observation_section = config["observation"]
    action_section = config["action"]
    model_section = config["model"]
    search_section = config["search"]

    env_config = dict(env_section)
    centerline_csv = Path(env_config["centerline_csv"])
    centerline = np.loadtxt(centerline_csv, delimiter=",", skiprows=1)[:, :2]

    observation_config = F1TenthObservationConfig(**observation_section)
    if observation_config.include_waypoints:
        observation_config = with_resampled_waypoints(observation_config, centerline)

    steering_bins, velocity_bins = action_bins_from_config(action_section, env_config)
    action_lattice = ActionLattice(steering_bins, velocity_bins)

    native_obs_config = native_observation_config(observation_config)
    dynamics_params = native_dynamics_params(env_config)
    track_map, car_length, car_width = load_native_track_map(
        Path(env_config["map"]),
        env_config.get("map_ext", ".png"),
    )

    checkpoint = (
        Path(args.checkpoint)
        if args.checkpoint
        else Path(config["output"]["dir"]) / "current_model.pt"
    )

    print(f"MuZero device: {device_name(device)}", flush=True)
    print(f"MuZero checkpoint: {checkpoint}", flush=True)
    print(f"MuZero obs dim: {rk.observation_dim(native_obs_config)}", flush=True)

    search: Any = MuZeroSearchAdapter(
        str(checkpoint),
        search_section["num_iters"],
        0.0,
        search_section["c_puct"],
        batch_size=1,
        action_count=action_lattice.action_count,
        hidden_size=model_section["hidden_size"],
        device=device_name(device),
        print_metrics=search_section.get("print_metrics", False),
        dirichlet_alpha=0.0,
        dirichlet_epsilon=0.0,
    )

    viewer = F110Viewer(
        ViewerConfig(
            map_path=env_config["map"],
            map_ext=env_config.get("map_ext", ".png"),
            width=1000,
            height=800,
            target_fps=60.0,
            initial_zoom=1.0,
        ),
        callbacks=[WaypointOverlay(centerline)],
    )

    state = initial_state_from_waypoints(centerline, 0.0)
    prev_action = np.zeros(2, dtype=np.float32)
    step_idx = 0
    start_pose = initial_pose_from_waypoints(centerline)[0]
    start_rot = np.array(
        [
            [math.cos(-start_pose[2]), -math.sin(-start_pose[2])],
            [math.sin(-start_pose[2]), math.cos(-start_pose[2])],
        ]
    )
    toggle_count = 0
    near_start = True
    laps_to_complete = int(env_config.get("laps_to_complete", 1))
    lap_counts = np.array([0.0], dtype=np.float64)
    lap_times = np.array([0.0], dtype=np.float64)

    viewer.update(
        viewer_obs(state, step_idx, dynamics_params.timestep, lap_times, lap_counts)
    )
    viewer.render()

    while True:
        scan = rk.get_scan(state.x, state.y, state.yaw_angle, track_map)
        collision = (
            bool(rk.check_ttc(scan, state.velocity, track_map))
            or track_map.distance_at(state.x, state.y)
            <= 0.5 * math.hypot(car_length, car_width)
            or state.in_collision
        )
        obs_dict = native_obs_dict(
            state, prev_action, collision, step_idx, dynamics_params.timestep
        )
        obs_dict["scans"] = np.array([scan], dtype=np.float64)
        obs_dict["lap_counts"] = lap_counts
        obs_dict["lap_times"] = lap_times
        obs = build_obs_tensor(obs_dict, observation_config)

        with torch.no_grad():
            search_result = search.search_batch(obs)
        action_index = int(
            search_result.action_probs.detach().cpu().numpy()[0].argmax()
        )
        action = np.asarray(
            action_lattice.normalized_action(action_index), dtype=np.float64
        )

        step_result = rk.step(
            state,
            rk.F110Action(float(action[0]), float(action[1])),
            dynamics_params,
            rk.Integrator.RK4,
        )
        state = step_result.state

        scan = rk.get_scan(state.x, state.y, state.yaw_angle, track_map)
        collision = bool(rk.check_ttc(scan, state.velocity, track_map)) or (
            track_map.distance_at(state.x, state.y)
            <= 0.5 * math.hypot(car_length, car_width)
        )
        state.in_collision = collision
        if collision:
            state.velocity = 0.0
            state.yaw_angle = 0.0
            state.yaw_rate = 0.0
            state.slip_angle = 0.0

        step_idx += 1
        current_time = float(step_idx) * float(dynamics_params.timestep)
        terminated, _checkpoint_done, toggle_count, near_start = checkpoint_done(
            state, start_pose, start_rot, toggle_count, near_start, laps_to_complete
        )
        terminated = collision or terminated
        if toggle_count < 2 * laps_to_complete:
            lap_times[0] = current_time
        lap_counts[0] = float(toggle_count // 2)

        prev_action = action.astype(np.float32)

        viewer.update(
            viewer_obs(state, step_idx, dynamics_params.timestep, lap_times, lap_counts)
        )
        viewer.render()

        if terminated:
            print(
                f"terminated at step {step_idx}: collision={collision}, x={state.x:.3f}, y={state.y:.3f}",
                flush=True,
            )
            break

    while not viewer.closed:
        viewer.render()


if __name__ == "__main__":
    main()
