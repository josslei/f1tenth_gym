from __future__ import annotations

# pyright: reportAttributeAccessIssue=none, reportArgumentType=none, reportCallIssue=none

import argparse
import faulthandler
from pathlib import Path

import lightning as pl
import numpy as np
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger

import f110_gym  # noqa: F401 - registers f110-v0
from f110_gym.envs.f110_env import DEFAULT_PARAMS, _resolve_map_path
from models.muzero import (
    DiscreteActionConfig,
    DiscreteActionSpace,
    F110MuZeroNet,
    LightningMuZero,
    MuZeroReplayBuffer,
    SelfPlayCallback,
    TorchScriptExportCallback,
    load_muzero_config,
)
from planner.f110_self_play.backend import (
    ActionLattice,
    F110Params,
    MuZeroSearchAdapter,
    ObservationConfig,
    SelfPlayEngine,
)
from utils.f110_env import (
    F1TenthObservationConfig,
    observation_dim,
    with_resampled_waypoints,
)
from utils.track_map import load_track_map
from utils.waypoint_view import initial_pose_from_waypoints
from utils.waypoint_utils import cumulative_arc_lengths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/muzero/default.yaml")
    parser.add_argument(
        "--resume",
        default=None,
        help="Lightning checkpoint to resume training from.",
    )
    parser.add_argument(
        "--tensorboard-version",
        default="run",
        help="TensorBoard version subdirectory under output/tensorboard.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Torch device for MuZero model and native TorchScript search.",
    )
    return parser.parse_args()


def resolve_device(device_name: str):
    if device_name != "auto":
        return torch.device(device_name)  # type: ignore[attr-defined]
    return torch.device(  # type: ignore[attr-defined]
        "cuda" if torch.cuda.is_available() else "cpu"
    )


def device_name(device) -> str:
    return str(device).split(":", maxsplit=1)[0]


def native_observation_config(config: F1TenthObservationConfig) -> ObservationConfig:
    native = ObservationConfig()
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


def native_dynamics_params(env_config: dict) -> F110Params:
    params = dict(DEFAULT_PARAMS)
    params.update(env_config.get("params", {}))
    native = F110Params()
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


def initial_states_from_pose(
    reset_pose: np.ndarray, batch_size: int, initial_velocity: float
) -> np.ndarray:
    pose = np.asarray(reset_pose, dtype=np.float64).reshape(-1, 3)[0]
    states = np.zeros((batch_size, 7), dtype=np.float64)
    states[:, 0] = pose[0]
    states[:, 1] = pose[1]
    states[:, 4] = pose[2]
    states[:, 3] = initial_velocity
    return states


def action_bins_from_config(
    action_section: dict, env_config: dict
) -> tuple[np.ndarray, np.ndarray]:
    params = dict(DEFAULT_PARAMS)
    params.update(env_config.get("params", {}))
    steering_config = action_section["steering_bins"]
    velocity_config = action_section["velocity_bins"]
    steering_bins = (
        np.linspace(
            params["s_min"], params["s_max"], int(steering_config), dtype=np.float32
        )
        if np.isscalar(steering_config)
        else np.asarray(steering_config, dtype=np.float32)
    )
    velocity_bins = (
        np.linspace(
            float(action_section["velocity_min"]),
            float(action_section["velocity_max"]),
            int(velocity_config),
            dtype=np.float32,
        )
        if np.isscalar(velocity_config)
        else np.asarray(velocity_config, dtype=np.float32)
    )
    return steering_bins, velocity_bins


def main() -> None:
    faulthandler.enable(all_threads=True)

    args = parse_args()
    device = resolve_device(args.device)
    print(f"MuZero device: {device_name(device)}", flush=True)
    config = load_muzero_config(args.config)
    env_section = config["env"]
    observation_section = config["observation"]
    action_section = config["action"]
    reward_section = config["reward"]
    model_section = config["model"]
    search_section = config["search"]
    self_play_section = config["self_play"]
    replay_section = config["replay"]
    training_section = config["training"]
    runtime_section = config["runtime"]

    torch.manual_seed(runtime_section["seed"])
    np.random.seed(runtime_section["seed"])

    output_dir = Path(config["output"]["dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    env_config = dict(env_section)
    centerline_csv = env_config.pop("centerline_csv")
    env_config.pop("initial_pose", None)
    centerline = np.loadtxt(centerline_csv, delimiter=",", skiprows=1)[:, :2]
    reset_pose = initial_pose_from_waypoints(centerline)

    observation_config = F1TenthObservationConfig(**observation_section)
    if observation_config.include_waypoints:
        observation_config = with_resampled_waypoints(observation_config, centerline)
    obs_dim = observation_dim(observation_config)

    steering_bins, velocity_bins = action_bins_from_config(action_section, env_config)
    discrete_action_config = DiscreteActionConfig(
        steering_bins=steering_bins,
        velocity_bins=velocity_bins,
    )
    action_space = DiscreteActionSpace(discrete_action_config)
    action_lattice = ActionLattice(
        discrete_action_config.steering_bins,
        discrete_action_config.velocity_bins,
    )

    model = F110MuZeroNet(
        obs_dim=obs_dim,
        action_count=action_space.action_count,
        hidden_size=model_section["hidden_size"],
        trunk_size=model_section.get("trunk_size", 256),
    ).to(device)
    replay_buffer = MuZeroReplayBuffer(
        max_size=replay_section["capacity"],
        unroll_steps=replay_section["unroll_steps"],
        td_steps=replay_section["td_steps"],
        discount=self_play_section["discount"],
    )
    module = LightningMuZero(
        model=model,
        replay_buffer=replay_buffer,
        training_config=training_section,
        replay_config=replay_section,
    )

    map_path = str(_resolve_map_path(env_config["map"]))
    track_map, car_length, car_width = load_track_map(
        map_path,
        env_config.get("map_ext", ".png"),
        car_length=DEFAULT_PARAMS["length"],
        car_width=DEFAULT_PARAMS["width"],
    )
    native_obs_config = native_observation_config(observation_config)
    waypoint_path = (
        observation_config._waypoints
        if observation_config.include_waypoints
        else centerline
    )
    waypoints = np.asarray(waypoint_path, dtype=np.float64)
    cum_arc_lengths = cumulative_arc_lengths(waypoints)
    dynamics_params = native_dynamics_params(env_config)
    initial_states = initial_states_from_pose(
        reset_pose,
        search_section["batch_size"],
        float(self_play_section["initial_velocity"]),
    )

    model_path = output_dir / "current_model.pt"
    scripted = torch.jit.script(model.eval()).to(device)
    scripted.save(str(model_path))
    model.train()
    print(f"MuZero scripted model: {model_path}", flush=True)

    search = MuZeroSearchAdapter(
        str(model_path),
        search_section["num_iters"],
        search_section["temperature"],
        search_section["c_puct"],
        dirichlet_alpha=search_section.get("dirichlet_alpha", 0.3),
        dirichlet_epsilon=search_section.get("dirichlet_epsilon", 0.25),
        batch_size=search_section["batch_size"],
        action_count=action_lattice.action_count,
        hidden_size=model.hidden_size,
        max_nodes=0,
        device=device_name(device),
        print_metrics=search_section.get("print_metrics", False),
    )
    print("MuZero native search constructed", flush=True)
    engine = SelfPlayEngine(
        search,
        track_map,
        native_obs_config,
        action_lattice,
        self_play_section["discount"],
        True,
        self_play_section.get("print_metrics", False),
        waypoints[:, 0],
        waypoints[:, 1],
        cum_arc_lengths,
        dynamics_params,
        car_length,
        car_width,
        reward_section["q_s_progress"],
        reward_section["q_s_alpha"],
        reward_section["q_s_smooth"],
        reward_section["terminal_penalty"],
        reward_section["alpha_th"],
        reward_section["slip_terminal_penalty"],
        reward_section["q_offtrack_grad"],
    )

    # Seed replay before Lightning asks the module for a dataloader.
    result = engine.generate(
        self_play_section["rollout_steps"],
        search_section["batch_size"],
        initial_states,
    )
    for trajectory in result.trajectories:
        replay_buffer.push(trajectory)

    logger = TensorBoardLogger(
        save_dir=output_dir,
        name="tensorboard",
        version=args.tensorboard_version,
    )
    callbacks = [
        SelfPlayCallback(
            track_map=track_map,
            observation_config=native_obs_config,
            action_lattice=action_lattice,
            waypoints=waypoints,
            cum_arc_lengths=cum_arc_lengths,
            dynamics_params=dynamics_params,
            initial_states=initial_states,
            car_length=car_length,
            car_width=car_width,
            reward_config=reward_section,
            rollout_steps=self_play_section["rollout_steps"],
            num_iters=search_section["num_iters"],
            c_puct=search_section["c_puct"],
            dirichlet_alpha=search_section.get("dirichlet_alpha", 0.3),
            dirichlet_epsilon=search_section.get("dirichlet_epsilon", 0.25),
            temperature=search_section["temperature"],
            search_print_metrics=search_section.get("print_metrics", False),
            self_play_print_metrics=self_play_section.get("print_metrics", False),
            discount=self_play_section["discount"],
            device=device,
            model_path=str(output_dir / "current_model.pt"),
        ),
        TorchScriptExportCallback(str(output_dir / "checkpoints")),
        ModelCheckpoint(
            dirpath=output_dir / "checkpoints",
            filename="muzero-{epoch:04d}",
            monitor="train/loss",
            mode="min",
            save_top_k=3,
            every_n_epochs=runtime_section.get("checkpoint_every_n_epochs", 10),
        ),
    ]

    trainer = pl.Trainer(
        accelerator=device_name(device),
        max_epochs=training_section["epochs"],
        enable_progress_bar=runtime_section["progress_bar"],
        logger=logger,
        callbacks=callbacks,
        reload_dataloaders_every_n_epochs=1,
    )
    trainer.fit(module, ckpt_path=args.resume)


if __name__ == "__main__":
    main()
