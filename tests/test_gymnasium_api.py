import f110_gym  # noqa: F401  # registers the environment
import gymnasium as gym
import numpy as np
from typing import cast

from f110_gym.envs.f110_env import F110Env


def _center_poses(env):
    scan_sim = env.unwrapped.sim.agents[0].scan_simulator
    origin_x, origin_y, _ = scan_sim.origin
    center_x = origin_x + scan_sim.map_width * scan_sim.map_resolution / 2.0
    center_y = origin_y + scan_sim.map_height * scan_sim.map_resolution / 2.0
    return np.array(
        [
            [center_x, center_y, 0.0],
            [center_x + 1.0, center_y, 0.0],
        ],
        dtype=np.float64,
    )


def _check_obs_against_space(env, obs):
    space = env.observation_space
    assert space.contains(obs), "observation is not within the declared space"


def test_gymnasium_registration_reset_and_step():
    env = gym.make("f110-v0")
    try:
        f110_env = cast(F110Env, env.unwrapped)
        poses = _center_poses(f110_env)
        obs, info = f110_env.reset(options={"poses": poses})

        assert isinstance(info, dict)
        _check_obs_against_space(f110_env, obs)
        assert obs["scans"].shape == (f110_env.num_agents, 1080)
        assert obs["poses_x"].shape == (f110_env.num_agents,)

        action = np.zeros((f110_env.num_agents, 2), dtype=np.float64)
        obs, reward, terminated, truncated, step_info = f110_env.step(action)

        _check_obs_against_space(f110_env, obs)
        assert obs["scans"].shape == (f110_env.num_agents, 1080)
        assert reward == f110_env.timestep
        assert isinstance(terminated, (bool, np.bool_))
        assert truncated is False
        assert "checkpoint_done" in step_info
    finally:
        env.close()


def test_legacy_reset_dict_and_update_params_aliases():
    env = gym.make("f110-v0")
    try:
        f110_env = cast(F110Env, env.unwrapped)
        scan_sim = f110_env.sim.agents[0].scan_simulator
        origin_x, origin_y, _ = scan_sim.origin
        center_x = origin_x + scan_sim.map_width * scan_sim.map_resolution / 2.0
        center_y = origin_y + scan_sim.map_height * scan_sim.map_resolution / 2.0
        poses = {
            "x": [center_x, center_x + 1.0],
            "y": [center_y, center_y],
            "theta": [0.0, 0.0],
        }

        obs, info = f110_env.reset(poses)
        assert isinstance(info, dict)
        _check_obs_against_space(f110_env, obs)
        assert obs["poses_x"].shape == (f110_env.num_agents,)

        f110_env.update_params(
            1.0,
            -1,
            0.07,
            0.17,
            4.0,
            5.0,
            0.05,
            3.7,
            "/tmp",
            double_finish=True,
        )
    finally:
        env.close()


def test_make_viewer_uses_environment_map_without_opening_window():
    env = gym.make("f110-v0")
    try:
        f110_env = cast(F110Env, env.unwrapped)
        viewer = f110_env.make_viewer(width=320, height=240, target_fps=None)

        assert viewer.config.map_path == f110_env.map_stem
        assert viewer.config.map_ext == f110_env.map_ext
        assert viewer.config.width == 320
        assert viewer.config.height == 240
        assert viewer.config.target_fps is None
    finally:
        env.close()
