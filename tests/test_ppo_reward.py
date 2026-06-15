from pathlib import Path

import numpy as np
import pytest

from models.ppo import PPOConfig
from runs.train_ppo_controller import F1TenthPPOReward


def _obs(*, vx: float = 8.0, collision: bool = False, theta: float = 0.0):
    return {
        "ego_idx": 0,
        "linear_vels_x": np.array([vx], dtype=np.float64),
        "linear_vels_y": np.array([0.0], dtype=np.float64),
        "collisions": np.array([collision], dtype=bool),
        "poses_theta": np.array([theta], dtype=np.float64),
        "poses_x": np.array([0.0], dtype=np.float64),
        "poses_y": np.array([0.0], dtype=np.float64),
    }


def _reward(tmp_path: Path, **kwargs) -> F1TenthPPOReward:
    waypoint_path = tmp_path / "waypoints.csv"
    waypoint_path.write_text("0.0,0.0\n1.0,0.0\n", encoding="utf-8")
    return F1TenthPPOReward(
        waypoints_path=waypoint_path,
        delimiter=",",
        usecols=(0, 1),
        **kwargs,
    )


def test_reward_uses_scaled_speed_component(tmp_path: Path):
    reward = _reward(
        tmp_path,
        speed_reward_weight=0.1,
        dense_progress_weight=0.0,
        waypoint_bonus_weight=0.0,
    )

    assert np.isclose(reward(_obs(vx=8.0), terminated=False), 0.8)


def test_dense_progress_signal_near_waypoint(tmp_path: Path):
    reward = _reward(
        tmp_path,
        speed_reward_weight=0.1,
        dense_progress_weight=0.5,
        waypoint_bonus_weight=1.0,
    )

    step_reward = reward(_obs(vx=8.0), terminated=False)

    assert step_reward > 2.0


def test_collision_returns_growth_penalty(tmp_path: Path):
    reward = _reward(
        tmp_path,
        speed_reward_weight=0.1,
        collision_penalty=2.0,
        collision_growth=0.005,
    )

    assert reward(_obs(vx=8.0, collision=True), terminated=True) == -2.0

    # simulate 500 waypoints of progress before crash
    reward.idx = 500
    assert reward(_obs(vx=8.0, collision=True), terminated=True) == pytest.approx(-4.5)


def test_default_ppo_config_uses_convergence_safe_settings():
    config = PPOConfig.from_yaml(Path("configs/ppo/default.yaml"))

    assert config.action["velocity_min"] > 0.0
    assert config.reward["speed_reward_weight"] == 0.1
    assert config.reward["dense_progress_weight"] == 0.5
    assert config.reward["waypoint_bonus_weight"] == 1.0
    assert config.reward["collision_penalty"] == 2.0
    assert config.reward["collision_growth"] == 0.005
    assert config.training.c2 == 0.01
