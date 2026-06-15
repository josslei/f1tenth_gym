from pathlib import Path

import numpy as np

from models.ppo import PPOConfig
from runs.train_ppo_controller import F1TenthPPOReward


def _obs(
    *,
    vx: float = 0.0,
    collision: bool = False,
    theta: float = 0.0,
    steer: float = 0.0,
    px: float = 0.0,
    py: float = 0.0,
):
    return {
        "ego_idx": 0,
        "linear_vels_x": np.array([vx], dtype=np.float64),
        "linear_vels_y": np.array([0.0], dtype=np.float64),
        "collisions": np.array([collision], dtype=bool),
        "poses_theta": np.array([theta], dtype=np.float64),
        "poses_x": np.array([px], dtype=np.float64),
        "poses_y": np.array([py], dtype=np.float64),
        "steer_angle": np.array([steer], dtype=np.float64),
    }


def _reward(tmp_path: Path, **kwargs) -> F1TenthPPOReward:
    path_len = 10.0
    waypoint_path = tmp_path / "waypoints.csv"
    lines = "\n".join(
        f"{x},0.0" for x in [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, path_len]
    )
    waypoint_path.write_text(lines, encoding="utf-8")
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
        progress_weight=0.0,
        steer_smoothness_weight=0.0,
    )

    assert np.isclose(reward(_obs(vx=8.0), terminated=False), 0.8)


def test_forward_progress_rewarded(tmp_path: Path):
    reward = _reward(
        tmp_path,
        speed_reward_weight=0.0,
        progress_weight=2.0,
        steer_smoothness_weight=0.0,
    )

    reward.prev_arc_length = 0.0
    reward.prev_steer = 0.0
    # Car at (0.6, 0).  Nearest waypoint is (1, 0) → arc ≈ 1.0.
    # Progress = 1.0 − 0.0 = 1.0, reward = 2.0 × 1.0 = 2.0.
    r = reward(_obs(px=0.6, py=0.0, steer=0.0), terminated=False)
    assert np.isclose(r, 2.0, atol=0.1)


def test_collision_returns_fixed_penalty(tmp_path: Path):
    reward = _reward(
        tmp_path,
        speed_reward_weight=0.0,
        progress_weight=0.0,
        steer_smoothness_weight=0.0,
        collision_penalty=2.0,
    )

    assert reward(_obs(vx=8.0, collision=True), terminated=True) == -2.0


def test_steer_smoothness_penalty(tmp_path: Path):
    reward = _reward(
        tmp_path,
        speed_reward_weight=0.0,
        progress_weight=0.0,
        steer_smoothness_weight=1.0,
    )

    # steer=0.5, prev_steer=0.0 (initial) → delta=0.5 → penalty=0.5
    r = reward(_obs(vx=0.0, steer=0.5), terminated=False)
    assert np.isclose(r, -0.5)


def test_default_ppo_config_uses_progress_reward(tmp_path: Path):
    config = PPOConfig.from_yaml(Path("configs/ppo/default.yaml"))

    assert config.action["velocity_min"] > 0.0
    assert config.reward["speed_reward_weight"] == 0.1
    assert config.reward["progress_weight"] == 2.0
    assert config.reward["steer_smoothness_weight"] == 0.5
    assert config.reward["collision_penalty"] == 50
    assert config.training.c2 == 0.01
