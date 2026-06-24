from pathlib import Path

import numpy as np

from models.ppo import PPOConfig
from runs.train_ppo_controller import F1TenthPPOReward


def _obs(
    *,
    vx: float = 0.0,
    collision: bool = False,
    theta: float = 0.0,
    action_0: float = 0.0,
    action_1: float = 0.0,
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
        "prev_action": np.array([action_0, action_1], dtype=np.float64),
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


def test_zero_progress_reward_is_zero(tmp_path: Path):
    reward = _reward(
        tmp_path,
        q_s_progress=0.0,
        q_s_alpha=0.0,
        q_s_smooth=0.0,
    )

    assert np.isclose(reward(_obs(vx=8.0, px=5.0), terminated=False), 0.0)


def test_forward_progress_rewarded(tmp_path: Path):
    reward = _reward(
        tmp_path,
        q_s_progress=2.0,
        q_s_alpha=0.0,
        q_s_smooth=0.0,
    )

    reward.prev_arc_length = 0.0
    reward.last_action_0 = 0.0
    reward.last_action_1 = 0.0
    reward.has_last_action = False
    # Car at (0.6, 0).  Projection onto the line gives s ≈ 0.6.
    # Reward = 2.0 × 0.6 = 1.2.
    r = reward(_obs(px=0.6, py=0.0), terminated=False)
    assert np.isclose(r, 1.2, atol=0.1)


def test_backward_terminal_penalty(tmp_path: Path):
    reward = _reward(
        tmp_path,
        q_s_progress=0.0,
        q_s_alpha=0.0,
        q_s_smooth=0.0,
        terminal_penalty=2.0,
    )

    assert reward(_obs(vx=8.0, theta=3.2, px=5.0), terminated=True) == -2.0


def test_steer_smoothness_penalty(tmp_path: Path):
    reward = _reward(
        tmp_path,
        q_s_progress=0.0,
        q_s_alpha=0.0,
        q_s_smooth=1.0,
    )

    reward(_obs(vx=0.0, px=5.0, action_0=0.0, action_1=0.0), terminated=False)
    # action_0=0.5, prev_action=(0.0, 0.0) → delta^2 = 0.25
    r = reward(_obs(vx=0.0, px=5.0, action_0=0.5, action_1=0.0), terminated=False)
    assert np.isclose(r, -0.25)


def test_default_ppo_config_uses_progress_reward(tmp_path: Path):
    config = PPOConfig.from_yaml(Path("configs/ppo/default.yaml"))

    assert config.action["velocity_min"] > 0.0
    assert config.reward["q_s_progress"] == 1.0
    assert config.reward["q_s_alpha"] == 1.0
    assert config.reward["q_s_smooth"] == 0.0
    assert config.reward["terminal_penalty"] == 1000000.0
    assert config.training.c2 == 0.01
