from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np

from .controller_base import ControlCommand, Controller, VehicleState


@dataclass(frozen=True)
class PPOObservationConfig:
    scan_size: int = 108
    scan_max_m: float = 30.0
    include_ego_state: bool = True
    speed_scale: float = 8.0
    yaw_rate_scale: float = 10.0


@dataclass(frozen=True)
class PPOActionConfig:
    velocity_min: float = 0.0
    velocity_max: float = 8.0


def _coerce_observation_config(
    config: PPOObservationConfig | Mapping[str, Any] | None,
) -> PPOObservationConfig:
    if config is None:
        return PPOObservationConfig()
    if isinstance(config, PPOObservationConfig):
        return config
    return PPOObservationConfig(**dict(config))


def _coerce_action_config(
    config: PPOActionConfig | Mapping[str, Any] | None,
) -> PPOActionConfig:
    if config is None:
        return PPOActionConfig()
    if isinstance(config, PPOActionConfig):
        return config
    return PPOActionConfig(**dict(config))


def _get_ego_index(obs: Mapping[str, Any]) -> int:
    return int(np.asarray(obs.get("ego_idx", 0)).item())


def observation_dim(config: PPOObservationConfig | None = None) -> int:
    obs_config = config or PPOObservationConfig()
    ego_state_dim = 6 if obs_config.include_ego_state else 0
    return obs_config.scan_size + ego_state_dim


def build_ppo_observation(
    obs: Mapping[str, Any],
    config: PPOObservationConfig | None = None,
) -> np.ndarray:
    obs_config = config or PPOObservationConfig()
    ego = _get_ego_index(obs)
    scans = np.asarray(obs["scans"], dtype=np.float64)
    ego_scan = scans[ego] if scans.ndim > 1 else scans
    scan_indices = np.linspace(0, ego_scan.shape[0] - 1, obs_config.scan_size).astype(
        np.int64
    )
    scan = np.nan_to_num(
        ego_scan[scan_indices],
        nan=obs_config.scan_max_m,
        posinf=obs_config.scan_max_m,
        neginf=0.0,
    )
    scan = np.clip(scan, 0.0, obs_config.scan_max_m) / obs_config.scan_max_m
    features = [scan]

    if obs_config.include_ego_state:
        linear_vels_x = np.asarray(obs["linear_vels_x"], dtype=np.float64)[ego]
        linear_vels_y = np.asarray(obs["linear_vels_y"], dtype=np.float64)[ego]
        ang_vels_z = np.asarray(obs["ang_vels_z"], dtype=np.float64)[ego]
        theta = np.asarray(obs["poses_theta"], dtype=np.float64)[ego]
        collision = np.asarray(obs["collisions"], dtype=np.float64)[ego]
        features.append(
            np.array(
                [
                    np.clip(linear_vels_x / obs_config.speed_scale, -1.0, 1.0),
                    np.clip(linear_vels_y / obs_config.speed_scale, -1.0, 1.0),
                    np.clip(ang_vels_z / obs_config.yaw_rate_scale, -1.0, 1.0),
                    np.sin(theta),
                    np.cos(theta),
                    np.clip(collision, 0.0, 1.0),
                ],
                dtype=np.float64,
            )
        )

    return np.nan_to_num(
        np.concatenate(features).astype(np.float32),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )


def scale_ppo_action(
    action: np.ndarray,
    f110_env: gym.Env,
    config: PPOActionConfig | None = None,
) -> np.ndarray:
    action_config = config or PPOActionConfig()
    normalized_action = np.clip(
        np.asarray(action, dtype=np.float64).reshape(2), -1.0, 1.0
    )
    unwrapped: Any = f110_env.unwrapped
    params = unwrapped.params
    steering = np.interp(
        normalized_action[0],
        [-1.0, 1.0],
        [params["s_min"], params["s_max"]],
    )
    velocity = np.interp(
        normalized_action[1],
        [-1.0, 1.0],
        [action_config.velocity_min, action_config.velocity_max],
    )
    return np.array(
        [
            [
                float(np.clip(steering, params["s_min"], params["s_max"])),
                float(np.clip(velocity, params["v_min"], params["v_max"])),
            ]
        ],
        dtype=np.float64,
    )


def _load_torch() -> tuple[Any, Any]:
    try:
        import torch
        from torch import nn
    except ImportError as exc:
        raise ImportError(
            'PyTorch is required for PPO. Install it with `pip install -e ".[rl]"`.'
        ) from exc
    return torch, nn


def make_policy(
    obs_dim: int,
    action_dim: int = 2,
    hidden_size: int = 128,
) -> Any:
    torch, nn = _load_torch()

    class PPOPolicy(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(obs_dim, hidden_size),
                nn.Tanh(),
                nn.Linear(hidden_size, hidden_size),
                nn.Tanh(),
            )
            self.action_mean = nn.Linear(hidden_size, action_dim)
            self.value = nn.Linear(hidden_size, 1)
            self.log_std = nn.Parameter(torch.zeros(action_dim))

        def forward(self, obs):
            features = self.net(obs)
            mean = torch.tanh(self.action_mean(features))
            value = self.value(features).squeeze(-1)
            return mean, value

        def distribution(self, obs):
            mean, value = self(obs)
            std = torch.exp(self.log_std).expand_as(mean)
            return torch.distributions.Normal(mean, std), value

        def act(self, obs, deterministic: bool = False):
            dist, value = self.distribution(obs)
            raw_action = dist.mean if deterministic else dist.rsample()
            action = torch.clamp(raw_action, -1.0, 1.0)
            log_prob = dist.log_prob(raw_action).sum(dim=-1)
            return action, log_prob, value

        def evaluate_actions(self, obs, actions):
            dist, value = self.distribution(obs)
            clipped_actions = torch.clamp(actions, -1.0, 1.0)
            log_prob = dist.log_prob(clipped_actions).sum(dim=-1)
            entropy = dist.entropy().sum(dim=-1)
            return log_prob, entropy, value

    return PPOPolicy()


def load_policy_checkpoint(
    model_path: str | Path,
    obs_dim: int,
    action_dim: int = 2,
    hidden_size: int = 128,
    map_location: str = "cpu",
) -> Any:
    torch, _nn = _load_torch()
    checkpoint_path = Path(model_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"PPO model not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    if isinstance(checkpoint, Mapping):
        hparams = checkpoint.get("hyper_parameters", {})
        obs_dim = int(checkpoint.get("obs_dim", hparams.get("obs_dim", obs_dim)))
        action_dim = int(
            checkpoint.get("action_dim", hparams.get("action_dim", action_dim))
        )
        hidden_size = int(
            checkpoint.get("hidden_size", hparams.get("hidden_size", hidden_size))
        )

    policy = make_policy(
        obs_dim=obs_dim, action_dim=action_dim, hidden_size=hidden_size
    )
    if isinstance(checkpoint, Mapping):
        state_dict = checkpoint.get(
            "policy_state_dict", checkpoint.get("state_dict", checkpoint)
        )
    else:
        state_dict = checkpoint

    policy_state = {}
    for key, value in state_dict.items():
        policy_key = key.removeprefix("policy.")
        policy_state[policy_key] = value
    policy.load_state_dict(policy_state)
    policy.eval()
    return policy


class PPOController(Controller):
    def __init__(
        self,
        model_path: str | Path,
        *,
        env: gym.Env | None = None,
        observation_config: PPOObservationConfig | Mapping[str, Any] | None = None,
        action_config: PPOActionConfig | Mapping[str, Any] | None = None,
        hidden_size: int = 128,
    ) -> None:
        self.model_path = Path(model_path)
        self.env = env
        self.observation_config = _coerce_observation_config(observation_config)
        self.action_config = _coerce_action_config(action_config)
        self.hidden_size = hidden_size
        self.vehicle_state = VehicleState(0.0, 0.0, 0.0, 0.0)
        self._raw_observation: Mapping[str, Any] | None = None
        self._policy: Any | None = None

    def reset(self) -> None:
        self.vehicle_state = VehicleState(0.0, 0.0, 0.0, 0.0)
        self._raw_observation = None

    def update(self, vehicle_state: VehicleState, *args, **kwargs) -> None:
        self.vehicle_state = vehicle_state

    def update_from_observation(self, obs: Mapping[str, Any]) -> None:
        self._raw_observation = obs

    def set_environment(self, env: gym.Env) -> None:
        self.env = env

    def _load_policy(self) -> Any:
        if self._policy is None:
            self._policy = load_policy_checkpoint(
                self.model_path,
                obs_dim=observation_dim(self.observation_config),
                hidden_size=self.hidden_size,
            )
        return self._policy

    def control(self) -> ControlCommand:
        if self._raw_observation is None:
            raise RuntimeError(
                "PPOController.control() requires "
                "update_from_observation(obs) before control()."
            )
        if self.env is None:
            raise RuntimeError(
                "PPOController.control() requires an F110 environment. "
                "Pass env=... when constructing the controller or call "
                "set_environment(env) first."
            )

        torch, _nn = _load_torch()
        policy = self._load_policy()
        observation = build_ppo_observation(
            self._raw_observation, self.observation_config
        )
        obs_tensor = torch.as_tensor(observation, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            action, _log_prob, _value = policy.act(obs_tensor, deterministic=True)
        scaled_action = scale_ppo_action(
            action.squeeze(0).cpu().numpy(),
            self.env,
            self.action_config,
        )
        steering, velocity = np.asarray(scaled_action, dtype=np.float64).reshape(2)
        return ControlCommand(steering=float(steering), velocity=float(velocity))


__all__ = [
    "PPOActionConfig",
    "PPOController",
    "PPOObservationConfig",
    "build_ppo_observation",
    "load_policy_checkpoint",
    "make_policy",
    "observation_dim",
    "scale_ppo_action",
]
