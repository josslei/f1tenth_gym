"""PPO-trained neural-network controller.

Loads a Lightning PPO policy checkpoint and provides the standard
:class:`~controllers.controller_base.Controller` interface so it can be
used interchangeably with classical controllers in drive scripts.

The policy maps the full F1TENTH observation (lidar scans + optional ego
state) to a continuous action in [-1, 1]², which is then scaled to the
environment's steering and velocity ranges.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from .controller_base import ControlCommand, Controller, VehicleState
from models.policies import Policy, make_policy
from models.ppo import PolicyConfig
from utils.f110_env import (
    F1TenthActionConfig,
    F1TenthObservationConfig,
    build_observation,
    DEFAULT_DEVICE,
    with_resampled_waypoints,
)

# ── Default action bounds ─────────────────────────────────────────────────────
# Match the default F1TENTH environment limits and Stanley's DELTA_MAX.

STEERING_MIN: float = -0.4189
STEERING_MAX: float = 0.4189
VELOCITY_MIN: float = 0.0
VELOCITY_MAX: float = 8.0


class PPOController(Controller):
    """Controller that uses a trained PPO policy for steering and velocity.

    Parameters
    ----------
    policy:
        A loaded, eval-mode :class:`models.policies.Policy`.
    observation_config:
        Observation preprocessing config matching the one used at training time.
    steering_min:
        Lower steering bound (rad).  Default matches ``f110-v0`` default.
    steering_max:
        Upper steering bound (rad).  Default matches ``f110-v0`` default.
    velocity_min:
        Minimum velocity command (m/s).
    velocity_max:
        Maximum velocity command (m/s).
    device:
        Torch device for inference.
    """

    def __init__(
        self,
        policy: Policy,
        observation_config: F1TenthObservationConfig,
        steering_min: float = STEERING_MIN,
        steering_max: float = STEERING_MAX,
        velocity_min: float = VELOCITY_MIN,
        velocity_max: float = VELOCITY_MAX,
        device: torch.device = DEFAULT_DEVICE,
    ) -> None:
        self.policy = policy.to(device).eval()
        self.observation_config = observation_config
        self.steering_min = steering_min
        self.steering_max = steering_max
        self.velocity_min = velocity_min
        self.velocity_max = velocity_max
        self.device = device

        self.vehicle_state = VehicleState(0, 0, 0, 0)
        self._obs: dict[str, Any] | None = None

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        observation_config: F1TenthObservationConfig | None = None,
        action_config: F1TenthActionConfig | None = None,
        steering_min: float = STEERING_MIN,
        steering_max: float = STEERING_MAX,
        waypoint_path: str | Path | None = None,
        device: torch.device = DEFAULT_DEVICE,
    ) -> PPOController:
        """Construct from a ``final_model.pt`` checkpoint saved by
        :func:`runs.train_ppo_controller.save_policy`.

        The checkpoint is expected to contain ``obs_dim``, ``action_dim``,
        ``policy`` (a :class:`PolicyConfig`-shaped dict), and
        ``policy_state_dict``.
        """
        checkpoint = torch.load(
            str(checkpoint_path), map_location=device, weights_only=False
        )
        obs_dim: int = checkpoint["obs_dim"]
        action_dim: int = checkpoint["action_dim"]
        policy_config = PolicyConfig(**checkpoint["policy"])
        policy = make_policy(policy_config, obs_dim=obs_dim, action_dim=action_dim)
        policy.load_state_dict(checkpoint["policy_state_dict"])

        if observation_config is None:
            if "observation_config" in checkpoint:
                observation_config = F1TenthObservationConfig(
                    **checkpoint["observation_config"]
                )
            else:
                observation_config = F1TenthObservationConfig()

        if observation_config.include_waypoints and waypoint_path is not None:
            waypoints_xy = np.loadtxt(str(waypoint_path), delimiter=",", skiprows=1)[
                :, :2
            ]
            observation_config = with_resampled_waypoints(
                observation_config, waypoints_xy
            )

        velocity_min = VELOCITY_MIN
        velocity_max = VELOCITY_MAX
        if action_config is not None:
            velocity_min = action_config.velocity_min
            velocity_max = action_config.velocity_max

        return cls(
            policy=policy,
            observation_config=observation_config,
            steering_min=steering_min,
            steering_max=steering_max,
            velocity_min=velocity_min,
            velocity_max=velocity_max,
            device=device,
        )

    # ── Controller interface ───────────────────────────────────────────────

    def reset(self) -> None:
        """Reset vehicle state to origin and clear the cached observation."""
        self.vehicle_state = VehicleState(0, 0, 0, 0)
        self._obs = None

    def update(
        self, vehicle_state: VehicleState, obs: dict[str, Any] | None = None
    ) -> None:
        """Receive the latest vehicle state and an optional raw gym observation.

        The PPO-augmented ``obs`` dict is required for :meth:`control` to
        produce a meaningful action.  It should contain the raw environment
        keys plus ``"steer_angle"`` from ``utils.f110_env.add_control_state``
        when ``include_ego_state`` is active.
        """
        self.vehicle_state = vehicle_state
        if obs is not None:
            self._obs = obs

    def control(self) -> ControlCommand:
        """Compute steering and velocity from the latest observation.

        Returns zero command when no observation has been received.
        """
        if self._obs is None:
            return ControlCommand(steering=0.0, velocity=0.0)

        obs_np = build_observation(self._obs, self.observation_config)
        obs_tensor = torch.as_tensor(
            obs_np, dtype=torch.float32, device=self.device
        ).unsqueeze(0)

        with torch.no_grad():
            action, _log_prob, _value = self.policy.act(obs_tensor, deterministic=True)

        action_np = action.squeeze(0).cpu().numpy()
        steering = float(
            np.interp(action_np[0], [-1.0, 1.0], [self.steering_min, self.steering_max])
        )
        velocity = float(
            np.interp(action_np[1], [-1.0, 1.0], [self.velocity_min, self.velocity_max])
        )

        return ControlCommand(steering=steering, velocity=velocity)
