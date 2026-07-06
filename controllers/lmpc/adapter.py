"""Python-only Gym API adapter for the C++ LMPC state types."""

from __future__ import annotations

from typing import Any

from .binding import GymVehicleState


def obs_to_gym_vehicle_state(obs: dict[str, Any]) -> GymVehicleState:
    ego = int(obs["ego_idx"])
    return GymVehicleState(
        x=float(obs["poses_x"][ego]),
        y=float(obs["poses_y"][ego]),
        yaw=float(obs["poses_theta"][ego]),
        v_x=float(obs["linear_vels_x"][ego]),
        v_y=float(obs["linear_vels_y"][ego]),
        omega=float(obs["ang_vels_z"][ego]),
    )
