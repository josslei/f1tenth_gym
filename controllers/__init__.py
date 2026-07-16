"""Controller interfaces and reference implementations."""

from typing import Any

from .controller_base import ControlCommand, Controller, VehicleState

__all__ = [
    "ControlCommand",
    "Controller",
    "LMPCController",
    "PPOController",
    "PurePursuit",
    "Stanley",
    "VehicleState",
]


def __getattr__(name: str) -> Any:
    if name == "LMPCController":
        from .lmpc import LMPCController

        return LMPCController
    if name == "PPOController":
        from .ppo import PPOController

        return PPOController
    if name == "PurePursuit":
        from .pure_pursuit import PurePursuit

        return PurePursuit
    if name == "Stanley":
        from .stanley import Stanley

        return Stanley
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
