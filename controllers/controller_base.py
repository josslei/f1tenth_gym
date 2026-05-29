from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class VehicleState:
    x: float
    y: float
    yaw: float
    speed: float


@dataclass(frozen=True)
class ControlCommand:
    steering: float
    velocity: float


class Controller(ABC):
    @abstractmethod
    def reset(self) -> None: ...

    @abstractmethod
    def update(self, vehicle_state: VehicleState, *args, **kwargs) -> None: ...

    @abstractmethod
    def control(self) -> ControlCommand: ...
