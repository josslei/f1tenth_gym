"""Shared conversion from recorded states to aligned LMPC lap arrays."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class LapSample:
    """Plant state sampled at one transition boundary."""

    x: np.ndarray
    actual_delta: float
    raw_speed: float


def build_lap_arrays(
    samples: Sequence[LapSample], dt: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build x_0..x_T, realized u_0..u_(T-1), and J_0..J_T."""
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError("dt must be finite and strictly positive")
    if len(samples) < 2:
        raise ValueError("at least two lap samples are required")

    states: list[np.ndarray] = []
    raw_speeds: list[float] = []
    actual_deltas: list[float] = []
    for sample in samples:
        state = np.asarray(sample.x, dtype=np.float64)
        if state.shape != (6,):
            raise ValueError(
                f"each sample state must have shape (6,), got {state.shape}"
            )
        if not np.isfinite(state).all():
            raise ValueError("sample states must contain only finite values")
        if not np.isfinite(sample.raw_speed):
            raise ValueError("sample raw speeds must be finite")
        if not np.isfinite(sample.actual_delta):
            raise ValueError("sample steering angles must be finite")
        states.append(state)
        raw_speeds.append(float(sample.raw_speed))
        actual_deltas.append(float(sample.actual_delta))

    x_lap = np.ascontiguousarray(np.column_stack(states), dtype=np.float64)
    speed = np.asarray(raw_speeds, dtype=np.float64)
    acceleration = np.diff(speed) / dt
    steering = np.asarray(actual_deltas[:-1], dtype=np.float64)
    u_lap = np.ascontiguousarray(np.vstack([acceleration, steering]), dtype=np.float64)
    transitions = len(samples) - 1
    J_lap = np.ascontiguousarray(np.arange(transitions, -1, -1, dtype=np.float64))

    assert x_lap.shape == (6, transitions + 1)
    assert u_lap.shape == (2, transitions)
    assert J_lap.shape == (transitions + 1,)
    assert J_lap[0] == transitions
    assert J_lap[-1] == 0.0
    return x_lap, u_lap, J_lap
