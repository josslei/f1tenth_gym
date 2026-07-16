"""Python shim for the native F110 rollout kernel."""

from importlib import import_module

_native = import_module("f110_gym.rollout_kernel.natives._f110_rollout_kernel")

DEFAULT_PARAMS = _native.DEFAULT_PARAMS
STATE_COLUMNS = _native.STATE_COLUMNS
F110Action = _native.F110Action
F110Params = _native.F110Params
F110State = _native.F110State
F110ProgressReward = _native.F110ProgressReward
F110StepResult = _native.F110StepResult
Integrator = _native.Integrator


def step(
    state, action, params=None, integrator=Integrator.RK4, direct_accel_control=False
):
    return _native.step(
        state,
        action,
        DEFAULT_PARAMS if params is None else params,
        integrator,
        direct_accel_control,
    )


def step_batch(states, actions, params=None, integrator=Integrator.RK4):
    return _native.step_batch(
        states, actions, DEFAULT_PARAMS if params is None else params, integrator
    )


__all__ = [
    "DEFAULT_PARAMS",
    "STATE_COLUMNS",
    "F110Action",
    "F110Params",
    "F110ProgressReward",
    "F110State",
    "F110StepResult",
    "Integrator",
    "step",
    "step_batch",
]
