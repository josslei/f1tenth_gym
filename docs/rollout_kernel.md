# F110 Rollout Kernel

`gym/f110_gym/rollout_kernel/` is the native C++ rollout kernel for high-throughput planning and search. It is F110-specific, but it is not a second Gymnasium environment.

## Purpose

The rollout kernel exists so batched MCTS, AlphaZero-style search, MuZero actors, and other planners can evaluate many imagined F110 transitions without crossing the Python/Gym boundary for each node expansion.

The Python simulator remains the behavioral reference:

```text
gym/f110_gym/envs/        authoritative Python/Numba simulator and Gym API
gym/f110_gym/rollout_kernel/  native F110 transition kernel for planner rollouts
planner/tree_search/      generic search algorithms; no F110 simulator logic
```

## Why This Lives Under `gym/f110_gym/`

The rollout kernel contains F110-specific simulation logic: vehicle dynamics, map interpretation, scan behavior, collision checks, and termination/reward semantics needed by search. Keeping it under `gym/f110_gym/` makes that ownership explicit and keeps `planner/tree_search/` reusable.

It should not live directly under top-level `gym/` because `gym/` is only the source root used to install the `f110_gym` package. Code under top-level `gym/` but outside `f110_gym/` is easy to miss in package discovery and makes ownership less clear.

## Naming

Use `rollout_kernel`, not `kernel`, `cpp_env`, or `native_env`.

The name is intentionally narrow:

- `rollout` means it is for imagined planner transitions.
- `kernel` means it exposes compact transition primitives, not a full environment wrapper.
- It avoids implying that `F110Env` itself is backed by C++.

## Non-Goals

Do not add these responsibilities to the rollout kernel:

- Gymnasium `Env` compatibility.
- Rendering or viewer integration.
- Python observation dictionaries.
- Training loop orchestration.
- Generic MCTS, AlphaZero, or MuZero tree logic.

Those belong in `gym/f110_gym/envs/`, viewer modules, Python wrappers, or `planner/tree_search/` respectively.

## Expected Interface Shape

Prefer compact, deterministic, batched APIs:

```cpp
struct F110State;
struct F110Action;
struct F110StepResult;

void step_batch(
    const F110State* states,
    const F110Action* actions,
    F110StepResult* results,
    int batch_size
);
```

For planner use, avoid per-transition callbacks into Python. If neural inference is needed, batch it at the planner boundary rather than calling Python from each rollout expansion.

## Build

The kernel is a compiled C++17 pybind11 module. Build it from the kernel directory:

```bash
cmake -S gym/f110_gym/rollout_kernel -B gym/f110_gym/rollout_kernel/build
cmake --build gym/f110_gym/rollout_kernel/build --config Release
```

The extension is written to:

```text
gym/f110_gym/rollout_kernel/natives/_f110_rollout_kernel.*.so
```

`natives/` is ignored by git because it contains local build artifacts.

## Current Scope

The first kernel slice mirrors the Python control preprocessing and vehicle integration path:

- state layout: `[x, y, steer_angle, velocity, yaw_angle, yaw_rate, slip_angle]`
- two-step steering delay buffer
- PID conversion from desired steering/speed to steering velocity/acceleration
- acceleration and steering constraints
- kinematic low-speed fallback
- single-track dynamics
- Euler and RK4 integration
- yaw wrapping
- batched stepping over flat arrays

It does not yet implement map loading, distance-transform LiDAR, TTC, vehicle-vehicle collision, lap/checkpoint completion, or scan observations. Add those incrementally with parity tests against `gym/f110_gym/envs/`.

## Parity Rule

When simulation behavior changes in `gym/f110_gym/envs/`, check whether the rollout kernel mirrors that behavior.

Update the rollout kernel in the same change when modifying:

- vehicle state layout or semantics
- PID/control preprocessing
- steering delay behavior
- Euler/RK4 integration
- `vehicle_dynamics_st` or `vehicle_dynamics_ks`
- map YAML/image interpretation
- distance-transform scan behavior
- scan noise assumptions
- TTC collision checks
- vehicle-vehicle collision checks
- reward timing
- lap/checkpoint termination semantics used by planners

Add or update parity tests for changed behavior. Parity tests should compare the Python simulator and native rollout kernel on deterministic inputs.

## Determinism

Search rollouts should be deterministic by default. The Python simulator can add LiDAR noise, but the rollout kernel should not add stochastic scan noise unless a planner explicitly models stochastic transitions.

## Planner Integration

F110-specific planner adapters may compose the generic search package with this kernel:

```text
planner/tree_search/       generic batched search
gym/f110_gym/rollout_kernel/  F110 transition kernel
planner/f110*/             optional adapters that combine both
```

Do not place F110 dynamics, scan, map, or collision code under `planner/tree_search/`.
