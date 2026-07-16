# F110 Rollout Kernel

This directory contains the native C++ rollout kernel for planner/search rollouts.

It is F110-specific, but it is not a Gymnasium environment. The Python simulator in `gym/f110_gym/envs/` remains the behavioral reference.

## Build

```bash
cmake -S gym/f110_gym/rollout_kernel -B gym/f110_gym/rollout_kernel/build
cmake --build gym/f110_gym/rollout_kernel/build --config Release
```

The compiled extension is written under `natives/`.

## Current Scope

Implemented:

- compact C++ state/action/parameter types
- Python-compatible default F110 parameters
- PID control preprocessing
- steering delay buffer
- optional direct-acceleration preprocessing for `step()`; velocity control
  remains the default
- acceleration and steering constraints
- kinematic low-speed fallback
- single-track dynamics
- Euler and RK4 integration
- batched stepping through `step_batch()`

Not implemented yet:

- map loading
- distance-transform LiDAR
- TTC checks
- vehicle-vehicle collision
- lap/checkpoint termination
- Gymnasium wrappers or rendering

Update this kernel and its parity tests whenever simulator behavior changes in `gym/f110_gym/envs/`.
