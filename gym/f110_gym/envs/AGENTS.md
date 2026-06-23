# Simulator Internals Notes

## Scope

- Applies to `gym/f110_gym/envs/`.
- Focus: simulator behavior, maps, scan generation, dynamics, collisions.
- Root `AGENTS.md` covers setup, package layout, and test entry points.
- The Python simulator here is the behavioral reference for `gym/f110_gym/rollout_kernel/`, the native F110 rollout kernel used by planners.

## Where To Look

| Task | File | Notes |
|------|------|-------|
| Public env API | `f110_env.py` | `F110Env`, Gymnasium `reset`/`step`, map resolution, render hooks |
| Agent simulation | `base_classes.py` | `Simulator`, `RaceCar`, integrator selection, scan updates |
| Vehicle dynamics | `dynamic_models.py` | Constraints, PID, kinematic and single-track models |
| Collision checks | `collision_models.py` | GJK-style polygon helpers and multi-agent collision |
| LIDAR | `laser_models.py` | Distance transform, ray tracing, scan generation, TTC |
| Native rollout mirror | `../rollout_kernel/` | C++ rollout kernel that should mirror the semantics needed for planner search |
| Renderer backend | `rendering.py` | Low-level pyglet renderer used by viewer facade |
| Bundled maps | `maps/` | YAML plus PNG/PGM assets included as package data |

## Gymnasium Behavior

- `reset()` returns `(obs, info)` and accepts poses directly or through `options["poses"]`.
- `options["initial_poses"]` is also accepted for compatibility.
- Pose shape is `(num_agents, 3)`; dict poses need `x`, `y`, and `theta`.
- `step()` returns `(obs, reward, terminated, truncated, info)`.
- Reward is currently the timestep; `truncated` is currently always `False`.
- `terminated` is driven by collision or lap/checkpoint completion logic.
- `info["checkpoint_done"]` carries checkpoint/lap completion state.

## Simulator State

- Vehicle state order: `[x, y, steer_angle, vel, yaw_angle, yaw_rate, slip_angle]`.
- Control input order: `[desired_steering_angle, desired_velocity]`.
- Supported integrators: `Integrator.Euler`, `Integrator.RK4`.
- `Simulator` owns agents, collision state, observations, and multi-agent stepping.
- `RaceCar` owns one vehicle state, control buffers, TTC state, scan RNG, and opponent poses.
- `RaceCar.scan_simulator`, scan angles, cosines, and side distances are class-level shared state; map/scan config changes affect all agents.

## Maps

- Built-in names: `berlin`, `vegas`, `skirk`, `levine`, `stata_basement`.
- Most bundled maps use PNG; `levine` uses PGM.
- YAML and image stems must match the selected `map_ext`.
- YAML must provide `resolution` and `origin`.
- Image loading flips the bitmap vertically before scan use.
- Pixels `<= 128` are occupied; pixels `> 128` are free.
- Distance transform uses the thresholded image and YAML resolution.

## Numba Gotchas

- Hot-path functions use `@njit(cache=True)`.
- Keep array shapes and numeric dtypes stable around numba entry points.
- After changing numba-backed logic, stale `.nbc`/`.nbi` cache files can mask behavior changes.

## Rollout Kernel Parity

- Treat `gym/f110_gym/envs/` as the authoritative simulator semantics.
- When changing vehicle dynamics, control preprocessing, integration, scan behavior, map interpretation, TTC, collision checks, reward timing, or termination semantics, update `gym/f110_gym/rollout_kernel/` in the same change if the native kernel implements that behavior.
- Add or update parity tests that compare Python simulator outputs against the native rollout kernel for the changed behavior.
- The rollout kernel is for high-throughput planner/search rollouts. It should not grow Gymnasium wrappers, rendering, or Python observation-dict plumbing.

## Anti-Patterns

- Do not copy behavior from `f110_env_backup.py`; it is a legacy snapshot excluded from type checking.
- Do not treat map images as interchangeable without matching YAML stem, extension, origin, and resolution.
- Do not change shared `RaceCar` scan state assuming it is per-agent.
- Do not put F110-specific native simulation code in `planner/tree_search/`; that package should remain generic search machinery.
