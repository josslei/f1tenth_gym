# LMPC Controller Boundary

This package is the Gym-facing entry point for the Racing-LMPC integration.

## Design

- The MPC/LMPC implementation lives in C++.
- Python is only used to bind the C++ controller into the Gym loop.
- Reuse and modify `thirdparty/Racing-LMPC-ROS2` where possible instead of copying large code blocks.
- Keep Racing-LMPC ROS behavior intact while adding a reusable non-ROS controller path.

## Immediate Gym Target

1. Build a C++ adapter that accepts Gym-derived vehicle state and a Racing-LMPC trajectory table.
2. Convert Gym/global state to Racing-LMPC-compatible Frenet state using the table `PX/PY` path.
3. Run normal MPC first with learning disabled.
4. Return Gym action format: `[desired_steering_angle, desired_velocity]`.

`LMPCController` should be constructed from a generated Racing-LMPC trajectory
table via `LMPCController.from_trajectory_table(...)`. On each `update(...)`, it
projects `VehicleState.x/y/yaw` into `s`, `e_y`, and `e_psi` before calling the
native C++ controller. The Gym adapter interpolates the table at the current
Frenet `s` and passes local `SPEED`, `CURVATURE`, and left/right lateral bounds
into the native C++ MPC before each solve.

`LMPCController.from_centerline_csv(...)` is kept for simple projection tests and
early experiments, but it is not the preferred LMPC entry point.

For full Gym observations, prefer `update_from_observation(obs)`. It preserves
Gym's lateral velocity and yaw-rate fields instead of defaulting them to zero.

## Generate Trajectory Table

The upstream `RacingTrajectory` format is a whitespace-delimited numeric table
with 17 columns and no header:

```text
PX PY PZ YAW SPEED CURVATURE DIST_TO_SF_BWD DIST_TO_SF_FWD REGION LEFT_BOUND_X LEFT_BOUND_Y RIGHT_BOUND_X RIGHT_BOUND_Y BANK LON_ACC LAT_ACC TIME
```

Generate a conservative centerline-based table from a centerline-width CSV:

```bash
python scripts/generate_lmpc_trajectory.py \
  maps/f1tenth_racetracks/Spielberg/Spielberg_centerline.csv \
  -o outputs/lmpc_trajectories/Spielberg_centerline.txt
```

The script reads `configs/raceline/f110.ini` by default. It uses:

- `GENERAL_OPTIONS.stepsize_opts.stepsize_interp_after_opt` for output spacing
- `GENERAL_OPTIONS.veh_params.v_max` for the speed cap
- `OPTIMIZATION_OPTIONS.optim_opts_mintime.ay_safe` for lateral acceleration if set
- otherwise `OPTIMIZATION_OPTIONS.optim_opts_mintime.mue * GENERAL_OPTIONS.veh_params.g`

Default speed mode is curvature-limited:

```text
speed = min(v_max, sqrt(ay_limit / (abs(curvature) + eps)))
```

The generated profile is then constrained by forward/backward acceleration passes
using `ax_pos_safe` and `ax_neg_safe` when present, otherwise `mue * g`. This is
important: a purely local curvature speed cap can command full straight-line
speed immediately before a corner.

Regenerate this file after changes to `configs/raceline/f110.ini` or
`scripts/generate_lmpc_trajectory.py`.

Run the Gym demo after generating the table:

```bash
python runs/waypoint_drive.py
```

## Build Native Binding

From the repository root:

```bash
cmake -S controllers/lmpc -B controllers/lmpc/build
cmake --build controllers/lmpc/build --config Release
```

The build writes `lmpc_native` into `controllers/lmpc/` so `binding.py` can import it directly.

## Later Paper Target

After normal MPC runs in Gym, add the sparse affine error-dynamics update from `ref/lmpc.tex`:

- update `A^e`
- update `B^e`
- update `C^e`
- reuse/fix the regression logic in `thirdparty/Racing-LMPC-ROS2/src/vehicle_dynamics_models/racing_trajectory/src/safe_set.cpp`

## Licensing

Racing-LMPC is LGPLv3. Preserve upstream copyright/license headers in modified
submodule files and add attribution for any reused code in this package.
