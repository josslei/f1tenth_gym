# LMPC Integration

The production controller is the LearningMPC implementation from
`ref/LearningMPC` with project-specific behavior restricted to adaptation
files.

## Reference Files

These files are copied byte-for-byte:

- `src/lmpc_core.cpp` from `ref/LearningMPC/src_gym/cpp/lmpc_core.cpp`
- `include/track.h`
- `include/occupancy_grid.h`
- `include/spline.h`
- `include/CSVReader.h`
- `include/car_params.h`

The five headers retain their original `LearningMPC/...` include directives.
`CMakeLists.txt` creates that compatibility layout in the build directory; the
source include directory remains flat.

## Adaptation Boundary

- `include/lmpc_controller.hpp` is the stable C++ controller API.
- `src/lmpc_controller.cpp` adapts that API to the copied `LMPCCore`.
- `src/bindings.cpp` provides NumPy/`casadi::DM` conversion at the Python
  boundary.
- `lmpc.py` converts the map and existing seed-lap formats into the files and
  occupancy grid expected by LearningMPC.

LearningMPC owns its safe set, lap detection, cost-to-go update, warm start,
linearization, and OSQP solve. Its affine dynamics constraints retain the
per-stage `Ad`, `Bd`, and `hd` matrices needed for future dynamics-error
corrections.

## Fixed Conventions

- Simulator and controller timestep: `0.025` seconds.
- Controller state: `[x, y, yaw, speed, yaw_rate, slip_angle]`.
- Controller output: `[acceleration, steering_angle]`.
- Gym must be constructed with `direct_accel_control=True`; its action remains
  `[steering_angle, longitudinal_command]`.
- The optional direct-acceleration mode bypasses only the velocity PID. Gym's
  steering delay, steering-rate limit, acceleration limits, and velocity limits
  remain active.
