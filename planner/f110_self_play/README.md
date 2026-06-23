# F110 Self-Play

Native F110 self-play orchestration for planner training.

Shared C++ backend guidance lives in `../../docs/`.

This package owns:

- action lattice discretization
- environment backend abstraction
- trajectory recording
- episode-level logging metrics
- search algorithm selection

It composes generic tree search from `planner/tree_search/` and does not own
F110 dynamics itself.

## Build

```bash
cmake --preset default
cmake --build build
```

The presets intentionally do not force a CMake generator.

Run `scripts/build_native_backends_release.sh` or
`scripts/build_native_backends_debug.sh` from the repo root.
