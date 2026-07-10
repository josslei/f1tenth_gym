# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

See `AGENTS.md` for the canonical project knowledge base (structure, conventions,
anti-patterns, code map). This file adds Claude-specific operating notes and
should be read alongside it. Subsystem-local `AGENTS.md` files exist under
`gym/f110_gym/envs/AGENTS.md` (and similar) — check for one before editing a
subsystem in depth.

## What This Is

Gymnasium-compatible F1TENTH simulator (`f110-v0`) plus several planner/controller
experiments built on top: Pure Pursuit/Stanley/PPO/LMPC controllers, a MuZero
self-play training stack, and raceline/map generation tooling.

`gym/`, `scripts/raceline_opt/`, and everything under `thirdparty/` and `ref/`
are vendored/downloaded from elsewhere — treat as black boxes unless a task
specifically requires changing them.

## Commands

```bash
# Install
pip install -e ".[dev,render]"      # core + test tooling + pyglet renderer
pip install -e ".[tools]"           # raceline/map-generation plotting deps
pip install -e ".[rl,dev]"          # PPO/RL training deps

# Test (run from repo root; tests/conftest.py inserts gym/ into sys.path)
pytest -q
pytest tests/test_dynamics.py -q                 # single file
pytest tests/test_dynamics.py::test_name -q      # single test

# Lint/format/type-check (what actually runs in .pre-commit-config.yaml)
pre-commit run --files <changed-file>...
pre-commit run --all-files

# Native builds
cmake -S gym/f110_gym/rollout_kernel -B gym/f110_gym/rollout_kernel/build && cmake --build gym/f110_gym/rollout_kernel/build --config Release
scripts/build_native_backends_release.sh          # builds planner/f110_self_play only (what MuZero training loads)
scripts/build_native_backends_release.sh --all    # also builds planner/tree_search and the rollout kernel standalone
scripts/build_native_backends_debug.sh[.--all]    # debug preset: symbols + assertions

# Example CLI workflows
python scripts/optimize_mintime.py --track maps/f1tenth_racetracks/Spielberg/Spielberg_centerline.csv --output outputs/waypoints/Spielberg_mintime.csv --save_plot
python scripts/generate_map.py maps/f1tenth_racetracks/Spielberg/Spielberg_centerline.csv -o outputs/maps/Spielberg -r 1.0
python runs/train_ppo_controller.py --config configs/ppo/default.yaml
python runs/train_muzero_planner.py --config configs/muzero/default.yaml
```

Note: `docs/coding_style.md` describes a black/isort/flake8/mypy stack, but the
actual `.pre-commit-config.yaml` runs `ruff` (lint+format), `clang-format`
(C/C++/CUDA), and `pyright` — trust the config file over that doc for tooling.
Ruff has no repo-specific config, so it runs with defaults.

## Architecture

**Non-standard source layout.** `pyproject.toml` maps `gym/f110_gym` to the
installed `f110_gym` package, alongside top-level `controllers`, `utils`,
`models`, `planner`. `tests/conftest.py` inserts `gym/` into `sys.path` and
force-reloads `f110_gym` — always run pytest from the repo root.
`scripts/optimize_mintime.py` similarly inserts `scripts/raceline_opt` into
`sys.path` for legacy nested imports; don't relocate raceline internals
without updating that, `pyrightconfig.json` extraPaths, and the nested
imports together.

**Two-tier native split — Python simulator is the source of truth.**
`gym/f110_gym/envs/` (numba-heavy Python) is the authoritative simulator
semantics (dynamics, scan, collision, termination). `gym/f110_gym/rollout_kernel/`
is a separate compiled C++17 pybind11 kernel that mirrors a subset of that
behavior for high-throughput planner rollouts (batched transitions, no
Gymnasium wrapper, no rendering, no Python observation dicts). **Any change to
vehicle dynamics, control preprocessing, integration, scan behavior, map
interpretation, TTC, collision, reward timing, or termination in `envs/` must
be mirrored in `rollout_kernel/` in the same change**, with parity tests added
under `tests/f110_rollout_kernel/` (compares Python vs. native output on
deterministic inputs — `test_rollout_kernel_parity.py` at the top level runs
these). Never put F110-specific simulation code in `planner/tree_search/`
(generic search only) — F110 logic composes it via `planner/f110_self_play/`
or lives in `gym/f110_gym/rollout_kernel/`.

**Native backend packages** (each its own CMake project + pybind11 module):
- `gym/f110_gym/rollout_kernel/` — F110 transition kernel for planner search
- `planner/tree_search/` — generic batched MCTS/MuZero search, header-only-ish scaffold
- `planner/f110_self_play/` — composes tree_search + rollout_kernel for MuZero self-play training; the only native module MuZero training actually loads, hence the default (non-`--all`) build scripts only build this one
- All expect vendored deps under `thirdparty/` (`pybind11`, `libtorch`)

**LMPC (`controllers/lmpc/`, being rebuilt from scratch).** The prior C++ port
of `Racing-LMPC-ROS2` (LGPLv3) was removed after it never completed a clean
lap despite extensive tuning; upstream source remains available for
reference at `ref/Racing-LMPC-ROS2` (untracked, outside this repo's history,
not vendored/buildable here). `controllers/lmpc/lmpc.py` is currently just a
`Controller` stub — no trajectory generation, seed-lap, or native solver
exists yet. The intended design going forward is for trajectory generation
to happen online in C++ rather than via an offline Python script/table, so
don't recreate the old `scripts/generate_lmpc_trajectory.py` /
`generate_lmpc_seed_lap.py` two-artifact offline workflow.

**Config-driven training entry points.** `runs/train_ppo_controller.py` and
`runs/train_muzero_planner.py` take a single `--config` YAML under `configs/`
that owns nearly all tunables (iteration counts, rollout length, reward
params, map, hyperparameters) — prefer editing/adding a config over passing
ad hoc flags. `runs/train_muzero_planner.py` writes a scripted model to
`outputs/rl/muzero_f110_gym_10/current_model.pt` before native self-play loads
it; if a native module was rebuilt in a different mode (debug vs. release)
than expected, or CPU/GPU device handling looks stale, rebuild before
debugging simulation logic. Use `lldb` for planner segfaults (see
`docs/cpp-backend.md`); `faulthandler` is already enabled in the training
entrypoint.

**Driving demo scripts under `runs/`** — currently just `waypoint_drive.py`
(Pure Pursuit/Stanley). There is no LMPC demo script yet; `lmpc_drive.py` was
removed with the old C++ port and should stay split from `waypoint_drive.py`
if/when it's recreated, so safe-set initialization requirements don't leak
into non-LMPC demos.

**Outputs are gitignored and regenerated locally**, not committed:
`outputs/{waypoints,centerlines,maps,rl}/`. Regenerate raceline outputs after
changing `configs/raceline/f110.ini`.

## Repo-Wide Conventions Not to Rediscover

- `maps/f1tenth_racetracks/` is a git submodule (`f1tenth/f1tenth_racetracks`), not ordinary source.
- Bundled map package-data must stay under `gym/f110_gym/envs/maps/` to be picked up by setuptools.
- Never copy from `*_backup.py` files (e.g. `gym/f110_gym/envs/f110_env_backup.py`) — legacy snapshots excluded from type checking.
- **Assume-correct, don't guard**: functions have a fixed input contract; don't add defensive `isinstance`/`None` checks or `ValueError`s inside implementation code for conditions callers already guarantee.
- Don't document/implement variable-friction raceline support — the current wrapper forces constant friction; assets for the variable case aren't restored.
- `tests/f110_gym/legacy_scan.npz` is an intentional binary fixture, not stray data.
