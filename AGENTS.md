# PROJECT KNOWLEDGE BASE

**Generated:** 2026-06-02 11:47:11 CDT
**Commit:** cbab2fa
**Branch:** main

## OVERVIEW

Gymnasium-compatible F1TENTH simulator with optional realtime rendering, pure-pursuit control, map generation, and minimum-time raceline tooling. Python 3.11+; package source is split across non-standard top-level directories.

## STRUCTURE

```text
./
├── gym/f110_gym/          # installed as f110_gym; env, models, viewer, rollout kernel, bundled maps
├── controllers/           # installed as controllers; controller ABC + PurePursuit
├── utils/                 # installed as utils; waypoint/viewer helpers
├── scripts/               # CLI tools; raceline optimizer and map generation
├── configs/raceline/      # minimum-time optimization parameters
├── tests/                 # pytest suite; injects gym/ into sys.path
├── maps/                         # map assets (submodules + custom)
│   ├── f1tenth_racetracks/       # git submodule: f1tenth racetrack CSV/YAML/PNG assets
│   └── custom/                   # user-provided maps
└── outputs/                      # gitignored generated racelines, plots, maps
```

## WHERE TO LOOK

| Task | Location | Notes |
|------|----------|-------|
| Gymnasium env API | `gym/f110_gym/envs/f110_env.py` | `F110Env`; `reset`, `step`, map resolution, rendering hooks |
| Simulator internals | `gym/f110_gym/envs/` | See local `AGENTS.md`; numba-heavy dynamics/scan/collision code |
| Native rollout kernel | `gym/f110_gym/rollout_kernel/` | F110-specific C++ kernel for high-throughput planner rollouts; mirror Python simulator semantics |
| Realtime viewer | `gym/f110_gym/viewer.py`, `gym/f110_gym/envs/rendering.py` | `pyglet` is optional via `render` extra |
| Controllers | `controllers/controller_base.py`, `controllers/pure_pursuit.py` | Waypoint CSV format consumed by `PurePursuit.from_csv` |
| Waypoint demo | `runs/waypoint_drive.py` | Uses `outputs/waypoints/Spielberg_mintime.csv` and `maps/f1tenth_racetracks` submodule map |
| Raceline CLI | `scripts/optimize_mintime.py` | Public wrapper around nested legacy optimizer |
| Raceline internals | `scripts/raceline_opt/` | See local `AGENTS.md`; CasADi/IPOPT, helper imports, constant friction |
| Map generation | `scripts/generate_map.py` | Converts centerline CSV to YAML+PNG occupancy-grid assets |
| Tests | `tests/` | Gym API, dynamics, scan sim, collision checks |

## CODE MAP

| Symbol | Type | Location | Role |
|--------|------|----------|------|
| `F110Env` | class | `gym/f110_gym/envs/f110_env.py` | Main Gymnasium environment; registered as `f110-v0` |
| `Simulator` | class | `gym/f110_gym/envs/base_classes.py` | Multi-agent stepping, observations, collision checks |
| `RaceCar` | class | `gym/f110_gym/envs/base_classes.py` | Per-agent state, scan update, TTC, control integration |
| `Integrator` | enum-like class | `gym/f110_gym/envs/base_classes.py` | `Euler` / `RK4` selector |
| `ScanSimulator2D` | class | `gym/f110_gym/envs/laser_models.py` | Distance-transform scan model |
| `vehicle_dynamics_st` | numba function | `gym/f110_gym/envs/dynamic_models.py` | Single-track vehicle dynamics |
| `collision_multiple` | numba function | `gym/f110_gym/envs/collision_models.py` | Multi-agent collision checks |
| `F110Viewer` | class | `gym/f110_gym/viewer.py` | Realtime viewer facade over renderer |
| `Controller` | ABC | `controllers/controller_base.py` | Controller contract |
| `PurePursuit` | class | `controllers/pure_pursuit.py` | Reference waypoint follower |
| `main` | function | `scripts/optimize_mintime.py` | Minimum-time raceline command |

## CONVENTIONS

- Source layout is non-standard: `pyproject.toml` maps `gym/f110_gym` to `f110_gym`, plus top-level `controllers` and `utils` packages.
- Importing `f110_gym` registers `f110-v0`; tests and demos rely on that import side effect.
- `tests/conftest.py` inserts `gym/` into `sys.path` and reloads `f110_gym`; run pytest from repo root.
- `gym/f110_gym/rollout_kernel/` is not a second Gym implementation. It is a native F110 rollout kernel for planner/search throughput. When changing simulation behavior in `gym/f110_gym/envs/`, update the rollout kernel and its parity tests/docs in the same change.
- `scripts/optimize_mintime.py` inserts `scripts/raceline_opt` into `sys.path` for legacy nested imports.
- `pyrightconfig.json` adds `scripts/raceline_opt` to `extraPaths` and excludes `gym/f110_gym/envs/f110_env_backup.py`.
- Ruff has no custom config; defaults come from the pre-commit hook.
- Bundled package map data must stay under `gym/f110_gym/envs/maps/` to be included by setuptools package-data.
- **Assume-correct, don't guard.** Functions define a fixed input contract (shape, type, format) documented in their signatures and docstrings. They trust callers to honour it. Do **not** write `if x is not None` guards, `isinstance` checks, shape validation, or defensive `ValueError`s inside implementation code.

## ANTI-PATTERNS (THIS PROJECT)

- Do not copy from `gym/f110_gym/envs/f110_env_backup.py`; it is a legacy snapshot excluded from pyright.
- Do not commit generated outputs, waypoints, plots, numba cache, or backup/scratch files; `.gitignore` covers `outputs/`, `*.nbc`, `*.nbi`, `*_backup.py`.
- Do not assume `maps/f1tenth_racetracks/` is ordinary source; it is a git submodule (`git@github.com:f1tenth/f1tenth_racetracks`).
- Do not move raceline internals without updating `sys.path` insertion, pyright extra paths, and nested imports together.
- Do not document variable-friction raceline support unless code and assets are restored; current wrapper forces constant friction.
- Do not put F110-specific dynamics, scan, map, or collision code under generic planner/search packages. Keep it in `gym/f110_gym/rollout_kernel/` and compose it from planner adapters.

## COMMANDS

```bash
pip install -e .
pip install -e ".[dev,render]"
pip install -e ".[tools]"
pytest -q
pre-commit run --files <changed-file>...
python scripts/optimize_mintime.py --track maps/f1tenth_racetracks/Spielberg/Spielberg_centerline.csv --output outputs/waypoints/Spielberg_mintime.csv --save_plot
python scripts/generate_map.py maps/f1tenth_racetracks/Spielberg/Spielberg_centerline.csv -o outputs/maps/Spielberg -r 1.0
```

## NOTES

- Performance-critical models use `@njit(cache=True)`. If numba-backed behavior seems stale after edits, clear `__pycache__/` plus `.nbc`/`.nbi` artifacts.
- Raceline defaults live in `configs/raceline/f110.ini`; CLI flags override step sizes, `width_opt`, `step_non_reg`, IPOPT iterations, and tolerance.
- Finer raceline step sizes can trigger `prep_track()` normal-crossing failures before IPOPT starts; increasing `reg_smooth_opts.s_reg` or coarsening spacing changes geometry.
- `tests/f110_gym/legacy_scan.npz` is an intentional binary fixture for scan regression checks.
- See `docs/rollout_kernel.md` before editing native rollout code or changing simulator semantics.
