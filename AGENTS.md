# AGENTS.md

## Setup

```bash
pip install -e ".[dev,render]"   # full dev install with renderer
pip install -e .                 # core only
```

Requires Python 3.11+.

## Lint / TypeCheck / Test

Coding style is enforced by pre-commit hooks defined in `.pre-commit-config.yaml`:
ruff (default rules), ruff-format, pyright, and basic checks (trailing-whitespace,
end-of-file-fixer, check-yaml, check-merge-conflict). No custom `[tool.ruff]`
section — ruff uses its defaults.

After file-editing tool use, `.codex/hooks/post_edit_checks.py` runs pre-commit
on only the changed files. To verify manually, run pre-commit on just the files
you touched:

```bash
pre-commit run --files <changed-file>...
```

To run the tests:

```bash
pytest -q
```

## Architecture

Single-package repo. Source lives in non-standard locations:

| Source dir          | Installed as    | Key contents                          |
|---------------------|-----------------|---------------------------------------|
| `gym/f110_gym/`     | `f110_gym`      | `F110Env`, viewer, env models, maps   |
| `controllers/`      | `controllers`   | `Controller` ABC, `PurePursuit`       |
| `utils/`            | `utils`         | utility helpers                       |

`gym/f110_gym/__init__.py` registers `f110-v0` with Gymnasium on import.

## Key files

- `gym/f110_gym/envs/f110_env.py` — `F110Env(gym.Env)`, the main environment
- `gym/f110_gym/envs/base_classes.py` — `Simulator`, `Integrator` (Euler/RK4)
- `gym/f110_gym/envs/dynamic_models.py` — single-track vehicle dynamics (numba `@njit`)
- `gym/f110_gym/envs/collision_models.py` — GJK collision detection (numba)
- `gym/f110_gym/envs/laser_models.py` — 2D LIDAR simulation (numba)
- `gym/f110_gym/viewer.py` — `F110Viewer`, `ViewerConfig` for realtime rendering
- `controllers/controller_base.py` — `VehicleState`, `ControlCommand`, `Controller` ABC
- `controllers/pure_pursuit.py` — reference pure pursuit controller

## Excluded from type checking

- `gym/f110_gym/envs/f110_env_backup.py` — legacy env snapshot

Listed in `pyrightconfig.json` excludes.

## Testing notes

- `tests/conftest.py` injects `gym/` into `sys.path` so `f110_gym` import works; run tests from repo root
- `test_gymnasium_api.py` tests the Gymnasium registration, `reset`, `step`, and `make_viewer`
- `test_dynamics.py`, `test_scan_sim.py`, `test_collision_checks.py` are `unittest.TestCase`-style tests that pytest collects automatically
- `tests/f110_gym/legacy_scan.npz` is a binary test fixture for scan simulator benchmarks

## Maps

Maps live in `gym/f110_gym/envs/maps/`. Each map is a YAML file with metadata plus a PNG or PGM image. Bundled maps: `berlin`, `levine`, `skirk`, `stata_basement`, `vegas`.

## Numba

Performance-critical models use `@njit(cache=True)`. If you change these functions, delete `__pycache__/` and any `.nbc`/`.nbi` files so numba re-JITs.

## Race line scripts

- `scripts/optimize_raceline.py` — main CLI for minimum-curvature raceline generation. Requires `pip install -e ".[tools]"` (matplotlib, casadi, rich, trajectory_planning_helpers).
- `scripts/raceline_opt/` — legacy raceline optimization pipeline (global traj optimization, friction mapping, helper functions).

## Outputs

- `outputs/` is gitignored; use it for generated race lines, plots, etc.
