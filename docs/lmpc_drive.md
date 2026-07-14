# Running `runs/lmpc_drive.py`

`runs/lmpc_drive.py` drives the native LMPC controller (`controllers/lmpc/`)
on the `f110_gym_10` map in the pyglet viewer, using the paper's
lap-as-iteration scheme: each completed lap is fed back into the safe set,
so later iterations have more demonstrated data to draw on. Design
background lives in `controllers/lmpc/DESIGN.md`; this doc only covers how
to get it running.

## 1. Prerequisites

Install the render extra (the viewer needs pyglet) and dev tooling:

```bash
pip install -e ".[dev,render]"
```

### Build the native controller

`controllers/lmpc/lmpc.py` imports a compiled `lmpc_native` pybind11
module. Build it once, and rebuild after any change under
`controllers/lmpc/{include,src}/`:

```bash
cmake -S controllers/lmpc -B controllers/lmpc/build
cmake --build controllers/lmpc/build -j
```

This also builds CasADi's `qrqp` and `ipopt` QP/NLP solver plugins from
source (`controllers/lmpc/CMakeLists.txt`'s `WITH_IPOPT`/`WITH_BUILD_IPOPT`
superbuild) — the first build takes a while because it compiles CasADi,
MUMPS, and IPOPT; later builds are incremental.

### Generate the D^0 seed lap

The controller needs a recorded "seed lap" (D^0: a closed-loop trajectory
from a simple low-speed centerline-tracking controller) before it has
anything to warm-start or query against. It's gitignored under `outputs/`,
so generate it locally:

```bash
python scripts/lmpc_collect_seed_lap.py          # headless
python scripts/lmpc_collect_seed_lap.py --visualize  # watch it drive
```

This writes `outputs/lmpc_seed_laps/f110_gym_10_seed_lap.csv` by default —
the same path `runs/lmpc_drive.py`'s `SEED_LAP_CSV` constant expects. Only
needs to be regenerated when the map, `SIM_TIMESTEP`, or the seed-lap
driving policy changes (its own file header explains why Pure Pursuit, not
Stanley, is used to record it).

## 2. Run it

From the repo root:

```bash
python runs/lmpc_drive.py
```

A pyglet window opens showing:

- the reference centerline (`WaypointOverlay`),
- the car's actual driven path so far (`DrivenLineOverlay`),
- the solver's receding-horizon prediction (`RecedingHorizonOverlay`).

The console prints one line per completed lap (iteration), e.g.:

```text
iteration 0: lap completed in 27.77s (transitions=1111, states=1112)
iteration 1: lap completed in 22.70s (transitions=908, states=909)
```

Only iteration 0 launches from a standing start; the simulator and
controller are reset exactly once for the whole run, so every later
iteration drives straight through the finish line at whatever speed it
crossed with — no per-lap restart.

The run stops after `MAX_ITERATIONS` laps, when the vehicle crashes (a
crashed/truncated lap is never added to the safe set and ends the run), or
when the viewer window is closed.

## 3. Config knobs

All tunables are module-level constants near the top of
`runs/lmpc_drive.py` — edit and rerun, no CLI flags:

| Constant | Meaning |
| --- | --- |
| `MAP`, `CENTERLINE_CSV` | Track to drive. Must match what the seed lap in `SEED_LAP_CSV` was recorded against. |
| `SEED_LAP_CSV` | D^0 path from step 1. |
| `HORIZON_STEPS` | FHOCP prediction horizon `N`. |
| `SIM_TIMESTEP` | Must match the seed lap's own `dt` (`scripts/lmpc_collect_seed_lap.py`'s `SIM_TIMESTEP`). |
| `MAX_ITERATIONS` | Laps to drive before stopping. |
| `CONFIG_OVERRIDES` | Any `LmpcConfig` field by name (cost weights, `solver_name`, `ey_max`, ...) — see `controllers/lmpc/include/lmpc_config.hpp` for what each one scales. |
| `FALLBACK_BRAKE_DELTA_V`, `MAX_CONSECUTIVE_FALLBACK_STEPS` | Controlled-brake fallback when a QP solve fails; abandons the iteration if the solver doesn't recover within this many consecutive steps. |
| `LOW_SPEED_STEER_ZERO_BELOW`, `LOW_SPEED_STEER_RESTORE_AT` | Actuator-level guard that suppresses steering below this speed band — works around a real gym low-speed dynamics divergence, not a controller bug. |

## 4. Switching tracks

There is no `--map` flag; a track is a matched trio of files that all have
to agree, so switching requires editing three places:

1. `scripts/lmpc_collect_seed_lap.py`: `WAYPOINTS_CSV` and `OUTPUT_CSV`.
2. `runs/lmpc_drive.py`: `MAP` and `CENTERLINE_CSV`.
3. Regenerate the seed lap for the new track (step 1 above) before running.

Also check `LmpcConfig::ey_max` (`controllers/lmpc/include/lmpc_config.hpp`)
— its default is sized for `f110_gym_10`'s ~1.5m centerline half-width, and
will be wrong (too permissive or infeasible) on a track with a different
width; override it via `CONFIG_OVERRIDES` if needed.

## 5. Debugging a stalled/misbehaving solve

Two env vars, read directly by the native controller, print per-solve
diagnostics to stderr:

```bash
LMPC_DEBUG_STAGES=1 python runs/lmpc_drive.py     # per-stage x_ref/u_ref/|A|/|B| before each solve
LMPC_DEBUG_TERMINAL=1 python runs/lmpc_drive.py   # x0, terminal safe-set query, lambda, slack per solve
```

## 6. Known limitations (not bugs to "fix" by tuning this script)

- **§5/§6 error-dynamics regression is unimplemented** (`DESIGN.md`): the
  nominal model overestimates cornering grip above the demonstrated
  speeds, so aggressive `cost_to_go_weight` settings can produce
  sprint-then-brake plans that fail a solve mid-corner. The fallback brake
  usually recovers it; a persistent failure there is this known issue, not
  a regression in the driving loop.
- **D^0 stays a standing-start recording.** Iteration 0 launches from rest,
  but every iteration after that begins flying (at whatever speed it
  crossed the finish line with) — the safe set's only near-`s=0` data is
  still D^0's low-speed launch samples. This can make the terminal
  safe-set query and hard terminal equality pull toward stale slow data
  right at the start of iterations 1+.
