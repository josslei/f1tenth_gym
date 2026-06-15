# f1tenth_gym_jl

Gymnasium-compatible F1TENTH environment.

## Install

```bash
pip install -e .
pip install -e ".[render]"
pip install -e ".[tools]"
pip install -e ".[dev]"
```

Use the core install for the simulation package only. Add `render` if you want
the pyglet/OpenGL renderer, and add `dev` if you want the test tooling.
Add `tools` if you want the plotting dependencies used by the race-line and
map-generation scripts.

Combine extras when needed, for example `pip install -e ".[dev,render]"` if you
want both the test tooling and the renderer in one install.

## Use

```python
import gymnasium as gym
import f110_gym  # registers f110-v0

env = gym.make("f110-v0")
obs, info = env.reset(options={"poses": poses})
obs, reward, terminated, truncated, info = env.step(action)
```

## Realtime Visualization

See [docs/realtime_visualization.md](/Users/josslei/projects/f1tenth_gym_jl/docs/realtime_visualization.md)
for the viewer API and usage examples.

## Testing

Run the pytest-native checks with:

```bash
pytest -q
```

The pytest suite exercises `F110Env` directly and ignores the old helper
tests under `tests/`.

## PPO Controller Experiment

Install the optional RL and dev dependencies with:

```bash
pip install -e ".[rl,dev]"
```

Train a Lightning PPO policy with:

```bash
python runs/train_ppo_controller.py --config configs/ppo/default.yaml
```

The config owns the PPO iteration count, rollout length, seed, output path,
observation settings, action bounds, policy selection, reward parameters,
single training map, initial pose, and PPO hyperparameters.
The default config trains on ``f110_gym_10`` and writes to ``outputs/rl/ppo_f110_gym_10/``.

Per-update episode returns are appended to
``outputs/rl/ppo_f110_gym_10/metrics.jsonl``.
TensorBoard event files are written under
``outputs/rl/ppo_f110_gym_10/tensorboard/`` and can be monitored with:

.. code:: bash

   tensorboard --logdir outputs/rl/ppo_f110_gym_10/tensorboard

Generated PPO outputs live under ignored `outputs/rl/` paths and should not be
committed.

### Temporal Context / Memory — TODO

The current PPO policy observes only one simulator frame at a time: a
downsampled LiDAR scan plus optional ego-state features. There is no temporal
context, so the policy cannot directly infer momentum, whether it is stuck, or
how the scene is changing while entering and exiting corners.

Future PPO convergence work should add one of these memory mechanisms:

1. **Frame stacking** — concatenate the last `N` observations along the feature
   axis, add an `observation.frame_stack` config field, and update
   `observation_dim()` plus the rollout observation path to maintain per-env
   history buffers across resets.
2. **Recurrent policy** — add an LSTM/GRU policy variant, extend the `Policy`
   interface to accept and return hidden state, and teach `RolloutDataset` to
   carry/reset hidden states on episode boundaries.

Do not treat model-size increases as a substitute for this; a larger feedforward
MLP still receives only a single-frame observation.

## Race Line Optimization

Generate a minimum-time optimized raceline from a track CSV with centerline
and width columns:

```bash
python scripts/optimize_mintime.py \
  --track maps/f1tenth_racetracks/Spielberg/Spielberg_centerline.csv \
  --output outputs/waypoints/Spielberg_mintime.csv \
  --save_plot
```

Use `--stepsize_reg` and related flags to trade runtime for line quality.
Smaller values create more optimization nodes and consume exponentially more
memory. Useful presets are:

```bash
# Fastest, coarsest line
--stepsize_prep 1.0 --stepsize_reg 5.0 --stepsize_interp_after_opt 1.0 --step_non_reg 10

# Current defaults, finest quality (no flags needed)
# --stepsize_prep 0.1 --stepsize_reg 0.3 --stepsize_interp_after_opt 0.2 --step_non_reg 0
```

Use `--width_opt` to control boundary clearance. It is the effective vehicle
width used by the optimizer, so larger values keep the optimized car center
farther from the track edge. The default config uses `0.9` m.

The script runs minimum-time optimization via CasADi + IPOPT. The output is
a semicolon-delimited CSV in the standard raceline format:

```text
# s_m; x_m; y_m; psi_rad; kappa_radpm; vx_mps; ax_mps2
```

The input track is expected to provide the centerline and left/right widths;
this script does not extract a centerline from an occupancy map. Place external
track CSVs under `maps/`. `--params`
selects the vehicle and optimization config; the default is
`configs/raceline/f110.ini`. Pass `--save_plot` to save a PNG next to the
output CSV showing the raceline overlaid on the track boundaries and centerline.

### Generating a Compatible Map

If a track does not ship with its own `map.yaml` and `map.png`, create them
from the centerline CSV:

```bash
python scripts/generate_map.py \
  maps/f1tenth_racetracks/Spielberg/Spielberg_centerline.csv \
  -o outputs/maps/Spielberg \
  -r 1.0
```

- `-o` sets the output prefix (produces `<prefix>.yaml` and `<prefix>.png`).
- `-r` controls the resolution in meters per pixel (default 0.05; use a larger
  value for big tracks to keep image dimensions reasonable).

Pass the resulting YAML path (or the track's own map YAML) to the waypoint
drive script, e.g. ``MAP = "maps/custom/f110_gym_10/f110_gym_map"``.

### Generating a Centerline From an Existing Map

If you already have an occupancy-grid map and want to derive a standard
centerline CSV from it, use:

```bash
python scripts/generate_centerline.py \
  --map maps/f1tenth_maps/maps/f1_aut.yaml \
  --output outputs/centerlines/f1tenth_maps/f1_aut_centerline.csv \
  --save_plot
```

This script estimates the loop center and track width from the white track
pixels in the map image, then writes a 4-column CSV in the format used by the
reward and waypoint loaders:

```text
# x_m, y_m, w_tr_right_m, w_tr_left_m
```

Use `--num-points` to control the density of the generated loop and
`--white-threshold` if your map image needs a different intensity cutoff.
With `--save_plot`, the script also writes a PNG visualization next to the
CSV output.

## Notes

- The environment now targets Gymnasium's reset/step API.
- Python 3.11 is the supported floor for this migration.
- `pyglet` is optional and only needed for rendering or the legacy pyglet test
  scripts.
