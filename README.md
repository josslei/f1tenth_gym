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

## Race Line Optimization

Generate a minimum-time optimized raceline from a track CSV with centerline
and width columns:

```bash
python scripts/optimize_mintime.py \
  --track scripts/raceline_opt/inputs/tracks/berlin_2018.csv \
  --output outputs/waypoints/berlin_mintime.csv \
  --save_plot
```

Use `--stepsize_reg` to trade runtime for line quality. Smaller values create
more mintime optimization nodes and consume exponentially more memory. The
default config uses `1.0` m. Useful overrides are:

```bash
# Very fast, rough line
--stepsize_reg 10.0

# Moderate quality, balanced speed
--stepsize_reg 5.0

# Default / higher quality
--stepsize_reg 1.0
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
this script does not extract a centerline from an occupancy map. `--params`
selects the vehicle and optimization config; the default is
`configs/raceline/f110.ini`. Pass `--save_plot` to save a PNG next to the
output CSV showing the raceline overlaid on the track boundaries and centerline.

## Notes

- The environment now targets Gymnasium's reset/step API.
- Python 3.11 is the supported floor for this migration.
- `pyglet` is optional and only needed for rendering or the legacy pyglet test
  scripts.
