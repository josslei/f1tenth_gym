# f1tenth_gym_jl

Gymnasium-compatible F1TENTH environment.

## Install

```bash
pip install -e .
pip install -e ".[render]"
pip install -e ".[tools]"
pip install -e ".[dev,render]"
```

Use the core install for the simulation package only. Add `render` if you want
the pyglet/OpenGL renderer, and add `dev` if you want the test tooling.
Add `tools` if you want the plotting dependencies used by the race-line and
legacy helper scripts.

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
tests under `tests/`, while the manual helper programs live in
[`scripts/f110_gym/`](/Users/josslei/projects/f1tenth_gym_jl/scripts/f110_gym).
That keeps `pytest -q` focused on the real automated tests instead of the
standalone demo and data-generation scripts.

If you want to run those helpers directly, execute the files in
`scripts/f110_gym/` as scripts and pass the required arguments.

## Race Line Generation

You can generate a minimum-curvature racing line from one of the bundled maps
with:

```bash
python3 scripts/generate_race_line.py \
  --map_path maps/berlin.yaml \
  --output /tmp/berlin_race_line.csv
```

The output is a CSV with `x,y,yaw,speed` columns in world coordinates. The
script extracts a centerline, measures left/right track width at every
centerline point, optimizes lateral offsets with L-BFGS-B, then computes a
curvature and acceleration-limited speed profile. The optimizer uses
interpolated control offsets by default; pass `--control_stride 1` to optimize
every waypoint, or use a larger stride for faster generation. Add `--timing`
to print elapsed time for each major processing step, and add `--visualize` to
display the map, centerline, and generated line.

The script uses the image path stored in the YAML by default; pass `--map_ext`
if you want to override the image extension explicitly. If required arguments
are missing or invalid, the script prints usage help before exiting.

## Notes

- The environment now targets Gymnasium's reset/step API.
- Python 3.11 is the supported floor for this migration.
- `pyglet` is optional and only needed for rendering or the legacy pyglet test
  scripts.
