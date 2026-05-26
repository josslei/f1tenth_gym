# f1tenth_gym_jl

Gymnasium-compatible F1TENTH environment.

## Install

```bash
pip install -e .
pip install -e ".[render]"
pip install -e ".[dev,render]"
```

Use the core install for the simulation package only. Add `render` if you want
the pyglet/OpenGL renderer, and add `dev` if you want the test tooling.

## Use

```python
import gymnasium as gym
import f110_gym  # registers f110-v0

env = gym.make("f110-v0")
obs, info = env.reset(options={"poses": poses})
obs, reward, terminated, truncated, info = env.step(action)
```

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

## Notes

- The environment now targets Gymnasium's reset/step API.
- Python 3.11 is the supported floor for this migration.
- `pyglet` is optional and only needed for rendering or the legacy pyglet test
  scripts.
