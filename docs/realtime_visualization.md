# Realtime Visualization

This project exposes a small viewer API for drawing simulation observations
without making the viewer own controller or simulator state.

## Install

Install the optional renderer dependencies first:

```bash
pip install -e ".[render]"
```

## Explicit Viewer Loop

Create a viewer from an environment and update it with each observation:

```python
import gymnasium as gym
import numpy as np
import f110_gym  # registers f110-v0

env = gym.make("f110-v0")
f110_env = env.unwrapped
viewer = f110_env.make_viewer(target_fps=60.0)

obs, info = f110_env.reset(options={"poses": poses})
try:
    while True:
        action = np.zeros((f110_env.num_agents, 2), dtype=np.float64)
        obs, reward, terminated, truncated, info = f110_env.step(action)
        viewer.update(obs)
        viewer.render()
        if terminated or truncated:
            break
finally:
    viewer.close()
    env.close()
```

Use `target_fps=None` to draw as fast as possible:

```python
viewer = f110_env.make_viewer(target_fps=None)
```

Use `width` and `height` to choose a smaller window:

```python
viewer = f110_env.make_viewer(width=640, height=480, target_fps=60.0)
```

## Gymnasium Render Path

You can also use the Gymnasium-style render path:

```python
env = gym.make("f110-v0", render_mode="human")
f110_env = env.unwrapped

obs, info = f110_env.reset(options={"poses": poses})
while True:
    action = np.zeros((f110_env.num_agents, 2), dtype=np.float64)
    obs, reward, terminated, truncated, info = f110_env.step(action)
    f110_env.render()
    if terminated or truncated:
        break

env.close()
```

Use `render_mode="human_fast"` to render without FPS throttling.

## Notes

- The viewer consumes observations only.
- The viewer uses the existing pyglet/OpenGL `EnvRenderer` internally.
- Initial poses must be inside the selected map's drivable area.
