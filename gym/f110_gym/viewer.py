"""Realtime viewer API for F110 Gym simulations."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Protocol


Observation = dict[str, Any]
RenderCallback = Callable[[Any], None]


class Viewer(Protocol):
    """Common interface for realtime simulation viewers."""

    def update(self, obs: Observation) -> None:
        """Update the viewer with the latest simulator observation."""

    def render(self) -> None:
        """Draw the latest observation."""

    def close(self) -> None:
        """Release viewer resources."""


@dataclass(frozen=True)
class ViewerConfig:
    """Configuration for a realtime F110 viewer.

    Args:
        map_path: Map path without image extension, or a YAML path.
        map_ext: Map image extension used by the renderer.
        width: Viewer window width in pixels.
        height: Viewer window height in pixels.
        target_fps: Maximum draw rate. Use ``None`` to render as fast as possible.
    """

    map_path: str | Path
    map_ext: str = ".png"
    width: int = 1000
    height: int = 800
    target_fps: float | None = 60.0


class F110Viewer:
    """Pyglet-backed realtime viewer for F110 Gym observations."""

    def __init__(
        self,
        config: ViewerConfig,
        *,
        callbacks: list[RenderCallback] | None = None,
    ) -> None:
        """Create a realtime viewer.

        Args:
            config: Viewer configuration.
            callbacks: Optional drawing callbacks that receive the low-level
                renderer after the observation has been applied.
        """
        self.config = config
        self.callbacks = list(callbacks or [])
        self._renderer: Any | None = None
        self._latest_obs: Observation | None = None
        self._last_render_time = 0.0
        self._closed = False

    @classmethod
    def from_env(
        cls,
        env: Any,
        *,
        width: int = 1000,
        height: int = 800,
        target_fps: float | None = 60.0,
        callbacks: list[RenderCallback] | None = None,
    ) -> F110Viewer:
        """Create a viewer configured from an ``F110Env`` instance.

        Args:
            env: Environment exposing ``map_stem`` and ``map_ext`` attributes.
            width: Viewer window width in pixels.
            height: Viewer window height in pixels.
            target_fps: Maximum draw rate. Use ``None`` to render as fast as
                possible.
            callbacks: Optional drawing callbacks.

        Returns:
            F110Viewer: Viewer configured for the environment's current map.
        """
        config = ViewerConfig(
            map_path=env.map_stem,
            map_ext=env.map_ext,
            width=width,
            height=height,
            target_fps=target_fps,
        )
        return cls(config, callbacks=callbacks)

    def update(self, obs: Observation) -> None:
        """Update the viewer with the latest simulator observation.

        Args:
            obs: Observation containing pose and lap fields from ``F110Env``.
        """
        self._latest_obs = {
            "ego_idx": obs["ego_idx"],
            "poses_x": obs["poses_x"],
            "poses_y": obs["poses_y"],
            "poses_theta": obs["poses_theta"],
            "lap_times": obs["lap_times"],
            "lap_counts": obs["lap_counts"],
        }

    @property
    def closed(self) -> bool:
        """True after the window has been closed."""
        return self._closed

    def render(self) -> None:
        """Draw the latest observation."""
        if self._closed:
            return
        if self._latest_obs is None:
            raise RuntimeError("F110Viewer.update() must be called before render().")

        renderer = self._ensure_renderer()
        renderer.update_obs(self._latest_obs)

        for callback in self.callbacks:
            callback(renderer)

        renderer.dispatch_events()
        if renderer.window_closed:
            self._renderer = None
            self._closed = True
            return

        renderer.on_draw()
        renderer.flip()
        self._throttle()

    def close(self) -> None:
        """Release the pyglet window if it has been created."""
        if self._renderer is None:
            return
        self._renderer.close()
        self._renderer = None
        self._closed = True

    def _ensure_renderer(self) -> Any:
        if self._renderer is not None:
            return self._renderer

        from f110_gym.envs.rendering import EnvRenderer

        renderer = EnvRenderer(self.config.width, self.config.height)
        map_stem = str(Path(self.config.map_path).with_suffix(""))
        renderer.update_map(map_stem, self.config.map_ext)
        self._renderer = renderer
        return renderer

    def _throttle(self) -> None:
        if self.config.target_fps is None:
            return

        min_period = 1.0 / self.config.target_fps
        now = time.perf_counter()
        remaining = min_period - (now - self._last_render_time)
        if remaining > 0.0:
            time.sleep(remaining)
        self._last_render_time = time.perf_counter()
