"""Track discovery, train/test split, curriculum pool, and start-pose extraction.

Used by the multi-track PPO training pipeline to manage a growing pool of
tracks that all parallel envs sample from independently.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

TRACKS_ROOT = Path(__file__).resolve().parents[1] / "tracks"
DEFAULT_MAP_EXT = ".png"


def _normalize(name: str) -> str:
    return name.replace(" ", "")


def _discover_track_files(root: Path) -> dict[str, dict[str, Path]]:
    """Return ``{normalized_name: {"csv": Path, "map_stem": Path}}``.

    Discovers actual files on disk, so it handles irregular names like
    ``Mexico City / MexicoCity_*`` transparently.
    """
    result: dict[str, dict[str, Path]] = {}
    for csv_path in sorted(root.glob("*/*_centerline.csv")):
        name = _normalize(csv_path.parent.name)
        map_yamls = list(csv_path.parent.glob("*_map.yaml"))
        if not map_yamls:
            continue
        map_stem = map_yamls[0].with_suffix("")
        result[name] = {"csv": csv_path, "map_stem": map_stem}
    return result


def _heading_from_csv(csv_path: Path) -> list[float]:
    pts = np.loadtxt(str(csv_path), delimiter=",", skiprows=1, max_rows=2)
    x = float(pts[0, 0])
    y = float(pts[0, 1])
    dx = float(pts[1, 0] - pts[0, 0])
    dy = float(pts[1, 1] - pts[0, 1])
    return [x, y, float(np.arctan2(dy, dx))]


class TrackScheduler:
    """Manages the set of available tracks, train/test split, and curriculum.

    Parameters
    ----------
    root:
        Path to the ``tracks/`` directory.
    holdout:
        Track name(s) that are **never** used for training (e.g. test-only).
    test_ratio:
        Fraction of the *remaining* tracks (after removing *holdout*) to set
        aside as a held-out test set.  The test set is fixed for the run.
    seed:
        RNG seed for reproducibility of the split.
    initial:
        Number of training tracks in the pool at iteration 0.
    increment:
        Number of new training tracks to add at each milestone.
    interval_frac:
        Fraction of total iterations between milestones
        (e.g. 0.1 -> every 10 %).
    """

    def __init__(
        self,
        root: str | Path = TRACKS_ROOT,
        holdout: list[str] | None = None,
        test_ratio: float = 0.2,
        seed: int = 42,
        initial: int = 4,
        increment: int = 4,
        interval_frac: float = 0.1,
    ) -> None:
        self._root = Path(root)
        self._files = _discover_track_files(self._root)
        all_tracks = sorted(self._files)
        known = set(all_tracks)

        holdout = holdout or []
        for name in holdout:
            if name not in known:
                import warnings

                warnings.warn(f"Holdout track '{name}' not found under {root}")

        pool = [t for t in all_tracks if t not in holdout]
        rng = np.random.default_rng(seed)
        rng.shuffle(pool)

        split_idx = max(1, round(len(pool) * (1.0 - test_ratio)))
        self._train_tracks = sorted(pool[:split_idx])
        self._test_tracks = sorted(pool[split_idx:])
        self._holdout = list(holdout)
        self._initial = int(initial)
        self._increment = int(increment)
        self._interval_frac = float(interval_frac)

        self._current_iteration = -1

        # Pre-compute start poses for all discovered tracks
        self._start_poses: dict[str, list[float]] = {}
        for name in all_tracks:
            csv = self._files[name]["csv"]
            if csv.exists():
                self._start_poses[name] = _heading_from_csv(csv)
            else:
                self._start_poses[name] = [0.0, 0.0, 0.0]

    # -- Iteration tracking (advanced externally by RolloutDataset) -------

    def step_iteration(self) -> None:
        self._current_iteration += 1

    @property
    def current_iteration(self) -> int:
        return max(0, self._current_iteration)

    # -- Public properties ------------------------------------------------

    @property
    def train_tracks(self) -> list[str]:
        return list(self._train_tracks)

    @property
    def test_tracks(self) -> list[str]:
        return list(self._test_tracks)

    @property
    def holdout_tracks(self) -> list[str]:
        return list(self._holdout)

    # -- File path resolution ---------------------------------------------

    def map_stem(self, track: str) -> str:
        """Filesystem stem (without extension) suitable for ``update_map()``."""
        return str(self._files[track]["map_stem"])

    def map_yaml(self, track: str) -> str:
        """Absolute ``.yaml`` path suitable for ``gym.make(map=...)``."""
        return str(self._files[track]["map_stem"].with_suffix(".yaml"))

    # -- Curriculum -------------------------------------------------------

    def current_pool(self, iteration: int, total_iterations: int) -> list[str]:
        """Subset of train tracks available at a given PPO iteration.

        The pool grows linearly from ``initial`` to ``len(train_tracks)``
        over milestones spaced ``interval_frac`` apart.  All training tracks
        are guaranteed to be in the pool by ``total_iterations / 2``.
        """
        num_train = len(self._train_tracks)
        if num_train == 0:
            return []

        target = self._initial
        if total_iterations > 0 and iteration > 0:
            step = int(iteration / (total_iterations * self._interval_frac))
            target = self._initial + step * self._increment

        pool_size = min(num_train, max(self._initial, target))
        return self._train_tracks[:pool_size]

    @staticmethod
    def sample(pool: list[str]) -> str:
        return str(np.random.choice(pool))

    # -- Start poses ------------------------------------------------------

    def start_pose(self, track: str) -> np.ndarray:
        return np.array([self._start_poses[track]], dtype=np.float64)

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}("
            f"train={len(self._train_tracks)}, "
            f"test={len(self._test_tracks)}, "
            f"holdout={len(self._holdout)})"
        )


def make_track_reset_fn(
    scheduler: TrackScheduler,
    total_iterations: int,
    map_ext: str = DEFAULT_MAP_EXT,
) -> Any:
    """Build a callable suitable as ``rollout(…)``'s ``reset_fn``.

    The returned function accepts a ``gym.Env`` as its only argument, reads
    the current iteration from *scheduler* to determine the curriculum pool,
    samples a track, calls ``env.unwrapped.update_map()``, and returns the
    ``options`` dict for ``env.reset(options=…)``.
    """

    def _reset(env: Any) -> dict[str, Any]:
        pool = scheduler.current_pool(scheduler.current_iteration, total_iterations)
        track = scheduler.sample(pool)
        pose = scheduler.start_pose(track)
        env.unwrapped.update_map(scheduler.map_yaml(track), map_ext)
        return {"poses": pose}

    return _reset


__all__ = [
    "TRACKS_ROOT",
    "DEFAULT_MAP_EXT",
    "TrackScheduler",
    "make_track_reset_fn",
]
