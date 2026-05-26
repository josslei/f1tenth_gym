"""Pytest configuration for the F110 Gym test suite."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = ROOT / "gym"

if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

sys.modules.pop("f110_gym", None)
sys.modules.pop("f110_gym.envs", None)

import f110_gym  # noqa: F401
