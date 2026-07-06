"""Python Gym binding for the C++ LMPC controller.

This module intentionally contains no MPC implementation. It loads the compiled
C++ extension used by Gym-facing tests and runners.
"""

from __future__ import annotations

import os
from pathlib import Path

_package_path = Path(__file__).resolve().parent
_plugin_paths = [
    str(_package_path),
    str(_package_path / "build" / "casadi" / "lib"),
]
_casadi_path = os.environ.get("CASADIPATH")
os.environ["CASADIPATH"] = (
    os.pathsep.join(_plugin_paths)
    if not _casadi_path
    else os.pathsep.join([*_plugin_paths, _casadi_path])
)

from . import lmpc_native as _native  # noqa: E402

PAPER_STATE_COLUMNS = _native.PAPER_STATE_COLUMNS
RACING_STATE_COLUMNS = _native.RACING_STATE_COLUMNS
CenterlineTrack = _native.CenterlineTrack
FrenetProjection = _native.FrenetProjection
GymVehicleState = _native.GymVehicleState
PaperLmpcState = _native.PaperLmpcState
RacingLmpcState = _native.RacingLmpcState
normalize_angle = _native.normalize_angle
racing_to_paper = _native.racing_to_paper
LmpcConfig = getattr(_native, "LmpcConfig", None)
LmpcControlCommand = getattr(_native, "LmpcControlCommand", None)
LmpcReference = getattr(_native, "LmpcReference", None)
NativeLMPCController = getattr(_native, "NativeLMPCController", None)
SparseErrorModel = getattr(_native, "SparseErrorModel", None)
