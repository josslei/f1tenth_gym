"""Gym-facing LMPC controller bindings.

The LMPC implementation is C++. Python code in this package should remain a
thin Gym API adapter around the compiled controller/binding.
"""

from .controller import LMPCController

__all__ = ["LMPCController"]
