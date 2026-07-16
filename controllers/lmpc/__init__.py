from typing import Any

__all__ = ["LMPCController"]


def __getattr__(name: str) -> Any:
    if name == "LMPCController":
        from .lmpc import LMPCController

        return LMPCController
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
