from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_muzero_config(
    path: str | Path = "configs/muzero/default.yaml",
) -> dict[str, Any]:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


__all__ = ["load_muzero_config"]
