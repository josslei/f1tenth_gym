"""Run repo style checks after a file-editing tool use."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]


def _read_hook_payload() -> dict[str, Any] | None:
    """Read the hook payload from stdin if Codex provides one."""

    if sys.stdin.isatty():
        return None
    raw_input = sys.stdin.read().strip()
    if not raw_input:
        return None
    try:
        payload = json.loads(raw_input)
    except json.JSONDecodeError:
        return None
    if isinstance(payload, dict):
        return payload
    return None


def _extract_candidate_files(value: Any) -> set[str]:
    """Extract likely repo-relative file paths from a hook payload."""

    candidates: set[str] = set()
    if isinstance(value, dict):
        for key, nested_value in value.items():
            if key in {"path", "file", "filename"} and isinstance(nested_value, str):
                candidates.add(nested_value)
            elif key in {"paths", "files", "filePaths", "changedFiles"}:
                candidates.update(_extract_candidate_files(nested_value))
            elif key == "changes" and isinstance(nested_value, list):
                for change in nested_value:
                    candidates.update(_extract_candidate_files(change))
            else:
                candidates.update(_extract_candidate_files(nested_value))
    elif isinstance(value, list):
        for item in value:
            candidates.update(_extract_candidate_files(item))
    elif isinstance(value, str):
        candidates.add(value)
    return candidates


def _filter_repo_files(candidates: set[str]) -> list[str]:
    """Keep only changed files that live in the repository and are checkable."""

    files = {
        path
        for path in candidates
        if path
        and not path.startswith(".codex/")
        and not path.startswith("/")
        and not path.endswith("/")
        and (REPO_ROOT / path).exists()
    }
    return sorted(files)


def _changed_files() -> list[str]:
    """Return files touched by the current tool invocation."""

    payload = _read_hook_payload()
    if payload is not None:
        files = _filter_repo_files(_extract_candidate_files(payload))
        return files
    return []


def _run_pre_commit(files: list[str]) -> None:
    """Run the repo's pre-commit stack against the provided files."""

    completed = subprocess.run(
        [sys.executable, "-m", "pre_commit", "run", "--files", *files],
        cwd=REPO_ROOT,
        check=False,
    )
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


def main() -> int:
    """Entry point for the post-edit hook."""

    files = _changed_files()
    if not files:
        return 0

    _run_pre_commit(files)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
