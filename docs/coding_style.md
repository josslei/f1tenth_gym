# Coding Style Guide

This project follows **[Google Python Style Guide](google_python_style_guide.md)** as its baseline.
This document records project-specific conventions, exceptions, and tooling setup.

---

## 1. General Principles

- Readability and consistency matter more than cleverness.
- Prefer the **smallest complete fix** over general-purpose solutions.
- All new code must be reviewable by another team member.

---

## 2. Deviations & Clarifications from Google Style

| Google Rule | Project Convention |
|---|---|
| **2.2 Imports** — avoid wildcard imports | Wildcard (`from module import *`) is **only permitted** in `__init__.py` for re-exporting a public API surface. Never use wildcard imports in other modules. |
| **2.2 Imports** — import ordering | Google's groups are followed. Use **absolute imports** for local modules (`from f110_gym.envs.base_classes import Simulator`). |
| **3.4 Strings** — prefer `'` for string literals | Use **single quotes** `'...'` for string literals that do not contain a single-quote character. Use double quotes `"..."` when the string contains a single quote, or for docstrings (`"""..."""`). |
| **3.6.2 Type Hints** — Google recommends them | Type hints are **encouraged but not yet required** for existing code. All new public functions **must** include type hints. |
| **3.11.2 Line length** — 80 characters | Keep to **88 characters** (Black default). Existing lines over 88 are tolerated but should be refactored when touched. |

**Rationale** for wildcard exceptions in `__init__.py`:
Keeps internal package structure flexible while exposing a stable public API.

---

## 3. Naming Conventions

| Category | Convention | Example |
|---|---|---|
| Package / Module | `snake_case` | `f110_gym`, `laser_models` |
| Class | `PascalCase` | `F110Env`, `ScanSimulator2D` |
| Function / Method | `snake_case` | `get_vertices`, `check_ttc_jit` |
| Variable | `snake_case` | `scan_angles`, `collision_check` |
| Private (module/class) | `_leading_underscore` | `_ray_cast_worker` |
| Constant | `UPPER_SNAKE_CASE` | `DEFAULT_PARAMS`, `MAX_STEERING_ANGLE` |
| Enum member | `UPPER_SNAKE_CASE` | `Integrator.RK4` (as set by the stdlib `Enum`) |

---

## 4. Project-Specific Patterns

### 4.1 Docstrings

Use Google-style docstrings (as recommended by the Google guide):

```python
def vehicle_dynamics_st(x, u, params):
    """Single-track vehicle dynamics model.

    Args:
        x (np.ndarray): State vector [x, y, psi, v, ...].
        u (np.ndarray): Control vector [steering, acceleration].
        params (dict): Vehicle parameters dict (mu, C_Sf, ...).

    Returns:
        np.ndarray: Time derivative of the state vector.
    """
```

All public modules must have a module-level docstring:

```python
"""Prototype of Utility functions and classes for simulating 2D LIDAR scans.

Author: Hongrui Zheng
"""
```

### 4.2 NUMBA JIT Functions

- Use `@njit` (not `@jit`) for NUMBA-compiled functions.
- Keep NUMBA functions **pure** — no side effects, no I/O.
- Unit-test NUMBA functions with and without the JIT path:

```python
# Test the Python fallback when NUMBA_DISABLE_JIT=1
```

### 4.3 Gymnasium Environment Registration

Register environments in the top-level `__init__.py`:

```python
from gymnasium.envs.registration import register

register(
    id='f110-v0',
    entry_point='f110_gym.envs:F110Env',
)
```

Keep registration calls **minimal** — only `id`, `entry_point`, and kwargs if needed.

### 4.4 Imports Order (within groups)

Within each import group, sort **alphabetically** by the full import path:

```python
# Standard library
import time
from pathlib import Path

# Third-party
import gymnasium as gym
import numpy as np
from gymnasium import spaces
from numba import njit

# Local application
from f110_gym.envs.base_classes import Simulator
from f110_gym.envs.collision_models import get_vertices
```

### 4.5 Variable Type Annotations

Annotate variables whose type is not immediately obvious from the assignment.
Use standard library types (`list`, `dict`, `tuple`, `set`, `Optional`, `Union`)
rather than the `typing` module's deprecated aliases in Python 3.10+.

```python
# Good — type clarified from context
scan_angles: np.ndarray = np.linspace(-2.0, 2.0, 1080)
params: dict[str, float] = DEFAULT_PARAMS.copy()
collision_check: bool | None = None

# Acceptable — type is obvious from literal
scan_angles = np.linspace(-2.0, 2.0, 1080)   # no annotation needed
count = 0                                      # no annotation needed
```

Rules of thumb:
- **Annotate** when the variable is a public module/class attribute, or when the
  assigned value does not reveal the type (e.g. `result = some_function()`).
- **Skip** the annotation when the type is trivially clear (e.g. `count = 0`,
  `name = 'map'`, `values = []`).
- For NUMBA `@njit` functions, variable annotations inside the JIT-compiled
  body are optional — NUMBA infers types at compile time.

### 4.6 Constants

Define module-level constants **after imports, before class/function definitions**:

```python
import numpy as np

# ----------
# Constants
# ----------
DEFAULT_PARAMS = {
    'mu': 1.0489,
    'C_Sf': 4.718,
    # ...
}
```

### 4.7 Backup / Scratch Files

- **Never commit** `*_backup.py`, `*_scratch.py`, or `*_playground.py` files.
- These are already covered by `.gitignore`. Use them for local experiments only.
- If you need a scratch module, add it to `.gitignore` explicitly.

---

## 5. Pre-commit Hooks

We use [`pre-commit`](https://pre-commit.com) to enforce style automatically on every commit.

### 5.1 Setup (one-time)

```bash
# Install pre-commit
pip install pre-commit

# Install the hooks into this repo
pre-commit install

# (Optional) Run on all files once to check
pre-commit run --all-files
```

### 5.2 Usage

After setup, hooks run automatically on `git commit`. If a hook modifies files
(e.g. trailing whitespace removal), stage the changes and re-commit.

To skip hooks temporarily (emergency only):

```bash
git commit --no-verify
```

### 5.3 Tool Stack

The following tools are configured in `.pre-commit-config.yaml`:

| Tool | Purpose | Stage |
|---|---|---|
| [`trailing-whitespace`](https://github.com/pre-commit/pre-commit-hooks) | Trim trailing whitespace | Pre-commit |
| [`end-of-file-fixer`](https://github.com/pre-commit/pre-commit-hooks) | Ensure files end with one newline | Pre-commit |
| [`check-yaml`](https://github.com/pre-commit/pre-commit-hooks) | Validate YAML files | Pre-commit |
| [`check-added-large-files`](https://github.com/pre-commit/pre-commit-hooks) | Warn on files > 500 KB | Pre-commit |
| [`check-merge-conflict`](https://github.com/pre-commit/pre-commit-hooks) | Detect unresolved merge conflict markers | Pre-commit |
| [`black`](https://github.com/psf/black) | Auto-format Python (line-length 88) | Pre-commit |
| [`isort`](https://pycqa.github.io/isort/) | Sort imports (compatible with Black) | Pre-commit |
| [`flake8`](https://flake8.pycqa.org/) | Lint for errors, style violations | Pre-commit |
| [`mypy`](https://mypy-lang.org/) | Static type checking (optional for existing code) | Pre-commit |

> **Note**: `flake8` is configured to respect Black's line length (88) and ignore
> style rules that conflict with Black.
>
> **Note**: `mypy` is set to `--ignore-missing-imports` so missing third-party stubs
> do not block commits.

---

## 6. Commit Message Convention

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>: <short summary>

[optional body]

[optional footer]
```

Types:

| Type | Usage |
|---|---|
| `feat` | New feature |
| `fix` | Bug fix |
| `refactor` | Code change that is neither fix nor feature |
| `style` | Formatting, linting, pre-commit changes only |
| `docs` | Documentation-only changes |
| `test` | Adding or updating tests |
| `chore` | Build, CI, tooling, dependency changes |

Examples:

```
feat: add render mode selection for pyglet viewer

fix: handle NaN in laser scan when vehicle is out of map bounds

docs: add pre-commit setup instructions to coding_style.md
```

---

## 7. Enforcement

- **CI pipeline** will run `black --check`, `isort --check`, `flake8`, and `mypy`.
  A commit that fails any check will be blocked.
- Exceptions for legacy code are documented inline with `# noqa` (flake8) or
  `# type: ignore` (mypy), accompanied by a comment explaining why.

---

*Last updated: 2026-05-26*
