#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

BUILD_ALL=0
if [[ "${1:-}" == "--all" ]]; then
  BUILD_ALL=1
elif [[ $# -gt 0 ]]; then
  echo "usage: $0 [--all]" >&2
  exit 2
fi

if [[ "$BUILD_ALL" -eq 1 ]]; then
  echo "=== planner/tree_search (debug) ==="
  (cd "$ROOT/planner/tree_search" && cmake --preset debug && cmake --build build-debug)

  echo
fi

echo "=== planner/f110_self_play (debug) ==="
(cd "$ROOT/planner/f110_self_play" && cmake --preset debug && cmake --build build-debug)

if [[ "$BUILD_ALL" -eq 1 ]]; then
  echo
  echo "=== gym/f110_gym/rollout_kernel (debug) ==="
  (cd "$ROOT/gym/f110_gym/rollout_kernel" && cmake --preset debug && cmake --build build-debug)
fi
