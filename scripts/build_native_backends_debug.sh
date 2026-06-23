#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "=== planner/tree_search (debug) ==="
(cd "$ROOT/planner/tree_search" && cmake --preset debug && cmake --build build-debug)

echo
echo "=== planner/f110_self_play (debug) ==="
(cd "$ROOT/planner/f110_self_play" && cmake --preset debug && cmake --build build-debug)

echo
echo "=== gym/f110_gym/rollout_kernel (debug) ==="
(cd "$ROOT/gym/f110_gym/rollout_kernel" && cmake --preset debug && cmake --build build-debug)
