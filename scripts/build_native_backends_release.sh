#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "=== planner/tree_search (release) ==="
(cd "$ROOT/planner/tree_search" && cmake --preset default && cmake --build build)

echo
echo "=== planner/f110_self_play (release) ==="
(cd "$ROOT/planner/f110_self_play" && cmake --preset default && cmake --build build)

echo
echo "=== gym/f110_gym/rollout_kernel (release) ==="
(cd "$ROOT/gym/f110_gym/rollout_kernel" && cmake --preset default && cmake --build build)
