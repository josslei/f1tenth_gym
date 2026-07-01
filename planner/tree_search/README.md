# Tree Search

Native batched tree-search components for planner use.

Shared C++ backend guidance lives in `../../docs/`.

This package is split into:

- `include/`: header-only tree, payload, and search functor interfaces.
- `src/`: pybind11 module entrypoints and future compiled implementation code.
- `backend.py`: Python shim for the compiled `tree_search_native` extension.
- `mcts.py` / `muzero.py`: Python-side planner wrappers.

The current C++ API is still a scaffold. Keep changes small and preserve the split between generic tree-search code here and F110-specific rollout logic under `gym/f110_gym/rollout_kernel/`.

## Search API

Current public entry points:

- `search_batch(...)`: batched MuZero search for exactly `B` roots.
- `search_one(...)`: placeholder for a future single-instance fast path.
- `get_metrics()`: aggregated scalar metrics from the most recent search call,
  intended for Python-side TensorBoard logging.

The `MuZeroSearch` constructor accepts `print_metrics=false` by default.
Native C++ search metrics are printed only when `print_metrics` is enabled;
callers should use
`get_metrics()` for training logs.

The search object stores its own scratch state internally. There is no separate scratch class.

## Dependencies

The CMake build expects vendored third-party dependencies under the repository `thirdparty/` directory:

- `thirdparty/pybind11/`
- `thirdparty/libtorch/`

`thirdparty/libtorch/` should be the extracted LibTorch distribution root, containing `include/`, `lib/`, and `share/cmake/Torch/`. The repo intentionally does not pin a LibTorch version or platform build. Choose the CPU/CUDA/macOS/Linux package that matches the target machine.

`thirdparty/libtorch/` and downloaded zip archives are ignored by git.

## Configure And Build

From this directory:

```bash
cmake --preset default
cmake --build build
```

The default preset builds `Release`. For a debug build:

```bash
cmake --preset debug
cmake --build build-debug
```

Equivalent commands from the repository root:

```bash
cmake -S planner/tree_search -B planner/tree_search/build
cmake --build planner/tree_search/build
```

The default release build includes local CPU tuning.

Run `scripts/build_native_backends_release.sh --all` or
`scripts/build_native_backends_debug.sh --all` from the repo root to include this
standalone binding in the repo-level build. Without `--all`, the scripts build
only `planner/f110_self_play`, which embeds the tree-search headers needed by
MuZero training.

The build creates the pybind11 extension target `tree_search_native` and writes
the extension into `planner/tree_search/` so `backend.py` can import it directly.

## Compile Commands

`CMakePresets.json` configures the build directory as:

```text
planner/tree_search/build/
```

and enables:

```cmake
CMAKE_EXPORT_COMPILE_COMMANDS=ON
```

After configure, CMake writes:

```text
planner/tree_search/build/compile_commands.json
```

VS Code should point C/C++ or clangd tooling at that file.

## Design Notes

The native binding builds its internal tree shape from primitive constructor
arguments. `B`, `Nmax`, `A`, and `H` are fixed for the lifetime of one
`MuZeroSearch` instance, but they are not part of the Python surface.

MuZero hidden state is intentionally stored as a `torch::Tensor`. When GPU inference is available, hidden state should stay device-resident instead of being copied through native CPU storage.
