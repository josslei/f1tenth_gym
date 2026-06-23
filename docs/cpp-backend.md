# Native C++ Backend

This document covers the shared native backend used by planner training.
Module-specific details stay in each module README.

## What It Is

The backend is split into two native packages:

- `planner/tree_search/`: generic MuZero/tree-search code
- `planner/f110_self_play/`: F110 self-play orchestration around that search

The Python training entrypoint loads both modules and saves a TorchScript model
that the native search adapter reads from disk.

## Dependencies

The native build expects vendored third-party libraries under `thirdparty/`:

- `thirdparty/pybind11/`
- `thirdparty/libtorch/`

`thirdparty/libtorch/` must be an extracted LibTorch distribution root with
`include/`, `lib/`, and `share/cmake/Torch/`.

## Build

Each package has its own CMake project and debug preset.

```bash
# release build
scripts/build_native_backends_release.sh

# debug build (symbols, assertions)
scripts/build_native_backends_debug.sh
```

From `planner/tree_search/`:

```bash
cmake --preset debug
cmake --build build-debug
```

From `planner/f110_self_play/`:

```bash
cmake --preset debug
cmake --build build-debug
```

Use the debug preset when you need symbols, assertions, or a crash trace.

## Debugging A Crash

If the planner segfaults:

1. Rebuild both native packages with the debug preset.
2. Run the Python entrypoint again.
3. If the crash is still opaque, run under `lldb`:

```bash
lldb -- python runs/train_muzero_planner.py --config configs/muzero/default.yaml
```

4. `runs/train_muzero_planner.py` enables `faulthandler`, so Python will print a
   traceback when the process dies from a fatal signal.

When `lldb` stops at the crash, collect the full native stack:

```lldb
bt all
```

If a device crash occurs, first rebuild the native modules and run with CPU:

```bash
python runs/train_muzero_planner.py --config configs/muzero/default.yaml --device cpu
```

The script prints the resolved MuZero device before constructing native search:

```text
MuZero device: cpu
```

Rebuild `planner/f110_self_play` after any native search adapter header change;
otherwise Python can keep loading an old `.so` with stale device handling.

If CPU still reaches GPU code, treat it as stale native code or a stray device
tensor before changing search or environment logic.

## Runtime Flow

1. `runs/train_muzero_planner.py` loads config.
1. It builds the model and writes a scripted copy to
   `outputs/rl/muzero_f110_gym_10/current_model.pt`.
1. `planner/f110_self_play.MuZeroSearchAdapter` loads that scripted model from
   disk.
1. Native self-play calls into the search backend and the Gym backend.

## Device Behavior

The MuZero native backend supports CUDA and CPU only.

The search adapter chooses a Torch device and moves the observation batch to
that device before calling TorchScript inference.

## Output And Metrics

Native metrics are not printed by default.

- `search.print_metrics` prints the detailed native search timing/tree block.
- `self_play.print_metrics` prints the rollout and episode summary block.
- training logs should use the structured metrics returned to Python.

## Where To Look First

- `planner/tree_search/README.md` for generic search-specific notes
- `planner/f110_self_play/README.md` for F110 orchestration notes
- `runs/train_muzero_planner.py` for the Python-side startup order

## Troubleshooting Checklist

- Rebuild in debug mode if the `.so` was built in release mode.
- Make sure `current_model.pt` exists before constructing the search adapter.
- Confirm the TorchScript model device matches the tensor device used by the
  native search call.
- Use `lldb` when the failure is a segfault and the Python traceback is not
  enough.
