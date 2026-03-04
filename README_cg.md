# CUDA Graph And Benchmark Notes (Current Code)

This file documents the current behavior in this repo.

## Scope

- Focus: TensorRT backend CUDA Graph behavior, batching semantics, and benchmark tooling.
- Code paths covered:
  - `cpp/program/setup.cpp`
  - `cpp/neuralnet/nneval.cpp`
  - `cpp/neuralnet/trtbackend.cpp`
  - `cpp/command/benchmark.cpp`
  - `run.sh`, `benchmark.sh`, `python/benchmark.py`, `python/visualize_benchmark.py`

## Key Runtime Behavior

### `trtUseCudaGraph`

- Config key: `trtUseCudaGraph` (or generic `useCudaGraph` alias via setup parsing).
- Effective on TensorRT backend.

When enabled in TensorRT:

- CUDA Graphs are pre-captured at startup for every batch size `1..nnMaxBatchSize`.
- Capture includes:
  - H2D input copies (`cudaMemcpyAsync`)
  - inference launch (`enqueueV3`)
  - D2H output copies (`cudaMemcpyAsync`)
- Host input/output arrays are pinned (`cudaMallocHost`) so graph-captured async copies are valid.
- Runtime executes `cudaGraphLaunch` with pre-captured graph exec objects.
- Failure mode is fail-fast: capture/instantiate/launch issues throw immediately.
- Startup pre-capture is globally serialized across threads to avoid TensorRT/Myelin legacy-stream race.

### Batching (`nnMaxBatchSize` only)

- The only batch-size config knob is `nnMaxBatchSize`.
- Scheduler behavior matches former `nnMinBatchSize == nnMaxBatchSize` semantics:
  - If GPU side has no additional pending work to wait for, it launches immediately.
  - Otherwise it waits to accumulate up to the current target batch (bounded by `nnMaxBatchSize`).

Removed knobs (no longer supported):

- `nnMinBatchSize`
- `trtSetTacticSources` / `setTacticSources`
- `trtMultiProfile` / `multiProfile`

## TensorRT Build/Engine Knobs

These are parsed in `setup.cpp` and passed into TensorRT backend.

### `trtBuilderOptimizationLevel`

- Range: `-1..5`
- `-1` means do not call `setBuilderOptimizationLevel` (TensorRT default).

### `trtAvgTimingIterations`

- Range: `-1..1000`
- `-1` means do not call `setAvgTimingIterations` (TensorRT default).

### `trtMaxAuxStreams`

- Range: `-1..1024`
- `-1` means do not call `setMaxAuxStreams` (TensorRT default).

### Single optimization profile only

- TensorRT runtime is single-profile only.
- Cached plans containing multiple optimization profiles are rejected and must be rebuilt.

## Multi-GPU Thread Mapping

- Main knob: `numNNServerThreadsPerModel`.
- Per-thread device mapping:
  - `trtDeviceToUseThread0`, `trtDeviceToUseThread1`, ...

## Benchmark Command Behavior

`katago benchmark` supports fixed evaluator batch size via:

- `-fixed-batch-size <N>`
- `-half-batch-size`

Example:

```bash
./katago benchmark \
  -config cpp/configs/gtp_example.cfg \
  -model /path/to/model.onnx \
  -fixed-batch-size 9 \
  -threads 27
```

## Included Scripts

### `run.sh` (GTP run helper)

- Launches `katago gtp` with composed `-override-config`.
- Exposes knobs near top of script:
  - `TRT_BUILDER_OPT_LEVEL`
  - `TRT_AVG_TIMING_ITERS`
  - `TRT_MAX_AUX_STREAMS`
  - `NN_MAX_BATCHSIZE`
  - `TRT_CUDA_STREAMS`
  - `TRT_DEVICE_ID` / `TRT_DEVICE_IDS`

### `benchmark.sh` (KataGo benchmark helper)

- Launches `katago benchmark` with:
  - `-fixed-batch-size "${NN_MAX_BATCHSIZE}"`
  - `numNNServerThreadsPerModel=${TRT_CUDA_STREAMS}`
  - TensorRT tuning overrides listed above.
- Supports CLI args (instead of `PGO_*` env overrides), for example:

```bash
./benchmark.sh --visits 2000 --cuda-streams 4 --batch-size 16
./benchmark.sh --katago-bin ./cpp/build_pgo_gen/katago --visits 1000
```

### `build.sh --pgo` (PGO build entrypoint)

- Use `build.sh` directly for PGO builds.
- Example:

```bash
./build.sh --pgo --pgo-bench-visits 2000
```

### `python/benchmark.py` (trtexec runner)

- Default workflow (without `--plan-file`):
  - For each tested batch size, call `katago gtp` once to build/ensure that batch's plan cache.
  - Run `trtexec` matrix for that batch over stream counts.
- Optional: use existing plans via `--plan-file`.
- Persists resumable JSON results with parsed throughput/latency metrics.
- Default output paths are under `./benchmark/`.
  - Filenames include model filename and GPU model.

Quick smoke:

```bash
python3 python/benchmark.py --smoke
```

### `python/visualize_benchmark.py`

- Reads JSON from `python/benchmark.py` and plots curves.

## Practical Defaults In Repo Configs

- `cpp/configs/gtp_example.cfg`:
  - `trtUseCudaGraph = true` (enabled)
- `analysis_example.cfg` / `match_example.cfg`:
  - `trtUseCudaGraph` remains optional/commented.
