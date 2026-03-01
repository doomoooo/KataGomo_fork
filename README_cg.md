# CUDA Graph And Benchmark Notes (Current Code)

This file documents the current behavior in this repo (not a commit-by-commit history).

## Scope

- Focus: TensorRT backend CUDA Graph behavior, batching knobs, and benchmark tooling.
- Code paths covered:
  - `cpp/program/setup.cpp`
  - `cpp/neuralnet/nneval.cpp`
  - `cpp/neuralnet/trtbackend.cpp`
  - `cpp/command/benchmark.cpp`
  - `run.sh`, `benchmark.sh`, `python/benchmark.py`, `python/visualize_benchmark.py`

## Key Runtime Behavior

### `trtUseCudaGraph`

- Config key: `trtUseCudaGraph` (or generic `useCudaGraph` alias via setup parsing).
- Effective today only on TensorRT backend. Other backends parse but ignore this flag.
- In `cpp/configs/gtp_example.cfg`, it is enabled by default:
  - `trtUseCudaGraph = true`

When enabled in TensorRT:

- CUDA Graphs are pre-captured at startup for every batch size `1..nnMaxBatchSize`.
- Capture includes:
  - H2D input copies (`cudaMemcpyAsync`)
  - inference launch (`enqueueV3`)
  - D2H output copies (`cudaMemcpyAsync`)
- Host input/output arrays are pinned (`cudaMallocHost`) so graph-captured async copies are valid.
- Runtime executes `cudaGraphLaunch` using pre-captured graph exec objects (no lazy capture in first request path).
- Failure mode is fail-fast:
  - capture/instantiate/launch issues throw immediately (no silent fallback path).
- Startup pre-capture is globally serialized across threads to avoid TensorRT/Myelin legacy-stream race:
  - `"operation would make the legacy stream depend on a capturing blocking stream"`

### `nnMinBatchSize`

- Config key: `nnMinBatchSize` (default `1`).
- Constraint: `1 <= nnMinBatchSize <= nnMaxBatchSize`.
- Queue behavior:
  - NN server threads try to dequeue at least `nnMinBatchSize` requests.
  - If queue has fewer, they may wait for more.
  - They still make forward progress when pending-eval upper bound indicates no more requests are coming immediately.
- Practical guidance:
  - Keep at `1` for lower latency.
  - Increase only if intentionally trading latency for larger effective batch size / throughput.

## TensorRT Build/Engine Knobs

All are parsed in `setup.cpp` and passed into TensorRT backend.

### `trtBuilderOptimizationLevel`

- Range: `-1..5`
- `-1` means do not call `setBuilderOptimizationLevel` (TensorRT default).

### `trtAvgTimingIterations`

- Range: `-1..1000`
- `-1` means do not call `setAvgTimingIterations` (TensorRT default).

### `trtMaxAuxStreams`

- Range: `-1..1024`
- `-1` means do not call `setMaxAuxStreams` (TensorRT default).

### `trtSetTacticSources`

- Bool, default `true`.
- Controls whether backend constrains tactic sources (architecture-dependent path in `trtbackend.cpp`).

### `trtMultiProfile`

- Bool, default `false`.
- If enabled:
  - Build multiple optimization profiles keyed by batch ranges.
  - Runtime maps each batch size to a specific profile.
  - Plan cache can reuse a previously built larger-batch multi-profile plan and clip mapping to current `nnMaxBatchSize`.

## Multi-GPU Thread Mapping

- Main knob: `numNNServerThreadsPerModel`.
- Per-thread device mapping:
  - `trtDeviceToUseThread0`, `trtDeviceToUseThread1`, ...
- Also supports model/thread-scoped forms via setup parser, but thread-scoped keys above are the common path.

## Benchmark Command Behavior

`katago benchmark` currently supports exact fixed batch size in evaluator init:

- `-fixed-batch-size <N>`
- `-half-batch-size`

It no longer forces internal round-up to older alignment heuristics when fixed batch is requested.

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

- Launches `katago gtp` with a composed `-override-config`.
- Exposes env-editable knobs near top of script:
  - `TRT_BUILDER_OPT_LEVEL`
  - `TRT_AVG_TIMING_ITERS`
  - `TRT_MAX_AUX_STREAMS`
  - `TRT_SET_TACTIC_SOURCES`
  - `TRT_MULTI_PROFILE`
  - `NN_MAX_BATCHSIZE`
  - `NN_MIN_BATCHSIZE`
  - `TRT_CUDA_STREAMS`
  - `TRT_DEVICE_ID`
- Adds per-thread `trtDeviceToUseThreadN` automatically.

### `benchmark.sh` (KataGo benchmark helper)

- Launches `katago benchmark` with:
  - `-fixed-batch-size "${NN_MAX_BATCHSIZE}"`
  - `numNNServerThreadsPerModel=${TRT_CUDA_STREAMS}`
  - `nnMinBatchSize=${NN_MIN_BATCHSIZE}`
  - TensorRT tuning overrides listed above.

### `python/benchmark.py` (trtexec matrix runner)

- Can either:
  - build/ensure plan via `katago gtp`, or
  - use prebuilt plans via `--plan-file`.
- Runs `trtexec` over combinations of:
  - plan batch (`pb`)
  - infer batch (`ib`)
  - streams
  - cuda graph on/off (`--graph-modes off,on`)
- Persists resumable JSON results with parsed throughput/latency metrics.

Quick smoke:

```bash
python3 python/benchmark.py --smoke
```

### `python/visualize_benchmark.py`

- Reads JSON from `python/benchmark.py`.
- Plots fixed-plan curves by `ib`, split by stream count and graph mode.
- Default metric is `nn_evals_per_sec` (`throughput_qps * infer_batch`).

Example:

```bash
python3 python/visualize_benchmark.py \
  --input-json build/trtexec_benchmark.json \
  --output-dir build/benchmark_plots
```

## Practical Defaults In Repo Configs

- `cpp/configs/gtp_example.cfg`:
  - `trtUseCudaGraph = true` (enabled)
  - `nnMinBatchSize` documented, default remains `1` unless set.
- `analysis_example.cfg` / `match_example.cfg`:
  - include `nnMinBatchSize` docs
  - keep `trtUseCudaGraph` commented (`false` if explicitly set).
