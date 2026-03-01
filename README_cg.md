# CUDA Graph And Benchmark Notes

This document summarizes the branch changes introduced after commit `55a48792` (`add QAT settings to onnx proto`), excluding that commit itself.

## New Commits In This Branch

1. `285ca441` - Add TensorRT cudaGraph option with per-batch graph capture
2. `b9f691af` - benchmark: allow exact fixed batch size values
3. `085853a5` - Add python benchmark using trtexec

## What Changed And How To Use

### 1) TensorRT cudaGraph option (`285ca441`)

- Added config key: `trtUseCudaGraph`.
- Added/updated TensorRT build-time option behavior:
  - `trtBuilderOptimizationLevel`
  - `trtAvgTimingIterations`
  - `trtMaxAuxStreams`
- Behavior:
  - When enabled, TensorRT pre-captures CUDA Graph for batch sizes `1..nnMaxBatchSize`.
  - Runtime uses graph launch path when capture succeeds.
  - Current branch behavior is fail-fast for cudaGraph issues in TensorRT path (no silent fallback), to expose problems during development.
  - `trtBuilderOptimizationLevel = -1` (default): do not call `setBuilderOptimizationLevel`, use TensorRT default.
  - `trtAvgTimingIterations = -1` (default): do not call `setAvgTimingIterations`, use TensorRT default.
  - `trtMaxAuxStreams = -1` (default): do not call `setMaxAuxStreams`, use TensorRT default.
  - Set explicit values to force calls:
    - `trtBuilderOptimizationLevel = 0..5`
    - `trtAvgTimingIterations = 0..1000`
    - `trtMaxAuxStreams = 0..1024`
- Usage:
  - In config file:
    - `trtUseCudaGraph = true`
    - `trtBuilderOptimizationLevel = -1`
    - `trtAvgTimingIterations = -1`
    - `trtMaxAuxStreams = -1`
  - Or CLI override:
    - `-override-config trtUseCudaGraph=true`
    - `-override-config trtBuilderOptimizationLevel=-1`
    - `-override-config trtAvgTimingIterations=-1`
    - `-override-config trtMaxAuxStreams=-1`

### 2) Benchmark exact fixed batch size (`b9f691af`)

- `benchmark` now supports exact requested batch size without forced round-up behavior in evaluator init path.
- Related flags:
  - `--fixed-batch-size <N>`
  - `--half-batch-size`
- Example:
  - `./katago benchmark -config cpp/configs/gtp_example.cfg -model /path/model.onnx --fixed-batch-size 9 -threads 27`

### 3) Python TensorRT benchmark pipeline (`085853a5`)

- Added:
  - `python/benchmark.py`
  - `python/visualize_benchmark.py`
- Purpose:
  - Generate/ensure TensorRT plans via `katago gtp`.
  - Run `trtexec` benchmark across combinations of:
    - plan batch (`pb`)
    - infer batch (`ib`)
    - stream count
    - cudaGraph mode on/off
  - Save structured results to JSON.
- Typical commands:
  - Run smoke:
    - `python3 python/benchmark.py --smoke`
  - Full run:
    - `python3 python/benchmark.py --katago-bin build/katago --config cpp/configs/gtp_example.cfg --model /path/model.onnx --output-json build/trtexec_benchmark.json`
  - Visualize:
    - `python3 python/visualize_benchmark.py --input-json build/trtexec_benchmark.json --output-dir build/benchmark_plots`

### 4) Fail-fast cudaGraph behavior (`a7188629`)

- cudaGraph failures in TensorRT path now throw immediately (development-mode fail-fast).

### 5) cudaGraph startup crash fix (post `a7188629`)

- Fixed an intermittent TensorRT cudaGraph pre-capture startup crash:
  - `operation would make the legacy stream depend on a capturing blocking stream`
  - Seen as warmup failure during pre-capture of some batch sizes.
- Change:
  - Serialize pre-capture across NN server threads to avoid the startup race with TensorRT/Myelin internal legacy-stream operations.
- Impact:
  - Slightly slower startup pre-capture when multiple NN server threads initialize on the same GPU.
  - Runtime inference path for this commit stayed unchanged (later updated in section 8).

### 6) cudaGraph now captures copy + inference

- TensorRT cudaGraph capture path now includes:
  - host-to-device input copies
  - `enqueueV3` inference launch
  - device-to-host output copies
- `InputBuffers` host arrays are now allocated as pinned memory (`cudaMallocHost`) to support async copy nodes in graph capture.
- Graph pre-capture runs during NN server thread initialization (after per-thread `InputBuffers` creation), so the first `getOutput` does not pay capture cost.

### 7) NN minimum dequeue batch size (`nnMinBatchSize`)

- Added config key: `nnMinBatchSize` (default `1`).
- Behavior:
  - NN server threads target at least `nnMinBatchSize` requests when dequeuing from the shared NN queue.
  - If fewer are queued, they wait for more requests.
  - If no additional requests are pending, they proceed with the currently queued requests to avoid stalling.
- Guidance:
  - Keep `nnMinBatchSize = 1` for latency-sensitive use.
  - Increase only when intentionally trading latency for larger batches / throughput.

## Default Config On This Branch

The default `gtp` config has been set as:

- `trtUseCudaGraph = true`

Location: `cpp/configs/gtp_example.cfg`
