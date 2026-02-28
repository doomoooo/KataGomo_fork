# CUDA Graph And Benchmark Notes

This document summarizes the branch changes introduced after commit `55a48792` (`add QAT settings to onnx proto`), excluding that commit itself.

## New Commits In This Branch

1. `285ca441` - Add TensorRT cudaGraph option with per-batch graph capture
2. `b9f691af` - benchmark: allow exact fixed batch size values
3. `8f5ad2ec` - Add search thread state stat
4. `085853a5` - Add python benchmark using trtexec

## What Changed And How To Use

### 1) TensorRT cudaGraph option (`285ca441`)

- Added config key: `trtUseCudaGraph`.
- Behavior:
  - When enabled, TensorRT pre-captures CUDA Graph for batch sizes `1..nnMaxBatchSize`.
  - Runtime uses graph launch path when capture succeeds.
  - Current branch behavior is fail-fast for cudaGraph issues in TensorRT path (no silent fallback), to expose problems during development.
- Usage:
  - In config file:
    - `trtUseCudaGraph = true`
  - Or CLI override:
    - `-override-config trtUseCudaGraph=true`

### 2) Benchmark exact fixed batch size (`b9f691af`)

- `benchmark` now supports exact requested batch size without forced round-up behavior in evaluator init path.
- Related flags:
  - `--fixed-batch-size <N>`
  - `--half-batch-size`
- Example:
  - `./katago benchmark -config cpp/configs/gtp_example.cfg -model /path/model.onnx --fixed-batch-size 9 -threads 27`

### 3) Search thread GPU-state sampling (`8f5ad2ec`)

- Added search param: `sampleSearchThreadStates`.
- Behavior when enabled:
  - Samples each search thread GPU state every 10ms.
  - Aggregates histograms for:
    - tree searching (no queued GPU task)
    - queued/waiting for idle GPU
    - waiting stream1
    - waiting stream2
  - Logs summary on evaluator shutdown.
- Usage:
  - In config:
    - `sampleSearchThreadStates = true`
  - Keep disabled for normal runs if you do not need telemetry (small overhead).

### 4) Python TensorRT benchmark pipeline (`085853a5`)

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

### 5) Fail-fast and benchmark-only batch histogram (`a7188629`)

- cudaGraph failures in TensorRT path now throw immediately (development-mode fail-fast).
- Batch size distribution logging is gated to benchmark mode only, and only when explicitly enabled.
- New config switch for benchmark telemetry:
  - `trtRecordBatchSizeHistogram = true|false` (default off unless set).

### 6) cudaGraph startup crash fix (post `a7188629`)

- Fixed an intermittent TensorRT cudaGraph pre-capture startup crash:
  - `operation would make the legacy stream depend on a capturing blocking stream`
  - Seen as warmup failure during pre-capture of some batch sizes.
- Change:
  - Serialize pre-capture across NN server threads to avoid the startup race with TensorRT/Myelin internal legacy-stream operations.
- Impact:
  - Slightly slower startup pre-capture when multiple NN server threads initialize on the same GPU.
  - Runtime inference path is unchanged.

### 7) Additional benchmark telemetry (post `a7188629`)

- Added benchmark output metric:
  - `gpuDupRows = <count> (<pct>)`
  - Meaning: repeated GPU inference rows actually executed on GPU (not just CPU-side contention).
- Extended existing 10ms state sampler to also track NN server threads:
  - Each server thread has two states: `GPU_IDLE` / `GPU_BUSY`.
  - Logs histogram of idle NN server thread count on shutdown:
    - `NNEval server idle-thread count (GPU idle): ...`

## Default Config On This Branch

The default `gtp` config has been set as:

- `trtUseCudaGraph = true`
- `sampleSearchThreadStates = false`

Location: `cpp/configs/gtp_example.cfg`
