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
  - If capture/instantiate fails, backend logs reason and safely falls back to normal enqueue path.
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

## Default Config On This Branch

The default `gtp` config has been set as:

- `trtUseCudaGraph = true`
- `sampleSearchThreadStates = false`

Location: `cpp/configs/gtp_example.cfg`
