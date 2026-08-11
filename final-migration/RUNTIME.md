# Plan-driven CUDA runtime

The runtime consumes only `best-tactic-plan.json` files emitted by the unified
autotuner. Loading is fail-closed: the plan must be production-ready, have a
complete positive-history closure and full correctness certificate, match the
model SHA-256 and exact receiver GPU fingerprint, and contain the selected
exact batch. The fingerprint binds product name, compute capability, global
memory, SM/L2/memory-bus topology, execution limits, and copy/concurrency
capabilities. Old development plans and implicit tactic fallback are rejected.

## Single GPU

For a two-stream plan on device 0, add the following to the normal GTP config:

```cfg
cudaTacticPlanFile = /absolute/path/to/best-tactic-plan.json
cudaTacticPlanBatch = 12

numNNServerThreadsPerModel = 2
cudaDeviceToUseThread0 = 0
cudaDeviceToUseThread1 = 0

cudaAsyncInferPipeline = true
cudaEventPipelineUseGraph = false
```

`cudaTacticPlanBatch` may be omitted when the plan contains exactly one
certified batch. The loader supplies `nnMaxBatchSize`, exact 19x19 shape,
FP16/NHWC, `nnBatchAwareDispatch=true`, and all tactic overrides from that
batch. Conflicting user settings are rejected instead of silently changed.

## Multiple GPUs

A plan's stream count is per device. For two devices with a two-stream plan,
configure four server threads and keep each pair on its owning device:

```cfg
numNNServerThreadsPerModel = 4
cudaDeviceToUseThread0 = 0
cudaDeviceToUseThread1 = 0
cudaDeviceToUseThread2 = 1
cudaDeviceToUseThread3 = 1
```

Every persistent submission worker selects its receiver device before touching
that device's cuBLAS handle, streams, events, copies, graph, or allocations.
The batch dispatcher tracks idleness per physical GPU, so one device cannot
authorize a partial launch on another device.

## Scheduling contract

With `nnBatchAwareDispatch=true`, a full physical batch may launch immediately.
An underfilled batch waits for more search requests unless its target GPU is
completely idle. Every backend launch still has the plan's exact physical
batch; an idle partial launch pads missing rows rather than selecting another
kernel shape.

With `cudaAsyncInferPipeline=true`, each inference lane has a persistent host
submission worker, pinned host staging, dedicated H2D and D2H streams, and one
device storage slot protected by input-consumed and output-consumed events.
This permits the next transfer and search-side production to overlap compute
without ping-pong storage or an in-place output race.

Search must provide enough independent work to keep every lane fed. A practical
starting point is:

```text
numSearchThreads = batch * (total inference lanes + 1) + C
```

where `C` is a modest long-tail allowance, usually 12 to 32, and must be tuned
against CPU availability and search strength. `visits/s` must remain strictly
greater than real `nnEval/s`; physical throughput comparisons use launched
batches multiplied by the exact plan batch, including padded rows.

## CUDA Graph

`cudaEventPipelineUseGraph=true` requires both the async pipeline and fixed
batch dispatch. External input-ready and output-consumed events remain outside
the captured graph and gate replay on the owning streams. Graph replay is
functional on SM89, but the 2026-08-09 RTX 4090 D certificate measured eager
submission about 0.8% faster, so eager is the current default recommendation.
SM120 graph replay is not yet a certified production path.

The exact SM89 hardware certificate and retained remote logs are recorded in
`records/plan-runtime-sm89-20260809.md`.
