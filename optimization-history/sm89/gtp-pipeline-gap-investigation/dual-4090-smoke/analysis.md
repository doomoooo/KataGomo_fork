# Dual RTX 4090 async-pipeline smoke test

Date: 2026-08-06 19:32-19:36 UTC

## Device mapping and locking

The CUDA runtime enumerates the two RTX 4090 cards as devices 0 and 1. These
map to `nvidia-smi` devices 0 and 2. Every test acquired both cards together
with `gpu-lock with --gpu 0,1`; the RTX 5090 D was not used.

The production-shape test used exact 19x19, FP16 NHWC, B13, eager event
submission, and CUDA Graph disabled. The primary dual-card topology was four
NN slots `[0,0,1,1]`, preserving S2 independently on each 4090.

## Result: reproducible fatal error

The four-slot dual-card GTP benchmark fails during initial search with:

```text
CUDA event pipeline scheduler failed: CUBLAS Error, for model.linear_global
... cublasHgemm(...) ... CUBLAS_STATUS_INTERNAL_ERROR
```

A smaller one-stream-per-card `[0,1]` test reproduced the same failure:

| path | exit status | result |
|---|---:|---|
| async event pipeline | 1 | `CUBLAS_STATUS_INTERNAL_ERROR` |
| existing synchronous path | 0 | completed 2/2 positions |

The synchronous result is only a functional control, not a throughput result.
Its very short final line was 4836.64 visits/s, 4641.40 real nnEval/s, and
386.04 batches/s.

## Correctness controls

- Current GPU-0 S2 output versus the historical eager S2 output: all reported
  numerical errors exactly zero.
- GPU-1 S2 output versus GPU-0 S2 output: all reported numerical errors exactly
  zero.
- Four slots on GPU 0 and mixed `[0,0,1,1]` both produced the same small
  non-bit-exact signature relative to S2: policy top-1 remained 100%, policy
  probability RMSE was 0.00014249, and value outcome RMSE was 0.00189625.

The 26-row replay is therefore insufficient as a multi-device execution smoke:
the mixed result has the same signature as four slots on one device and does
not trigger the failing second-device submission path. Real GTP does.

## Root cause

The event scheduler creates one persistent host submit thread per slot in
`nneval.cpp`. The worker lambda calls `launchEventPipelineInference()` without
first selecting `slotPtr->gpuIdx` with `cudaSetDevice`.

CUDA device selection currently occurs only while the stream and compute
handle are created. CUDA current-device state is per host thread, so that setup
does not propagate into a newly created submit worker. The second-card worker
therefore invokes a cuBLAS handle and streams created on CUDA device 1 while
its host thread remains on the default CUDA device 0. The first cuBLAS boundary
then reports `CUBLAS_STATUS_INTERNAL_ERROR`.

The immediate fault is at the submit worker, but a robust fix should audit the
whole per-handle API boundary rather than add only one isolated call:

- bind every submit worker to its slot GPU before any CUDA call;
- store the owning device in `ComputeHandle` and select/guard that device for
  event queries, H2D/D2H enqueue, output completion, and destruction;
- make compute-stream destruction device-aware as well;
- retain a `[0,1]` async-vs-sync GTP regression test that actually forces work
  onto both devices.

No production source was modified during this diagnostic.

## Artifacts

- `gtp.raw`: four-slot `[0,0,1,1]` fatal log
- `dual-vs-single-current.json`: mixed four-slot replay comparison
- `second-gpu-vs-first-gpu.json`: per-card S2 equality control
- `four-vs-two-slots-same-gpu.json`: slot-count control
- sibling directory `dual-4090-minimal-repro/`: async failure and synchronous
  success logs/status files
- parent scripts `run-dual-4090-smoke.sh` and
  `run-dual-4090-minimal-repro.sh`
