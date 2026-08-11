# H24: Reopen FP16 cuBLASLt for fixed-B13 single-stream MatMul layers

## Frozen protocol

- GPU: RTX 5090 D, CUDA device 2, exclusively held by `gpu-lock`.
- Model: `/workspace/models/b11c768h12nbt3tflrs-fson-silu.bin.gz`.
- Shape: fixed 19x19, batch 13 only.
- Backend/precision: CUDA SM120, FP16.
- Primary topology: one NN server thread and one CUDA stream.
- Control: the current Stage-23 S1-specific fused FFN and wide-QKV schedules.
- Candidate variable: route only remaining `MatMulLayer` calls whose matrix N is
  13 or `13*361` through a cached cuBLASLt FP16-compute plan.

## Evidence

- Stage 2 measured about `3136 -> 3414 nnEval/s` from the FP16-compute
  cuBLASLt route on the earlier backend, but rejected it after `replaynn`
  reported `invalid resource handle`.
- Stage 3 independently traced that failure to the FA4 CuTe DSL module being
  unloaded when replay warmup models were destroyed. FA4 is now loaded once
  for process lifetime, and full replay passes.
- The current S1 Nsys trace still attributes about 23.6% of forward time to
  generic cuBLAS GEMMs after fused FFN and QKV are removed.
- An exact residual-GEMM probe found no cuBLASLt advantage for beta=1 linear2
  or attention out-projection. This hypothesis therefore targets only the
  beta=0 `MatMulLayer` boundary that produced the old positive result.

## Mechanism and prediction

Legacy `cublasHgemm` fixes library routing for every shape. cuBLASLt exposes
multiple supported algorithms, including SM120-native kernels. During untimed
model warmup, each NN server can measure returned heuristics for its exact
matrix shape and cache the fastest valid plan. The timed forward then pays only
the cached `cublasLtMatmul` launch.

Prediction: remaining beta=0 spatial/head GEMM time and serial forward time
will fall. A whole-network improvement of at least 0.5% in short S1 ABBA is
required before long confirmation. The old 7% gain is an upper bound because
the largest FFN/QKV GEMMs are now handled by AOT kernels.

## Risks

- FP16 accumulation may change rounding and requires the full 8,192-row,
  all-head comparison against the fixed FP32 reference.
- Heuristic order is not a performance guarantee, so plans are selected by
  short CUDA-event measurements during warmup, not by list position.
- Plan construction, tuning, event synchronization, and workspace allocation
  must occur before the benchmark timing interval.
- A cuBLASLt algorithm can be faster alone but worse under S2 contention. S1
  is the primary decision topology; S2 is measured separately only after S1
  acceptance.

## Validation and decision

1. Build and fixed-B13 smoke; confirm the activation log.
2. Short B13/S1 ABBA with only `cudaUseProjectionGemmLt` changing.
3. If positive by at least 0.5%, Nsys the control/candidate and confirm plan
   construction is outside the timed interval and targeted GEMMs changed.
4. Long symmetric S1 confirmation.
5. Full 8,192-row replay against the FP32 reference.
6. Measure S2 separately, without using it to veto an accepted S1 route.

Reject on crash, numerical gate failure, or less than 0.5% confirmed S1 gain.
Keep all artifacts and record the stale Stage-2 rejection explicitly.
