# Stage 7 hypothesis: SM89 FlashAttention for exact 19x19 B13

Date: 2026-08-05 UTC

## Frozen target

- GPU: CUDA device 0, RTX 4090 (SM89), protected by `gpu-lock`.
- Shape: FP16 BSHD `[13,361,12,32]`, non-causal self attention, no mask,
  two independent server streams.
- Control: accepted stage-6 exact-19x19 path using cuDNN frontend SDPA.
- Candidate source: Dao-AILab/flash-attention commit
  `5835c733e7e9c07606b045255768e8a7e9e851bd`, BSD-3-Clause, CUTLASS
  submodule `7127592069c2fe01b041e174ba4345ef9b279671`, compiled by CUDA
  13.2.86 into SM89-only AOT SASS.

Upstream `main` at `c68c592fd9da1e40a4fb0b56229caae6754ac5c9` was audited.
The commits after the pinned candidate do not change the SM8x forward kernel or
its Ada tile policy, so updating them would add unrelated SM100/ROCm changes.

## Evidence -> mechanism -> predicted change

The exact-B13 Nsys trace records 2178 cuDNN SDPA launches at an average
`60.13 us` in the dual-stream trace. On one representative B13 launch, NCU
reports:

- `54.98 us`, grid `(3,12,13)`, block `(256,1,1)`;
- `16.61%` achieved occupancy and 7.97 active warps/SM;
- one CTA/SM due to the register limit (shared memory would permit two);
- `32.26%` compute and `20.05%` DRAM throughput, with no spilling;
- 5.9 of 10.1 cycles between issued instructions stalled for a math pipeline,
  or 58.5% of the issue interval.

This is a latency/occupancy and tile-quantization limit rather than bandwidth
saturation. The stock FlashAttention SM86/89 policy for rounded head dimension
64 uses a `128x112` tile, four warps and one stage. It is a plausible lower-
thread-count alternative to cuDNN's 256-thread kernel. The falsifiable first
prediction is an isolated latency below `50 us`; only then will it be connected
to KataGo. A connected candidate must improve B13/S2 whole-network throughput
by at least 1% beyond dispersion.

## Risks and validation

- The kernel is specialized only for B13/S361/H12/D32 FP16 with no mask. All
  other shapes and masked calls retain cuDNN.
- The microbenchmark checks BSHD strides and one complete batch/head against an
  FP32 CPU softmax reference before its timing is accepted.
- Integration must preserve the caller's CUDA stream and allocate LSE scratch
  outside the repeated attention call.
- Any accepted candidate must pass all 8192 fixed 19x19 rows and every output
  head directly against the FP32 reference, then pass Nsys/NCU and forward ABBA
  plus reverse-order testing.

The first implementation tests the upstream tile unchanged. If it is close but
register- or wave-limited, the next single variable is an SM89/B13 tile sweep;
the current library version is not a reason to stop that sweep.
