# Stage 23 hypothesis: share ordinary matmul weights across S2

Date: 2026-08-06 UTC

## Frozen target

- RTX 4090 (SM89), exact 19x19, B13, FP16, S2.
- Accepted Stage22 configuration is the control.
- This stage changes only `cudaShareModelWeights=false/true`.
- Only `Sm89MatMul` device allocations are shared. Specialized QKV, FFN,
  pre/post projection, convolution, normalization, and bias buffers remain
  private per server.

## Pre-implementation evidence

- Current Nsys has 33 C384 attention out-projection GEMMs per forward. Across
  the complete capture they consume about 72ms and are the largest remaining
  ordinary-matmul category.
- Representative NCU samples are 11.65-11.81us with only 111 CTAs (0.43
  waves/SM), 186 registers/thread, and about 8.3% achieved occupancy. L2 hit is
  already 99.25-99.36%, so sharing is not expected to remove substantial DRAM
  traffic from an isolated hot invocation.
- Two server models currently allocate separate physical buffers for identical
  descriptor weights. Sharing removes the second ordinary-matmul allocation,
  which can reduce live S2 L2 working-set pressure even when an isolated replay
  sees a hot weight matrix. The expected gain is therefore small and must be
  established only by concurrent Nsys and full-graph ABBA.

## Mechanism and falsifiable tests

Use a process-local, mutex-protected weak cache keyed by descriptor address,
CUDA device, and precision. The second server receives a shared-ownership view
of the first server's allocation; the last owner frees it. With the switch off,
the existing per-instance allocation path is unchanged.

1. Startup Nsys must show fewer H2D copies/bytes and allocations with sharing,
   proving physical deduplication. Kernel count, geometry, arithmetic, copies in
   the timed region, and synchronization must not change.
2. NCU rechecks the representative ordinary GEMM. A large isolated improvement
   is not expected; any duration/L2 change must be interpreted with replay
   serialization in mind.
3. Concurrent steady-state Nsys must not increase the full S2 kernel union.
4. Only a positive profiler screen proceeds to locked forward/reverse ABBA and
   the complete 8192-row replay, which should be byte-identical to Stage22.

The route is rejected if cache contention, construction synchronization, or
weight lifetime handling introduces instability or if the S2 signal is absent.

## Result

- Startup Nsys proved the mechanism: 476 `cudaMalloc` calls and 476 H2D copies
  were removed, totaling 254,435,328 fewer H2D bytes.
- Representative NCU remained flat: median 11.680us -> 11.725us, with the same
  186 registers/thread, 0.43 waves/SM, about 8.30% achieved occupancy, and no
  spills. The ordinary GEMM was already about 99.3% L2-hit.
- Forward and reverse Nsys captures kept exactly 311 kernels/forward/stream and
  unchanged launch geometry. Their union results had opposite signs, while
  summed kernel time improved only 0.21-0.27%.
- Three 500-iteration ABBA rounds were bimodal and initially looked positive.
  The completed 1500-iteration forward/reverse ABBA instead regressed in both
  directions: pooled 3185.972 -> 3163.154 nnEval/s (-0.716%). A second long
  round was stopped at the user's request and is retained as contaminated.

Decision: rejected on RTX 4090 exact-19x19/B13/S2. The experimental switch is
kept for reproducibility but remains disabled by default. No full accuracy
replay was run because kernel arithmetic and the executed graph are unchanged
and performance already failed the acceptance condition.
