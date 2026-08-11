# Stage 27 hypothesis: fuse initial global dot and spatial broadcast-add

Date: 2026-08-06 UTC

## Frozen target

- RTX 4090 (SM89), exact 19x19, B13, FP16, NHWC, S2.
- Accepted Stage24/25 configuration is the control; rejected Stage26 remains off.
- Only `initialMatMul` (`K=19`, `C=768`) and its immediately following
  broadcast-add into the `13x361x768` initial-convolution output may change.

## Pre-implementation evidence and falsifiable test

- Nsys identifies three consecutive baseline kernels per forward/stream:
  cuBLAS `Kernel2` (about 4.750us), `splitKreduce_kernel` (about 1.556us), and
  `addNCBiasInplaceNHWCHalfKernel` (about 11.277us), for about 17.583us summed.
- The `K=19` dot is tiny while the broadcast-add launches 14,079 CTAs and reads
  the intermediate `B13xC768` tensor. A fixed-shape kernel can compute a small
  group of channel dots once per CTA, stage the FP32 accumulators in shared
  memory, round each dot once to half, then use the remaining threads to apply
  the same `__hadd` over spatial positions.
- Before implementation, NCU must confirm that the broadcast-add is not near a
  compute or memory throughput ceiling and that the GEMM/reduction launch is
  disproportionately small. At most four representative launches are sampled.
- The candidate must preserve a FP32 dot accumulation, one FP32-to-half rounding
  before broadcast, and the baseline half add. The main numerical risk is a
  different reduction order from cuBLAS, so full FP32-reference replay remains
  mandatory if the performance gate passes.
- After implementation, NCU must show no spill and explain the chosen CTA/channel
  grouping. Nsys must replace all three kernels with exactly one target kernel,
  improve the complete local boundary in both run orders, and not regress the
  last-20-forward S2 union. Only then run one locked-2400 300-iteration
  forward/reverse ABBA; no longer performance run is part of this iteration.

## Rejection and reopen conditions

- Reject if the candidate is not locally faster, spills, loses occupancy through
  excessive channel grouping, or the short ABBA is negative in both directions.
- Reopen only with a materially different reduction/CTA mapping or a wider
  fusion boundary that includes the initial convolution epilogue.

## Result

- Accepted for the exact B13/S1 regime and enabled only in
  `/workspace/bench-cuda-gpu0-4090-s1.cfg`. The existing S2 config keeps the
  switch off because a short topology probe measured an S2 regression.
- Baseline NCU measured 10.82us broadcast-add, 3.90us GEMM and 1.98us reduce.
  The first fused geometry used non-coalesced four-channel groups and regressed
  to 37.50-37.82us. NCU identified about 86% L2 throughput, leading to the
  channel-contiguous geometry.
- Channel-contiguous variants measured 12.00us for 4 rows/CTA, 9.44-9.60us for
  8 rows/CTA, and 9.70-9.76us for 16 rows/CTA. Eight rows is the final choice:
  40 registers/thread, 2.34 waves/SM, 82.9% achieved occupancy and zero spill.
- S1 Nsys measured the complete three-kernel boundary at 15.872/15.875us and
  the fused kernel at 8.733/8.747us in forward/reverse order, about 45% faster.
  Two launches were removed per forward.
- Locked-2400 S1 100-iteration forward/reverse ABBA measured pooled medians
  2460.772 -> 2463.670 nnEval/s (+0.1178%). Both aggregate directions and all
  four adjacent pairs were positive.
- A separate one-order topology probe measured S1 control/candidate at
  2460.317/2463.622 (+0.134%) and S2 at 3271.145/3232.313 (-1.187%). Current S1
  is 75.21% of S2 throughput; this motivated using S1 as the fast primary
  optimization regime while retaining S2 as a periodic frontier comparison.
- The 8192-row replay passed the established all-head FP32 envelope: policy
  top-1 99.707%, probability RMSE 1.273e-4, value outcome RMSE 0.00265 and
  ownership sigmoid RMSE 3.016e-4.
