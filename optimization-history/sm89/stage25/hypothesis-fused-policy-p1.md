# Stage 25 hypothesis: fuse the policy P1 conversion, global bias, and BN+SiLU

Date: 2026-08-06 UTC

## Frozen target

- RTX 4090 (SM89), exact 19x19, B13, FP16, NHWC, S2.
- Accepted Stage24 configuration is the control.
- Only the policy P1 path may change. All trunk, value-head, pass-policy, and
  non-target shape behavior must retain the existing implementation.

## Pre-implementation evidence and falsifiable test

- Current Nsys shows one consecutive three-kernel P1 boundary per
  forward/stream for the `13x361x96` tensor:
  `copyFromHalfKernel` -> `addNCBiasInplaceNHWCKernel` ->
  `applyCScaleBiasNHWCSiluKernel`.
- Across 64 complete observed boundaries, summed kernel duration averages
  8.217us and the elapsed boundary averages 11.382us. The path materializes a
  1.72MiB FP32 map between kernels and performs redundant global-memory
  round-trips.
- NCU must show the unique bias-add and BN+SiLU kernels are low-arithmetic
  pointwise work whose cost is compatible with launch and memory traffic. If
  they are compute-saturated or the boundary is not consecutive, fusion is
  rejected before implementation.
- The candidate will preserve the exact operation order in FP32:
  `half_to_float`, global-bias add, BN scale/bias, then SiLU. It may only run
  for B13/19x19/FP16/NHWC/P1-C96 with no mask; the existing three-kernel path is
  the fallback.
- After implementation, NCU must show one replacement kernel without spill,
  Nsys must show three launches replaced by one and a smaller local boundary,
  and one locked 300-iteration forward/reverse ABBA must be non-negative before
  full 8192-row accuracy replay.

## Result

- Accepted as a small fixed-shape optimization. Baseline NCU measured the
  bias-add at 2.88/2.94us and BN+SiLU at 3.17/3.14us. Both were L2-hot,
  approximately 18-20% compute SOL and 32-34% memory SOL, with 16 registers and
  no spill, supporting a launch/intermediate-traffic fusion.
- Candidate v1 used 96-thread CTAs and measured 4.38-4.48us with only 29%
  achieved occupancy; its short smoke was negative. NCU explained the geometry
  issue, so v2 matched the original 96x5 thread layout. V2 measured
  3.20-3.23us, 66.5-68.3% occupancy, 20 registers and no spill.
- Nsys confirmed 311 -> 309 kernels per forward/stream. The target summed
  boundary changed from 8.616 -> 3.115us in forward order (-63.85%) and
  9.073 -> 3.344us in reverse order (-63.15%). Last-20-forward S2 union changed
  164.069 -> 161.169ms (-1.77%) and 164.659 -> 163.750ms (-0.55%).
- One locked-2400 300-iteration forward/reverse ABBA measured pooled medians
  3220.066 -> 3222.436 nnEval/s (+0.074%), with 3/4 adjacent pairs positive.
  Forward aggregate was +0.760%; reverse aggregate was -0.042% (effectively
  flat), so the accepted end-to-end benefit is explicitly low-confidence and
  close to the theoretical 5-6us-per-forward ceiling.
- The full 8192-row replay is byte-identical to Stage24 and preserves the same
  FP32 error envelope.
- One initial combined-regex NCU invocation returned application code 11 before
  collecting a sample. It is retained as a no-sample diagnostic failure; the
  two exact-name baseline reports completed normally with four total samples.
