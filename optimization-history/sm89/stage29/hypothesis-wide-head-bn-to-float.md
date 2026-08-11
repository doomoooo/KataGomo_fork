# Stage 29 hypothesis: wide-head BN directly emits FP32

Date: 2026-08-06 UTC

## Frozen target

- RTX 4090 (SM89), exact 19x19, B13, FP16, NHWC, S1.
- Accepted Stage28 S1 configuration is the control.
- Only the policy g1 and value v1 wide-slice BN/copy boundaries may change.

## Pre-implementation evidence and falsifiable test

- Stage28 Nsys observes policy g1 strided BN plus half-to-float copy at
  2.686+2.065=4.751us and value v1 at 4.166+3.181=7.347us. The four launches
  total 12.098us per forward.
- Stage28 NCU measures the two strided BN kernels at 3.20us and 4.45us with
  23.78%/31.70% compute SOL, 18 registers/thread and zero spill. NCU must also
  confirm the two copy kernels are traffic/launch work rather than hidden
  compute before implementation.
- The candidate reads g1/v1 directly from their C384 wide-head slices. Policy
  g1 writes only the contiguous FP32 pooling input. Value v1 writes both the
  original rounded contiguous half ownership input and its exact FP32
  conversion. It must preserve half FMA, float SiLU, half rounding and
  half-to-float conversion order.
- Candidate NCU must show zero spill. Nsys must remove exactly two copy launches
  and reduce both complete boundaries. Only then may one locked-2400 S1
  100-iteration forward/reverse ABBA decide whole-graph value.
- If performance passes, the 8192-row all-head FP32 replay is mandatory.

## Rejection and reopen conditions

- Reject if either local boundary grows, spill appears, or the short ABBA is
  negative in both orders.
- Reopen only if pooling or ownership convolution is fused into the producer.

## Result

- Accepted with low magnitude for exact 19x19/B13/S1 and enabled together with
  the Stage28 wide head. The prior Stage26 S2 rejection remains unchanged.
- Baseline copy NCU measured g1/v1 at 2.72/3.62us, only 11.71%/16.78%
  compute SOL, 98.43%/99.21% L2 hit and zero spill. Together with the Stage28
  BN samples, the baseline boundary was 13.99us.
- Candidate NCU measured g1/v1 at 3.20/4.86us, 20 registers/thread,
  63.43-66.50% achieved occupancy and zero local/shared spill. The summed
  boundary was 8.06us (-42.39%).
- Forward/reverse Nsys removed exactly two copy launches per forward and
  measured 12.089->7.133us (-41.00%) and 12.064->7.143us (-40.79%). Whole-graph
  short runs improved 2435.530->2447.518 (+0.492%) and
  2436.735->2438.783 (+0.084%).
- Locked-2400 S1 100-iteration forward/reverse ABBA measured pooled medians
  2473.996->2475.926 nnEval/s (+0.078%). All four adjacent pairs were positive
  (+0.117%, +0.078%, +0.059%, +0.078%). No longer throughput run was used.
- The 8192-row candidate replay is byte-identical to Stage28 for every raw
  output, so its established all-head FP32 error envelope is unchanged.
