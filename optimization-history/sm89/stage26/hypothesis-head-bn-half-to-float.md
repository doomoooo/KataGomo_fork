# Stage 26 hypothesis: head BN consumes FP16 and emits FP32 directly

Date: 2026-08-06 UTC

## Frozen target

- RTX 4090 (SM89), exact 19x19, B13, FP16, NHWC, S2.
- Accepted Stage25 configuration is the control.
- Only the policy g1 and value v1 head BN/copy boundaries may change.

## Pre-implementation evidence and falsifiable test

- Policy g1 is a consecutive C96 half BN+SiLU followed by a 450,528-element
  half-to-float copy. The float result is consumed by pooling; the half BN
  output has no other consumer.
- Value v1 is a consecutive C192 half BN+SiLU followed by a 901,056-element
  half-to-float copy. Its half result is also consumed by the ownership
  convolution, so a correct fusion must write both the original rounded half
  result and its exact float conversion.
- Forward/reverse Nsys observes mean summed boundaries of 6.102/6.269us for
  policy g1 and 7.447/7.730us for value v1. In every complete observation the
  copy immediately follows the BN kernel.
- NCU must show the two BN kernels have pointwise/launch or memory headroom and
  no hidden compute saturation. The candidate must preserve the official half
  FMA, half rounding after SiLU, and `__half2float` conversion order.
- After implementation, NCU must show no spill, Nsys must remove exactly two
  copy launches per forward/stream and reduce both local boundaries, and a
  single locked 300-iteration forward/reverse ABBA must be non-negative before
  full accuracy replay.

## Result

- Rejected and left behind default-disabled `cudaUseHeadBNHalfToFloat`.
- Baseline NCU measured C96 at 3.10us and C192 at 4.61us. Compute and memory
  SOL were at most 27.5%, with 16 registers and no spill, so the local fusion
  mechanism was supported.
- Candidate NCU measured C96 at 3.20us and dual-output C192 at 4.74us, with
  18/19 registers, 63.5-67.9% achieved occupancy and no spill. Relative to each
  complete BN+copy boundary this was locally faster.
- Nsys confirmed 309 -> 307 kernels per forward/stream. Policy g1 improved by
  48.9%/46.4% and value v1 by 39.3%/36.7% in forward/reverse order. However,
  last-20-forward S2 union regressed 163.848 -> 165.508ms (+1.01%) forward and
  improved 162.575 -> 161.541ms (-0.64%) reverse, an order-sensitive conflict.
- The single locked-2400 300-iteration ABBA was decisively negative: all four
  adjacent pairs regressed, pooled medians were 3246.156 -> 3205.214 nnEval/s
  (-1.261%), forward -1.215% and reverse -1.279%.
- Full accuracy replay was not run because performance failed the acceptance
  gate. The candidate explicitly preserved half rounding and the value-head
  ownership half output, but that implementation property is not a substitute
  for replay evidence.
- Reopen only if policy/value pooling and ownership projection are fused into a
  wider head boundary that changes the present dual-stream scheduling tradeoff.
