# Stage 52 hypothesis: exact-B13 initial-conv cuDNN engine 47

## Frozen target and profiler evidence

- Control: commit `91f0e80`, option disabled, RTX 5090 D, exact 19x19 B13,
  FP16/NHWC, natural S2.
- Accepted Stage 47 full profile ordinals 3--5 are two NHWC padding kernels
  plus the legacy initial-convolution kernel.  The S1 boundary is about
  22.3 us and its S2 excess is about 5.3 us/forward.
- Stage 30 enumerated all 36 unique supported cuDNN frontend plans for the
  exact `[13,22,19,19] * [768,22,3,3]` shape.  Exact tag
  `eng47_k2=2_k6=1_k13=1_k14=0_k22=2` reduced the CUDA-event boundary
  `29.150 -> 16.172 us` and the Nsys kernel boundary about 27.7%.
- Existing NCU shows the legacy main kernel at 94 registers and 81.92 KiB
  dynamic shared memory (one CTA/SM, 8.32% achieved occupancy); engine 47 is
  128 registers but only 4.10 KiB shared memory (four CTAs/SM register limit,
  26.07% achieved occupancy), with no workspace.

## Single mechanism

Build a per-handle cuDNN frontend graph only for the exact target and query
heuristic A by the complete plan tag, never by enumeration index.  For runtime
B13 and beta zero, execute that plan with the existing input, NHWC filter, and
output buffers.  If the exact tag cannot be validated/built, fail closed to
the legacy convolution.  B1--B12 and every other device, shape, precision,
layout, or accumulate mode remain on the legacy path.  The option defaults
false during evaluation.

## Gates

1. Compile and run fixed-B13 plus B12 correctness smoke; the frontend log must
   occur only for B13.  Compare all raw outputs before performance testing.
2. Short Nsys must show ordinals 3--5 replaced by the single engine-47 kernel,
   with no unexpected synchronization or workspace activity.
3. Run natural-S2 100/20 ABBA plus reverse BAAB.  Promote only when both order
   aggregates are non-negative and the pooled mean is positive; confirm at
   400/40.
4. If S2 is inconclusive, retain default-off only when fixed-B13 correctness,
   strict profiler resource/work reduction, and both S1 order aggregates are
   positive.
5. Default-on acceptance requires stable fixed-B13 8,192-row all-head accuracy,
   a fresh full S2 Nsys, complete S1 NCU, history update, and one commit.
