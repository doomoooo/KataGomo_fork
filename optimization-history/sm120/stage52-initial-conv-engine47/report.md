# Stage 52 result: retain exact-B13 cuDNN engine 47 default-off

Target: RTX 5090 D, exact 19x19 B13, FP16/NHWC.  Control is commit
`91f0e80`; `cudaUseInitialConvFrontend` is false by default.

The implementation builds a per-handle cuDNN frontend convolution graph only
for `[13,22,19,19] * [768,22,3,3] -> [13,768,19,19]`, queries heuristic A,
and selects the complete tag
`eng47_k2=2_k6=1_k13=1_k14=0_k22=2`.  It never relies on enumeration index.
Missing/build-failed plans fall back to legacy.  B1--B12, non-target shapes,
layouts, precision, GPUs, and accumulate mode always use legacy.

## Mechanism and correctness

- B13 26-row replay: byte-identical to control; exact engine tag logged.
- B12 26-row fallback replay: byte-identical; no frontend log.
- Short S1 Nsys: legacy main calls `28 -> 24`, padding calls `56 -> 48`, and
  four B13 engine-47 kernels appear at about 16.24 us each.  Thus every B13
  boundary changes from two padding kernels plus legacy main to one kernel,
  while B1--B12 warmup remains unchanged.
- Existing targeted NCU: legacy 94 registers/thread and 81.92 KiB dynamic
  shared memory, one CTA/SM and 8.32% achieved occupancy; engine 47 uses 128
  registers but only 4.10 KiB shared memory, four CTAs/SM register limit and
  26.07% achieved occupancy.  Workspace is zero.
- Stable fixed-B13 8,192-row replay is byte-identical to control, SHA256
  `3a7329716ccef8cababb2ca71ebe0734ddd6b9f6f1a90411d3035b4dac7fe7ff`.
  All-head metrics versus FP32 match the current accepted graph.

## Whole-forward throughput

Natural S2, 100/20 ABBA plus reverse BAAB:

| order | control (nnEval/s) | candidate (nnEval/s) | delta |
|---|---:|---:|---:|
| forward | 3987.223 | 3985.034 | -0.055% |
| reverse | 3979.408 | 3984.179 | +0.120% |
| pooled | 3983.315 | 3984.606 | +0.032% |

S1, 400/40 confirmation:

| order | control (nnEval/s) | candidate (nnEval/s) | delta |
|---|---:|---:|---:|
| forward | 3330.807 | 3336.879 | +0.182% |
| reverse | 3325.843 | 3332.609 | +0.203% |
| pooled | 3328.325 | 3334.744 | +0.193% |

## Decision

Retain behind `cudaUseInitialConvFrontend=false`.  The candidate is exact at
the whole-network output, deletes two launches, sharply reduces the limiting
shared-memory footprint, and improves both S1 order aggregates.  Natural S2 is
below resolution with conflicting order signs, so it is not promoted and no
400-run S2 confirmation is justified.  The default/accepted graph remains
Stage 47 and its complete profile remains the priority source.
