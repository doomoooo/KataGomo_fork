# Stage 24 hypothesis: fixed frontend plan for initial 3x3 convolution

Date: 2026-08-06 UTC

## Frozen target

- RTX 4090 (SM89), exact 19x19, B13, FP16, NHWC, S2.
- Accepted Stage22 configuration is the control; Stage23 sharing remains off.
- Only the trunk initial 3x3 convolution plan may change. All 1x1 nested and
  head convolutions retain their existing paths.

## Pre-implementation evidence and falsifiable test

- Current Nsys identifies one initial-convolution launch per forward/stream:
  `sm80_xmma_fprop_implicit_gemm_f16f16_f16f32_f32_nhwckrsc_nhwc_`
  `tilesize256x64x32_stage3...`, about 32.8us in the complete capture.
- The historical SM120 optimization selected a fixed cuDNN frontend engine,
  but engine numbers and knobs are architecture/cuDNN specific and will not be
  copied without SM89 profiling.
- NCU must first establish the current kernel's launch geometry, occupancy,
  memory/compute balance, and duration. A frontend candidate proceeds only if
  its exact 19x19/B13 kernel or complete initial-conv boundary is measurably
  faster with unchanged output.
- Because this convolution occurs only once per forward, any candidate that is
  flat locally is rejected before full-graph ABBA. A locally positive candidate
  receives forward/reverse Nsys and one 300-iteration locked ABBA screen.

## Result

- Accepted for the frozen target. cuDNN frontend logging confirms the selected
  plan is `eng45_k14=2_k2=0` with 557,056 bytes of workspace; the heuristic
  fallback was not used.
- Baseline NCU (four correct samples) measured a 31.200us median, 254
  registers/thread, 8.32% achieved occupancy, 1.78 waves/SM and no spill.
  Candidate NCU's two correctly targeted grid-296 samples measured 22.048us
  median (-29.33%), 244 registers/thread, 12.70% occupancy, 1.16 waves/SM and
  no spill. Two later grid-111 generic GEMMs accidentally matched the generic
  kernel-name filter and are explicitly excluded.
- Nsys measured the initial convolution at 33.041 -> 24.328us in forward order
  (-26.37%) and 32.533 -> 23.867us in reverse order (-26.64%). Kernel count was
  unchanged. The last-20-forward S2 union was order-sensitive: +1.20% forward
  and -1.64% reverse, for an equal-order mean reduction of 0.224%.
- One locked-2400 300-iteration `A-B-B-A / B-A-A-B` screen had all four
  adjacent control/candidate pairs positive. Pooled medians were 3251.925 ->
  3257.140 nnEval/s (+0.160%). This is consistent with the convolution's small
  critical-path share and the local reduction, so no long throughput run was
  required.
- The 8192-row replay is byte-identical to accepted Stage22 output and therefore
  preserves its complete FP32 error envelope.
- A separate cuDNN library-wide debug-log probe crashed during concurrent model
  construction and is marked diagnostic-only. The narrower frontend logger ran
  successfully and provided the plan identity above.
