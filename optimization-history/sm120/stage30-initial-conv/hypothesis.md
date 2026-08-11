# H30: RTX 5090 D fixed-B13 initial-convolution frontend tactic

## Frozen production boundary

- Device/topology target: RTX 5090 D (SM120), fixed 19x19, B13. This stage is
  isolated S1 discovery for a later S2 integration decision; it does not
  generalize to other batches, boards, convolutions, or accumulation modes.
- Ordinal 5 is the initial convolution after the two NHWC input-padding
  kernels. Its exact graph is FP16 NHWC logical NCHW:
  - X `[13,22,19,19]`, strides `[7942,1,418,22]`;
  - W `[768,22,3,3]`, strides `[198,1,66,22]`;
  - Y `[13,768,19,19]`, strides `[277248,1,14592,768]`;
  - cross-correlation, pre/post padding `[1,1]`, stride `[1,1]`, dilation
    `[1,1]`, FP32 compute/intermediate, beta 0.
- Current production path is legacy `cudnnConvolutionForward` with
  `CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM`. Ordinal attribution
  measures 19.569 us isolated median and 21.697 us S2 median, once per forward.
  Its observed launch is grid `296x3`, block 128, 94 registers/thread, and
  81,920 B dynamic shared memory.

## Evidence and mechanism

- On RTX 5080, frontend engine 45 with tile 0/stages 2 measured 36.417 us and
  improved full S2 throughput by 0.461%.
- On RTX 4090 B13, the same explicit plan reduced the local kernel from 31.200
  to 22.048 us and produced a smaller but positive 0.160% whole-network gain.
- The 5090 D legacy tactic is already substantially faster. A frontend plan
  can still help if cuDNN exposes an engine/mainloop with less padding or a
  better SM120 schedule, but engine 45 is a hypothesis rather than a portable
  identity.

The probe will construct the exact graph independently for explicit engine 45
(`TILE_SIZE=0`, `STAGES=2`), heuristic A, heuristic B, and fallback. Every
enumerated config is built and recorded separately, under a 64 MiB workspace
cap, then timed on the same non-blocking stream and device buffers against the
exact legacy call. Plan name tags are retained because they encode engine and
all selected knob values.

## Falsifiable gate

- All graph stages must succeed: validate, operation graph, enumeration,
  per-plan support/build, workspace query, execution, and finite output.
- Each timed plan must agree with legacy within max absolute error 0.05 and
  relative L2 0.02. Exact equality is recorded but not required.
- A plan justifies later S2 integration only if it is at least 10% faster than
  legacy in both control-candidate and candidate-control isolated runs, with
  stable long confirmation. This is deliberately stronger than noise because
  ordinal 5 is only about 0.3% of a full forward and integration has nontrivial
  lifetime/workspace complexity.
- Retain the top three materially distinct passing tags. If no plan clears the
  local margin, stop without changing shared KataGo source or configuration.

## Risks and later gates

- Heuristic source lists can overlap; duplicate tags are not independent
  candidates.
- Frontend graph-call overhead, not only kernel duration, belongs in timing.
- Isolated S1 cannot predict S2 stream phase or coexistence. Any local nominee
  still needs fixed-plan per-server construction, S2 ABBA and reverse order,
  ordinal-aware Nsys/NCU, zero fallback, and full 8192-row accuracy.
- Engine indices and knob semantics are private to the current cuDNN/runtime,
  GPU, and graph. The production route must retain exact guards and legacy
  fallback.
