# H40: Fast FP32 quotient in the accepted fused-FFN SwiGLU epilogue

## Frozen protocol

- RTX 5090 D, fixed B13/19x19, accepted A-reuse FFN geometry and copy schedule.
- Replace only the four source-level FP32 divisions in the unrolled 64-value
  SwiGLU epilogue with `__fdividef`.
- Keep the FP16 exponential, FP32 gate multiply, MMA order, output layout,
  grid, shared memory, and final FP16 rounding unchanged.
- Pre-screen boundary finiteness/error, interleaved S1, and NCU. No local S2
  proxy is allowed. A survivor goes directly to natural whole-graph S2.

## Evidence and prediction

Current PTX/SASS contains 64 precise quotients per thread, with reciprocal
refinement and special-case scaffolding. Stage38 reports math-pipe throttle
`2.169/issue`, wait `3.188/issue`, and only 43.38% tensor elapsed activity.
The candidate should remove the 64 precise-divide slow paths, reduce dynamic
instructions by at least 5%, keep resources at or below 136 registers and
32,768 B dynamic shared memory, and improve S1 by roughly 3-8%.

## Accuracy and stop rules

Boundary output need not be bit-identical, but must be finite with bounded
FP16 error. Any S1 regression or resource/spill regression stops the route.
If the real S2 graph is positive, the established 8,192-row all-head replay
against the frozen FP32 reference is mandatory before acceptance.
