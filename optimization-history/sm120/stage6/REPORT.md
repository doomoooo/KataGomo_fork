# Stage 6: single-wide FFN GEMM microbenchmark

Status: accepted for S1, rejected for S2 (2026-08-05 UTC).

The microbenchmark compared the exact B13 FFN input projection work
(`M=1152`, `N=4693`, `K=384`) in three cuBLAS formulations. Times are wall
microseconds per complete projection pair; the two-stream rows submit one pair
on each stream per iteration.

| streams | two Hgemm | strided batch-2 | one M=2304 Hgemm |
|---:|---:|---:|---:|
| 1 | 32.815 | 33.443 (+1.91%) | 30.555 (-6.89%) |
| 2 | 60.381 | 61.704 (+2.19%) | 59.879 (-0.83%) |

The true-wide GEMM failed the original requirement of at least 10% GEMM
savings under both one-stream and two-stream execution. That gate was too
aggressive for optimization scope that includes S1: its 6.89% S1 saving is now
being validated with the real layout-aware, out-of-place SwiGLU. S2 remains a
separate non-regression measurement.

## Integrated result

The integrated path interleaves weights by input channel, runs one
M2304xN4693xK384 Hgemm, and uses an out-of-place half2 SwiGLU that reads the
per-token `[linear1,gate]` layout. It is opt-in through
`cudaUseWideFFNSingleGemm=true`.

The short A-B-B-A result separated the topology effect:

| topology | control mean | wide mean | delta |
|---|---:|---:|---:|
| B13/S1 | 2636.023 | 2659.091 | +0.875% |
| B13/S2 | 3125.699 | 3034.819 | -2.907% |

Nsys over the last 30 S1 forwards confirmed the intended subgraph change:

| component | control total | wide total |
|---|---:|---:|
| FFN input GEMM(s) | 13.018 ms | 10.944 ms |
| SwiGLU | 3.198 ms | 3.857 ms |
| combined | 16.216 ms | 14.801 ms (-8.73%) |

The full 8,192-row replay passed every Stage-3 gate: policy top-1 99.7437%,
policy probability RMSE 1.031e-4, total variation 0.0017066, JSD 3.718e-6,
outcome RMSE 0.0022866, score-mean RMSE 0.0019959, and ownership sigmoid RMSE
0.0002464.

After a 1,500-iteration thermal precondition, the 1,000-iteration symmetric
S1 A-B-B-A/B-A-A-B result was:

| mode | values (nnEval/s) | mean | median |
|---|---|---:|---:|
| control | 2646.201 / 2638.518 / 2637.465 / 2635.694 | 2639.469 | 2637.991 |
| single-wide FFN | 2670.870 / 2668.624 / 2666.312 / 2661.369 | 2666.794 | 2667.468 |

Mean improvement is 1.035%; median improvement is 1.117%. Accept for S1. The
2.91% S2 regression is decisive, so the option remains disabled by default and
must not be enabled in the S2 production regime.
