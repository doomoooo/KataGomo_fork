# Stage 20: fixed-B13 fused FFN

Status: accepted for RTX 5090D B13/S2 (2026-08-06 UTC).

The accepted TileLang kernel is M128-N64-K32, two pipeline stages, 128
threads, minimum three blocks per SM, and FP16 MMA accumulation. It loads the
C384 input tile once, computes the independent linear/gate C1152 projections,
and writes the SwiGLU result directly. Unsupported shapes fall back.

## Selection

FP32 accumulation was rejected despite a faster isolated boundary because it
used 168 registers/thread and unbalanced the two server streams; whole-network
throughput regressed 4.74%. FP16 accumulation reduced the dual-stream micro
boundary from 91.38 us unfused to 65.34 us fused and restored balanced
scheduling. Direct whole-network comparisons also rejected K64/S1 and K32/S3
in favor of K32/S2.

## Whole network

The 1,000-iteration A-B-B-A result was:

| mode | values (nnEval/s) | mean |
|---|---:|---:|
| control | 3556.172 / 3572.987 | 3564.580 |
| fused FFN | 3666.963 / 3646.556 | 3656.759 |

Mean improvement is 2.586%.

## Profiler and accuracy

Nsys measured the fused kernel at 46.83 us under the mixed two-stream load,
146 registers/thread and 32 KiB dynamic shared memory. Main stream spans were
358.64 and 356.63 ms. The 8,192-row comparison passed every established gate:
policy top-1 99.7925%, optimistic top-1 99.6094%, probability RMSE 1.012e-4,
total variation 0.001697, outcome RMSE 0.002291, and ownership sigmoid RMSE
0.000245.

Artifacts are in this directory and `accuracy/`.
