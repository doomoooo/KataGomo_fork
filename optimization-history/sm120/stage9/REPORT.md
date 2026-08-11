# Stage 9: GEMM beta residual fusion

Status: accepted for S1 and S2 (2026-08-05 UTC).

For fixed full-board FP16 transformer projections, attention out-proj and FFN
linear2 now write directly to trunk using Hgemm beta=1. This removes the
temporary-to-trunk residual kernel. Masked or unsupported shapes retain the
official path.

Short A-B-B-A showed +3.59% at B13/S1 and +3.16% at B13/S2. The full 8,192-row
replay passed every Stage-3 full-FP32 gate; policy top-1 was 99.7437%, policy
RMSE 1.031e-4, TV 0.0017066, outcome RMSE 0.0022867, score-mean RMSE 0.0019960,
and ownership sigmoid RMSE 0.0002464.

S2 Nsys over the last 60 forward instances reported:

| component | control | fused |
|---|---:|---:|
| residual kernel | 1,440 launches / 8.533 ms | 0 launches |
| grid-148 GEMMs | 64.746 ms | 69.283 ms |
| direct net of those rows | 73.279 ms | 69.283 ms (-5.45%) |

The 1,000-iteration symmetric S2 A-B-B-A/B-A-A-B result was:

| mode | values (nnEval/s) | mean | median |
|---|---|---:|---:|
| control | 3162.017 / 3126.374 / 3115.859 / 3110.265 | 3128.629 | 3121.117 |
| fused | 3239.866 / 3228.068 / 3210.479 / 3206.197 | 3221.153 | 3219.274 |

Mean improvement is 2.957%; median improvement is 3.145%. The first symmetric
block improves about 2.86% and the second about 3.06%, despite monotonic
platform drift. Accept and enable by default on the gated SM120 path.
