# Stage 7: strided-batched QKV projection

Status: accepted for S1, rejected for S2 (2026-08-05 UTC).

For the fixed transformer shape, the candidate concatenates the three
M384xK384 FP16 weight matrices once and submits Q/K/V as one
`cublasHgemmStridedBatched` call with batch count 3 and shared-input stride 0.
Q/K/V outputs remain three contiguous buffers, so RoPE, FA4, and every later
operation are unchanged. The option is `cudaUseQKVStridedSm120=true` and
defaults false.

## Evidence

Isolated B13 GEMM work:

| streams | three Hgemm | strided batch-3 | delta |
|---:|---:|---:|---:|
| 1 | 22.895 us | 19.823 us | -13.42% |
| 2 | 32.014 us | 31.464 us | -1.72% |

Short whole-network A-B-B-A:

| topology | control mean | QKV mean | delta |
|---|---:|---:|---:|
| B13/S1 | 2648.992 | 2686.590 | +1.419% |
| B13/S2 | 3150.912 | 3133.659 | -0.548% |

Nsys over the last 30 S1 forwards shows approximately 1,077 ordinary QKV
launches (10.60 ms estimated from the same grid group) replaced by 359 batch-3
launches (7.461 ms), saving about 104 us per forward.

The full 8,192-row replay is byte-identical to the accepted Stage-6 replay and
passes every Stage-3 full-FP32 gate: policy top-1 99.7437%, probability RMSE
1.031e-4, total variation 0.0017066, JSD 3.718e-6, outcome RMSE 0.0022866,
score-mean RMSE 0.0019959, and ownership sigmoid RMSE 0.0002464.

The 1,000-iteration symmetric S1 A-B-B-A/B-A-A-B result after thermal
preconditioning was:

| mode | values (nnEval/s) | mean | median |
|---|---|---:|---:|
| control | 2646.700 / 2642.268 / 2639.581 / 2637.764 | 2641.578 | 2640.924 |
| strided QKV | 2683.254 / 2678.742 / 2676.501 / 2676.254 | 2678.688 | 2677.621 |

Mean improvement is 1.405%; median improvement is 1.390%. Accept for S1. The
S2 regression means it remains disabled in the S2 production regime.
