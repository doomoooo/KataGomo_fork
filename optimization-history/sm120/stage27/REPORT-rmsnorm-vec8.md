# Stage 27: B13/S2 C384 RMSNorm vec8

Status: accepted for the fixed 19x19 B13/S2 configuration on RTX 5090D
(2026-08-06 UTC). The runtime option remains default-off and is enabled only
in `bench-cuda-gpu2-5090d-s2.cfg`.

## Mechanism

The accepted exact-tree kernel used six separated `half2` input, gamma, beta,
and output accesses per lane. The vec8 candidate divides each aligned C384 row
into 256-half and 128-half regions, then uses one aligned `uint4` and one
aligned `uint2` access per lane for each tensor. It retains one warp per row
and four rows per CTA. The new lane mapping changes the FP32 reduction tree,
so full accuracy regression was required.

NCU on the same B13 one-stream launch reports:

| metric | exact-tree control | vec8 | change |
|---|---:|---:|---:|
| duration | 5.28 us | 4.58 us | -13.3% |
| executed instructions | 1,102,879 | 699,281 | -36.6% |
| registers/thread | 38 | 41 | +3 |
| achieved occupancy | 49.08% | 45.16% | -3.92 pp |
| memory throughput | 684.36 GB/s | 789.37 GB/s | +15.3% |

The lower instruction count and higher effective memory throughput dominate
the modest register and occupancy regression. In Nsys, the 3,960 timed S2
RMSNorm launches changed from 19.305 ms with 4.608 us median to 16.461 ms with
4.032 us median. The candidate trace retained a near-zero, stable stream phase
(3.66 us median).

## Whole-network performance

The 400-iteration S2 A-B-B-A screen measured control 3740.933 and vec8
3784.805 nnEval/s, a 1.173% gain.

The 1,000-iteration symmetric A-B-B-A / B-A-A-B confirmation measured:

| mode | values (nnEval/s) | mean | median |
|---|---|---:|---:|
| exact-tree control | 3723.503 / 3700.802 / 3664.697 / 3671.819 | 3690.205 | 3686.310 |
| vec8 | 3745.470 / 3737.492 / 3719.331 / 3676.554 | 3719.712 | 3728.412 |

Mean improvement is 0.800%; median improvement is 1.142%. The forward and
reverse halves improve by 0.790% and 0.809%, respectively.

## Accuracy

The candidate was compared directly with the Stage-1 full-FP32 reference over
all 8,192 fixed 19x19 rows. All established gates pass: policy top-1 99.7803%,
optimistic-policy top-1 99.7070%, probability RMSE 1.019e-4, total variation
1.692e-3, JSD 3.656e-6, max policy absolute error 0.01843, weighted p0loss
delta 3.016e-5, outcome RMSE 2.270e-3, score-mean RMSE 1.873e-3, and ownership
sigmoid RMSE 2.458e-4.

Artifacts are in `rmsnorm-vec8-s2-abba/`, `rmsnorm-vec8-s2-long/`,
`rmsnorm-vec8-accuracy/`, `rmsnorm-vec8-s2.nsys-rep`, and
`rmsnorm-vec8-b13.ncu-rep`.
