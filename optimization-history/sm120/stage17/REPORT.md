# Stage 17: final fixed-19x19 topology and validation

Status: superseded; optimization loop reopened on 2026-08-06 UTC after the
5080-history cross-check found unimplemented retained routes.

Scope is RTX 5090 D, fixed 19x19, FP16, model
`b11c768h12nbt3tflrs-fson-silu`, and pure device-resident network forward.
Masked and non-19x19 paths are not optimization targets and retain fallback
behavior. The final binary SHA256 is
`cf380e0dce0d8d3487cdc0487ec21b2668991f986e8cfd3037f0ae3bba30400f`.

## Final topology scan

The coarse scan used 200 measured iterations per point after 20 warmups.
S1 enabled the two projection paths accepted only for single-stream execution;
S2-S4 disabled them.

| topology | B1 | B2 | B4 | B6 | B8 | B10 | B13 |
|---|---:|---:|---:|---:|---:|---:|---:|
| S1 | 765.5 | 1284.6 | 1745.5 | 2131.3 | 2191.9 | 2218.6 | 2982.1 |
| S2 | 999.2 | 1749.9 | 2432.6 | 3014.2 | 3016.3 | 3256.8 | 3472.3 |
| S3 | 1256.5 | 2044.1 | 2636.6 | 3221.4 | 3145.1 | 3348.2 | 3446.0 |
| S4 | 1396.3 | 2193.5 | 2686.7 | 3246.6 | 3104.7 | 3343.0 | 3411.4 |

All values are combined nnEval/s. Refinement at B11/B12 stayed below B13:
S2 `3364.7/3411.6`, S3 `3383.9/3408.8`, and S4 `3345.4/3379.8`.

The ordered 1,000-iteration B13 confirmation used
`S2-S3-S4-S4-S3-S2`:

| topology | mean nnEval/s | range |
|---|---:|---:|
| S2/B13 | 3461.079 | 3434.861-3487.297 |
| S3/B13 | 3400.520 | 3389.331-3411.709 |
| S4/B13 | 3362.087 | 3355.405-3368.769 |

S2/B13 is the throughput optimum. The final S1/B13 configuration measured
2996.035 nnEval/s over two 1,000-iteration runs
(2993.207-2998.863). Use
`/workspace/bench-cuda-gpu2-5090d-s1.cfg` for lowest server concurrency and
`/workspace/bench-cuda-gpu2-5090d-s2.cfg` for maximum pure-forward throughput.

## Stream validation

The final S2 Nsys trace used two independent CUDA streams, IDs 65 and 82,
with 25,762 kernels on each. Their kernel busy intervals overlapped for
256.777 ms; overlap was 70.85% of the union busy time. Traced throughput was
3474.409 nnEval/s. This verifies real concurrent execution rather than merely
two configured host threads.

## Accuracy

Both final topology configurations were replayed over all 8,192 fixed rows
and compared directly with the full-FP32 reference. Both pass every
predeclared gate. S2 retains the accepted replay SHA256
`9ead43c9e5567242defbba4b7b45110ce2b802c39146ee8f9c23a1a4863c3d62`.

| metric | S1 | S2 | gate |
|---|---:|---:|---:|
| policy top-1 | 99.7437% | 99.7437% | >=99.70% |
| optimistic top-1 | 99.7314% | 99.7314% | >=99.60% |
| probability RMSE | 1.03074e-4 | 1.03075e-4 | <=1.5e-4 |
| total variation | 0.00170660 | 0.00170662 | <=0.0025 |
| JSD | 3.71788e-6 | 3.71786e-6 | <=8e-6 |
| outcome RMSE | 0.00228663 | 0.00228669 | <=0.015 |
| score-mean RMSE | 0.00199595 | 0.00199598 | <=0.01 |
| ownership sigmoid RMSE | 0.00024643 | 0.00024640 | <=0.0004 |

## Withdrawn stop condition

This stage originally claimed no remaining profiler-supported candidate. That
claim was incorrect: it treated plain-GEMM probes as coverage for the fused
FFN and fixed projection AOT kernels, stopped the batch scan at B13, and
mistook parsed historical options for implemented paths. See
`/workspace/results/rebuild/5080-CROSSCHECK.md`. Stage 17 remains a valid B1-B13
measurement checkpoint, not a final optimization result.
