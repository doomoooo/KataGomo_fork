# Stage 5: strided-batched FFN input projection

Status: rejected (2026-08-05 UTC).

The candidate concatenated each FFN block's `linear1` and `linearGate`
weights, stored their outputs contiguously, and replaced two identical
`cublasHgemm` submissions with one `cublasHgemmStridedBatched` submission
using batch count 2 and shared-input stride 0. The rest of the FFN and network
was unchanged. The fixed scope was FP16, 19x19, B1-B13.

Under one GPU lock, after a 500-iteration thermal precondition, the short
ordered A-B-B-A result was:

| mode | throughput (nnEval/s) | mean |
|---|---|---:|
| official two GEMMs | 3108.031 / 3114.990 | 3111.511 |
| strided-batched | 3040.424 / 3048.978 | 3044.701 |

The candidate is 2.147% slower by mean, with both same-mode repeats agreeing.
It failed the predeclared early performance gate, so Nsys, full 8,192-row
accuracy replay, and long whole-network A/B were not run. The implementation
was removed and the official two-GEMM path restored.

This rejects `cublasHgemmStridedBatched(batchCount=2)` on SM120 for this
shape. It does not reject an actual single `M=2304` GEMM with interleaved
weights and a layout-aware SwiGLU; that is a separate candidate requiring an
isolated GEMM microbenchmark first.
