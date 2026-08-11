# 5080 history cross-check against the 5090D rebuild

Created: 2026-08-06 UTC.

The Stage-17 stop condition is withdrawn. A same-day, same-source,
TRT-CUDA-CUDA-TRT comparison at fixed 19x19, B13, S2, FP16, 1000 timed
iterations and 50 warmups measured TensorRT at 3260.834 nnEval/s and the
current CUDA path at 3509.727 nnEval/s. CUDA leads by only 7.633%, whereas the
5080 history contains several material optimization families not rebuilt here.

## Cross-check

| 5080 retained route | Current 5090D state | Assessment |
|---|---|---|
| FA4 both16 | Implemented and accepted | Complete for the current FA4 tile |
| Q/K/V strided-batched projection | Implemented, accepted only for S1 | Partial; disabled in S2 |
| Wide QKV AOT | Implemented as fixed-B13 TileLang planar-output GEMM | Complete; Stage 21 +3.806% |
| GEMM beta=1 residual | Implemented for out-projection and linear2 | Functional baseline, not the later AOT epilogues |
| Shared-A dual GEMM + SwiGLU | Implemented as fixed-B13 TileLang fused FFN | Complete; Stage 20 +2.586% |
| TileLang fused FFN | Implemented and dispatched at B13 | Complete |
| Linear2 + residual AOT | No fixed AOT kernel | Missing, high priority |
| Out-projection + residual AOT | No fixed AOT kernel | Missing, high priority |
| Fused Q/K learnable RoPE | Implemented and accepted | Complete |
| Batch-shared RoPE + B19 unroll | Simplified B13 kernel tested and rejected; unroll/two-way options are dead | Partial, implementation not equivalent |
| Half2 RoPE | Implemented; direct gain but below the old whole-network threshold | Tested, may be reconsidered in a bundle |
| Ordered C384 RMSNorm | One-warp exact-tree half2 version implemented | Partial; old vec8 load schedule not rebuilt |
| C768 affine-SiLU vec8 | Only one-half2-per-thread kernel implemented | Partial; old flat vec8 schedule missing |
| Persisting-L2 trunk and inner windows | Options parsed, no runtime code | Missing |
| Outer C384/C768 projection AOT | Option parsed, no kernel or dispatch | Missing |
| Ordinary matmul weight sharing across S2 servers | Option parsed, no shared per-GPU cache | Missing |
| Initial-conv frontend tactic | Option parsed, no runtime code | Missing |
| Initial global matmul-add | Option parsed, no runtime code | Missing |
| Fused policy P1 | Option parsed, no runtime code | Missing |
| Head BN half-to-float | Option parsed, no runtime code | Missing |
| Wide head no-split projection | Option parsed, no runtime code | Missing |
| QKV/outer tile refinements | No corresponding AOT kernels | Missing after their parent kernels |

The following 5080 routes were rejected there and do not need to be rebuilt
before the retained routes: CUDA Graph, RMSNorm algebraic folding, QKV+RoPE
AOT fusion, DSM/cluster launch, row480 RMSNorm, projection swizzle 8,
share-all special weights, folded RMS-QKV, two-way RoPE, lower-stage or
persistent QKV, outer fused-SiLU, fused head pooling, FA4 N48/N96,
register-Q, M128/256-thread FA4, and BF16.

## Dead scaffold switches

The current `Options` structure contains historical defaults that create a
false impression of completeness. These fields have no `options.<field>`
runtime use in `cudabackend_sm120.cpp`:

`useQKVGemmRopeAot`,
`useBatchSharedRoPEUnroll19`, `useBatchSharedRoPETwoWay`,
`useProjectionGemmLt`, `useFusedRMSNormFFN`,
`useRMSNormQKVGemmAot`, `useGraph`, `usePersistingL2Trunk`,
`usePersistingL2Inner`, `useOuterProjectionAot`, `shareModelWeights`,
`shareWideQKVWeights`, `shareOuterProjectionWeights`,
`useInitialConvFrontend`, `useInitialConvBiasFrontend`,
`useInitialGlobalMatMulAdd`, `useFusedPolicyP1`,
`useHeadBNHalfToFloat`, and `useWideHeadProjection`.

## Reopened priority order

1. Rebuild fixed-shape projection AOT kernels: linear2 residual,
   out-projection residual, then outer expand/contract. Fused FFN and wide QKV
   are complete for fixed B13/S2.
2. Rebuild vec8 RMSNorm/affine-SiLU and persisting-L2 windows.
3. Rebuild and measure the small frontend/head fusions as independent changes.
4. Repeat the fixed B13 TensorRT comparison, full accuracy, and independent-stream
   Nsys validation only after the major kernels settle. Batch scanning is outside
   the current task scope.
