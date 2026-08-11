# Stage 45: optional outer postConv to following C768 affine-SiLU fusion

Date: 2026-08-06 UTC

## Decision

Retain the exact-B13 RTX 5090 D implementation behind the default-false
`cudaUsePostConvBNSiluSm120` switch, but do not enable it in the accepted
natural-S2 deployment graph.

The final CUTLASS2 tactic is resource-positive and reproducibly improves S1,
so it satisfies the retained-candidate rule. Natural S2 regresses, so Stage 44
remains the deployment baseline and its accepted full-graph profiles remain
the current checkpoint.

## Implemented boundary

The 11 outer C384-to-C768 post-projection residual GEMMs each produce two
outputs in one epilogue:

1. the rounded FP16 residual written back to the trunk buffer;
2. the following FP16 affine plus FP32-expf SiLU output written to the existing
   trunk scratch buffer.

The next outer pre-normalization, or final trunk-tip normalization, is skipped
only when the fused kernel succeeds. The guard requires RTX 5090 D, exact B13,
19x19, FP16 NHWC, C384-to-C768, SiLU, and no mask. Every other shape and backend
uses the existing path.

## Evidence-driven tactic correction

The direct RTX 4090 transfer used a 128-thread 128x128x32/warp-64x64/stage-3
CUTLASS2 kernel. Its short whole-graph result was negative:

| topology | control (nnEval/s) | candidate (nnEval/s) | delta |
|---|---:|---:|---:|
| S1 | 3266.712 | 3251.398 | -0.469% |
| S2 | 4040.302 | 4001.215 | -0.967% |

Targeted NCU showed 168 registers/thread, 50.176 KiB total shared memory, 0.65
waves/SM, about 0.197 eligible warps/scheduler, and zero spills. The accepted
5090D postConv library GEMM uses a 256-thread block. The only evidence-backed
tactic change was therefore to divide the same 128x128 tile among eight warps
instead of four, using a 64x32 warp tile. This reduced per-thread accumulator
and epilogue live state without changing the CTA tile, K tile, or pipeline
stage count.

Final targeted NCU over two real launches reports:

| metric | launch 0 | launch 1 |
|---|---:|---:|
| duration | 21.376 us | 22.048 us |
| registers/thread | 108 | 108 |
| allocated registers/thread | 112 | 112 |
| total shared memory/block | 50.176 KiB | 50.176 KiB |
| active warps | 21.907% | 21.939% |
| eligible warps/scheduler | 0.3412 | 0.3412 |
| register spills | 0 | 0 |

The accepted standalone postConv uses 154 registers/thread and 73.728 KiB
dynamic shared memory, followed by a separate affine-SiLU launch. The final
candidate therefore has strictly lower register and shared-memory footprint,
zero spills, and deletes 11 launches and their full C768 input reads.

No warp-orientation or further tile sweep was performed: NCU does not provide
evidence that the symmetric warp orientation would be better.

## Whole-graph result

The final S1 result is four control and four candidate measurements comprising
forward ABBA and reverse BAAB, each with 20 timed and 5 warmup iterations:

| topology | control mean (nnEval/s) | candidate mean (nnEval/s) | delta |
|---|---:|---:|---:|
| S1 | 3267.010 | 3272.912 | +0.181% |

All two adjacent comparisons in the first ABBA were non-negative. The reverse
round preserved the pooled positive mean.

Natural S2 used a short ABBA under the same iteration count:

| topology | control mean (nnEval/s) | candidate mean (nnEval/s) | delta |
|---|---:|---:|---:|
| S2 | 4047.917 | 4030.194 | -0.438% |

Both adjacent S2 comparisons were negative. The switch therefore remains
default false and no post-change accepted Nsys/NCU checkpoint was captured.

## Accuracy

- The 26-row B13 smoke output was byte-identical to the Stage 44 path.
- The complete 8,192-row fixed-19x19 replay was byte-identical to the Stage 44
  accepted replay.
- Both files have SHA256
  `1503a84be6541685881380d4fee57461960ba58563ad23b94d1d42d8ecb18d96`.

This is stronger than re-running the numeric gates: every stored head output
is unchanged.

## Next direction

This implementation uses CUTLASS C++ 2.x with an `arch::Sm80` TensorOp kernel.
The modern SM120-native candidate is a CuTe DSL dense GEMM with the same
verified residual plus dual-output affine-SiLU epilogue. Stage 44 already
proved the CuTe AOT bridge and process-lifetime handling on this backend. A
CuTe version should be compared against both the accepted cuBLAS-plus-affine
boundary and this default-off CUTLASS2 reference before any further tactic
change.

## Artifacts

- `hypothesis.md`: frozen baseline and falsifiable gates
- `accuracy-smoke/`: 26-row byte-identical replay
- `accuracy-full/`: 8,192-row byte-identical replay
- `nsys-short/`: initial 128-thread natural-S1 trace
- `ncu-targeted/`: 128-thread and final 256-thread reports
- `short-abba/`: rejected 128-thread whole-graph screen
- `short-abba-warp64x32/`: final S1/S2 measurements

