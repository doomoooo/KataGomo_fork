# Stage 1: FA4 AOT attention on SM120 (5090D)

Status: accepted (2026-08-05). All evidence from this machine (RTX 5090D,
CUDA 13.2.86, driver 595.80, KataGo aac3a3d + FA4 AOT stage-1 diff), following
the SKILL.md evidence chain.

## 1. Baseline / critical path (Nsys)

Same regime note: `benchmarknn` and `replaynn` now use `requireExactNNLen=true`
(full-board fixed inputs; mask all ones => identical semantics, removes
needless mask/bias work). This is a deliberate measurement-regime change from
the earlier B13 baseline (which used `requireExactNNLen=false`).

Per-forward timed-window kernel composition (B13, S2, 5090D; official backend,
FA4 off):

| component | kernel | per-iter pair ms | per stream-fwd ms |
|---|---|---:|---:|
| GEMMs | `cutlass_80_tensorop_h16816gemm_*` (SM80) | 5.03 | 2.51 |
| attention | cuDNN SDPA SM80 wmma | 1.75 | 0.87 |
| RMSNorm | `rmsNormGammaBetaNHWCHalfVecKernel` | 0.49 | 0.24 |
| SwiGLU | `swiGLUHalfStrideKernel` | 0.47 | 0.24 |
| residual | `maskedResidualAddNHWCHalfKernel` | 0.41 | 0.21 |
| RoPE | `applyRoPELearnableRecomputeHalfKernel` | 0.39 | 0.20 |
| head SiLU | `applyCScaleBiasNHWCSiluHalfKernel` | 0.24 | 0.12 |

NCU on the official attention kernel (SM80 wmma on SM120): 36 CTAs, achieved
occupancy 16.6% (register-limited), 73% cycles no eligible warp, 36% stalls on
math pipe.

NCU on the dominant GEMM (`cutlass_80_tensorop_h16816gemm_128x128_32x5`):
achieved occupancy 8.3% (shared-memory-limited, 1 CTA/SM), 84% cycles no
eligible warp, 54% fixed-latency dependency stalls.

## 2. Hypothesis H1 (from `hypothesis-h1-cublaslt-gemm.md`)

The FP16 GEMMs use legacy `cublasHgemm`, which on SM120 selects SM80 CUTLASS
kernels at 8.3% occupancy instead of SM120-native `nvjet` kernels. Not yet
implemented; retained for the next stage (GEMM path remains the largest
component, ~52%).

## 3. Stage 1 change (attention)

- FA4 (flash-attn 4.0.0 beta21 / cutlass-dsl 4.6.0.dev0) SM120 forward kernel,
  compiled AOT via `cute.compile` + `export_to_c` (`cpp/neuralnet/fa4_aot/`).
- Kernel: FP16, head_dim 32, MHA, non-causal, tile 128x128, 128 threads,
  1 stage, FP32 accumulators (this FA4 version has no both16 option on SM120;
  `cudaFlashAttentionSm120Accum=both16` currently selects the FP32-accum AOT
  kernel — both16 pending).
- Dispatch: thin hook in `TransformerAttentionBlock::apply`; all SM120 logic in
  `cudabackend_sm120.cpp`. Gate: FP16, MHA, D=32, seqLen=361, maskBuf==NULL,
  `cudaUseFlashAttentionSm120=true`.
- Host side is statically linked: the DSL `libcute_dsl_runtime.so` dependency
  is replaced by `fa4_cuda_bridge.cpp` (8 thin `_cuda*` wrappers). Final binary
  has no venv/Python dependency.

## 4. Results

### Nsys (same regime, B13 S2)

| metric | official | FA4 AOT |
|---|---:|---:|
| attention kernel | cuDNN SM80 wmma, 49 us/block | FA4 SM120, 19 us/block |
| attention per stream-fwd | ~0.87 ms | ~0.45 ms |
| combined throughput under trace | 2808 nnEval/s | 3063 nnEval/s |

### ABBA benchmark (B13, S2, 3x300, no trace, same regime)

| order | config | median nnEval/s |
|---|---:|---:|
| A1 | FA4 on | 3143.68 |
| B1 | FA4 off | 2818.53 |
| B2 | FA4 off | 2816.85 |
| A2 | FA4 on | 3129.88 |

FA4 median 3136.8 vs official 2817.7 => +11.3%. No clock locking (regime S2).

### NCU on FA4 kernel

Still SM80-MMA (m16n8k16) on SM120: grid (3,12,1)=36 CTAs, achieved occupancy
8.3% (theoretical 16.7%, register-limited; grid too small to reach 2 CTA/SM),
83% cycles no eligible warp, 37% stalls on MMA pipe. Faster than cuDNN mainly
via instruction mix/tile choice; tcgen05-native attention is the obvious next
lever.

### Accuracy (8192 rows, all heads, FP32 reference regenerated in the new regime)

`fa4-fp16-vs-fp32-stage1.json`:

| metric | official FP16 vs FP32 (old regime) | FA4 FP16 vs FP32 |
|---|---:|---:|
| policy top-1 | 99.805% | 99.768% |
| optimistic top-1 | 99.695% | 99.707% |
| prob RMSE | 9.74e-05 | 9.86e-05 |
| TV | 1.62e-03 | 1.63e-03 |
| JSD | 3.35e-06 | 3.38e-06 |
| p0loss (ref/cand) | 1.591528 / 1.591523 | 1.591526 / 1.591465 |
| outcome RMSE | 2.33e-03 | 2.37e-03 |
| score mean RMSE | 2.12e-03 | 2.01e-03 |
| ownership sigmoid RMSE | 2.40e-04 | 2.40e-04 |
| max policy abs err | 0.01511 | 0.01244 |

All gates pass; differences are at the same order as the official FP16-vs-FP32
baseline.

## 5. Decision

Accept FA4 AOT attention (stage 1). Next stage candidates, ranked by evidence:
1. FP16 GEMM path (52%): cuBLASLt/nvjet or CuTe AOT GEMMs (H1).
2. tcgen05-native attention (FA4 SM100-style / newer FA4) — 19 us/block has
   large headroom (8.3% achieved occupancy).
3. Elementwise bundle (RMSNorm/RoPE/SwiGLU/residual ~20%) only after 1-2.

## 6. Tools note (nsys hang root cause)

nsys 2026.1.3 on this container hangs after report generation because the
symbol downloader uses debuginfod and the server is unreachable. Fix:
`DEBUGINFOD_URLS= DEBUGINFOD_TIMEOUT=1` (verified: mini program + katago both
exit cleanly). Proxy env did not help (debuginfod ignores it). NCU is
unaffected.
