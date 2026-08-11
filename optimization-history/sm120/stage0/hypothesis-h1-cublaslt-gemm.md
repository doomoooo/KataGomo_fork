# Hypothesis H1: FP16 transformer GEMMs should use cuBLASLt on SM120

Created: 2026-08-05 (UTC), before any implementation. Evidence and mechanism
are from this machine (RTX 5090D, CUDA 13.2.86, KataGo aac3a3d, stage-0
scaffold), measured with Nsys 2026.1.3 / NCU 2026.2.1 under gpu-lock.

## Evidence (B13, S2, 5090D, timed region)

Nsys per-forward kernel-time composition (one server stream, 20 blocks,
20 timed iterations, window cut at last 400 SDPA launches):

| component | kernel family | per-forward ms | share |
|---|---|---:|---:|
| FP16 GEMM | `cutlass::Kernel2<cutlass_80_tensorop_h16816gemm_...>` | 5.12 | 52.4% |
| attention | `cudnn_generated_..._sdpa_sm80_flash_fprop_wmma_...` | 2.16 | 22.1% |
| RMSNorm | `rmsNormGammaBetaNHWCHalfVecKernel` | 0.63 | 6.4% |
| residual | `maskedResidualAddNHWCHalfKernel` | 0.47 | 4.8% |
| RoPE | `applyRoPELearnableRecomputeHalfKernel` | 0.46 | 4.7% |
| SwiGLU | `swiGLUHalfStrideKernel` | 0.45 | 4.6% |
| mask→bias | `maskToAttnBiasFullHalfKernel` | 0.33 | 3.4% |
| head SiLU | `applyCScaleBiasNHWCSiluMaskHalfKernel` | 0.28 | 2.8% |
| other | heads, copy, bias, pools | ~0.3 | ~3% |

The FP16 GEMMs all go through `MatMulLayer::apply` -> legacy `cublasHgemm`.
On SM120 cuBLAS picks SM80 CUTLASS kernels, not the SM120-native `nvjet`
kernels (only 670 nvjet launches in the same window, ~0.04 ms/forward).

NCU on the dominant GEMM instantiation
`cutlass_80_tensorop_h16816gemm_128x128_32x5_nn_align8` (grid 96x2, 128 thr):

- Achieved occupancy 8.3% (theoretical 8.3%), block limit = shared memory:
  only one 128x128x5-stage tile fits per SM.
- Active warps per scheduler 1.0 (max 12); 84.3% of cycles no eligible warp;
  issued warp per scheduler 0.16.
- 54% of issue stalls are fixed-latency execution dependency (short
  scoreboard), consistent with too few warps to hide MMA/load latency.

## Mechanism

Legacy `cublasHgemm` uses cuBLAS's default kernel heuristic. On SM120 that
heuristic selects large-tile SM80 tensor-op kernels whose shared-memory
footprint caps occupancy at 1 CTA/SM; SM120-native `nvjet` (tcgen05) kernels
exist in the same cuBLAS distribution and are selected only for a few other
shapes. cuBLASLt exposes heuristic/plan control and is the supported route to
SM120 kernels for FP16 GEMMs.

## Predicted change

Switching the FP16 `MatMulLayer` GEMMs on SM120 to cuBLASLt
(fp32 accumulation, per-shape cached algorithm, workspace per handle) will:

1. Replace the SM80 h16816 GEMM launches with higher-occupancy / SM120-native
   kernels for the main transformer GEMMs.
2. Reduce the GEMM share from ~5.12 ms/forward; kernel-level NCU occupancy
   should rise above 8.3% and issue stalls drop.
3. Keep numerics in the same domain (FP16 inputs, FP32 accumulate), so the
   full 8192-row FP32-reference regression should stay within the same gates
   as the official FP16 baseline; exact bit equality is NOT assumed.

## Risks

- cuBLASLt heuristic may still select a suboptimal kernel for some shapes:
  measured outcome decides, no tuning beyond one heuristic call per shape.
- Per-shape plan/workspace setup at first call; mitigated by caching the
  algorithm per (inChannels, outChannels, batchSize) and one reusable
  workspace buffer.
- Stream semantics: cuBLASLt must be launched on `CudaHandles::stream` (the
  server thread's private non-blocking stream), same as cuBLAS today.
- 4090 / official path must stay untouched (hook only installed on SM120).

## Validation

1. Build; 4090 benchmarknn unchanged (no SM120 model => no hook).
2. 5090D smoke: log line confirms cuBLASLt matmul active; benchmarknn works.
3. Nsys: `nvjet_sm120_*` launches increase; `cutlass_80_tensorop_h16816gemm_...`
   launches drop for the main GEMMs; per-forward GEMM ms decreases.
4. NCU on the new dominant GEMM: occupancy / tensor-pipe / stall evidence.
5. A/B at B13/S2: same regime, ABBA, 3x300 iterations; report absolute
   nnEval/s and kernel times separately.
6. Full 8192-row FP16 replay vs FP32 reference (all heads + p0loss gates).
7. Keep nsys/ncu artifacts and update the optimization history with timestamps.

## Reopen condition

If cuBLASLt does not select better kernels or throughput does not improve
outside noise, the hypothesis is rejected and the fallback is to keep
`cublasHgemm` and pursue SM120-native AOT GEMMs (CuTe/CUTLASS) instead.
