# Hypothesis H2: FP16 transformer GEMMs should use cuBLASLt on SM120

Created: 2026-08-05 (UTC), before implementation. Evidence is from this
machine (RTX 5090D, CUDA 13.2.86, KataGo 090caa6 with stage-1 FA4 attention),
measured with Nsys 2026.1.3 / NCU 2026.2.1 under gpu-lock.

## Evidence

Stage-1 timed-window Nsys (B13, S2, FA4 on, `requireExactNNLen=true`):

| component | kernel family | per stream-fwd ms | share |
|---|---|---:|---:|
| FP16 GEMM | `cutlass::Kernel2<cutlass_80_tensorop_h16816gemm_...>` | ~2.5 | ~52% |
| FA4 attention | `kernel_cutlass_kernel_flash_attn...` | ~0.45 | ~9% |
| RMSNorm / RoPE / SwiGLU / residual / head | custom kernels | ~1.4 | ~29% |

All FP16 `MatMulLayer` GEMMs call legacy `cublasHgemm`. NCU on the dominant
instantiation `cutlass_80_tensorop_h16816gemm_128x128_32x5_nn_align8`
(grid 96x2, 128 thr):

- Achieved occupancy 8.3% (theoretical 8.3%), block limit = shared memory
  (one 128x128x5-stage tile per SM).
- Active warps per scheduler 1.0; 84% cycles no eligible warp;
  issued warp per scheduler 0.16.
- 54% of issue stalls are fixed-latency execution dependency.

The same cuBLAS distribution already contains SM120-native `nvjet` kernels
(they appear for ~670 launches in the same window, e.g.
`nvjet_sm120_hhh_mma_64x64x128_3_32x32x128_tmaAB_alignCD4_bz_NNNN`), so the
hardware path exists; `cublasHgemm`'s heuristic simply does not select it for
the transformer shapes.

## Mechanism

`cublasHgemm` uses a fixed heuristic that, for these shapes on SM120, chooses
large-tile SM80 tensor-op kernels whose shared-memory footprint caps occupancy
at 1 CTA/SM. `cublasLtMatmul` exposes heuristic/plan control and is the
supported route to SM120 kernels for FP16 GEMMs.

## Predicted change

Routing the SM120 FP16 `MatMulLayer` GEMMs through cuBLASLt
(fp32 accumulation, per-shape cached algorithm, one reusable workspace,
launched on the server thread's private stream) will:

1. Replace the SM80 h16816 GEMM launches with higher-occupancy or SM120-native
   kernels for the main transformer GEMMs.
2. Reduce the GEMM share from ~2.5 ms/stream-fwd; NCU occupancy should rise
   above 8.3% and issue stalls drop.
3. Keep numerics in the same domain (FP16 inputs, FP32 accumulation), so the
   full 8192-row FP32-reference regression stays within the same gates.

## Risks

- cuBLASLt heuristic may still pick a suboptimal kernel for some shapes: the
  measurement decides; no manual tuning beyond one heuristic call per shape.
- Per-shape plan/workspace setup: cached per (inChannels, outChannels,
  batchSize); warmup covers all shapes before the timed loop.
- CPU launch overhead: benchmarknn is GPU-event-timed, but with two streams
  the CPU must stay ahead; cuBLASLt call path is kept minimal (no per-call
  descriptor creation after warmup).
- 4090 / official path must stay untouched (hook installed only on SM120).

## Validation

1. Build; 4090 unchanged (no SM120 model => no hook).
2. 5090D smoke + log line confirms cuBLASLt matmul active.
3. Nsys: `nvjet_sm120_*` launches increase and
   `cutlass_80_tensorop_h16816gemm_...` drops for the main GEMMs;
   per-forward GEMM time decreases.
4. NCU on the new dominant GEMM: occupancy / tensor-pipe / stall evidence.
5. ABBA at B13/S2, 3x300, same regime; absolute nnEval/s and kernel times.
6. Full 8192-row FP16 replay vs the stage-1 FP32 reference (all heads).
7. Update optimization history with artifact timestamps.

## Reopen condition

If cuBLASLt does not select better kernels or throughput does not improve
outside noise, keep `cublasHgemm` and pursue SM120-native AOT GEMMs
(CuTe/CUTLASS tcgen05) instead.
