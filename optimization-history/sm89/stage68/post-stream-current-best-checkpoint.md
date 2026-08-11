# Post-Stage68 current-best macro checkpoint

The clean `bd6b8a6` binary uses two caller-owned non-blocking CUDA streams and reports
3431.046 nnEval/s under a shortened 5-warmup/10-timed Nsys run. The combined kernel span is
132.982 ms, busy union is 131.180 ms (98.645%), so launch gaps are no longer a primary target.

| Family | Launches | Raw duration (ms) | Union (ms) | Exclusive (ms) | Busy union | Busy exclusive |
|---|---:|---:|---:|---:|---:|---:|
| FFN dual projection + SwiGLU | 1122 | 48.540 | 48.501 | 24.129 | 36.97% | 18.39% |
| QKV + RoPE | 1122 | 34.120 | 34.076 | 14.756 | 25.98% | 11.25% |
| FlashAttention | 1122 | 30.435 | 30.365 | 6.800 | 23.15% | 5.18% |
| FFN linear2 + residual | 1122 | 30.395 | 30.242 | 2.353 | 23.05% | 1.79% |
| attention out-projection | 1122 | 24.413 | 24.328 | 1.697 | 18.55% | 1.29% |
| RMSNorm | 2244 | 13.466 | 13.466 | 1.365 | 10.27% | 1.04% |
| outer postConv | 374 | 10.038 | 10.029 | 0.911 | 7.64% | 0.69% |
| outer preConv | 374 | 7.272 | 7.249 | 0.631 | 5.53% | 0.48% |
| heads/frontend/other | 1032 | 6.018 | 4.621 | 2.994 | 3.52% | 2.28% |
| outer/trunk C384 BN + SiLU | 374 | 3.061 | 3.061 | 0.436 | 2.33% | 0.33% |
| outer C768 BN + SiLU | 34 | 0.267 | 0.267 | 0.082 | 0.20% | 0.06% |

Broad NCU sampled one representative of 61 launch geometries. The three leading kernels are:

| Kernel family | Duration (us) | SM | DRAM | L2 | Issue | Active warps | Registers | Dynamic shared | Waves/SM |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| FFN dual projection + SwiGLU | 41.632 | 67.6% | 16.0% | 73.0% | 19.8% | 15.3% | 168 | 49.152 KiB | 2.60 |
| QKV + RoPE | 30.528 | 45.9% | 15.1% | 53.6% | 20.8% | 14.9% | 240 | 49.152 KiB | 1.30 |
| FlashAttention both16 | 23.584 | 42.7% | 46.9% | 49.5% | 48.1% | 29.3% | 117 | 16.768 KiB | 1.83 |

The next investment order is therefore dual FFN, QKV, then FlashAttention. NCU duration is used
only to explain each unchanged kernel boundary; natural Nsys union/exclusive time determines
whole-graph priority.
