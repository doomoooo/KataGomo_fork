# Stage 52 post-decision current-best macro checkpoint

Frozen S2 Nsys uses streams [114, 99] and reports 3186.189 nnEval/s. The combined kernel span is 273.265 ms, busy union 267.965 ms (98.060%), and uncovered gaps 5.300 ms.

| Family | Launches | Raw duration (ms) | Union (ms) | Exclusive (ms) | Busy union | Busy exclusive |
|---|---:|---:|---:|---:|---:|---:|
| FFN dual projection + SwiGLU | 2112 | 89.711 | 87.675 | 59.115 | 32.72% | 22.06% |
| FlashAttention | 2112 | 67.110 | 61.582 | 31.753 | 22.98% | 11.85% |
| QKV + RoPE | 2112 | 62.646 | 60.352 | 29.064 | 22.52% | 10.85% |
| FFN linear2 + residual | 2112 | 62.871 | 52.658 | 16.817 | 19.65% | 6.28% |
| attention out-projection | 2112 | 39.818 | 34.788 | 10.128 | 12.98% | 3.78% |
| RMSNorm | 4224 | 25.979 | 25.976 | 2.191 | 9.69% | 0.82% |
| outer postConv | 704 | 16.392 | 16.205 | 2.727 | 6.05% | 1.02% |
| outer preConv | 704 | 12.914 | 12.334 | 2.731 | 4.60% | 1.02% |
| heads/frontend/other | 1932 | 11.233 | 8.290 | 6.485 | 3.09% | 2.42% |
| outer C768 BN + SiLU | 768 | 7.509 | 7.507 | 0.422 | 2.80% | 0.16% |
| outer/trunk C384 BN + SiLU | 704 | 6.497 | 6.497 | 0.632 | 2.42% | 0.24% |
| RMSNorm inverse-only | 0 | 0.000 | 0.000 | 0.000 | 0.00% | 0.00% |

Broad NCU covered 63 distinct S2 launch geometries.

| Family | Geometries | Median (us) | Max (us) | SM | DRAM | L2 | Issue | Active warps | Regs | Shared (B) | Waves/SM |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| FFN dual projection + SwiGLU | 1 | 42.272 | 42.272 | 66.5% | 15.7% | 71.8% | 21.6% | 15.3% | 168 | 49152 | 2.60 |
| QKV + RoPE | 1 | 30.976 | 30.976 | 45.2% | 18.9% | 52.9% | 20.7% | 15.0% | 240 | 49152 | 1.30 |
| FlashAttention | 1 | 30.144 | 30.144 | 32.6% | 36.6% | 38.4% | 25.7% | 22.5% | 168 | 16768 | 2.44 |
| FFN linear2 + residual | 1 | 23.072 | 23.072 | 61.7% | 51.9% | 68.2% | 15.4% | 8.3% | 162 | 65536 | 0.87 |
| heads/frontend/other | 39 | 7.488 | 22.240 | 31.0% | 2.7% | 39.5% | 14.5% | 12.7% | 244 | 49152 | 1.16 |
| outer postConv | 9 | 20.384 | 20.864 | 45.6% | 36.4% | 71.9% | 15.0% | 13.6% | 186 | 40960 | 0.87 |
| outer preConv | 1 | 16.768 | 16.768 | 57.5% | 47.6% | 61.8% | 15.3% | 8.3% | 162 | 81920 | 0.87 |
| attention out-projection | 7 | 12.704 | 14.176 | 33.1% | 49.3% | 44.2% | 14.5% | 8.3% | 186 | 32768 | 0.43 |
| outer C768 BN + SiLU | 1 | 9.152 | 9.152 | 26.9% | 74.0% | 39.4% | 57.0% | 76.8% | 31 | 0 | 2.29 |
| outer/trunk C384 BN + SiLU | 1 | 7.616 | 7.616 | 30.9% | 55.9% | 29.4% | 44.8% | 62.4% | 16 | 0 | 9.17 |
| RMSNorm | 1 | 5.376 | 5.376 | 32.3% | 31.8% | 41.3% | 34.1% | 55.5% | 40 | 0 | 0.76 |
