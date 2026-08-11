# Stage 48 current-best macro checkpoint

Frozen S2 Nsys uses streams [114, 100] and reports 3175.175 nnEval/s. The combined kernel span is 267.092 ms, busy union 263.149 ms (98.524%), and uncovered gaps 3.943 ms.

| Family | Launches | Raw duration (ms) | Union (ms) | Exclusive (ms) | Busy union | Busy exclusive |
|---|---:|---:|---:|---:|---:|---:|
| FFN dual projection + SwiGLU | 2112 | 91.157 | 90.748 | 48.608 | 34.49% | 18.47% |
| FlashAttention | 2112 | 69.176 | 68.180 | 27.854 | 25.91% | 10.58% |
| QKV + RoPE | 2112 | 63.245 | 62.829 | 27.776 | 23.88% | 10.56% |
| FFN linear2 + residual | 2112 | 59.409 | 57.548 | 7.122 | 21.87% | 2.71% |
| attention out-projection | 2112 | 43.249 | 42.338 | 4.919 | 16.09% | 1.87% |
| RMSNorm | 4224 | 24.951 | 24.950 | 2.705 | 9.48% | 1.03% |
| outer postConv | 704 | 15.726 | 15.690 | 2.323 | 5.96% | 0.88% |
| outer preConv | 704 | 12.334 | 12.227 | 1.792 | 4.65% | 0.68% |
| heads/frontend/other | 2124 | 11.410 | 8.684 | 6.201 | 3.30% | 2.36% |
| outer C768 BN + SiLU | 768 | 6.829 | 6.829 | 0.362 | 2.60% | 0.14% |
| outer/trunk C384 BN + SiLU | 704 | 5.846 | 5.846 | 0.766 | 2.22% | 0.29% |

Broad NCU covered 69 distinct S2 launch geometries.

| Family | Geometries | Median (us) | Max (us) | SM | DRAM | L2 | Issue | Active warps | Regs | Shared (B) | Waves/SM |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| FFN dual projection + SwiGLU | 1 | 42.176 | 42.176 | 66.6% | 15.7% | 71.9% | 21.7% | 15.3% | 168 | 49152 | 2.60 |
| QKV + RoPE | 1 | 30.688 | 30.688 | 45.6% | 15.1% | 53.3% | 20.7% | 14.9% | 240 | 49152 | 1.30 |
| FlashAttention | 1 | 30.304 | 30.304 | 32.4% | 36.4% | 38.2% | 25.7% | 22.5% | 168 | 16768 | 2.44 |
| FFN linear2 + residual | 1 | 23.360 | 23.360 | 60.8% | 52.1% | 67.2% | 15.4% | 8.3% | 162 | 65536 | 0.87 |
| heads/frontend/other | 46 | 7.488 | 22.080 | 31.2% | 2.7% | 39.8% | 14.5% | 12.7% | 244 | 49152 | 1.16 |
| outer postConv | 8 | 20.432 | 20.704 | 46.0% | 36.4% | 72.5% | 15.0% | 13.5% | 186 | 40960 | 0.87 |
| outer preConv | 1 | 16.704 | 16.704 | 57.2% | 47.8% | 61.6% | 15.3% | 8.3% | 162 | 81920 | 0.87 |
| attention out-projection | 7 | 13.408 | 14.784 | 31.7% | 48.0% | 42.2% | 14.5% | 8.3% | 186 | 32768 | 0.43 |
| outer C768 BN + SiLU | 1 | 9.088 | 9.088 | 27.2% | 72.1% | 39.8% | 57.5% | 76.8% | 31 | 0 | 2.29 |
| outer/trunk C384 BN + SiLU | 1 | 6.976 | 6.976 | 33.9% | 21.2% | 32.3% | 44.9% | 62.6% | 16 | 0 | 9.17 |
| RMSNorm | 1 | 6.528 | 6.528 | 26.3% | 42.6% | 33.8% | 33.1% | 55.4% | 40 | 0 | 0.76 |
