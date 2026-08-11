# Stage 48 current-best macro checkpoint

Frozen S2 Nsys uses streams [114, 112] and reports 3236.040 nnEval/s. The combined kernel span is 269.364 ms, busy union 262.646 ms (97.506%), and uncovered gaps 6.718 ms.

| Family | Launches | Raw duration (ms) | Union (ms) | Exclusive (ms) | Busy union | Busy exclusive |
|---|---:|---:|---:|---:|---:|---:|
| FFN dual projection + SwiGLU | 2112 | 90.366 | 89.535 | 50.490 | 34.09% | 19.22% |
| FlashAttention | 2112 | 69.097 | 67.422 | 28.276 | 25.67% | 10.77% |
| QKV + RoPE | 2112 | 63.402 | 62.568 | 27.773 | 23.82% | 10.57% |
| FFN linear2 + residual | 2112 | 59.783 | 56.619 | 7.873 | 21.56% | 3.00% |
| attention out-projection | 2112 | 43.573 | 41.940 | 4.909 | 15.97% | 1.87% |
| RMSNorm | 4224 | 25.931 | 25.931 | 1.999 | 9.87% | 0.76% |
| outer postConv | 704 | 15.918 | 15.731 | 1.915 | 5.99% | 0.73% |
| outer preConv | 704 | 12.362 | 12.135 | 1.625 | 4.62% | 0.62% |
| heads/frontend/other | 2124 | 11.558 | 8.337 | 6.063 | 3.17% | 2.31% |
| outer C768 BN + SiLU | 768 | 6.999 | 6.999 | 0.466 | 2.66% | 0.18% |
| outer/trunk C384 BN + SiLU | 704 | 5.961 | 5.961 | 0.726 | 2.27% | 0.28% |
| RMSNorm inverse-only | 0 | 0.000 | 0.000 | 0.000 | 0.00% | 0.00% |

Broad NCU covered 81 distinct S2 launch geometries.

| Family | Geometries | Median (us) | Max (us) | SM | DRAM | L2 | Issue | Active warps | Regs | Shared (B) | Waves/SM |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| FFN dual projection + SwiGLU | 1 | 42.432 | 42.432 | 66.2% | 15.5% | 71.5% | 21.8% | 15.4% | 168 | 49152 | 2.60 |
| QKV + RoPE | 1 | 30.688 | 30.688 | 45.5% | 15.0% | 53.3% | 20.7% | 15.0% | 240 | 49152 | 1.30 |
| FlashAttention | 1 | 30.272 | 30.272 | 32.5% | 36.5% | 38.2% | 25.8% | 22.5% | 168 | 16768 | 2.44 |
| FFN linear2 + residual | 1 | 23.264 | 23.264 | 61.1% | 51.4% | 67.5% | 15.4% | 8.3% | 162 | 65536 | 0.87 |
| heads/frontend/other | 45 | 7.520 | 22.272 | 30.9% | 2.7% | 39.4% | 14.5% | 12.7% | 244 | 49152 | 1.16 |
| outer postConv | 13 | 20.416 | 20.704 | 46.0% | 34.2% | 72.4% | 15.0% | 13.5% | 186 | 40960 | 0.87 |
| outer preConv | 1 | 16.640 | 16.640 | 57.3% | 47.9% | 61.8% | 15.3% | 8.3% | 162 | 81920 | 0.87 |
| attention out-projection | 15 | 12.832 | 14.656 | 32.0% | 49.6% | 42.7% | 14.4% | 8.3% | 186 | 32768 | 0.43 |
| outer C768 BN + SiLU | 1 | 7.424 | 7.424 | 33.6% | 52.5% | 49.0% | 57.1% | 78.9% | 31 | 0 | 2.29 |
| RMSNorm | 1 | 7.328 | 7.328 | 23.4% | 49.6% | 30.0% | 33.4% | 55.6% | 40 | 0 | 0.76 |
| outer/trunk C384 BN + SiLU | 1 | 7.072 | 7.072 | 33.4% | 20.0% | 31.8% | 44.9% | 62.8% | 16 | 0 | 9.17 |
