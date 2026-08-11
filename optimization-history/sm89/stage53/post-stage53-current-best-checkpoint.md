# Stage 53 post-decision current-best macro checkpoint

Frozen S2 Nsys uses streams [114, 101] and reports 3206.994 nnEval/s. The combined kernel span is 266.187 ms, busy union 262.260 ms (98.525%), and uncovered gaps 3.928 ms.

| Family | Launches | Raw duration (ms) | Union (ms) | Exclusive (ms) | Busy union | Busy exclusive |
|---|---:|---:|---:|---:|---:|---:|
| FFN dual projection + SwiGLU | 2112 | 91.398 | 90.658 | 47.877 | 34.57% | 18.26% |
| FlashAttention | 2112 | 70.548 | 68.787 | 27.607 | 26.23% | 10.53% |
| QKV + RoPE | 2112 | 63.223 | 62.467 | 27.804 | 23.82% | 10.60% |
| FFN linear2 + residual | 2112 | 59.001 | 55.738 | 8.305 | 21.25% | 3.17% |
| attention out-projection | 2112 | 44.020 | 42.418 | 5.785 | 16.17% | 2.21% |
| RMSNorm | 4224 | 24.009 | 24.009 | 3.219 | 9.15% | 1.23% |
| outer postConv | 704 | 15.882 | 15.809 | 1.999 | 6.03% | 0.76% |
| outer preConv | 704 | 11.805 | 11.614 | 1.803 | 4.43% | 0.69% |
| heads/frontend/other | 1932 | 11.093 | 8.446 | 6.090 | 3.22% | 2.32% |
| outer C768 BN + SiLU | 768 | 7.010 | 7.010 | 0.333 | 2.67% | 0.13% |
| outer/trunk C384 BN + SiLU | 704 | 5.866 | 5.866 | 0.878 | 2.24% | 0.33% |
| RMSNorm inverse-only | 0 | 0.000 | 0.000 | 0.000 | 0.00% | 0.00% |

Broad NCU covered 75 distinct S2 launch geometries.

| Family | Geometries | Median (us) | Max (us) | SM | DRAM | L2 | Issue | Active warps | Regs | Shared (B) | Waves/SM |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| FFN dual projection + SwiGLU | 1 | 42.048 | 42.048 | 66.9% | 15.3% | 72.2% | 21.7% | 15.4% | 168 | 49152 | 2.60 |
| QKV + RoPE | 1 | 30.560 | 30.560 | 45.8% | 18.4% | 53.6% | 20.7% | 15.1% | 240 | 49152 | 1.30 |
| FlashAttention | 1 | 29.984 | 29.984 | 32.8% | 36.8% | 38.6% | 25.7% | 22.5% | 168 | 16768 | 2.44 |
| FFN linear2 + residual | 1 | 23.104 | 23.104 | 61.5% | 51.7% | 67.9% | 15.4% | 8.3% | 162 | 65536 | 0.87 |
| heads/frontend/other | 44 | 7.600 | 21.952 | 31.4% | 2.7% | 40.0% | 14.5% | 12.7% | 244 | 49152 | 1.16 |
| outer postConv | 9 | 20.416 | 20.608 | 46.1% | 31.3% | 72.7% | 15.0% | 13.5% | 186 | 40960 | 0.87 |
| outer preConv | 1 | 16.576 | 16.576 | 57.7% | 48.2% | 62.1% | 15.3% | 8.3% | 162 | 81920 | 0.87 |
| attention out-projection | 14 | 12.688 | 15.008 | 31.2% | 44.9% | 41.6% | 14.6% | 8.3% | 186 | 32768 | 0.43 |
| outer C768 BN + SiLU | 1 | 7.552 | 7.552 | 33.0% | 50.4% | 48.4% | 57.2% | 78.0% | 31 | 0 | 2.29 |
| outer/trunk C384 BN + SiLU | 1 | 7.424 | 7.424 | 31.7% | 58.6% | 30.0% | 46.8% | 63.7% | 16 | 0 | 9.17 |
| RMSNorm | 1 | 5.472 | 5.472 | 31.7% | 32.4% | 40.6% | 34.2% | 55.1% | 40 | 0 | 0.76 |
