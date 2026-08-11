# Stage 56 post-decision current-best macro checkpoint

Frozen S2 Nsys uses streams [114, 113] and reports 3267.780 nnEval/s. The combined kernel span is 263.452 ms, busy union 259.147 ms (98.366%), and uncovered gaps 4.305 ms.

| Family | Launches | Raw duration (ms) | Union (ms) | Exclusive (ms) | Busy union | Busy exclusive |
|---|---:|---:|---:|---:|---:|---:|
| FFN dual projection + SwiGLU | 2112 | 91.105 | 91.105 | 45.943 | 35.16% | 17.73% |
| FlashAttention | 2112 | 71.786 | 71.786 | 25.688 | 27.70% | 9.91% |
| QKV + RoPE | 2112 | 64.287 | 64.287 | 28.061 | 24.81% | 10.83% |
| FFN linear2 + residual | 2112 | 56.439 | 56.439 | 4.526 | 21.78% | 1.75% |
| attention out-projection | 2112 | 43.606 | 43.606 | 3.321 | 16.83% | 1.28% |
| RMSNorm | 4224 | 24.881 | 24.881 | 2.457 | 9.60% | 0.95% |
| outer postConv | 704 | 18.178 | 18.178 | 1.710 | 7.01% | 0.66% |
| outer preConv | 704 | 13.718 | 13.718 | 1.223 | 5.29% | 0.47% |
| heads/frontend/other | 1932 | 11.360 | 8.534 | 5.606 | 3.29% | 2.16% |
| outer/trunk C384 BN + SiLU | 704 | 5.768 | 5.768 | 0.845 | 2.23% | 0.33% |
| outer C768 BN + SiLU | 64 | 0.457 | 0.457 | 0.155 | 0.18% | 0.06% |
| RMSNorm inverse-only | 0 | 0.000 | 0.000 | 0.000 | 0.00% | 0.00% |

Broad NCU covered 57 distinct S2 launch geometries.

| Family | Geometries | Median (us) | Max (us) | SM | DRAM | L2 | Issue | Active warps | Regs | Shared (B) | Waves/SM |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| FFN dual projection + SwiGLU | 1 | 42.240 | 42.240 | 66.6% | 18.3% | 71.8% | 21.8% | 15.3% | 168 | 49152 | 2.60 |
| FlashAttention | 1 | 30.688 | 30.688 | 32.0% | 39.9% | 37.7% | 25.7% | 22.5% | 168 | 16768 | 2.44 |
| QKV + RoPE | 1 | 30.464 | 30.464 | 46.0% | 15.2% | 53.6% | 20.8% | 14.9% | 240 | 49152 | 1.30 |
| FFN linear2 + residual | 1 | 23.104 | 23.104 | 61.6% | 51.8% | 68.0% | 15.4% | 8.3% | 162 | 65536 | 0.87 |
| heads/frontend/other | 40 | 7.504 | 22.080 | 31.2% | 2.7% | 39.8% | 14.5% | 12.7% | 244 | 49152 | 1.16 |
| outer postConv | 1 | 21.536 | 21.536 | 44.0% | 20.2% | 65.8% | 20.1% | 13.8% | 164 | 49152 | 0.87 |
| outer preConv | 1 | 16.576 | 16.576 | 57.5% | 48.1% | 61.9% | 15.3% | 8.3% | 162 | 81920 | 0.87 |
| attention out-projection | 8 | 13.472 | 15.456 | 30.2% | 50.5% | 40.4% | 14.5% | 8.4% | 186 | 32768 | 0.43 |
| outer C768 BN + SiLU | 1 | 7.712 | 7.712 | 32.2% | 50.6% | 47.0% | 56.9% | 79.0% | 31 | 0 | 2.29 |
| outer/trunk C384 BN + SiLU | 1 | 7.008 | 7.008 | 33.7% | 20.6% | 31.9% | 44.9% | 62.6% | 16 | 0 | 9.17 |
| RMSNorm | 1 | 5.376 | 5.376 | 32.3% | 30.4% | 41.2% | 33.8% | 55.7% | 40 | 0 | 0.76 |
