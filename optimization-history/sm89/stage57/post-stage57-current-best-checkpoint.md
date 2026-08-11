# Current-best macro checkpoint

Frozen S2 Nsys uses streams [114, 105] and reports 3247.331 nnEval/s. The combined kernel span is 262.725 ms, busy union 258.851 ms (98.526%), and uncovered gaps 3.874 ms.

| Family | Launches | Raw duration (ms) | Union (ms) | Exclusive (ms) | Busy union | Busy exclusive |
|---|---:|---:|---:|---:|---:|---:|
| FFN dual projection + SwiGLU | 2112 | 91.249 | 90.821 | 45.749 | 35.09% | 17.67% |
| FlashAttention | 2112 | 71.637 | 70.533 | 26.385 | 27.25% | 10.19% |
| QKV + RoPE | 2112 | 63.486 | 63.029 | 27.878 | 24.35% | 10.77% |
| FFN linear2 + residual | 2112 | 57.678 | 55.663 | 6.866 | 21.50% | 2.65% |
| attention out-projection | 2112 | 42.279 | 41.273 | 5.124 | 15.94% | 1.98% |
| RMSNorm | 4224 | 23.561 | 23.561 | 3.443 | 9.10% | 1.33% |
| outer postConv | 704 | 17.863 | 17.740 | 2.140 | 6.85% | 0.83% |
| outer preConv | 704 | 13.951 | 13.637 | 1.722 | 5.27% | 0.67% |
| heads/frontend/other | 1932 | 11.358 | 8.719 | 6.067 | 3.37% | 2.34% |
| outer/trunk C384 BN + SiLU | 704 | 5.843 | 5.843 | 0.875 | 2.26% | 0.34% |
| outer C768 BN + SiLU | 64 | 0.467 | 0.465 | 0.167 | 0.18% | 0.06% |
| RMSNorm inverse-only | 0 | 0.000 | 0.000 | 0.000 | 0.00% | 0.00% |

Broad NCU covered 57 distinct S2 launch geometries.

| Family | Geometries | Median (us) | Max (us) | SM | DRAM | L2 | Issue | Active warps | Regs | Shared (B) | Waves/SM |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| FFN dual projection + SwiGLU | 1 | 42.336 | 42.336 | 66.4% | 17.3% | 71.7% | 21.6% | 15.3% | 168 | 49152 | 2.60 |
| QKV + RoPE | 1 | 30.496 | 30.496 | 45.9% | 15.2% | 53.7% | 20.8% | 14.9% | 240 | 49152 | 1.30 |
| FlashAttention | 1 | 30.080 | 30.080 | 32.7% | 36.7% | 38.5% | 25.8% | 22.5% | 168 | 16768 | 2.44 |
| FFN linear2 + residual | 1 | 23.072 | 23.072 | 61.7% | 51.8% | 68.1% | 15.4% | 8.3% | 162 | 65536 | 0.87 |
| heads/frontend/other | 38 | 7.328 | 22.592 | 27.1% | 59.5% | 29.1% | 15.8% | 8.3% | 186 | 40960 | 0.29 |
| outer postConv | 1 | 21.600 | 21.600 | 43.8% | 20.2% | 65.8% | 20.2% | 13.8% | 164 | 49152 | 0.87 |
| outer preConv | 1 | 16.544 | 16.544 | 57.8% | 48.3% | 62.3% | 15.3% | 8.3% | 162 | 81920 | 0.87 |
| attention out-projection | 10 | 13.104 | 14.880 | 31.5% | 45.6% | 42.0% | 14.6% | 8.4% | 186 | 32768 | 0.43 |
| outer C768 BN + SiLU | 1 | 8.736 | 8.736 | 28.4% | 73.1% | 41.5% | 56.9% | 78.2% | 31 | 0 | 2.29 |
| outer/trunk C384 BN + SiLU | 1 | 7.136 | 7.136 | 33.1% | 19.7% | 31.7% | 45.4% | 62.7% | 16 | 0 | 9.17 |
| RMSNorm | 1 | 6.944 | 6.944 | 24.8% | 51.9% | 32.0% | 33.8% | 55.0% | 40 | 0 | 0.76 |
