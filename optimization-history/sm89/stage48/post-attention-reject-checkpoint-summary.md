# Stage 48 current-best macro checkpoint

Frozen S2 Nsys uses streams [114, 99] and reports 3164.779 nnEval/s. The combined kernel span is 268.151 ms, busy union 264.696 ms (98.711%), and uncovered gaps 3.456 ms.

| Family | Launches | Raw duration (ms) | Union (ms) | Exclusive (ms) | Busy union | Busy exclusive |
|---|---:|---:|---:|---:|---:|---:|
| FFN dual projection + SwiGLU | 2112 | 90.763 | 89.638 | 53.091 | 33.86% | 20.06% |
| FlashAttention | 2112 | 68.275 | 65.529 | 29.238 | 24.76% | 11.05% |
| QKV + RoPE | 2112 | 63.027 | 61.876 | 27.414 | 23.38% | 10.36% |
| FFN linear2 + residual | 2112 | 61.286 | 56.200 | 10.459 | 21.23% | 3.95% |
| attention out-projection | 2112 | 42.195 | 39.694 | 6.684 | 15.00% | 2.53% |
| RMSNorm | 4224 | 25.884 | 25.884 | 2.097 | 9.78% | 0.79% |
| outer postConv | 704 | 15.912 | 15.841 | 2.433 | 5.98% | 0.92% |
| outer preConv | 704 | 12.502 | 12.314 | 1.909 | 4.65% | 0.72% |
| heads/frontend/other | 2124 | 11.607 | 9.007 | 6.023 | 3.40% | 2.28% |
| outer C768 BN + SiLU | 768 | 7.022 | 7.022 | 0.345 | 2.65% | 0.13% |
| outer/trunk C384 BN + SiLU | 704 | 6.023 | 6.023 | 0.669 | 2.28% | 0.25% |
| RMSNorm inverse-only | 0 | 0.000 | 0.000 | 0.000 | 0.00% | 0.00% |

Broad NCU covered 62 distinct S2 launch geometries.

| Family | Geometries | Median (us) | Max (us) | SM | DRAM | L2 | Issue | Active warps | Regs | Shared (B) | Waves/SM |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| FFN dual projection + SwiGLU | 1 | 42.304 | 42.304 | 66.5% | 16.7% | 71.8% | 21.6% | 15.3% | 168 | 49152 | 2.60 |
| FlashAttention | 1 | 30.528 | 30.528 | 32.2% | 36.2% | 38.0% | 25.7% | 22.5% | 168 | 16768 | 2.44 |
| QKV + RoPE | 1 | 30.368 | 30.368 | 46.1% | 15.2% | 53.9% | 20.8% | 14.9% | 240 | 49152 | 1.30 |
| FFN linear2 + residual | 1 | 23.072 | 23.072 | 61.7% | 51.8% | 68.2% | 15.4% | 8.3% | 162 | 65536 | 0.87 |
| heads/frontend/other | 40 | 7.040 | 22.176 | 31.1% | 2.7% | 39.6% | 14.5% | 12.7% | 244 | 49152 | 1.16 |
| outer postConv | 7 | 20.384 | 20.480 | 46.5% | 35.1% | 73.3% | 14.9% | 13.5% | 186 | 40960 | 0.87 |
| outer preConv | 1 | 16.640 | 16.640 | 57.4% | 48.0% | 61.8% | 15.3% | 8.3% | 162 | 81920 | 0.87 |
| attention out-projection | 7 | 12.928 | 14.112 | 33.3% | 47.1% | 44.4% | 14.6% | 8.3% | 186 | 32768 | 0.43 |
| outer/trunk C384 BN + SiLU | 1 | 7.936 | 7.936 | 29.6% | 55.3% | 28.2% | 45.0% | 62.8% | 16 | 0 | 9.17 |
| outer C768 BN + SiLU | 1 | 7.744 | 7.744 | 32.2% | 51.6% | 47.0% | 56.9% | 76.6% | 31 | 0 | 2.29 |
| RMSNorm | 1 | 6.752 | 6.752 | 25.4% | 41.7% | 32.7% | 33.4% | 55.1% | 40 | 0 | 0.76 |
