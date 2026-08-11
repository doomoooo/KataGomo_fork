# Stage 50 current-best macro checkpoint

Frozen S2 Nsys uses streams [114, 99] and reports 3216.453 nnEval/s. The combined kernel span is 265.661 ms, busy union 261.563 ms (98.457%), and uncovered gaps 4.099 ms.

| Family | Launches | Raw duration (ms) | Union (ms) | Exclusive (ms) | Busy union | Busy exclusive |
|---|---:|---:|---:|---:|---:|---:|
| FFN dual projection + SwiGLU | 2112 | 91.331 | 90.711 | 47.214 | 34.68% | 18.05% |
| FlashAttention | 2112 | 70.619 | 69.063 | 27.460 | 26.40% | 10.50% |
| QKV + RoPE | 2112 | 63.405 | 62.755 | 27.577 | 23.99% | 10.54% |
| FFN linear2 + residual | 2112 | 58.898 | 56.023 | 7.756 | 21.42% | 2.97% |
| attention out-projection | 2112 | 44.129 | 42.708 | 5.551 | 16.33% | 2.12% |
| RMSNorm | 4224 | 23.983 | 23.983 | 3.118 | 9.17% | 1.19% |
| outer postConv | 704 | 15.792 | 15.737 | 1.865 | 6.02% | 0.71% |
| outer preConv | 704 | 11.788 | 11.636 | 1.786 | 4.45% | 0.68% |
| heads/frontend/other | 1932 | 11.005 | 8.371 | 5.959 | 3.20% | 2.28% |
| outer C768 BN + SiLU | 768 | 6.876 | 6.876 | 0.328 | 2.63% | 0.13% |
| outer/trunk C384 BN + SiLU | 704 | 5.766 | 5.766 | 0.880 | 2.20% | 0.34% |
| RMSNorm inverse-only | 0 | 0.000 | 0.000 | 0.000 | 0.00% | 0.00% |

Broad NCU covered 77 distinct S2 launch geometries.

| Family | Geometries | Median (us) | Max (us) | SM | DRAM | L2 | Issue | Active warps | Regs | Shared (B) | Waves/SM |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| FFN dual projection + SwiGLU | 1 | 42.016 | 42.016 | 67.0% | 14.6% | 72.3% | 21.7% | 15.3% | 168 | 49152 | 2.60 |
| QKV + RoPE | 1 | 30.784 | 30.784 | 45.4% | 15.0% | 53.2% | 20.8% | 14.9% | 240 | 49152 | 1.30 |
| FlashAttention | 1 | 29.856 | 29.856 | 33.0% | 37.0% | 38.8% | 25.8% | 22.5% | 168 | 16768 | 2.44 |
| FFN linear2 + residual | 1 | 23.136 | 23.136 | 61.5% | 51.7% | 68.0% | 15.4% | 8.3% | 162 | 65536 | 0.87 |
| heads/frontend/other | 42 | 7.568 | 22.304 | 30.9% | 2.7% | 39.3% | 14.5% | 12.7% | 244 | 49152 | 1.16 |
| outer postConv | 10 | 20.400 | 20.576 | 46.2% | 37.2% | 72.9% | 15.0% | 13.5% | 186 | 40960 | 0.87 |
| outer preConv | 1 | 16.608 | 16.608 | 57.5% | 48.3% | 61.9% | 15.3% | 8.3% | 162 | 81920 | 0.87 |
| attention out-projection | 17 | 12.832 | 15.808 | 29.6% | 49.4% | 39.6% | 14.6% | 8.3% | 186 | 32768 | 0.43 |
| outer C768 BN + SiLU | 1 | 8.480 | 8.480 | 29.5% | 72.0% | 43.1% | 57.3% | 77.5% | 31 | 0 | 2.29 |
| RMSNorm | 1 | 7.616 | 7.616 | 22.5% | 56.6% | 29.0% | 34.2% | 55.6% | 40 | 0 | 0.76 |
| outer/trunk C384 BN + SiLU | 1 | 7.168 | 7.168 | 33.0% | 17.5% | 31.3% | 45.0% | 62.9% | 16 | 0 | 9.17 |
