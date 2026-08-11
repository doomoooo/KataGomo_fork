# Current-best macro checkpoint

Frozen S2 Nsys uses streams [114, 99] and reports 3418.728 nnEval/s. The combined kernel span is 255.247 ms, busy union 247.599 ms (97.004%), and uncovered gaps 7.648 ms.

| Family | Launches | Raw duration (ms) | Union (ms) | Exclusive (ms) | Busy union | Busy exclusive |
|---|---:|---:|---:|---:|---:|---:|
| FFN dual projection + SwiGLU | 2112 | 91.252 | 90.842 | 45.209 | 36.69% | 18.26% |
| QKV + RoPE | 2112 | 63.186 | 62.746 | 28.476 | 25.34% | 11.50% |
| FlashAttention | 2112 | 57.516 | 56.667 | 14.414 | 22.89% | 5.82% |
| FFN linear2 + residual | 2112 | 56.844 | 54.988 | 7.883 | 22.21% | 3.18% |
| attention out-projection | 2112 | 44.094 | 43.105 | 5.652 | 17.41% | 2.28% |
| RMSNorm | 4224 | 22.752 | 22.751 | 4.380 | 9.19% | 1.77% |
| outer postConv | 704 | 18.491 | 18.283 | 2.433 | 7.38% | 0.98% |
| outer preConv | 704 | 13.375 | 13.069 | 2.006 | 5.28% | 0.81% |
| heads/frontend/other | 1932 | 11.243 | 8.631 | 6.119 | 3.49% | 2.47% |
| outer/trunk C384 BN + SiLU | 704 | 5.786 | 5.786 | 1.082 | 2.34% | 0.44% |
| outer C768 BN + SiLU | 64 | 0.512 | 0.512 | 0.165 | 0.21% | 0.07% |
| RMSNorm inverse-only | 0 | 0.000 | 0.000 | 0.000 | 0.00% | 0.00% |

Broad NCU covered 63 distinct S2 launch geometries.

| Family | Geometries | Median (us) | Max (us) | SM | DRAM | L2 | Issue | Active warps | Regs | Shared (B) | Waves/SM |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| FFN dual projection + SwiGLU | 1 | 41.824 | 41.824 | 67.3% | 16.0% | 72.7% | 19.9% | 15.3% | 168 | 49152 | 2.60 |
| QKV + RoPE | 1 | 30.784 | 30.784 | 45.4% | 15.0% | 53.1% | 20.8% | 14.9% | 240 | 49152 | 1.30 |
| FlashAttention | 1 | 24.352 | 24.352 | 41.2% | 50.2% | 47.8% | 48.2% | 29.4% | 117 | 16768 | 1.83 |
| FFN linear2 + residual | 1 | 23.104 | 23.104 | 61.6% | 51.8% | 68.1% | 15.4% | 8.3% | 162 | 65536 | 0.87 |
| heads/frontend/other | 39 | 7.424 | 22.112 | 31.2% | 2.7% | 39.7% | 14.5% | 12.7% | 244 | 49152 | 1.16 |
| outer postConv | 1 | 21.600 | 21.600 | 43.8% | 20.1% | 65.7% | 20.2% | 13.8% | 164 | 49152 | 0.87 |
| outer preConv | 1 | 16.800 | 16.800 | 57.4% | 47.6% | 61.7% | 15.3% | 8.3% | 162 | 81920 | 0.87 |
| attention out-projection | 15 | 12.512 | 15.392 | 30.4% | 50.2% | 40.6% | 14.5% | 8.4% | 186 | 32768 | 0.43 |
| outer/trunk C384 BN + SiLU | 1 | 8.192 | 8.192 | 28.7% | 54.5% | 27.3% | 45.7% | 63.3% | 16 | 0 | 9.17 |
| outer C768 BN + SiLU | 1 | 7.616 | 7.616 | 32.7% | 49.1% | 47.7% | 57.5% | 78.0% | 31 | 0 | 2.29 |
| RMSNorm | 1 | 5.376 | 5.376 | 32.3% | 32.0% | 41.4% | 33.9% | 54.9% | 40 | 0 | 0.76 |
