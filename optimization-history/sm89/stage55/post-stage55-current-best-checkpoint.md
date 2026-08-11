# Stage 55 post-decision current-best macro checkpoint

Frozen S2 Nsys uses streams [114, 99] and reports 3211.621 nnEval/s. The combined kernel span is 268.346 ms, busy union 262.615 ms (97.864%), and uncovered gaps 5.731 ms.

| Family | Launches | Raw duration (ms) | Union (ms) | Exclusive (ms) | Busy union | Busy exclusive |
|---|---:|---:|---:|---:|---:|---:|
| FFN dual projection + SwiGLU | 2112 | 91.683 | 91.570 | 45.080 | 34.87% | 17.17% |
| FlashAttention | 2112 | 71.418 | 71.084 | 27.256 | 27.07% | 10.38% |
| QKV + RoPE | 2112 | 63.569 | 63.428 | 28.533 | 24.15% | 10.87% |
| FFN linear2 + residual | 2112 | 57.019 | 56.395 | 6.647 | 21.47% | 2.53% |
| attention out-projection | 2112 | 44.775 | 44.470 | 4.924 | 16.93% | 1.88% |
| RMSNorm | 4224 | 22.993 | 22.993 | 4.125 | 8.76% | 1.57% |
| outer postConv | 704 | 15.487 | 15.471 | 2.014 | 5.89% | 0.77% |
| outer preConv | 704 | 11.256 | 11.220 | 1.797 | 4.27% | 0.68% |
| heads/frontend/other | 1932 | 10.962 | 8.482 | 5.931 | 3.23% | 2.26% |
| outer C768 BN + SiLU | 768 | 6.621 | 6.621 | 0.468 | 2.52% | 0.18% |
| outer/trunk C384 BN + SiLU | 704 | 5.562 | 5.562 | 1.160 | 2.12% | 0.44% |
| RMSNorm inverse-only | 0 | 0.000 | 0.000 | 0.000 | 0.00% | 0.00% |

Broad NCU covered 82 distinct S2 launch geometries.

| Family | Geometries | Median (us) | Max (us) | SM | DRAM | L2 | Issue | Active warps | Regs | Shared (B) | Waves/SM |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| FFN dual projection + SwiGLU | 1 | 42.304 | 42.304 | 66.5% | 15.6% | 71.8% | 21.7% | 15.3% | 168 | 49152 | 2.60 |
| QKV + RoPE | 1 | 30.848 | 30.848 | 45.3% | 18.9% | 53.0% | 20.8% | 15.0% | 240 | 49152 | 1.30 |
| FlashAttention | 1 | 30.304 | 30.304 | 32.4% | 36.4% | 38.2% | 25.7% | 22.5% | 168 | 16768 | 2.44 |
| FFN linear2 + residual | 1 | 23.104 | 23.104 | 61.6% | 51.8% | 68.0% | 15.4% | 8.3% | 162 | 65536 | 0.87 |
| heads/frontend/other | 46 | 7.568 | 22.144 | 31.1% | 2.7% | 39.6% | 14.5% | 12.7% | 244 | 49152 | 1.16 |
| outer postConv | 13 | 20.416 | 20.672 | 46.1% | 35.7% | 72.5% | 15.1% | 13.5% | 186 | 40960 | 0.87 |
| outer preConv | 1 | 16.544 | 16.544 | 57.8% | 48.3% | 62.2% | 15.3% | 8.3% | 162 | 81920 | 0.87 |
| attention out-projection | 15 | 12.768 | 14.304 | 32.9% | 48.5% | 43.8% | 14.5% | 8.4% | 186 | 32768 | 0.43 |
| outer C768 BN + SiLU | 1 | 7.424 | 7.424 | 33.5% | 52.5% | 48.9% | 57.5% | 77.9% | 31 | 0 | 2.29 |
| outer/trunk C384 BN + SiLU | 1 | 7.008 | 7.008 | 33.7% | 20.5% | 32.1% | 45.3% | 62.9% | 16 | 0 | 9.17 |
| RMSNorm | 1 | 5.184 | 5.184 | 33.3% | 31.4% | 42.6% | 33.2% | 56.0% | 40 | 0 | 0.76 |
