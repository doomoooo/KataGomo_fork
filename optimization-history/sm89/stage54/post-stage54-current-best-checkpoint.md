# Stage 54 post-decision current-best macro checkpoint

Frozen S2 Nsys uses streams [114, 101] and reports 3205.757 nnEval/s. The combined kernel span is 263.100 ms, busy union 259.434 ms (98.607%), and uncovered gaps 3.666 ms.

| Family | Launches | Raw duration (ms) | Union (ms) | Exclusive (ms) | Busy union | Busy exclusive |
|---|---:|---:|---:|---:|---:|---:|
| FFN dual projection + SwiGLU | 2112 | 91.848 | 91.498 | 44.074 | 35.27% | 16.99% |
| FlashAttention | 2112 | 71.921 | 71.061 | 26.023 | 27.39% | 10.03% |
| QKV + RoPE | 2112 | 63.729 | 63.353 | 26.635 | 24.42% | 10.27% |
| FFN linear2 + residual | 2112 | 58.585 | 57.003 | 5.314 | 21.97% | 2.05% |
| attention out-projection | 2112 | 45.683 | 44.907 | 4.323 | 17.31% | 1.67% |
| RMSNorm | 4224 | 23.543 | 23.543 | 3.124 | 9.07% | 1.20% |
| outer postConv | 704 | 15.727 | 15.714 | 1.507 | 6.06% | 0.58% |
| outer preConv | 704 | 11.584 | 11.538 | 1.297 | 4.45% | 0.50% |
| heads/frontend/other | 1932 | 11.105 | 8.578 | 5.826 | 3.31% | 2.25% |
| outer C768 BN + SiLU | 768 | 6.790 | 6.790 | 0.216 | 2.62% | 0.08% |
| outer/trunk C384 BN + SiLU | 704 | 5.654 | 5.654 | 0.891 | 2.18% | 0.34% |
| RMSNorm inverse-only | 0 | 0.000 | 0.000 | 0.000 | 0.00% | 0.00% |

Broad NCU covered 75 distinct S2 launch geometries.

| Family | Geometries | Median (us) | Max (us) | SM | DRAM | L2 | Issue | Active warps | Regs | Shared (B) | Waves/SM |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| FFN dual projection + SwiGLU | 1 | 42.080 | 42.080 | 66.8% | 15.7% | 72.2% | 21.7% | 15.3% | 168 | 49152 | 2.60 |
| QKV + RoPE | 1 | 30.880 | 30.880 | 45.3% | 18.6% | 52.9% | 20.8% | 15.0% | 240 | 49152 | 1.30 |
| FlashAttention | 1 | 30.080 | 30.080 | 32.7% | 36.7% | 38.4% | 25.8% | 22.6% | 168 | 16768 | 2.44 |
| outer postConv | 11 | 20.448 | 25.184 | 37.2% | 60.7% | 59.1% | 14.9% | 16.1% | 186 | 40960 | 0.87 |
| FFN linear2 + residual | 1 | 23.104 | 23.104 | 61.5% | 51.8% | 68.0% | 15.4% | 8.3% | 162 | 65536 | 0.87 |
| heads/frontend/other | 42 | 7.568 | 21.984 | 31.3% | 2.7% | 39.8% | 14.5% | 12.7% | 244 | 49152 | 1.16 |
| outer preConv | 1 | 16.768 | 16.768 | 57.4% | 48.1% | 61.6% | 15.3% | 8.3% | 162 | 81920 | 0.87 |
| attention out-projection | 14 | 12.736 | 14.848 | 31.5% | 56.3% | 42.4% | 14.5% | 8.3% | 186 | 32768 | 0.43 |
| outer C768 BN + SiLU | 1 | 7.456 | 7.456 | 33.3% | 51.7% | 48.7% | 56.4% | 78.5% | 31 | 0 | 2.29 |
| outer/trunk C384 BN + SiLU | 1 | 7.136 | 7.136 | 33.3% | 5.1% | 31.6% | 45.1% | 62.5% | 16 | 0 | 9.17 |
| RMSNorm | 1 | 5.408 | 5.408 | 31.8% | 32.0% | 40.8% | 33.1% | 55.6% | 40 | 0 | 0.76 |
