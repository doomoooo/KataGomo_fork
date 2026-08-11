# Current-best macro checkpoint

Frozen S2 Nsys uses streams [114, 100] and reports 3401.752 nnEval/s. The combined kernel span is 260.863 ms, busy union 248.671 ms (95.326%), and uncovered gaps 12.192 ms.

| Family | Launches | Raw duration (ms) | Union (ms) | Exclusive (ms) | Busy union | Busy exclusive |
|---|---:|---:|---:|---:|---:|---:|
| FFN dual projection + SwiGLU | 2112 | 92.494 | 92.215 | 45.504 | 37.08% | 18.30% |
| QKV + RoPE | 2112 | 63.258 | 62.991 | 28.534 | 25.33% | 11.47% |
| FlashAttention | 2112 | 57.614 | 57.045 | 14.148 | 22.94% | 5.69% |
| FFN linear2 + residual | 2112 | 56.689 | 55.459 | 7.232 | 22.30% | 2.91% |
| attention out-projection | 2112 | 44.048 | 43.389 | 5.247 | 17.45% | 2.11% |
| RMSNorm | 4224 | 22.942 | 22.942 | 4.296 | 9.23% | 1.73% |
| outer postConv | 704 | 18.485 | 18.367 | 2.506 | 7.39% | 1.01% |
| outer preConv | 704 | 13.591 | 13.381 | 1.886 | 5.38% | 0.76% |
| heads/frontend/other | 1932 | 11.123 | 8.616 | 6.076 | 3.46% | 2.44% |
| outer/trunk C384 BN + SiLU | 704 | 5.756 | 5.756 | 1.084 | 2.31% | 0.44% |
| outer C768 BN + SiLU | 64 | 0.507 | 0.507 | 0.161 | 0.20% | 0.06% |
| RMSNorm inverse-only | 0 | 0.000 | 0.000 | 0.000 | 0.00% | 0.00% |

Broad NCU covered 62 distinct S2 launch geometries.

| Family | Geometries | Median (us) | Max (us) | SM | DRAM | L2 | Issue | Active warps | Regs | Shared (B) | Waves/SM |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| FFN dual projection + SwiGLU | 1 | 42.208 | 42.208 | 66.6% | 17.4% | 72.0% | 21.7% | 15.3% | 168 | 49152 | 2.60 |
| QKV + RoPE | 1 | 30.400 | 30.400 | 46.1% | 15.2% | 53.9% | 20.8% | 14.9% | 240 | 49152 | 1.30 |
| FlashAttention | 1 | 23.712 | 23.712 | 42.5% | 46.6% | 49.0% | 48.3% | 29.4% | 117 | 16768 | 1.83 |
| FFN linear2 + residual | 1 | 23.104 | 23.104 | 61.6% | 51.8% | 68.0% | 15.4% | 8.3% | 162 | 65536 | 0.87 |
| heads/frontend/other | 42 | 7.456 | 22.144 | 31.4% | 2.7% | 39.9% | 14.5% | 12.7% | 244 | 49152 | 1.16 |
| outer postConv | 1 | 21.504 | 21.504 | 44.0% | 24.4% | 66.0% | 20.2% | 13.8% | 164 | 49152 | 0.87 |
| outer preConv | 1 | 16.544 | 16.544 | 57.8% | 48.3% | 62.2% | 15.3% | 8.3% | 162 | 81920 | 0.87 |
| attention out-projection | 11 | 12.672 | 15.456 | 30.3% | 44.0% | 40.5% | 14.5% | 8.4% | 186 | 32768 | 0.43 |
| outer C768 BN + SiLU | 1 | 8.672 | 8.672 | 28.5% | 73.3% | 41.7% | 57.2% | 77.7% | 31 | 0 | 2.29 |
| outer/trunk C384 BN + SiLU | 1 | 7.488 | 7.488 | 31.6% | 58.4% | 30.0% | 46.7% | 63.3% | 16 | 0 | 9.17 |
| RMSNorm | 1 | 6.304 | 6.304 | 27.4% | 43.9% | 35.2% | 34.0% | 55.5% | 40 | 0 | 0.76 |
