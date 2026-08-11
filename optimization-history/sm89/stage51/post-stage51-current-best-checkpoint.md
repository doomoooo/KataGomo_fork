# Stage 51 post-decision current-best macro checkpoint

Frozen S2 Nsys uses streams [114, 99] and reports 3206.437 nnEval/s. The combined kernel span is 269.347 ms, busy union 262.341 ms (97.399%), and uncovered gaps 7.006 ms.

| Family | Launches | Raw duration (ms) | Union (ms) | Exclusive (ms) | Busy union | Busy exclusive |
|---|---:|---:|---:|---:|---:|---:|
| FFN dual projection + SwiGLU | 2112 | 91.493 | 91.493 | 44.429 | 34.88% | 16.94% |
| FlashAttention | 2112 | 71.371 | 71.371 | 27.130 | 27.21% | 10.34% |
| QKV + RoPE | 2112 | 63.676 | 63.676 | 28.489 | 24.27% | 10.86% |
| FFN linear2 + residual | 2112 | 56.873 | 56.873 | 6.019 | 21.68% | 2.29% |
| attention out-projection | 2112 | 44.869 | 44.869 | 4.558 | 17.10% | 1.74% |
| RMSNorm | 4224 | 22.814 | 22.814 | 4.085 | 8.70% | 1.56% |
| outer postConv | 704 | 15.443 | 15.443 | 2.098 | 5.89% | 0.80% |
| outer preConv | 704 | 11.372 | 11.372 | 1.759 | 4.33% | 0.67% |
| heads/frontend/other | 1932 | 10.986 | 8.526 | 5.951 | 3.25% | 2.27% |
| outer C768 BN + SiLU | 768 | 6.589 | 6.589 | 0.453 | 2.51% | 0.17% |
| outer/trunk C384 BN + SiLU | 704 | 5.512 | 5.512 | 1.172 | 2.10% | 0.45% |
| RMSNorm inverse-only | 0 | 0.000 | 0.000 | 0.000 | 0.00% | 0.00% |

Broad NCU covered 77 distinct S2 launch geometries.

| Family | Geometries | Median (us) | Max (us) | SM | DRAM | L2 | Issue | Active warps | Regs | Shared (B) | Waves/SM |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| FFN dual projection + SwiGLU | 1 | 42.528 | 42.528 | 66.1% | 17.1% | 71.4% | 21.7% | 15.3% | 168 | 49152 | 2.60 |
| QKV + RoPE | 1 | 30.880 | 30.880 | 45.2% | 14.9% | 52.9% | 20.7% | 15.0% | 240 | 49152 | 1.30 |
| FlashAttention | 1 | 30.112 | 30.112 | 32.7% | 36.7% | 38.5% | 25.8% | 22.5% | 168 | 16768 | 2.44 |
| FFN linear2 + residual | 1 | 23.072 | 23.072 | 61.7% | 51.9% | 68.2% | 15.4% | 8.3% | 162 | 65536 | 0.87 |
| heads/frontend/other | 44 | 7.536 | 22.176 | 31.1% | 2.7% | 39.6% | 14.5% | 12.7% | 244 | 49152 | 1.16 |
| outer postConv | 13 | 20.352 | 20.832 | 45.6% | 34.5% | 71.9% | 14.9% | 13.5% | 186 | 40960 | 0.87 |
| outer preConv | 1 | 16.608 | 16.608 | 57.5% | 48.1% | 61.9% | 15.3% | 8.3% | 162 | 81920 | 0.87 |
| attention out-projection | 12 | 12.528 | 14.848 | 31.6% | 47.5% | 42.1% | 14.5% | 8.3% | 186 | 32768 | 0.43 |
| outer/trunk C384 BN + SiLU | 1 | 7.552 | 7.552 | 31.3% | 56.1% | 29.7% | 45.3% | 63.1% | 16 | 0 | 9.17 |
| outer C768 BN + SiLU | 1 | 7.456 | 7.456 | 33.5% | 51.6% | 48.9% | 57.3% | 78.2% | 31 | 0 | 2.29 |
| RMSNorm | 1 | 6.496 | 6.496 | 26.4% | 43.2% | 34.0% | 34.2% | 55.4% | 40 | 0 | 0.76 |
