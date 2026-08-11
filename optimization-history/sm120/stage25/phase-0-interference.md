# Nsys stream interference report

- Timed iterations: 20; streams: 65, 82
- Kernels per forward: 65=344, 82=344
- Iteration start offset stream 82 - 65: median -3.66 us, p10..p90 -3.91..-3.35 us, range -3.97..-0.99 us.

## Kernel families

| family | instances | total ms | median us | p90 us | peer overlap | S2/S1 median | summed excess ms | matched |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| fused_ffn | 1320 | 60.004 | 45.441 | 49.316 | 76.7% | 1.085x | 4.542 | 1320 |
| library_gemm | 2760 | 45.093 | 16.273 | 22.400 | 95.0% | 1.686x | 19.410 | 2760 |
| linear2_residual | 1320 | 44.974 | 34.240 | 37.216 | 98.2% | 1.618x | 17.079 | 1320 |
| wide_qkv | 1320 | 34.093 | 24.337 | 31.748 | 77.3% | 1.252x | 8.329 | 1320 |
| fa4 | 1320 | 20.253 | 15.200 | 16.225 | 47.4% | 1.256x | 4.311 | 1320 |
| rmsnorm | 2640 | 12.858 | 4.720 | 6.400 | 100.0% | 1.759x | 5.691 | 2640 |
| qk_rope | 1320 | 9.985 | 7.408 | 9.632 | 99.7% | 1.787x | 4.471 | 1320 |
| affine_silu | 920 | 5.754 | 6.144 | 9.348 | 97.6% | 1.346x | 1.799 | 920 |
| head_elementwise | 480 | 1.540 | 2.368 | 8.068 | 79.7% | 1.123x | 0.337 | 480 |
| cudnn | 120 | 1.001 | 1.792 | 21.728 | 25.0% | 1.116x | 0.105 | 120 |
| copy_reformat | 200 | 0.360 | 1.728 | 2.496 | 58.4% | 1.180x | 0.072 | 200 |
| sumChannelsNCHWKernel | 40 | 0.076 | 1.856 | 2.179 | 44.0% | 1.137x | 0.011 | 40 |

## Dominant interference pairs

| running family | dominant peer | instances | median overlap | median us | S2/S1 median | matched |
|---|---|---:|---:|---:|---:|---:|
| library_gemm | fused_ffn | 531 | 95.2% | 21.184 | 2.450x | 531 |
| rmsnorm | fused_ffn | 660 | 100.0% | 6.304 | 2.314x | 660 |
| qk_rope | wide_qkv | 660 | 100.0% | 9.344 | 2.237x | 660 |
| rmsnorm | wide_qkv | 660 | 100.0% | 5.728 | 2.109x | 660 |
| library_gemm | linear2_residual | 220 | 96.7% | 26.576 | 1.813x | 220 |
| linear2_residual | wide_qkv | 440 | 97.6% | 35.904 | 1.698x | 440 |
| library_gemm | library_gemm | 1251 | 96.5% | 14.689 | 1.691x | 1251 |
| linear2_residual | library_gemm | 220 | 97.6% | 34.385 | 1.628x | 220 |
| wide_qkv | linear2_residual | 440 | 97.4% | 30.848 | 1.575x | 440 |
| copy_reformat | idle | 46 | 0.0% | 1.488 | 1.550x | 46 |
| library_gemm | cudnn | 30 | 95.3% | 4.032 | 1.537x | 30 |
| linear2_residual | fused_ffn | 660 | 99.1% | 32.256 | 1.524x | 660 |
| library_gemm | affine_silu | 357 | 95.1% | 20.736 | 1.475x | 357 |
| rmsnorm | linear2_residual | 440 | 100.0% | 3.872 | 1.424x | 440 |
| affine_silu | library_gemm | 689 | 100.0% | 7.296 | 1.414x | 689 |
| qk_rope | fa4 | 660 | 100.0% | 5.872 | 1.405x | 660 |
| fa4 | fa4 | 18 | 64.5% | 16.816 | 1.389x | 18 |
| wide_qkv | library_gemm | 215 | 67.0% | 26.976 | 1.381x | 215 |
| wide_qkv | rmsnorm | 5 | 66.2% | 25.920 | 1.330x | 5 |
| library_gemm | wide_qkv | 217 | 93.6% | 15.968 | 1.322x | 217 |
| rmsnorm | library_gemm | 880 | 100.0% | 3.488 | 1.279x | 880 |
| fa4 | qk_rope | 651 | 45.1% | 15.297 | 1.272x | 651 |
| head_elementwise | idle | 19 | 0.0% | 1.504 | 1.270x | 19 |
| affine_silu | linear2_residual | 220 | 100.0% | 4.256 | 1.259x | 220 |
| fa4 | library_gemm | 651 | 48.4% | 15.041 | 1.245x | 651 |
| library_gemm | head_elementwise | 132 | 85.5% | 6.752 | 1.222x | 132 |
| affine_silu | head_elementwise | 6 | 22.2% | 5.888 | 1.187x | 6 |
| head_elementwise | copy_reformat | 31 | 82.9% | 3.776 | 1.172x | 31 |
| copy_reformat | library_gemm | 95 | 73.3% | 1.824 | 1.167x | 95 |
| sumChannelsNCHWKernel | cudnn | 19 | 40.6% | 1.888 | 1.157x | 19 |
| cudnn | cudnn | 79 | 48.2% | 1.792 | 1.150x | 79 |
| wide_qkv | qk_rope | 660 | 63.3% | 22.400 | 1.148x | 660 |
| affine_silu | affine_silu | 5 | 36.4% | 5.632 | 1.135x | 5 |
| copy_reformat | head_elementwise | 45 | 78.6% | 2.432 | 1.132x | 45 |
| head_elementwise | library_gemm | 357 | 100.0% | 2.176 | 1.125x | 357 |
| cudnn | sumChannelsNCHWKernel | 17 | 79.5% | 1.440 | 1.125x | 17 |
| fused_ffn | fused_ffn | 667 | 67.6% | 47.713 | 1.123x | 667 |
| sumChannelsNCHWKernel | head_elementwise | 19 | 47.4% | 1.824 | 1.118x | 19 |
| head_elementwise | head_elementwise | 34 | 100.0% | 2.416 | 1.108x | 34 |
| cudnn | library_gemm | 19 | 33.0% | 21.664 | 1.106x | 19 |
