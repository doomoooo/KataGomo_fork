# Nsys stream interference report

- Timed iterations: 20; streams: 65, 82
- Kernels per forward: 65=344, 82=344
- Iteration start offset stream 82 - 65: median 2247.64 us, p10..p90 2209.73..2262.00 us, range 1506.13..2295.16 us.

## Kernel families

| family | instances | total ms | median us | p90 us | peer overlap | S2/S1 median | summed excess ms | matched |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| fused_ffn | 1320 | 63.318 | 47.297 | 55.200 | 65.3% | 1.109x | 7.723 | 1320 |
| library_gemm | 2760 | 46.923 | 19.360 | 23.553 | 96.0% | 1.825x | 21.239 | 2760 |
| linear2_residual | 1320 | 38.725 | 29.408 | 32.387 | 95.1% | 1.386x | 10.830 | 1320 |
| wide_qkv | 1320 | 34.140 | 26.944 | 29.120 | 80.3% | 1.382x | 8.377 | 1320 |
| fa4 | 1320 | 24.277 | 18.913 | 21.568 | 80.6% | 1.569x | 8.335 | 1320 |
| rmsnorm | 2640 | 12.447 | 4.416 | 6.592 | 97.2% | 1.630x | 5.283 | 2640 |
| qk_rope | 1320 | 7.661 | 5.760 | 6.720 | 96.5% | 1.374x | 2.150 | 1320 |
| affine_silu | 920 | 5.682 | 6.400 | 7.424 | 73.5% | 1.401x | 1.727 | 920 |
| head_elementwise | 480 | 1.713 | 3.009 | 8.010 | 80.2% | 1.281x | 0.510 | 480 |
| cudnn | 120 | 1.114 | 2.864 | 23.434 | 41.0% | 1.417x | 0.218 | 120 |
| copy_reformat | 200 | 0.440 | 2.336 | 3.040 | 93.1% | 1.429x | 0.152 | 200 |
| sumChannelsNCHWKernel | 40 | 0.119 | 3.072 | 3.808 | 95.5% | 1.882x | 0.054 | 40 |

## Dominant interference pairs

| running family | dominant peer | instances | median overlap | median us | S2/S1 median | matched |
|---|---|---:|---:|---:|---:|---:|
| cudnn | wide_qkv | 23 | 96.7% | 3.264 | 2.550x | 23 |
| rmsnorm | fused_ffn | 690 | 100.0% | 6.272 | 2.299x | 690 |
| library_gemm | wide_qkv | 458 | 95.3% | 19.857 | 2.287x | 458 |
| head_elementwise | wide_qkv | 41 | 100.0% | 5.120 | 2.275x | 41 |
| library_gemm | fused_ffn | 888 | 97.8% | 20.464 | 2.260x | 888 |
| qk_rope | wide_qkv | 42 | 100.0% | 9.216 | 2.201x | 42 |
| qk_rope | fused_ffn | 12 | 100.0% | 8.784 | 2.124x | 12 |
| affine_silu | fused_ffn | 25 | 100.0% | 7.104 | 2.105x | 25 |
| sumChannelsNCHWKernel | wide_qkv | 27 | 100.0% | 3.392 | 2.078x | 27 |
| rmsnorm | wide_qkv | 48 | 100.0% | 5.553 | 2.059x | 48 |
| copy_reformat | wide_qkv | 16 | 90.5% | 2.288 | 1.950x | 16 |
| affine_silu | wide_qkv | 108 | 100.0% | 6.480 | 1.879x | 108 |
| copy_reformat | rmsnorm | 19 | 100.0% | 1.952 | 1.733x | 19 |
| head_elementwise | rmsnorm | 58 | 88.3% | 3.520 | 1.730x | 58 |
| fa4 | linear2_residual | 648 | 95.7% | 20.384 | 1.685x | 648 |
| rmsnorm | fa4 | 738 | 100.0% | 4.416 | 1.620x | 738 |
| rmsnorm | library_gemm | 440 | 89.9% | 4.416 | 1.616x | 440 |
| fa4 | fused_ffn | 55 | 66.8% | 19.264 | 1.597x | 55 |
| linear2_residual | wide_qkv | 126 | 97.1% | 32.784 | 1.562x | 126 |
| rmsnorm | rmsnorm | 13 | 91.2% | 4.224 | 1.553x | 13 |
| library_gemm | library_gemm | 932 | 97.4% | 19.168 | 1.529x | 932 |
| wide_qkv | linear2_residual | 126 | 94.0% | 29.904 | 1.527x | 126 |
| fa4 | library_gemm | 203 | 72.0% | 17.985 | 1.523x | 203 |
| cudnn | qk_rope | 15 | 68.4% | 2.336 | 1.521x | 15 |
| copy_reformat | linear2_residual | 107 | 100.0% | 2.304 | 1.500x | 107 |
| library_gemm | qk_rope | 16 | 91.6% | 3.856 | 1.493x | 16 |
| rmsnorm | linear2_residual | 496 | 100.0% | 4.000 | 1.476x | 496 |
| affine_silu | fa4 | 389 | 100.0% | 5.056 | 1.430x | 389 |
| affine_silu | linear2_residual | 24 | 100.0% | 7.329 | 1.421x | 24 |
| wide_qkv | fused_ffn | 660 | 94.9% | 27.808 | 1.419x | 660 |
| cudnn | fused_ffn | 9 | 100.0% | 2.176 | 1.417x | 9 |
| linear2_residual | fused_ffn | 411 | 96.6% | 29.952 | 1.409x | 411 |
| qk_rope | fa4 | 42 | 100.0% | 5.824 | 1.393x | 42 |
| qk_rope | linear2_residual | 648 | 100.0% | 5.824 | 1.392x | 648 |
| cudnn | linear2_residual | 25 | 100.0% | 2.048 | 1.375x | 25 |
| rmsnorm | cudnn | 9 | 100.0% | 3.776 | 1.372x | 9 |
| linear2_residual | fa4 | 648 | 97.1% | 28.960 | 1.372x | 648 |
| library_gemm | fa4 | 190 | 96.0% | 15.376 | 1.369x | 190 |
| library_gemm | rmsnorm | 4 | 85.8% | 4.736 | 1.352x | 4 |
| library_gemm | linear2_residual | 187 | 100.0% | 6.433 | 1.348x | 187 |
