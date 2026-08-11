# Nsys stream interference report

- Timed iterations: 30; streams: 65, 82
- Kernels per forward: 65=344, 82=344
- Iteration start offset stream 82 - 65: median -3.78 us, p10..p90 -6.70..-3.36 us, range -8.38..34.18 us.

## Kernel families

| family | instances | total ms | median us | p90 us | peer overlap | S2/S1 median | matched |
|---|---:|---:|---:|---:|---:|---:|---:|
| fused_ffn | 1980 | 90.599 | 45.664 | 49.888 | 76.5% | n/a | 0 |
| linear2_residual | 1980 | 67.943 | 34.433 | 37.505 | 98.1% | n/a | 0 |
| library_gemm | 4140 | 67.827 | 16.256 | 22.592 | 95.0% | n/a | 0 |
| wide_qkv | 1980 | 51.451 | 24.672 | 31.968 | 77.1% | n/a | 0 |
| fa4 | 1980 | 30.561 | 15.328 | 16.256 | 47.5% | n/a | 0 |
| rmsnorm | 3960 | 19.305 | 4.608 | 6.432 | 100.0% | n/a | 0 |
| qk_rope | 1980 | 15.129 | 7.488 | 9.696 | 99.7% | n/a | 0 |
| affine_silu | 1380 | 8.672 | 6.064 | 9.664 | 97.5% | n/a | 0 |
| head_elementwise | 720 | 2.314 | 2.368 | 8.096 | 81.2% | n/a | 0 |
| cudnn | 180 | 1.502 | 1.760 | 21.824 | 25.3% | n/a | 0 |
| copy_reformat | 300 | 0.546 | 1.760 | 2.496 | 63.4% | n/a | 0 |
| sumChannelsNCHWKernel | 60 | 0.121 | 1.856 | 2.272 | 43.6% | n/a | 0 |

## Dominant interference pairs

| running family | dominant peer | instances | median overlap | median us | S2/S1 median | matched |
|---|---|---:|---:|---:|---:|---:|
| library_gemm | library_gemm | 1995 | 96.5% | 15.008 | n/a | 0 |
| rmsnorm | library_gemm | 1320 | 100.0% | 3.456 | n/a | 0 |
| affine_silu | library_gemm | 1040 | 100.0% | 6.992 | n/a | 0 |
| fused_ffn | fused_ffn | 1008 | 67.2% | 47.872 | n/a | 0 |
| rmsnorm | wide_qkv | 990 | 100.0% | 5.744 | n/a | 0 |
| wide_qkv | qk_rope | 990 | 63.2% | 22.593 | n/a | 0 |
| qk_rope | fa4 | 990 | 100.0% | 5.920 | n/a | 0 |
| rmsnorm | fused_ffn | 990 | 100.0% | 6.336 | n/a | 0 |
| qk_rope | wide_qkv | 990 | 100.0% | 9.440 | n/a | 0 |
| linear2_residual | fused_ffn | 990 | 99.1% | 32.448 | n/a | 0 |
| fa4 | library_gemm | 979 | 48.8% | 15.232 | n/a | 0 |
| fa4 | qk_rope | 979 | 45.3% | 15.488 | n/a | 0 |
| fused_ffn | linear2_residual | 972 | 86.5% | 43.584 | n/a | 0 |
| library_gemm | fused_ffn | 753 | 95.2% | 21.089 | n/a | 0 |
| linear2_residual | wide_qkv | 660 | 97.5% | 36.192 | n/a | 0 |
| rmsnorm | linear2_residual | 660 | 100.0% | 3.904 | n/a | 0 |
| wide_qkv | linear2_residual | 660 | 97.2% | 31.072 | n/a | 0 |
| head_elementwise | library_gemm | 533 | 100.0% | 2.272 | n/a | 0 |
| library_gemm | affine_silu | 423 | 95.4% | 21.440 | n/a | 0 |
| library_gemm | wide_qkv | 330 | 93.5% | 15.968 | n/a | 0 |
| linear2_residual | library_gemm | 330 | 97.6% | 34.689 | n/a | 0 |
| affine_silu | linear2_residual | 330 | 100.0% | 4.320 | n/a | 0 |
| library_gemm | linear2_residual | 330 | 96.3% | 25.888 | n/a | 0 |
| wide_qkv | library_gemm | 319 | 66.9% | 27.072 | n/a | 0 |
| library_gemm | head_elementwise | 222 | 84.3% | 6.816 | n/a | 0 |
| copy_reformat | library_gemm | 156 | 76.8% | 1.856 | n/a | 0 |
| cudnn | cudnn | 97 | 64.7% | 1.792 | n/a | 0 |
| copy_reformat | idle | 63 | 0.0% | 1.408 | n/a | 0 |
| copy_reformat | head_elementwise | 56 | 83.9% | 1.984 | n/a | 0 |
| head_elementwise | head_elementwise | 55 | 100.0% | 2.368 | n/a | 0 |
| library_gemm | cudnn | 50 | 94.6% | 3.504 | n/a | 0 |
| head_elementwise | copy_reformat | 46 | 79.8% | 3.760 | n/a | 0 |
| library_gemm | copy_reformat | 37 | 51.2% | 1.376 | n/a | 0 |
| cudnn | library_gemm | 33 | 33.7% | 21.569 | n/a | 0 |
| sumChannelsNCHWKernel | cudnn | 29 | 38.8% | 1.984 | n/a | 0 |
| head_elementwise | idle | 28 | 0.0% | 1.600 | n/a | 0 |
| head_elementwise | sumChannelsNCHWKernel | 25 | 60.0% | 1.280 | n/a | 0 |
| sumChannelsNCHWKernel | head_elementwise | 25 | 49.1% | 1.824 | n/a | 0 |
| cudnn | sumChannelsNCHWKernel | 25 | 61.4% | 1.440 | n/a | 0 |
| head_elementwise | cudnn | 23 | 33.6% | 8.128 | n/a | 0 |
