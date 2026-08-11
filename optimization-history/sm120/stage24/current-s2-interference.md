# Nsys stream interference report

- Timed iterations: 30; streams: 65, 82
- Kernels per forward: 65=344, 82=344
- Iteration start offset stream 82 - 65: median -3.78 us, p10..p90 -6.70..-3.36 us, range -8.38..34.18 us.

## Kernel families

| family | instances | total ms | median us | p90 us | peer overlap | S2/S1 median | summed excess ms | matched |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| fused_ffn | 1980 | 90.599 | 45.664 | 49.888 | 76.5% | 1.100x | 7.843 | 1980 |
| linear2_residual | 1980 | 67.943 | 34.433 | 37.505 | 98.1% | 1.633x | 26.233 | 1980 |
| library_gemm | 4140 | 67.827 | 16.256 | 22.592 | 95.0% | 1.703x | 29.420 | 4140 |
| wide_qkv | 1980 | 51.451 | 24.672 | 31.968 | 77.1% | 1.278x | 12.955 | 1980 |
| fa4 | 1980 | 30.561 | 15.328 | 16.256 | 47.5% | 1.272x | 6.727 | 1980 |
| rmsnorm | 3960 | 19.305 | 4.608 | 6.432 | 100.0% | 1.719x | 8.615 | 3960 |
| qk_rope | 1980 | 15.129 | 7.488 | 9.696 | 99.7% | 1.812x | 6.906 | 1980 |
| affine_silu | 1380 | 8.672 | 6.064 | 9.664 | 97.5% | 1.330x | 2.757 | 1380 |
| head_elementwise | 720 | 2.314 | 2.368 | 8.096 | 81.2% | 1.113x | 0.521 | 720 |
| cudnn | 180 | 1.502 | 1.760 | 21.824 | 25.3% | 1.122x | 0.159 | 180 |
| copy_reformat | 300 | 0.546 | 1.760 | 2.496 | 63.4% | 1.167x | 0.116 | 300 |
| sumChannelsNCHWKernel | 60 | 0.121 | 1.856 | 2.272 | 43.6% | 1.137x | 0.023 | 60 |

## Dominant interference pairs

| running family | dominant peer | instances | median overlap | median us | S2/S1 median | matched |
|---|---|---:|---:|---:|---:|---:|
| library_gemm | fused_ffn | 753 | 95.2% | 21.089 | 2.449x | 753 |
| rmsnorm | fused_ffn | 990 | 100.0% | 6.336 | 2.334x | 990 |
| qk_rope | wide_qkv | 990 | 100.0% | 9.440 | 2.274x | 990 |
| rmsnorm | wide_qkv | 990 | 100.0% | 5.744 | 2.131x | 990 |
| library_gemm | linear2_residual | 330 | 96.3% | 25.888 | 1.773x | 330 |
| linear2_residual | wide_qkv | 660 | 97.5% | 36.192 | 1.717x | 660 |
| library_gemm | library_gemm | 1995 | 96.5% | 15.008 | 1.714x | 1995 |
| linear2_residual | library_gemm | 330 | 97.6% | 34.689 | 1.647x | 330 |
| wide_qkv | linear2_residual | 660 | 97.2% | 31.072 | 1.588x | 660 |
| linear2_residual | fused_ffn | 990 | 99.1% | 32.448 | 1.538x | 990 |
| library_gemm | affine_silu | 423 | 95.4% | 21.440 | 1.494x | 423 |
| copy_reformat | idle | 63 | 0.0% | 1.408 | 1.482x | 63 |
| rmsnorm | linear2_residual | 660 | 100.0% | 3.904 | 1.446x | 660 |
| qk_rope | fa4 | 990 | 100.0% | 5.920 | 1.430x | 990 |
| fa4 | fa4 | 22 | 65.1% | 16.752 | 1.403x | 22 |
| wide_qkv | library_gemm | 319 | 66.9% | 27.072 | 1.395x | 319 |
| sumChannelsNCHWKernel | idle | 4 | 0.0% | 2.272 | 1.392x | 4 |
| affine_silu | library_gemm | 1040 | 100.0% | 6.992 | 1.366x | 1040 |
| library_gemm | cudnn | 50 | 94.6% | 3.504 | 1.364x | 50 |
| wide_qkv | rmsnorm | 11 | 65.8% | 26.144 | 1.354x | 11 |
| head_elementwise | idle | 28 | 0.0% | 1.600 | 1.351x | 28 |
| library_gemm | wide_qkv | 330 | 93.5% | 15.968 | 1.325x | 330 |
| fa4 | qk_rope | 979 | 45.3% | 15.488 | 1.286x | 979 |
| affine_silu | linear2_residual | 330 | 100.0% | 4.320 | 1.276x | 330 |
| rmsnorm | library_gemm | 1320 | 100.0% | 3.456 | 1.275x | 1320 |
| fa4 | library_gemm | 979 | 48.8% | 15.232 | 1.260x | 979 |
| sumChannelsNCHWKernel | cudnn | 29 | 38.8% | 1.984 | 1.216x | 29 |
| library_gemm | head_elementwise | 222 | 84.3% | 6.816 | 1.198x | 222 |
| copy_reformat | library_gemm | 156 | 76.8% | 1.856 | 1.196x | 156 |
| affine_silu | head_elementwise | 10 | 25.3% | 5.856 | 1.181x | 10 |
| head_elementwise | copy_reformat | 46 | 79.8% | 3.760 | 1.167x | 46 |
| copy_reformat | copy_reformat | 4 | 78.5% | 1.104 | 1.167x | 4 |
| wide_qkv | qk_rope | 990 | 63.2% | 22.593 | 1.159x | 990 |
| fused_ffn | fused_ffn | 1008 | 67.2% | 47.872 | 1.138x | 1008 |
| cudnn | head_elementwise | 5 | 31.1% | 22.048 | 1.133x | 5 |
| copy_reformat | head_elementwise | 56 | 83.9% | 1.984 | 1.132x | 56 |
| cudnn | cudnn | 97 | 64.7% | 1.792 | 1.125x | 97 |
| head_elementwise | library_gemm | 533 | 100.0% | 2.272 | 1.125x | 533 |
| cudnn | idle | 12 | 0.0% | 1.712 | 1.125x | 12 |
| sumChannelsNCHWKernel | head_elementwise | 25 | 49.1% | 1.824 | 1.118x | 25 |
