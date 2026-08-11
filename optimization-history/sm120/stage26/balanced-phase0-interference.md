# Nsys stream interference report

- Timed iterations: 20; streams: 65, 82
- Kernels per forward: 65=344, 82=344
- Iteration start offset stream 82 - 65: median 373.52 us, p10..p90 45.71..469.73 us, range 4.45..499.81 us.

## Kernel families

| family | instances | total ms | median us | p90 us | peer overlap | S2/S1 median | summed excess ms | matched |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| fused_ffn | 1320 | 62.117 | 46.273 | 53.255 | 66.4% | n/a | 0.000 | 0 |
| library_gemm | 2760 | 46.958 | 19.329 | 24.576 | 96.2% | n/a | 0.000 | 0 |
| linear2_residual | 1320 | 43.567 | 30.704 | 42.560 | 97.1% | n/a | 0.000 | 0 |
| wide_qkv | 1320 | 33.849 | 26.400 | 29.600 | 78.3% | n/a | 0.000 | 0 |
| fa4 | 1320 | 22.189 | 16.000 | 20.000 | 64.4% | n/a | 0.000 | 0 |
| rmsnorm | 2640 | 13.813 | 5.344 | 6.688 | 98.8% | n/a | 0.000 | 0 |
| qk_rope | 1320 | 9.607 | 7.008 | 9.472 | 97.3% | n/a | 0.000 | 0 |
| affine_silu | 920 | 6.440 | 6.624 | 9.987 | 85.5% | n/a | 0.000 | 0 |
| head_elementwise | 480 | 1.774 | 2.704 | 8.096 | 83.1% | n/a | 0.000 | 0 |
| cudnn | 120 | 1.123 | 2.432 | 24.704 | 48.6% | n/a | 0.000 | 0 |
| copy_reformat | 200 | 0.482 | 2.112 | 4.384 | 91.3% | n/a | 0.000 | 0 |
| sumChannelsNCHWKernel | 40 | 0.108 | 2.688 | 3.779 | 92.9% | n/a | 0.000 | 0 |

## Dominant interference pairs

| running family | dominant peer | instances | median overlap | median us | S2/S1 median | matched |
|---|---|---:|---:|---:|---:|---:|
| library_gemm | library_gemm | 1043 | 96.8% | 17.889 | n/a | 0 |
| library_gemm | fused_ffn | 807 | 97.7% | 21.344 | n/a | 0 |
| rmsnorm | fused_ffn | 635 | 100.0% | 6.464 | n/a | 0 |
| rmsnorm | library_gemm | 634 | 100.0% | 3.584 | n/a | 0 |
| rmsnorm | linear2_residual | 501 | 100.0% | 5.792 | n/a | 0 |
| rmsnorm | fa4 | 476 | 100.0% | 4.672 | n/a | 0 |
| linear2_residual | linear2_residual | 445 | 98.5% | 40.257 | n/a | 0 |
| fused_ffn | fused_ffn | 425 | 70.5% | 45.920 | n/a | 0 |
| fa4 | library_gemm | 413 | 52.1% | 15.104 | n/a | 0 |
| fused_ffn | library_gemm | 388 | 46.6% | 44.256 | n/a | 0 |
| qk_rope | library_gemm | 384 | 100.0% | 5.344 | n/a | 0 |
| wide_qkv | fused_ffn | 377 | 98.0% | 28.288 | n/a | 0 |
| qk_rope | linear2_residual | 375 | 100.0% | 8.928 | n/a | 0 |
| fa4 | linear2_residual | 375 | 77.9% | 18.944 | n/a | 0 |
| library_gemm | wide_qkv | 364 | 95.1% | 19.729 | n/a | 0 |
| linear2_residual | wide_qkv | 356 | 96.7% | 28.672 | n/a | 0 |
| wide_qkv | library_gemm | 343 | 60.3% | 23.136 | n/a | 0 |
| fused_ffn | wide_qkv | 307 | 72.4% | 51.904 | n/a | 0 |
| linear2_residual | fused_ffn | 299 | 96.7% | 29.600 | n/a | 0 |
| wide_qkv | linear2_residual | 263 | 88.6% | 28.545 | n/a | 0 |
| affine_silu | fa4 | 262 | 100.0% | 6.048 | n/a | 0 |
| rmsnorm | wide_qkv | 262 | 100.0% | 5.392 | n/a | 0 |
| qk_rope | wide_qkv | 260 | 100.0% | 9.216 | n/a | 0 |
| wide_qkv | qk_rope | 260 | 64.2% | 22.032 | n/a | 0 |
| qk_rope | fa4 | 260 | 100.0% | 5.856 | n/a | 0 |
| fa4 | qk_rope | 256 | 46.1% | 15.200 | n/a | 0 |
| fa4 | rmsnorm | 218 | 60.1% | 15.873 | n/a | 0 |
| library_gemm | linear2_residual | 206 | 96.8% | 19.424 | n/a | 0 |
| affine_silu | library_gemm | 196 | 100.0% | 6.720 | n/a | 0 |
| fused_ffn | linear2_residual | 184 | 81.5% | 48.561 | n/a | 0 |
| library_gemm | fa4 | 175 | 95.7% | 15.040 | n/a | 0 |
| affine_silu | rmsnorm | 155 | 47.5% | 6.784 | n/a | 0 |
| affine_silu | wide_qkv | 120 | 100.0% | 6.192 | n/a | 0 |
| affine_silu | linear2_residual | 118 | 100.0% | 7.744 | n/a | 0 |
| head_elementwise | library_gemm | 114 | 100.0% | 3.136 | n/a | 0 |
| head_elementwise | fused_ffn | 99 | 100.0% | 2.368 | n/a | 0 |
| head_elementwise | linear2_residual | 85 | 100.0% | 4.640 | n/a | 0 |
| rmsnorm | affine_silu | 82 | 92.6% | 3.616 | n/a | 0 |
| linear2_residual | fa4 | 69 | 96.7% | 26.849 | n/a | 0 |
| copy_reformat | library_gemm | 58 | 100.0% | 2.240 | n/a | 0 |
