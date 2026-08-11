# Nsys stream interference report

- Timed iterations: 30; streams: 65, 82
- Kernels per forward: 65=344, 82=344
- Iteration start offset stream 82 - 65: median -4.14 us, p10..p90 -69.60..-3.36 us, range -69.73..-1.06 us.

## Kernel families

| family | instances | total ms | median us | p90 us | peer overlap | S2/S1 median | summed excess ms | matched |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| fused_ffn | 1980 | 90.160 | 44.993 | 52.353 | 73.3% | 1.158x | 13.444 | 1980 |
| library_gemm | 4140 | 64.871 | 17.376 | 20.896 | 95.3% | 1.686x | 26.945 | 4140 |
| linear2_residual | 1980 | 63.488 | 32.097 | 35.904 | 97.7% | 1.540x | 22.220 | 1980 |
| wide_qkv | 1980 | 49.962 | 25.601 | 29.152 | 82.3% | 1.323x | 11.824 | 1980 |
| fa4 | 1980 | 33.512 | 15.553 | 21.376 | 67.0% | 1.308x | 9.963 | 1980 |
| rmsnorm | 3960 | 15.602 | 3.456 | 5.408 | 99.0% | 1.421x | 5.924 | 3960 |
| qk_rope | 1980 | 13.134 | 5.856 | 9.408 | 99.0% | 1.423x | 4.987 | 1980 |
| affine_silu | 1380 | 8.216 | 5.665 | 9.600 | 86.4% | 1.358x | 2.371 | 1380 |
| head_elementwise | 720 | 2.193 | 2.400 | 8.000 | 71.6% | 1.129x | 0.408 | 720 |
| cudnn | 180 | 1.522 | 1.936 | 21.504 | 35.5% | 1.104x | 0.185 | 180 |
| copy_reformat | 300 | 0.559 | 1.760 | 2.787 | 70.0% | 1.206x | 0.133 | 300 |
| sumChannelsNCHWKernel | 60 | 0.129 | 2.000 | 2.787 | 64.8% | 1.202x | 0.029 | 60 |

## Dominant interference pairs

| running family | dominant peer | instances | median overlap | median us | S2/S1 median | matched |
|---|---|---:|---:|---:|---:|---:|
| cudnn | wide_qkv | 21 | 100.0% | 3.232 | 2.525x | 21 |
| library_gemm | fused_ffn | 999 | 95.9% | 19.584 | 2.307x | 999 |
| library_gemm | wide_qkv | 483 | 94.8% | 19.200 | 2.264x | 483 |
| qk_rope | wide_qkv | 530 | 100.0% | 9.312 | 2.261x | 530 |
| rmsnorm | fused_ffn | 1121 | 100.0% | 5.120 | 2.091x | 1121 |
| rmsnorm | wide_qkv | 530 | 100.0% | 5.008 | 2.040x | 530 |
| affine_silu | fused_ffn | 8 | 100.0% | 10.080 | 2.000x | 8 |
| affine_silu | wide_qkv | 9 | 100.0% | 5.953 | 1.781x | 9 |
| fa4 | linear2_residual | 591 | 96.1% | 20.672 | 1.733x | 591 |
| linear2_residual | wide_qkv | 363 | 97.6% | 35.616 | 1.711x | 363 |
| library_gemm | library_gemm | 1990 | 96.6% | 14.849 | 1.694x | 1990 |
| sumChannelsNCHWKernel | rmsnorm | 13 | 83.1% | 2.784 | 1.673x | 13 |
| linear2_residual | library_gemm | 155 | 97.7% | 34.113 | 1.643x | 155 |
| linear2_residual | linear2_residual | 37 | 97.7% | 33.792 | 1.643x | 37 |
| copy_reformat | linear2_residual | 28 | 100.0% | 2.416 | 1.561x | 28 |
| linear2_residual | fused_ffn | 834 | 98.5% | 32.321 | 1.553x | 834 |
| cudnn | qk_rope | 5 | 66.2% | 2.272 | 1.479x | 5 |
| wide_qkv | linear2_residual | 363 | 97.3% | 28.513 | 1.479x | 363 |
| copy_reformat | idle | 39 | 0.0% | 1.344 | 1.448x | 39 |
| wide_qkv | fused_ffn | 591 | 98.1% | 27.617 | 1.434x | 591 |
| qk_rope | fa4 | 530 | 100.0% | 5.824 | 1.417x | 530 |
| linear2_residual | fa4 | 591 | 97.3% | 29.345 | 1.409x | 591 |
| library_gemm | linear2_residual | 190 | 96.2% | 20.208 | 1.408x | 190 |
| qk_rope | linear2_residual | 591 | 100.0% | 5.792 | 1.408x | 591 |
| affine_silu | rmsnorm | 18 | 47.3% | 7.008 | 1.395x | 18 |
| fused_ffn | wide_qkv | 286 | 69.7% | 53.472 | 1.393x | 286 |
| fa4 | fa4 | 52 | 65.5% | 16.448 | 1.391x | 52 |
| head_elementwise | linear2_residual | 28 | 100.0% | 4.400 | 1.387x | 28 |
| affine_silu | library_gemm | 847 | 100.0% | 6.944 | 1.377x | 847 |
| fa4 | fused_ffn | 297 | 61.8% | 16.321 | 1.376x | 297 |
| affine_silu | fa4 | 295 | 100.0% | 4.608 | 1.354x | 295 |
| rmsnorm | fa4 | 612 | 100.0% | 3.296 | 1.351x | 612 |
| rmsnorm | linear2_residual | 669 | 100.0% | 3.296 | 1.342x | 669 |
| cudnn | head_elementwise | 5 | 82.4% | 1.728 | 1.325x | 5 |
| library_gemm | affine_silu | 215 | 94.9% | 18.784 | 1.317x | 215 |
| library_gemm | cudnn | 48 | 93.1% | 3.728 | 1.308x | 48 |
| fa4 | rmsnorm | 9 | 59.1% | 15.456 | 1.298x | 9 |
| qk_rope | library_gemm | 316 | 100.0% | 5.280 | 1.281x | 316 |
| library_gemm | qk_rope | 5 | 94.2% | 17.536 | 1.250x | 5 |
| fa4 | library_gemm | 526 | 44.7% | 14.784 | 1.240x | 526 |

## Logical operation groups

Isolated reference total is the isolated median for each ordinal multiplied by its S2 call count; it is a normalized reference, not a second trace total.

| logical group | families | positions | calls | isolated reference ms | S2 total ms | total ratio | excess ms |
|---|---|---:|---:|---:|---:|---:|---:|
| transformer.ffn_linear2_residual | linear2_residual | 33 | 1980 | 41.268 | 63.488 | 1.538x | 22.220 |
| transformer.attention_out_projection_residual | library_gemm | 33 | 1980 | 16.795 | 36.365 | 2.165x | 19.570 |
| transformer.ffn_linear1_gate_swiglu | fused_ffn | 33 | 1980 | 76.751 | 90.160 | 1.175x | 13.444 |
| transformer.attention_qkv_projection | wide_qkv | 33 | 1980 | 38.137 | 49.962 | 1.310x | 11.824 |
| transformer.attention_fa4 | fa4 | 33 | 1980 | 23.548 | 33.512 | 1.423x | 9.963 |
| transformer.attention_qk_rope | qk_rope | 33 | 1980 | 8.148 | 13.134 | 1.612x | 4.987 |
| transformer.attention_rmsnorm | rmsnorm | 33 | 1980 | 4.831 | 8.431 | 1.745x | 3.600 |
| outer.pre_projection_c768_to_c384 | library_gemm | 11 | 660 | 7.865 | 11.201 | 1.424x | 3.336 |
| outer.post_projection_c384_to_c768_residual | library_gemm | 11 | 660 | 9.502 | 12.376 | 1.302x | 2.874 |
| transformer.ffn_rmsnorm | rmsnorm | 33 | 1980 | 4.847 | 7.171 | 1.479x | 2.324 |
| outer.pre_norm_silu | affine_silu | 11 | 660 | 3.332 | 4.982 | 1.495x | 1.650 |
| outer.post_norm_silu | affine_silu | 11 | 660 | 2.209 | 2.786 | 1.261x | 0.577 |
| policy.g1_conv | library_gemm | 1 | 60 | 0.355 | 0.671 | 1.890x | 0.316 |
| policy.p1_conv | library_gemm | 1 | 60 | 0.372 | 0.562 | 1.510x | 0.190 |
| trunk.tip_norm_silu | affine_silu | 1 | 60 | 0.303 | 0.448 | 1.478x | 0.145 |
| policy.g1_global_pool | head_elementwise | 1 | 60 | 0.269 | 0.389 | 1.448x | 0.120 |
| frontend.initial_conv | cudnn | 1 | 60 | 1.168 | 1.289 | 1.103x | 0.120 |
| value.v1_conv | library_gemm | 1 | 60 | 0.478 | 0.595 | 1.245x | 0.117 |
| policy.p2_conv | library_gemm | 1 | 60 | 0.234 | 0.335 | 1.429x | 0.100 |
| value.v2_matmul | library_gemm | 1 | 60 | 0.570 | 0.651 | 1.142x | 0.081 |
| frontend.initial_global_matmul | library_gemm | 1 | 60 | 0.157 | 0.224 | 1.421x | 0.066 |
| policy.gpool_to_pass_matmul2 | library_gemm | 1 | 60 | 0.137 | 0.199 | 1.450x | 0.062 |
| policy.gpool_to_bias_matmul | library_gemm | 1 | 60 | 0.321 | 0.381 | 1.187x | 0.060 |
| policy.gpool_to_pass_matmul | library_gemm | 1 | 60 | 0.320 | 0.369 | 1.154x | 0.049 |
| value.v1_norm_silu | head_elementwise | 1 | 60 | 0.188 | 0.234 | 1.244x | 0.046 |
| policy.g1_norm_silu | head_elementwise | 1 | 60 | 0.127 | 0.166 | 1.311x | 0.039 |
| frontend.initial_global_broadcast_add | head_elementwise | 1 | 60 | 0.463 | 0.501 | 1.084x | 0.039 |
| value.ownership_half_to_float | copy_reformat | 1 | 60 | 0.056 | 0.094 | 1.691x | 0.038 |
| value.v1_global_pool | head_elementwise | 1 | 60 | 0.194 | 0.232 | 1.198x | 0.038 |
| value.ownership_conv | library_gemm | 1 | 60 | 0.242 | 0.279 | 1.154x | 0.038 |
| frontend.initial_conv_nhwc_padding_0 | cudnn | 1 | 60 | 0.077 | 0.111 | 1.446x | 0.034 |
| policy.p1_norm_silu | head_elementwise | 1 | 60 | 0.131 | 0.164 | 1.259x | 0.034 |
| frontend.initial_conv_nhwc_padding_1 | cudnn | 1 | 60 | 0.092 | 0.123 | 1.330x | 0.030 |
| value.score_matmul | library_gemm | 1 | 60 | 0.209 | 0.239 | 1.143x | 0.030 |
| policy.pass_bias_silu | head_elementwise | 1 | 60 | 0.061 | 0.091 | 1.482x | 0.030 |
| value.v1_half_to_float | copy_reformat | 1 | 60 | 0.131 | 0.160 | 1.225x | 0.029 |
| input.mask_sum | sumChannelsNCHWKernel | 1 | 60 | 0.100 | 0.129 | 1.291x | 0.029 |
| policy.p1_half_to_float | copy_reformat | 1 | 60 | 0.090 | 0.119 | 1.315x | 0.028 |
| value.v3_matmul | library_gemm | 1 | 60 | 0.209 | 0.234 | 1.118x | 0.025 |
| value.ownership_conv_splitk_reduce | library_gemm | 1 | 60 | 0.083 | 0.105 | 1.274x | 0.023 |
| policy.g1_half_to_float | copy_reformat | 1 | 60 | 0.094 | 0.116 | 1.231x | 0.022 |
| policy.gpool_bias_add | head_elementwise | 1 | 60 | 0.108 | 0.127 | 1.181x | 0.019 |
| input.extract_mask | head_elementwise | 1 | 60 | 0.071 | 0.090 | 1.261x | 0.019 |
| input.mask_half_to_float | copy_reformat | 1 | 60 | 0.056 | 0.071 | 1.272x | 0.015 |
| value.score_bias | head_elementwise | 1 | 60 | 0.056 | 0.064 | 1.151x | 0.008 |
| frontend.initial_global_matmul_splitk_reduce | library_gemm | 1 | 60 | 0.077 | 0.084 | 1.093x | 0.008 |
| value.v3_bias | head_elementwise | 1 | 60 | 0.058 | 0.066 | 1.138x | 0.008 |
| value.v2_bias_silu | head_elementwise | 1 | 60 | 0.061 | 0.068 | 1.109x | 0.007 |

## `library_gemm` logical breakdown

| logical group | positions | calls | isolated reference ms | S2 total ms | total ratio | excess ms |
|---|---:|---:|---:|---:|---:|---:|
| transformer.attention_out_projection_residual | 33 | 1980 | 16.795 | 36.365 | 2.165x | 19.570 |
| outer.pre_projection_c768_to_c384 | 11 | 660 | 7.865 | 11.201 | 1.424x | 3.336 |
| outer.post_projection_c384_to_c768_residual | 11 | 660 | 9.502 | 12.376 | 1.302x | 2.874 |
| policy.g1_conv | 1 | 60 | 0.355 | 0.671 | 1.890x | 0.316 |
| policy.p1_conv | 1 | 60 | 0.372 | 0.562 | 1.510x | 0.190 |
| value.v1_conv | 1 | 60 | 0.478 | 0.595 | 1.245x | 0.117 |
| policy.p2_conv | 1 | 60 | 0.234 | 0.335 | 1.429x | 0.100 |
| value.v2_matmul | 1 | 60 | 0.570 | 0.651 | 1.142x | 0.081 |
| frontend.initial_global_matmul | 1 | 60 | 0.157 | 0.224 | 1.421x | 0.066 |
| policy.gpool_to_pass_matmul2 | 1 | 60 | 0.137 | 0.199 | 1.450x | 0.062 |
| policy.gpool_to_bias_matmul | 1 | 60 | 0.321 | 0.381 | 1.187x | 0.060 |
| policy.gpool_to_pass_matmul | 1 | 60 | 0.320 | 0.369 | 1.154x | 0.049 |
| value.ownership_conv | 1 | 60 | 0.242 | 0.279 | 1.154x | 0.038 |
| value.score_matmul | 1 | 60 | 0.209 | 0.239 | 1.143x | 0.030 |
| value.v3_matmul | 1 | 60 | 0.209 | 0.234 | 1.118x | 0.025 |
| value.ownership_conv_splitk_reduce | 1 | 60 | 0.083 | 0.105 | 1.274x | 0.023 |
| frontend.initial_global_matmul_splitk_reduce | 1 | 60 | 0.077 | 0.084 | 1.093x | 0.008 |

## Top ordinal hotspots by summed excess

The worst peer is the highest median S2/S1 slowdown among peer families observed at least four times for that ordinal.

| rank | ordinal | logical position | family | calls | isolated us | S2 us | S2/S1 | excess ms | common peer | worst peer |
|---:|---:|---|---|---:|---:|---:|---:|---:|---|---|
| 1 | 214 | outer_07.transformer_0.block_21.ffn_linear2_residual | linear2_residual | 60 | 21.056 | 35.456 | 1.684x | 0.796 | fused_ffn (30) | wide_qkv (1.784x; 16) |
| 2 | 222 | outer_07.transformer_1.block_22.ffn_linear2_residual | linear2_residual | 60 | 21.056 | 34.368 | 1.632x | 0.796 | fused_ffn (30) | wide_qkv (1.789x; 16) |
| 3 | 250 | outer_08.transformer_1.block_25.ffn_linear2_residual | linear2_residual | 60 | 20.977 | 32.944 | 1.571x | 0.741 | fused_ffn (30) | wide_qkv (1.712x; 16) |
| 4 | 82 | outer_02.transformer_1.block_07.ffn_linear2_residual | linear2_residual | 60 | 20.800 | 32.289 | 1.552x | 0.725 | fused_ffn (30) | wide_qkv (1.782x; 16) |
| 5 | 158 | outer_05.transformer_0.block_15.ffn_linear2_residual | linear2_residual | 60 | 20.896 | 33.408 | 1.599x | 0.719 | fused_ffn (30) | wide_qkv (1.755x; 16) |
| 6 | 230 | outer_07.transformer_2.block_23.ffn_linear2_residual | linear2_residual | 60 | 21.137 | 34.032 | 1.610x | 0.717 | fa4 (28) | library_gemm (1.741x; 16) |
| 7 | 138 | outer_04.transformer_1.block_13.ffn_linear2_residual | linear2_residual | 60 | 20.849 | 32.673 | 1.567x | 0.710 | fused_ffn (30) | wide_qkv (1.669x; 16) |
| 8 | 54 | outer_01.transformer_1.block_04.ffn_linear2_residual | linear2_residual | 60 | 20.768 | 32.321 | 1.556x | 0.707 | fused_ffn (30) | wide_qkv (1.727x; 16) |
| 9 | 74 | outer_02.transformer_0.block_06.ffn_linear2_residual | linear2_residual | 60 | 20.816 | 32.721 | 1.572x | 0.701 | fused_ffn (30) | wide_qkv (1.773x; 16) |
| 10 | 242 | outer_08.transformer_0.block_24.ffn_linear2_residual | linear2_residual | 60 | 20.960 | 33.520 | 1.599x | 0.698 | fused_ffn (30) | wide_qkv (1.714x; 16) |
| 11 | 202 | outer_06.transformer_2.block_20.ffn_linear2_residual | linear2_residual | 60 | 21.040 | 33.248 | 1.580x | 0.695 | fa4 (28) | library_gemm (1.712x; 16) |
| 12 | 146 | outer_04.transformer_2.block_14.ffn_linear2_residual | linear2_residual | 60 | 21.009 | 33.040 | 1.573x | 0.695 | fa4 (28) | library_gemm (1.736x; 16) |
| 13 | 166 | outer_05.transformer_1.block_16.ffn_linear2_residual | linear2_residual | 60 | 20.768 | 32.305 | 1.555x | 0.695 | fused_ffn (30) | wide_qkv (1.709x; 16) |
| 14 | 298 | outer_10.transformer_0.block_30.ffn_linear2_residual | linear2_residual | 60 | 20.848 | 33.328 | 1.599x | 0.692 | fused_ffn (30) | wide_qkv (1.688x; 16) |
| 15 | 194 | outer_06.transformer_1.block_19.ffn_linear2_residual | linear2_residual | 60 | 20.864 | 32.545 | 1.560x | 0.692 | fused_ffn (30) | wide_qkv (1.629x; 16) |
| 16 | 26 | outer_00.transformer_1.block_01.ffn_linear2_residual | linear2_residual | 60 | 20.800 | 32.001 | 1.538x | 0.690 | fused_ffn (30) | wide_qkv (1.717x; 17) |
| 17 | 46 | outer_01.transformer_0.block_03.ffn_linear2_residual | linear2_residual | 60 | 20.848 | 32.272 | 1.548x | 0.687 | fused_ffn (30) | wide_qkv (1.751x; 16) |
| 18 | 306 | outer_10.transformer_1.block_31.ffn_linear2_residual | linear2_residual | 60 | 20.944 | 32.465 | 1.550x | 0.685 | fused_ffn (30) | wide_qkv (1.664x; 16) |
| 19 | 130 | outer_04.transformer_0.block_12.ffn_linear2_residual | linear2_residual | 60 | 20.832 | 32.128 | 1.542x | 0.681 | fused_ffn (30) | wide_qkv (1.737x; 16) |
| 20 | 278 | outer_09.transformer_1.block_28.ffn_linear2_residual | linear2_residual | 60 | 20.640 | 31.504 | 1.526x | 0.679 | fused_ffn (30) | wide_qkv (1.709x; 16) |
| 21 | 110 | outer_03.transformer_1.block_10.ffn_linear2_residual | linear2_residual | 60 | 20.592 | 31.441 | 1.527x | 0.667 | fused_ffn (30) | wide_qkv (1.702x; 16) |
| 22 | 270 | outer_09.transformer_0.block_27.ffn_linear2_residual | linear2_residual | 60 | 20.768 | 32.176 | 1.549x | 0.652 | fused_ffn (30) | wide_qkv (1.713x; 16) |
| 23 | 258 | outer_08.transformer_2.block_26.ffn_linear2_residual | linear2_residual | 60 | 20.928 | 32.321 | 1.544x | 0.649 | fa4 (27) | library_gemm (1.605x; 16) |
| 24 | 102 | outer_03.transformer_0.block_09.ffn_linear2_residual | linear2_residual | 60 | 20.688 | 31.744 | 1.534x | 0.643 | fused_ffn (29) | wide_qkv (1.690x; 16) |
| 25 | 18 | outer_00.transformer_0.block_00.ffn_linear2_residual | linear2_residual | 60 | 20.976 | 31.344 | 1.494x | 0.627 | fused_ffn (30) | wide_qkv (1.712x; 17) |
| 26 | 191 | outer_06.transformer_1.block_19.attention_out_projection_residual | library_gemm | 60 | 8.496 | 19.504 | 2.296x | 0.626 | fused_ffn (24) | fused_ffn (2.557x; 24) |
| 27 | 211 | outer_07.transformer_0.block_21.attention_out_projection_residual | library_gemm | 60 | 8.512 | 20.064 | 2.357x | 0.622 | library_gemm (37) | wide_qkv (2.408x; 14) |
| 28 | 163 | outer_05.transformer_1.block_16.attention_out_projection_residual | library_gemm | 60 | 8.496 | 19.488 | 2.294x | 0.621 | fused_ffn (28) | fused_ffn (2.490x; 28) |
| 29 | 247 | outer_08.transformer_1.block_25.attention_out_projection_residual | library_gemm | 60 | 8.496 | 19.616 | 2.309x | 0.621 | fused_ffn (25) | fused_ffn (2.569x; 25) |
| 30 | 143 | outer_04.transformer_2.block_14.attention_out_projection_residual | library_gemm | 60 | 8.496 | 20.032 | 2.358x | 0.620 | library_gemm (37) | fused_ffn (2.456x; 23) |
| 31 | 186 | outer_06.transformer_0.block_18.ffn_linear2_residual | linear2_residual | 60 | 20.849 | 31.184 | 1.496x | 0.618 | fused_ffn (30) | wide_qkv (1.559x; 16) |
| 32 | 199 | outer_06.transformer_2.block_20.attention_out_projection_residual | library_gemm | 60 | 8.480 | 19.712 | 2.325x | 0.618 | library_gemm (36) | fused_ffn (2.419x; 24) |
| 33 | 239 | outer_08.transformer_0.block_24.attention_out_projection_residual | library_gemm | 60 | 8.592 | 20.064 | 2.335x | 0.615 | library_gemm (40) | wide_qkv (2.363x; 14) |
| 34 | 87 | outer_02.transformer_2.block_08.attention_out_projection_residual | library_gemm | 60 | 8.528 | 19.520 | 2.289x | 0.615 | library_gemm (31) | fused_ffn (2.398x; 29) |
| 35 | 255 | outer_08.transformer_2.block_26.attention_out_projection_residual | library_gemm | 60 | 8.480 | 19.664 | 2.319x | 0.614 | library_gemm (34) | fused_ffn (2.470x; 26) |
| 36 | 89 | outer_02.transformer_2.block_08.ffn_linear1_gate_swiglu | fused_ffn | 60 | 37.952 | 48.337 | 1.274x | 0.610 | linear2_residual (26) | wide_qkv (1.460x; 14) |
| 37 | 79 | outer_02.transformer_1.block_07.attention_out_projection_residual | library_gemm | 60 | 8.512 | 19.328 | 2.271x | 0.610 | fused_ffn (28) | fused_ffn (2.445x; 28) |
| 38 | 311 | outer_10.transformer_2.block_32.attention_out_projection_residual | library_gemm | 60 | 8.448 | 19.601 | 2.320x | 0.607 | library_gemm (37) | fused_ffn (2.394x; 23) |
| 39 | 314 | outer_10.transformer_2.block_32.ffn_linear2_residual | linear2_residual | 60 | 20.960 | 32.336 | 1.543x | 0.605 | library_gemm (30) | library_gemm (1.573x; 30) |
| 40 | 155 | outer_05.transformer_0.block_15.attention_out_projection_residual | library_gemm | 60 | 8.512 | 19.792 | 2.325x | 0.604 | library_gemm (37) | fused_ffn (2.346x; 9) |

## Full fixed-forward ordinal map

| ordinal | logical position | family | resource signature | calls | isolated us | S2 us | S2/S1 | overlap | excess ms | common peer | worst peer |
|---:|---|---|---|---:|---:|---:|---:|---:|---:|---|---|
| 0 | input.extract_mask | head_elementwise | head_elementwise; extractChannel0KernelNHWC; g10x1x1; b512x1x1; r16; s0 | 60 | 1.184 | 1.344 | 1.135x | 56.0% | 0.019 | library_gemm (22) | library_gemm (1.189x; 22) |
| 1 | input.mask_half_to_float | copy_reformat | copy_reformat; copyFromHalfKernel; g10x1x1; b512x1x1; r16; s0 | 60 | 0.928 | 1.136 | 1.224x | 42.4% | 0.015 | idle (22) | idle (1.448x; 22) |
| 2 | input.mask_sum | sumChannelsNCHWKernel | sumChannelsNCHWKernel; sumChannelsNCHWKernel; g1x1x13; b256x2x1; r22; s2048 | 60 | 1.664 | 2.000 | 1.202x | 64.8% | 0.029 | head_elementwise (26) | rmsnorm (1.673x; 13) |
| 3 | frontend.initial_conv_nhwc_padding_0 | cudnn | cudnn; nhwcAddPaddingKernel; g340x1x1; b768x1x1; r34; s0 | 60 | 1.280 | 1.520 | 1.188x | 81.8% | 0.034 | cudnn (17) | wide_qkv (2.475x; 13) |
| 4 | frontend.initial_conv_nhwc_padding_1 | cudnn | cudnn; nhwcAddPaddingKernel; g340x1x1; b768x1x1; r34; s0 | 60 | 1.536 | 1.696 | 1.104x | 72.6% | 0.030 | cudnn (26) | wide_qkv (2.604x; 8) |
| 5 | frontend.initial_conv | cudnn | cudnn; Kernel; g296x3x1; b128x1x1; r94; s81920 | 60 | 19.473 | 21.312 | 1.094x | 28.0% | 0.120 | library_gemm (29) | cudnn (1.104x; 17) |
| 6 | frontend.initial_global_matmul | library_gemm | library_gemm; Kernel2; g8x1x3; b128x1x1; r128; s24576 | 60 | 2.624 | 3.488 | 1.329x | 90.6% | 0.066 | cudnn (16) | fa4 (1.756x; 13) |
| 7 | frontend.initial_global_matmul_splitk_reduce | library_gemm | library_gemm; splitKreduce_kernel; g24x1x1; b32x16x1; r49; s0 | 60 | 1.280 | 1.376 | 1.075x | 89.4% | 0.008 | library_gemm (24) | fa4 (1.150x; 13) |
| 8 | frontend.initial_global_broadcast_add | head_elementwise | head_elementwise; addNCBiasInplaceNHWCHalfKernel; g3x361x13; b256x1x1; r16; s0 | 60 | 7.713 | 8.048 | 1.044x | 32.8% | 0.039 | library_gemm (42) | library_gemm (1.048x; 42) |
| 9 | outer_00.pre_norm_silu | affine_silu | affine_silu; affineSiluHalf2Kernel; g4693x1x1; b384x1x1; r17; s0 | 60 | 4.928 | 6.384 | 1.295x | 61.5% | 0.090 | library_gemm (46) | library_gemm (1.396x; 46) |
| 10 | outer_00.pre_projection_c768_to_c384 | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 60 | 11.584 | 14.432 | 1.246x | 88.4% | 0.213 | wide_qkv (16) | wide_qkv (1.465x; 16) |
| 11 | outer_00.transformer_0.block_00.attention_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | 2.368 | 3.744 | 1.581x | 96.6% | 0.077 | wide_qkv (16) | wide_qkv (1.953x; 16) |
| 12 | outer_00.transformer_0.block_00.attention_qkv_projection | wide_qkv | wide_qkv; wide_qkv_kernel; g9x37x1; b128x1x1; r136; s65536 | 60 | 18.752 | 22.512 | 1.201x | 67.9% | 0.258 | library_gemm (17) | fused_ffn (1.379x; 13) |
| 13 | outer_00.transformer_0.block_00.attention_qk_rope | qk_rope | qk_rope; fusedQKRoPE19HalfKernel; g361x13x1; b192x1x1; r16; s0 | 60 | 4.064 | 5.664 | 1.394x | 86.3% | 0.130 | wide_qkv (17) | wide_qkv (2.220x; 17) |
| 14 | outer_00.transformer_0.block_00.attention_fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 60 | 11.681 | 15.297 | 1.310x | 66.1% | 0.311 | library_gemm (25) | linear2_residual (1.959x; 13) |
| 15 | outer_00.transformer_0.block_00.attention_out_projection_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 60 | 8.432 | 14.416 | 1.710x | 94.9% | 0.435 | library_gemm (23) | fused_ffn (2.330x; 11) |
| 16 | outer_00.transformer_0.block_00.ffn_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | 2.432 | 3.392 | 1.395x | 99.8% | 0.071 | library_gemm (30) | fused_ffn (2.039x; 16) |
| 17 | outer_00.transformer_0.block_00.ffn_linear1_gate_swiglu | fused_ffn | fused_ffn; fused_ffn_candidate_a_reuse_kernel; g18x37x1; b128x1x1; r136; s32768 | 60 | 39.408 | 42.785 | 1.086x | 65.3% | 0.240 | fused_ffn (23) | wide_qkv (1.216x; 13) |
| 18 | outer_00.transformer_0.block_00.ffn_linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 60 | 20.976 | 31.344 | 1.494x | 97.6% | 0.627 | fused_ffn (30) | wide_qkv (1.712x; 17) |
| 19 | outer_00.transformer_1.block_01.attention_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | 2.464 | 3.856 | 1.565x | 100.0% | 0.100 | linear2_residual (17) | fused_ffn (2.143x; 13) |
| 20 | outer_00.transformer_1.block_01.attention_qkv_projection | wide_qkv | wide_qkv; wide_qkv_kernel; g9x37x1; b128x1x1; r136; s65536 | 60 | 19.408 | 24.704 | 1.273x | 84.7% | 0.339 | linear2_residual (17) | fused_ffn (1.482x; 13) |
| 21 | outer_00.transformer_1.block_01.attention_qk_rope | qk_rope | qk_rope; fusedQKRoPE19HalfKernel; g361x13x1; b192x1x1; r16; s0 | 60 | 4.128 | 5.808 | 1.407x | 100.0% | 0.154 | fa4 (17) | wide_qkv (2.287x; 17) |
| 22 | outer_00.transformer_1.block_01.attention_fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 60 | 11.968 | 15.152 | 1.266x | 62.8% | 0.266 | library_gemm (17) | linear2_residual (1.759x; 13) |
| 23 | outer_00.transformer_1.block_01.attention_out_projection_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 60 | 8.480 | 19.073 | 2.249x | 96.1% | 0.589 | fused_ffn (26) | fused_ffn (2.391x; 26) |
| 24 | outer_00.transformer_1.block_01.ffn_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | 2.464 | 3.376 | 1.370x | 100.0% | 0.082 | fused_ffn (17) | fused_ffn (2.325x; 17) |
| 25 | outer_00.transformer_1.block_01.ffn_linear1_gate_swiglu | fused_ffn | fused_ffn; fused_ffn_candidate_a_reuse_kernel; g18x37x1; b128x1x1; r136; s32768 | 60 | 38.016 | 43.664 | 1.149x | 70.7% | 0.376 | fused_ffn (25) | linear2_residual (1.304x; 22) |
| 26 | outer_00.transformer_1.block_01.ffn_linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 60 | 20.800 | 32.001 | 1.538x | 97.9% | 0.690 | fused_ffn (30) | wide_qkv (1.717x; 17) |
| 27 | outer_00.transformer_2.block_02.attention_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | 2.400 | 3.792 | 1.580x | 100.0% | 0.084 | linear2_residual (17) | wide_qkv (1.973x; 17) |
| 28 | outer_00.transformer_2.block_02.attention_qkv_projection | wide_qkv | wide_qkv; wide_qkv_kernel; g9x37x1; b128x1x1; r136; s65536 | 60 | 19.248 | 23.921 | 1.243x | 82.0% | 0.299 | linear2_residual (17) | linear2_residual (1.431x; 17) |
| 29 | outer_00.transformer_2.block_02.attention_qk_rope | qk_rope | qk_rope; fusedQKRoPE19HalfKernel; g361x13x1; b192x1x1; r16; s0 | 60 | 4.096 | 5.744 | 1.402x | 100.0% | 0.151 | fa4 (17) | wide_qkv (2.278x; 16) |
| 30 | outer_00.transformer_2.block_02.attention_fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 60 | 11.872 | 14.592 | 1.229x | 62.1% | 0.241 | library_gemm (18) | linear2_residual (1.644x; 13) |
| 31 | outer_00.transformer_2.block_02.attention_out_projection_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 60 | 8.480 | 19.440 | 2.292x | 96.8% | 0.595 | library_gemm (33) | fused_ffn (2.396x; 27) |
| 32 | outer_00.transformer_2.block_02.ffn_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | 2.480 | 3.168 | 1.277x | 96.9% | 0.076 | fused_ffn (17) | fused_ffn (2.400x; 17) |
| 33 | outer_00.transformer_2.block_02.ffn_linear1_gate_swiglu | fused_ffn | fused_ffn; fused_ffn_candidate_a_reuse_kernel; g18x37x1; b128x1x1; r136; s32768 | 60 | 38.273 | 46.240 | 1.208x | 75.4% | 0.520 | fused_ffn (25) | wide_qkv (1.404x; 14) |
| 34 | outer_00.transformer_2.block_02.ffn_linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 60 | 20.704 | 29.808 | 1.440x | 97.8% | 0.577 | fa4 (27) | linear2_residual (1.641x; 6) |
| 35 | outer_00.post_norm_silu | affine_silu | affine_silu; affineSiluHalf2Kernel; g4693x1x1; b192x1x1; r17; s0 | 60 | 3.328 | 3.936 | 1.183x | 99.3% | 0.048 | fa4 (27) | fa4 (1.346x; 27) |
| 36 | outer_00.post_projection_c384_to_c768_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b256x1x1; r154; s73728 | 60 | 14.304 | 17.168 | 1.200x | 96.7% | 0.205 | library_gemm (33) | linear2_residual (1.352x; 16) |
| 37 | outer_01.pre_norm_silu | affine_silu | affine_silu; affineSiluHalf2Kernel; g4693x1x1; b384x1x1; r17; s0 | 60 | 5.088 | 6.976 | 1.371x | 79.1% | 0.134 | library_gemm (59) | library_gemm (1.371x; 59) |
| 38 | outer_01.pre_projection_c768_to_c384 | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 60 | 11.921 | 15.104 | 1.267x | 95.5% | 0.308 | fused_ffn (27) | library_gemm (1.667x; 15) |
| 39 | outer_01.transformer_0.block_03.attention_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | 2.432 | 4.880 | 2.007x | 100.0% | 0.133 | fused_ffn (28) | wide_qkv (2.191x; 16) |
| 40 | outer_01.transformer_0.block_03.attention_qkv_projection | wide_qkv | wide_qkv; wide_qkv_kernel; g9x37x1; b128x1x1; r136; s65536 | 60 | 19.073 | 24.864 | 1.304x | 82.3% | 0.362 | fused_ffn (28) | fused_ffn (1.439x; 28) |
| 41 | outer_01.transformer_0.block_03.attention_qk_rope | qk_rope | qk_rope; fusedQKRoPE19HalfKernel; g361x13x1; b192x1x1; r16; s0 | 60 | 4.064 | 5.856 | 1.441x | 100.0% | 0.160 | linear2_residual (28) | wide_qkv (2.295x; 16) |
| 42 | outer_01.transformer_0.block_03.attention_fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 60 | 11.840 | 15.088 | 1.274x | 71.5% | 0.302 | linear2_residual (28) | linear2_residual (1.641x; 28) |
| 43 | outer_01.transformer_0.block_03.attention_out_projection_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 60 | 8.480 | 19.344 | 2.281x | 96.1% | 0.582 | library_gemm (32) | fused_ffn (2.321x; 14) |
| 44 | outer_01.transformer_0.block_03.ffn_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | 2.432 | 2.992 | 1.230x | 96.5% | 0.068 | fused_ffn (16) | fused_ffn (2.217x; 16) |
| 45 | outer_01.transformer_0.block_03.ffn_linear1_gate_swiglu | fused_ffn | fused_ffn; fused_ffn_candidate_a_reuse_kernel; g18x37x1; b128x1x1; r136; s32768 | 60 | 39.633 | 42.992 | 1.085x | 66.6% | 0.290 | fused_ffn (17) | wide_qkv (1.321x; 14) |
| 46 | outer_01.transformer_0.block_03.ffn_linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 60 | 20.848 | 32.272 | 1.548x | 97.6% | 0.687 | fused_ffn (30) | wide_qkv (1.751x; 16) |
| 47 | outer_01.transformer_1.block_04.attention_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | 2.448 | 3.984 | 1.627x | 100.0% | 0.097 | linear2_residual (16) | fused_ffn (1.967x; 14) |
| 48 | outer_01.transformer_1.block_04.attention_qkv_projection | wide_qkv | wide_qkv; wide_qkv_kernel; g9x37x1; b128x1x1; r136; s65536 | 60 | 19.328 | 24.976 | 1.292x | 82.9% | 0.344 | linear2_residual (16) | linear2_residual (1.475x; 16) |
| 49 | outer_01.transformer_1.block_04.attention_qk_rope | qk_rope | qk_rope; fusedQKRoPE19HalfKernel; g361x13x1; b192x1x1; r16; s0 | 60 | 4.096 | 5.760 | 1.406x | 98.1% | 0.150 | fa4 (16) | wide_qkv (2.297x; 16) |
| 50 | outer_01.transformer_1.block_04.attention_fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 60 | 11.904 | 15.632 | 1.313x | 65.5% | 0.300 | fused_ffn (14) | linear2_residual (1.825x; 14) |
| 51 | outer_01.transformer_1.block_04.attention_out_projection_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 60 | 8.512 | 19.312 | 2.269x | 96.0% | 0.599 | fused_ffn (28) | fused_ffn (2.419x; 28) |
| 52 | outer_01.transformer_1.block_04.ffn_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | 2.432 | 3.392 | 1.395x | 100.0% | 0.086 | fused_ffn (16) | fused_ffn (2.454x; 16) |
| 53 | outer_01.transformer_1.block_04.ffn_linear1_gate_swiglu | fused_ffn | fused_ffn; fused_ffn_candidate_a_reuse_kernel; g18x37x1; b128x1x1; r136; s32768 | 60 | 38.608 | 44.337 | 1.148x | 70.1% | 0.370 | fused_ffn (26) | linear2_residual (1.298x; 20) |
| 54 | outer_01.transformer_1.block_04.ffn_linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 60 | 20.768 | 32.321 | 1.556x | 97.8% | 0.707 | fused_ffn (30) | wide_qkv (1.727x; 16) |
| 55 | outer_01.transformer_2.block_05.attention_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | 2.432 | 3.808 | 1.566x | 100.0% | 0.094 | linear2_residual (16) | wide_qkv (2.118x; 16) |
| 56 | outer_01.transformer_2.block_05.attention_qkv_projection | wide_qkv | wide_qkv; wide_qkv_kernel; g9x37x1; b128x1x1; r136; s65536 | 60 | 19.184 | 24.368 | 1.270x | 82.0% | 0.317 | linear2_residual (16) | linear2_residual (1.447x; 16) |
| 57 | outer_01.transformer_2.block_05.attention_qk_rope | qk_rope | qk_rope; fusedQKRoPE19HalfKernel; g361x13x1; b192x1x1; r16; s0 | 60 | 4.128 | 5.712 | 1.384x | 99.0% | 0.144 | fa4 (16) | wide_qkv (2.240x; 16) |
| 58 | outer_01.transformer_2.block_05.attention_fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 60 | 11.809 | 15.376 | 1.302x | 63.8% | 0.264 | library_gemm (15) | linear2_residual (1.672x; 14) |
| 59 | outer_01.transformer_2.block_05.attention_out_projection_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 60 | 8.448 | 19.328 | 2.288x | 96.7% | 0.603 | library_gemm (36) | fused_ffn (2.506x; 24) |
| 60 | outer_01.transformer_2.block_05.ffn_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | 2.432 | 3.120 | 1.283x | 96.6% | 0.079 | fused_ffn (16) | fused_ffn (2.539x; 16) |
| 61 | outer_01.transformer_2.block_05.ffn_linear1_gate_swiglu | fused_ffn | fused_ffn; fused_ffn_candidate_a_reuse_kernel; g18x37x1; b128x1x1; r136; s32768 | 60 | 38.320 | 46.624 | 1.217x | 75.6% | 0.556 | fused_ffn (26) | wide_qkv (1.419x; 14) |
| 62 | outer_01.transformer_2.block_05.ffn_linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 60 | 20.752 | 30.448 | 1.467x | 97.8% | 0.603 | fa4 (28) | linear2_residual (1.668x; 4) |
| 63 | outer_01.post_norm_silu | affine_silu | affine_silu; affineSiluHalf2Kernel; g4693x1x1; b192x1x1; r17; s0 | 60 | 3.328 | 3.808 | 1.144x | 99.0% | 0.039 | fa4 (28) | fa4 (1.269x; 28) |
| 64 | outer_01.post_projection_c384_to_c768_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b256x1x1; r154; s73728 | 60 | 14.320 | 17.408 | 1.216x | 97.1% | 0.213 | library_gemm (37) | linear2_residual (1.344x; 16) |
| 65 | outer_02.pre_norm_silu | affine_silu | affine_silu; affineSiluHalf2Kernel; g4693x1x1; b384x1x1; r17; s0 | 60 | 5.024 | 6.912 | 1.376x | 71.0% | 0.121 | library_gemm (55) | rmsnorm (1.395x; 5) |
| 66 | outer_02.pre_projection_c768_to_c384 | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 60 | 11.905 | 15.248 | 1.281x | 95.4% | 0.337 | fused_ffn (28) | library_gemm (2.271x; 16) |
| 67 | outer_02.transformer_0.block_06.attention_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | 2.432 | 4.976 | 2.046x | 100.0% | 0.134 | fused_ffn (28) | wide_qkv (2.125x; 16) |
| 68 | outer_02.transformer_0.block_06.attention_qkv_projection | wide_qkv | wide_qkv; wide_qkv_kernel; g9x37x1; b128x1x1; r136; s65536 | 60 | 19.105 | 25.104 | 1.314x | 82.3% | 0.371 | fused_ffn (28) | fused_ffn (1.436x; 28) |
| 69 | outer_02.transformer_0.block_06.attention_qk_rope | qk_rope | qk_rope; fusedQKRoPE19HalfKernel; g361x13x1; b192x1x1; r16; s0 | 60 | 4.144 | 5.888 | 1.421x | 100.0% | 0.155 | linear2_residual (28) | wide_qkv (2.247x; 16) |
| 70 | outer_02.transformer_0.block_06.attention_fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 60 | 11.856 | 15.825 | 1.335x | 72.1% | 0.327 | linear2_residual (28) | linear2_residual (1.638x; 28) |
| 71 | outer_02.transformer_0.block_06.attention_out_projection_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 60 | 8.448 | 19.329 | 2.288x | 96.1% | 0.590 | library_gemm (36) | fused_ffn (2.348x; 10) |
| 72 | outer_02.transformer_0.block_06.ffn_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | 2.432 | 3.216 | 1.322x | 97.1% | 0.070 | fused_ffn (16) | fused_ffn (2.211x; 16) |
| 73 | outer_02.transformer_0.block_06.ffn_linear1_gate_swiglu | fused_ffn | fused_ffn; fused_ffn_candidate_a_reuse_kernel; g18x37x1; b128x1x1; r136; s32768 | 60 | 40.752 | 43.856 | 1.076x | 64.5% | 0.312 | fused_ffn (23) | wide_qkv (1.340x; 14) |
| 74 | outer_02.transformer_0.block_06.ffn_linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 60 | 20.816 | 32.721 | 1.572x | 97.6% | 0.701 | fused_ffn (30) | wide_qkv (1.773x; 16) |
| 75 | outer_02.transformer_1.block_07.attention_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | 2.464 | 3.840 | 1.558x | 100.0% | 0.094 | linear2_residual (16) | wide_qkv (2.071x; 16) |
| 76 | outer_02.transformer_1.block_07.attention_qkv_projection | wide_qkv | wide_qkv; wide_qkv_kernel; g9x37x1; b128x1x1; r136; s65536 | 60 | 19.248 | 24.928 | 1.295x | 83.5% | 0.345 | linear2_residual (16) | linear2_residual (1.472x; 16) |
| 77 | outer_02.transformer_1.block_07.attention_qk_rope | qk_rope | qk_rope; fusedQKRoPE19HalfKernel; g361x13x1; b192x1x1; r16; s0 | 60 | 4.144 | 5.840 | 1.409x | 99.0% | 0.144 | fa4 (16) | wide_qkv (2.174x; 16) |
| 78 | outer_02.transformer_1.block_07.attention_fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 60 | 11.857 | 15.456 | 1.304x | 66.0% | 0.299 | library_gemm (15) | linear2_residual (1.826x; 14) |
| 79 | outer_02.transformer_1.block_07.attention_out_projection_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 60 | 8.512 | 19.328 | 2.271x | 96.1% | 0.610 | fused_ffn (28) | fused_ffn (2.445x; 28) |
| 80 | outer_02.transformer_1.block_07.ffn_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | 2.464 | 3.392 | 1.377x | 100.0% | 0.070 | fused_ffn (16) | fused_ffn (2.052x; 16) |
| 81 | outer_02.transformer_1.block_07.ffn_linear1_gate_swiglu | fused_ffn | fused_ffn; fused_ffn_candidate_a_reuse_kernel; g18x37x1; b128x1x1; r136; s32768 | 60 | 41.056 | 44.592 | 1.086x | 71.6% | 0.248 | fused_ffn (27) | linear2_residual (1.216x; 19) |
| 82 | outer_02.transformer_1.block_07.ffn_linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 60 | 20.800 | 32.289 | 1.552x | 97.8% | 0.725 | fused_ffn (30) | wide_qkv (1.782x; 16) |
| 83 | outer_02.transformer_2.block_08.attention_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | 2.464 | 3.984 | 1.617x | 100.0% | 0.087 | linear2_residual (16) | wide_qkv (1.974x; 16) |
| 84 | outer_02.transformer_2.block_08.attention_qkv_projection | wide_qkv | wide_qkv; wide_qkv_kernel; g9x37x1; b128x1x1; r136; s65536 | 60 | 19.152 | 24.704 | 1.290x | 82.0% | 0.341 | linear2_residual (16) | linear2_residual (1.475x; 16) |
| 85 | outer_02.transformer_2.block_08.attention_qk_rope | qk_rope | qk_rope; fusedQKRoPE19HalfKernel; g361x13x1; b192x1x1; r16; s0 | 60 | 4.128 | 5.824 | 1.411x | 100.0% | 0.152 | fa4 (16) | wide_qkv (2.298x; 16) |
| 86 | outer_02.transformer_2.block_08.attention_fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 60 | 11.872 | 15.184 | 1.279x | 62.9% | 0.264 | library_gemm (16) | linear2_residual (1.678x; 14) |
| 87 | outer_02.transformer_2.block_08.attention_out_projection_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 60 | 8.528 | 19.520 | 2.289x | 96.7% | 0.615 | library_gemm (31) | fused_ffn (2.398x; 29) |
| 88 | outer_02.transformer_2.block_08.ffn_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | 2.449 | 3.152 | 1.287x | 96.5% | 0.072 | fused_ffn (16) | fused_ffn (2.274x; 16) |
| 89 | outer_02.transformer_2.block_08.ffn_linear1_gate_swiglu | fused_ffn | fused_ffn; fused_ffn_candidate_a_reuse_kernel; g18x37x1; b128x1x1; r136; s32768 | 60 | 37.952 | 48.337 | 1.274x | 75.8% | 0.610 | linear2_residual (26) | wide_qkv (1.460x; 14) |
| 90 | outer_02.transformer_2.block_08.ffn_linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 60 | 20.881 | 30.800 | 1.475x | 97.7% | 0.603 | fa4 (28) | library_gemm (1.594x; 16) |
| 91 | outer_02.post_norm_silu | affine_silu | affine_silu; affineSiluHalf2Kernel; g4693x1x1; b192x1x1; r17; s0 | 60 | 3.360 | 4.064 | 1.210x | 99.8% | 0.049 | fa4 (28) | fa4 (1.243x; 28) |
| 92 | outer_02.post_projection_c384_to_c768_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b256x1x1; r154; s73728 | 60 | 14.480 | 17.984 | 1.242x | 96.6% | 0.245 | library_gemm (30) | linear2_residual (1.417x; 16) |
| 93 | outer_03.pre_norm_silu | affine_silu | affine_silu; affineSiluHalf2Kernel; g4693x1x1; b384x1x1; r17; s0 | 60 | 5.040 | 7.088 | 1.406x | 80.7% | 0.157 | library_gemm (56) | library_gemm (1.410x; 56) |
| 94 | outer_03.pre_projection_c768_to_c384 | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 60 | 12.000 | 15.392 | 1.283x | 95.1% | 0.287 | fused_ffn (28) | library_gemm (1.693x; 13) |
| 95 | outer_03.transformer_0.block_09.attention_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | 2.432 | 5.072 | 2.086x | 100.0% | 0.141 | fused_ffn (28) | wide_qkv (2.158x; 16) |
| 96 | outer_03.transformer_0.block_09.attention_qkv_projection | wide_qkv | wide_qkv; wide_qkv_kernel; g9x37x1; b128x1x1; r136; s65536 | 60 | 19.376 | 25.361 | 1.309x | 82.1% | 0.369 | fused_ffn (28) | fused_ffn (1.420x; 28) |
| 97 | outer_03.transformer_0.block_09.attention_qk_rope | qk_rope | qk_rope; fusedQKRoPE19HalfKernel; g361x13x1; b192x1x1; r16; s0 | 60 | 4.096 | 5.952 | 1.453x | 100.0% | 0.164 | linear2_residual (28) | wide_qkv (2.320x; 16) |
| 98 | outer_03.transformer_0.block_09.attention_fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 60 | 11.936 | 15.793 | 1.323x | 71.9% | 0.325 | linear2_residual (28) | linear2_residual (1.634x; 28) |
| 99 | outer_03.transformer_0.block_09.attention_out_projection_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 60 | 8.512 | 19.328 | 2.271x | 96.2% | 0.593 | library_gemm (37) | fused_ffn (2.335x; 9) |
| 100 | outer_03.transformer_0.block_09.ffn_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | 2.432 | 3.168 | 1.303x | 96.7% | 0.080 | fused_ffn (16) | fused_ffn (2.493x; 16) |
| 101 | outer_03.transformer_0.block_09.ffn_linear1_gate_swiglu | fused_ffn | fused_ffn; fused_ffn_candidate_a_reuse_kernel; g18x37x1; b128x1x1; r136; s32768 | 60 | 38.240 | 43.792 | 1.145x | 64.6% | 0.447 | fused_ffn (24) | wide_qkv (1.437x; 14) |
| 102 | outer_03.transformer_0.block_09.ffn_linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 60 | 20.688 | 31.744 | 1.534x | 97.6% | 0.643 | fused_ffn (29) | wide_qkv (1.690x; 16) |
| 103 | outer_03.transformer_1.block_10.attention_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | 2.432 | 3.616 | 1.487x | 100.0% | 0.088 | linear2_residual (16) | fused_ffn (1.947x; 14) |
| 104 | outer_03.transformer_1.block_10.attention_qkv_projection | wide_qkv | wide_qkv; wide_qkv_kernel; g9x37x1; b128x1x1; r136; s65536 | 60 | 19.136 | 24.241 | 1.267x | 83.5% | 0.306 | linear2_residual (16) | linear2_residual (1.421x; 16) |
| 105 | outer_03.transformer_1.block_10.attention_qk_rope | qk_rope | qk_rope; fusedQKRoPE19HalfKernel; g361x13x1; b192x1x1; r16; s0 | 60 | 4.096 | 5.664 | 1.383x | 97.0% | 0.143 | fa4 (16) | wide_qkv (2.258x; 16) |
| 106 | outer_03.transformer_1.block_10.attention_fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 60 | 11.808 | 15.600 | 1.321x | 65.8% | 0.279 | fused_ffn (14) | linear2_residual (1.759x; 14) |
| 107 | outer_03.transformer_1.block_10.attention_out_projection_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 60 | 8.416 | 19.008 | 2.259x | 96.0% | 0.591 | fused_ffn (27) | fused_ffn (2.437x; 27) |
| 108 | outer_03.transformer_1.block_10.ffn_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | 2.432 | 3.232 | 1.329x | 100.0% | 0.071 | fused_ffn (16) | fused_ffn (2.217x; 16) |
| 109 | outer_03.transformer_1.block_10.ffn_linear1_gate_swiglu | fused_ffn | fused_ffn; fused_ffn_candidate_a_reuse_kernel; g18x37x1; b128x1x1; r136; s32768 | 60 | 40.160 | 43.105 | 1.073x | 70.3% | 0.227 | fused_ffn (27) | linear2_residual (1.211x; 19) |
| 110 | outer_03.transformer_1.block_10.ffn_linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 60 | 20.592 | 31.441 | 1.527x | 97.8% | 0.667 | fused_ffn (30) | wide_qkv (1.702x; 16) |
| 111 | outer_03.transformer_2.block_11.attention_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | 2.368 | 3.856 | 1.628x | 100.0% | 0.094 | linear2_residual (16) | wide_qkv (2.135x; 16) |
| 112 | outer_03.transformer_2.block_11.attention_qkv_projection | wide_qkv | wide_qkv; wide_qkv_kernel; g9x37x1; b128x1x1; r136; s65536 | 60 | 19.105 | 23.968 | 1.255x | 81.4% | 0.293 | linear2_residual (16) | linear2_residual (1.416x; 16) |
| 113 | outer_03.transformer_2.block_11.attention_qk_rope | qk_rope | qk_rope; fusedQKRoPE19HalfKernel; g361x13x1; b192x1x1; r16; s0 | 60 | 4.096 | 5.616 | 1.371x | 99.0% | 0.137 | fa4 (16) | wide_qkv (2.211x; 16) |
| 114 | outer_03.transformer_2.block_11.attention_fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 60 | 11.760 | 14.768 | 1.256x | 63.7% | 0.242 | library_gemm (15) | linear2_residual (1.642x; 14) |
| 115 | outer_03.transformer_2.block_11.attention_out_projection_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 60 | 8.400 | 18.816 | 2.240x | 96.7% | 0.579 | library_gemm (32) | fused_ffn (2.383x; 28) |
| 116 | outer_03.transformer_2.block_11.ffn_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | 2.464 | 3.057 | 1.240x | 96.9% | 0.074 | library_gemm (17) | fused_ffn (2.468x; 16) |
| 117 | outer_03.transformer_2.block_11.ffn_linear1_gate_swiglu | fused_ffn | fused_ffn; fused_ffn_candidate_a_reuse_kernel; g18x37x1; b128x1x1; r136; s32768 | 60 | 40.657 | 46.849 | 1.152x | 75.8% | 0.353 | fused_ffn (24) | wide_qkv (1.300x; 13) |
| 118 | outer_03.transformer_2.block_11.ffn_linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 60 | 20.544 | 29.776 | 1.449x | 97.8% | 0.572 | fa4 (27) | library_gemm (1.625x; 9) |
| 119 | outer_03.post_norm_silu | affine_silu | affine_silu; affineSiluHalf2Kernel; g4693x1x1; b192x1x1; r17; s0 | 60 | 3.296 | 3.888 | 1.180x | 99.5% | 0.045 | fa4 (27) | linear2_residual (1.199x; 16) |
| 120 | outer_03.post_projection_c384_to_c768_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b256x1x1; r154; s73728 | 60 | 14.161 | 17.216 | 1.216x | 96.9% | 0.211 | library_gemm (30) | linear2_residual (1.316x; 16) |
| 121 | outer_04.pre_norm_silu | affine_silu | affine_silu; affineSiluHalf2Kernel; g4693x1x1; b384x1x1; r17; s0 | 60 | 4.992 | 6.848 | 1.372x | 79.9% | 0.144 | library_gemm (56) | library_gemm (1.372x; 56) |
| 122 | outer_04.pre_projection_c768_to_c384 | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 60 | 11.760 | 14.784 | 1.257x | 95.2% | 0.290 | fused_ffn (27) | library_gemm (1.646x; 17) |
| 123 | outer_04.transformer_0.block_12.attention_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | 2.432 | 4.816 | 1.980x | 100.0% | 0.129 | fused_ffn (27) | wide_qkv (2.178x; 16) |
| 124 | outer_04.transformer_0.block_12.attention_qkv_projection | wide_qkv | wide_qkv; wide_qkv_kernel; g9x37x1; b128x1x1; r136; s65536 | 60 | 18.945 | 24.464 | 1.291x | 81.3% | 0.344 | fused_ffn (27) | fused_ffn (1.412x; 27) |
| 125 | outer_04.transformer_0.block_12.attention_qk_rope | qk_rope | qk_rope; fusedQKRoPE19HalfKernel; g361x13x1; b192x1x1; r16; s0 | 60 | 4.064 | 5.792 | 1.425x | 98.1% | 0.158 | linear2_residual (27) | wide_qkv (2.315x; 16) |
| 126 | outer_04.transformer_0.block_12.attention_fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 60 | 11.745 | 16.464 | 1.402x | 72.5% | 0.316 | linear2_residual (27) | linear2_residual (1.638x; 27) |
| 127 | outer_04.transformer_0.block_12.attention_out_projection_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 60 | 8.448 | 19.088 | 2.259x | 96.1% | 0.565 | library_gemm (35) | fused_ffn (2.337x; 11) |
| 128 | outer_04.transformer_0.block_12.ffn_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | 2.432 | 3.280 | 1.349x | 97.0% | 0.073 | library_gemm (17) | fused_ffn (2.289x; 16) |
| 129 | outer_04.transformer_0.block_12.ffn_linear1_gate_swiglu | fused_ffn | fused_ffn; fused_ffn_candidate_a_reuse_kernel; g18x37x1; b128x1x1; r136; s32768 | 60 | 38.129 | 42.129 | 1.105x | 68.0% | 0.299 | fused_ffn (16) | wide_qkv (1.299x; 14) |
| 130 | outer_04.transformer_0.block_12.ffn_linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 60 | 20.832 | 32.128 | 1.542x | 97.5% | 0.681 | fused_ffn (30) | wide_qkv (1.737x; 16) |
| 131 | outer_04.transformer_1.block_13.attention_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | 2.432 | 3.920 | 1.612x | 100.0% | 0.112 | linear2_residual (16) | fused_ffn (2.461x; 14) |
| 132 | outer_04.transformer_1.block_13.attention_qkv_projection | wide_qkv | wide_qkv; wide_qkv_kernel; g9x37x1; b128x1x1; r136; s65536 | 60 | 19.328 | 25.233 | 1.305x | 83.7% | 0.361 | linear2_residual (16) | linear2_residual (1.475x; 16) |
| 133 | outer_04.transformer_1.block_13.attention_qk_rope | qk_rope | qk_rope; fusedQKRoPE19HalfKernel; g361x13x1; b192x1x1; r16; s0 | 60 | 4.160 | 5.920 | 1.423x | 99.1% | 0.150 | fa4 (16) | wide_qkv (2.269x; 16) |
| 134 | outer_04.transformer_1.block_13.attention_fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 60 | 11.936 | 15.376 | 1.288x | 65.1% | 0.314 | library_gemm (15) | linear2_residual (1.887x; 14) |
| 135 | outer_04.transformer_1.block_13.attention_out_projection_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 60 | 8.464 | 19.424 | 2.295x | 96.0% | 0.602 | fused_ffn (27) | fused_ffn (2.401x; 27) |
| 136 | outer_04.transformer_1.block_13.ffn_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | 2.432 | 3.377 | 1.388x | 100.0% | 0.072 | fused_ffn (16) | fused_ffn (2.066x; 16) |
| 137 | outer_04.transformer_1.block_13.ffn_linear1_gate_swiglu | fused_ffn | fused_ffn; fused_ffn_candidate_a_reuse_kernel; g18x37x1; b128x1x1; r136; s32768 | 60 | 38.368 | 44.273 | 1.154x | 78.8% | 0.382 | linear2_residual (30) | fused_ffn (1.190x; 16) |
| 138 | outer_04.transformer_1.block_13.ffn_linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 60 | 20.849 | 32.673 | 1.567x | 97.8% | 0.710 | fused_ffn (30) | wide_qkv (1.669x; 16) |
| 139 | outer_04.transformer_2.block_14.attention_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | 2.464 | 3.841 | 1.559x | 100.0% | 0.095 | linear2_residual (16) | fused_ffn (2.149x; 14) |
| 140 | outer_04.transformer_2.block_14.attention_qkv_projection | wide_qkv | wide_qkv; wide_qkv_kernel; g9x37x1; b128x1x1; r136; s65536 | 60 | 19.424 | 25.312 | 1.303x | 83.0% | 0.375 | linear2_residual (16) | linear2_residual (1.502x; 16) |
| 141 | outer_04.transformer_2.block_14.attention_qk_rope | qk_rope | qk_rope; fusedQKRoPE19HalfKernel; g361x13x1; b192x1x1; r16; s0 | 60 | 4.128 | 5.840 | 1.415x | 99.0% | 0.148 | fa4 (16) | wide_qkv (2.244x; 16) |
| 142 | outer_04.transformer_2.block_14.attention_fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 60 | 11.920 | 15.504 | 1.301x | 64.1% | 0.291 | library_gemm (15) | linear2_residual (1.769x; 14) |
| 143 | outer_04.transformer_2.block_14.attention_out_projection_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 60 | 8.496 | 20.032 | 2.358x | 96.6% | 0.620 | library_gemm (37) | fused_ffn (2.456x; 23) |
| 144 | outer_04.transformer_2.block_14.ffn_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | 2.432 | 3.040 | 1.250x | 95.9% | 0.054 | fused_ffn (16) | fused_ffn (1.842x; 16) |
| 145 | outer_04.transformer_2.block_14.ffn_linear1_gate_swiglu | fused_ffn | fused_ffn; fused_ffn_candidate_a_reuse_kernel; g18x37x1; b128x1x1; r136; s32768 | 60 | 35.488 | 42.897 | 1.209x | 82.8% | 0.573 | linear2_residual (28) | wide_qkv (1.428x; 15) |
| 146 | outer_04.transformer_2.block_14.ffn_linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 60 | 21.009 | 33.040 | 1.573x | 97.9% | 0.695 | fa4 (28) | library_gemm (1.736x; 16) |
| 147 | outer_04.post_norm_silu | affine_silu | affine_silu; affineSiluHalf2Kernel; g4693x1x1; b192x1x1; r17; s0 | 60 | 3.392 | 4.320 | 1.274x | 99.9% | 0.067 | fa4 (28) | fa4 (1.462x; 28) |
| 148 | outer_04.post_projection_c384_to_c768_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b256x1x1; r154; s73728 | 60 | 14.528 | 19.008 | 1.308x | 96.6% | 0.308 | library_gemm (30) | linear2_residual (1.548x; 16) |
| 149 | outer_05.pre_norm_silu | affine_silu | affine_silu; affineSiluHalf2Kernel; g4693x1x1; b384x1x1; r17; s0 | 60 | 5.120 | 7.328 | 1.431x | 84.6% | 0.165 | library_gemm (60) | library_gemm (1.431x; 60) |
| 150 | outer_05.pre_projection_c768_to_c384 | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 60 | 12.081 | 15.936 | 1.319x | 95.2% | 0.323 | fused_ffn (28) | library_gemm (1.799x; 13) |
| 151 | outer_05.transformer_0.block_15.attention_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | 2.464 | 5.024 | 2.039x | 100.0% | 0.139 | fused_ffn (28) | wide_qkv (2.253x; 16) |
| 152 | outer_05.transformer_0.block_15.attention_qkv_projection | wide_qkv | wide_qkv; wide_qkv_kernel; g9x37x1; b128x1x1; r136; s65536 | 60 | 19.360 | 26.561 | 1.372x | 81.8% | 0.410 | fused_ffn (28) | fused_ffn (1.469x; 28) |
| 153 | outer_05.transformer_0.block_15.attention_qk_rope | qk_rope | qk_rope; fusedQKRoPE19HalfKernel; g361x13x1; b192x1x1; r16; s0 | 60 | 4.128 | 5.984 | 1.450x | 99.1% | 0.164 | linear2_residual (28) | wide_qkv (2.322x; 16) |
| 154 | outer_05.transformer_0.block_15.attention_fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 60 | 11.921 | 16.496 | 1.384x | 73.5% | 0.370 | linear2_residual (28) | linear2_residual (1.758x; 28) |
| 155 | outer_05.transformer_0.block_15.attention_out_projection_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 60 | 8.512 | 19.792 | 2.325x | 96.2% | 0.604 | library_gemm (37) | fused_ffn (2.346x; 9) |
| 156 | outer_05.transformer_0.block_15.ffn_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | 2.448 | 3.200 | 1.307x | 96.9% | 0.066 | fused_ffn (16) | fused_ffn (2.124x; 16) |
| 157 | outer_05.transformer_0.block_15.ffn_linear1_gate_swiglu | fused_ffn | fused_ffn; fused_ffn_candidate_a_reuse_kernel; g18x37x1; b128x1x1; r136; s32768 | 60 | 41.072 | 45.440 | 1.106x | 67.8% | 0.395 | fused_ffn (23) | wide_qkv (1.392x; 14) |
| 158 | outer_05.transformer_0.block_15.ffn_linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 60 | 20.896 | 33.408 | 1.599x | 97.6% | 0.719 | fused_ffn (30) | wide_qkv (1.755x; 16) |
| 159 | outer_05.transformer_1.block_16.attention_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | 2.432 | 4.016 | 1.651x | 100.0% | 0.102 | linear2_residual (16) | fused_ffn (2.039x; 14) |
| 160 | outer_05.transformer_1.block_16.attention_qkv_projection | wide_qkv | wide_qkv; wide_qkv_kernel; g9x37x1; b128x1x1; r136; s65536 | 60 | 19.345 | 25.184 | 1.302x | 84.2% | 0.357 | linear2_residual (16) | linear2_residual (1.490x; 16) |
| 161 | outer_05.transformer_1.block_16.attention_qk_rope | qk_rope | qk_rope; fusedQKRoPE19HalfKernel; g361x13x1; b192x1x1; r16; s0 | 60 | 4.096 | 5.856 | 1.430x | 99.1% | 0.150 | fa4 (16) | wide_qkv (2.270x; 16) |
| 162 | outer_05.transformer_1.block_16.attention_fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 60 | 11.937 | 15.632 | 1.310x | 65.2% | 0.303 | library_gemm (15) | linear2_residual (1.815x; 14) |
| 163 | outer_05.transformer_1.block_16.attention_out_projection_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 60 | 8.496 | 19.488 | 2.294x | 96.1% | 0.621 | fused_ffn (28) | fused_ffn (2.490x; 28) |
| 164 | outer_05.transformer_1.block_16.ffn_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | 2.432 | 3.328 | 1.368x | 100.0% | 0.083 | fused_ffn (16) | fused_ffn (2.414x; 16) |
| 165 | outer_05.transformer_1.block_16.ffn_linear1_gate_swiglu | fused_ffn | fused_ffn; fused_ffn_candidate_a_reuse_kernel; g18x37x1; b128x1x1; r136; s32768 | 60 | 40.481 | 44.912 | 1.109x | 70.4% | 0.281 | fused_ffn (25) | linear2_residual (1.249x; 21) |
| 166 | outer_05.transformer_1.block_16.ffn_linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 60 | 20.768 | 32.305 | 1.555x | 97.8% | 0.695 | fused_ffn (30) | wide_qkv (1.709x; 16) |
| 167 | outer_05.transformer_2.block_17.attention_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | 2.432 | 3.824 | 1.572x | 100.0% | 0.086 | linear2_residual (16) | wide_qkv (2.013x; 16) |
| 168 | outer_05.transformer_2.block_17.attention_qkv_projection | wide_qkv | wide_qkv; wide_qkv_kernel; g9x37x1; b128x1x1; r136; s65536 | 60 | 19.264 | 24.672 | 1.281x | 82.2% | 0.326 | linear2_residual (16) | linear2_residual (1.449x; 16) |
| 169 | outer_05.transformer_2.block_17.attention_qk_rope | qk_rope | qk_rope; fusedQKRoPE19HalfKernel; g361x13x1; b192x1x1; r16; s0 | 60 | 4.096 | 5.760 | 1.406x | 99.0% | 0.144 | fa4 (16) | wide_qkv (2.238x; 16) |
| 170 | outer_05.transformer_2.block_17.attention_fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 60 | 11.856 | 15.312 | 1.291x | 64.3% | 0.262 | library_gemm (15) | linear2_residual (1.672x; 14) |
| 171 | outer_05.transformer_2.block_17.attention_out_projection_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 60 | 8.448 | 19.201 | 2.273x | 96.6% | 0.595 | library_gemm (35) | fused_ffn (2.519x; 25) |
| 172 | outer_05.transformer_2.block_17.ffn_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | 2.400 | 3.056 | 1.273x | 97.0% | 0.084 | library_gemm (17) | fused_ffn (2.713x; 16) |
| 173 | outer_05.transformer_2.block_17.ffn_linear1_gate_swiglu | fused_ffn | fused_ffn; fused_ffn_candidate_a_reuse_kernel; g18x37x1; b128x1x1; r136; s32768 | 60 | 38.672 | 47.809 | 1.236x | 75.7% | 0.534 | fused_ffn (25) | wide_qkv (1.415x; 13) |
| 174 | outer_05.transformer_2.block_17.ffn_linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 60 | 20.704 | 30.576 | 1.477x | 97.8% | 0.601 | fa4 (27) | linear2_residual (1.643x; 5) |
| 175 | outer_05.post_norm_silu | affine_silu | affine_silu; affineSiluHalf2Kernel; g4693x1x1; b192x1x1; r17; s0 | 60 | 3.328 | 3.905 | 1.173x | 99.7% | 0.043 | fa4 (27) | linear2_residual (1.197x; 16) |
| 176 | outer_05.post_projection_c384_to_c768_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b256x1x1; r154; s73728 | 60 | 14.336 | 18.145 | 1.266x | 96.6% | 0.230 | library_gemm (28) | linear2_residual (1.336x; 16) |
| 177 | outer_06.pre_norm_silu | affine_silu | affine_silu; affineSiluHalf2Kernel; g4693x1x1; b384x1x1; r17; s0 | 60 | 5.056 | 6.944 | 1.373x | 82.6% | 0.155 | library_gemm (56) | library_gemm (1.373x; 56) |
| 178 | outer_06.pre_projection_c768_to_c384 | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 60 | 11.936 | 15.104 | 1.265x | 95.0% | 0.286 | fused_ffn (28) | library_gemm (1.672x; 16) |
| 179 | outer_06.transformer_0.block_18.attention_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | 2.432 | 4.816 | 1.980x | 100.0% | 0.127 | fused_ffn (27) | wide_qkv (2.125x; 16) |
| 180 | outer_06.transformer_0.block_18.attention_qkv_projection | wide_qkv | wide_qkv; wide_qkv_kernel; g9x37x1; b128x1x1; r136; s65536 | 60 | 19.232 | 24.864 | 1.293x | 82.2% | 0.360 | fused_ffn (27) | fused_ffn (1.414x; 27) |
| 181 | outer_06.transformer_0.block_18.attention_qk_rope | qk_rope | qk_rope; fusedQKRoPE19HalfKernel; g361x13x1; b192x1x1; r16; s0 | 60 | 4.128 | 5.936 | 1.438x | 100.0% | 0.160 | linear2_residual (27) | wide_qkv (2.287x; 16) |
| 182 | outer_06.transformer_0.block_18.attention_fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 60 | 11.873 | 15.536 | 1.309x | 72.3% | 0.358 | linear2_residual (27) | linear2_residual (1.774x; 27) |
| 183 | outer_06.transformer_0.block_18.attention_out_projection_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 60 | 8.464 | 19.409 | 2.293x | 96.1% | 0.584 | library_gemm (33) | fused_ffn (2.367x; 13) |
| 184 | outer_06.transformer_0.block_18.ffn_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | 2.496 | 3.136 | 1.257x | 97.2% | 0.064 | library_gemm (17) | fused_ffn (2.122x; 16) |
| 185 | outer_06.transformer_0.block_18.ffn_linear1_gate_swiglu | fused_ffn | fused_ffn; fused_ffn_candidate_a_reuse_kernel; g18x37x1; b128x1x1; r136; s32768 | 60 | 40.544 | 45.856 | 1.131x | 70.1% | 0.414 | fused_ffn (16) | wide_qkv (1.326x; 14) |
| 186 | outer_06.transformer_0.block_18.ffn_linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 60 | 20.849 | 31.184 | 1.496x | 97.5% | 0.618 | fused_ffn (30) | wide_qkv (1.559x; 16) |
| 187 | outer_06.transformer_1.block_19.attention_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | 2.464 | 3.984 | 1.617x | 100.0% | 0.101 | linear2_residual (16) | fused_ffn (2.247x; 14) |
| 188 | outer_06.transformer_1.block_19.attention_qkv_projection | wide_qkv | wide_qkv; wide_qkv_kernel; g9x37x1; b128x1x1; r136; s65536 | 60 | 19.329 | 25.696 | 1.329x | 85.0% | 0.421 | linear2_residual (16) | fused_ffn (1.587x; 14) |
| 189 | outer_06.transformer_1.block_19.attention_qk_rope | qk_rope | qk_rope; fusedQKRoPE19HalfKernel; g361x13x1; b192x1x1; r16; s0 | 60 | 4.128 | 5.824 | 1.411x | 100.0% | 0.148 | fa4 (16) | wide_qkv (2.229x; 16) |
| 190 | outer_06.transformer_1.block_19.attention_fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 60 | 11.968 | 15.184 | 1.269x | 63.6% | 0.284 | library_gemm (16) | linear2_residual (1.775x; 14) |
| 191 | outer_06.transformer_1.block_19.attention_out_projection_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 60 | 8.496 | 19.504 | 2.296x | 96.0% | 0.626 | fused_ffn (24) | fused_ffn (2.557x; 24) |
| 192 | outer_06.transformer_1.block_19.ffn_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | 2.432 | 3.424 | 1.408x | 100.0% | 0.071 | fused_ffn (16) | fused_ffn (2.026x; 16) |
| 193 | outer_06.transformer_1.block_19.ffn_linear1_gate_swiglu | fused_ffn | fused_ffn; fused_ffn_candidate_a_reuse_kernel; g18x37x1; b128x1x1; r136; s32768 | 60 | 39.216 | 44.624 | 1.138x | 77.0% | 0.386 | linear2_residual (30) | fused_ffn (1.174x; 16) |
| 194 | outer_06.transformer_1.block_19.ffn_linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 60 | 20.864 | 32.545 | 1.560x | 97.8% | 0.692 | fused_ffn (30) | wide_qkv (1.629x; 16) |
| 195 | outer_06.transformer_2.block_20.attention_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | 2.432 | 3.888 | 1.599x | 100.0% | 0.097 | linear2_residual (16) | fused_ffn (2.171x; 14) |
| 196 | outer_06.transformer_2.block_20.attention_qkv_projection | wide_qkv | wide_qkv; wide_qkv_kernel; g9x37x1; b128x1x1; r136; s65536 | 60 | 19.280 | 25.056 | 1.300x | 82.3% | 0.375 | linear2_residual (16) | linear2_residual (1.524x; 16) |
| 197 | outer_06.transformer_2.block_20.attention_qk_rope | qk_rope | qk_rope; fusedQKRoPE19HalfKernel; g361x13x1; b192x1x1; r16; s0 | 60 | 4.144 | 5.920 | 1.429x | 100.0% | 0.150 | fa4 (16) | wide_qkv (2.236x; 16) |
| 198 | outer_06.transformer_2.block_20.attention_fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 60 | 11.904 | 15.168 | 1.274x | 63.1% | 0.280 | library_gemm (16) | linear2_residual (1.761x; 14) |
| 199 | outer_06.transformer_2.block_20.attention_out_projection_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 60 | 8.480 | 19.712 | 2.325x | 96.6% | 0.618 | library_gemm (36) | fused_ffn (2.419x; 24) |
| 200 | outer_06.transformer_2.block_20.ffn_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | 2.448 | 3.152 | 1.288x | 96.1% | 0.054 | fused_ffn (16) | fused_ffn (1.817x; 16) |
| 201 | outer_06.transformer_2.block_20.ffn_linear1_gate_swiglu | fused_ffn | fused_ffn; fused_ffn_candidate_a_reuse_kernel; g18x37x1; b128x1x1; r136; s32768 | 60 | 35.312 | 42.529 | 1.204x | 83.7% | 0.591 | linear2_residual (30) | wide_qkv (1.428x; 14) |
| 202 | outer_06.transformer_2.block_20.ffn_linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 60 | 21.040 | 33.248 | 1.580x | 97.8% | 0.695 | fa4 (28) | library_gemm (1.712x; 16) |
| 203 | outer_06.post_norm_silu | affine_silu | affine_silu; affineSiluHalf2Kernel; g4693x1x1; b192x1x1; r17; s0 | 60 | 3.376 | 4.288 | 1.270x | 100.0% | 0.066 | fa4 (28) | fa4 (1.427x; 28) |
| 204 | outer_06.post_projection_c384_to_c768_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b256x1x1; r154; s73728 | 60 | 14.544 | 19.136 | 1.316x | 96.2% | 0.308 | library_gemm (28) | linear2_residual (1.539x; 16) |
| 205 | outer_07.pre_norm_silu | affine_silu | affine_silu; affineSiluHalf2Kernel; g4693x1x1; b384x1x1; r17; s0 | 60 | 5.088 | 7.377 | 1.450x | 85.7% | 0.179 | library_gemm (60) | library_gemm (1.450x; 60) |
| 206 | outer_07.pre_projection_c768_to_c384 | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 60 | 12.032 | 16.016 | 1.331x | 95.1% | 0.310 | fused_ffn (28) | library_gemm (1.790x; 14) |
| 207 | outer_07.transformer_0.block_21.attention_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | 2.464 | 5.040 | 2.046x | 100.0% | 0.142 | fused_ffn (28) | wide_qkv (2.188x; 16) |
| 208 | outer_07.transformer_0.block_21.attention_qkv_projection | wide_qkv | wide_qkv; wide_qkv_kernel; g9x37x1; b128x1x1; r136; s65536 | 60 | 19.376 | 25.872 | 1.335x | 81.9% | 0.395 | fused_ffn (28) | fused_ffn (1.451x; 28) |
| 209 | outer_07.transformer_0.block_21.attention_qk_rope | qk_rope | qk_rope; fusedQKRoPE19HalfKernel; g361x13x1; b192x1x1; r16; s0 | 60 | 4.128 | 5.984 | 1.450x | 100.0% | 0.158 | linear2_residual (28) | wide_qkv (2.256x; 16) |
| 210 | outer_07.transformer_0.block_21.attention_fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 60 | 11.984 | 15.968 | 1.332x | 73.9% | 0.379 | linear2_residual (28) | linear2_residual (1.742x; 28) |
| 211 | outer_07.transformer_0.block_21.attention_out_projection_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 60 | 8.512 | 20.064 | 2.357x | 96.2% | 0.622 | library_gemm (37) | wide_qkv (2.408x; 14) |
| 212 | outer_07.transformer_0.block_21.ffn_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | 2.432 | 3.248 | 1.336x | 96.5% | 0.060 | library_gemm (17) | fused_ffn (1.836x; 16) |
| 213 | outer_07.transformer_0.block_21.ffn_linear1_gate_swiglu | fused_ffn | fused_ffn; fused_ffn_candidate_a_reuse_kernel; g18x37x1; b128x1x1; r136; s32768 | 60 | 35.712 | 42.304 | 1.185x | 73.5% | 0.465 | fused_ffn (16) | wide_qkv (1.445x; 14) |
| 214 | outer_07.transformer_0.block_21.ffn_linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 60 | 21.056 | 35.456 | 1.684x | 97.7% | 0.796 | fused_ffn (30) | wide_qkv (1.784x; 16) |
| 215 | outer_07.transformer_1.block_22.attention_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | 2.432 | 4.144 | 1.704x | 100.0% | 0.110 | linear2_residual (16) | fused_ffn (2.224x; 14) |
| 216 | outer_07.transformer_1.block_22.attention_qkv_projection | wide_qkv | wide_qkv; wide_qkv_kernel; g9x37x1; b128x1x1; r136; s65536 | 60 | 19.488 | 25.760 | 1.322x | 85.0% | 0.403 | linear2_residual (16) | linear2_residual (1.548x; 16) |
| 217 | outer_07.transformer_1.block_22.attention_qk_rope | qk_rope | qk_rope; fusedQKRoPE19HalfKernel; g361x13x1; b192x1x1; r16; s0 | 60 | 4.160 | 5.968 | 1.435x | 100.0% | 0.154 | fa4 (16) | wide_qkv (2.250x; 16) |
| 218 | outer_07.transformer_1.block_22.attention_fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 60 | 12.016 | 15.393 | 1.281x | 64.9% | 0.327 | library_gemm (16) | linear2_residual (1.947x; 14) |
| 219 | outer_07.transformer_1.block_22.attention_out_projection_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 60 | 8.576 | 19.601 | 2.286x | 96.0% | 0.596 | fused_ffn (26) | wide_qkv (2.364x; 14) |
| 220 | outer_07.transformer_1.block_22.ffn_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | 2.496 | 3.456 | 1.385x | 99.3% | 0.062 | fused_ffn (16) | fused_ffn (1.853x; 16) |
| 221 | outer_07.transformer_1.block_22.ffn_linear1_gate_swiglu | fused_ffn | fused_ffn; fused_ffn_candidate_a_reuse_kernel; g18x37x1; b128x1x1; r136; s32768 | 60 | 35.825 | 41.569 | 1.160x | 79.0% | 0.421 | linear2_residual (30) | linear2_residual (1.184x; 30) |
| 222 | outer_07.transformer_1.block_22.ffn_linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 60 | 21.056 | 34.368 | 1.632x | 97.9% | 0.796 | fused_ffn (30) | wide_qkv (1.789x; 16) |
| 223 | outer_07.transformer_2.block_23.attention_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | 2.496 | 4.048 | 1.622x | 100.0% | 0.102 | linear2_residual (16) | fused_ffn (2.224x; 14) |
| 224 | outer_07.transformer_2.block_23.attention_qkv_projection | wide_qkv | wide_qkv; wide_qkv_kernel; g9x37x1; b128x1x1; r136; s65536 | 60 | 19.520 | 25.921 | 1.328x | 82.5% | 0.393 | linear2_residual (16) | linear2_residual (1.545x; 16) |
| 225 | outer_07.transformer_2.block_23.attention_qk_rope | qk_rope | qk_rope; fusedQKRoPE19HalfKernel; g361x13x1; b192x1x1; r16; s0 | 60 | 4.128 | 5.984 | 1.450x | 100.0% | 0.149 | fa4 (16) | wide_qkv (2.198x; 16) |
| 226 | outer_07.transformer_2.block_23.attention_fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 60 | 12.016 | 15.649 | 1.302x | 65.4% | 0.309 | library_gemm (16) | linear2_residual (1.842x; 14) |
| 227 | outer_07.transformer_2.block_23.attention_out_projection_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 60 | 8.560 | 19.505 | 2.279x | 96.6% | 0.597 | library_gemm (35) | fused_ffn (2.288x; 25) |
| 228 | outer_07.transformer_2.block_23.ffn_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | 2.432 | 3.072 | 1.263x | 95.7% | 0.055 | fused_ffn (16) | fused_ffn (1.842x; 16) |
| 229 | outer_07.transformer_2.block_23.ffn_linear1_gate_swiglu | fused_ffn | fused_ffn; fused_ffn_candidate_a_reuse_kernel; g18x37x1; b128x1x1; r136; s32768 | 60 | 36.033 | 43.441 | 1.206x | 83.7% | 0.578 | linear2_residual (30) | wide_qkv (1.411x; 14) |
| 230 | outer_07.transformer_2.block_23.ffn_linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 60 | 21.137 | 34.032 | 1.610x | 97.9% | 0.717 | fa4 (28) | library_gemm (1.741x; 16) |
| 231 | outer_07.post_norm_silu | affine_silu | affine_silu; affineSiluHalf2Kernel; g4693x1x1; b192x1x1; r17; s0 | 60 | 3.360 | 4.352 | 1.295x | 100.0% | 0.072 | fa4 (28) | fa4 (1.481x; 28) |
| 232 | outer_07.post_projection_c384_to_c768_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b256x1x1; r154; s73728 | 60 | 14.608 | 19.072 | 1.306x | 96.4% | 0.318 | library_gemm (28) | linear2_residual (1.573x; 16) |
| 233 | outer_08.pre_norm_silu | affine_silu | affine_silu; affineSiluHalf2Kernel; g4693x1x1; b384x1x1; r17; s0 | 60 | 5.120 | 7.424 | 1.450x | 85.5% | 0.179 | library_gemm (60) | library_gemm (1.450x; 60) |
| 234 | outer_08.pre_projection_c768_to_c384 | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 60 | 12.064 | 16.336 | 1.354x | 95.2% | 0.324 | fused_ffn (28) | library_gemm (1.790x; 14) |
| 235 | outer_08.transformer_0.block_24.attention_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | 2.464 | 5.056 | 2.052x | 100.0% | 0.140 | fused_ffn (28) | wide_qkv (2.175x; 16) |
| 236 | outer_08.transformer_0.block_24.attention_qkv_projection | wide_qkv | wide_qkv; wide_qkv_kernel; g9x37x1; b128x1x1; r136; s65536 | 60 | 19.440 | 26.208 | 1.348x | 81.9% | 0.414 | fused_ffn (28) | fused_ffn (1.469x; 28) |
| 237 | outer_08.transformer_0.block_24.attention_qk_rope | qk_rope | qk_rope; fusedQKRoPE19HalfKernel; g361x13x1; b192x1x1; r16; s0 | 60 | 4.144 | 5.920 | 1.429x | 100.0% | 0.155 | linear2_residual (28) | wide_qkv (2.270x; 16) |
| 238 | outer_08.transformer_0.block_24.attention_fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 60 | 12.032 | 16.273 | 1.352x | 74.0% | 0.364 | linear2_residual (28) | linear2_residual (1.713x; 28) |
| 239 | outer_08.transformer_0.block_24.attention_out_projection_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 60 | 8.592 | 20.064 | 2.335x | 96.3% | 0.615 | library_gemm (40) | wide_qkv (2.363x; 14) |
| 240 | outer_08.transformer_0.block_24.ffn_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | 2.464 | 3.040 | 1.234x | 96.8% | 0.065 | fused_ffn (16) | fused_ffn (2.065x; 16) |
| 241 | outer_08.transformer_0.block_24.ffn_linear1_gate_swiglu | fused_ffn | fused_ffn; fused_ffn_candidate_a_reuse_kernel; g18x37x1; b128x1x1; r136; s32768 | 60 | 40.401 | 45.776 | 1.133x | 70.7% | 0.453 | fused_ffn (22) | wide_qkv (1.388x; 14) |
| 242 | outer_08.transformer_0.block_24.ffn_linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 60 | 20.960 | 33.520 | 1.599x | 97.6% | 0.698 | fused_ffn (30) | wide_qkv (1.714x; 16) |
| 243 | outer_08.transformer_1.block_25.attention_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | 2.464 | 3.952 | 1.604x | 100.0% | 0.108 | linear2_residual (16) | fused_ffn (2.318x; 14) |
| 244 | outer_08.transformer_1.block_25.attention_qkv_projection | wide_qkv | wide_qkv; wide_qkv_kernel; g9x37x1; b128x1x1; r136; s65536 | 60 | 19.456 | 25.840 | 1.328x | 83.5% | 0.408 | linear2_residual (16) | fused_ffn (1.562x; 14) |
| 245 | outer_08.transformer_1.block_25.attention_qk_rope | qk_rope | qk_rope; fusedQKRoPE19HalfKernel; g361x13x1; b192x1x1; r16; s0 | 60 | 4.128 | 5.904 | 1.430x | 98.0% | 0.151 | fa4 (16) | wide_qkv (2.260x; 16) |
| 246 | outer_08.transformer_1.block_25.attention_fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 60 | 11.968 | 16.000 | 1.337x | 65.4% | 0.318 | fused_ffn (14) | linear2_residual (1.853x; 14) |
| 247 | outer_08.transformer_1.block_25.attention_out_projection_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 60 | 8.496 | 19.616 | 2.309x | 96.1% | 0.621 | fused_ffn (25) | fused_ffn (2.569x; 25) |
| 248 | outer_08.transformer_1.block_25.ffn_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | 2.464 | 3.376 | 1.370x | 100.0% | 0.074 | fused_ffn (16) | fused_ffn (2.104x; 16) |
| 249 | outer_08.transformer_1.block_25.ffn_linear1_gate_swiglu | fused_ffn | fused_ffn; fused_ffn_candidate_a_reuse_kernel; g18x37x1; b128x1x1; r136; s32768 | 60 | 41.041 | 45.968 | 1.120x | 73.2% | 0.287 | fused_ffn (24) | linear2_residual (1.247x; 22) |
| 250 | outer_08.transformer_1.block_25.ffn_linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 60 | 20.977 | 32.944 | 1.571x | 97.9% | 0.741 | fused_ffn (30) | wide_qkv (1.712x; 16) |
| 251 | outer_08.transformer_2.block_26.attention_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | 2.496 | 4.048 | 1.622x | 100.0% | 0.096 | linear2_residual (16) | wide_qkv (2.038x; 16) |
| 252 | outer_08.transformer_2.block_26.attention_qkv_projection | wide_qkv | wide_qkv; wide_qkv_kernel; g9x37x1; b128x1x1; r136; s65536 | 60 | 19.376 | 25.168 | 1.299x | 82.5% | 0.353 | linear2_residual (16) | linear2_residual (1.494x; 16) |
| 253 | outer_08.transformer_2.block_26.attention_qk_rope | qk_rope | qk_rope; fusedQKRoPE19HalfKernel; g361x13x1; b192x1x1; r16; s0 | 60 | 4.128 | 5.872 | 1.422x | 100.0% | 0.151 | fa4 (16) | wide_qkv (2.271x; 16) |
| 254 | outer_08.transformer_2.block_26.attention_fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 60 | 11.953 | 15.601 | 1.305x | 64.1% | 0.289 | library_gemm (16) | linear2_residual (1.764x; 14) |
| 255 | outer_08.transformer_2.block_26.attention_out_projection_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 60 | 8.480 | 19.664 | 2.319x | 96.6% | 0.614 | library_gemm (34) | fused_ffn (2.470x; 26) |
| 256 | outer_08.transformer_2.block_26.ffn_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | 2.464 | 3.200 | 1.299x | 96.7% | 0.063 | library_gemm (17) | fused_ffn (2.071x; 16) |
| 257 | outer_08.transformer_2.block_26.ffn_linear1_gate_swiglu | fused_ffn | fused_ffn; fused_ffn_candidate_a_reuse_kernel; g18x37x1; b128x1x1; r136; s32768 | 60 | 40.224 | 47.456 | 1.180x | 79.4% | 0.472 | linear2_residual (28) | wide_qkv (1.336x; 13) |
| 258 | outer_08.transformer_2.block_26.ffn_linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 60 | 20.928 | 32.321 | 1.544x | 97.8% | 0.649 | fa4 (27) | library_gemm (1.605x; 16) |
| 259 | outer_08.post_norm_silu | affine_silu | affine_silu; affineSiluHalf2Kernel; g4693x1x1; b192x1x1; r17; s0 | 60 | 3.360 | 4.160 | 1.238x | 99.9% | 0.057 | fa4 (27) | linear2_residual (1.257x; 16) |
| 260 | outer_08.post_projection_c384_to_c768_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b256x1x1; r154; s73728 | 60 | 14.448 | 18.832 | 1.303x | 96.3% | 0.309 | library_gemm (26) | linear2_residual (1.578x; 16) |
| 261 | outer_09.pre_norm_silu | affine_silu | affine_silu; affineSiluHalf2Kernel; g4693x1x1; b384x1x1; r17; s0 | 60 | 5.056 | 7.345 | 1.453x | 84.7% | 0.172 | library_gemm (55) | library_gemm (1.456x; 55) |
| 262 | outer_09.pre_projection_c768_to_c384 | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 60 | 12.016 | 15.793 | 1.314x | 95.0% | 0.295 | fused_ffn (27) | library_gemm (1.712x; 6) |
| 263 | outer_09.transformer_0.block_27.attention_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | 2.400 | 4.896 | 2.040x | 100.0% | 0.142 | fused_ffn (27) | wide_qkv (2.360x; 16) |
| 264 | outer_09.transformer_0.block_27.attention_qkv_projection | wide_qkv | wide_qkv; wide_qkv_kernel; g9x37x1; b128x1x1; r136; s65536 | 60 | 19.280 | 26.128 | 1.355x | 81.9% | 0.398 | fused_ffn (27) | fused_ffn (1.451x; 27) |
| 265 | outer_09.transformer_0.block_27.attention_qk_rope | qk_rope | qk_rope; fusedQKRoPE19HalfKernel; g361x13x1; b192x1x1; r16; s0 | 60 | 4.112 | 5.936 | 1.444x | 100.0% | 0.160 | linear2_residual (27) | wide_qkv (2.276x; 16) |
| 266 | outer_09.transformer_0.block_27.attention_fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 60 | 11.936 | 15.968 | 1.338x | 72.6% | 0.344 | linear2_residual (27) | linear2_residual (1.694x; 27) |
| 267 | outer_09.transformer_0.block_27.attention_out_projection_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 60 | 8.480 | 19.441 | 2.293x | 96.2% | 0.597 | library_gemm (37) | fused_ffn (2.396x; 9) |
| 268 | outer_09.transformer_0.block_27.ffn_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | 2.464 | 3.328 | 1.351x | 97.2% | 0.078 | library_gemm (17) | fused_ffn (2.442x; 16) |
| 269 | outer_09.transformer_0.block_27.ffn_linear1_gate_swiglu | fused_ffn | fused_ffn; fused_ffn_candidate_a_reuse_kernel; g18x37x1; b128x1x1; r136; s32768 | 60 | 39.328 | 44.704 | 1.137x | 64.4% | 0.426 | fused_ffn (25) | wide_qkv (1.429x; 14) |
| 270 | outer_09.transformer_0.block_27.ffn_linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 60 | 20.768 | 32.176 | 1.549x | 97.5% | 0.652 | fused_ffn (30) | wide_qkv (1.713x; 16) |
| 271 | outer_09.transformer_1.block_28.attention_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | 2.432 | 3.968 | 1.632x | 100.0% | 0.100 | linear2_residual (16) | wide_qkv (2.105x; 16) |
| 272 | outer_09.transformer_1.block_28.attention_qkv_projection | wide_qkv | wide_qkv; wide_qkv_kernel; g9x37x1; b128x1x1; r136; s65536 | 60 | 19.184 | 24.273 | 1.265x | 83.7% | 0.314 | linear2_residual (16) | linear2_residual (1.433x; 16) |
| 273 | outer_09.transformer_1.block_28.attention_qk_rope | qk_rope | qk_rope; fusedQKRoPE19HalfKernel; g361x13x1; b192x1x1; r16; s0 | 60 | 4.096 | 5.760 | 1.406x | 98.1% | 0.144 | fa4 (16) | wide_qkv (2.219x; 16) |
| 274 | outer_09.transformer_1.block_28.attention_fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 60 | 11.840 | 15.152 | 1.280x | 64.8% | 0.275 | library_gemm (14) | linear2_residual (1.742x; 14) |
| 275 | outer_09.transformer_1.block_28.attention_out_projection_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 60 | 8.464 | 19.088 | 2.255x | 95.9% | 0.596 | library_gemm (24) | fused_ffn (2.563x; 22) |
| 276 | outer_09.transformer_1.block_28.ffn_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | 2.464 | 3.296 | 1.338x | 100.0% | 0.086 | fused_ffn (16) | fused_ffn (2.513x; 16) |
| 277 | outer_09.transformer_1.block_28.ffn_linear1_gate_swiglu | fused_ffn | fused_ffn; fused_ffn_candidate_a_reuse_kernel; g18x37x1; b128x1x1; r136; s32768 | 60 | 38.400 | 43.169 | 1.124x | 69.9% | 0.333 | fused_ffn (23) | linear2_residual (1.267x; 23) |
| 278 | outer_09.transformer_1.block_28.ffn_linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 60 | 20.640 | 31.504 | 1.526x | 97.8% | 0.679 | fused_ffn (30) | wide_qkv (1.709x; 16) |
| 279 | outer_09.transformer_2.block_29.attention_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | 2.400 | 4.016 | 1.673x | 100.0% | 0.092 | linear2_residual (16) | wide_qkv (2.073x; 16) |
| 280 | outer_09.transformer_2.block_29.attention_qkv_projection | wide_qkv | wide_qkv; wide_qkv_kernel; g9x37x1; b128x1x1; r136; s65536 | 60 | 19.041 | 23.985 | 1.260x | 81.3% | 0.304 | linear2_residual (16) | linear2_residual (1.433x; 16) |
| 281 | outer_09.transformer_2.block_29.attention_qk_rope | qk_rope | qk_rope; fusedQKRoPE19HalfKernel; g361x13x1; b192x1x1; r16; s0 | 60 | 4.064 | 5.664 | 1.394x | 100.0% | 0.143 | fa4 (16) | wide_qkv (2.252x; 16) |
| 282 | outer_09.transformer_2.block_29.attention_fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 60 | 11.777 | 14.656 | 1.244x | 63.1% | 0.242 | library_gemm (16) | linear2_residual (1.638x; 14) |
| 283 | outer_09.transformer_2.block_29.attention_out_projection_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 60 | 8.448 | 19.232 | 2.277x | 96.6% | 0.552 | library_gemm (33) | fused_ffn (2.470x; 27) |
| 284 | outer_09.transformer_2.block_29.ffn_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | 2.464 | 3.168 | 1.286x | 97.8% | 0.069 | library_gemm (21) | fused_ffn (2.247x; 16) |
| 285 | outer_09.transformer_2.block_29.ffn_linear1_gate_swiglu | fused_ffn | fused_ffn; fused_ffn_candidate_a_reuse_kernel; g18x37x1; b128x1x1; r136; s32768 | 60 | 40.224 | 45.713 | 1.136x | 76.6% | 0.351 | fused_ffn (29) | wide_qkv (1.322x; 9) |
| 286 | outer_09.transformer_2.block_29.ffn_linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 60 | 20.544 | 30.384 | 1.479x | 97.8% | 0.601 | fa4 (23) | linear2_residual (1.642x; 14) |
| 287 | outer_09.post_norm_silu | affine_silu | affine_silu; affineSiluHalf2Kernel; g4693x1x1; b192x1x1; r17; s0 | 60 | 3.328 | 3.872 | 1.163x | 99.1% | 0.047 | fa4 (23) | wide_qkv (1.769x; 5) |
| 288 | outer_09.post_projection_c384_to_c768_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b256x1x1; r154; s73728 | 60 | 14.209 | 17.840 | 1.256x | 96.7% | 0.258 | library_gemm (23) | fused_ffn (2.054x; 5) |
| 289 | outer_10.pre_norm_silu | affine_silu | affine_silu; affineSiluHalf2Kernel; g4693x1x1; b384x1x1; r17; s0 | 60 | 5.024 | 6.912 | 1.376x | 84.6% | 0.154 | library_gemm (50) | fused_ffn (2.025x; 5) |
| 290 | outer_10.pre_projection_c768_to_c384 | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 60 | 11.792 | 15.136 | 1.284x | 95.3% | 0.363 | fused_ffn (23) | library_gemm (1.674x; 21) |
| 291 | outer_10.transformer_0.block_30.attention_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | 2.432 | 5.088 | 2.092x | 100.0% | 0.133 | fused_ffn (23) | wide_qkv (2.217x; 16) |
| 292 | outer_10.transformer_0.block_30.attention_qkv_projection | wide_qkv | wide_qkv; wide_qkv_kernel; g9x37x1; b128x1x1; r136; s65536 | 60 | 19.040 | 24.544 | 1.289x | 80.8% | 0.345 | fused_ffn (23) | linear2_residual (1.487x; 5) |
| 293 | outer_10.transformer_0.block_30.attention_qk_rope | qk_rope | qk_rope; fusedQKRoPE19HalfKernel; g361x13x1; b192x1x1; r16; s0 | 60 | 4.064 | 5.856 | 1.441x | 97.3% | 0.160 | linear2_residual (23) | wide_qkv (2.280x; 16) |
| 294 | outer_10.transformer_0.block_30.attention_fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 60 | 11.808 | 17.088 | 1.447x | 71.6% | 0.322 | linear2_residual (23) | linear2_residual (1.664x; 23) |
| 295 | outer_10.transformer_0.block_30.attention_out_projection_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 60 | 8.416 | 19.264 | 2.289x | 96.0% | 0.538 | library_gemm (37) | wide_qkv (2.363x; 14) |
| 296 | outer_10.transformer_0.block_30.ffn_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | 2.432 | 3.392 | 1.395x | 97.8% | 0.059 | library_gemm (21) | fused_ffn (1.829x; 16) |
| 297 | outer_10.transformer_0.block_30.ffn_linear1_gate_swiglu | fused_ffn | fused_ffn; fused_ffn_candidate_a_reuse_kernel; g18x37x1; b128x1x1; r136; s32768 | 60 | 36.016 | 42.416 | 1.178x | 74.4% | 0.409 | fused_ffn (16) | wide_qkv (1.338x; 14) |
| 298 | outer_10.transformer_0.block_30.ffn_linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 60 | 20.848 | 33.328 | 1.599x | 97.4% | 0.692 | fused_ffn (30) | wide_qkv (1.688x; 16) |
| 299 | outer_10.transformer_1.block_31.attention_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | 2.480 | 3.968 | 1.600x | 100.0% | 0.118 | linear2_residual (16) | fused_ffn (2.600x; 14) |
| 300 | outer_10.transformer_1.block_31.attention_qkv_projection | wide_qkv | wide_qkv; wide_qkv_kernel; g9x37x1; b128x1x1; r136; s65536 | 60 | 19.409 | 25.808 | 1.330x | 83.9% | 0.405 | linear2_residual (16) | linear2_residual (1.542x; 16) |
| 301 | outer_10.transformer_1.block_31.attention_qk_rope | qk_rope | qk_rope; fusedQKRoPE19HalfKernel; g361x13x1; b192x1x1; r16; s0 | 60 | 4.128 | 5.888 | 1.426x | 100.0% | 0.155 | fa4 (16) | wide_qkv (2.287x; 16) |
| 302 | outer_10.transformer_1.block_31.attention_fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 60 | 11.952 | 15.264 | 1.277x | 64.2% | 0.304 | library_gemm (16) | linear2_residual (1.869x; 14) |
| 303 | outer_10.transformer_1.block_31.attention_out_projection_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 60 | 8.496 | 19.456 | 2.290x | 96.0% | 0.601 | fused_ffn (26) | fused_ffn (2.341x; 26) |
| 304 | outer_10.transformer_1.block_31.ffn_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | 2.432 | 3.440 | 1.414x | 100.0% | 0.065 | fused_ffn (16) | fused_ffn (1.855x; 16) |
| 305 | outer_10.transformer_1.block_31.ffn_linear1_gate_swiglu | fused_ffn | fused_ffn; fused_ffn_candidate_a_reuse_kernel; g18x37x1; b128x1x1; r136; s32768 | 60 | 37.312 | 43.569 | 1.168x | 78.0% | 0.451 | linear2_residual (30) | fused_ffn (1.195x; 16) |
| 306 | outer_10.transformer_1.block_31.ffn_linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 60 | 20.944 | 32.465 | 1.550x | 97.7% | 0.685 | fused_ffn (30) | wide_qkv (1.664x; 16) |
| 307 | outer_10.transformer_2.block_32.attention_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | 2.432 | 3.664 | 1.507x | 100.0% | 0.106 | linear2_residual (16) | fused_ffn (2.421x; 14) |
| 308 | outer_10.transformer_2.block_32.attention_qkv_projection | wide_qkv | wide_qkv; wide_qkv_kernel; g9x37x1; b128x1x1; r136; s65536 | 60 | 19.392 | 26.081 | 1.345x | 82.3% | 0.421 | linear2_residual (16) | fused_ffn (1.554x; 14) |
| 309 | outer_10.transformer_2.block_32.attention_qk_rope | qk_rope | qk_rope; fusedQKRoPE19HalfKernel; g361x13x1; b192x1x1; r16; s0 | 60 | 4.160 | 5.872 | 1.412x | 100.0% | 0.150 | fa4 (16) | wide_qkv (2.269x; 16) |
| 310 | outer_10.transformer_2.block_32.attention_fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 60 | 11.968 | 15.424 | 1.289x | 63.8% | 0.293 | library_gemm (16) | linear2_residual (1.821x; 14) |
| 311 | outer_10.transformer_2.block_32.attention_out_projection_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 60 | 8.448 | 19.601 | 2.320x | 96.6% | 0.607 | library_gemm (37) | fused_ffn (2.394x; 23) |
| 312 | outer_10.transformer_2.block_32.ffn_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | 2.480 | 3.120 | 1.258x | 96.3% | 0.065 | fused_ffn (16) | fused_ffn (2.071x; 16) |
| 313 | outer_10.transformer_2.block_32.ffn_linear1_gate_swiglu | fused_ffn | fused_ffn; fused_ffn_candidate_a_reuse_kernel; g18x37x1; b128x1x1; r136; s32768 | 60 | 40.304 | 47.009 | 1.166x | 73.6% | 0.393 | fused_ffn (23) | linear2_residual (1.237x; 23) |
| 314 | outer_10.transformer_2.block_32.ffn_linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 60 | 20.960 | 32.336 | 1.543x | 96.7% | 0.605 | library_gemm (30) | library_gemm (1.573x; 30) |
| 315 | outer_10.post_norm_silu | affine_silu | affine_silu; affineSiluHalf2Kernel; g4693x1x1; b192x1x1; r17; s0 | 60 | 3.361 | 4.112 | 1.224x | 82.3% | 0.045 | library_gemm (16) | linear2_residual (1.257x; 16) |
| 316 | outer_10.post_projection_c384_to_c768_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b256x1x1; r154; s73728 | 60 | 14.432 | 18.960 | 1.314x | 94.9% | 0.269 | library_gemm (29) | linear2_residual (1.480x; 16) |
| 317 | trunk.tip_norm_silu | affine_silu | affine_silu; affineSiluHalf2Kernel; g4693x1x1; b384x1x1; r17; s0 | 60 | 5.056 | 7.040 | 1.392x | 80.5% | 0.145 | library_gemm (58) | library_gemm (1.392x; 58) |
| 318 | policy.p1_conv | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r80; s98304 | 60 | 6.208 | 9.600 | 1.546x | 92.1% | 0.190 | library_gemm (45) | library_gemm (1.670x; 45) |
| 319 | policy.g1_conv | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r80; s98304 | 60 | 5.920 | 9.040 | 1.527x | 94.9% | 0.316 | library_gemm (29) | fused_ffn (2.973x; 14) |
| 320 | policy.g1_norm_silu | head_elementwise | head_elementwise; applyCScaleBiasNHWCSiluHalfKernel; g1x73x13; b96x5x1; r16; s0 | 60 | 2.112 | 2.688 | 1.273x | 87.6% | 0.039 | library_gemm (17) | linear2_residual (1.462x; 14) |
| 321 | policy.g1_half_to_float | copy_reformat | copy_reformat; copyFromHalfKernel; g880x1x1; b512x1x1; r16; s0 | 60 | 1.568 | 1.824 | 1.163x | 74.4% | 0.022 | library_gemm (18) | linear2_residual (1.500x; 14) |
| 322 | policy.g1_global_pool | head_elementwise | head_elementwise; gPoolChannelsNHWCKernel; g2x1x13; b64x8x1; r22; s4096 | 60 | 4.480 | 5.344 | 1.193x | 95.5% | 0.120 | library_gemm (31) | library_gemm (1.871x; 31) |
| 323 | policy.gpool_to_bias_matmul | library_gemm | library_gemm; gemmSN_NN_kernel; g2x2x1; b256x1x1; r64; s21504 | 60 | 5.344 | 6.368 | 1.192x | 89.9% | 0.060 | head_elementwise (31) | head_elementwise (1.251x; 31) |
| 324 | policy.p1_half_to_float | copy_reformat | copy_reformat; copyFromHalfKernel; g880x1x1; b512x1x1; r16; s0 | 60 | 1.504 | 1.680 | 1.117x | 87.0% | 0.028 | library_gemm (38) | linear2_residual (1.830x; 14) |
| 325 | policy.gpool_bias_add | head_elementwise | head_elementwise; addNCBiasInplaceNHWCKernel; g1x73x13; b96x5x1; r16; s0 | 60 | 1.792 | 2.032 | 1.134x | 79.8% | 0.019 | library_gemm (34) | affine_silu (1.179x; 12) |
| 326 | policy.p1_norm_silu | head_elementwise | head_elementwise; applyCScaleBiasNHWCSiluKernel; g1x73x13; b96x5x1; r16; s0 | 60 | 2.176 | 2.464 | 1.132x | 79.4% | 0.034 | library_gemm (59) | library_gemm (1.132x; 59) |
| 327 | policy.p2_conv | library_gemm | library_gemm; Kernel2; g74x1x1; b128x1x1; r90; s98304 | 60 | 3.904 | 4.704 | 1.205x | 93.0% | 0.100 | library_gemm (45) | head_elementwise (1.369x; 15) |
| 328 | policy.gpool_to_pass_matmul | library_gemm | library_gemm; gemmSN_NN_kernel; g2x2x1; b256x1x1; r64; s21504 | 60 | 5.328 | 5.984 | 1.123x | 87.8% | 0.049 | library_gemm (47) | affine_silu (1.345x; 13) |
| 329 | policy.pass_bias_silu | head_elementwise | head_elementwise; addCBiasInplaceNCKernelSilu; g1x3x1; b96x5x1; r16; s0 | 60 | 1.024 | 1.152 | 1.125x | 90.2% | 0.030 | library_gemm (46) | library_gemm (1.141x; 46) |
| 330 | policy.gpool_to_pass_matmul2 | library_gemm | library_gemm; gemmSN_NN_kernel; g1x2x1; b256x1x1; r64; s21504 | 60 | 2.288 | 2.784 | 1.217x | 95.3% | 0.062 | library_gemm (58) | library_gemm (1.244x; 58) |
| 331 | value.v1_conv | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r118; s98304 | 60 | 7.968 | 9.777 | 1.227x | 81.8% | 0.117 | library_gemm (35) | library_gemm (1.341x; 35) |
| 332 | value.v1_norm_silu | head_elementwise | head_elementwise; applyCScaleBiasNHWCSiluHalfKernel; g1x181x13; b192x2x1; r16; s0 | 60 | 3.136 | 3.473 | 1.107x | 70.7% | 0.046 | library_gemm (30) | library_gemm (1.332x; 30) |
| 333 | value.v1_half_to_float | copy_reformat | copy_reformat; copyFromHalfKernel; g1760x1x1; b512x1x1; r16; s0 | 60 | 2.176 | 2.592 | 1.191x | 66.6% | 0.029 | head_elementwise (37) | head_elementwise (1.309x; 37) |
| 334 | value.v1_global_pool | head_elementwise | head_elementwise; valueHeadPoolChannelsNHWCKernel; g3x1x13; b64x8x1; r22; s2048 | 60 | 3.232 | 3.680 | 1.139x | 78.7% | 0.038 | library_gemm (17) | copy_reformat (1.505x; 15) |
| 335 | value.v2_matmul | library_gemm | library_gemm; gemmSN_NN_kernel; g3x2x1; b256x1x1; r64; s21504 | 60 | 9.504 | 10.096 | 1.062x | 88.6% | 0.081 | library_gemm (46) | cudnn (1.380x; 13) |
| 336 | value.v2_bias_silu | head_elementwise | head_elementwise; addCBiasInplaceNCKernelSilu; g1x7x1; b192x2x1; r16; s0 | 60 | 1.024 | 1.120 | 1.094x | 90.4% | 0.007 | library_gemm (31) | head_elementwise (1.156x; 15) |
| 337 | value.v3_matmul | library_gemm | library_gemm; gemmSN_NN_kernel; g1x2x1; b256x1x1; r64; s21504 | 60 | 3.488 | 3.824 | 1.096x | 80.8% | 0.025 | library_gemm (38) | library_gemm (1.138x; 38) |
| 338 | value.v3_bias | head_elementwise | head_elementwise; addCBiasInplaceNCKernel; g1x1x1; b3x170x1; r16; s0 | 60 | 0.960 | 1.072 | 1.117x | 91.8% | 0.008 | library_gemm (58) | library_gemm (1.133x; 58) |
| 339 | value.score_matmul | library_gemm | library_gemm; gemmSN_NN_kernel; g1x2x1; b256x1x1; r64; s21504 | 60 | 3.488 | 3.968 | 1.138x | 82.1% | 0.030 | library_gemm (42) | head_elementwise (1.202x; 17) |
| 340 | value.score_bias | head_elementwise | head_elementwise; addCBiasInplaceNCKernel; g1x1x1; b6x85x1; r16; s0 | 60 | 0.928 | 1.056 | 1.138x | 85.8% | 0.008 | library_gemm (45) | library_gemm (1.172x; 45) |
| 341 | value.ownership_conv | library_gemm | library_gemm; Kernel2; g8x19x3; b128x1x1; r118; s33792 | 60 | 4.032 | 4.608 | 1.143x | 59.4% | 0.038 | library_gemm (46) | head_elementwise (1.163x; 12) |
| 342 | value.ownership_conv_splitk_reduce | library_gemm | library_gemm; splitKreduce_kernel; g147x1x1; b32x16x1; r49; s0 | 60 | 1.376 | 1.536 | 1.116x | 85.0% | 0.023 | library_gemm (43) | library_gemm (1.186x; 43) |
| 343 | value.ownership_half_to_float | copy_reformat | copy_reformat; copyFromHalfKernel; g10x1x1; b512x1x1; r16; s0 | 60 | 0.928 | 1.216 | 1.310x | 69.6% | 0.038 | library_gemm (43) | idle (1.569x; 16) |
