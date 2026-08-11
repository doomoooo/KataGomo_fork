# Nsys stream interference report

- Timed iterations: 30; streams: 65, 82
- Kernels per forward: 65=344, 82=344
- Iteration start offset stream 82 - 65: median 3.76 us, p10..p90 3.45..5.00 us, range 3.33..35.23 us.

## Kernel families

| family | instances | total ms | median us | p90 us | peer overlap | S2/S1 median | summed excess ms | matched |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| fused_ffn | 1980 | 88.655 | 44.288 | 48.740 | 79.0% | 1.089x | 7.147 | 1980 |
| library_gemm | 4140 | 65.173 | 15.808 | 21.536 | 94.7% | 1.694x | 27.553 | 4140 |
| linear2_residual | 1980 | 64.618 | 32.672 | 35.233 | 98.2% | 1.579x | 23.723 | 1980 |
| wide_qkv | 1980 | 50.034 | 23.665 | 31.012 | 77.6% | 1.245x | 12.253 | 1980 |
| fa4 | 1980 | 29.553 | 14.784 | 15.552 | 46.0% | 1.252x | 6.220 | 1980 |
| rmsnorm | 3960 | 16.347 | 4.032 | 5.600 | 100.0% | 1.678x | 6.733 | 3960 |
| qk_rope | 1980 | 14.845 | 7.392 | 9.504 | 98.4% | 1.817x | 6.752 | 1980 |
| affine_silu | 1380 | 8.563 | 6.096 | 9.472 | 97.5% | 1.334x | 2.761 | 1380 |
| head_elementwise | 720 | 2.282 | 2.496 | 8.064 | 76.5% | 1.133x | 0.506 | 720 |
| cudnn | 180 | 1.497 | 1.728 | 21.799 | 22.4% | 1.111x | 0.161 | 180 |
| copy_reformat | 300 | 0.552 | 1.728 | 2.528 | 63.5% | 1.188x | 0.128 | 300 |
| sumChannelsNCHWKernel | 60 | 0.118 | 1.872 | 2.208 | 42.8% | 1.104x | 0.016 | 60 |

## Dominant interference pairs

| running family | dominant peer | instances | median overlap | median us | S2/S1 median | matched |
|---|---|---:|---:|---:|---:|---:|
| library_gemm | fused_ffn | 963 | 95.1% | 20.896 | 2.481x | 963 |
| qk_rope | wide_qkv | 990 | 100.0% | 9.280 | 2.271x | 990 |
| rmsnorm | fused_ffn | 990 | 100.0% | 5.504 | 2.260x | 990 |
| rmsnorm | wide_qkv | 990 | 100.0% | 4.928 | 2.040x | 990 |
| library_gemm | library_gemm | 1661 | 96.5% | 14.368 | 1.707x | 1661 |
| library_gemm | linear2_residual | 330 | 95.6% | 23.424 | 1.646x | 330 |
| linear2_residual | wide_qkv | 660 | 97.8% | 33.728 | 1.634x | 660 |
| library_gemm | cudnn | 38 | 93.5% | 4.208 | 1.616x | 38 |
| wide_qkv | linear2_residual | 660 | 96.4% | 30.384 | 1.585x | 660 |
| linear2_residual | library_gemm | 330 | 97.7% | 32.176 | 1.562x | 330 |
| linear2_residual | fused_ffn | 990 | 98.5% | 31.584 | 1.525x | 990 |
| copy_reformat | idle | 68 | 0.0% | 1.424 | 1.500x | 68 |
| library_gemm | idle | 5 | 0.0% | 2.015 | 1.464x | 5 |
| affine_silu | library_gemm | 1023 | 100.0% | 7.104 | 1.423x | 1023 |
| library_gemm | affine_silu | 580 | 94.7% | 18.944 | 1.419x | 580 |
| qk_rope | fa4 | 990 | 100.0% | 5.792 | 1.419x | 990 |
| fa4 | fa4 | 124 | 65.0% | 16.337 | 1.391x | 124 |
| rmsnorm | linear2_residual | 660 | 100.0% | 3.264 | 1.342x | 660 |
| wide_qkv | library_gemm | 329 | 66.4% | 25.344 | 1.332x | 329 |
| library_gemm | wide_qkv | 326 | 93.4% | 15.457 | 1.311x | 326 |
| head_elementwise | idle | 29 | 0.0% | 1.504 | 1.270x | 29 |
| affine_silu | linear2_residual | 330 | 100.0% | 4.160 | 1.257x | 330 |
| fa4 | library_gemm | 928 | 44.6% | 14.784 | 1.252x | 928 |
| fa4 | qk_rope | 928 | 42.4% | 14.688 | 1.246x | 928 |
| library_gemm | head_elementwise | 201 | 84.7% | 6.528 | 1.202x | 201 |
| copy_reformat | library_gemm | 147 | 74.3% | 1.824 | 1.191x | 147 |
| head_elementwise | copy_reformat | 43 | 79.3% | 3.808 | 1.190x | 43 |
| sumChannelsNCHWKernel | cudnn | 29 | 40.3% | 2.016 | 1.189x | 29 |
| affine_silu | head_elementwise | 19 | 22.1% | 5.856 | 1.188x | 19 |
| head_elementwise | head_elementwise | 57 | 94.3% | 2.560 | 1.182x | 57 |
| library_gemm | copy_reformat | 36 | 61.4% | 2.224 | 1.162x | 36 |
| wide_qkv | qk_rope | 990 | 64.9% | 21.985 | 1.153x | 990 |
| copy_reformat | head_elementwise | 66 | 80.1% | 2.256 | 1.149x | 66 |
| affine_silu | affine_silu | 7 | 36.4% | 5.632 | 1.143x | 7 |
| fused_ffn | fused_ffn | 1022 | 69.4% | 47.521 | 1.139x | 1022 |
| head_elementwise | library_gemm | 533 | 100.0% | 2.208 | 1.129x | 533 |
| rmsnorm | library_gemm | 1312 | 100.0% | 2.736 | 1.120x | 1312 |
| cudnn | cudnn | 103 | 65.3% | 1.728 | 1.117x | 103 |
| cudnn | idle | 10 | 0.0% | 1.712 | 1.115x | 10 |
| cudnn | library_gemm | 32 | 15.4% | 21.601 | 1.111x | 32 |

## Logical operation groups

Isolated reference total is the isolated median for each ordinal multiplied by its S2 call count; it is a normalized reference, not a second trace total.

| logical group | families | positions | calls | isolated reference ms | S2 total ms | total ratio | excess ms |
|---|---|---:|---:|---:|---:|---:|---:|
| transformer.ffn_linear2_residual | linear2_residual | 33 | 1980 | 40.895 | 64.618 | 1.580x | 23.723 |
| transformer.attention_out_projection_residual | library_gemm | 33 | 1980 | 16.664 | 34.892 | 2.094x | 18.228 |
| transformer.attention_qkv_projection | wide_qkv | 33 | 1980 | 37.780 | 50.034 | 1.324x | 12.253 |
| transformer.ffn_linear1_gate_swiglu | fused_ffn | 33 | 1980 | 81.608 | 88.655 | 1.086x | 7.147 |
| transformer.attention_qk_rope | qk_rope | 33 | 1980 | 8.093 | 14.845 | 1.834x | 6.752 |
| transformer.attention_fa4 | fa4 | 33 | 1980 | 23.333 | 29.553 | 1.267x | 6.220 |
| outer.post_projection_c384_to_c768_residual | library_gemm | 11 | 660 | 9.421 | 14.270 | 1.515x | 4.849 |
| outer.pre_projection_c768_to_c384 | library_gemm | 11 | 660 | 7.792 | 11.355 | 1.457x | 3.563 |
| transformer.attention_rmsnorm | rmsnorm | 33 | 1980 | 4.796 | 8.252 | 1.721x | 3.456 |
| transformer.ffn_rmsnorm | rmsnorm | 33 | 1980 | 4.817 | 8.095 | 1.680x | 3.278 |
| outer.pre_norm_silu | affine_silu | 11 | 660 | 3.306 | 5.386 | 1.629x | 2.080 |
| outer.post_norm_silu | affine_silu | 11 | 660 | 2.195 | 2.698 | 1.229x | 0.503 |
| policy.p1_conv | library_gemm | 1 | 60 | 0.369 | 0.615 | 1.668x | 0.246 |
| trunk.tip_norm_silu | affine_silu | 1 | 60 | 0.301 | 0.478 | 1.587x | 0.177 |
| policy.g1_global_pool | head_elementwise | 1 | 60 | 0.265 | 0.440 | 1.661x | 0.175 |
| frontend.initial_conv | cudnn | 1 | 60 | 1.167 | 1.300 | 1.114x | 0.133 |
| policy.g1_conv | library_gemm | 1 | 60 | 0.351 | 0.466 | 1.326x | 0.115 |
| value.v1_conv | library_gemm | 1 | 60 | 0.474 | 0.567 | 1.197x | 0.093 |
| policy.gpool_to_bias_matmul | library_gemm | 1 | 60 | 0.323 | 0.401 | 1.242x | 0.078 |
| value.v1_norm_silu | head_elementwise | 1 | 60 | 0.188 | 0.262 | 1.392x | 0.074 |
| frontend.initial_global_matmul | library_gemm | 1 | 60 | 0.157 | 0.231 | 1.467x | 0.073 |
| policy.p2_conv | library_gemm | 1 | 60 | 0.232 | 0.292 | 1.257x | 0.060 |
| value.v2_matmul | library_gemm | 1 | 60 | 0.568 | 0.626 | 1.102x | 0.058 |
| policy.gpool_to_pass_matmul | library_gemm | 1 | 60 | 0.315 | 0.372 | 1.182x | 0.057 |
| value.v1_half_to_float | copy_reformat | 1 | 60 | 0.129 | 0.183 | 1.426x | 0.055 |
| frontend.initial_global_broadcast_add | head_elementwise | 1 | 60 | 0.463 | 0.513 | 1.109x | 0.051 |
| policy.pass_bias_silu | head_elementwise | 1 | 60 | 0.060 | 0.110 | 1.847x | 0.050 |
| value.v1_global_pool | head_elementwise | 1 | 60 | 0.192 | 0.234 | 1.221x | 0.042 |
| policy.gpool_to_pass_matmul2 | library_gemm | 1 | 60 | 0.138 | 0.174 | 1.259x | 0.036 |
| policy.g1_norm_silu | head_elementwise | 1 | 60 | 0.127 | 0.160 | 1.264x | 0.034 |
| value.ownership_conv | library_gemm | 1 | 60 | 0.242 | 0.269 | 1.111x | 0.028 |
| policy.p1_norm_silu | head_elementwise | 1 | 60 | 0.129 | 0.156 | 1.212x | 0.027 |
| value.score_matmul | library_gemm | 1 | 60 | 0.209 | 0.235 | 1.122x | 0.025 |
| input.extract_mask | head_elementwise | 1 | 60 | 0.071 | 0.094 | 1.323x | 0.023 |
| policy.g1_half_to_float | copy_reformat | 1 | 60 | 0.092 | 0.114 | 1.232x | 0.021 |
| value.v3_matmul | library_gemm | 1 | 60 | 0.208 | 0.229 | 1.097x | 0.020 |
| value.ownership_half_to_float | copy_reformat | 1 | 60 | 0.056 | 0.075 | 1.345x | 0.019 |
| frontend.initial_conv_nhwc_padding_0 | cudnn | 1 | 60 | 0.077 | 0.095 | 1.241x | 0.019 |
| input.mask_half_to_float | copy_reformat | 1 | 60 | 0.058 | 0.075 | 1.310x | 0.018 |
| input.mask_sum | sumChannelsNCHWKernel | 1 | 60 | 0.102 | 0.118 | 1.161x | 0.016 |
| policy.gpool_bias_add | head_elementwise | 1 | 60 | 0.108 | 0.122 | 1.136x | 0.015 |
| policy.p1_half_to_float | copy_reformat | 1 | 60 | 0.090 | 0.105 | 1.162x | 0.015 |
| value.ownership_conv_splitk_reduce | library_gemm | 1 | 60 | 0.083 | 0.096 | 1.164x | 0.014 |
| frontend.initial_conv_nhwc_padding_1 | cudnn | 1 | 60 | 0.092 | 0.101 | 1.100x | 0.009 |
| frontend.initial_global_matmul_splitk_reduce | library_gemm | 1 | 60 | 0.077 | 0.083 | 1.080x | 0.008 |
| value.score_bias | head_elementwise | 1 | 60 | 0.056 | 0.062 | 1.116x | 0.006 |
| value.v3_bias | head_elementwise | 1 | 60 | 0.058 | 0.063 | 1.097x | 0.006 |
| value.v2_bias_silu | head_elementwise | 1 | 60 | 0.061 | 0.065 | 1.053x | 0.003 |

## `library_gemm` logical breakdown

| logical group | positions | calls | isolated reference ms | S2 total ms | total ratio | excess ms |
|---|---:|---:|---:|---:|---:|---:|
| transformer.attention_out_projection_residual | 33 | 1980 | 16.664 | 34.892 | 2.094x | 18.228 |
| outer.post_projection_c384_to_c768_residual | 11 | 660 | 9.421 | 14.270 | 1.515x | 4.849 |
| outer.pre_projection_c768_to_c384 | 11 | 660 | 7.792 | 11.355 | 1.457x | 3.563 |
| policy.p1_conv | 1 | 60 | 0.369 | 0.615 | 1.668x | 0.246 |
| policy.g1_conv | 1 | 60 | 0.351 | 0.466 | 1.326x | 0.115 |
| value.v1_conv | 1 | 60 | 0.474 | 0.567 | 1.197x | 0.093 |
| policy.gpool_to_bias_matmul | 1 | 60 | 0.323 | 0.401 | 1.242x | 0.078 |
| frontend.initial_global_matmul | 1 | 60 | 0.157 | 0.231 | 1.467x | 0.073 |
| policy.p2_conv | 1 | 60 | 0.232 | 0.292 | 1.257x | 0.060 |
| value.v2_matmul | 1 | 60 | 0.568 | 0.626 | 1.102x | 0.058 |
| policy.gpool_to_pass_matmul | 1 | 60 | 0.315 | 0.372 | 1.182x | 0.057 |
| policy.gpool_to_pass_matmul2 | 1 | 60 | 0.138 | 0.174 | 1.259x | 0.036 |
| value.ownership_conv | 1 | 60 | 0.242 | 0.269 | 1.111x | 0.028 |
| value.score_matmul | 1 | 60 | 0.209 | 0.235 | 1.122x | 0.025 |
| value.v3_matmul | 1 | 60 | 0.208 | 0.229 | 1.097x | 0.020 |
| value.ownership_conv_splitk_reduce | 1 | 60 | 0.083 | 0.096 | 1.164x | 0.014 |
| frontend.initial_global_matmul_splitk_reduce | 1 | 60 | 0.077 | 0.083 | 1.080x | 0.008 |

## Top ordinal hotspots by summed excess

The worst peer is the highest median S2/S1 slowdown among peer families observed at least four times for that ordinal.

| rank | ordinal | logical position | family | calls | isolated us | S2 us | S2/S1 | excess ms | common peer | worst peer |
|---:|---:|---|---|---:|---:|---:|---:|---:|---|---|
| 1 | 214 | outer_07.transformer_0.block_21.ffn_linear2_residual | linear2_residual | 60 | 20.816 | 35.920 | 1.726x | 0.909 | fused_ffn (30) | wide_qkv (1.779x; 30) |
| 2 | 222 | outer_07.transformer_1.block_22.ffn_linear2_residual | linear2_residual | 60 | 20.816 | 35.216 | 1.692x | 0.881 | fused_ffn (30) | wide_qkv (1.789x; 30) |
| 3 | 230 | outer_07.transformer_2.block_23.ffn_linear2_residual | linear2_residual | 60 | 20.832 | 35.104 | 1.685x | 0.871 | fused_ffn (30) | library_gemm (1.722x; 30) |
| 4 | 202 | outer_06.transformer_2.block_20.ffn_linear2_residual | linear2_residual | 60 | 20.768 | 34.240 | 1.649x | 0.839 | fused_ffn (30) | library_gemm (1.731x; 30) |
| 5 | 146 | outer_04.transformer_2.block_14.ffn_linear2_residual | linear2_residual | 60 | 20.768 | 33.904 | 1.633x | 0.820 | fused_ffn (30) | library_gemm (1.716x; 30) |
| 6 | 298 | outer_10.transformer_0.block_30.ffn_linear2_residual | linear2_residual | 60 | 20.720 | 34.081 | 1.645x | 0.807 | fused_ffn (30) | wide_qkv (1.688x; 30) |
| 7 | 138 | outer_04.transformer_1.block_13.ffn_linear2_residual | linear2_residual | 60 | 20.625 | 33.600 | 1.629x | 0.783 | fused_ffn (30) | wide_qkv (1.672x; 30) |
| 8 | 242 | outer_08.transformer_0.block_24.ffn_linear2_residual | linear2_residual | 60 | 20.881 | 33.840 | 1.621x | 0.779 | fused_ffn (30) | wide_qkv (1.645x; 30) |
| 9 | 306 | outer_10.transformer_1.block_31.ffn_linear2_residual | linear2_residual | 60 | 20.704 | 32.913 | 1.590x | 0.759 | fused_ffn (30) | wide_qkv (1.677x; 30) |
| 10 | 130 | outer_04.transformer_0.block_12.ffn_linear2_residual | linear2_residual | 60 | 20.704 | 32.864 | 1.587x | 0.752 | fused_ffn (30) | wide_qkv (1.684x; 30) |
| 11 | 194 | outer_06.transformer_1.block_19.ffn_linear2_residual | linear2_residual | 60 | 20.736 | 33.120 | 1.597x | 0.741 | fused_ffn (30) | fused_ffn (1.605x; 30) |
| 12 | 46 | outer_01.transformer_0.block_03.ffn_linear2_residual | linear2_residual | 60 | 20.704 | 32.448 | 1.567x | 0.734 | fused_ffn (30) | wide_qkv (1.685x; 30) |
| 13 | 186 | outer_06.transformer_0.block_18.ffn_linear2_residual | linear2_residual | 60 | 20.736 | 32.880 | 1.586x | 0.733 | fused_ffn (30) | wide_qkv (1.610x; 30) |
| 14 | 250 | outer_08.transformer_1.block_25.ffn_linear2_residual | linear2_residual | 60 | 20.784 | 32.736 | 1.575x | 0.731 | fused_ffn (30) | wide_qkv (1.643x; 30) |
| 15 | 74 | outer_02.transformer_0.block_06.ffn_linear2_residual | linear2_residual | 60 | 20.640 | 32.496 | 1.574x | 0.730 | fused_ffn (30) | wide_qkv (1.683x; 30) |
| 16 | 258 | outer_08.transformer_2.block_26.ffn_linear2_residual | linear2_residual | 60 | 20.752 | 32.608 | 1.571x | 0.712 | fused_ffn (30) | library_gemm (1.584x; 30) |
| 17 | 18 | outer_00.transformer_0.block_00.ffn_linear2_residual | linear2_residual | 60 | 20.832 | 32.144 | 1.543x | 0.711 | fused_ffn (30) | wide_qkv (1.634x; 30) |
| 18 | 158 | outer_05.transformer_0.block_15.ffn_linear2_residual | linear2_residual | 60 | 20.753 | 32.576 | 1.570x | 0.709 | fused_ffn (30) | wide_qkv (1.593x; 30) |
| 19 | 82 | outer_02.transformer_1.block_07.ffn_linear2_residual | linear2_residual | 60 | 20.608 | 32.256 | 1.565x | 0.708 | fused_ffn (30) | wide_qkv (1.632x; 30) |
| 20 | 314 | outer_10.transformer_2.block_32.ffn_linear2_residual | linear2_residual | 60 | 20.752 | 32.689 | 1.575x | 0.701 | fused_ffn (30) | fused_ffn (1.584x; 30) |
| 21 | 166 | outer_05.transformer_1.block_16.ffn_linear2_residual | linear2_residual | 60 | 20.544 | 32.096 | 1.562x | 0.691 | fused_ffn (30) | wide_qkv (1.621x; 30) |
| 22 | 54 | outer_01.transformer_1.block_04.ffn_linear2_residual | linear2_residual | 60 | 20.480 | 31.312 | 1.529x | 0.679 | fused_ffn (30) | wide_qkv (1.617x; 30) |
| 23 | 270 | outer_09.transformer_0.block_27.ffn_linear2_residual | linear2_residual | 60 | 20.625 | 31.248 | 1.515x | 0.659 | fused_ffn (30) | wide_qkv (1.611x; 30) |
| 24 | 90 | outer_02.transformer_2.block_08.ffn_linear2_residual | linear2_residual | 60 | 20.640 | 31.648 | 1.533x | 0.658 | fused_ffn (30) | library_gemm (1.548x; 30) |
| 25 | 102 | outer_03.transformer_0.block_09.ffn_linear2_residual | linear2_residual | 60 | 20.433 | 31.360 | 1.535x | 0.654 | fused_ffn (30) | wide_qkv (1.617x; 30) |
| 26 | 26 | outer_00.transformer_1.block_01.ffn_linear2_residual | linear2_residual | 60 | 20.704 | 31.280 | 1.511x | 0.653 | fused_ffn (30) | wide_qkv (1.599x; 30) |
| 27 | 110 | outer_03.transformer_1.block_10.ffn_linear2_residual | linear2_residual | 60 | 20.384 | 30.544 | 1.498x | 0.650 | fused_ffn (30) | wide_qkv (1.612x; 30) |
| 28 | 278 | outer_09.transformer_1.block_28.ffn_linear2_residual | linear2_residual | 60 | 20.480 | 30.672 | 1.498x | 0.648 | fused_ffn (30) | wide_qkv (1.602x; 30) |
| 29 | 62 | outer_01.transformer_2.block_05.ffn_linear2_residual | linear2_residual | 60 | 20.512 | 30.928 | 1.508x | 0.629 | fused_ffn (30) | library_gemm (1.542x; 30) |
| 30 | 34 | outer_00.transformer_2.block_02.ffn_linear2_residual | linear2_residual | 60 | 20.545 | 30.480 | 1.484x | 0.608 | fused_ffn (30) | library_gemm (1.509x; 30) |
| 31 | 174 | outer_05.transformer_2.block_17.ffn_linear2_residual | linear2_residual | 60 | 20.512 | 30.624 | 1.493x | 0.607 | fused_ffn (30) | library_gemm (1.518x; 30) |
| 32 | 286 | outer_09.transformer_2.block_29.ffn_linear2_residual | linear2_residual | 60 | 20.417 | 30.144 | 1.476x | 0.595 | fused_ffn (30) | library_gemm (1.502x; 30) |
| 33 | 239 | outer_08.transformer_0.block_24.attention_out_projection_residual | library_gemm | 60 | 8.544 | 18.241 | 2.135x | 0.585 | library_gemm (31) | fused_ffn (2.494x; 29) |
| 34 | 118 | outer_03.transformer_2.block_11.ffn_linear2_residual | linear2_residual | 60 | 20.369 | 29.888 | 1.467x | 0.581 | fused_ffn (30) | library_gemm (1.506x; 30) |
| 35 | 99 | outer_03.transformer_0.block_09.attention_out_projection_residual | library_gemm | 60 | 8.432 | 17.968 | 2.131x | 0.579 | fused_ffn (30) | fused_ffn (2.516x; 30) |
| 36 | 183 | outer_06.transformer_0.block_18.attention_out_projection_residual | library_gemm | 60 | 8.384 | 17.504 | 2.088x | 0.575 | fused_ffn (30) | fused_ffn (2.498x; 30) |
| 37 | 211 | outer_07.transformer_0.block_21.attention_out_projection_residual | library_gemm | 60 | 8.448 | 17.936 | 2.123x | 0.575 | library_gemm (32) | fused_ffn (2.496x; 28) |
| 38 | 155 | outer_05.transformer_0.block_15.attention_out_projection_residual | library_gemm | 60 | 8.448 | 17.776 | 2.104x | 0.570 | library_gemm (31) | fused_ffn (2.477x; 29) |
| 39 | 71 | outer_02.transformer_0.block_06.attention_out_projection_residual | library_gemm | 60 | 8.384 | 17.552 | 2.094x | 0.568 | fused_ffn (30) | fused_ffn (2.496x; 30) |
| 40 | 267 | outer_09.transformer_0.block_27.attention_out_projection_residual | library_gemm | 60 | 8.416 | 17.601 | 2.091x | 0.568 | library_gemm (31) | fused_ffn (2.479x; 29) |

## Full fixed-forward ordinal map

| ordinal | logical position | family | resource signature | calls | isolated us | S2 us | S2/S1 | overlap | excess ms | common peer | worst peer |
|---:|---|---|---|---:|---:|---:|---:|---:|---:|---|---|
| 0 | input.extract_mask | head_elementwise | head_elementwise; extractChannel0KernelNHWC; g10x1x1; b512x1x1; r16; s0 | 60 | 1.184 | 1.312 | 1.108x | 29.0% | 0.023 | idle (29) | idle (1.270x; 29) |
| 1 | input.mask_half_to_float | copy_reformat | copy_reformat; copyFromHalfKernel; g10x1x1; b512x1x1; r16; s0 | 60 | 0.960 | 1.088 | 1.133x | 13.8% | 0.018 | idle (41) | idle (1.467x; 41) |
| 2 | input.mask_sum | sumChannelsNCHWKernel | sumChannelsNCHWKernel; sumChannelsNCHWKernel; g1x1x13; b256x2x1; r22; s2048 | 60 | 1.696 | 1.872 | 1.104x | 42.8% | 0.016 | cudnn (29) | cudnn (1.189x; 29) |
| 3 | frontend.initial_conv_nhwc_padding_0 | cudnn | cudnn; nhwcAddPaddingKernel; g340x1x1; b768x1x1; r34; s0 | 60 | 1.280 | 1.520 | 1.188x | 67.0% | 0.019 | cudnn (28) | cudnn (1.325x; 28) |
| 4 | frontend.initial_conv_nhwc_padding_1 | cudnn | cudnn; nhwcAddPaddingKernel; g340x1x1; b768x1x1; r34; s0 | 60 | 1.536 | 1.664 | 1.083x | 59.2% | 0.009 | cudnn (46) | idle (1.125x; 7) |
| 5 | frontend.initial_conv | cudnn | cudnn; Kernel; g296x3x1; b128x1x1; r94; s81920 | 60 | 19.456 | 21.633 | 1.112x | 16.3% | 0.133 | cudnn (29) | cudnn (1.117x; 29) |
| 6 | frontend.initial_global_matmul | library_gemm | library_gemm; Kernel2; g8x1x3; b128x1x1; r128; s24576 | 60 | 2.624 | 3.744 | 1.427x | 88.1% | 0.073 | cudnn (30) | cudnn (1.640x; 30) |
| 7 | frontend.initial_global_matmul_splitk_reduce | library_gemm | library_gemm; splitKreduce_kernel; g24x1x1; b32x16x1; r49; s0 | 60 | 1.280 | 1.280 | 1.000x | 84.3% | 0.008 | library_gemm (23) | cudnn (1.137x; 8) |
| 8 | frontend.initial_global_broadcast_add | head_elementwise | head_elementwise; addNCBiasInplaceNHWCHalfKernel; g3x361x13; b256x1x1; r16; s0 | 60 | 7.712 | 8.096 | 1.050x | 35.3% | 0.051 | library_gemm (29) | library_gemm (1.058x; 29) |
| 9 | outer_00.pre_norm_silu | affine_silu | affine_silu; affineSiluHalf2Kernel; g4693x1x1; b384x1x1; r17; s0 | 60 | 4.928 | 6.096 | 1.237x | 61.5% | 0.093 | library_gemm (33) | library_gemm (1.519x; 33) |
| 10 | outer_00.pre_projection_c768_to_c384 | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 60 | 11.601 | 14.656 | 1.263x | 94.3% | 0.258 | wide_qkv (26) | library_gemm (1.668x; 8) |
| 11 | outer_00.transformer_0.block_00.attention_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | 2.368 | 3.728 | 1.574x | 98.4% | 0.093 | wide_qkv (30) | wide_qkv (2.020x; 30) |
| 12 | outer_00.transformer_0.block_00.attention_qkv_projection | wide_qkv | wide_qkv; wide_qkv_kernel; g9x37x1; b128x1x1; r136; s65536 | 60 | 18.784 | 22.336 | 1.189x | 69.1% | 0.255 | library_gemm (30) | library_gemm (1.285x; 30) |
| 13 | outer_00.transformer_0.block_00.attention_qk_rope | qk_rope | qk_rope; fusedQKRoPE19HalfKernel; g361x13x1; b192x1x1; r16; s0 | 60 | 4.064 | 7.232 | 1.780x | 97.3% | 0.188 | fa4 (30) | wide_qkv (2.193x; 30) |
| 14 | outer_00.transformer_0.block_00.attention_fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 60 | 11.680 | 14.464 | 1.238x | 49.5% | 0.174 | library_gemm (27) | fa4 (1.378x; 6) |
| 15 | outer_00.transformer_0.block_00.attention_out_projection_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 60 | 8.448 | 16.320 | 1.932x | 95.8% | 0.531 | library_gemm (31) | fused_ffn (2.439x; 29) |
| 16 | outer_00.transformer_0.block_00.ffn_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | 2.432 | 3.840 | 1.579x | 100.0% | 0.093 | fused_ffn (30) | fused_ffn (2.171x; 30) |
| 17 | outer_00.transformer_0.block_00.ffn_linear1_gate_swiglu | fused_ffn | fused_ffn; fused_ffn_kernel; g18x37x1; b128x1x1; r146; s32768 | 60 | 40.992 | 44.608 | 1.088x | 77.8% | 0.213 | fused_ffn (30) | fused_ffn (1.134x; 30) |
| 18 | outer_00.transformer_0.block_00.ffn_linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 60 | 20.832 | 32.144 | 1.543x | 98.1% | 0.711 | fused_ffn (30) | wide_qkv (1.634x; 30) |
| 19 | outer_00.transformer_1.block_01.attention_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | 2.464 | 3.728 | 1.513x | 100.0% | 0.092 | linear2_residual (30) | wide_qkv (1.909x; 30) |
| 20 | outer_00.transformer_1.block_01.attention_qkv_projection | wide_qkv | wide_qkv; wide_qkv_kernel; g9x37x1; b128x1x1; r136; s65536 | 60 | 19.312 | 25.584 | 1.325x | 83.3% | 0.410 | linear2_residual (30) | linear2_residual (1.573x; 30) |
| 21 | outer_00.transformer_1.block_01.attention_qk_rope | qk_rope | qk_rope; fusedQKRoPE19HalfKernel; g361x13x1; b192x1x1; r16; s0 | 60 | 4.128 | 7.441 | 1.802x | 99.2% | 0.200 | fa4 (30) | wide_qkv (2.209x; 30) |
| 22 | outer_00.transformer_1.block_01.attention_fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 60 | 11.904 | 14.624 | 1.229x | 44.3% | 0.169 | library_gemm (29) | library_gemm (1.234x; 29) |
| 23 | outer_00.transformer_1.block_01.attention_out_projection_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 60 | 8.480 | 15.920 | 1.877x | 95.8% | 0.546 | library_gemm (31) | fused_ffn (2.472x; 29) |
| 24 | outer_00.transformer_1.block_01.ffn_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | 2.432 | 4.000 | 1.645x | 100.0% | 0.104 | fused_ffn (30) | fused_ffn (2.296x; 30) |
| 25 | outer_00.transformer_1.block_01.ffn_linear1_gate_swiglu | fused_ffn | fused_ffn; fused_ffn_kernel; g18x37x1; b128x1x1; r146; s32768 | 60 | 42.672 | 45.089 | 1.057x | 76.4% | 0.152 | fused_ffn (31) | fused_ffn (1.116x; 31) |
| 26 | outer_00.transformer_1.block_01.ffn_linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 60 | 20.704 | 31.280 | 1.511x | 98.3% | 0.653 | fused_ffn (30) | wide_qkv (1.599x; 30) |
| 27 | outer_00.transformer_2.block_02.attention_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | 2.400 | 3.648 | 1.520x | 100.0% | 0.086 | linear2_residual (30) | wide_qkv (1.880x; 30) |
| 28 | outer_00.transformer_2.block_02.attention_qkv_projection | wide_qkv | wide_qkv; wide_qkv_kernel; g9x37x1; b128x1x1; r136; s65536 | 60 | 19.040 | 24.768 | 1.301x | 83.1% | 0.352 | linear2_residual (30) | linear2_residual (1.492x; 30) |
| 29 | outer_00.transformer_2.block_02.attention_qk_rope | qk_rope | qk_rope; fusedQKRoPE19HalfKernel; g361x13x1; b192x1x1; r16; s0 | 60 | 4.080 | 7.392 | 1.812x | 96.6% | 0.198 | fa4 (30) | wide_qkv (2.259x; 30) |
| 30 | outer_00.transformer_2.block_02.attention_fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 60 | 11.745 | 14.464 | 1.232x | 46.9% | 0.179 | library_gemm (26) | fa4 (1.400x; 8) |
| 31 | outer_00.transformer_2.block_02.attention_out_projection_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 60 | 8.416 | 15.920 | 1.892x | 95.8% | 0.528 | library_gemm (32) | fused_ffn (2.470x; 28) |
| 32 | outer_00.transformer_2.block_02.ffn_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | 2.464 | 4.048 | 1.643x | 100.0% | 0.099 | fused_ffn (30) | fused_ffn (2.260x; 30) |
| 33 | outer_00.transformer_2.block_02.ffn_linear1_gate_swiglu | fused_ffn | fused_ffn; fused_ffn_kernel; g18x37x1; b128x1x1; r146; s32768 | 60 | 42.337 | 45.056 | 1.064x | 76.3% | 0.183 | fused_ffn (34) | fused_ffn (1.130x; 34) |
| 34 | outer_00.transformer_2.block_02.ffn_linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 60 | 20.545 | 30.480 | 1.484x | 98.2% | 0.608 | fused_ffn (30) | library_gemm (1.509x; 30) |
| 35 | outer_00.post_norm_silu | affine_silu | affine_silu; affineSiluHalf2Kernel; g4693x1x1; b192x1x1; r17; s0 | 60 | 3.296 | 3.984 | 1.209x | 100.0% | 0.040 | library_gemm (30) | linear2_residual (1.233x; 30) |
| 36 | outer_00.post_projection_c384_to_c768_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b256x1x1; r154; s73728 | 60 | 14.177 | 19.600 | 1.383x | 95.2% | 0.381 | affine_silu (30) | linear2_residual (1.593x; 30) |
| 37 | outer_01.pre_norm_silu | affine_silu | affine_silu; affineSiluHalf2Kernel; g4693x1x1; b384x1x1; r17; s0 | 60 | 5.056 | 8.080 | 1.598x | 100.0% | 0.190 | library_gemm (60) | library_gemm (1.598x; 60) |
| 38 | outer_01.pre_projection_c768_to_c384 | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 60 | 11.840 | 16.368 | 1.382x | 94.0% | 0.294 | wide_qkv (30) | library_gemm (1.662x; 5) |
| 39 | outer_01.transformer_0.block_03.attention_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | 2.400 | 4.369 | 1.820x | 100.0% | 0.122 | library_gemm (30) | wide_qkv (2.180x; 30) |
| 40 | outer_01.transformer_0.block_03.attention_qkv_projection | wide_qkv | wide_qkv; wide_qkv_kernel; g9x37x1; b128x1x1; r136; s65536 | 60 | 18.945 | 23.233 | 1.226x | 65.5% | 0.259 | library_gemm (30) | library_gemm (1.303x; 30) |
| 41 | outer_01.transformer_0.block_03.attention_qk_rope | qk_rope | qk_rope; fusedQKRoPE19HalfKernel; g361x13x1; b192x1x1; r16; s0 | 60 | 4.032 | 7.568 | 1.877x | 99.2% | 0.213 | fa4 (30) | wide_qkv (2.306x; 30) |
| 42 | outer_01.transformer_0.block_03.attention_fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 60 | 11.761 | 14.720 | 1.252x | 44.7% | 0.182 | library_gemm (29) | library_gemm (1.254x; 29) |
| 43 | outer_01.transformer_0.block_03.attention_out_projection_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 60 | 8.416 | 16.944 | 2.013x | 95.9% | 0.564 | fused_ffn (30) | fused_ffn (2.500x; 30) |
| 44 | outer_01.transformer_0.block_03.ffn_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | 2.432 | 4.096 | 1.684x | 100.0% | 0.104 | fused_ffn (30) | fused_ffn (2.250x; 30) |
| 45 | outer_01.transformer_0.block_03.ffn_linear1_gate_swiglu | fused_ffn | fused_ffn; fused_ffn_kernel; g18x37x1; b128x1x1; r146; s32768 | 60 | 41.409 | 45.184 | 1.091x | 77.6% | 0.195 | fused_ffn (30) | fused_ffn (1.128x; 30) |
| 46 | outer_01.transformer_0.block_03.ffn_linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 60 | 20.704 | 32.448 | 1.567x | 98.1% | 0.734 | fused_ffn (30) | wide_qkv (1.685x; 30) |
| 47 | outer_01.transformer_1.block_04.attention_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | 2.432 | 3.744 | 1.539x | 100.0% | 0.093 | linear2_residual (30) | wide_qkv (1.961x; 30) |
| 48 | outer_01.transformer_1.block_04.attention_qkv_projection | wide_qkv | wide_qkv; wide_qkv_kernel; g9x37x1; b128x1x1; r136; s65536 | 60 | 19.104 | 25.648 | 1.343x | 83.2% | 0.426 | linear2_residual (30) | linear2_residual (1.600x; 30) |
| 49 | outer_01.transformer_1.block_04.attention_qk_rope | qk_rope | qk_rope; fusedQKRoPE19HalfKernel; g361x13x1; b192x1x1; r16; s0 | 60 | 4.112 | 7.360 | 1.790x | 100.0% | 0.199 | fa4 (30) | wide_qkv (2.210x; 30) |
| 50 | outer_01.transformer_1.block_04.attention_fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 60 | 11.777 | 14.720 | 1.250x | 45.4% | 0.179 | library_gemm (30) | qk_rope (1.253x; 30) |
| 51 | outer_01.transformer_1.block_04.attention_out_projection_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 60 | 8.448 | 17.136 | 2.028x | 95.8% | 0.550 | fused_ffn (30) | fused_ffn (2.462x; 30) |
| 52 | outer_01.transformer_1.block_04.ffn_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | 2.432 | 4.112 | 1.691x | 100.0% | 0.104 | fused_ffn (30) | fused_ffn (2.296x; 30) |
| 53 | outer_01.transformer_1.block_04.ffn_linear1_gate_swiglu | fused_ffn | fused_ffn; fused_ffn_kernel; g18x37x1; b128x1x1; r146; s32768 | 60 | 42.449 | 45.584 | 1.074x | 76.4% | 0.185 | fused_ffn (31) | fused_ffn (1.139x; 31) |
| 54 | outer_01.transformer_1.block_04.ffn_linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 60 | 20.480 | 31.312 | 1.529x | 98.4% | 0.679 | fused_ffn (30) | wide_qkv (1.617x; 30) |
| 55 | outer_01.transformer_2.block_05.attention_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | 2.401 | 4.016 | 1.673x | 100.0% | 0.093 | linear2_residual (30) | wide_qkv (1.899x; 30) |
| 56 | outer_01.transformer_2.block_05.attention_qkv_projection | wide_qkv | wide_qkv; wide_qkv_kernel; g9x37x1; b128x1x1; r136; s65536 | 60 | 18.992 | 24.881 | 1.310x | 82.9% | 0.365 | linear2_residual (30) | linear2_residual (1.494x; 30) |
| 57 | outer_01.transformer_2.block_05.attention_qk_rope | qk_rope | qk_rope; fusedQKRoPE19HalfKernel; g361x13x1; b192x1x1; r16; s0 | 60 | 4.096 | 7.504 | 1.832x | 94.9% | 0.201 | fa4 (30) | wide_qkv (2.273x; 30) |
| 58 | outer_01.transformer_2.block_05.attention_fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 60 | 11.728 | 14.592 | 1.244x | 47.9% | 0.193 | library_gemm (24) | fa4 (1.383x; 12) |
| 59 | outer_01.transformer_2.block_05.attention_out_projection_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 60 | 8.400 | 15.936 | 1.897x | 95.7% | 0.529 | library_gemm (32) | fused_ffn (2.478x; 28) |
| 60 | outer_01.transformer_2.block_05.ffn_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | 2.400 | 4.240 | 1.767x | 100.0% | 0.110 | fused_ffn (30) | fused_ffn (2.387x; 30) |
| 61 | outer_01.transformer_2.block_05.ffn_linear1_gate_swiglu | fused_ffn | fused_ffn; fused_ffn_kernel; g18x37x1; b128x1x1; r146; s32768 | 60 | 42.065 | 45.825 | 1.089x | 76.0% | 0.209 | fused_ffn (32) | fused_ffn (1.143x; 32) |
| 62 | outer_01.transformer_2.block_05.ffn_linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 60 | 20.512 | 30.928 | 1.508x | 98.2% | 0.629 | fused_ffn (30) | library_gemm (1.542x; 30) |
| 63 | outer_01.post_norm_silu | affine_silu | affine_silu; affineSiluHalf2Kernel; g4693x1x1; b192x1x1; r17; s0 | 60 | 3.296 | 4.032 | 1.223x | 100.0% | 0.042 | library_gemm (30) | linear2_residual (1.243x; 30) |
| 64 | outer_01.post_projection_c384_to_c768_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b256x1x1; r154; s73728 | 60 | 14.208 | 20.497 | 1.443x | 95.2% | 0.400 | affine_silu (30) | linear2_residual (1.581x; 30) |
| 65 | outer_02.pre_norm_silu | affine_silu | affine_silu; affineSiluHalf2Kernel; g4693x1x1; b384x1x1; r17; s0 | 60 | 4.992 | 8.048 | 1.612x | 100.0% | 0.192 | library_gemm (60) | library_gemm (1.612x; 60) |
| 66 | outer_02.pre_projection_c768_to_c384 | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 60 | 11.745 | 16.416 | 1.398x | 94.1% | 0.319 | wide_qkv (30) | library_gemm (1.646x; 13) |
| 67 | outer_02.transformer_0.block_06.attention_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | 2.400 | 4.416 | 1.840x | 100.0% | 0.121 | library_gemm (30) | wide_qkv (2.127x; 30) |
| 68 | outer_02.transformer_0.block_06.attention_qkv_projection | wide_qkv | wide_qkv; wide_qkv_kernel; g9x37x1; b128x1x1; r136; s65536 | 60 | 18.976 | 23.456 | 1.236x | 65.7% | 0.277 | library_gemm (30) | library_gemm (1.327x; 30) |
| 69 | outer_02.transformer_0.block_06.attention_qk_rope | qk_rope | qk_rope; fusedQKRoPE19HalfKernel; g361x13x1; b192x1x1; r16; s0 | 60 | 4.096 | 7.648 | 1.867x | 100.0% | 0.203 | fa4 (30) | wide_qkv (2.258x; 30) |
| 70 | outer_02.transformer_0.block_06.attention_fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 60 | 11.744 | 15.008 | 1.278x | 45.8% | 0.204 | library_gemm (30) | qk_rope (1.286x; 30) |
| 71 | outer_02.transformer_0.block_06.attention_out_projection_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 60 | 8.384 | 17.552 | 2.094x | 95.9% | 0.568 | fused_ffn (30) | fused_ffn (2.496x; 30) |
| 72 | outer_02.transformer_0.block_06.ffn_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | 2.432 | 4.016 | 1.651x | 100.0% | 0.099 | fused_ffn (30) | fused_ffn (2.197x; 30) |
| 73 | outer_02.transformer_0.block_06.ffn_linear1_gate_swiglu | fused_ffn | fused_ffn; fused_ffn_kernel; g18x37x1; b128x1x1; r146; s32768 | 60 | 42.017 | 46.977 | 1.118x | 76.2% | 0.257 | fused_ffn (30) | fused_ffn (1.163x; 30) |
| 74 | outer_02.transformer_0.block_06.ffn_linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 60 | 20.640 | 32.496 | 1.574x | 98.1% | 0.730 | fused_ffn (30) | wide_qkv (1.683x; 30) |
| 75 | outer_02.transformer_1.block_07.attention_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | 2.432 | 4.016 | 1.651x | 100.0% | 0.092 | linear2_residual (30) | wide_qkv (1.888x; 30) |
| 76 | outer_02.transformer_1.block_07.attention_qkv_projection | wide_qkv | wide_qkv; wide_qkv_kernel; g9x37x1; b128x1x1; r136; s65536 | 60 | 19.104 | 25.456 | 1.333x | 83.3% | 0.395 | linear2_residual (30) | linear2_residual (1.544x; 30) |
| 77 | outer_02.transformer_1.block_07.attention_qk_rope | qk_rope | qk_rope; fusedQKRoPE19HalfKernel; g361x13x1; b192x1x1; r16; s0 | 60 | 4.096 | 7.489 | 1.828x | 99.2% | 0.206 | fa4 (30) | wide_qkv (2.277x; 30) |
| 78 | outer_02.transformer_1.block_07.attention_fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 60 | 11.760 | 14.656 | 1.246x | 44.8% | 0.180 | library_gemm (29) | library_gemm (1.252x; 29) |
| 79 | outer_02.transformer_1.block_07.attention_out_projection_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 60 | 8.416 | 16.768 | 1.992x | 95.8% | 0.547 | fused_ffn (30) | fused_ffn (2.458x; 30) |
| 80 | outer_02.transformer_1.block_07.ffn_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | 2.432 | 3.888 | 1.599x | 100.0% | 0.095 | fused_ffn (30) | fused_ffn (2.184x; 30) |
| 81 | outer_02.transformer_1.block_07.ffn_linear1_gate_swiglu | fused_ffn | fused_ffn; fused_ffn_kernel; g18x37x1; b128x1x1; r146; s32768 | 60 | 42.673 | 46.480 | 1.089x | 78.5% | 0.238 | fused_ffn (31) | fused_ffn (1.147x; 31) |
| 82 | outer_02.transformer_1.block_07.ffn_linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 60 | 20.608 | 32.256 | 1.565x | 98.4% | 0.708 | fused_ffn (30) | wide_qkv (1.632x; 30) |
| 83 | outer_02.transformer_2.block_08.attention_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | 2.432 | 3.760 | 1.546x | 100.0% | 0.090 | linear2_residual (30) | wide_qkv (1.895x; 30) |
| 84 | outer_02.transformer_2.block_08.attention_qkv_projection | wide_qkv | wide_qkv; wide_qkv_kernel; g9x37x1; b128x1x1; r136; s65536 | 60 | 19.040 | 25.776 | 1.354x | 83.3% | 0.426 | linear2_residual (30) | linear2_residual (1.592x; 30) |
| 85 | outer_02.transformer_2.block_08.attention_qk_rope | qk_rope | qk_rope; fusedQKRoPE19HalfKernel; g361x13x1; b192x1x1; r16; s0 | 60 | 4.064 | 7.456 | 1.835x | 100.0% | 0.208 | fa4 (30) | wide_qkv (2.272x; 30) |
| 86 | outer_02.transformer_2.block_08.attention_fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 60 | 11.776 | 14.689 | 1.247x | 43.4% | 0.177 | library_gemm (30) | library_gemm (1.255x; 30) |
| 87 | outer_02.transformer_2.block_08.attention_out_projection_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 60 | 8.464 | 17.536 | 2.072x | 95.8% | 0.552 | fused_ffn (30) | fused_ffn (2.454x; 30) |
| 88 | outer_02.transformer_2.block_08.ffn_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | 2.432 | 4.016 | 1.651x | 100.0% | 0.100 | fused_ffn (30) | fused_ffn (2.237x; 30) |
| 89 | outer_02.transformer_2.block_08.ffn_linear1_gate_swiglu | fused_ffn | fused_ffn; fused_ffn_kernel; g18x37x1; b128x1x1; r146; s32768 | 60 | 42.705 | 46.065 | 1.079x | 76.7% | 0.188 | fused_ffn (30) | fused_ffn (1.130x; 30) |
| 90 | outer_02.transformer_2.block_08.ffn_linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 60 | 20.640 | 31.648 | 1.533x | 98.2% | 0.658 | fused_ffn (30) | library_gemm (1.548x; 30) |
| 91 | outer_02.post_norm_silu | affine_silu | affine_silu; affineSiluHalf2Kernel; g4693x1x1; b192x1x1; r17; s0 | 60 | 3.328 | 4.064 | 1.221x | 100.0% | 0.042 | library_gemm (30) | linear2_residual (1.250x; 30) |
| 92 | outer_02.post_projection_c384_to_c768_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b256x1x1; r154; s73728 | 60 | 14.273 | 20.689 | 1.450x | 95.1% | 0.423 | affine_silu (30) | linear2_residual (1.648x; 30) |
| 93 | outer_03.pre_norm_silu | affine_silu | affine_silu; affineSiluHalf2Kernel; g4693x1x1; b384x1x1; r17; s0 | 60 | 4.992 | 8.144 | 1.631x | 100.0% | 0.194 | library_gemm (60) | library_gemm (1.631x; 60) |
| 94 | outer_03.pre_projection_c768_to_c384 | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 60 | 11.809 | 17.040 | 1.443x | 94.1% | 0.322 | wide_qkv (30) | affine_silu (1.593x; 29) |
| 95 | outer_03.transformer_0.block_09.attention_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | 2.400 | 4.416 | 1.840x | 100.0% | 0.125 | library_gemm (30) | wide_qkv (2.260x; 30) |
| 96 | outer_03.transformer_0.block_09.attention_qkv_projection | wide_qkv | wide_qkv; wide_qkv_kernel; g9x37x1; b128x1x1; r136; s65536 | 60 | 19.105 | 23.872 | 1.250x | 65.5% | 0.288 | library_gemm (30) | library_gemm (1.338x; 30) |
| 97 | outer_03.transformer_0.block_09.attention_qk_rope | qk_rope | qk_rope; fusedQKRoPE19HalfKernel; g361x13x1; b192x1x1; r16; s0 | 60 | 4.032 | 7.552 | 1.873x | 100.0% | 0.211 | fa4 (30) | wide_qkv (2.294x; 30) |
| 98 | outer_03.transformer_0.block_09.attention_fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 60 | 11.840 | 15.216 | 1.285x | 46.1% | 0.206 | library_gemm (30) | qk_rope (1.288x; 30) |
| 99 | outer_03.transformer_0.block_09.attention_out_projection_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 60 | 8.432 | 17.968 | 2.131x | 95.9% | 0.579 | fused_ffn (30) | fused_ffn (2.516x; 30) |
| 100 | outer_03.transformer_0.block_09.ffn_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | 2.432 | 4.160 | 1.711x | 100.0% | 0.107 | fused_ffn (30) | fused_ffn (2.329x; 30) |
| 101 | outer_03.transformer_0.block_09.ffn_linear1_gate_swiglu | fused_ffn | fused_ffn; fused_ffn_kernel; g18x37x1; b128x1x1; r146; s32768 | 60 | 42.369 | 45.409 | 1.072x | 75.7% | 0.184 | fused_ffn (36) | fused_ffn (1.141x; 36) |
| 102 | outer_03.transformer_0.block_09.ffn_linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 60 | 20.433 | 31.360 | 1.535x | 98.0% | 0.654 | fused_ffn (30) | wide_qkv (1.617x; 30) |
| 103 | outer_03.transformer_1.block_10.attention_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | 2.432 | 4.016 | 1.651x | 100.0% | 0.090 | linear2_residual (30) | wide_qkv (1.842x; 30) |
| 104 | outer_03.transformer_1.block_10.attention_qkv_projection | wide_qkv | wide_qkv; wide_qkv_kernel; g9x37x1; b128x1x1; r136; s65536 | 60 | 18.896 | 24.401 | 1.291x | 83.2% | 0.346 | linear2_residual (30) | linear2_residual (1.475x; 30) |
| 105 | outer_03.transformer_1.block_10.attention_qk_rope | qk_rope | qk_rope; fusedQKRoPE19HalfKernel; g361x13x1; b192x1x1; r16; s0 | 60 | 4.064 | 7.296 | 1.795x | 97.5% | 0.197 | fa4 (30) | wide_qkv (2.236x; 30) |
| 106 | outer_03.transformer_1.block_10.attention_fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 60 | 11.665 | 14.336 | 1.229x | 46.2% | 0.172 | library_gemm (27) | fa4 (1.388x; 6) |
| 107 | outer_03.transformer_1.block_10.attention_out_projection_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 60 | 8.320 | 16.208 | 1.948x | 95.7% | 0.538 | library_gemm (31) | fused_ffn (2.492x; 29) |
| 108 | outer_03.transformer_1.block_10.ffn_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | 2.417 | 4.144 | 1.715x | 100.0% | 0.108 | fused_ffn (30) | fused_ffn (2.384x; 30) |
| 109 | outer_03.transformer_1.block_10.ffn_linear1_gate_swiglu | fused_ffn | fused_ffn; fused_ffn_kernel; g18x37x1; b128x1x1; r146; s32768 | 60 | 41.728 | 44.913 | 1.076x | 75.9% | 0.171 | fused_ffn (31) | fused_ffn (1.132x; 31) |
| 110 | outer_03.transformer_1.block_10.ffn_linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 60 | 20.384 | 30.544 | 1.498x | 98.4% | 0.650 | fused_ffn (30) | wide_qkv (1.612x; 30) |
| 111 | outer_03.transformer_2.block_11.attention_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | 2.368 | 3.776 | 1.595x | 100.0% | 0.095 | linear2_residual (30) | wide_qkv (1.905x; 30) |
| 112 | outer_03.transformer_2.block_11.attention_qkv_projection | wide_qkv | wide_qkv; wide_qkv_kernel; g9x37x1; b128x1x1; r136; s65536 | 60 | 18.944 | 24.209 | 1.278x | 82.9% | 0.333 | linear2_residual (30) | linear2_residual (1.466x; 30) |
| 113 | outer_03.transformer_2.block_11.attention_qk_rope | qk_rope | qk_rope; fusedQKRoPE19HalfKernel; g361x13x1; b192x1x1; r16; s0 | 60 | 4.064 | 7.296 | 1.795x | 94.8% | 0.192 | fa4 (30) | wide_qkv (2.220x; 30) |
| 114 | outer_03.transformer_2.block_11.attention_fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 60 | 11.616 | 14.368 | 1.237x | 48.8% | 0.185 | library_gemm (24) | fa4 (1.369x; 12) |
| 115 | outer_03.transformer_2.block_11.attention_out_projection_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 60 | 8.352 | 15.776 | 1.889x | 95.7% | 0.528 | library_gemm (32) | fused_ffn (2.467x; 28) |
| 116 | outer_03.transformer_2.block_11.ffn_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | 2.433 | 4.016 | 1.651x | 100.0% | 0.103 | fused_ffn (30) | fused_ffn (2.328x; 30) |
| 117 | outer_03.transformer_2.block_11.ffn_linear1_gate_swiglu | fused_ffn | fused_ffn; fused_ffn_kernel; g18x37x1; b128x1x1; r146; s32768 | 60 | 41.600 | 44.257 | 1.064x | 76.5% | 0.181 | fused_ffn (32) | fused_ffn (1.137x; 32) |
| 118 | outer_03.transformer_2.block_11.ffn_linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 60 | 20.369 | 29.888 | 1.467x | 98.2% | 0.581 | fused_ffn (30) | library_gemm (1.506x; 30) |
| 119 | outer_03.post_norm_silu | affine_silu | affine_silu; affineSiluHalf2Kernel; g4693x1x1; b192x1x1; r17; s0 | 60 | 3.296 | 3.968 | 1.204x | 100.0% | 0.040 | library_gemm (30) | linear2_residual (1.233x; 30) |
| 120 | outer_03.post_projection_c384_to_c768_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b256x1x1; r154; s73728 | 60 | 14.112 | 19.488 | 1.381x | 95.1% | 0.358 | affine_silu (30) | linear2_residual (1.565x; 30) |
| 121 | outer_04.pre_norm_silu | affine_silu | affine_silu; affineSiluHalf2Kernel; g4693x1x1; b384x1x1; r17; s0 | 60 | 4.992 | 7.936 | 1.590x | 100.0% | 0.183 | library_gemm (60) | library_gemm (1.590x; 60) |
| 122 | outer_04.pre_projection_c768_to_c384 | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 60 | 11.680 | 16.112 | 1.379x | 94.1% | 0.294 | wide_qkv (30) | library_gemm (1.649x; 5) |
| 123 | outer_04.transformer_0.block_12.attention_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | 2.400 | 4.352 | 1.813x | 100.0% | 0.121 | library_gemm (30) | wide_qkv (2.134x; 30) |
| 124 | outer_04.transformer_0.block_12.attention_qkv_projection | wide_qkv | wide_qkv; wide_qkv_kernel; g9x37x1; b128x1x1; r136; s65536 | 60 | 18.768 | 22.704 | 1.210x | 65.4% | 0.252 | qk_rope (30) | library_gemm (1.292x; 29) |
| 125 | outer_04.transformer_0.block_12.attention_qk_rope | qk_rope | qk_rope; fusedQKRoPE19HalfKernel; g361x13x1; b192x1x1; r16; s0 | 60 | 4.064 | 7.536 | 1.854x | 99.2% | 0.205 | fa4 (30) | wide_qkv (2.276x; 30) |
| 126 | outer_04.transformer_0.block_12.attention_fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 60 | 11.664 | 14.592 | 1.251x | 44.8% | 0.182 | library_gemm (29) | library_gemm (1.254x; 29) |
| 127 | outer_04.transformer_0.block_12.attention_out_projection_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 60 | 8.416 | 16.128 | 1.916x | 95.9% | 0.552 | library_gemm (31) | fused_ffn (2.487x; 29) |
| 128 | outer_04.transformer_0.block_12.ffn_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | 2.416 | 4.256 | 1.762x | 100.0% | 0.109 | fused_ffn (30) | fused_ffn (2.331x; 30) |
| 129 | outer_04.transformer_0.block_12.ffn_linear1_gate_swiglu | fused_ffn | fused_ffn; fused_ffn_kernel; g18x37x1; b128x1x1; r146; s32768 | 60 | 40.720 | 43.505 | 1.068x | 79.2% | 0.149 | fused_ffn (31) | fused_ffn (1.106x; 31) |
| 130 | outer_04.transformer_0.block_12.ffn_linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 60 | 20.704 | 32.864 | 1.587x | 98.1% | 0.752 | fused_ffn (30) | wide_qkv (1.684x; 30) |
| 131 | outer_04.transformer_1.block_13.attention_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | 2.432 | 3.888 | 1.599x | 100.0% | 0.098 | linear2_residual (30) | wide_qkv (2.039x; 30) |
| 132 | outer_04.transformer_1.block_13.attention_qkv_projection | wide_qkv | wide_qkv; wide_qkv_kernel; g9x37x1; b128x1x1; r136; s65536 | 60 | 19.217 | 25.793 | 1.342x | 83.3% | 0.413 | linear2_residual (30) | linear2_residual (1.579x; 30) |
| 133 | outer_04.transformer_1.block_13.attention_qk_rope | qk_rope | qk_rope; fusedQKRoPE19HalfKernel; g361x13x1; b192x1x1; r16; s0 | 60 | 4.096 | 7.568 | 1.848x | 99.2% | 0.208 | fa4 (30) | wide_qkv (2.281x; 30) |
| 134 | outer_04.transformer_1.block_13.attention_fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 60 | 11.793 | 14.720 | 1.248x | 44.5% | 0.179 | library_gemm (29) | library_gemm (1.254x; 29) |
| 135 | outer_04.transformer_1.block_13.attention_out_projection_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 60 | 8.384 | 17.504 | 2.088x | 95.8% | 0.553 | library_gemm (31) | fused_ffn (2.470x; 29) |
| 136 | outer_04.transformer_1.block_13.ffn_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | 2.432 | 3.632 | 1.493x | 100.0% | 0.078 | fused_ffn (30) | fused_ffn (1.987x; 30) |
| 137 | outer_04.transformer_1.block_13.ffn_linear1_gate_swiglu | fused_ffn | fused_ffn; fused_ffn_kernel; g18x37x1; b128x1x1; r146; s32768 | 60 | 42.416 | 44.449 | 1.048x | 82.3% | 0.109 | fused_ffn (30) | fused_ffn (1.079x; 30) |
| 138 | outer_04.transformer_1.block_13.ffn_linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 60 | 20.625 | 33.600 | 1.629x | 98.4% | 0.783 | fused_ffn (30) | wide_qkv (1.672x; 30) |
| 139 | outer_04.transformer_2.block_14.attention_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | 2.432 | 4.016 | 1.651x | 100.0% | 0.105 | linear2_residual (30) | wide_qkv (2.053x; 30) |
| 140 | outer_04.transformer_2.block_14.attention_qkv_projection | wide_qkv | wide_qkv; wide_qkv_kernel; g9x37x1; b128x1x1; r136; s65536 | 60 | 19.072 | 26.529 | 1.391x | 82.3% | 0.459 | linear2_residual (30) | linear2_residual (1.633x; 30) |
| 141 | outer_04.transformer_2.block_14.attention_qk_rope | qk_rope | qk_rope; fusedQKRoPE19HalfKernel; g361x13x1; b192x1x1; r16; s0 | 60 | 4.112 | 7.440 | 1.809x | 99.2% | 0.206 | fa4 (30) | wide_qkv (2.253x; 30) |
| 142 | outer_04.transformer_2.block_14.attention_fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 60 | 11.808 | 14.784 | 1.252x | 44.8% | 0.185 | library_gemm (29) | library_gemm (1.252x; 29) |
| 143 | outer_04.transformer_2.block_14.attention_out_projection_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 60 | 8.384 | 16.992 | 2.027x | 95.7% | 0.556 | fused_ffn (30) | fused_ffn (2.489x; 30) |
| 144 | outer_04.transformer_2.block_14.ffn_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | 2.400 | 4.048 | 1.687x | 100.0% | 0.102 | fused_ffn (30) | fused_ffn (2.293x; 30) |
| 145 | outer_04.transformer_2.block_14.ffn_linear1_gate_swiglu | fused_ffn | fused_ffn; fused_ffn_kernel; g18x37x1; b128x1x1; r146; s32768 | 60 | 36.800 | 41.281 | 1.122x | 82.5% | 0.245 | fused_ffn (30) | fused_ffn (1.147x; 30) |
| 146 | outer_04.transformer_2.block_14.ffn_linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 60 | 20.768 | 33.904 | 1.633x | 98.3% | 0.820 | fused_ffn (30) | library_gemm (1.716x; 30) |
| 147 | outer_04.post_norm_silu | affine_silu | affine_silu; affineSiluHalf2Kernel; g4693x1x1; b192x1x1; r17; s0 | 60 | 3.360 | 4.320 | 1.286x | 100.0% | 0.057 | library_gemm (30) | linear2_residual (1.314x; 30) |
| 148 | outer_04.post_projection_c384_to_c768_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b256x1x1; r154; s73728 | 60 | 14.432 | 21.281 | 1.475x | 95.3% | 0.476 | affine_silu (30) | linear2_residual (1.680x; 30) |
| 149 | outer_05.pre_norm_silu | affine_silu | affine_silu; affineSiluHalf2Kernel; g4693x1x1; b384x1x1; r17; s0 | 60 | 5.056 | 8.320 | 1.646x | 100.0% | 0.210 | library_gemm (60) | library_gemm (1.646x; 60) |
| 150 | outer_05.pre_projection_c768_to_c384 | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 60 | 11.937 | 17.280 | 1.448x | 94.5% | 0.383 | wide_qkv (30) | library_gemm (1.793x; 18) |
| 151 | outer_05.transformer_0.block_15.attention_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | 2.432 | 4.384 | 1.803x | 100.0% | 0.129 | library_gemm (30) | wide_qkv (2.309x; 30) |
| 152 | outer_05.transformer_0.block_15.attention_qkv_projection | wide_qkv | wide_qkv; wide_qkv_kernel; g9x37x1; b128x1x1; r136; s65536 | 60 | 19.136 | 24.432 | 1.277x | 65.4% | 0.322 | library_gemm (30) | library_gemm (1.379x; 30) |
| 153 | outer_05.transformer_0.block_15.attention_qk_rope | qk_rope | qk_rope; fusedQKRoPE19HalfKernel; g361x13x1; b192x1x1; r16; s0 | 60 | 4.096 | 7.648 | 1.867x | 99.2% | 0.208 | fa4 (30) | wide_qkv (2.309x; 30) |
| 154 | outer_05.transformer_0.block_15.attention_fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 60 | 11.840 | 15.296 | 1.292x | 47.5% | 0.220 | library_gemm (29) | qk_rope (1.311x; 29) |
| 155 | outer_05.transformer_0.block_15.attention_out_projection_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 60 | 8.448 | 17.776 | 2.104x | 95.9% | 0.570 | library_gemm (31) | fused_ffn (2.477x; 29) |
| 156 | outer_05.transformer_0.block_15.ffn_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | 2.432 | 4.144 | 1.704x | 100.0% | 0.103 | fused_ffn (30) | fused_ffn (2.283x; 30) |
| 157 | outer_05.transformer_0.block_15.ffn_linear1_gate_swiglu | fused_ffn | fused_ffn; fused_ffn_kernel; g18x37x1; b128x1x1; r146; s32768 | 60 | 42.321 | 47.425 | 1.121x | 78.4% | 0.323 | fused_ffn (30) | fused_ffn (1.177x; 30) |
| 158 | outer_05.transformer_0.block_15.ffn_linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 60 | 20.753 | 32.576 | 1.570x | 98.0% | 0.709 | fused_ffn (30) | wide_qkv (1.593x; 30) |
| 159 | outer_05.transformer_1.block_16.attention_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | 2.432 | 4.048 | 1.664x | 100.0% | 0.102 | linear2_residual (30) | wide_qkv (2.000x; 30) |
| 160 | outer_05.transformer_1.block_16.attention_qkv_projection | wide_qkv | wide_qkv; wide_qkv_kernel; g9x37x1; b128x1x1; r136; s65536 | 60 | 19.056 | 25.808 | 1.354x | 82.8% | 0.445 | linear2_residual (30) | linear2_residual (1.613x; 30) |
| 161 | outer_05.transformer_1.block_16.attention_qk_rope | qk_rope | qk_rope; fusedQKRoPE19HalfKernel; g361x13x1; b192x1x1; r16; s0 | 60 | 4.064 | 7.520 | 1.850x | 99.2% | 0.206 | fa4 (30) | wide_qkv (2.272x; 30) |
| 162 | outer_05.transformer_1.block_16.attention_fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 60 | 11.792 | 14.720 | 1.248x | 44.6% | 0.182 | library_gemm (29) | library_gemm (1.256x; 29) |
| 163 | outer_05.transformer_1.block_16.attention_out_projection_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 60 | 8.400 | 17.056 | 2.030x | 95.7% | 0.556 | fused_ffn (30) | fused_ffn (2.505x; 30) |
| 164 | outer_05.transformer_1.block_16.ffn_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | 2.432 | 4.048 | 1.664x | 100.0% | 0.103 | fused_ffn (30) | fused_ffn (2.322x; 30) |
| 165 | outer_05.transformer_1.block_16.ffn_linear1_gate_swiglu | fused_ffn | fused_ffn; fused_ffn_kernel; g18x37x1; b128x1x1; r146; s32768 | 60 | 42.673 | 45.536 | 1.067x | 76.5% | 0.184 | fused_ffn (33) | fused_ffn (1.133x; 33) |
| 166 | outer_05.transformer_1.block_16.ffn_linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 60 | 20.544 | 32.096 | 1.562x | 98.4% | 0.691 | fused_ffn (30) | wide_qkv (1.621x; 30) |
| 167 | outer_05.transformer_2.block_17.attention_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | 2.432 | 3.984 | 1.638x | 100.0% | 0.097 | linear2_residual (30) | wide_qkv (1.947x; 30) |
| 168 | outer_05.transformer_2.block_17.attention_qkv_projection | wide_qkv | wide_qkv; wide_qkv_kernel; g9x37x1; b128x1x1; r136; s65536 | 60 | 19.008 | 24.960 | 1.313x | 83.4% | 0.369 | linear2_residual (30) | linear2_residual (1.512x; 30) |
| 169 | outer_05.transformer_2.block_17.attention_qk_rope | qk_rope | qk_rope; fusedQKRoPE19HalfKernel; g361x13x1; b192x1x1; r16; s0 | 60 | 4.048 | 7.408 | 1.830x | 93.1% | 0.202 | fa4 (30) | wide_qkv (2.304x; 30) |
| 170 | outer_05.transformer_2.block_17.attention_fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 60 | 11.745 | 14.625 | 1.245x | 49.5% | 0.201 | library_gemm (22) | fa4 (1.403x; 16) |
| 171 | outer_05.transformer_2.block_17.attention_out_projection_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 60 | 8.352 | 15.344 | 1.837x | 95.7% | 0.530 | library_gemm (34) | fused_ffn (2.525x; 26) |
| 172 | outer_05.transformer_2.block_17.ffn_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | 2.400 | 4.096 | 1.707x | 100.0% | 0.108 | fused_ffn (30) | fused_ffn (2.360x; 30) |
| 173 | outer_05.transformer_2.block_17.ffn_linear1_gate_swiglu | fused_ffn | fused_ffn; fused_ffn_kernel; g18x37x1; b128x1x1; r146; s32768 | 60 | 42.736 | 45.600 | 1.067x | 76.3% | 0.191 | fused_ffn (32) | fused_ffn (1.131x; 32) |
| 174 | outer_05.transformer_2.block_17.ffn_linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 60 | 20.512 | 30.624 | 1.493x | 98.2% | 0.607 | fused_ffn (30) | library_gemm (1.518x; 30) |
| 175 | outer_05.post_norm_silu | affine_silu | affine_silu; affineSiluHalf2Kernel; g4693x1x1; b192x1x1; r17; s0 | 60 | 3.296 | 4.032 | 1.223x | 100.0% | 0.041 | library_gemm (30) | linear2_residual (1.243x; 30) |
| 176 | outer_05.post_projection_c384_to_c768_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b256x1x1; r154; s73728 | 60 | 14.208 | 20.241 | 1.425x | 95.2% | 0.398 | affine_silu (30) | linear2_residual (1.597x; 30) |
| 177 | outer_06.pre_norm_silu | affine_silu | affine_silu; affineSiluHalf2Kernel; g4693x1x1; b384x1x1; r17; s0 | 60 | 4.992 | 8.320 | 1.667x | 100.0% | 0.204 | library_gemm (60) | library_gemm (1.667x; 60) |
| 178 | outer_06.pre_projection_c768_to_c384 | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 60 | 11.776 | 16.128 | 1.370x | 94.1% | 0.303 | wide_qkv (30) | affine_silu (1.568x; 27) |
| 179 | outer_06.transformer_0.block_18.attention_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | 2.400 | 4.048 | 1.687x | 100.0% | 0.116 | library_gemm (30) | wide_qkv (2.133x; 30) |
| 180 | outer_06.transformer_0.block_18.attention_qkv_projection | wide_qkv | wide_qkv; wide_qkv_kernel; g9x37x1; b128x1x1; r136; s65536 | 60 | 18.977 | 24.208 | 1.276x | 65.7% | 0.282 | library_gemm (30) | library_gemm (1.330x; 30) |
| 181 | outer_06.transformer_0.block_18.attention_qk_rope | qk_rope | qk_rope; fusedQKRoPE19HalfKernel; g361x13x1; b192x1x1; r16; s0 | 60 | 4.096 | 7.296 | 1.781x | 100.0% | 0.209 | fa4 (30) | wide_qkv (2.289x; 30) |
| 182 | outer_06.transformer_0.block_18.attention_fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 60 | 11.728 | 14.896 | 1.270x | 44.7% | 0.196 | library_gemm (30) | library_gemm (1.273x; 30) |
| 183 | outer_06.transformer_0.block_18.attention_out_projection_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 60 | 8.384 | 17.504 | 2.088x | 95.8% | 0.575 | fused_ffn (30) | fused_ffn (2.498x; 30) |
| 184 | outer_06.transformer_0.block_18.ffn_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | 2.464 | 3.840 | 1.558x | 100.0% | 0.092 | fused_ffn (30) | fused_ffn (2.117x; 30) |
| 185 | outer_06.transformer_0.block_18.ffn_linear1_gate_swiglu | fused_ffn | fused_ffn; fused_ffn_kernel; g18x37x1; b128x1x1; r146; s32768 | 60 | 41.488 | 46.448 | 1.120x | 79.2% | 0.284 | fused_ffn (30) | fused_ffn (1.155x; 30) |
| 186 | outer_06.transformer_0.block_18.ffn_linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 60 | 20.736 | 32.880 | 1.586x | 98.1% | 0.733 | fused_ffn (30) | wide_qkv (1.610x; 30) |
| 187 | outer_06.transformer_1.block_19.attention_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | 2.464 | 3.936 | 1.597x | 100.0% | 0.100 | linear2_residual (30) | wide_qkv (2.052x; 30) |
| 188 | outer_06.transformer_1.block_19.attention_qkv_projection | wide_qkv | wide_qkv; wide_qkv_kernel; g9x37x1; b128x1x1; r136; s65536 | 60 | 19.153 | 26.209 | 1.368x | 82.8% | 0.445 | linear2_residual (30) | linear2_residual (1.619x; 30) |
| 189 | outer_06.transformer_1.block_19.attention_qk_rope | qk_rope | qk_rope; fusedQKRoPE19HalfKernel; g361x13x1; b192x1x1; r16; s0 | 60 | 4.096 | 7.520 | 1.836x | 100.0% | 0.206 | fa4 (30) | wide_qkv (2.258x; 30) |
| 190 | outer_06.transformer_1.block_19.attention_fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 60 | 11.808 | 14.704 | 1.245x | 44.5% | 0.176 | library_gemm (30) | library_gemm (1.247x; 30) |
| 191 | outer_06.transformer_1.block_19.attention_out_projection_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 60 | 8.416 | 17.488 | 2.078x | 95.8% | 0.560 | fused_ffn (30) | fused_ffn (2.492x; 30) |
| 192 | outer_06.transformer_1.block_19.ffn_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | 2.432 | 3.760 | 1.546x | 100.0% | 0.086 | fused_ffn (30) | fused_ffn (2.033x; 30) |
| 193 | outer_06.transformer_1.block_19.ffn_linear1_gate_swiglu | fused_ffn | fused_ffn; fused_ffn_kernel; g18x37x1; b128x1x1; r146; s32768 | 60 | 41.153 | 46.305 | 1.125x | 82.6% | 0.284 | fused_ffn (30) | fused_ffn (1.158x; 30) |
| 194 | outer_06.transformer_1.block_19.ffn_linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 60 | 20.736 | 33.120 | 1.597x | 98.4% | 0.741 | fused_ffn (30) | fused_ffn (1.605x; 30) |
| 195 | outer_06.transformer_2.block_20.attention_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | 2.432 | 4.160 | 1.711x | 100.0% | 0.105 | linear2_residual (30) | wide_qkv (2.099x; 30) |
| 196 | outer_06.transformer_2.block_20.attention_qkv_projection | wide_qkv | wide_qkv; wide_qkv_kernel; g9x37x1; b128x1x1; r136; s65536 | 60 | 19.168 | 26.768 | 1.396x | 80.3% | 0.452 | linear2_residual (30) | linear2_residual (1.625x; 30) |
| 197 | outer_06.transformer_2.block_20.attention_qk_rope | qk_rope | qk_rope; fusedQKRoPE19HalfKernel; g361x13x1; b192x1x1; r16; s0 | 60 | 4.144 | 7.504 | 1.811x | 100.0% | 0.203 | fa4 (30) | wide_qkv (2.216x; 30) |
| 198 | outer_06.transformer_2.block_20.attention_fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 60 | 11.840 | 14.752 | 1.246x | 43.4% | 0.174 | library_gemm (30) | library_gemm (1.249x; 30) |
| 199 | outer_06.transformer_2.block_20.attention_out_projection_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 60 | 8.384 | 17.632 | 2.103x | 95.8% | 0.560 | fused_ffn (30) | fused_ffn (2.498x; 30) |
| 200 | outer_06.transformer_2.block_20.ffn_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | 2.432 | 3.856 | 1.586x | 100.0% | 0.094 | fused_ffn (30) | fused_ffn (2.164x; 30) |
| 201 | outer_06.transformer_2.block_20.ffn_linear1_gate_swiglu | fused_ffn | fused_ffn; fused_ffn_kernel; g18x37x1; b128x1x1; r146; s32768 | 60 | 38.177 | 40.688 | 1.066x | 83.3% | 0.148 | fused_ffn (30) | fused_ffn (1.094x; 30) |
| 202 | outer_06.transformer_2.block_20.ffn_linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 60 | 20.768 | 34.240 | 1.649x | 98.3% | 0.839 | fused_ffn (30) | library_gemm (1.731x; 30) |
| 203 | outer_06.post_norm_silu | affine_silu | affine_silu; affineSiluHalf2Kernel; g4693x1x1; b192x1x1; r17; s0 | 60 | 3.328 | 4.336 | 1.303x | 100.0% | 0.060 | library_gemm (30) | linear2_residual (1.308x; 30) |
| 204 | outer_06.post_projection_c384_to_c768_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b256x1x1; r154; s73728 | 60 | 14.384 | 21.808 | 1.516x | 95.3% | 0.478 | affine_silu (30) | linear2_residual (1.687x; 30) |
| 205 | outer_07.pre_norm_silu | affine_silu | affine_silu; affineSiluHalf2Kernel; g4693x1x1; b384x1x1; r17; s0 | 60 | 5.056 | 8.464 | 1.674x | 100.0% | 0.214 | library_gemm (60) | library_gemm (1.674x; 60) |
| 206 | outer_07.pre_projection_c768_to_c384 | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 60 | 11.920 | 17.856 | 1.498x | 94.5% | 0.391 | wide_qkv (30) | library_gemm (1.805x; 20) |
| 207 | outer_07.transformer_0.block_21.attention_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | 2.432 | 4.448 | 1.829x | 100.0% | 0.124 | library_gemm (30) | wide_qkv (2.165x; 30) |
| 208 | outer_07.transformer_0.block_21.attention_qkv_projection | wide_qkv | wide_qkv; wide_qkv_kernel; g9x37x1; b128x1x1; r136; s65536 | 60 | 19.152 | 24.224 | 1.265x | 65.2% | 0.312 | library_gemm (30) | library_gemm (1.357x; 30) |
| 209 | outer_07.transformer_0.block_21.attention_qk_rope | qk_rope | qk_rope; fusedQKRoPE19HalfKernel; g361x13x1; b192x1x1; r16; s0 | 60 | 4.096 | 7.632 | 1.863x | 100.0% | 0.207 | fa4 (30) | wide_qkv (2.269x; 30) |
| 210 | outer_07.transformer_0.block_21.attention_fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 60 | 11.840 | 15.280 | 1.291x | 45.8% | 0.208 | library_gemm (30) | qk_rope (1.296x; 30) |
| 211 | outer_07.transformer_0.block_21.attention_out_projection_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 60 | 8.448 | 17.936 | 2.123x | 95.7% | 0.575 | library_gemm (32) | fused_ffn (2.496x; 28) |
| 212 | outer_07.transformer_0.block_21.ffn_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | 2.417 | 3.985 | 1.649x | 100.0% | 0.103 | fused_ffn (30) | fused_ffn (2.284x; 30) |
| 213 | outer_07.transformer_0.block_21.ffn_linear1_gate_swiglu | fused_ffn | fused_ffn; fused_ffn_kernel; g18x37x1; b128x1x1; r146; s32768 | 60 | 39.377 | 42.049 | 1.068x | 83.7% | 0.168 | fused_ffn (30) | fused_ffn (1.086x; 30) |
| 214 | outer_07.transformer_0.block_21.ffn_linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 60 | 20.816 | 35.920 | 1.726x | 98.2% | 0.909 | fused_ffn (30) | wide_qkv (1.779x; 30) |
| 215 | outer_07.transformer_1.block_22.attention_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | 2.400 | 3.664 | 1.527x | 100.0% | 0.097 | linear2_residual (30) | wide_qkv (1.993x; 30) |
| 216 | outer_07.transformer_1.block_22.attention_qkv_projection | wide_qkv | wide_qkv; wide_qkv_kernel; g9x37x1; b128x1x1; r136; s65536 | 60 | 19.312 | 27.041 | 1.400x | 83.8% | 0.477 | linear2_residual (30) | linear2_residual (1.610x; 30) |
| 217 | outer_07.transformer_1.block_22.attention_qk_rope | qk_rope | qk_rope; fusedQKRoPE19HalfKernel; g361x13x1; b192x1x1; r16; s0 | 60 | 4.128 | 7.696 | 1.864x | 99.1% | 0.220 | fa4 (30) | wide_qkv (2.357x; 30) |
| 218 | outer_07.transformer_1.block_22.attention_fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 60 | 11.889 | 14.944 | 1.257x | 45.5% | 0.192 | library_gemm (29) | qk_rope (1.262x; 29) |
| 219 | outer_07.transformer_1.block_22.attention_out_projection_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 60 | 8.480 | 17.392 | 2.051x | 95.7% | 0.561 | fused_ffn (30) | fused_ffn (2.489x; 30) |
| 220 | outer_07.transformer_1.block_22.ffn_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | 2.464 | 3.840 | 1.558x | 100.0% | 0.088 | fused_ffn (30) | fused_ffn (2.065x; 30) |
| 221 | outer_07.transformer_1.block_22.ffn_linear1_gate_swiglu | fused_ffn | fused_ffn; fused_ffn_kernel; g18x37x1; b128x1x1; r146; s32768 | 60 | 38.128 | 42.176 | 1.106x | 84.3% | 0.244 | fused_ffn (30) | fused_ffn (1.138x; 30) |
| 222 | outer_07.transformer_1.block_22.ffn_linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 60 | 20.816 | 35.216 | 1.692x | 98.5% | 0.881 | fused_ffn (30) | wide_qkv (1.789x; 30) |
| 223 | outer_07.transformer_2.block_23.attention_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | 2.464 | 4.000 | 1.623x | 100.0% | 0.094 | linear2_residual (30) | wide_qkv (1.948x; 30) |
| 224 | outer_07.transformer_2.block_23.attention_qkv_projection | wide_qkv | wide_qkv; wide_qkv_kernel; g9x37x1; b128x1x1; r136; s65536 | 60 | 19.313 | 26.928 | 1.394x | 83.1% | 0.480 | linear2_residual (30) | linear2_residual (1.626x; 30) |
| 225 | outer_07.transformer_2.block_23.attention_qk_rope | qk_rope | qk_rope; fusedQKRoPE19HalfKernel; g361x13x1; b192x1x1; r16; s0 | 60 | 4.128 | 7.553 | 1.830x | 99.1% | 0.211 | fa4 (30) | wide_qkv (2.295x; 30) |
| 226 | outer_07.transformer_2.block_23.attention_fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 60 | 11.904 | 14.976 | 1.258x | 45.5% | 0.192 | library_gemm (29) | qk_rope (1.266x; 29) |
| 227 | outer_07.transformer_2.block_23.attention_out_projection_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 60 | 8.496 | 17.377 | 2.045x | 95.8% | 0.552 | fused_ffn (30) | fused_ffn (2.461x; 30) |
| 228 | outer_07.transformer_2.block_23.ffn_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | 2.432 | 4.000 | 1.645x | 100.0% | 0.096 | fused_ffn (30) | fused_ffn (2.204x; 30) |
| 229 | outer_07.transformer_2.block_23.ffn_linear1_gate_swiglu | fused_ffn | fused_ffn; fused_ffn_kernel; g18x37x1; b128x1x1; r146; s32768 | 60 | 37.185 | 41.809 | 1.124x | 84.4% | 0.279 | fused_ffn (30) | fused_ffn (1.154x; 30) |
| 230 | outer_07.transformer_2.block_23.ffn_linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 60 | 20.832 | 35.104 | 1.685x | 98.4% | 0.871 | fused_ffn (30) | library_gemm (1.722x; 30) |
| 231 | outer_07.post_norm_silu | affine_silu | affine_silu; affineSiluHalf2Kernel; g4693x1x1; b192x1x1; r17; s0 | 60 | 3.360 | 4.320 | 1.286x | 100.0% | 0.058 | library_gemm (30) | linear2_residual (1.324x; 30) |
| 232 | outer_07.post_projection_c384_to_c768_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b256x1x1; r154; s73728 | 60 | 14.448 | 21.904 | 1.516x | 95.5% | 0.504 | affine_silu (30) | linear2_residual (1.755x; 30) |
| 233 | outer_08.pre_norm_silu | affine_silu | affine_silu; affineSiluHalf2Kernel; g4693x1x1; b384x1x1; r17; s0 | 60 | 5.056 | 8.512 | 1.684x | 100.0% | 0.214 | library_gemm (60) | library_gemm (1.684x; 60) |
| 234 | outer_08.pre_projection_c768_to_c384 | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 60 | 11.953 | 17.760 | 1.486x | 94.4% | 0.378 | wide_qkv (30) | library_gemm (1.796x; 14) |
| 235 | outer_08.transformer_0.block_24.attention_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | 2.432 | 4.384 | 1.803x | 100.0% | 0.125 | library_gemm (30) | wide_qkv (2.178x; 30) |
| 236 | outer_08.transformer_0.block_24.attention_qkv_projection | wide_qkv | wide_qkv; wide_qkv_kernel; g9x37x1; b128x1x1; r136; s65536 | 60 | 19.232 | 24.529 | 1.275x | 65.3% | 0.321 | library_gemm (30) | library_gemm (1.376x; 30) |
| 237 | outer_08.transformer_0.block_24.attention_qk_rope | qk_rope | qk_rope; fusedQKRoPE19HalfKernel; g361x13x1; b192x1x1; r16; s0 | 60 | 4.128 | 7.664 | 1.857x | 100.0% | 0.202 | fa4 (30) | wide_qkv (2.240x; 30) |
| 238 | outer_08.transformer_0.block_24.attention_fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 60 | 11.872 | 15.392 | 1.296x | 47.1% | 0.215 | library_gemm (30) | qk_rope (1.313x; 30) |
| 239 | outer_08.transformer_0.block_24.attention_out_projection_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 60 | 8.544 | 18.241 | 2.135x | 95.9% | 0.585 | library_gemm (31) | fused_ffn (2.494x; 29) |
| 240 | outer_08.transformer_0.block_24.ffn_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | 2.432 | 4.032 | 1.658x | 100.0% | 0.101 | fused_ffn (30) | fused_ffn (2.263x; 30) |
| 241 | outer_08.transformer_0.block_24.ffn_linear1_gate_swiglu | fused_ffn | fused_ffn; fused_ffn_kernel; g18x37x1; b128x1x1; r146; s32768 | 60 | 41.936 | 46.753 | 1.115x | 81.9% | 0.328 | fused_ffn (33) | fused_ffn (1.187x; 33) |
| 242 | outer_08.transformer_0.block_24.ffn_linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 60 | 20.881 | 33.840 | 1.621x | 98.1% | 0.779 | fused_ffn (30) | wide_qkv (1.645x; 30) |
| 243 | outer_08.transformer_1.block_25.attention_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | 2.464 | 3.888 | 1.578x | 100.0% | 0.100 | linear2_residual (30) | wide_qkv (2.032x; 30) |
| 244 | outer_08.transformer_1.block_25.attention_qkv_projection | wide_qkv | wide_qkv; wide_qkv_kernel; g9x37x1; b128x1x1; r136; s65536 | 60 | 19.265 | 26.896 | 1.396x | 82.7% | 0.470 | linear2_residual (30) | linear2_residual (1.639x; 30) |
| 245 | outer_08.transformer_1.block_25.attention_qk_rope | qk_rope | qk_rope; fusedQKRoPE19HalfKernel; g361x13x1; b192x1x1; r16; s0 | 60 | 4.080 | 7.504 | 1.839x | 99.2% | 0.207 | fa4 (30) | wide_qkv (2.282x; 30) |
| 246 | outer_08.transformer_1.block_25.attention_fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 60 | 11.872 | 14.816 | 1.248x | 45.0% | 0.181 | library_gemm (29) | library_gemm (1.248x; 29) |
| 247 | outer_08.transformer_1.block_25.attention_out_projection_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 60 | 8.416 | 16.928 | 2.011x | 95.8% | 0.557 | fused_ffn (30) | fused_ffn (2.483x; 30) |
| 248 | outer_08.transformer_1.block_25.ffn_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | 2.448 | 4.000 | 1.634x | 100.0% | 0.099 | fused_ffn (30) | fused_ffn (2.242x; 30) |
| 249 | outer_08.transformer_1.block_25.ffn_linear1_gate_swiglu | fused_ffn | fused_ffn; fused_ffn_kernel; g18x37x1; b128x1x1; r146; s32768 | 60 | 42.225 | 46.992 | 1.113x | 79.3% | 0.277 | fused_ffn (30) | fused_ffn (1.160x; 30) |
| 250 | outer_08.transformer_1.block_25.ffn_linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 60 | 20.784 | 32.736 | 1.575x | 98.4% | 0.731 | fused_ffn (30) | wide_qkv (1.643x; 30) |
| 251 | outer_08.transformer_2.block_26.attention_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | 2.464 | 3.856 | 1.565x | 100.0% | 0.097 | linear2_residual (30) | wide_qkv (1.987x; 30) |
| 252 | outer_08.transformer_2.block_26.attention_qkv_projection | wide_qkv | wide_qkv; wide_qkv_kernel; g9x37x1; b128x1x1; r136; s65536 | 60 | 19.280 | 26.241 | 1.361x | 83.3% | 0.432 | linear2_residual (30) | linear2_residual (1.612x; 30) |
| 253 | outer_08.transformer_2.block_26.attention_qk_rope | qk_rope | qk_rope; fusedQKRoPE19HalfKernel; g361x13x1; b192x1x1; r16; s0 | 60 | 4.096 | 7.408 | 1.809x | 100.0% | 0.199 | fa4 (30) | wide_qkv (2.242x; 30) |
| 254 | outer_08.transformer_2.block_26.attention_fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 60 | 11.841 | 14.752 | 1.246x | 45.0% | 0.181 | library_gemm (30) | library_gemm (1.246x; 30) |
| 255 | outer_08.transformer_2.block_26.attention_out_projection_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 60 | 8.432 | 17.280 | 2.049x | 95.8% | 0.546 | fused_ffn (30) | fused_ffn (2.429x; 30) |
| 256 | outer_08.transformer_2.block_26.ffn_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | 2.433 | 3.872 | 1.591x | 100.0% | 0.094 | fused_ffn (30) | fused_ffn (2.190x; 30) |
| 257 | outer_08.transformer_2.block_26.ffn_linear1_gate_swiglu | fused_ffn | fused_ffn; fused_ffn_kernel; g18x37x1; b128x1x1; r146; s32768 | 60 | 41.425 | 45.792 | 1.105x | 80.0% | 0.259 | fused_ffn (30) | fused_ffn (1.146x; 30) |
| 258 | outer_08.transformer_2.block_26.ffn_linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 60 | 20.752 | 32.608 | 1.571x | 98.3% | 0.712 | fused_ffn (30) | library_gemm (1.584x; 30) |
| 259 | outer_08.post_norm_silu | affine_silu | affine_silu; affineSiluHalf2Kernel; g4693x1x1; b192x1x1; r17; s0 | 60 | 3.360 | 4.096 | 1.219x | 100.0% | 0.041 | library_gemm (30) | linear2_residual (1.229x; 30) |
| 260 | outer_08.post_projection_c384_to_c768_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b256x1x1; r154; s73728 | 60 | 14.368 | 22.544 | 1.569x | 95.6% | 0.510 | affine_silu (30) | linear2_residual (1.811x; 30) |
| 261 | outer_09.pre_norm_silu | affine_silu | affine_silu; affineSiluHalf2Kernel; g4693x1x1; b384x1x1; r17; s0 | 60 | 4.992 | 8.240 | 1.651x | 100.0% | 0.199 | library_gemm (60) | library_gemm (1.651x; 60) |
| 262 | outer_09.pre_projection_c768_to_c384 | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 60 | 11.920 | 17.184 | 1.442x | 94.1% | 0.324 | affine_silu (30) | affine_silu (1.573x; 30) |
| 263 | outer_09.transformer_0.block_27.attention_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | 2.400 | 4.624 | 1.927x | 100.0% | 0.132 | library_gemm (30) | wide_qkv (2.234x; 30) |
| 264 | outer_09.transformer_0.block_27.attention_qkv_projection | wide_qkv | wide_qkv; wide_qkv_kernel; g9x37x1; b128x1x1; r136; s65536 | 60 | 19.136 | 24.592 | 1.285x | 65.5% | 0.305 | library_gemm (30) | library_gemm (1.345x; 30) |
| 265 | outer_09.transformer_0.block_27.attention_qk_rope | qk_rope | qk_rope; fusedQKRoPE19HalfKernel; g361x13x1; b192x1x1; r16; s0 | 60 | 4.064 | 7.616 | 1.874x | 99.2% | 0.213 | fa4 (30) | wide_qkv (2.354x; 30) |
| 266 | outer_09.transformer_0.block_27.attention_fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 60 | 11.872 | 15.264 | 1.286x | 47.0% | 0.218 | library_gemm (29) | qk_rope (1.315x; 29) |
| 267 | outer_09.transformer_0.block_27.attention_out_projection_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 60 | 8.416 | 17.601 | 2.091x | 95.7% | 0.568 | library_gemm (31) | fused_ffn (2.479x; 29) |
| 268 | outer_09.transformer_0.block_27.ffn_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | 2.464 | 4.144 | 1.682x | 100.0% | 0.108 | fused_ffn (30) | fused_ffn (2.351x; 30) |
| 269 | outer_09.transformer_0.block_27.ffn_linear1_gate_swiglu | fused_ffn | fused_ffn; fused_ffn_kernel; g18x37x1; b128x1x1; r146; s32768 | 60 | 42.609 | 46.225 | 1.085x | 75.1% | 0.206 | fused_ffn (30) | fused_ffn (1.154x; 30) |
| 270 | outer_09.transformer_0.block_27.ffn_linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 60 | 20.625 | 31.248 | 1.515x | 98.0% | 0.659 | fused_ffn (30) | wide_qkv (1.611x; 30) |
| 271 | outer_09.transformer_1.block_28.attention_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | 2.432 | 3.904 | 1.605x | 100.0% | 0.093 | linear2_residual (30) | wide_qkv (1.895x; 30) |
| 272 | outer_09.transformer_1.block_28.attention_qkv_projection | wide_qkv | wide_qkv; wide_qkv_kernel; g9x37x1; b128x1x1; r136; s65536 | 60 | 18.976 | 24.673 | 1.300x | 83.3% | 0.361 | linear2_residual (30) | linear2_residual (1.501x; 30) |
| 273 | outer_09.transformer_1.block_28.attention_qk_rope | qk_rope | qk_rope; fusedQKRoPE19HalfKernel; g361x13x1; b192x1x1; r16; s0 | 60 | 4.064 | 7.536 | 1.854x | 94.1% | 0.206 | fa4 (30) | wide_qkv (2.327x; 30) |
| 274 | outer_09.transformer_1.block_28.attention_fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 60 | 11.712 | 14.528 | 1.240x | 48.7% | 0.192 | library_gemm (23) | fa4 (1.396x; 14) |
| 275 | outer_09.transformer_1.block_28.attention_out_projection_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 60 | 8.384 | 15.440 | 1.842x | 95.7% | 0.531 | library_gemm (34) | fused_ffn (2.519x; 26) |
| 276 | outer_09.transformer_1.block_28.ffn_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | 2.432 | 4.096 | 1.684x | 100.0% | 0.106 | fused_ffn (30) | fused_ffn (2.322x; 30) |
| 277 | outer_09.transformer_1.block_28.ffn_linear1_gate_swiglu | fused_ffn | fused_ffn; fused_ffn_kernel; g18x37x1; b128x1x1; r146; s32768 | 60 | 41.937 | 45.425 | 1.083x | 76.1% | 0.188 | fused_ffn (31) | fused_ffn (1.139x; 31) |
| 278 | outer_09.transformer_1.block_28.ffn_linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 60 | 20.480 | 30.672 | 1.498x | 98.4% | 0.648 | fused_ffn (30) | wide_qkv (1.602x; 30) |
| 279 | outer_09.transformer_2.block_29.attention_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | 2.368 | 3.889 | 1.642x | 100.0% | 0.096 | linear2_residual (30) | wide_qkv (1.939x; 30) |
| 280 | outer_09.transformer_2.block_29.attention_qkv_projection | wide_qkv | wide_qkv; wide_qkv_kernel; g9x37x1; b128x1x1; r136; s65536 | 60 | 18.913 | 24.416 | 1.291x | 82.8% | 0.337 | linear2_residual (30) | linear2_residual (1.471x; 30) |
| 281 | outer_09.transformer_2.block_29.attention_qk_rope | qk_rope | qk_rope; fusedQKRoPE19HalfKernel; g361x13x1; b192x1x1; r16; s0 | 60 | 4.096 | 7.296 | 1.781x | 94.9% | 0.193 | fa4 (30) | wide_qkv (2.227x; 30) |
| 282 | outer_09.transformer_2.block_29.attention_fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 60 | 11.681 | 14.400 | 1.233x | 48.7% | 0.183 | library_gemm (24) | fa4 (1.385x; 12) |
| 283 | outer_09.transformer_2.block_29.attention_out_projection_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 60 | 8.400 | 15.952 | 1.899x | 95.7% | 0.531 | library_gemm (31) | fused_ffn (2.461x; 29) |
| 284 | outer_09.transformer_2.block_29.ffn_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | 2.432 | 4.032 | 1.658x | 100.0% | 0.099 | fused_ffn (30) | fused_ffn (2.263x; 30) |
| 285 | outer_09.transformer_2.block_29.ffn_linear1_gate_swiglu | fused_ffn | fused_ffn; fused_ffn_kernel; g18x37x1; b128x1x1; r146; s32768 | 60 | 41.296 | 44.865 | 1.086x | 76.7% | 0.215 | fused_ffn (34) | fused_ffn (1.145x; 34) |
| 286 | outer_09.transformer_2.block_29.ffn_linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 60 | 20.417 | 30.144 | 1.476x | 98.2% | 0.595 | fused_ffn (30) | library_gemm (1.502x; 30) |
| 287 | outer_09.post_norm_silu | affine_silu | affine_silu; affineSiluHalf2Kernel; g4693x1x1; b192x1x1; r17; s0 | 60 | 3.296 | 3.968 | 1.204x | 100.0% | 0.039 | library_gemm (30) | linear2_residual (1.238x; 30) |
| 288 | outer_09.post_projection_c384_to_c768_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b256x1x1; r154; s73728 | 60 | 14.080 | 19.761 | 1.403x | 95.3% | 0.378 | affine_silu (30) | linear2_residual (1.572x; 30) |
| 289 | outer_10.pre_norm_silu | affine_silu | affine_silu; affineSiluHalf2Kernel; g4693x1x1; b384x1x1; r17; s0 | 60 | 4.992 | 7.872 | 1.577x | 100.0% | 0.186 | library_gemm (60) | library_gemm (1.577x; 60) |
| 290 | outer_10.pre_projection_c768_to_c384 | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 60 | 11.681 | 16.113 | 1.379x | 94.1% | 0.298 | wide_qkv (30) | library_gemm (1.647x; 7) |
| 291 | outer_10.transformer_0.block_30.attention_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | 2.432 | 4.224 | 1.737x | 100.0% | 0.114 | library_gemm (30) | wide_qkv (2.112x; 30) |
| 292 | outer_10.transformer_0.block_30.attention_qkv_projection | wide_qkv | wide_qkv; wide_qkv_kernel; g9x37x1; b128x1x1; r136; s65536 | 60 | 18.912 | 23.392 | 1.237x | 65.7% | 0.262 | library_gemm (30) | library_gemm (1.316x; 30) |
| 293 | outer_10.transformer_0.block_30.attention_qk_rope | qk_rope | qk_rope; fusedQKRoPE19HalfKernel; g361x13x1; b192x1x1; r16; s0 | 60 | 4.032 | 7.408 | 1.837x | 98.3% | 0.207 | fa4 (30) | wide_qkv (2.321x; 30) |
| 294 | outer_10.transformer_0.block_30.attention_fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 60 | 11.712 | 14.672 | 1.253x | 45.4% | 0.186 | library_gemm (28) | fa4 (1.418x; 4) |
| 295 | outer_10.transformer_0.block_30.attention_out_projection_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 60 | 8.368 | 16.288 | 1.946x | 95.9% | 0.552 | library_gemm (31) | fused_ffn (2.493x; 29) |
| 296 | outer_10.transformer_0.block_30.ffn_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | 2.432 | 3.968 | 1.632x | 100.0% | 0.094 | fused_ffn (30) | fused_ffn (2.164x; 30) |
| 297 | outer_10.transformer_0.block_30.ffn_linear1_gate_swiglu | fused_ffn | fused_ffn; fused_ffn_kernel; g18x37x1; b128x1x1; r146; s32768 | 60 | 39.904 | 42.145 | 1.056x | 83.8% | 0.131 | fused_ffn (30) | fused_ffn (1.077x; 30) |
| 298 | outer_10.transformer_0.block_30.ffn_linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 60 | 20.720 | 34.081 | 1.645x | 98.1% | 0.807 | fused_ffn (30) | wide_qkv (1.688x; 30) |
| 299 | outer_10.transformer_1.block_31.attention_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | 2.464 | 4.192 | 1.701x | 100.0% | 0.109 | linear2_residual (30) | wide_qkv (2.045x; 30) |
| 300 | outer_10.transformer_1.block_31.attention_qkv_projection | wide_qkv | wide_qkv; wide_qkv_kernel; g9x37x1; b128x1x1; r136; s65536 | 60 | 19.200 | 26.576 | 1.384x | 82.7% | 0.452 | linear2_residual (30) | linear2_residual (1.607x; 30) |
| 301 | outer_10.transformer_1.block_31.attention_qk_rope | qk_rope | qk_rope; fusedQKRoPE19HalfKernel; g361x13x1; b192x1x1; r16; s0 | 60 | 4.096 | 7.505 | 1.832x | 98.3% | 0.207 | fa4 (30) | wide_qkv (2.297x; 30) |
| 302 | outer_10.transformer_1.block_31.attention_fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 60 | 11.824 | 14.800 | 1.252x | 45.9% | 0.188 | library_gemm (28) | fa4 (1.411x; 4) |
| 303 | outer_10.transformer_1.block_31.attention_out_projection_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 60 | 8.416 | 17.248 | 2.049x | 95.8% | 0.551 | fused_ffn (30) | fused_ffn (2.479x; 30) |
| 304 | outer_10.transformer_1.block_31.ffn_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | 2.432 | 3.984 | 1.638x | 100.0% | 0.098 | fused_ffn (30) | fused_ffn (2.230x; 30) |
| 305 | outer_10.transformer_1.block_31.ffn_linear1_gate_swiglu | fused_ffn | fused_ffn; fused_ffn_kernel; g18x37x1; b128x1x1; r146; s32768 | 60 | 38.961 | 43.984 | 1.129x | 81.2% | 0.281 | fused_ffn (30) | fused_ffn (1.159x; 30) |
| 306 | outer_10.transformer_1.block_31.ffn_linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 60 | 20.704 | 32.913 | 1.590x | 98.4% | 0.759 | fused_ffn (30) | wide_qkv (1.677x; 30) |
| 307 | outer_10.transformer_2.block_32.attention_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | 2.400 | 4.176 | 1.740x | 100.0% | 0.112 | linear2_residual (30) | wide_qkv (2.153x; 30) |
| 308 | outer_10.transformer_2.block_32.attention_qkv_projection | wide_qkv | wide_qkv; wide_qkv_kernel; g9x37x1; b128x1x1; r136; s65536 | 60 | 19.184 | 26.800 | 1.397x | 83.1% | 0.472 | linear2_residual (30) | linear2_residual (1.645x; 30) |
| 309 | outer_10.transformer_2.block_32.attention_qk_rope | qk_rope | qk_rope; fusedQKRoPE19HalfKernel; g361x13x1; b192x1x1; r16; s0 | 60 | 4.128 | 7.616 | 1.845x | 97.5% | 0.213 | fa4 (30) | wide_qkv (2.326x; 30) |
| 310 | outer_10.transformer_2.block_32.attention_fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 60 | 11.857 | 14.816 | 1.250x | 46.3% | 0.191 | library_gemm (27) | fa4 (1.407x; 6) |
| 311 | outer_10.transformer_2.block_32.attention_out_projection_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 60 | 8.384 | 17.136 | 2.044x | 95.8% | 0.548 | library_gemm (31) | fused_ffn (2.469x; 29) |
| 312 | outer_10.transformer_2.block_32.ffn_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | 2.464 | 3.920 | 1.591x | 100.0% | 0.090 | fused_ffn (30) | fused_ffn (2.117x; 30) |
| 313 | outer_10.transformer_2.block_32.ffn_linear1_gate_swiglu | fused_ffn | fused_ffn; fused_ffn_kernel; g18x37x1; b128x1x1; r146; s32768 | 60 | 41.649 | 47.184 | 1.133x | 80.8% | 0.299 | fused_ffn (30) | fused_ffn (1.161x; 30) |
| 314 | outer_10.transformer_2.block_32.ffn_linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 60 | 20.752 | 32.689 | 1.575x | 98.2% | 0.701 | fused_ffn (30) | fused_ffn (1.584x; 30) |
| 315 | outer_10.post_norm_silu | affine_silu | affine_silu; affineSiluHalf2Kernel; g4693x1x1; b192x1x1; r17; s0 | 60 | 3.360 | 4.128 | 1.229x | 100.0% | 0.043 | library_gemm (30) | linear2_residual (1.267x; 30) |
| 316 | outer_10.post_projection_c384_to_c768_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b256x1x1; r154; s73728 | 60 | 14.320 | 22.704 | 1.585x | 95.5% | 0.543 | affine_silu (30) | linear2_residual (1.828x; 30) |
| 317 | trunk.tip_norm_silu | affine_silu | affine_silu; affineSiluHalf2Kernel; g4693x1x1; b384x1x1; r17; s0 | 60 | 5.024 | 8.240 | 1.640x | 86.7% | 0.177 | library_gemm (60) | library_gemm (1.640x; 60) |
| 318 | policy.p1_conv | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r80; s98304 | 60 | 6.144 | 10.368 | 1.688x | 93.4% | 0.246 | library_gemm (60) | library_gemm (1.688x; 60) |
| 319 | policy.g1_conv | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r80; s98304 | 60 | 5.856 | 7.840 | 1.339x | 91.3% | 0.115 | head_elementwise (30) | library_gemm (1.503x; 30) |
| 320 | policy.g1_norm_silu | head_elementwise | head_elementwise; applyCScaleBiasNHWCSiluHalfKernel; g1x73x13; b96x5x1; r16; s0 | 60 | 2.112 | 2.656 | 1.258x | 99.7% | 0.034 | head_elementwise (30) | library_gemm (1.326x; 30) |
| 321 | policy.g1_half_to_float | copy_reformat | copy_reformat; copyFromHalfKernel; g880x1x1; b512x1x1; r16; s0 | 60 | 1.536 | 1.872 | 1.219x | 57.7% | 0.021 | library_gemm (40) | library_gemm (1.271x; 40) |
| 322 | policy.g1_global_pool | head_elementwise | head_elementwise; gPoolChannelsNHWCKernel; g2x1x13; b64x8x1; r22; s4096 | 60 | 4.416 | 6.608 | 1.496x | 94.5% | 0.175 | library_gemm (60) | library_gemm (1.496x; 60) |
| 323 | policy.gpool_to_bias_matmul | library_gemm | library_gemm; gemmSN_NN_kernel; g2x2x1; b256x1x1; r64; s21504 | 60 | 5.376 | 6.592 | 1.226x | 82.1% | 0.078 | head_elementwise (60) | head_elementwise (1.226x; 60) |
| 324 | policy.p1_half_to_float | copy_reformat | copy_reformat; copyFromHalfKernel; g880x1x1; b512x1x1; r16; s0 | 60 | 1.504 | 1.696 | 1.128x | 77.0% | 0.015 | library_gemm (50) | head_elementwise (1.191x; 10) |
| 325 | policy.gpool_bias_add | head_elementwise | head_elementwise; addNCBiasInplaceNHWCKernel; g1x73x13; b96x5x1; r16; s0 | 60 | 1.792 | 1.984 | 1.107x | 97.9% | 0.015 | library_gemm (60) | library_gemm (1.107x; 60) |
| 326 | policy.p1_norm_silu | head_elementwise | head_elementwise; applyCScaleBiasNHWCSiluKernel; g1x73x13; b96x5x1; r16; s0 | 60 | 2.144 | 2.688 | 1.254x | 86.9% | 0.027 | library_gemm (58) | library_gemm (1.246x; 58) |
| 327 | policy.p2_conv | library_gemm | library_gemm; Kernel2; g74x1x1; b128x1x1; r90; s98304 | 60 | 3.872 | 4.528 | 1.169x | 89.2% | 0.060 | library_gemm (30) | head_elementwise (1.376x; 26) |
| 328 | policy.gpool_to_pass_matmul | library_gemm | library_gemm; gemmSN_NN_kernel; g2x2x1; b256x1x1; r64; s21504 | 60 | 5.248 | 6.048 | 1.152x | 84.2% | 0.057 | library_gemm (56) | head_elementwise (1.171x; 4) |
| 329 | policy.pass_bias_silu | head_elementwise | head_elementwise; addCBiasInplaceNCKernelSilu; g1x3x1; b96x5x1; r16; s0 | 60 | 0.992 | 1.200 | 1.210x | 99.3% | 0.050 | library_gemm (60) | library_gemm (1.210x; 60) |
| 330 | policy.gpool_to_pass_matmul2 | library_gemm | library_gemm; gemmSN_NN_kernel; g1x2x1; b256x1x1; r64; s21504 | 60 | 2.304 | 2.640 | 1.146x | 97.3% | 0.036 | library_gemm (56) | head_elementwise (1.181x; 4) |
| 331 | value.v1_conv | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r118; s98304 | 60 | 7.904 | 9.312 | 1.178x | 85.2% | 0.093 | head_elementwise (28) | copy_reformat (1.198x; 7) |
| 332 | value.v1_norm_silu | head_elementwise | head_elementwise; applyCScaleBiasNHWCSiluHalfKernel; g1x181x13; b192x2x1; r16; s0 | 60 | 3.136 | 3.680 | 1.173x | 77.3% | 0.074 | library_gemm (30) | library_gemm (1.561x; 30) |
| 333 | value.v1_half_to_float | copy_reformat | copy_reformat; copyFromHalfKernel; g1760x1x1; b512x1x1; r16; s0 | 60 | 2.144 | 2.528 | 1.179x | 92.8% | 0.055 | head_elementwise (33) | library_gemm (1.284x; 27) |
| 334 | value.v1_global_pool | head_elementwise | head_elementwise; valueHeadPoolChannelsNHWCKernel; g3x1x13; b64x8x1; r22; s2048 | 60 | 3.200 | 3.712 | 1.160x | 89.7% | 0.042 | library_gemm (30) | head_elementwise (1.370x; 7) |
| 335 | value.v2_matmul | library_gemm | library_gemm; gemmSN_NN_kernel; g3x2x1; b256x1x1; r64; s21504 | 60 | 9.473 | 10.016 | 1.057x | 93.0% | 0.058 | library_gemm (59) | library_gemm (1.057x; 59) |
| 336 | value.v2_bias_silu | head_elementwise | head_elementwise; addCBiasInplaceNCKernelSilu; g1x7x1; b192x2x1; r16; s0 | 60 | 1.024 | 1.056 | 1.032x | 97.9% | 0.003 | library_gemm (59) | library_gemm (1.032x; 59) |
| 337 | value.v3_matmul | library_gemm | library_gemm; gemmSN_NN_kernel; g1x2x1; b256x1x1; r64; s21504 | 60 | 3.472 | 3.808 | 1.097x | 76.0% | 0.020 | library_gemm (51) | library_gemm (1.097x; 51) |
| 338 | value.v3_bias | head_elementwise | head_elementwise; addCBiasInplaceNCKernel; g1x1x1; b3x170x1; r16; s0 | 60 | 0.960 | 1.024 | 1.067x | 96.9% | 0.006 | library_gemm (58) | library_gemm (1.067x; 58) |
| 339 | value.score_matmul | library_gemm | library_gemm; gemmSN_NN_kernel; g1x2x1; b256x1x1; r64; s21504 | 60 | 3.488 | 3.872 | 1.110x | 77.9% | 0.025 | library_gemm (56) | library_gemm (1.128x; 56) |
| 340 | value.score_bias | head_elementwise | head_elementwise; addCBiasInplaceNCKernel; g1x1x1; b6x85x1; r16; s0 | 60 | 0.928 | 1.024 | 1.103x | 96.9% | 0.006 | library_gemm (57) | library_gemm (1.103x; 57) |
| 341 | value.ownership_conv | library_gemm | library_gemm; Kernel2; g8x19x3; b128x1x1; r118; s33792 | 60 | 4.032 | 4.544 | 1.127x | 64.2% | 0.028 | library_gemm (56) | library_gemm (1.127x; 56) |
| 342 | value.ownership_conv_splitk_reduce | library_gemm | library_gemm; splitKreduce_kernel; g147x1x1; b32x16x1; r49; s0 | 60 | 1.376 | 1.440 | 1.047x | 61.4% | 0.014 | library_gemm (30) | idle (1.464x; 5) |
| 343 | value.ownership_half_to_float | copy_reformat | copy_reformat; copyFromHalfKernel; g10x1x1; b512x1x1; r16; s0 | 60 | 0.928 | 1.168 | 1.259x | 31.8% | 0.019 | library_gemm (28) | idle (1.552x; 27) |
