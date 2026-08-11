# Nsys stream interference report

- Timed iterations: 30; streams: 65, 82
- Kernels per forward: 65=344, 82=344
- Iteration start offset stream 82 - 65: median 3.66 us, p10..p90 -3.84..4.49 us, range -8.32..8.70 us.

## Kernel families

| family | instances | total ms | median us | p90 us | peer overlap | S2/S1 median | summed excess ms | matched |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| fused_ffn | 1980 | 91.499 | 46.112 | 50.081 | 78.3% | 1.113x | 8.556 | 1980 |
| linear2_residual | 1980 | 67.809 | 34.113 | 37.440 | 98.2% | 1.620x | 26.099 | 1980 |
| library_gemm | 4140 | 67.351 | 16.224 | 22.624 | 94.9% | 1.697x | 28.945 | 4140 |
| wide_qkv | 1980 | 52.352 | 25.409 | 32.196 | 77.9% | 1.313x | 13.857 | 1980 |
| fa4 | 1980 | 30.461 | 15.232 | 16.256 | 47.2% | 1.264x | 6.627 | 1980 |
| rmsnorm | 3960 | 16.461 | 4.032 | 5.664 | 99.9% | n/a | 0.000 | 0 |
| qk_rope | 1980 | 15.065 | 7.472 | 9.600 | 98.9% | 1.824x | 6.842 | 1980 |
| affine_silu | 1380 | 8.590 | 6.240 | 9.408 | 97.5% | 1.327x | 2.676 | 1380 |
| head_elementwise | 720 | 2.326 | 2.400 | 8.064 | 79.5% | 1.133x | 0.533 | 720 |
| cudnn | 180 | 1.494 | 1.792 | 21.665 | 24.4% | 1.111x | 0.151 | 180 |
| copy_reformat | 300 | 0.544 | 1.728 | 2.496 | 64.8% | 1.176x | 0.115 | 300 |
| sumChannelsNCHWKernel | 60 | 0.113 | 1.856 | 2.115 | 39.8% | 1.137x | 0.015 | 60 |

## Dominant interference pairs

| running family | dominant peer | instances | median overlap | median us | S2/S1 median | matched |
|---|---|---:|---:|---:|---:|---:|
| library_gemm | fused_ffn | 900 | 95.1% | 20.896 | 2.423x | 900 |
| qk_rope | wide_qkv | 990 | 100.0% | 9.376 | 2.256x | 990 |
| head_elementwise | idle | 26 | 0.0% | 2.496 | 2.108x | 26 |
| library_gemm | linear2_residual | 330 | 96.1% | 26.336 | 1.798x | 330 |
| library_gemm | library_gemm | 1807 | 96.5% | 14.752 | 1.699x | 1807 |
| linear2_residual | wide_qkv | 660 | 97.7% | 35.552 | 1.685x | 660 |
| wide_qkv | linear2_residual | 660 | 96.9% | 31.537 | 1.613x | 660 |
| linear2_residual | library_gemm | 330 | 97.8% | 33.648 | 1.603x | 330 |
| linear2_residual | fused_ffn | 990 | 98.7% | 32.833 | 1.558x | 990 |
| library_gemm | affine_silu | 480 | 95.2% | 21.168 | 1.487x | 480 |
| copy_reformat | idle | 58 | 0.0% | 1.408 | 1.467x | 58 |
| library_gemm | cudnn | 49 | 94.2% | 3.776 | 1.439x | 49 |
| wide_qkv | library_gemm | 330 | 67.6% | 27.729 | 1.429x | 330 |
| qk_rope | fa4 | 990 | 100.0% | 5.920 | 1.426x | 990 |
| fa4 | fa4 | 84 | 65.2% | 16.800 | 1.404x | 84 |
| affine_silu | library_gemm | 1037 | 100.0% | 7.168 | 1.394x | 1037 |
| library_gemm | wide_qkv | 328 | 93.5% | 15.904 | 1.320x | 328 |
| fa4 | qk_rope | 948 | 44.1% | 15.329 | 1.274x | 948 |
| affine_silu | linear2_residual | 330 | 100.0% | 4.288 | 1.265x | 330 |
| fa4 | library_gemm | 948 | 47.0% | 15.136 | 1.253x | 948 |
| library_gemm | idle | 5 | 0.0% | 1.696 | 1.232x | 5 |
| library_gemm | head_elementwise | 211 | 85.0% | 6.848 | 1.227x | 211 |
| head_elementwise | copy_reformat | 52 | 72.7% | 3.712 | 1.216x | 52 |
| affine_silu | head_elementwise | 9 | 21.6% | 5.888 | 1.187x | 9 |
| copy_reformat | library_gemm | 148 | 76.4% | 1.824 | 1.184x | 148 |
| wide_qkv | qk_rope | 990 | 65.0% | 23.009 | 1.180x | 990 |
| affine_silu | affine_silu | 4 | 40.7% | 5.824 | 1.174x | 4 |
| sumChannelsNCHWKernel | cudnn | 28 | 38.4% | 1.872 | 1.147x | 28 |
| sumChannelsNCHWKernel | head_elementwise | 26 | 42.8% | 1.872 | 1.147x | 26 |
| cudnn | cudnn | 96 | 62.4% | 1.856 | 1.146x | 96 |
| fused_ffn | fused_ffn | 1015 | 68.5% | 48.353 | 1.144x | 1015 |
| head_elementwise | library_gemm | 522 | 100.0% | 2.208 | 1.133x | 522 |
| copy_reformat | head_elementwise | 68 | 77.8% | 2.096 | 1.132x | 68 |
| head_elementwise | head_elementwise | 62 | 90.7% | 2.400 | 1.129x | 62 |
| cudnn | idle | 15 | 0.0% | 1.696 | 1.125x | 15 |
| cudnn | head_elementwise | 4 | 17.3% | 21.825 | 1.115x | 4 |
| head_elementwise | sumChannelsNCHWKernel | 26 | 59.7% | 1.312 | 1.108x | 26 |
| cudnn | sumChannelsNCHWKernel | 22 | 61.2% | 1.408 | 1.100x | 22 |
| copy_reformat | cudnn | 21 | 63.6% | 1.056 | 1.100x | 21 |
| cudnn | library_gemm | 32 | 31.8% | 21.408 | 1.096x | 32 |

## Logical operation groups

Isolated reference total is the isolated median for each ordinal multiplied by its S2 call count; it is a normalized reference, not a second trace total.

| logical group | families | positions | calls | isolated reference ms | S2 total ms | total ratio | excess ms |
|---|---|---:|---:|---:|---:|---:|---:|
| transformer.ffn_linear2_residual | linear2_residual | 33 | 1980 | 41.710 | 67.809 | 1.626x | 26.099 |
| transformer.attention_out_projection_residual | library_gemm | 33 | 1980 | 17.074 | 35.360 | 2.071x | 18.286 |
| transformer.attention_qkv_projection | wide_qkv | 33 | 1980 | 38.496 | 52.352 | 1.360x | 13.857 |
| transformer.ffn_linear1_gate_swiglu | fused_ffn | 33 | 1980 | 83.026 | 91.499 | 1.102x | 8.556 |
| transformer.attention_qk_rope | qk_rope | 33 | 1980 | 8.222 | 15.065 | 1.832x | 6.842 |
| transformer.attention_fa4 | fa4 | 33 | 1980 | 23.834 | 30.461 | 1.278x | 6.627 |
| outer.post_projection_c384_to_c768_residual | library_gemm | 11 | 660 | 9.604 | 15.843 | 1.650x | 6.239 |
| outer.pre_projection_c768_to_c384 | library_gemm | 11 | 660 | 7.936 | 11.409 | 1.438x | 3.473 |
| outer.pre_norm_silu | affine_silu | 11 | 660 | 3.376 | 5.351 | 1.585x | 1.974 |
| outer.post_norm_silu | affine_silu | 11 | 660 | 2.231 | 2.747 | 1.231x | 0.516 |
| policy.p1_conv | library_gemm | 1 | 60 | 0.374 | 0.674 | 1.801x | 0.300 |
| trunk.tip_norm_silu | affine_silu | 1 | 60 | 0.307 | 0.492 | 1.601x | 0.185 |
| policy.g1_global_pool | head_elementwise | 1 | 60 | 0.271 | 0.442 | 1.633x | 0.171 |
| policy.g1_conv | library_gemm | 1 | 60 | 0.357 | 0.480 | 1.345x | 0.123 |
| frontend.initial_conv | cudnn | 1 | 60 | 1.174 | 1.294 | 1.102x | 0.120 |
| policy.gpool_to_bias_matmul | library_gemm | 1 | 60 | 0.325 | 0.418 | 1.288x | 0.093 |
| value.v1_conv | library_gemm | 1 | 60 | 0.484 | 0.569 | 1.176x | 0.085 |
| frontend.initial_global_broadcast_add | head_elementwise | 1 | 60 | 0.464 | 0.546 | 1.177x | 0.082 |
| value.v1_norm_silu | head_elementwise | 1 | 60 | 0.188 | 0.270 | 1.434x | 0.082 |
| frontend.initial_global_matmul | library_gemm | 1 | 60 | 0.157 | 0.226 | 1.434x | 0.068 |
| policy.p2_conv | library_gemm | 1 | 60 | 0.238 | 0.301 | 1.264x | 0.063 |
| policy.gpool_to_pass_matmul | library_gemm | 1 | 60 | 0.323 | 0.369 | 1.144x | 0.046 |
| policy.gpool_to_pass_matmul2 | library_gemm | 1 | 60 | 0.140 | 0.182 | 1.301x | 0.042 |
| value.v1_half_to_float | copy_reformat | 1 | 60 | 0.131 | 0.173 | 1.323x | 0.042 |
| policy.pass_bias_silu | head_elementwise | 1 | 60 | 0.060 | 0.102 | 1.693x | 0.042 |
| value.v1_global_pool | head_elementwise | 1 | 60 | 0.196 | 0.236 | 1.205x | 0.040 |
| value.v2_matmul | library_gemm | 1 | 60 | 0.574 | 0.612 | 1.066x | 0.038 |
| value.ownership_conv | library_gemm | 1 | 60 | 0.242 | 0.275 | 1.135x | 0.033 |
| policy.g1_norm_silu | head_elementwise | 1 | 60 | 0.127 | 0.159 | 1.257x | 0.033 |
| input.extract_mask | head_elementwise | 1 | 60 | 0.071 | 0.095 | 1.341x | 0.024 |
| policy.p1_norm_silu | head_elementwise | 1 | 60 | 0.132 | 0.156 | 1.176x | 0.023 |
| policy.g1_half_to_float | copy_reformat | 1 | 60 | 0.094 | 0.117 | 1.240x | 0.023 |
| value.score_matmul | library_gemm | 1 | 60 | 0.211 | 0.232 | 1.097x | 0.021 |
| input.mask_half_to_float | copy_reformat | 1 | 60 | 0.058 | 0.076 | 1.324x | 0.019 |
| policy.gpool_bias_add | head_elementwise | 1 | 60 | 0.109 | 0.127 | 1.160x | 0.017 |
| value.ownership_half_to_float | copy_reformat | 1 | 60 | 0.056 | 0.072 | 1.300x | 0.017 |
| frontend.initial_conv_nhwc_padding_0 | cudnn | 1 | 60 | 0.077 | 0.094 | 1.218x | 0.017 |
| value.v3_matmul | library_gemm | 1 | 60 | 0.210 | 0.226 | 1.076x | 0.016 |
| input.mask_sum | sumChannelsNCHWKernel | 1 | 60 | 0.098 | 0.113 | 1.157x | 0.015 |
| policy.p1_half_to_float | copy_reformat | 1 | 60 | 0.092 | 0.106 | 1.155x | 0.014 |
| frontend.initial_conv_nhwc_padding_1 | cudnn | 1 | 60 | 0.092 | 0.106 | 1.154x | 0.014 |
| value.ownership_conv_splitk_reduce | library_gemm | 1 | 60 | 0.083 | 0.093 | 1.128x | 0.011 |
| value.v3_bias | head_elementwise | 1 | 60 | 0.058 | 0.064 | 1.118x | 0.007 |
| value.score_bias | head_elementwise | 1 | 60 | 0.056 | 0.062 | 1.110x | 0.006 |
| frontend.initial_global_matmul_splitk_reduce | library_gemm | 1 | 60 | 0.077 | 0.081 | 1.060x | 0.006 |
| value.v2_bias_silu | head_elementwise | 1 | 60 | 0.061 | 0.066 | 1.076x | 0.005 |
| transformer.ffn_rmsnorm | rmsnorm | 33 | 1980 | 0.000 | 8.239 | n/a | 0.000 |
| transformer.attention_rmsnorm | rmsnorm | 33 | 1980 | 0.000 | 8.223 | n/a | 0.000 |

## `library_gemm` logical breakdown

| logical group | positions | calls | isolated reference ms | S2 total ms | total ratio | excess ms |
|---|---:|---:|---:|---:|---:|---:|
| transformer.attention_out_projection_residual | 33 | 1980 | 17.074 | 35.360 | 2.071x | 18.286 |
| outer.post_projection_c384_to_c768_residual | 11 | 660 | 9.604 | 15.843 | 1.650x | 6.239 |
| outer.pre_projection_c768_to_c384 | 11 | 660 | 7.936 | 11.409 | 1.438x | 3.473 |
| policy.p1_conv | 1 | 60 | 0.374 | 0.674 | 1.801x | 0.300 |
| policy.g1_conv | 1 | 60 | 0.357 | 0.480 | 1.345x | 0.123 |
| policy.gpool_to_bias_matmul | 1 | 60 | 0.325 | 0.418 | 1.288x | 0.093 |
| value.v1_conv | 1 | 60 | 0.484 | 0.569 | 1.176x | 0.085 |
| frontend.initial_global_matmul | 1 | 60 | 0.157 | 0.226 | 1.434x | 0.068 |
| policy.p2_conv | 1 | 60 | 0.238 | 0.301 | 1.264x | 0.063 |
| policy.gpool_to_pass_matmul | 1 | 60 | 0.323 | 0.369 | 1.144x | 0.046 |
| policy.gpool_to_pass_matmul2 | 1 | 60 | 0.140 | 0.182 | 1.301x | 0.042 |
| value.v2_matmul | 1 | 60 | 0.574 | 0.612 | 1.066x | 0.038 |
| value.ownership_conv | 1 | 60 | 0.242 | 0.275 | 1.135x | 0.033 |
| value.score_matmul | 1 | 60 | 0.211 | 0.232 | 1.097x | 0.021 |
| value.v3_matmul | 1 | 60 | 0.210 | 0.226 | 1.076x | 0.016 |
| value.ownership_conv_splitk_reduce | 1 | 60 | 0.083 | 0.093 | 1.128x | 0.011 |
| frontend.initial_global_matmul_splitk_reduce | 1 | 60 | 0.077 | 0.081 | 1.060x | 0.006 |

## Top ordinal hotspots by summed excess

The worst peer is the highest median S2/S1 slowdown among peer families observed at least four times for that ordinal.

| rank | ordinal | logical position | family | calls | isolated us | S2 us | S2/S1 | excess ms | common peer | worst peer |
|---:|---:|---|---|---:|---:|---:|---:|---:|---|---|
| 1 | 214 | outer_07.transformer_0.block_21.ffn_linear2_residual | linear2_residual | 60 | 21.265 | 37.856 | 1.780x | 1.010 | fused_ffn (30) | wide_qkv (1.851x; 30) |
| 2 | 222 | outer_07.transformer_1.block_22.ffn_linear2_residual | linear2_residual | 60 | 21.265 | 37.633 | 1.770x | 0.990 | fused_ffn (30) | wide_qkv (1.844x; 30) |
| 3 | 202 | outer_06.transformer_2.block_20.ffn_linear2_residual | linear2_residual | 60 | 21.184 | 37.313 | 1.761x | 0.968 | fused_ffn (30) | library_gemm (1.820x; 30) |
| 4 | 230 | outer_07.transformer_2.block_23.ffn_linear2_residual | linear2_residual | 60 | 21.280 | 37.328 | 1.754x | 0.963 | fused_ffn (30) | library_gemm (1.771x; 30) |
| 5 | 146 | outer_04.transformer_2.block_14.ffn_linear2_residual | linear2_residual | 60 | 21.152 | 36.705 | 1.735x | 0.958 | fused_ffn (30) | library_gemm (1.848x; 30) |
| 6 | 298 | outer_10.transformer_0.block_30.ffn_linear2_residual | linear2_residual | 60 | 21.152 | 36.368 | 1.719x | 0.926 | fused_ffn (30) | wide_qkv (1.757x; 30) |
| 7 | 138 | outer_04.transformer_1.block_13.ffn_linear2_residual | linear2_residual | 60 | 21.073 | 35.968 | 1.707x | 0.893 | fused_ffn (30) | wide_qkv (1.749x; 30) |
| 8 | 306 | outer_10.transformer_1.block_31.ffn_linear2_residual | linear2_residual | 60 | 21.152 | 35.425 | 1.675x | 0.873 | fused_ffn (30) | wide_qkv (1.747x; 30) |
| 9 | 242 | outer_08.transformer_0.block_24.ffn_linear2_residual | linear2_residual | 60 | 21.265 | 35.441 | 1.667x | 0.850 | fused_ffn (30) | wide_qkv (1.692x; 30) |
| 10 | 130 | outer_04.transformer_0.block_12.ffn_linear2_residual | linear2_residual | 60 | 21.121 | 35.008 | 1.658x | 0.841 | fused_ffn (30) | wide_qkv (1.749x; 30) |
| 11 | 194 | outer_06.transformer_1.block_19.ffn_linear2_residual | linear2_residual | 60 | 21.169 | 34.992 | 1.653x | 0.823 | fused_ffn (30) | fused_ffn (1.658x; 30) |
| 12 | 46 | outer_01.transformer_0.block_03.ffn_linear2_residual | linear2_residual | 60 | 21.088 | 34.432 | 1.633x | 0.811 | fused_ffn (30) | wide_qkv (1.746x; 30) |
| 13 | 250 | outer_08.transformer_1.block_25.ffn_linear2_residual | linear2_residual | 60 | 21.216 | 34.864 | 1.643x | 0.803 | fused_ffn (30) | wide_qkv (1.690x; 30) |
| 14 | 74 | outer_02.transformer_0.block_06.ffn_linear2_residual | linear2_residual | 60 | 21.088 | 34.048 | 1.615x | 0.796 | fused_ffn (30) | wide_qkv (1.728x; 30) |
| 15 | 314 | outer_10.transformer_2.block_32.ffn_linear2_residual | linear2_residual | 60 | 21.120 | 34.273 | 1.623x | 0.790 | fused_ffn (30) | fused_ffn (1.630x; 30) |
| 16 | 82 | outer_02.transformer_1.block_07.ffn_linear2_residual | linear2_residual | 60 | 21.008 | 33.809 | 1.609x | 0.781 | fused_ffn (30) | wide_qkv (1.695x; 30) |
| 17 | 158 | outer_05.transformer_0.block_15.ffn_linear2_residual | linear2_residual | 60 | 21.216 | 34.080 | 1.606x | 0.777 | fused_ffn (30) | wide_qkv (1.659x; 30) |
| 18 | 258 | outer_08.transformer_2.block_26.ffn_linear2_residual | linear2_residual | 60 | 21.200 | 34.064 | 1.607x | 0.772 | fused_ffn (30) | library_gemm (1.611x; 30) |
| 19 | 18 | outer_00.transformer_0.block_00.ffn_linear2_residual | linear2_residual | 60 | 21.120 | 33.585 | 1.590x | 0.766 | fused_ffn (30) | wide_qkv (1.685x; 30) |
| 20 | 186 | outer_06.transformer_0.block_18.ffn_linear2_residual | linear2_residual | 60 | 21.200 | 33.856 | 1.597x | 0.762 | fused_ffn (30) | wide_qkv (1.614x; 30) |
| 21 | 54 | outer_01.transformer_1.block_04.ffn_linear2_residual | linear2_residual | 60 | 20.944 | 33.392 | 1.594x | 0.751 | fused_ffn (30) | wide_qkv (1.677x; 30) |
| 22 | 166 | outer_05.transformer_1.block_16.ffn_linear2_residual | linear2_residual | 60 | 21.025 | 33.296 | 1.584x | 0.747 | fused_ffn (30) | wide_qkv (1.664x; 30) |
| 23 | 90 | outer_02.transformer_2.block_08.ffn_linear2_residual | linear2_residual | 60 | 21.056 | 32.993 | 1.567x | 0.722 | fused_ffn (30) | library_gemm (1.599x; 30) |
| 24 | 270 | outer_09.transformer_0.block_27.ffn_linear2_residual | linear2_residual | 60 | 21.056 | 32.576 | 1.547x | 0.712 | fused_ffn (30) | wide_qkv (1.642x; 30) |
| 25 | 62 | outer_01.transformer_2.block_05.ffn_linear2_residual | linear2_residual | 60 | 20.960 | 32.513 | 1.551x | 0.703 | fused_ffn (30) | library_gemm (1.585x; 30) |
| 26 | 26 | outer_00.transformer_1.block_01.ffn_linear2_residual | linear2_residual | 60 | 20.945 | 32.464 | 1.550x | 0.697 | fused_ffn (30) | wide_qkv (1.624x; 30) |
| 27 | 316 | outer_10.post_projection_c384_to_c768_residual | library_gemm | 60 | 14.624 | 25.713 | 1.758x | 0.686 | linear2_residual (30) | linear2_residual (1.927x; 30) |
| 28 | 174 | outer_05.transformer_2.block_17.ffn_linear2_residual | linear2_residual | 60 | 20.992 | 32.336 | 1.540x | 0.680 | fused_ffn (30) | library_gemm (1.566x; 30) |
| 29 | 102 | outer_03.transformer_0.block_09.ffn_linear2_residual | linear2_residual | 60 | 20.928 | 31.680 | 1.514x | 0.673 | fused_ffn (30) | wide_qkv (1.617x; 30) |
| 30 | 110 | outer_03.transformer_1.block_10.ffn_linear2_residual | linear2_residual | 60 | 20.704 | 31.632 | 1.528x | 0.671 | fused_ffn (30) | wide_qkv (1.621x; 30) |
| 31 | 278 | outer_09.transformer_1.block_28.ffn_linear2_residual | linear2_residual | 60 | 20.848 | 31.552 | 1.513x | 0.671 | fused_ffn (30) | wide_qkv (1.622x; 30) |
| 32 | 34 | outer_00.transformer_2.block_02.ffn_linear2_residual | linear2_residual | 60 | 20.865 | 31.921 | 1.530x | 0.669 | fused_ffn (30) | library_gemm (1.566x; 30) |
| 33 | 232 | outer_07.post_projection_c384_to_c768_residual | library_gemm | 60 | 14.720 | 25.137 | 1.708x | 0.643 | linear2_residual (30) | linear2_residual (1.911x; 30) |
| 34 | 286 | outer_09.transformer_2.block_29.ffn_linear2_residual | linear2_residual | 60 | 20.736 | 31.120 | 1.501x | 0.639 | fused_ffn (30) | library_gemm (1.557x; 30) |
| 35 | 260 | outer_08.post_projection_c384_to_c768_residual | library_gemm | 60 | 14.624 | 24.960 | 1.707x | 0.633 | affine_silu (30) | linear2_residual (1.926x; 30) |
| 36 | 204 | outer_06.post_projection_c384_to_c768_residual | library_gemm | 60 | 14.688 | 24.337 | 1.657x | 0.616 | affine_silu (30) | linear2_residual (1.865x; 30) |
| 37 | 148 | outer_04.post_projection_c384_to_c768_residual | library_gemm | 60 | 14.736 | 24.928 | 1.692x | 0.615 | affine_silu (30) | linear2_residual (1.845x; 30) |
| 38 | 118 | outer_03.transformer_2.block_11.ffn_linear2_residual | linear2_residual | 60 | 20.768 | 30.832 | 1.485x | 0.611 | fused_ffn (30) | library_gemm (1.510x; 30) |
| 39 | 99 | outer_03.transformer_0.block_09.attention_out_projection_residual | library_gemm | 60 | 8.576 | 18.224 | 2.125x | 0.590 | library_gemm (35) | fused_ffn (2.489x; 25) |
| 40 | 43 | outer_01.transformer_0.block_03.attention_out_projection_residual | library_gemm | 60 | 8.512 | 18.256 | 2.145x | 0.590 | library_gemm (39) | fused_ffn (2.500x; 21) |

## Full fixed-forward ordinal map

| ordinal | logical position | family | resource signature | calls | isolated us | S2 us | S2/S1 | overlap | excess ms | common peer | worst peer |
|---:|---|---|---|---:|---:|---:|---:|---:|---:|---|---|
| 0 | input.extract_mask | head_elementwise | head_elementwise; extractChannel0KernelNHWC; g10x1x1; b512x1x1; r16; s0 | 60 | 1.184 | 1.328 | 1.122x | 27.0% | 0.024 | idle (26) | idle (2.108x; 26) |
| 1 | input.mask_half_to_float | copy_reformat | copy_reformat; copyFromHalfKernel; g10x1x1; b512x1x1; r16; s0 | 60 | 0.960 | 1.152 | 1.200x | 20.3% | 0.019 | idle (34) | idle (1.467x; 34) |
| 2 | input.mask_sum | sumChannelsNCHWKernel | sumChannelsNCHWKernel; sumChannelsNCHWKernel; g1x1x13; b256x2x1; r22; s2048 | 60 | 1.632 | 1.856 | 1.137x | 39.8% | 0.015 | cudnn (28) | cudnn (1.147x; 28) |
| 3 | frontend.initial_conv_nhwc_padding_0 | cudnn | cudnn; nhwcAddPaddingKernel; g340x1x1; b768x1x1; r34; s0 | 60 | 1.280 | 1.456 | 1.138x | 65.7% | 0.017 | cudnn (28) | cudnn (1.288x; 28) |
| 4 | frontend.initial_conv_nhwc_padding_1 | cudnn | cudnn; nhwcAddPaddingKernel; g340x1x1; b768x1x1; r34; s0 | 60 | 1.536 | 1.760 | 1.146x | 56.1% | 0.014 | cudnn (40) | cudnn (1.167x; 40) |
| 5 | frontend.initial_conv | cudnn | cudnn; Kernel; g296x3x1; b128x1x1; r94; s81920 | 60 | 19.569 | 21.520 | 1.100x | 18.8% | 0.120 | cudnn (28) | cudnn (1.101x; 28) |
| 6 | frontend.initial_global_matmul | library_gemm | library_gemm; Kernel2; g8x1x3; b128x1x1; r128; s24576 | 60 | 2.624 | 3.696 | 1.409x | 86.9% | 0.068 | cudnn (30) | cudnn (1.555x; 30) |
| 7 | frontend.initial_global_matmul_splitk_reduce | library_gemm | library_gemm; splitKreduce_kernel; g24x1x1; b32x16x1; r49; s0 | 60 | 1.280 | 1.360 | 1.062x | 82.4% | 0.006 | affine_silu (19) | cudnn (1.125x; 19) |
| 8 | frontend.initial_global_broadcast_add | head_elementwise | head_elementwise; addNCBiasInplaceNHWCHalfKernel; g3x361x13; b256x1x1; r16; s0 | 60 | 7.729 | 8.160 | 1.056x | 54.8% | 0.082 | library_gemm (29) | library_gemm (1.428x; 29) |
| 9 | outer_00.pre_norm_silu | affine_silu | affine_silu; affineSiluHalf2Kernel; g4693x1x1; b384x1x1; r17; s0 | 60 | 4.960 | 6.240 | 1.258x | 60.0% | 0.092 | library_gemm (47) | library_gemm (1.458x; 47) |
| 10 | outer_00.pre_projection_c768_to_c384 | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 60 | 11.680 | 14.592 | 1.249x | 94.6% | 0.219 | wide_qkv (28) | library_gemm (1.664x; 4) |
| 11 | outer_00.transformer_0.block_00.attention_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | n/a | 3.648 | n/a | 95.8% | 0.000 | wide_qkv (30) | n/a |
| 12 | outer_00.transformer_0.block_00.attention_qkv_projection | wide_qkv | wide_qkv; wide_qkv_kernel; g9x37x1; b128x1x1; r136; s65536 | 60 | 18.784 | 22.752 | 1.211x | 70.7% | 0.337 | library_gemm (30) | library_gemm (1.532x; 30) |
| 13 | outer_00.transformer_0.block_00.attention_qk_rope | qk_rope | qk_rope; fusedQKRoPE19HalfKernel; g361x13x1; b192x1x1; r16; s0 | 60 | 4.064 | 7.328 | 1.803x | 93.3% | 0.194 | fa4 (30) | wide_qkv (2.252x; 30) |
| 14 | outer_00.transformer_0.block_00.attention_fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 60 | 11.777 | 14.704 | 1.249x | 53.1% | 0.230 | library_gemm (22) | fa4 (1.395x; 16) |
| 15 | outer_00.transformer_0.block_00.attention_out_projection_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 60 | 8.576 | 16.496 | 1.923x | 95.7% | 0.510 | library_gemm (31) | fused_ffn (2.362x; 29) |
| 16 | outer_00.transformer_0.block_00.ffn_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | n/a | 3.824 | n/a | 100.0% | 0.000 | fused_ffn (30) | n/a |
| 17 | outer_00.transformer_0.block_00.ffn_linear1_gate_swiglu | fused_ffn | fused_ffn; fused_ffn_kernel; g18x37x1; b128x1x1; r146; s32768 | 60 | 41.409 | 45.665 | 1.103x | 77.4% | 0.260 | fused_ffn (30) | fused_ffn (1.136x; 30) |
| 18 | outer_00.transformer_0.block_00.ffn_linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 60 | 21.120 | 33.585 | 1.590x | 98.0% | 0.766 | fused_ffn (30) | wide_qkv (1.685x; 30) |
| 19 | outer_00.transformer_1.block_01.attention_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | n/a | 3.936 | n/a | 100.0% | 0.000 | linear2_residual (30) | n/a |
| 20 | outer_00.transformer_1.block_01.attention_qkv_projection | wide_qkv | wide_qkv; wide_qkv_kernel; g9x37x1; b128x1x1; r136; s65536 | 60 | 19.536 | 26.753 | 1.369x | 83.6% | 0.447 | linear2_residual (30) | linear2_residual (1.615x; 30) |
| 21 | outer_00.transformer_1.block_01.attention_qk_rope | qk_rope | qk_rope; fusedQKRoPE19HalfKernel; g361x13x1; b192x1x1; r16; s0 | 60 | 4.160 | 7.472 | 1.796x | 100.0% | 0.206 | fa4 (30) | wide_qkv (2.254x; 30) |
| 22 | outer_00.transformer_1.block_01.attention_fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 60 | 12.064 | 14.864 | 1.232x | 44.5% | 0.169 | library_gemm (30) | library_gemm (1.233x; 30) |
| 23 | outer_00.transformer_1.block_01.attention_out_projection_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 60 | 8.704 | 17.312 | 1.989x | 95.8% | 0.534 | fused_ffn (30) | fused_ffn (2.386x; 30) |
| 24 | outer_00.transformer_1.block_01.ffn_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | n/a | 4.128 | n/a | 100.0% | 0.000 | fused_ffn (30) | n/a |
| 25 | outer_00.transformer_1.block_01.ffn_linear1_gate_swiglu | fused_ffn | fused_ffn; fused_ffn_kernel; g18x37x1; b128x1x1; r146; s32768 | 60 | 43.104 | 45.920 | 1.065x | 75.9% | 0.160 | fused_ffn (30) | fused_ffn (1.120x; 30) |
| 26 | outer_00.transformer_1.block_01.ffn_linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 60 | 20.945 | 32.464 | 1.550x | 98.3% | 0.697 | fused_ffn (30) | wide_qkv (1.624x; 30) |
| 27 | outer_00.transformer_2.block_02.attention_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | n/a | 3.712 | n/a | 100.0% | 0.000 | linear2_residual (30) | n/a |
| 28 | outer_00.transformer_2.block_02.attention_qkv_projection | wide_qkv | wide_qkv; wide_qkv_kernel; g9x37x1; b128x1x1; r136; s65536 | 60 | 19.296 | 24.881 | 1.289x | 83.3% | 0.367 | linear2_residual (30) | linear2_residual (1.488x; 30) |
| 29 | outer_00.transformer_2.block_02.attention_qk_rope | qk_rope | qk_rope; fusedQKRoPE19HalfKernel; g361x13x1; b192x1x1; r16; s0 | 60 | 4.160 | 7.521 | 1.808x | 100.0% | 0.203 | fa4 (30) | wide_qkv (2.238x; 30) |
| 30 | outer_00.transformer_2.block_02.attention_fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 60 | 11.936 | 14.688 | 1.231x | 44.1% | 0.169 | library_gemm (30) | library_gemm (1.233x; 30) |
| 31 | outer_00.transformer_2.block_02.attention_out_projection_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 60 | 8.592 | 17.504 | 2.037x | 95.7% | 0.540 | fused_ffn (30) | fused_ffn (2.413x; 30) |
| 32 | outer_00.transformer_2.block_02.ffn_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | n/a | 4.160 | n/a | 100.0% | 0.000 | fused_ffn (30) | n/a |
| 33 | outer_00.transformer_2.block_02.ffn_linear1_gate_swiglu | fused_ffn | fused_ffn; fused_ffn_kernel; g18x37x1; b128x1x1; r146; s32768 | 60 | 42.785 | 45.937 | 1.074x | 75.7% | 0.180 | fused_ffn (30) | fused_ffn (1.125x; 30) |
| 34 | outer_00.transformer_2.block_02.ffn_linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 60 | 20.865 | 31.921 | 1.530x | 98.3% | 0.669 | fused_ffn (30) | library_gemm (1.566x; 30) |
| 35 | outer_00.post_norm_silu | affine_silu | affine_silu; affineSiluHalf2Kernel; g4693x1x1; b192x1x1; r17; s0 | 60 | 3.360 | 4.032 | 1.200x | 100.0% | 0.041 | library_gemm (30) | linear2_residual (1.229x; 30) |
| 36 | outer_00.post_projection_c384_to_c768_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b256x1x1; r154; s73728 | 60 | 14.416 | 22.160 | 1.537x | 95.7% | 0.482 | affine_silu (30) | linear2_residual (1.650x; 30) |
| 37 | outer_01.pre_norm_silu | affine_silu | affine_silu; affineSiluHalf2Kernel; g4693x1x1; b384x1x1; r17; s0 | 60 | 5.152 | 8.064 | 1.565x | 100.0% | 0.181 | library_gemm (60) | library_gemm (1.565x; 60) |
| 38 | outer_01.pre_projection_c768_to_c384 | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 60 | 12.000 | 16.048 | 1.337x | 94.1% | 0.299 | wide_qkv (30) | library_gemm (1.581x; 20) |
| 39 | outer_01.transformer_0.block_03.attention_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | n/a | 4.048 | n/a | 100.0% | 0.000 | library_gemm (30) | n/a |
| 40 | outer_01.transformer_0.block_03.attention_qkv_projection | wide_qkv | wide_qkv; wide_qkv_kernel; g9x37x1; b128x1x1; r136; s65536 | 60 | 19.296 | 25.761 | 1.335x | 66.7% | 0.356 | library_gemm (30) | library_gemm (1.406x; 30) |
| 41 | outer_01.transformer_0.block_03.attention_qk_rope | qk_rope | qk_rope; fusedQKRoPE19HalfKernel; g361x13x1; b192x1x1; r16; s0 | 60 | 4.096 | 7.632 | 1.863x | 100.0% | 0.211 | fa4 (30) | wide_qkv (2.274x; 30) |
| 42 | outer_01.transformer_0.block_03.attention_fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 60 | 12.000 | 15.680 | 1.307x | 47.3% | 0.221 | library_gemm (30) | qk_rope (1.324x; 30) |
| 43 | outer_01.transformer_0.block_03.attention_out_projection_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 60 | 8.512 | 18.256 | 2.145x | 95.1% | 0.590 | library_gemm (39) | fused_ffn (2.500x; 21) |
| 44 | outer_01.transformer_0.block_03.ffn_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | n/a | 4.144 | n/a | 100.0% | 0.000 | fused_ffn (30) | n/a |
| 45 | outer_01.transformer_0.block_03.ffn_linear1_gate_swiglu | fused_ffn | fused_ffn; fused_ffn_kernel; g18x37x1; b128x1x1; r146; s32768 | 60 | 42.097 | 45.344 | 1.077x | 76.3% | 0.243 | fused_ffn (30) | fused_ffn (1.156x; 30) |
| 46 | outer_01.transformer_0.block_03.ffn_linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 60 | 21.088 | 34.432 | 1.633x | 98.0% | 0.811 | fused_ffn (30) | wide_qkv (1.746x; 30) |
| 47 | outer_01.transformer_1.block_04.attention_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | n/a | 3.857 | n/a | 100.0% | 0.000 | linear2_residual (30) | n/a |
| 48 | outer_01.transformer_1.block_04.attention_qkv_projection | wide_qkv | wide_qkv; wide_qkv_kernel; g9x37x1; b128x1x1; r136; s65536 | 60 | 19.473 | 26.528 | 1.362x | 83.2% | 0.450 | linear2_residual (30) | linear2_residual (1.605x; 30) |
| 49 | outer_01.transformer_1.block_04.attention_qk_rope | qk_rope | qk_rope; fusedQKRoPE19HalfKernel; g361x13x1; b192x1x1; r16; s0 | 60 | 4.160 | 7.520 | 1.808x | 100.0% | 0.207 | fa4 (30) | wide_qkv (2.269x; 30) |
| 50 | outer_01.transformer_1.block_04.attention_fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 60 | 12.064 | 15.168 | 1.257x | 46.1% | 0.188 | library_gemm (30) | qk_rope (1.272x; 30) |
| 51 | outer_01.transformer_1.block_04.attention_out_projection_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 60 | 8.576 | 17.536 | 2.045x | 95.7% | 0.554 | fused_ffn (30) | fused_ffn (2.442x; 30) |
| 52 | outer_01.transformer_1.block_04.ffn_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | n/a | 4.240 | n/a | 100.0% | 0.000 | fused_ffn (30) | n/a |
| 53 | outer_01.transformer_1.block_04.ffn_linear1_gate_swiglu | fused_ffn | fused_ffn; fused_ffn_kernel; g18x37x1; b128x1x1; r146; s32768 | 60 | 43.200 | 46.769 | 1.083x | 75.7% | 0.202 | fused_ffn (30) | fused_ffn (1.134x; 30) |
| 54 | outer_01.transformer_1.block_04.ffn_linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 60 | 20.944 | 33.392 | 1.594x | 98.4% | 0.751 | fused_ffn (30) | wide_qkv (1.677x; 30) |
| 55 | outer_01.transformer_2.block_05.attention_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | n/a | 3.792 | n/a | 100.0% | 0.000 | linear2_residual (30) | n/a |
| 56 | outer_01.transformer_2.block_05.attention_qkv_projection | wide_qkv | wide_qkv; wide_qkv_kernel; g9x37x1; b128x1x1; r136; s65536 | 60 | 19.281 | 25.360 | 1.315x | 83.3% | 0.383 | linear2_residual (30) | linear2_residual (1.526x; 30) |
| 57 | outer_01.transformer_2.block_05.attention_qk_rope | qk_rope | qk_rope; fusedQKRoPE19HalfKernel; g361x13x1; b192x1x1; r16; s0 | 60 | 4.096 | 7.584 | 1.852x | 95.9% | 0.210 | fa4 (30) | wide_qkv (2.312x; 30) |
| 58 | outer_01.transformer_2.block_05.attention_fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 60 | 11.968 | 14.864 | 1.242x | 47.5% | 0.193 | library_gemm (25) | fa4 (1.401x; 10) |
| 59 | outer_01.transformer_2.block_05.attention_out_projection_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 60 | 8.528 | 15.408 | 1.807x | 95.7% | 0.528 | library_gemm (33) | fused_ffn (2.428x; 27) |
| 60 | outer_01.transformer_2.block_05.ffn_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | n/a | 4.288 | n/a | 100.0% | 0.000 | fused_ffn (30) | n/a |
| 61 | outer_01.transformer_2.block_05.ffn_linear1_gate_swiglu | fused_ffn | fused_ffn; fused_ffn_kernel; g18x37x1; b128x1x1; r146; s32768 | 60 | 42.961 | 47.361 | 1.102x | 75.5% | 0.246 | fused_ffn (33) | fused_ffn (1.145x; 33) |
| 62 | outer_01.transformer_2.block_05.ffn_linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 60 | 20.960 | 32.513 | 1.551x | 98.3% | 0.703 | fused_ffn (30) | library_gemm (1.585x; 30) |
| 63 | outer_01.post_norm_silu | affine_silu | affine_silu; affineSiluHalf2Kernel; g4693x1x1; b192x1x1; r17; s0 | 60 | 3.360 | 4.128 | 1.229x | 100.0% | 0.045 | library_gemm (30) | linear2_residual (1.248x; 30) |
| 64 | outer_01.post_projection_c384_to_c768_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b256x1x1; r154; s73728 | 60 | 14.480 | 22.721 | 1.569x | 95.8% | 0.524 | affine_silu (30) | linear2_residual (1.724x; 30) |
| 65 | outer_02.pre_norm_silu | affine_silu | affine_silu; affineSiluHalf2Kernel; g4693x1x1; b384x1x1; r17; s0 | 60 | 5.120 | 8.192 | 1.600x | 100.0% | 0.182 | library_gemm (60) | library_gemm (1.600x; 60) |
| 66 | outer_02.pre_projection_c768_to_c384 | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 60 | 12.001 | 16.512 | 1.376x | 94.1% | 0.313 | wide_qkv (30) | library_gemm (1.604x; 18) |
| 67 | outer_02.transformer_0.block_06.attention_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | n/a | 4.112 | n/a | 100.0% | 0.000 | library_gemm (30) | n/a |
| 68 | outer_02.transformer_0.block_06.attention_qkv_projection | wide_qkv | wide_qkv; wide_qkv_kernel; g9x37x1; b128x1x1; r136; s65536 | 60 | 19.312 | 26.177 | 1.355x | 66.8% | 0.395 | library_gemm (30) | library_gemm (1.435x; 30) |
| 69 | outer_02.transformer_0.block_06.attention_qk_rope | qk_rope | qk_rope; fusedQKRoPE19HalfKernel; g361x13x1; b192x1x1; r16; s0 | 60 | 4.160 | 7.520 | 1.807x | 100.0% | 0.210 | fa4 (30) | wide_qkv (2.257x; 30) |
| 70 | outer_02.transformer_0.block_06.attention_fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 60 | 12.016 | 15.936 | 1.326x | 49.2% | 0.236 | library_gemm (30) | qk_rope (1.350x; 30) |
| 71 | outer_02.transformer_0.block_06.attention_out_projection_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 60 | 8.640 | 18.080 | 2.093x | 95.9% | 0.577 | library_gemm (38) | fused_ffn (2.439x; 22) |
| 72 | outer_02.transformer_0.block_06.ffn_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | n/a | 4.064 | n/a | 100.0% | 0.000 | fused_ffn (30) | n/a |
| 73 | outer_02.transformer_0.block_06.ffn_linear1_gate_swiglu | fused_ffn | fused_ffn; fused_ffn_kernel; g18x37x1; b128x1x1; r146; s32768 | 60 | 43.105 | 47.344 | 1.098x | 75.1% | 0.274 | fused_ffn (32) | fused_ffn (1.176x; 32) |
| 74 | outer_02.transformer_0.block_06.ffn_linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 60 | 21.088 | 34.048 | 1.615x | 98.0% | 0.796 | fused_ffn (30) | wide_qkv (1.728x; 30) |
| 75 | outer_02.transformer_1.block_07.attention_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | n/a | 3.937 | n/a | 100.0% | 0.000 | linear2_residual (30) | n/a |
| 76 | outer_02.transformer_1.block_07.attention_qkv_projection | wide_qkv | wide_qkv; wide_qkv_kernel; g9x37x1; b128x1x1; r136; s65536 | 60 | 19.520 | 26.080 | 1.336x | 83.4% | 0.418 | linear2_residual (30) | linear2_residual (1.579x; 30) |
| 77 | outer_02.transformer_1.block_07.attention_qk_rope | qk_rope | qk_rope; fusedQKRoPE19HalfKernel; g361x13x1; b192x1x1; r16; s0 | 60 | 4.192 | 7.457 | 1.779x | 100.0% | 0.203 | fa4 (30) | wide_qkv (2.202x; 30) |
| 78 | outer_02.transformer_1.block_07.attention_fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 60 | 12.032 | 14.992 | 1.246x | 44.9% | 0.177 | library_gemm (30) | library_gemm (1.249x; 30) |
| 79 | outer_02.transformer_1.block_07.attention_out_projection_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 60 | 8.624 | 17.953 | 2.082x | 95.8% | 0.553 | library_gemm (32) | fused_ffn (2.423x; 28) |
| 80 | outer_02.transformer_1.block_07.ffn_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | n/a | 4.128 | n/a | 100.0% | 0.000 | fused_ffn (30) | n/a |
| 81 | outer_02.transformer_1.block_07.ffn_linear1_gate_swiglu | fused_ffn | fused_ffn; fused_ffn_kernel; g18x37x1; b128x1x1; r146; s32768 | 60 | 43.472 | 47.840 | 1.100x | 77.9% | 0.283 | fused_ffn (31) | fused_ffn (1.155x; 31) |
| 82 | outer_02.transformer_1.block_07.ffn_linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 60 | 21.008 | 33.809 | 1.609x | 98.3% | 0.781 | fused_ffn (30) | wide_qkv (1.695x; 30) |
| 83 | outer_02.transformer_2.block_08.attention_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | n/a | 3.840 | n/a | 100.0% | 0.000 | linear2_residual (30) | n/a |
| 84 | outer_02.transformer_2.block_08.attention_qkv_projection | wide_qkv | wide_qkv; wide_qkv_kernel; g9x37x1; b128x1x1; r136; s65536 | 60 | 19.488 | 26.464 | 1.358x | 83.5% | 0.452 | linear2_residual (30) | linear2_residual (1.618x; 30) |
| 85 | outer_02.transformer_2.block_08.attention_qk_rope | qk_rope | qk_rope; fusedQKRoPE19HalfKernel; g361x13x1; b192x1x1; r16; s0 | 60 | 4.160 | 7.632 | 1.835x | 99.2% | 0.207 | fa4 (30) | wide_qkv (2.273x; 30) |
| 86 | outer_02.transformer_2.block_08.attention_fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 60 | 12.064 | 15.024 | 1.245x | 45.6% | 0.184 | library_gemm (29) | qk_rope (1.249x; 29) |
| 87 | outer_02.transformer_2.block_08.attention_out_projection_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 60 | 8.640 | 17.296 | 2.002x | 95.7% | 0.538 | library_gemm (31) | fused_ffn (2.374x; 29) |
| 88 | outer_02.transformer_2.block_08.ffn_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | n/a | 4.240 | n/a | 100.0% | 0.000 | fused_ffn (30) | n/a |
| 89 | outer_02.transformer_2.block_08.ffn_linear1_gate_swiglu | fused_ffn | fused_ffn; fused_ffn_kernel; g18x37x1; b128x1x1; r146; s32768 | 60 | 43.552 | 46.849 | 1.076x | 76.5% | 0.241 | fused_ffn (30) | fused_ffn (1.142x; 30) |
| 90 | outer_02.transformer_2.block_08.ffn_linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 60 | 21.056 | 32.993 | 1.567x | 98.3% | 0.722 | fused_ffn (30) | library_gemm (1.599x; 30) |
| 91 | outer_02.post_norm_silu | affine_silu | affine_silu; affineSiluHalf2Kernel; g4693x1x1; b192x1x1; r17; s0 | 60 | 3.392 | 4.128 | 1.217x | 100.0% | 0.045 | library_gemm (30) | linear2_residual (1.264x; 30) |
| 92 | outer_02.post_projection_c384_to_c768_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b256x1x1; r154; s73728 | 60 | 14.560 | 23.680 | 1.626x | 95.6% | 0.572 | affine_silu (30) | linear2_residual (1.818x; 30) |
| 93 | outer_03.pre_norm_silu | affine_silu | affine_silu; affineSiluHalf2Kernel; g4693x1x1; b384x1x1; r17; s0 | 60 | 5.152 | 8.256 | 1.602x | 100.0% | 0.186 | library_gemm (60) | library_gemm (1.602x; 60) |
| 94 | outer_03.pre_projection_c768_to_c384 | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 60 | 12.064 | 16.800 | 1.393x | 94.2% | 0.315 | wide_qkv (30) | library_gemm (1.576x; 14) |
| 95 | outer_03.transformer_0.block_09.attention_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | n/a | 4.096 | n/a | 100.0% | 0.000 | library_gemm (30) | n/a |
| 96 | outer_03.transformer_0.block_09.attention_qkv_projection | wide_qkv | wide_qkv; wide_qkv_kernel; g9x37x1; b128x1x1; r136; s65536 | 60 | 19.440 | 25.729 | 1.323x | 66.7% | 0.394 | library_gemm (30) | library_gemm (1.433x; 30) |
| 97 | outer_03.transformer_0.block_09.attention_qk_rope | qk_rope | qk_rope; fusedQKRoPE19HalfKernel; g361x13x1; b192x1x1; r16; s0 | 60 | 4.192 | 7.632 | 1.821x | 100.0% | 0.210 | fa4 (30) | wide_qkv (2.229x; 30) |
| 98 | outer_03.transformer_0.block_09.attention_fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 60 | 12.081 | 16.016 | 1.326x | 49.0% | 0.238 | library_gemm (30) | qk_rope (1.347x; 30) |
| 99 | outer_03.transformer_0.block_09.attention_out_projection_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 60 | 8.576 | 18.224 | 2.125x | 95.5% | 0.590 | library_gemm (35) | fused_ffn (2.489x; 25) |
| 100 | outer_03.transformer_0.block_09.ffn_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | n/a | 4.256 | n/a | 100.0% | 0.000 | fused_ffn (30) | n/a |
| 101 | outer_03.transformer_0.block_09.ffn_linear1_gate_swiglu | fused_ffn | fused_ffn; fused_ffn_kernel; g18x37x1; b128x1x1; r146; s32768 | 60 | 43.153 | 46.016 | 1.066x | 73.7% | 0.187 | fused_ffn (34) | fused_ffn (1.133x; 34) |
| 102 | outer_03.transformer_0.block_09.ffn_linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 60 | 20.928 | 31.680 | 1.514x | 98.0% | 0.673 | fused_ffn (30) | wide_qkv (1.617x; 30) |
| 103 | outer_03.transformer_1.block_10.attention_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | n/a | 3.696 | n/a | 100.0% | 0.000 | linear2_residual (30) | n/a |
| 104 | outer_03.transformer_1.block_10.attention_qkv_projection | wide_qkv | wide_qkv; wide_qkv_kernel; g9x37x1; b128x1x1; r136; s65536 | 60 | 19.312 | 24.657 | 1.277x | 83.0% | 0.331 | linear2_residual (30) | linear2_residual (1.456x; 30) |
| 105 | outer_03.transformer_1.block_10.attention_qk_rope | qk_rope | qk_rope; fusedQKRoPE19HalfKernel; g361x13x1; b192x1x1; r16; s0 | 60 | 4.128 | 7.456 | 1.806x | 97.5% | 0.199 | fa4 (30) | wide_qkv (2.252x; 30) |
| 106 | outer_03.transformer_1.block_10.attention_fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 60 | 11.905 | 14.560 | 1.223x | 45.6% | 0.168 | library_gemm (27) | fa4 (1.366x; 6) |
| 107 | outer_03.transformer_1.block_10.attention_out_projection_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 60 | 8.512 | 15.904 | 1.868x | 95.6% | 0.533 | library_gemm (32) | fused_ffn (2.444x; 28) |
| 108 | outer_03.transformer_1.block_10.ffn_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | n/a | 4.192 | n/a | 100.0% | 0.000 | fused_ffn (30) | n/a |
| 109 | outer_03.transformer_1.block_10.ffn_linear1_gate_swiglu | fused_ffn | fused_ffn; fused_ffn_kernel; g18x37x1; b128x1x1; r146; s32768 | 60 | 42.352 | 45.089 | 1.065x | 75.3% | 0.169 | fused_ffn (32) | fused_ffn (1.122x; 32) |
| 110 | outer_03.transformer_1.block_10.ffn_linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 60 | 20.704 | 31.632 | 1.528x | 98.3% | 0.671 | fused_ffn (30) | wide_qkv (1.621x; 30) |
| 111 | outer_03.transformer_2.block_11.attention_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | n/a | 3.888 | n/a | 100.0% | 0.000 | linear2_residual (30) | n/a |
| 112 | outer_03.transformer_2.block_11.attention_qkv_projection | wide_qkv | wide_qkv; wide_qkv_kernel; g9x37x1; b128x1x1; r136; s65536 | 60 | 19.152 | 24.689 | 1.289x | 82.8% | 0.350 | linear2_residual (30) | linear2_residual (1.487x; 30) |
| 113 | outer_03.transformer_2.block_11.attention_qk_rope | qk_rope | qk_rope; fusedQKRoPE19HalfKernel; g361x13x1; b192x1x1; r16; s0 | 60 | 4.096 | 7.264 | 1.773x | 100.0% | 0.200 | fa4 (30) | wide_qkv (2.227x; 30) |
| 114 | outer_03.transformer_2.block_11.attention_fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 60 | 11.872 | 14.544 | 1.225x | 44.3% | 0.163 | library_gemm (30) | library_gemm (1.226x; 30) |
| 115 | outer_03.transformer_2.block_11.attention_out_projection_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 60 | 8.448 | 17.377 | 2.057x | 95.7% | 0.552 | library_gemm (31) | fused_ffn (2.485x; 29) |
| 116 | outer_03.transformer_2.block_11.ffn_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | n/a | 4.064 | n/a | 100.0% | 0.000 | fused_ffn (30) | n/a |
| 117 | outer_03.transformer_2.block_11.ffn_linear1_gate_swiglu | fused_ffn | fused_ffn; fused_ffn_kernel; g18x37x1; b128x1x1; r146; s32768 | 60 | 42.432 | 45.089 | 1.063x | 76.4% | 0.163 | fused_ffn (32) | fused_ffn (1.121x; 32) |
| 118 | outer_03.transformer_2.block_11.ffn_linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 60 | 20.768 | 30.832 | 1.485x | 98.2% | 0.611 | fused_ffn (30) | library_gemm (1.510x; 30) |
| 119 | outer_03.post_norm_silu | affine_silu | affine_silu; affineSiluHalf2Kernel; g4693x1x1; b192x1x1; r17; s0 | 60 | 3.328 | 4.000 | 1.202x | 100.0% | 0.039 | library_gemm (30) | linear2_residual (1.221x; 30) |
| 120 | outer_03.post_projection_c384_to_c768_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b256x1x1; r154; s73728 | 60 | 14.320 | 22.096 | 1.543x | 95.7% | 0.471 | affine_silu (30) | linear2_residual (1.635x; 30) |
| 121 | outer_04.pre_norm_silu | affine_silu | affine_silu; affineSiluHalf2Kernel; g4693x1x1; b384x1x1; r17; s0 | 60 | 5.056 | 7.968 | 1.576x | 100.0% | 0.174 | library_gemm (60) | library_gemm (1.576x; 60) |
| 122 | outer_04.pre_projection_c768_to_c384 | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 60 | 11.904 | 15.632 | 1.313x | 93.6% | 0.281 | wide_qkv (30) | library_gemm (1.554x; 20) |
| 123 | outer_04.transformer_0.block_12.attention_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | n/a | 3.952 | n/a | 100.0% | 0.000 | library_gemm (30) | n/a |
| 124 | outer_04.transformer_0.block_12.attention_qkv_projection | wide_qkv | wide_qkv; wide_qkv_kernel; g9x37x1; b128x1x1; r136; s65536 | 60 | 19.104 | 25.312 | 1.325x | 66.6% | 0.354 | library_gemm (30) | library_gemm (1.399x; 30) |
| 125 | outer_04.transformer_0.block_12.attention_qk_rope | qk_rope | qk_rope; fusedQKRoPE19HalfKernel; g361x13x1; b192x1x1; r16; s0 | 60 | 4.128 | 7.536 | 1.826x | 98.3% | 0.205 | fa4 (30) | wide_qkv (2.260x; 30) |
| 126 | outer_04.transformer_0.block_12.attention_fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 60 | 11.904 | 15.344 | 1.289x | 47.9% | 0.217 | library_gemm (28) | fa4 (1.442x; 4) |
| 127 | outer_04.transformer_0.block_12.attention_out_projection_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 60 | 8.608 | 17.648 | 2.050x | 95.7% | 0.563 | library_gemm (35) | fused_ffn (2.431x; 25) |
| 128 | outer_04.transformer_0.block_12.ffn_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | n/a | 4.208 | n/a | 100.0% | 0.000 | fused_ffn (30) | n/a |
| 129 | outer_04.transformer_0.block_12.ffn_linear1_gate_swiglu | fused_ffn | fused_ffn; fused_ffn_kernel; g18x37x1; b128x1x1; r146; s32768 | 60 | 41.184 | 44.240 | 1.074x | 77.7% | 0.217 | fused_ffn (30) | fused_ffn (1.125x; 30) |
| 130 | outer_04.transformer_0.block_12.ffn_linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 60 | 21.121 | 35.008 | 1.658x | 98.1% | 0.841 | fused_ffn (30) | wide_qkv (1.749x; 30) |
| 131 | outer_04.transformer_1.block_13.attention_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | n/a | 4.000 | n/a | 100.0% | 0.000 | linear2_residual (30) | n/a |
| 132 | outer_04.transformer_1.block_13.attention_qkv_projection | wide_qkv | wide_qkv; wide_qkv_kernel; g9x37x1; b128x1x1; r136; s65536 | 60 | 19.504 | 26.977 | 1.383x | 83.6% | 0.466 | linear2_residual (30) | linear2_residual (1.617x; 30) |
| 133 | outer_04.transformer_1.block_13.attention_qk_rope | qk_rope | qk_rope; fusedQKRoPE19HalfKernel; g361x13x1; b192x1x1; r16; s0 | 60 | 4.128 | 7.632 | 1.849x | 99.2% | 0.211 | fa4 (30) | wide_qkv (2.291x; 30) |
| 134 | outer_04.transformer_1.block_13.attention_fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 60 | 12.032 | 15.072 | 1.253x | 46.5% | 0.191 | library_gemm (29) | qk_rope (1.258x; 29) |
| 135 | outer_04.transformer_1.block_13.attention_out_projection_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 60 | 8.672 | 16.064 | 1.852x | 95.8% | 0.551 | library_gemm (31) | fused_ffn (2.435x; 29) |
| 136 | outer_04.transformer_1.block_13.ffn_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | n/a | 3.760 | n/a | 100.0% | 0.000 | fused_ffn (30) | n/a |
| 137 | outer_04.transformer_1.block_13.ffn_linear1_gate_swiglu | fused_ffn | fused_ffn; fused_ffn_kernel; g18x37x1; b128x1x1; r146; s32768 | 60 | 43.248 | 46.433 | 1.074x | 82.4% | 0.184 | fused_ffn (30) | fused_ffn (1.092x; 30) |
| 138 | outer_04.transformer_1.block_13.ffn_linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 60 | 21.073 | 35.968 | 1.707x | 98.4% | 0.893 | fused_ffn (30) | wide_qkv (1.749x; 30) |
| 139 | outer_04.transformer_2.block_14.attention_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | n/a | 4.000 | n/a | 100.0% | 0.000 | linear2_residual (30) | n/a |
| 140 | outer_04.transformer_2.block_14.attention_qkv_projection | wide_qkv | wide_qkv; wide_qkv_kernel; g9x37x1; b128x1x1; r136; s65536 | 60 | 19.488 | 27.137 | 1.392x | 83.0% | 0.502 | linear2_residual (30) | linear2_residual (1.652x; 30) |
| 141 | outer_04.transformer_2.block_14.attention_qk_rope | qk_rope | qk_rope; fusedQKRoPE19HalfKernel; g361x13x1; b192x1x1; r16; s0 | 60 | 4.192 | 7.552 | 1.802x | 100.0% | 0.203 | fa4 (30) | wide_qkv (2.202x; 30) |
| 142 | outer_04.transformer_2.block_14.attention_fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 60 | 12.049 | 15.296 | 1.270x | 47.0% | 0.197 | library_gemm (30) | qk_rope (1.289x; 30) |
| 143 | outer_04.transformer_2.block_14.attention_out_projection_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 60 | 8.704 | 17.856 | 2.051x | 95.8% | 0.553 | library_gemm (31) | fused_ffn (2.390x; 29) |
| 144 | outer_04.transformer_2.block_14.ffn_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | n/a | 3.984 | n/a | 100.0% | 0.000 | fused_ffn (30) | n/a |
| 145 | outer_04.transformer_2.block_14.ffn_linear1_gate_swiglu | fused_ffn | fused_ffn; fused_ffn_kernel; g18x37x1; b128x1x1; r146; s32768 | 60 | 37.552 | 42.737 | 1.138x | 82.3% | 0.309 | fused_ffn (30) | fused_ffn (1.157x; 30) |
| 146 | outer_04.transformer_2.block_14.ffn_linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 60 | 21.152 | 36.705 | 1.735x | 98.4% | 0.958 | fused_ffn (30) | library_gemm (1.848x; 30) |
| 147 | outer_04.post_norm_silu | affine_silu | affine_silu; affineSiluHalf2Kernel; g4693x1x1; b192x1x1; r17; s0 | 60 | 3.424 | 4.384 | 1.280x | 100.0% | 0.059 | library_gemm (30) | linear2_residual (1.327x; 30) |
| 148 | outer_04.post_projection_c384_to_c768_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b256x1x1; r154; s73728 | 60 | 14.736 | 24.928 | 1.692x | 96.0% | 0.615 | affine_silu (30) | linear2_residual (1.845x; 30) |
| 149 | outer_05.pre_norm_silu | affine_silu | affine_silu; affineSiluHalf2Kernel; g4693x1x1; b384x1x1; r17; s0 | 60 | 5.184 | 8.544 | 1.648x | 100.0% | 0.207 | library_gemm (60) | library_gemm (1.648x; 60) |
| 150 | outer_05.pre_projection_c768_to_c384 | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 60 | 12.161 | 17.792 | 1.463x | 94.5% | 0.384 | wide_qkv (30) | library_gemm (1.724x; 23) |
| 151 | outer_05.transformer_0.block_15.attention_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | n/a | 4.560 | n/a | 100.0% | 0.000 | library_gemm (30) | n/a |
| 152 | outer_05.transformer_0.block_15.attention_qkv_projection | wide_qkv | wide_qkv; wide_qkv_kernel; g9x37x1; b128x1x1; r136; s65536 | 60 | 19.569 | 26.273 | 1.343x | 66.2% | 0.386 | library_gemm (30) | library_gemm (1.428x; 30) |
| 153 | outer_05.transformer_0.block_15.attention_qk_rope | qk_rope | qk_rope; fusedQKRoPE19HalfKernel; g361x13x1; b192x1x1; r16; s0 | 60 | 4.192 | 7.808 | 1.862x | 100.0% | 0.223 | fa4 (30) | wide_qkv (2.328x; 30) |
| 154 | outer_05.transformer_0.block_15.attention_fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 60 | 12.128 | 15.969 | 1.317x | 47.5% | 0.235 | library_gemm (30) | qk_rope (1.347x; 30) |
| 155 | outer_05.transformer_0.block_15.attention_out_projection_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 60 | 8.640 | 18.256 | 2.113x | 96.0% | 0.585 | library_gemm (37) | fused_ffn (2.467x; 23) |
| 156 | outer_05.transformer_0.block_15.ffn_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | n/a | 4.096 | n/a | 100.0% | 0.000 | fused_ffn (30) | n/a |
| 157 | outer_05.transformer_0.block_15.ffn_linear1_gate_swiglu | fused_ffn | fused_ffn; fused_ffn_kernel; g18x37x1; b128x1x1; r146; s32768 | 60 | 43.153 | 49.169 | 1.139x | 78.5% | 0.387 | fused_ffn (31) | fused_ffn (1.205x; 31) |
| 158 | outer_05.transformer_0.block_15.ffn_linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 60 | 21.216 | 34.080 | 1.606x | 98.0% | 0.777 | fused_ffn (30) | wide_qkv (1.659x; 30) |
| 159 | outer_05.transformer_1.block_16.attention_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | n/a | 4.032 | n/a | 100.0% | 0.000 | linear2_residual (30) | n/a |
| 160 | outer_05.transformer_1.block_16.attention_qkv_projection | wide_qkv | wide_qkv; wide_qkv_kernel; g9x37x1; b128x1x1; r136; s65536 | 60 | 19.648 | 27.360 | 1.393x | 83.8% | 0.473 | linear2_residual (30) | linear2_residual (1.646x; 30) |
| 161 | outer_05.transformer_1.block_16.attention_qk_rope | qk_rope | qk_rope; fusedQKRoPE19HalfKernel; g361x13x1; b192x1x1; r16; s0 | 60 | 4.192 | 7.568 | 1.805x | 100.0% | 0.207 | fa4 (30) | wide_qkv (2.237x; 30) |
| 162 | outer_05.transformer_1.block_16.attention_fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 60 | 12.065 | 14.944 | 1.239x | 44.3% | 0.176 | library_gemm (30) | library_gemm (1.241x; 30) |
| 163 | outer_05.transformer_1.block_16.attention_out_projection_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 60 | 8.577 | 17.648 | 2.058x | 95.8% | 0.557 | library_gemm (32) | fused_ffn (2.442x; 28) |
| 164 | outer_05.transformer_1.block_16.ffn_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | n/a | 4.176 | n/a | 100.0% | 0.000 | fused_ffn (30) | n/a |
| 165 | outer_05.transformer_1.block_16.ffn_linear1_gate_swiglu | fused_ffn | fused_ffn; fused_ffn_kernel; g18x37x1; b128x1x1; r146; s32768 | 60 | 43.425 | 47.313 | 1.090x | 75.8% | 0.233 | fused_ffn (30) | fused_ffn (1.144x; 30) |
| 166 | outer_05.transformer_1.block_16.ffn_linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 60 | 21.025 | 33.296 | 1.584x | 98.3% | 0.747 | fused_ffn (30) | wide_qkv (1.664x; 30) |
| 167 | outer_05.transformer_2.block_17.attention_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | n/a | 4.064 | n/a | 100.0% | 0.000 | linear2_residual (30) | n/a |
| 168 | outer_05.transformer_2.block_17.attention_qkv_projection | wide_qkv | wide_qkv; wide_qkv_kernel; g9x37x1; b128x1x1; r136; s65536 | 60 | 19.440 | 25.392 | 1.306x | 83.2% | 0.384 | linear2_residual (30) | linear2_residual (1.530x; 30) |
| 169 | outer_05.transformer_2.block_17.attention_qk_rope | qk_rope | qk_rope; fusedQKRoPE19HalfKernel; g361x13x1; b192x1x1; r16; s0 | 60 | 4.160 | 7.632 | 1.835x | 97.5% | 0.208 | fa4 (30) | wide_qkv (2.277x; 30) |
| 170 | outer_05.transformer_2.block_17.attention_fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 60 | 12.000 | 14.848 | 1.237x | 46.9% | 0.183 | library_gemm (27) | fa4 (1.391x; 6) |
| 171 | outer_05.transformer_2.block_17.attention_out_projection_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 60 | 8.640 | 16.352 | 1.893x | 95.7% | 0.534 | library_gemm (31) | fused_ffn (2.400x; 29) |
| 172 | outer_05.transformer_2.block_17.ffn_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | n/a | 4.208 | n/a | 100.0% | 0.000 | fused_ffn (30) | n/a |
| 173 | outer_05.transformer_2.block_17.ffn_linear1_gate_swiglu | fused_ffn | fused_ffn; fused_ffn_kernel; g18x37x1; b128x1x1; r146; s32768 | 60 | 43.361 | 47.120 | 1.087x | 75.6% | 0.193 | fused_ffn (32) | fused_ffn (1.130x; 32) |
| 174 | outer_05.transformer_2.block_17.ffn_linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 60 | 20.992 | 32.336 | 1.540x | 98.3% | 0.680 | fused_ffn (30) | library_gemm (1.566x; 30) |
| 175 | outer_05.post_norm_silu | affine_silu | affine_silu; affineSiluHalf2Kernel; g4693x1x1; b192x1x1; r17; s0 | 60 | 3.328 | 4.048 | 1.216x | 100.0% | 0.043 | library_gemm (30) | linear2_residual (1.250x; 30) |
| 176 | outer_05.post_projection_c384_to_c768_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b256x1x1; r154; s73728 | 60 | 14.496 | 22.352 | 1.542x | 95.8% | 0.509 | affine_silu (30) | linear2_residual (1.692x; 30) |
| 177 | outer_06.pre_norm_silu | affine_silu | affine_silu; affineSiluHalf2Kernel; g4693x1x1; b384x1x1; r17; s0 | 60 | 5.088 | 8.048 | 1.582x | 100.0% | 0.181 | library_gemm (60) | library_gemm (1.582x; 60) |
| 178 | outer_06.pre_projection_c768_to_c384 | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 60 | 12.032 | 16.192 | 1.346x | 94.1% | 0.298 | wide_qkv (30) | library_gemm (1.569x; 19) |
| 179 | outer_06.transformer_0.block_18.attention_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | n/a | 4.176 | n/a | 100.0% | 0.000 | library_gemm (30) | n/a |
| 180 | outer_06.transformer_0.block_18.attention_qkv_projection | wide_qkv | wide_qkv; wide_qkv_kernel; g9x37x1; b128x1x1; r136; s65536 | 60 | 19.297 | 25.104 | 1.301x | 66.5% | 0.371 | library_gemm (30) | library_gemm (1.424x; 30) |
| 181 | outer_06.transformer_0.block_18.attention_qk_rope | qk_rope | qk_rope; fusedQKRoPE19HalfKernel; g361x13x1; b192x1x1; r16; s0 | 60 | 4.144 | 7.777 | 1.877x | 99.2% | 0.211 | fa4 (30) | wide_qkv (2.286x; 30) |
| 182 | outer_06.transformer_0.block_18.attention_fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 60 | 12.032 | 16.000 | 1.330x | 49.0% | 0.237 | library_gemm (29) | qk_rope (1.348x; 29) |
| 183 | outer_06.transformer_0.block_18.attention_out_projection_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 60 | 8.544 | 18.304 | 2.142x | 96.0% | 0.586 | library_gemm (33) | fused_ffn (2.476x; 27) |
| 184 | outer_06.transformer_0.block_18.ffn_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | n/a | 4.048 | n/a | 100.0% | 0.000 | fused_ffn (30) | n/a |
| 185 | outer_06.transformer_0.block_18.ffn_linear1_gate_swiglu | fused_ffn | fused_ffn; fused_ffn_kernel; g18x37x1; b128x1x1; r146; s32768 | 60 | 42.305 | 47.712 | 1.128x | 78.5% | 0.358 | fused_ffn (30) | fused_ffn (1.183x; 30) |
| 186 | outer_06.transformer_0.block_18.ffn_linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 60 | 21.200 | 33.856 | 1.597x | 98.0% | 0.762 | fused_ffn (30) | wide_qkv (1.614x; 30) |
| 187 | outer_06.transformer_1.block_19.attention_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | n/a | 4.048 | n/a | 100.0% | 0.000 | linear2_residual (30) | n/a |
| 188 | outer_06.transformer_1.block_19.attention_qkv_projection | wide_qkv | wide_qkv; wide_qkv_kernel; g9x37x1; b128x1x1; r136; s65536 | 60 | 19.600 | 27.184 | 1.387x | 82.9% | 0.474 | linear2_residual (30) | linear2_residual (1.630x; 30) |
| 189 | outer_06.transformer_1.block_19.attention_qk_rope | qk_rope | qk_rope; fusedQKRoPE19HalfKernel; g361x13x1; b192x1x1; r16; s0 | 60 | 4.160 | 7.568 | 1.819x | 96.6% | 0.208 | fa4 (30) | wide_qkv (2.281x; 30) |
| 190 | outer_06.transformer_1.block_19.attention_fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 60 | 12.080 | 15.088 | 1.249x | 47.8% | 0.196 | library_gemm (26) | fa4 (1.383x; 8) |
| 191 | outer_06.transformer_1.block_19.attention_out_projection_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 60 | 8.720 | 16.624 | 1.906x | 95.7% | 0.530 | library_gemm (31) | fused_ffn (2.371x; 29) |
| 192 | outer_06.transformer_1.block_19.ffn_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | n/a | 3.936 | n/a | 100.0% | 0.000 | fused_ffn (30) | n/a |
| 193 | outer_06.transformer_1.block_19.ffn_linear1_gate_swiglu | fused_ffn | fused_ffn; fused_ffn_kernel; g18x37x1; b128x1x1; r146; s32768 | 60 | 41.984 | 47.697 | 1.136x | 81.6% | 0.327 | fused_ffn (30) | fused_ffn (1.162x; 30) |
| 194 | outer_06.transformer_1.block_19.ffn_linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 60 | 21.169 | 34.992 | 1.653x | 98.4% | 0.823 | fused_ffn (30) | fused_ffn (1.658x; 30) |
| 195 | outer_06.transformer_2.block_20.attention_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | n/a | 4.208 | n/a | 100.0% | 0.000 | linear2_residual (30) | n/a |
| 196 | outer_06.transformer_2.block_20.attention_qkv_projection | wide_qkv | wide_qkv; wide_qkv_kernel; g9x37x1; b128x1x1; r136; s65536 | 60 | 19.488 | 27.761 | 1.424x | 80.8% | 0.500 | linear2_residual (30) | linear2_residual (1.663x; 30) |
| 197 | outer_06.transformer_2.block_20.attention_qk_rope | qk_rope | qk_rope; fusedQKRoPE19HalfKernel; g361x13x1; b192x1x1; r16; s0 | 60 | 4.176 | 7.552 | 1.808x | 100.0% | 0.203 | fa4 (30) | wide_qkv (2.218x; 30) |
| 198 | outer_06.transformer_2.block_20.attention_fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 60 | 12.096 | 15.152 | 1.253x | 46.1% | 0.188 | library_gemm (30) | qk_rope (1.286x; 30) |
| 199 | outer_06.transformer_2.block_20.attention_out_projection_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 60 | 8.704 | 17.712 | 2.035x | 95.7% | 0.545 | fused_ffn (30) | fused_ffn (2.393x; 30) |
| 200 | outer_06.transformer_2.block_20.ffn_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | n/a | 4.048 | n/a | 100.0% | 0.000 | fused_ffn (30) | n/a |
| 201 | outer_06.transformer_2.block_20.ffn_linear1_gate_swiglu | fused_ffn | fused_ffn; fused_ffn_kernel; g18x37x1; b128x1x1; r146; s32768 | 60 | 38.065 | 42.400 | 1.114x | 83.2% | 0.267 | fused_ffn (30) | fused_ffn (1.140x; 30) |
| 202 | outer_06.transformer_2.block_20.ffn_linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 60 | 21.184 | 37.313 | 1.761x | 98.4% | 0.968 | fused_ffn (30) | library_gemm (1.820x; 30) |
| 203 | outer_06.post_norm_silu | affine_silu | affine_silu; affineSiluHalf2Kernel; g4693x1x1; b192x1x1; r17; s0 | 60 | 3.392 | 4.416 | 1.302x | 100.0% | 0.060 | library_gemm (30) | linear2_residual (1.321x; 30) |
| 204 | outer_06.post_projection_c384_to_c768_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b256x1x1; r154; s73728 | 60 | 14.688 | 24.337 | 1.657x | 95.8% | 0.616 | affine_silu (30) | linear2_residual (1.865x; 30) |
| 205 | outer_07.pre_norm_silu | affine_silu | affine_silu; affineSiluHalf2Kernel; g4693x1x1; b384x1x1; r17; s0 | 60 | 5.152 | 8.432 | 1.637x | 100.0% | 0.207 | library_gemm (60) | library_gemm (1.637x; 60) |
| 206 | outer_07.pre_projection_c768_to_c384 | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 60 | 12.129 | 17.552 | 1.447x | 94.5% | 0.381 | wide_qkv (30) | library_gemm (1.697x; 26) |
| 207 | outer_07.transformer_0.block_21.attention_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | n/a | 4.368 | n/a | 100.0% | 0.000 | library_gemm (30) | n/a |
| 208 | outer_07.transformer_0.block_21.attention_qkv_projection | wide_qkv | wide_qkv; wide_qkv_kernel; g9x37x1; b128x1x1; r136; s65536 | 60 | 19.584 | 26.384 | 1.347x | 66.2% | 0.398 | library_gemm (30) | library_gemm (1.440x; 30) |
| 209 | outer_07.transformer_0.block_21.attention_qk_rope | qk_rope | qk_rope; fusedQKRoPE19HalfKernel; g361x13x1; b192x1x1; r16; s0 | 60 | 4.128 | 7.920 | 1.919x | 99.2% | 0.220 | fa4 (30) | wide_qkv (2.357x; 30) |
| 210 | outer_07.transformer_0.block_21.attention_fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 60 | 12.113 | 16.000 | 1.321x | 48.1% | 0.237 | library_gemm (29) | qk_rope (1.347x; 29) |
| 211 | outer_07.transformer_0.block_21.attention_out_projection_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 60 | 8.672 | 17.857 | 2.059x | 95.6% | 0.579 | library_gemm (40) | fused_ffn (2.441x; 20) |
| 212 | outer_07.transformer_0.block_21.ffn_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | n/a | 4.016 | n/a | 100.0% | 0.000 | fused_ffn (30) | n/a |
| 213 | outer_07.transformer_0.block_21.ffn_linear1_gate_swiglu | fused_ffn | fused_ffn; fused_ffn_kernel; g18x37x1; b128x1x1; r146; s32768 | 60 | 40.688 | 44.113 | 1.084x | 83.1% | 0.195 | fused_ffn (30) | fused_ffn (1.098x; 30) |
| 214 | outer_07.transformer_0.block_21.ffn_linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 60 | 21.265 | 37.856 | 1.780x | 98.2% | 1.010 | fused_ffn (30) | wide_qkv (1.851x; 30) |
| 215 | outer_07.transformer_1.block_22.attention_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | n/a | 3.968 | n/a | 100.0% | 0.000 | linear2_residual (30) | n/a |
| 216 | outer_07.transformer_1.block_22.attention_qkv_projection | wide_qkv | wide_qkv; wide_qkv_kernel; g9x37x1; b128x1x1; r136; s65536 | 60 | 19.697 | 27.776 | 1.410x | 83.9% | 0.504 | linear2_residual (30) | linear2_residual (1.660x; 30) |
| 217 | outer_07.transformer_1.block_22.attention_qk_rope | qk_rope | qk_rope; fusedQKRoPE19HalfKernel; g361x13x1; b192x1x1; r16; s0 | 60 | 4.160 | 7.504 | 1.804x | 100.0% | 0.212 | fa4 (30) | wide_qkv (2.258x; 30) |
| 218 | outer_07.transformer_1.block_22.attention_fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 60 | 12.128 | 15.409 | 1.270x | 46.2% | 0.199 | library_gemm (30) | library_gemm (1.270x; 30) |
| 219 | outer_07.transformer_1.block_22.attention_out_projection_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 60 | 8.608 | 17.985 | 2.089x | 95.8% | 0.566 | library_gemm (32) | fused_ffn (2.416x; 28) |
| 220 | outer_07.transformer_1.block_22.ffn_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | n/a | 3.952 | n/a | 100.0% | 0.000 | fused_ffn (30) | n/a |
| 221 | outer_07.transformer_1.block_22.ffn_linear1_gate_swiglu | fused_ffn | fused_ffn; fused_ffn_kernel; g18x37x1; b128x1x1; r146; s32768 | 60 | 38.145 | 44.129 | 1.157x | 83.4% | 0.353 | fused_ffn (30) | fused_ffn (1.172x; 30) |
| 222 | outer_07.transformer_1.block_22.ffn_linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 60 | 21.265 | 37.633 | 1.770x | 98.5% | 0.990 | fused_ffn (30) | wide_qkv (1.844x; 30) |
| 223 | outer_07.transformer_2.block_23.attention_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | n/a | 4.096 | n/a | 100.0% | 0.000 | linear2_residual (30) | n/a |
| 224 | outer_07.transformer_2.block_23.attention_qkv_projection | wide_qkv | wide_qkv; wide_qkv_kernel; g9x37x1; b128x1x1; r136; s65536 | 60 | 19.713 | 27.616 | 1.401x | 83.1% | 0.507 | linear2_residual (30) | linear2_residual (1.683x; 30) |
| 225 | outer_07.transformer_2.block_23.attention_qk_rope | qk_rope | qk_rope; fusedQKRoPE19HalfKernel; g361x13x1; b192x1x1; r16; s0 | 60 | 4.192 | 7.568 | 1.805x | 100.0% | 0.207 | fa4 (30) | wide_qkv (2.248x; 30) |
| 226 | outer_07.transformer_2.block_23.attention_fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 60 | 12.209 | 15.424 | 1.263x | 46.9% | 0.196 | library_gemm (30) | qk_rope (1.287x; 30) |
| 227 | outer_07.transformer_2.block_23.attention_out_projection_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 60 | 8.672 | 17.824 | 2.055x | 95.8% | 0.554 | fused_ffn (30) | fused_ffn (2.399x; 30) |
| 228 | outer_07.transformer_2.block_23.ffn_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | n/a | 3.984 | n/a | 100.0% | 0.000 | fused_ffn (30) | n/a |
| 229 | outer_07.transformer_2.block_23.ffn_linear1_gate_swiglu | fused_ffn | fused_ffn; fused_ffn_kernel; g18x37x1; b128x1x1; r146; s32768 | 60 | 37.665 | 43.968 | 1.167x | 84.4% | 0.377 | fused_ffn (30) | fused_ffn (1.178x; 30) |
| 230 | outer_07.transformer_2.block_23.ffn_linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 60 | 21.280 | 37.328 | 1.754x | 98.4% | 0.963 | fused_ffn (30) | library_gemm (1.771x; 30) |
| 231 | outer_07.post_norm_silu | affine_silu | affine_silu; affineSiluHalf2Kernel; g4693x1x1; b192x1x1; r17; s0 | 60 | 3.424 | 4.480 | 1.308x | 100.0% | 0.059 | library_gemm (30) | linear2_residual (1.332x; 30) |
| 232 | outer_07.post_projection_c384_to_c768_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b256x1x1; r154; s73728 | 60 | 14.720 | 25.137 | 1.708x | 96.1% | 0.643 | linear2_residual (30) | linear2_residual (1.911x; 30) |
| 233 | outer_08.pre_norm_silu | affine_silu | affine_silu; affineSiluHalf2Kernel; g4693x1x1; b384x1x1; r17; s0 | 60 | 5.184 | 8.496 | 1.639x | 100.0% | 0.199 | library_gemm (60) | library_gemm (1.639x; 60) |
| 234 | outer_08.pre_projection_c768_to_c384 | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 60 | 12.209 | 17.552 | 1.438x | 94.5% | 0.372 | wide_qkv (30) | library_gemm (1.664x; 25) |
| 235 | outer_08.transformer_0.block_24.attention_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | n/a | 4.240 | n/a | 100.0% | 0.000 | library_gemm (30) | n/a |
| 236 | outer_08.transformer_0.block_24.attention_qkv_projection | wide_qkv | wide_qkv; wide_qkv_kernel; g9x37x1; b128x1x1; r136; s65536 | 60 | 19.648 | 26.768 | 1.362x | 66.7% | 0.427 | library_gemm (30) | library_gemm (1.462x; 30) |
| 237 | outer_08.transformer_0.block_24.attention_qk_rope | qk_rope | qk_rope; fusedQKRoPE19HalfKernel; g361x13x1; b192x1x1; r16; s0 | 60 | 4.224 | 7.665 | 1.815x | 100.0% | 0.214 | fa4 (30) | wide_qkv (2.239x; 30) |
| 238 | outer_08.transformer_0.block_24.attention_fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 60 | 12.176 | 16.144 | 1.326x | 50.0% | 0.240 | library_gemm (30) | qk_rope (1.351x; 30) |
| 239 | outer_08.transformer_0.block_24.attention_out_projection_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 60 | 8.800 | 18.416 | 2.093x | 96.0% | 0.578 | library_gemm (35) | fused_ffn (2.418x; 25) |
| 240 | outer_08.transformer_0.block_24.ffn_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | n/a | 4.144 | n/a | 100.0% | 0.000 | fused_ffn (30) | n/a |
| 241 | outer_08.transformer_0.block_24.ffn_linear1_gate_swiglu | fused_ffn | fused_ffn; fused_ffn_kernel; g18x37x1; b128x1x1; r146; s32768 | 60 | 42.833 | 49.489 | 1.155x | 80.6% | 0.380 | fused_ffn (31) | fused_ffn (1.192x; 31) |
| 242 | outer_08.transformer_0.block_24.ffn_linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 60 | 21.265 | 35.441 | 1.667x | 98.1% | 0.850 | fused_ffn (30) | wide_qkv (1.692x; 30) |
| 243 | outer_08.transformer_1.block_25.attention_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | n/a | 4.048 | n/a | 100.0% | 0.000 | linear2_residual (30) | n/a |
| 244 | outer_08.transformer_1.block_25.attention_qkv_projection | wide_qkv | wide_qkv; wide_qkv_kernel; g9x37x1; b128x1x1; r136; s65536 | 60 | 19.648 | 27.680 | 1.409x | 82.7% | 0.490 | linear2_residual (30) | linear2_residual (1.650x; 30) |
| 245 | outer_08.transformer_1.block_25.attention_qk_rope | qk_rope | qk_rope; fusedQKRoPE19HalfKernel; g361x13x1; b192x1x1; r16; s0 | 60 | 4.192 | 7.584 | 1.809x | 100.0% | 0.207 | fa4 (30) | wide_qkv (2.248x; 30) |
| 246 | outer_08.transformer_1.block_25.attention_fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 60 | 12.160 | 15.216 | 1.251x | 46.3% | 0.190 | library_gemm (30) | qk_rope (1.272x; 30) |
| 247 | outer_08.transformer_1.block_25.attention_out_projection_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 60 | 8.672 | 17.792 | 2.052x | 95.8% | 0.554 | library_gemm (31) | fused_ffn (2.399x; 29) |
| 248 | outer_08.transformer_1.block_25.ffn_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | n/a | 4.064 | n/a | 100.0% | 0.000 | fused_ffn (30) | n/a |
| 249 | outer_08.transformer_1.block_25.ffn_linear1_gate_swiglu | fused_ffn | fused_ffn; fused_ffn_kernel; g18x37x1; b128x1x1; r146; s32768 | 60 | 43.169 | 48.513 | 1.124x | 78.6% | 0.320 | fused_ffn (30) | fused_ffn (1.168x; 30) |
| 250 | outer_08.transformer_1.block_25.ffn_linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 60 | 21.216 | 34.864 | 1.643x | 98.4% | 0.803 | fused_ffn (30) | wide_qkv (1.690x; 30) |
| 251 | outer_08.transformer_2.block_26.attention_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | n/a | 4.032 | n/a | 100.0% | 0.000 | linear2_residual (30) | n/a |
| 252 | outer_08.transformer_2.block_26.attention_qkv_projection | wide_qkv | wide_qkv; wide_qkv_kernel; g9x37x1; b128x1x1; r136; s65536 | 60 | 19.616 | 27.520 | 1.403x | 83.6% | 0.477 | linear2_residual (30) | linear2_residual (1.643x; 30) |
| 253 | outer_08.transformer_2.block_26.attention_qk_rope | qk_rope | qk_rope; fusedQKRoPE19HalfKernel; g361x13x1; b192x1x1; r16; s0 | 60 | 4.160 | 7.552 | 1.815x | 100.0% | 0.207 | fa4 (30) | wide_qkv (2.238x; 30) |
| 254 | outer_08.transformer_2.block_26.attention_fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 60 | 12.128 | 15.040 | 1.240x | 45.3% | 0.179 | library_gemm (30) | qk_rope (1.249x; 30) |
| 255 | outer_08.transformer_2.block_26.attention_out_projection_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 60 | 8.688 | 17.728 | 2.041x | 95.8% | 0.551 | library_gemm (31) | fused_ffn (2.383x; 29) |
| 256 | outer_08.transformer_2.block_26.ffn_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | n/a | 4.032 | n/a | 100.0% | 0.000 | fused_ffn (30) | n/a |
| 257 | outer_08.transformer_2.block_26.ffn_linear1_gate_swiglu | fused_ffn | fused_ffn; fused_ffn_kernel; g18x37x1; b128x1x1; r146; s32768 | 60 | 41.969 | 47.425 | 1.130x | 79.9% | 0.338 | fused_ffn (30) | fused_ffn (1.173x; 30) |
| 258 | outer_08.transformer_2.block_26.ffn_linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 60 | 21.200 | 34.064 | 1.607x | 98.3% | 0.772 | fused_ffn (30) | library_gemm (1.611x; 30) |
| 259 | outer_08.post_norm_silu | affine_silu | affine_silu; affineSiluHalf2Kernel; g4693x1x1; b192x1x1; r17; s0 | 60 | 3.392 | 4.128 | 1.217x | 100.0% | 0.043 | library_gemm (30) | linear2_residual (1.274x; 30) |
| 260 | outer_08.post_projection_c384_to_c768_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b256x1x1; r154; s73728 | 60 | 14.624 | 24.960 | 1.707x | 96.1% | 0.633 | affine_silu (30) | linear2_residual (1.926x; 30) |
| 261 | outer_09.pre_norm_silu | affine_silu | affine_silu; affineSiluHalf2Kernel; g4693x1x1; b384x1x1; r17; s0 | 60 | 5.152 | 8.400 | 1.630x | 100.0% | 0.192 | library_gemm (60) | library_gemm (1.630x; 60) |
| 262 | outer_09.pre_projection_c768_to_c384 | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 60 | 12.160 | 16.960 | 1.395x | 94.1% | 0.317 | wide_qkv (30) | library_gemm (1.621x; 5) |
| 263 | outer_09.transformer_0.block_27.attention_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | n/a | 4.320 | n/a | 100.0% | 0.000 | library_gemm (30) | n/a |
| 264 | outer_09.transformer_0.block_27.attention_qkv_projection | wide_qkv | wide_qkv; wide_qkv_kernel; g9x37x1; b128x1x1; r136; s65536 | 60 | 19.489 | 26.560 | 1.363x | 67.3% | 0.415 | library_gemm (30) | library_gemm (1.458x; 30) |
| 265 | outer_09.transformer_0.block_27.attention_qk_rope | qk_rope | qk_rope; fusedQKRoPE19HalfKernel; g361x13x1; b192x1x1; r16; s0 | 60 | 4.176 | 7.776 | 1.862x | 99.2% | 0.219 | fa4 (30) | wide_qkv (2.306x; 30) |
| 266 | outer_09.transformer_0.block_27.attention_fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 60 | 12.112 | 16.065 | 1.326x | 49.1% | 0.242 | library_gemm (29) | qk_rope (1.350x; 29) |
| 267 | outer_09.transformer_0.block_27.attention_out_projection_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 60 | 8.688 | 18.256 | 2.101x | 95.7% | 0.577 | library_gemm (40) | fused_ffn (2.433x; 20) |
| 268 | outer_09.transformer_0.block_27.ffn_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | n/a | 4.304 | n/a | 100.0% | 0.000 | fused_ffn (30) | n/a |
| 269 | outer_09.transformer_0.block_27.ffn_linear1_gate_swiglu | fused_ffn | fused_ffn; fused_ffn_kernel; g18x37x1; b128x1x1; r146; s32768 | 60 | 43.488 | 46.992 | 1.081x | 74.2% | 0.221 | fused_ffn (31) | fused_ffn (1.151x; 31) |
| 270 | outer_09.transformer_0.block_27.ffn_linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 60 | 21.056 | 32.576 | 1.547x | 98.0% | 0.712 | fused_ffn (30) | wide_qkv (1.642x; 30) |
| 271 | outer_09.transformer_1.block_28.attention_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | n/a | 3.856 | n/a | 100.0% | 0.000 | linear2_residual (30) | n/a |
| 272 | outer_09.transformer_1.block_28.attention_qkv_projection | wide_qkv | wide_qkv; wide_qkv_kernel; g9x37x1; b128x1x1; r136; s65536 | 60 | 19.441 | 24.928 | 1.282x | 83.2% | 0.353 | linear2_residual (30) | linear2_residual (1.485x; 30) |
| 273 | outer_09.transformer_1.block_28.attention_qk_rope | qk_rope | qk_rope; fusedQKRoPE19HalfKernel; g361x13x1; b192x1x1; r16; s0 | 60 | 4.160 | 7.408 | 1.781x | 94.9% | 0.199 | fa4 (30) | wide_qkv (2.254x; 30) |
| 274 | outer_09.transformer_1.block_28.attention_fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 60 | 12.000 | 14.800 | 1.233x | 49.8% | 0.189 | library_gemm (24) | fa4 (1.385x; 12) |
| 275 | outer_09.transformer_1.block_28.attention_out_projection_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 60 | 8.624 | 15.696 | 1.820x | 95.6% | 0.520 | library_gemm (32) | fused_ffn (2.404x; 28) |
| 276 | outer_09.transformer_1.block_28.ffn_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | n/a | 4.288 | n/a | 100.0% | 0.000 | fused_ffn (30) | n/a |
| 277 | outer_09.transformer_1.block_28.ffn_linear1_gate_swiglu | fused_ffn | fused_ffn; fused_ffn_kernel; g18x37x1; b128x1x1; r146; s32768 | 60 | 42.897 | 46.048 | 1.073x | 75.1% | 0.175 | fused_ffn (33) | fused_ffn (1.123x; 33) |
| 278 | outer_09.transformer_1.block_28.ffn_linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 60 | 20.848 | 31.552 | 1.513x | 98.3% | 0.671 | fused_ffn (30) | wide_qkv (1.622x; 30) |
| 279 | outer_09.transformer_2.block_29.attention_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | n/a | 3.728 | n/a | 100.0% | 0.000 | linear2_residual (30) | n/a |
| 280 | outer_09.transformer_2.block_29.attention_qkv_projection | wide_qkv | wide_qkv; wide_qkv_kernel; g9x37x1; b128x1x1; r136; s65536 | 60 | 19.328 | 24.784 | 1.282x | 82.8% | 0.343 | linear2_residual (30) | linear2_residual (1.469x; 30) |
| 281 | outer_09.transformer_2.block_29.attention_qk_rope | qk_rope | qk_rope; fusedQKRoPE19HalfKernel; g361x13x1; b192x1x1; r16; s0 | 60 | 4.128 | 7.424 | 1.798x | 97.5% | 0.198 | fa4 (30) | wide_qkv (2.229x; 30) |
| 282 | outer_09.transformer_2.block_29.attention_fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 60 | 11.921 | 14.656 | 1.229x | 47.0% | 0.175 | library_gemm (27) | fa4 (1.388x; 6) |
| 283 | outer_09.transformer_2.block_29.attention_out_projection_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 60 | 8.576 | 16.352 | 1.907x | 95.7% | 0.530 | library_gemm (32) | fused_ffn (2.425x; 28) |
| 284 | outer_09.transformer_2.block_29.ffn_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | n/a | 4.176 | n/a | 100.0% | 0.000 | fused_ffn (30) | n/a |
| 285 | outer_09.transformer_2.block_29.ffn_linear1_gate_swiglu | fused_ffn | fused_ffn; fused_ffn_kernel; g18x37x1; b128x1x1; r146; s32768 | 60 | 42.160 | 45.041 | 1.068x | 75.7% | 0.175 | fused_ffn (33) | fused_ffn (1.126x; 33) |
| 286 | outer_09.transformer_2.block_29.ffn_linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 60 | 20.736 | 31.120 | 1.501x | 98.3% | 0.639 | fused_ffn (30) | library_gemm (1.557x; 30) |
| 287 | outer_09.post_norm_silu | affine_silu | affine_silu; affineSiluHalf2Kernel; g4693x1x1; b192x1x1; r17; s0 | 60 | 3.360 | 3.937 | 1.172x | 100.0% | 0.037 | library_gemm (30) | linear2_residual (1.200x; 30) |
| 288 | outer_09.post_projection_c384_to_c768_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b256x1x1; r154; s73728 | 60 | 14.400 | 22.352 | 1.552x | 95.8% | 0.487 | affine_silu (30) | linear2_residual (1.648x; 30) |
| 289 | outer_10.pre_norm_silu | affine_silu | affine_silu; affineSiluHalf2Kernel; g4693x1x1; b384x1x1; r17; s0 | 60 | 5.072 | 8.080 | 1.593x | 100.0% | 0.173 | library_gemm (60) | library_gemm (1.593x; 60) |
| 290 | outer_10.pre_projection_c768_to_c384 | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 60 | 11.921 | 15.776 | 1.323x | 94.0% | 0.295 | wide_qkv (30) | library_gemm (1.565x; 25) |
| 291 | outer_10.transformer_0.block_30.attention_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | n/a | 3.872 | n/a | 100.0% | 0.000 | library_gemm (30) | n/a |
| 292 | outer_10.transformer_0.block_30.attention_qkv_projection | wide_qkv | wide_qkv; wide_qkv_kernel; g9x37x1; b128x1x1; r136; s65536 | 60 | 19.232 | 25.121 | 1.306x | 66.4% | 0.347 | library_gemm (30) | library_gemm (1.397x; 30) |
| 293 | outer_10.transformer_0.block_30.attention_qk_rope | qk_rope | qk_rope; fusedQKRoPE19HalfKernel; g361x13x1; b192x1x1; r16; s0 | 60 | 4.096 | 7.600 | 1.855x | 98.3% | 0.211 | fa4 (30) | wide_qkv (2.285x; 30) |
| 294 | outer_10.transformer_0.block_30.attention_fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 60 | 11.968 | 15.744 | 1.316x | 48.7% | 0.229 | library_gemm (28) | fa4 (1.445x; 4) |
| 295 | outer_10.transformer_0.block_30.attention_out_projection_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 60 | 8.544 | 16.768 | 1.963x | 95.7% | 0.570 | library_gemm (31) | fused_ffn (2.472x; 29) |
| 296 | outer_10.transformer_0.block_30.ffn_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | n/a | 4.032 | n/a | 100.0% | 0.000 | fused_ffn (30) | n/a |
| 297 | outer_10.transformer_0.block_30.ffn_linear1_gate_swiglu | fused_ffn | fused_ffn; fused_ffn_kernel; g18x37x1; b128x1x1; r146; s32768 | 60 | 40.865 | 44.353 | 1.085x | 83.5% | 0.205 | fused_ffn (30) | fused_ffn (1.099x; 30) |
| 298 | outer_10.transformer_0.block_30.ffn_linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 60 | 21.152 | 36.368 | 1.719x | 98.1% | 0.926 | fused_ffn (30) | wide_qkv (1.757x; 30) |
| 299 | outer_10.transformer_1.block_31.attention_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | n/a | 3.856 | n/a | 100.0% | 0.000 | linear2_residual (30) | n/a |
| 300 | outer_10.transformer_1.block_31.attention_qkv_projection | wide_qkv | wide_qkv; wide_qkv_kernel; g9x37x1; b128x1x1; r136; s65536 | 60 | 19.552 | 27.648 | 1.414x | 83.2% | 0.497 | linear2_residual (30) | linear2_residual (1.640x; 30) |
| 301 | outer_10.transformer_1.block_31.attention_qk_rope | qk_rope | qk_rope; fusedQKRoPE19HalfKernel; g361x13x1; b192x1x1; r16; s0 | 60 | 4.096 | 7.617 | 1.859x | 99.1% | 0.210 | fa4 (30) | wide_qkv (2.305x; 30) |
| 302 | outer_10.transformer_1.block_31.attention_fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 60 | 12.064 | 15.168 | 1.257x | 46.7% | 0.194 | library_gemm (29) | qk_rope (1.273x; 29) |
| 303 | outer_10.transformer_1.block_31.attention_out_projection_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 60 | 8.576 | 17.536 | 2.045x | 95.8% | 0.550 | library_gemm (31) | fused_ffn (2.422x; 29) |
| 304 | outer_10.transformer_1.block_31.ffn_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | n/a | 4.032 | n/a | 100.0% | 0.000 | fused_ffn (30) | n/a |
| 305 | outer_10.transformer_1.block_31.ffn_linear1_gate_swiglu | fused_ffn | fused_ffn; fused_ffn_kernel; g18x37x1; b128x1x1; r146; s32768 | 60 | 39.600 | 45.664 | 1.153x | 80.9% | 0.365 | fused_ffn (30) | fused_ffn (1.181x; 30) |
| 306 | outer_10.transformer_1.block_31.ffn_linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 60 | 21.152 | 35.425 | 1.675x | 98.4% | 0.873 | fused_ffn (30) | wide_qkv (1.747x; 30) |
| 307 | outer_10.transformer_2.block_32.attention_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | n/a | 4.144 | n/a | 100.0% | 0.000 | linear2_residual (30) | n/a |
| 308 | outer_10.transformer_2.block_32.attention_qkv_projection | wide_qkv | wide_qkv; wide_qkv_kernel; g9x37x1; b128x1x1; r136; s65536 | 60 | 19.616 | 27.921 | 1.423x | 83.6% | 0.509 | linear2_residual (30) | linear2_residual (1.688x; 30) |
| 309 | outer_10.transformer_2.block_32.attention_qk_rope | qk_rope | qk_rope; fusedQKRoPE19HalfKernel; g361x13x1; b192x1x1; r16; s0 | 60 | 4.192 | 7.568 | 1.805x | 100.0% | 0.202 | fa4 (30) | wide_qkv (2.225x; 30) |
| 310 | outer_10.transformer_2.block_32.attention_fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 60 | 12.096 | 15.456 | 1.278x | 47.9% | 0.203 | library_gemm (30) | qk_rope (1.295x; 30) |
| 311 | outer_10.transformer_2.block_32.attention_out_projection_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 60 | 8.704 | 17.712 | 2.035x | 95.7% | 0.554 | library_gemm (32) | fused_ffn (2.399x; 28) |
| 312 | outer_10.transformer_2.block_32.ffn_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | n/a | 3.952 | n/a | 100.0% | 0.000 | fused_ffn (30) | n/a |
| 313 | outer_10.transformer_2.block_32.ffn_linear1_gate_swiglu | fused_ffn | fused_ffn; fused_ffn_kernel; g18x37x1; b128x1x1; r146; s32768 | 60 | 42.385 | 48.401 | 1.142x | 80.3% | 0.370 | fused_ffn (30) | fused_ffn (1.179x; 30) |
| 314 | outer_10.transformer_2.block_32.ffn_linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 60 | 21.120 | 34.273 | 1.623x | 98.3% | 0.790 | fused_ffn (30) | fused_ffn (1.630x; 30) |
| 315 | outer_10.post_norm_silu | affine_silu | affine_silu; affineSiluHalf2Kernel; g4693x1x1; b192x1x1; r17; s0 | 60 | 3.424 | 4.240 | 1.238x | 100.0% | 0.046 | library_gemm (30) | linear2_residual (1.285x; 30) |
| 316 | outer_10.post_projection_c384_to_c768_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b256x1x1; r154; s73728 | 60 | 14.624 | 25.713 | 1.758x | 96.2% | 0.686 | linear2_residual (30) | linear2_residual (1.927x; 30) |
| 317 | trunk.tip_norm_silu | affine_silu | affine_silu; affineSiluHalf2Kernel; g4693x1x1; b384x1x1; r17; s0 | 60 | 5.120 | 8.416 | 1.644x | 88.7% | 0.185 | library_gemm (60) | library_gemm (1.644x; 60) |
| 318 | policy.p1_conv | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r80; s98304 | 60 | 6.240 | 11.200 | 1.795x | 93.7% | 0.300 | library_gemm (60) | library_gemm (1.795x; 60) |
| 319 | policy.g1_conv | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r80; s98304 | 60 | 5.952 | 8.112 | 1.363x | 91.4% | 0.123 | head_elementwise (30) | library_gemm (1.497x; 30) |
| 320 | policy.g1_norm_silu | head_elementwise | head_elementwise; applyCScaleBiasNHWCSiluHalfKernel; g1x73x13; b96x5x1; r16; s0 | 60 | 2.112 | 2.656 | 1.258x | 99.9% | 0.033 | head_elementwise (30) | library_gemm (1.364x; 30) |
| 321 | policy.g1_half_to_float | copy_reformat | copy_reformat; copyFromHalfKernel; g880x1x1; b512x1x1; r16; s0 | 60 | 1.568 | 1.904 | 1.214x | 58.9% | 0.023 | library_gemm (50) | library_gemm (1.255x; 50) |
| 322 | policy.g1_global_pool | head_elementwise | head_elementwise; gPoolChannelsNHWCKernel; g2x1x13; b64x8x1; r22; s4096 | 60 | 4.512 | 7.040 | 1.560x | 94.8% | 0.171 | library_gemm (60) | library_gemm (1.560x; 60) |
| 323 | policy.gpool_to_bias_matmul | library_gemm | library_gemm; gemmSN_NN_kernel; g2x2x1; b256x1x1; r64; s21504 | 60 | 5.409 | 6.800 | 1.257x | 82.2% | 0.093 | head_elementwise (60) | head_elementwise (1.257x; 60) |
| 324 | policy.p1_half_to_float | copy_reformat | copy_reformat; copyFromHalfKernel; g880x1x1; b512x1x1; r16; s0 | 60 | 1.536 | 1.696 | 1.104x | 81.3% | 0.014 | library_gemm (40) | head_elementwise (1.198x; 20) |
| 325 | policy.gpool_bias_add | head_elementwise | head_elementwise; addNCBiasInplaceNHWCKernel; g1x73x13; b96x5x1; r16; s0 | 60 | 1.824 | 2.096 | 1.149x | 99.3% | 0.017 | library_gemm (60) | library_gemm (1.149x; 60) |
| 326 | policy.p1_norm_silu | head_elementwise | head_elementwise; applyCScaleBiasNHWCSiluKernel; g1x73x13; b96x5x1; r16; s0 | 60 | 2.208 | 2.592 | 1.174x | 81.1% | 0.023 | library_gemm (50) | copy_reformat (1.268x; 10) |
| 327 | policy.p2_conv | library_gemm | library_gemm; Kernel2; g74x1x1; b128x1x1; r90; s98304 | 60 | 3.968 | 4.688 | 1.181x | 89.7% | 0.063 | library_gemm (30) | head_elementwise (1.306x; 29) |
| 328 | policy.gpool_to_pass_matmul | library_gemm | library_gemm; gemmSN_NN_kernel; g2x2x1; b256x1x1; r64; s21504 | 60 | 5.376 | 5.984 | 1.113x | 85.3% | 0.046 | library_gemm (59) | library_gemm (1.113x; 59) |
| 329 | policy.pass_bias_silu | head_elementwise | head_elementwise; addCBiasInplaceNCKernelSilu; g1x3x1; b96x5x1; r16; s0 | 60 | 1.008 | 1.216 | 1.206x | 96.6% | 0.042 | library_gemm (60) | library_gemm (1.206x; 60) |
| 330 | policy.gpool_to_pass_matmul2 | library_gemm | library_gemm; gemmSN_NN_kernel; g1x2x1; b256x1x1; r64; s21504 | 60 | 2.336 | 2.704 | 1.158x | 97.2% | 0.042 | library_gemm (59) | library_gemm (1.164x; 59) |
| 331 | value.v1_conv | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r118; s98304 | 60 | 8.064 | 9.424 | 1.169x | 86.5% | 0.085 | head_elementwise (30) | library_gemm (1.198x; 25) |
| 332 | value.v1_norm_silu | head_elementwise | head_elementwise; applyCScaleBiasNHWCSiluHalfKernel; g1x181x13; b192x2x1; r16; s0 | 60 | 3.136 | 3.712 | 1.184x | 73.8% | 0.082 | library_gemm (30) | library_gemm (1.633x; 30) |
| 333 | value.v1_half_to_float | copy_reformat | copy_reformat; copyFromHalfKernel; g1760x1x1; b512x1x1; r16; s0 | 60 | 2.176 | 2.496 | 1.147x | 89.8% | 0.042 | head_elementwise (32) | library_gemm (1.243x; 28) |
| 334 | value.v1_global_pool | head_elementwise | head_elementwise; valueHeadPoolChannelsNHWCKernel; g3x1x13; b64x8x1; r22; s2048 | 60 | 3.264 | 3.680 | 1.127x | 88.6% | 0.040 | library_gemm (31) | head_elementwise (1.392x; 4) |
| 335 | value.v2_matmul | library_gemm | library_gemm; gemmSN_NN_kernel; g3x2x1; b256x1x1; r64; s21504 | 60 | 9.568 | 10.080 | 1.054x | 93.4% | 0.038 | library_gemm (59) | library_gemm (1.054x; 59) |
| 336 | value.v2_bias_silu | head_elementwise | head_elementwise; addCBiasInplaceNCKernelSilu; g1x7x1; b192x2x1; r16; s0 | 60 | 1.024 | 1.088 | 1.062x | 98.6% | 0.005 | library_gemm (58) | library_gemm (1.062x; 58) |
| 337 | value.v3_matmul | library_gemm | library_gemm; gemmSN_NN_kernel; g1x2x1; b256x1x1; r64; s21504 | 60 | 3.505 | 3.776 | 1.077x | 76.7% | 0.016 | library_gemm (47) | library_gemm (1.077x; 47) |
| 338 | value.v3_bias | head_elementwise | head_elementwise; addCBiasInplaceNCKernel; g1x1x1; b3x170x1; r16; s0 | 60 | 0.960 | 1.056 | 1.100x | 95.9% | 0.007 | library_gemm (56) | head_elementwise (1.167x; 4) |
| 339 | value.score_matmul | library_gemm | library_gemm; gemmSN_NN_kernel; g1x2x1; b256x1x1; r64; s21504 | 60 | 3.520 | 3.824 | 1.086x | 78.5% | 0.021 | library_gemm (55) | library_gemm (1.091x; 55) |
| 340 | value.score_bias | head_elementwise | head_elementwise; addCBiasInplaceNCKernel; g1x1x1; b6x85x1; r16; s0 | 60 | 0.928 | 1.024 | 1.103x | 95.2% | 0.006 | library_gemm (55) | library_gemm (1.103x; 55) |
| 341 | value.ownership_conv | library_gemm | library_gemm; Kernel2; g8x19x3; b128x1x1; r118; s33792 | 60 | 4.032 | 4.560 | 1.131x | 61.1% | 0.033 | library_gemm (52) | head_elementwise (1.139x; 8) |
| 342 | value.ownership_conv_splitk_reduce | library_gemm | library_gemm; splitKreduce_kernel; g147x1x1; b32x16x1; r49; s0 | 60 | 1.377 | 1.440 | 1.046x | 68.6% | 0.011 | library_gemm (31) | idle (1.232x; 5) |
| 343 | value.ownership_half_to_float | copy_reformat | copy_reformat; copyFromHalfKernel; g10x1x1; b512x1x1; r16; s0 | 60 | 0.928 | 1.120 | 1.206x | 37.5% | 0.017 | library_gemm (27) | idle (1.430x; 24) |
