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

## Logical operation groups

Isolated reference total is the isolated median for each ordinal multiplied by its S2 call count; it is a normalized reference, not a second trace total.

| logical group | families | positions | calls | isolated reference ms | S2 total ms | total ratio | excess ms |
|---|---|---:|---:|---:|---:|---:|---:|
| transformer.ffn_linear2_residual | linear2_residual | 33 | 1980 | 41.710 | 67.943 | 1.629x | 26.233 |
| transformer.attention_out_projection_residual | library_gemm | 33 | 1980 | 17.074 | 35.735 | 2.093x | 18.661 |
| transformer.attention_qkv_projection | wide_qkv | 33 | 1980 | 38.496 | 51.451 | 1.337x | 12.955 |
| transformer.ffn_linear1_gate_swiglu | fused_ffn | 33 | 1980 | 83.026 | 90.599 | 1.091x | 7.843 |
| transformer.attention_qk_rope | qk_rope | 33 | 1980 | 8.222 | 15.129 | 1.840x | 6.906 |
| transformer.attention_fa4 | fa4 | 33 | 1980 | 23.834 | 30.561 | 1.282x | 6.727 |
| outer.post_projection_c384_to_c768_residual | library_gemm | 11 | 660 | 9.604 | 15.768 | 1.642x | 6.164 |
| transformer.attention_rmsnorm | rmsnorm | 33 | 1980 | 5.328 | 9.661 | 1.813x | 4.332 |
| transformer.ffn_rmsnorm | rmsnorm | 33 | 1980 | 5.363 | 9.645 | 1.799x | 4.282 |
| outer.pre_projection_c768_to_c384 | library_gemm | 11 | 660 | 7.936 | 11.614 | 1.464x | 3.678 |
| outer.pre_norm_silu | affine_silu | 11 | 660 | 3.376 | 5.400 | 1.599x | 2.024 |
| outer.post_norm_silu | affine_silu | 11 | 660 | 2.231 | 2.781 | 1.246x | 0.550 |
| policy.p1_conv | library_gemm | 1 | 60 | 0.374 | 0.682 | 1.822x | 0.308 |
| trunk.tip_norm_silu | affine_silu | 1 | 60 | 0.307 | 0.490 | 1.597x | 0.183 |
| policy.g1_global_pool | head_elementwise | 1 | 60 | 0.271 | 0.444 | 1.639x | 0.173 |
| frontend.initial_conv | cudnn | 1 | 60 | 1.174 | 1.305 | 1.112x | 0.131 |
| policy.g1_conv | library_gemm | 1 | 60 | 0.357 | 0.478 | 1.340x | 0.121 |
| policy.gpool_to_bias_matmul | library_gemm | 1 | 60 | 0.325 | 0.410 | 1.263x | 0.085 |
| frontend.initial_global_broadcast_add | head_elementwise | 1 | 60 | 0.464 | 0.546 | 1.177x | 0.082 |
| value.v1_norm_silu | head_elementwise | 1 | 60 | 0.188 | 0.269 | 1.428x | 0.081 |
| value.v1_conv | library_gemm | 1 | 60 | 0.484 | 0.557 | 1.152x | 0.074 |
| frontend.initial_global_matmul | library_gemm | 1 | 60 | 0.157 | 0.230 | 1.458x | 0.072 |
| policy.pass_bias_silu | head_elementwise | 1 | 60 | 0.060 | 0.114 | 1.891x | 0.054 |
| policy.p2_conv | library_gemm | 1 | 60 | 0.238 | 0.291 | 1.223x | 0.053 |
| policy.gpool_to_pass_matmul | library_gemm | 1 | 60 | 0.323 | 0.368 | 1.142x | 0.046 |
| value.v1_half_to_float | copy_reformat | 1 | 60 | 0.131 | 0.176 | 1.344x | 0.045 |
| value.v2_matmul | library_gemm | 1 | 60 | 0.574 | 0.611 | 1.065x | 0.037 |
| policy.g1_norm_silu | head_elementwise | 1 | 60 | 0.127 | 0.160 | 1.266x | 0.034 |
| policy.gpool_to_pass_matmul2 | library_gemm | 1 | 60 | 0.140 | 0.174 | 1.240x | 0.034 |
| value.ownership_conv | library_gemm | 1 | 60 | 0.242 | 0.273 | 1.130x | 0.032 |
| value.v1_global_pool | head_elementwise | 1 | 60 | 0.196 | 0.227 | 1.161x | 0.032 |
| input.mask_sum | sumChannelsNCHWKernel | 1 | 60 | 0.098 | 0.121 | 1.237x | 0.023 |
| value.v3_matmul | library_gemm | 1 | 60 | 0.210 | 0.231 | 1.100x | 0.021 |
| value.score_matmul | library_gemm | 1 | 60 | 0.211 | 0.231 | 1.095x | 0.020 |
| input.mask_half_to_float | copy_reformat | 1 | 60 | 0.058 | 0.077 | 1.329x | 0.019 |
| value.ownership_half_to_float | copy_reformat | 1 | 60 | 0.056 | 0.075 | 1.341x | 0.019 |
| policy.g1_half_to_float | copy_reformat | 1 | 60 | 0.094 | 0.112 | 1.189x | 0.018 |
| policy.gpool_bias_add | head_elementwise | 1 | 60 | 0.109 | 0.126 | 1.151x | 0.017 |
| policy.p1_norm_silu | head_elementwise | 1 | 60 | 0.132 | 0.149 | 1.123x | 0.016 |
| frontend.initial_conv_nhwc_padding_0 | cudnn | 1 | 60 | 0.077 | 0.093 | 1.206x | 0.016 |
| input.extract_mask | head_elementwise | 1 | 60 | 0.071 | 0.086 | 1.213x | 0.015 |
| policy.p1_half_to_float | copy_reformat | 1 | 60 | 0.092 | 0.107 | 1.162x | 0.015 |
| frontend.initial_conv_nhwc_padding_1 | cudnn | 1 | 60 | 0.092 | 0.105 | 1.134x | 0.012 |
| frontend.initial_global_matmul_splitk_reduce | library_gemm | 1 | 60 | 0.077 | 0.082 | 1.074x | 0.007 |
| value.ownership_conv_splitk_reduce | library_gemm | 1 | 60 | 0.083 | 0.089 | 1.076x | 0.007 |
| value.score_bias | head_elementwise | 1 | 60 | 0.056 | 0.062 | 1.121x | 0.007 |
| value.v3_bias | head_elementwise | 1 | 60 | 0.058 | 0.064 | 1.114x | 0.007 |
| value.v2_bias_silu | head_elementwise | 1 | 60 | 0.061 | 0.066 | 1.079x | 0.005 |

## `library_gemm` logical breakdown

| logical group | positions | calls | isolated reference ms | S2 total ms | total ratio | excess ms |
|---|---:|---:|---:|---:|---:|---:|
| transformer.attention_out_projection_residual | 33 | 1980 | 17.074 | 35.735 | 2.093x | 18.661 |
| outer.post_projection_c384_to_c768_residual | 11 | 660 | 9.604 | 15.768 | 1.642x | 6.164 |
| outer.pre_projection_c768_to_c384 | 11 | 660 | 7.936 | 11.614 | 1.464x | 3.678 |
| policy.p1_conv | 1 | 60 | 0.374 | 0.682 | 1.822x | 0.308 |
| policy.g1_conv | 1 | 60 | 0.357 | 0.478 | 1.340x | 0.121 |
| policy.gpool_to_bias_matmul | 1 | 60 | 0.325 | 0.410 | 1.263x | 0.085 |
| value.v1_conv | 1 | 60 | 0.484 | 0.557 | 1.152x | 0.074 |
| frontend.initial_global_matmul | 1 | 60 | 0.157 | 0.230 | 1.458x | 0.072 |
| policy.p2_conv | 1 | 60 | 0.238 | 0.291 | 1.223x | 0.053 |
| policy.gpool_to_pass_matmul | 1 | 60 | 0.323 | 0.368 | 1.142x | 0.046 |
| value.v2_matmul | 1 | 60 | 0.574 | 0.611 | 1.065x | 0.037 |
| policy.gpool_to_pass_matmul2 | 1 | 60 | 0.140 | 0.174 | 1.240x | 0.034 |
| value.ownership_conv | 1 | 60 | 0.242 | 0.273 | 1.130x | 0.032 |
| value.v3_matmul | 1 | 60 | 0.210 | 0.231 | 1.100x | 0.021 |
| value.score_matmul | 1 | 60 | 0.211 | 0.231 | 1.095x | 0.020 |
| frontend.initial_global_matmul_splitk_reduce | 1 | 60 | 0.077 | 0.082 | 1.074x | 0.007 |
| value.ownership_conv_splitk_reduce | 1 | 60 | 0.083 | 0.089 | 1.076x | 0.007 |

## Top ordinal hotspots by summed excess

The worst peer is the highest median S2/S1 slowdown among peer families observed at least four times for that ordinal.

| rank | ordinal | logical position | family | calls | isolated us | S2 us | S2/S1 | excess ms | common peer | worst peer |
|---:|---:|---|---|---:|---:|---:|---:|---:|---|---|
| 1 | 222 | outer_07.transformer_1.block_22.ffn_linear2_residual | linear2_residual | 60 | 21.265 | 37.456 | 1.761x | 0.977 | fused_ffn (30) | wide_qkv (1.836x; 30) |
| 2 | 214 | outer_07.transformer_0.block_21.ffn_linear2_residual | linear2_residual | 60 | 21.265 | 37.361 | 1.757x | 0.976 | fused_ffn (30) | wide_qkv (1.832x; 30) |
| 3 | 202 | outer_06.transformer_2.block_20.ffn_linear2_residual | linear2_residual | 60 | 21.184 | 37.024 | 1.748x | 0.974 | fused_ffn (30) | library_gemm (1.829x; 30) |
| 4 | 230 | outer_07.transformer_2.block_23.ffn_linear2_residual | linear2_residual | 60 | 21.280 | 37.088 | 1.743x | 0.960 | fused_ffn (30) | library_gemm (1.797x; 30) |
| 5 | 146 | outer_04.transformer_2.block_14.ffn_linear2_residual | linear2_residual | 60 | 21.152 | 36.512 | 1.726x | 0.956 | fused_ffn (30) | library_gemm (1.830x; 30) |
| 6 | 298 | outer_10.transformer_0.block_30.ffn_linear2_residual | linear2_residual | 60 | 21.152 | 36.304 | 1.716x | 0.911 | fused_ffn (30) | wide_qkv (1.756x; 30) |
| 7 | 138 | outer_04.transformer_1.block_13.ffn_linear2_residual | linear2_residual | 60 | 21.073 | 34.977 | 1.660x | 0.872 | fused_ffn (30) | wide_qkv (1.768x; 30) |
| 8 | 306 | outer_10.transformer_1.block_31.ffn_linear2_residual | linear2_residual | 60 | 21.152 | 35.824 | 1.694x | 0.871 | fused_ffn (30) | wide_qkv (1.769x; 30) |
| 9 | 130 | outer_04.transformer_0.block_12.ffn_linear2_residual | linear2_residual | 60 | 21.121 | 34.864 | 1.651x | 0.836 | fused_ffn (30) | wide_qkv (1.744x; 30) |
| 10 | 194 | outer_06.transformer_1.block_19.ffn_linear2_residual | linear2_residual | 60 | 21.169 | 34.689 | 1.639x | 0.827 | fused_ffn (30) | wide_qkv (1.686x; 30) |
| 11 | 242 | outer_08.transformer_0.block_24.ffn_linear2_residual | linear2_residual | 60 | 21.265 | 34.961 | 1.644x | 0.826 | fused_ffn (30) | wide_qkv (1.686x; 30) |
| 12 | 250 | outer_08.transformer_1.block_25.ffn_linear2_residual | linear2_residual | 60 | 21.216 | 34.240 | 1.614x | 0.811 | fused_ffn (30) | wide_qkv (1.717x; 30) |
| 13 | 158 | outer_05.transformer_0.block_15.ffn_linear2_residual | linear2_residual | 60 | 21.216 | 34.257 | 1.615x | 0.801 | fused_ffn (30) | wide_qkv (1.705x; 30) |
| 14 | 314 | outer_10.transformer_2.block_32.ffn_linear2_residual | linear2_residual | 60 | 21.120 | 34.320 | 1.625x | 0.793 | fused_ffn (30) | library_gemm (1.644x; 30) |
| 15 | 46 | outer_01.transformer_0.block_03.ffn_linear2_residual | linear2_residual | 60 | 21.088 | 34.112 | 1.618x | 0.792 | fused_ffn (30) | wide_qkv (1.729x; 30) |
| 16 | 166 | outer_05.transformer_1.block_16.ffn_linear2_residual | linear2_residual | 60 | 21.025 | 33.792 | 1.607x | 0.788 | fused_ffn (30) | wide_qkv (1.743x; 30) |
| 17 | 74 | outer_02.transformer_0.block_06.ffn_linear2_residual | linear2_residual | 60 | 21.088 | 33.424 | 1.585x | 0.786 | fused_ffn (30) | wide_qkv (1.729x; 30) |
| 18 | 18 | outer_00.transformer_0.block_00.ffn_linear2_residual | linear2_residual | 60 | 21.120 | 34.144 | 1.617x | 0.779 | fused_ffn (30) | wide_qkv (1.703x; 30) |
| 19 | 82 | outer_02.transformer_1.block_07.ffn_linear2_residual | linear2_residual | 60 | 21.008 | 33.520 | 1.596x | 0.775 | fused_ffn (30) | wide_qkv (1.711x; 30) |
| 20 | 258 | outer_08.transformer_2.block_26.ffn_linear2_residual | linear2_residual | 60 | 21.200 | 34.001 | 1.604x | 0.773 | fused_ffn (30) | library_gemm (1.643x; 30) |
| 21 | 54 | outer_01.transformer_1.block_04.ffn_linear2_residual | linear2_residual | 60 | 20.944 | 33.200 | 1.585x | 0.772 | fused_ffn (30) | wide_qkv (1.730x; 30) |
| 22 | 186 | outer_06.transformer_0.block_18.ffn_linear2_residual | linear2_residual | 60 | 21.200 | 33.601 | 1.585x | 0.748 | fused_ffn (30) | wide_qkv (1.606x; 30) |
| 23 | 90 | outer_02.transformer_2.block_08.ffn_linear2_residual | linear2_residual | 60 | 21.056 | 33.280 | 1.581x | 0.743 | fused_ffn (30) | library_gemm (1.657x; 30) |
| 24 | 62 | outer_01.transformer_2.block_05.ffn_linear2_residual | linear2_residual | 60 | 20.960 | 32.720 | 1.561x | 0.738 | fused_ffn (30) | library_gemm (1.668x; 30) |
| 25 | 270 | outer_09.transformer_0.block_27.ffn_linear2_residual | linear2_residual | 60 | 21.056 | 32.864 | 1.561x | 0.727 | fused_ffn (30) | wide_qkv (1.692x; 30) |
| 26 | 26 | outer_00.transformer_1.block_01.ffn_linear2_residual | linear2_residual | 60 | 20.945 | 32.480 | 1.551x | 0.716 | fused_ffn (30) | wide_qkv (1.681x; 30) |
| 27 | 278 | outer_09.transformer_1.block_28.ffn_linear2_residual | linear2_residual | 60 | 20.848 | 32.160 | 1.543x | 0.699 | fused_ffn (30) | wide_qkv (1.665x; 30) |
| 28 | 110 | outer_03.transformer_1.block_10.ffn_linear2_residual | linear2_residual | 60 | 20.704 | 32.337 | 1.562x | 0.698 | fused_ffn (30) | wide_qkv (1.666x; 30) |
| 29 | 174 | outer_05.transformer_2.block_17.ffn_linear2_residual | linear2_residual | 60 | 20.992 | 31.904 | 1.520x | 0.688 | fused_ffn (30) | library_gemm (1.617x; 30) |
| 30 | 316 | outer_10.post_projection_c384_to_c768_residual | library_gemm | 60 | 14.624 | 25.504 | 1.744x | 0.686 | affine_silu (30) | linear2_residual (1.945x; 30) |
| 31 | 102 | outer_03.transformer_0.block_09.ffn_linear2_residual | linear2_residual | 60 | 20.928 | 32.064 | 1.532x | 0.685 | fused_ffn (30) | wide_qkv (1.648x; 30) |
| 32 | 34 | outer_00.transformer_2.block_02.ffn_linear2_residual | linear2_residual | 60 | 20.865 | 31.825 | 1.525x | 0.681 | fused_ffn (30) | library_gemm (1.614x; 30) |
| 33 | 204 | outer_06.post_projection_c384_to_c768_residual | library_gemm | 60 | 14.688 | 25.200 | 1.716x | 0.635 | affine_silu (30) | linear2_residual (1.853x; 30) |
| 34 | 286 | outer_09.transformer_2.block_29.ffn_linear2_residual | linear2_residual | 60 | 20.736 | 30.800 | 1.485x | 0.631 | fused_ffn (30) | library_gemm (1.576x; 30) |
| 35 | 118 | outer_03.transformer_2.block_11.ffn_linear2_residual | linear2_residual | 60 | 20.768 | 30.896 | 1.488x | 0.623 | fused_ffn (30) | library_gemm (1.581x; 30) |
| 36 | 232 | outer_07.post_projection_c384_to_c768_residual | library_gemm | 60 | 14.720 | 24.241 | 1.647x | 0.619 | affine_silu (30) | linear2_residual (1.886x; 30) |
| 37 | 260 | outer_08.post_projection_c384_to_c768_residual | library_gemm | 60 | 14.624 | 24.304 | 1.662x | 0.618 | affine_silu (30) | linear2_residual (1.896x; 30) |
| 38 | 148 | outer_04.post_projection_c384_to_c768_residual | library_gemm | 60 | 14.736 | 24.608 | 1.670x | 0.616 | affine_silu (30) | linear2_residual (1.862x; 30) |
| 39 | 43 | outer_01.transformer_0.block_03.attention_out_projection_residual | library_gemm | 60 | 8.512 | 18.560 | 2.180x | 0.597 | library_gemm (42) | fused_ffn (2.522x; 18) |
| 40 | 155 | outer_05.transformer_0.block_15.attention_out_projection_residual | library_gemm | 60 | 8.640 | 18.320 | 2.120x | 0.596 | library_gemm (42) | fused_ffn (2.491x; 18) |

## Full fixed-forward ordinal map

| ordinal | logical position | family | resource signature | calls | isolated us | S2 us | S2/S1 | overlap | excess ms | common peer | worst peer |
|---:|---|---|---|---:|---:|---:|---:|---:|---:|---|---|
| 0 | input.extract_mask | head_elementwise | head_elementwise; extractChannel0KernelNHWC; g10x1x1; b512x1x1; r16; s0 | 60 | 1.184 | 1.280 | 1.081x | 32.9% | 0.015 | idle (28) | idle (1.351x; 28) |
| 1 | input.mask_half_to_float | copy_reformat | copy_reformat; copyFromHalfKernel; g10x1x1; b512x1x1; r16; s0 | 60 | 0.960 | 1.152 | 1.200x | 17.9% | 0.019 | idle (37) | idle (1.400x; 37) |
| 2 | input.mask_sum | sumChannelsNCHWKernel | sumChannelsNCHWKernel; sumChannelsNCHWKernel; g1x1x13; b256x2x1; r22; s2048 | 60 | 1.632 | 1.856 | 1.137x | 43.6% | 0.023 | cudnn (29) | idle (1.392x; 4) |
| 3 | frontend.initial_conv_nhwc_padding_0 | cudnn | cudnn; nhwcAddPaddingKernel; g340x1x1; b768x1x1; r34; s0 | 60 | 1.280 | 1.472 | 1.150x | 68.3% | 0.016 | cudnn (28) | cudnn (1.288x; 28) |
| 4 | frontend.initial_conv_nhwc_padding_1 | cudnn | cudnn; nhwcAddPaddingKernel; g340x1x1; b768x1x1; r34; s0 | 60 | 1.536 | 1.728 | 1.125x | 59.2% | 0.012 | cudnn (42) | cudnn (1.125x; 42) |
| 5 | frontend.initial_conv | cudnn | cudnn; Kernel; g296x3x1; b128x1x1; r94; s81920 | 60 | 19.569 | 21.697 | 1.109x | 19.5% | 0.131 | cudnn (27) | cudnn (1.115x; 27) |
| 6 | frontend.initial_global_matmul | library_gemm | library_gemm; Kernel2; g8x1x3; b128x1x1; r128; s24576 | 60 | 2.624 | 3.760 | 1.433x | 87.1% | 0.072 | cudnn (30) | cudnn (1.622x; 30) |
| 7 | frontend.initial_global_matmul_splitk_reduce | library_gemm | library_gemm; splitKreduce_kernel; g24x1x1; b32x16x1; r49; s0 | 60 | 1.280 | 1.344 | 1.050x | 83.3% | 0.007 | affine_silu (20) | cudnn (1.100x; 20) |
| 8 | frontend.initial_global_broadcast_add | head_elementwise | head_elementwise; addNCBiasInplaceNHWCHalfKernel; g3x361x13; b256x1x1; r16; s0 | 60 | 7.729 | 8.128 | 1.052x | 57.2% | 0.082 | library_gemm (30) | library_gemm (1.416x; 30) |
| 9 | outer_00.pre_norm_silu | affine_silu | affine_silu; affineSiluHalf2Kernel; g4693x1x1; b384x1x1; r17; s0 | 60 | 4.960 | 6.064 | 1.223x | 61.4% | 0.079 | library_gemm (50) | library_gemm (1.284x; 50) |
| 10 | outer_00.pre_projection_c768_to_c384 | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 60 | 11.680 | 14.656 | 1.255x | 94.8% | 0.205 | wide_qkv (30) | head_elementwise (1.264x; 20) |
| 11 | outer_00.transformer_0.block_00.attention_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Half2Kernel; g1174x1x1; b128x1x1; r38; s0 | 60 | 2.624 | 4.384 | 1.671x | 96.7% | 0.116 | library_gemm (30) | wide_qkv (1.774x; 30) |
| 12 | outer_00.transformer_0.block_00.attention_qkv_projection | wide_qkv | wide_qkv; wide_qkv_kernel; g9x37x1; b128x1x1; r136; s65536 | 60 | 18.784 | 22.688 | 1.208x | 69.2% | 0.308 | library_gemm (30) | library_gemm (1.470x; 30) |
| 13 | outer_00.transformer_0.block_00.attention_qk_rope | qk_rope | qk_rope; fusedQKRoPE19HalfKernel; g361x13x1; b192x1x1; r16; s0 | 60 | 4.064 | 7.152 | 1.760x | 97.5% | 0.197 | fa4 (30) | wide_qkv (2.252x; 30) |
| 14 | outer_00.transformer_0.block_00.attention_fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 60 | 11.777 | 14.560 | 1.236x | 48.6% | 0.182 | library_gemm (27) | fa4 (1.395x; 6) |
| 15 | outer_00.transformer_0.block_00.attention_out_projection_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 60 | 8.576 | 15.184 | 1.770x | 95.9% | 0.529 | library_gemm (34) | fused_ffn (2.429x; 26) |
| 16 | outer_00.transformer_0.block_00.ffn_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Half2Kernel; g1174x1x1; b128x1x1; r38; s0 | 60 | 2.688 | 4.688 | 1.744x | 100.0% | 0.122 | fused_ffn (30) | fused_ffn (2.244x; 30) |
| 17 | outer_00.transformer_0.block_00.ffn_linear1_gate_swiglu | fused_ffn | fused_ffn; fused_ffn_kernel; g18x37x1; b128x1x1; r146; s32768 | 60 | 41.409 | 46.065 | 1.112x | 76.0% | 0.254 | fused_ffn (30) | fused_ffn (1.143x; 30) |
| 18 | outer_00.transformer_0.block_00.ffn_linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 60 | 21.120 | 34.144 | 1.617x | 97.9% | 0.779 | fused_ffn (30) | wide_qkv (1.703x; 30) |
| 19 | outer_00.transformer_1.block_01.attention_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Half2Kernel; g1174x1x1; b128x1x1; r38; s0 | 60 | 2.688 | 4.608 | 1.714x | 100.0% | 0.121 | linear2_residual (30) | wide_qkv (2.042x; 30) |
| 20 | outer_00.transformer_1.block_01.attention_qkv_projection | wide_qkv | wide_qkv; wide_qkv_kernel; g9x37x1; b128x1x1; r136; s65536 | 60 | 19.536 | 26.224 | 1.342x | 83.5% | 0.422 | linear2_residual (30) | linear2_residual (1.586x; 30) |
| 21 | outer_00.transformer_1.block_01.attention_qk_rope | qk_rope | qk_rope; fusedQKRoPE19HalfKernel; g361x13x1; b192x1x1; r16; s0 | 60 | 4.160 | 7.504 | 1.804x | 100.0% | 0.209 | fa4 (30) | wide_qkv (2.273x; 30) |
| 22 | outer_00.transformer_1.block_01.attention_fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 60 | 12.064 | 14.816 | 1.228x | 44.8% | 0.170 | library_gemm (30) | library_gemm (1.232x; 30) |
| 23 | outer_00.transformer_1.block_01.attention_out_projection_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 60 | 8.704 | 17.776 | 2.042x | 95.9% | 0.542 | library_gemm (32) | fused_ffn (2.393x; 28) |
| 24 | outer_00.transformer_1.block_01.ffn_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Half2Kernel; g1174x1x1; b128x1x1; r38; s0 | 60 | 2.720 | 4.944 | 1.818x | 100.0% | 0.132 | fused_ffn (30) | fused_ffn (2.376x; 30) |
| 25 | outer_00.transformer_1.block_01.ffn_linear1_gate_swiglu | fused_ffn | fused_ffn; fused_ffn_kernel; g18x37x1; b128x1x1; r146; s32768 | 60 | 43.104 | 44.337 | 1.029x | 72.7% | 0.122 | fused_ffn (31) | fused_ffn (1.094x; 31) |
| 26 | outer_00.transformer_1.block_01.ffn_linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 60 | 20.945 | 32.480 | 1.551x | 98.2% | 0.716 | fused_ffn (30) | wide_qkv (1.681x; 30) |
| 27 | outer_00.transformer_2.block_02.attention_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Half2Kernel; g1174x1x1; b128x1x1; r38; s0 | 60 | 2.672 | 4.544 | 1.701x | 100.0% | 0.122 | linear2_residual (30) | wide_qkv (2.060x; 30) |
| 28 | outer_00.transformer_2.block_02.attention_qkv_projection | wide_qkv | wide_qkv; wide_qkv_kernel; g9x37x1; b128x1x1; r136; s65536 | 60 | 19.296 | 24.992 | 1.295x | 81.8% | 0.343 | linear2_residual (30) | linear2_residual (1.470x; 30) |
| 29 | outer_00.transformer_2.block_02.attention_qk_rope | qk_rope | qk_rope; fusedQKRoPE19HalfKernel; g361x13x1; b192x1x1; r16; s0 | 60 | 4.160 | 7.536 | 1.812x | 99.2% | 0.198 | fa4 (30) | wide_qkv (2.242x; 30) |
| 30 | outer_00.transformer_2.block_02.attention_fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 60 | 11.936 | 14.784 | 1.239x | 48.1% | 0.181 | library_gemm (29) | qk_rope (1.265x; 29) |
| 31 | outer_00.transformer_2.block_02.attention_out_projection_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 60 | 8.592 | 16.112 | 1.875x | 95.8% | 0.537 | library_gemm (35) | fused_ffn (2.425x; 25) |
| 32 | outer_00.transformer_2.block_02.ffn_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Half2Kernel; g1174x1x1; b128x1x1; r38; s0 | 60 | 2.688 | 4.864 | 1.810x | 100.0% | 0.133 | fused_ffn (30) | fused_ffn (2.417x; 30) |
| 33 | outer_00.transformer_2.block_02.ffn_linear1_gate_swiglu | fused_ffn | fused_ffn; fused_ffn_kernel; g18x37x1; b128x1x1; r146; s32768 | 60 | 42.785 | 45.072 | 1.053x | 73.5% | 0.153 | fused_ffn (32) | fused_ffn (1.119x; 32) |
| 34 | outer_00.transformer_2.block_02.ffn_linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 60 | 20.865 | 31.825 | 1.525x | 98.1% | 0.681 | fused_ffn (30) | library_gemm (1.614x; 30) |
| 35 | outer_00.post_norm_silu | affine_silu | affine_silu; affineSiluHalf2Kernel; g4693x1x1; b192x1x1; r17; s0 | 60 | 3.360 | 4.080 | 1.214x | 100.0% | 0.044 | library_gemm (30) | linear2_residual (1.262x; 30) |
| 36 | outer_00.post_projection_c384_to_c768_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b256x1x1; r154; s73728 | 60 | 14.416 | 22.176 | 1.538x | 95.9% | 0.475 | affine_silu (30) | linear2_residual (1.638x; 30) |
| 37 | outer_01.pre_norm_silu | affine_silu | affine_silu; affineSiluHalf2Kernel; g4693x1x1; b384x1x1; r17; s0 | 60 | 5.152 | 8.176 | 1.587x | 100.0% | 0.182 | library_gemm (60) | library_gemm (1.587x; 60) |
| 38 | outer_01.pre_projection_c768_to_c384 | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 60 | 12.000 | 17.024 | 1.419x | 94.4% | 0.323 | wide_qkv (30) | library_gemm (1.592x; 29) |
| 39 | outer_01.transformer_0.block_03.attention_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Half2Kernel; g1174x1x1; b128x1x1; r38; s0 | 60 | 2.688 | 4.912 | 1.827x | 100.0% | 0.142 | library_gemm (30) | wide_qkv (2.167x; 30) |
| 40 | outer_01.transformer_0.block_03.attention_qkv_projection | wide_qkv | wide_qkv; wide_qkv_kernel; g9x37x1; b128x1x1; r136; s65536 | 60 | 19.296 | 24.560 | 1.273x | 65.7% | 0.329 | library_gemm (30) | library_gemm (1.388x; 30) |
| 41 | outer_01.transformer_0.block_03.attention_qk_rope | qk_rope | qk_rope; fusedQKRoPE19HalfKernel; g361x13x1; b192x1x1; r16; s0 | 60 | 4.096 | 7.521 | 1.836x | 100.0% | 0.207 | fa4 (30) | wide_qkv (2.262x; 30) |
| 42 | outer_01.transformer_0.block_03.attention_fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 60 | 12.000 | 15.904 | 1.325x | 49.5% | 0.233 | library_gemm (30) | qk_rope (1.347x; 30) |
| 43 | outer_01.transformer_0.block_03.attention_out_projection_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 60 | 8.512 | 18.560 | 2.180x | 96.1% | 0.597 | library_gemm (42) | fused_ffn (2.522x; 18) |
| 44 | outer_01.transformer_0.block_03.ffn_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Half2Kernel; g1174x1x1; b128x1x1; r38; s0 | 60 | 2.688 | 4.768 | 1.774x | 100.0% | 0.127 | fused_ffn (30) | fused_ffn (2.286x; 30) |
| 45 | outer_01.transformer_0.block_03.ffn_linear1_gate_swiglu | fused_ffn | fused_ffn; fused_ffn_kernel; g18x37x1; b128x1x1; r146; s32768 | 60 | 42.097 | 46.480 | 1.104x | 75.8% | 0.271 | fused_ffn (30) | fused_ffn (1.171x; 30) |
| 46 | outer_01.transformer_0.block_03.ffn_linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 60 | 21.088 | 34.112 | 1.618x | 97.9% | 0.792 | fused_ffn (30) | wide_qkv (1.729x; 30) |
| 47 | outer_01.transformer_1.block_04.attention_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Half2Kernel; g1174x1x1; b128x1x1; r38; s0 | 60 | 2.688 | 4.528 | 1.685x | 100.0% | 0.120 | linear2_residual (30) | wide_qkv (2.060x; 30) |
| 48 | outer_01.transformer_1.block_04.attention_qkv_projection | wide_qkv | wide_qkv; wide_qkv_kernel; g9x37x1; b128x1x1; r136; s65536 | 60 | 19.473 | 25.856 | 1.328x | 82.8% | 0.433 | linear2_residual (30) | linear2_residual (1.585x; 30) |
| 49 | outer_01.transformer_1.block_04.attention_qk_rope | qk_rope | qk_rope; fusedQKRoPE19HalfKernel; g361x13x1; b192x1x1; r16; s0 | 60 | 4.160 | 7.712 | 1.854x | 100.0% | 0.214 | fa4 (30) | wide_qkv (2.300x; 30) |
| 50 | outer_01.transformer_1.block_04.attention_fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 60 | 12.064 | 15.200 | 1.260x | 46.2% | 0.191 | library_gemm (30) | qk_rope (1.268x; 30) |
| 51 | outer_01.transformer_1.block_04.attention_out_projection_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 60 | 8.576 | 17.808 | 2.076x | 95.8% | 0.564 | library_gemm (33) | fused_ffn (2.470x; 27) |
| 52 | outer_01.transformer_1.block_04.ffn_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Half2Kernel; g1174x1x1; b128x1x1; r38; s0 | 60 | 2.688 | 4.960 | 1.845x | 100.0% | 0.138 | fused_ffn (30) | fused_ffn (2.446x; 30) |
| 53 | outer_01.transformer_1.block_04.ffn_linear1_gate_swiglu | fused_ffn | fused_ffn; fused_ffn_kernel; g18x37x1; b128x1x1; r146; s32768 | 60 | 43.200 | 45.344 | 1.050x | 72.8% | 0.144 | fused_ffn (30) | fused_ffn (1.101x; 30) |
| 54 | outer_01.transformer_1.block_04.ffn_linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 60 | 20.944 | 33.200 | 1.585x | 98.3% | 0.772 | fused_ffn (30) | wide_qkv (1.730x; 30) |
| 55 | outer_01.transformer_2.block_05.attention_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Half2Kernel; g1174x1x1; b128x1x1; r38; s0 | 60 | 2.688 | 4.496 | 1.673x | 100.0% | 0.119 | linear2_residual (30) | wide_qkv (2.060x; 30) |
| 56 | outer_01.transformer_2.block_05.attention_qkv_projection | wide_qkv | wide_qkv; wide_qkv_kernel; g9x37x1; b128x1x1; r136; s65536 | 60 | 19.281 | 25.328 | 1.314x | 81.9% | 0.369 | linear2_residual (30) | linear2_residual (1.510x; 30) |
| 57 | outer_01.transformer_2.block_05.attention_qk_rope | qk_rope | qk_rope; fusedQKRoPE19HalfKernel; g361x13x1; b192x1x1; r16; s0 | 60 | 4.096 | 7.504 | 1.832x | 100.0% | 0.206 | fa4 (30) | wide_qkv (2.266x; 30) |
| 58 | outer_01.transformer_2.block_05.attention_fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 60 | 11.968 | 14.912 | 1.246x | 47.3% | 0.182 | library_gemm (30) | qk_rope (1.266x; 30) |
| 59 | outer_01.transformer_2.block_05.attention_out_projection_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 60 | 8.528 | 17.680 | 2.073x | 95.9% | 0.564 | library_gemm (31) | fused_ffn (2.473x; 29) |
| 60 | outer_01.transformer_2.block_05.ffn_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Half2Kernel; g1174x1x1; b128x1x1; r38; s0 | 60 | 2.704 | 4.992 | 1.846x | 100.0% | 0.140 | fused_ffn (30) | fused_ffn (2.444x; 30) |
| 61 | outer_01.transformer_2.block_05.ffn_linear1_gate_swiglu | fused_ffn | fused_ffn; fused_ffn_kernel; g18x37x1; b128x1x1; r146; s32768 | 60 | 42.961 | 46.049 | 1.072x | 73.0% | 0.173 | fused_ffn (30) | fused_ffn (1.110x; 30) |
| 62 | outer_01.transformer_2.block_05.ffn_linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 60 | 20.960 | 32.720 | 1.561x | 98.2% | 0.738 | fused_ffn (30) | library_gemm (1.668x; 30) |
| 63 | outer_01.post_norm_silu | affine_silu | affine_silu; affineSiluHalf2Kernel; g4693x1x1; b192x1x1; r17; s0 | 60 | 3.360 | 4.144 | 1.233x | 100.0% | 0.046 | library_gemm (30) | linear2_residual (1.248x; 30) |
| 64 | outer_01.post_projection_c384_to_c768_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b256x1x1; r154; s73728 | 60 | 14.480 | 22.945 | 1.585x | 95.9% | 0.527 | affine_silu (30) | linear2_residual (1.724x; 30) |
| 65 | outer_02.pre_norm_silu | affine_silu | affine_silu; affineSiluHalf2Kernel; g4693x1x1; b384x1x1; r17; s0 | 60 | 5.120 | 8.304 | 1.622x | 100.0% | 0.189 | library_gemm (60) | library_gemm (1.622x; 60) |
| 66 | outer_02.pre_projection_c768_to_c384 | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 60 | 12.001 | 17.456 | 1.455x | 94.4% | 0.346 | wide_qkv (30) | library_gemm (1.656x; 29) |
| 67 | outer_02.transformer_0.block_06.attention_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Half2Kernel; g1174x1x1; b128x1x1; r38; s0 | 60 | 2.672 | 4.976 | 1.862x | 100.0% | 0.153 | library_gemm (30) | wide_qkv (2.299x; 30) |
| 68 | outer_02.transformer_0.block_06.attention_qkv_projection | wide_qkv | wide_qkv; wide_qkv_kernel; g9x37x1; b128x1x1; r136; s65536 | 60 | 19.312 | 24.736 | 1.281x | 65.0% | 0.333 | qk_rope (30) | library_gemm (1.384x; 24) |
| 69 | outer_02.transformer_0.block_06.attention_qk_rope | qk_rope | qk_rope; fusedQKRoPE19HalfKernel; g361x13x1; b192x1x1; r16; s0 | 60 | 4.160 | 7.760 | 1.865x | 100.0% | 0.216 | fa4 (30) | wide_qkv (2.327x; 30) |
| 70 | outer_02.transformer_0.block_06.attention_fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 60 | 12.016 | 16.032 | 1.334x | 48.3% | 0.239 | library_gemm (30) | qk_rope (1.353x; 30) |
| 71 | outer_02.transformer_0.block_06.attention_out_projection_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 60 | 8.640 | 18.224 | 2.109x | 95.9% | 0.586 | library_gemm (53) | fused_ffn (2.500x; 7) |
| 72 | outer_02.transformer_0.block_06.ffn_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Half2Kernel; g1174x1x1; b128x1x1; r38; s0 | 60 | 2.720 | 4.896 | 1.800x | 100.0% | 0.133 | fused_ffn (30) | fused_ffn (2.324x; 30) |
| 73 | outer_02.transformer_0.block_06.ffn_linear1_gate_swiglu | fused_ffn | fused_ffn; fused_ffn_kernel; g18x37x1; b128x1x1; r146; s32768 | 60 | 43.105 | 47.040 | 1.091x | 73.8% | 0.280 | fused_ffn (36) | fused_ffn (1.182x; 36) |
| 74 | outer_02.transformer_0.block_06.ffn_linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 60 | 21.088 | 33.424 | 1.585x | 97.9% | 0.786 | fused_ffn (30) | wide_qkv (1.729x; 30) |
| 75 | outer_02.transformer_1.block_07.attention_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Half2Kernel; g1174x1x1; b128x1x1; r38; s0 | 60 | 2.704 | 4.640 | 1.716x | 100.0% | 0.128 | linear2_residual (30) | wide_qkv (2.095x; 30) |
| 76 | outer_02.transformer_1.block_07.attention_qkv_projection | wide_qkv | wide_qkv; wide_qkv_kernel; g9x37x1; b128x1x1; r136; s65536 | 60 | 19.520 | 25.680 | 1.316x | 82.4% | 0.398 | linear2_residual (30) | linear2_residual (1.539x; 30) |
| 77 | outer_02.transformer_1.block_07.attention_qk_rope | qk_rope | qk_rope; fusedQKRoPE19HalfKernel; g361x13x1; b192x1x1; r16; s0 | 60 | 4.192 | 7.456 | 1.779x | 100.0% | 0.200 | fa4 (30) | wide_qkv (2.214x; 30) |
| 78 | outer_02.transformer_1.block_07.attention_fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 60 | 12.032 | 15.168 | 1.261x | 47.8% | 0.191 | library_gemm (30) | qk_rope (1.279x; 30) |
| 79 | outer_02.transformer_1.block_07.attention_out_projection_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 60 | 8.624 | 17.840 | 2.069x | 95.9% | 0.566 | library_gemm (33) | fused_ffn (2.467x; 27) |
| 80 | outer_02.transformer_1.block_07.ffn_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Half2Kernel; g1174x1x1; b128x1x1; r38; s0 | 60 | 2.720 | 4.768 | 1.753x | 100.0% | 0.128 | fused_ffn (30) | fused_ffn (2.329x; 30) |
| 81 | outer_02.transformer_1.block_07.ffn_linear1_gate_swiglu | fused_ffn | fused_ffn; fused_ffn_kernel; g18x37x1; b128x1x1; r146; s32768 | 60 | 43.472 | 47.953 | 1.103x | 76.1% | 0.260 | fused_ffn (30) | fused_ffn (1.153x; 30) |
| 82 | outer_02.transformer_1.block_07.ffn_linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 60 | 21.008 | 33.520 | 1.596x | 98.3% | 0.775 | fused_ffn (30) | wide_qkv (1.711x; 30) |
| 83 | outer_02.transformer_2.block_08.attention_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Half2Kernel; g1174x1x1; b128x1x1; r38; s0 | 60 | 2.688 | 4.512 | 1.679x | 100.0% | 0.119 | linear2_residual (30) | wide_qkv (2.024x; 30) |
| 84 | outer_02.transformer_2.block_08.attention_qkv_projection | wide_qkv | wide_qkv; wide_qkv_kernel; g9x37x1; b128x1x1; r136; s65536 | 60 | 19.488 | 26.288 | 1.349x | 82.6% | 0.427 | linear2_residual (30) | linear2_residual (1.594x; 30) |
| 85 | outer_02.transformer_2.block_08.attention_qk_rope | qk_rope | qk_rope; fusedQKRoPE19HalfKernel; g361x13x1; b192x1x1; r16; s0 | 60 | 4.160 | 7.552 | 1.815x | 100.0% | 0.203 | fa4 (30) | wide_qkv (2.250x; 30) |
| 86 | outer_02.transformer_2.block_08.attention_fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 60 | 12.064 | 15.200 | 1.260x | 47.1% | 0.191 | library_gemm (30) | qk_rope (1.285x; 30) |
| 87 | outer_02.transformer_2.block_08.attention_out_projection_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 60 | 8.640 | 17.648 | 2.043x | 95.9% | 0.556 | library_gemm (32) | fused_ffn (2.419x; 28) |
| 88 | outer_02.transformer_2.block_08.ffn_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Half2Kernel; g1174x1x1; b128x1x1; r38; s0 | 60 | 2.688 | 4.800 | 1.786x | 100.0% | 0.129 | fused_ffn (30) | fused_ffn (2.333x; 30) |
| 89 | outer_02.transformer_2.block_08.ffn_linear1_gate_swiglu | fused_ffn | fused_ffn; fused_ffn_kernel; g18x37x1; b128x1x1; r146; s32768 | 60 | 43.552 | 46.769 | 1.074x | 74.4% | 0.191 | fused_ffn (30) | fused_ffn (1.122x; 30) |
| 90 | outer_02.transformer_2.block_08.ffn_linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 60 | 21.056 | 33.280 | 1.581x | 98.2% | 0.743 | fused_ffn (30) | library_gemm (1.657x; 30) |
| 91 | outer_02.post_norm_silu | affine_silu | affine_silu; affineSiluHalf2Kernel; g4693x1x1; b192x1x1; r17; s0 | 60 | 3.392 | 4.208 | 1.241x | 100.0% | 0.049 | library_gemm (30) | linear2_residual (1.264x; 30) |
| 92 | outer_02.post_projection_c384_to_c768_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b256x1x1; r154; s73728 | 60 | 14.560 | 23.633 | 1.623x | 95.9% | 0.561 | affine_silu (30) | linear2_residual (1.765x; 30) |
| 93 | outer_03.pre_norm_silu | affine_silu | affine_silu; affineSiluHalf2Kernel; g4693x1x1; b384x1x1; r17; s0 | 60 | 5.152 | 8.432 | 1.637x | 100.0% | 0.199 | library_gemm (60) | library_gemm (1.637x; 60) |
| 94 | outer_03.pre_projection_c768_to_c384 | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 60 | 12.064 | 17.360 | 1.439x | 94.5% | 0.354 | wide_qkv (30) | library_gemm (1.669x; 25) |
| 95 | outer_03.transformer_0.block_09.attention_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Half2Kernel; g1174x1x1; b128x1x1; r38; s0 | 60 | 2.688 | 5.072 | 1.887x | 100.0% | 0.145 | library_gemm (30) | wide_qkv (2.208x; 30) |
| 96 | outer_03.transformer_0.block_09.attention_qkv_projection | wide_qkv | wide_qkv; wide_qkv_kernel; g9x37x1; b128x1x1; r136; s65536 | 60 | 19.440 | 25.233 | 1.298x | 65.5% | 0.353 | library_gemm (30) | library_gemm (1.403x; 30) |
| 97 | outer_03.transformer_0.block_09.attention_qk_rope | qk_rope | qk_rope; fusedQKRoPE19HalfKernel; g361x13x1; b192x1x1; r16; s0 | 60 | 4.192 | 7.824 | 1.866x | 100.0% | 0.221 | fa4 (30) | wide_qkv (2.344x; 30) |
| 98 | outer_03.transformer_0.block_09.attention_fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 60 | 12.081 | 16.000 | 1.324x | 49.1% | 0.263 | library_gemm (30) | qk_rope (1.364x; 30) |
| 99 | outer_03.transformer_0.block_09.attention_out_projection_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 60 | 8.576 | 18.224 | 2.125x | 96.0% | 0.593 | library_gemm (45) | fused_ffn (2.496x; 15) |
| 100 | outer_03.transformer_0.block_09.ffn_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Half2Kernel; g1174x1x1; b128x1x1; r38; s0 | 60 | 2.720 | 4.992 | 1.835x | 100.0% | 0.140 | fused_ffn (30) | fused_ffn (2.412x; 30) |
| 101 | outer_03.transformer_0.block_09.ffn_linear1_gate_swiglu | fused_ffn | fused_ffn; fused_ffn_kernel; g18x37x1; b128x1x1; r146; s32768 | 60 | 43.153 | 44.721 | 1.036x | 71.8% | 0.160 | fused_ffn (31) | fused_ffn (1.115x; 31) |
| 102 | outer_03.transformer_0.block_09.ffn_linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 60 | 20.928 | 32.064 | 1.532x | 97.9% | 0.685 | fused_ffn (30) | wide_qkv (1.648x; 30) |
| 103 | outer_03.transformer_1.block_10.attention_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Half2Kernel; g1174x1x1; b128x1x1; r38; s0 | 60 | 2.688 | 4.432 | 1.649x | 100.0% | 0.119 | linear2_residual (30) | wide_qkv (2.036x; 30) |
| 104 | outer_03.transformer_1.block_10.attention_qkv_projection | wide_qkv | wide_qkv; wide_qkv_kernel; g9x37x1; b128x1x1; r136; s65536 | 60 | 19.312 | 24.945 | 1.292x | 82.1% | 0.318 | linear2_residual (30) | linear2_residual (1.432x; 30) |
| 105 | outer_03.transformer_1.block_10.attention_qk_rope | qk_rope | qk_rope; fusedQKRoPE19HalfKernel; g361x13x1; b192x1x1; r16; s0 | 60 | 4.128 | 7.440 | 1.802x | 100.0% | 0.202 | fa4 (30) | wide_qkv (2.225x; 30) |
| 106 | outer_03.transformer_1.block_10.attention_fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 60 | 11.905 | 14.688 | 1.234x | 47.1% | 0.168 | library_gemm (30) | qk_rope (1.247x; 30) |
| 107 | outer_03.transformer_1.block_10.attention_out_projection_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 60 | 8.512 | 17.440 | 2.049x | 95.9% | 0.551 | library_gemm (31) | fused_ffn (2.447x; 29) |
| 108 | outer_03.transformer_1.block_10.ffn_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Half2Kernel; g1174x1x1; b128x1x1; r38; s0 | 60 | 2.688 | 4.832 | 1.798x | 100.0% | 0.134 | fused_ffn (30) | fused_ffn (2.393x; 30) |
| 109 | outer_03.transformer_1.block_10.ffn_linear1_gate_swiglu | fused_ffn | fused_ffn; fused_ffn_kernel; g18x37x1; b128x1x1; r146; s32768 | 60 | 42.352 | 43.632 | 1.030x | 72.8% | 0.128 | fused_ffn (30) | fused_ffn (1.097x; 30) |
| 110 | outer_03.transformer_1.block_10.ffn_linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 60 | 20.704 | 32.337 | 1.562x | 98.3% | 0.698 | fused_ffn (30) | wide_qkv (1.666x; 30) |
| 111 | outer_03.transformer_2.block_11.attention_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Half2Kernel; g1174x1x1; b128x1x1; r38; s0 | 60 | 2.640 | 4.497 | 1.703x | 100.0% | 0.120 | linear2_residual (30) | wide_qkv (2.049x; 30) |
| 112 | outer_03.transformer_2.block_11.attention_qkv_projection | wide_qkv | wide_qkv; wide_qkv_kernel; g9x37x1; b128x1x1; r136; s65536 | 60 | 19.152 | 24.768 | 1.293x | 81.6% | 0.329 | linear2_residual (30) | linear2_residual (1.458x; 30) |
| 113 | outer_03.transformer_2.block_11.attention_qk_rope | qk_rope | qk_rope; fusedQKRoPE19HalfKernel; g361x13x1; b192x1x1; r16; s0 | 60 | 4.096 | 7.248 | 1.770x | 100.0% | 0.197 | fa4 (30) | wide_qkv (2.234x; 30) |
| 114 | outer_03.transformer_2.block_11.attention_fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 60 | 11.872 | 14.688 | 1.237x | 47.9% | 0.174 | library_gemm (30) | qk_rope (1.260x; 30) |
| 115 | outer_03.transformer_2.block_11.attention_out_projection_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 60 | 8.448 | 17.392 | 2.059x | 95.9% | 0.552 | library_gemm (33) | fused_ffn (2.462x; 27) |
| 116 | outer_03.transformer_2.block_11.ffn_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Half2Kernel; g1174x1x1; b128x1x1; r38; s0 | 60 | 2.688 | 4.848 | 1.804x | 100.0% | 0.133 | fused_ffn (30) | fused_ffn (2.405x; 30) |
| 117 | outer_03.transformer_2.block_11.ffn_linear1_gate_swiglu | fused_ffn | fused_ffn; fused_ffn_kernel; g18x37x1; b128x1x1; r146; s32768 | 60 | 42.432 | 44.177 | 1.041x | 73.7% | 0.140 | fused_ffn (33) | fused_ffn (1.112x; 33) |
| 118 | outer_03.transformer_2.block_11.ffn_linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 60 | 20.768 | 30.896 | 1.488x | 98.1% | 0.623 | fused_ffn (30) | library_gemm (1.581x; 30) |
| 119 | outer_03.post_norm_silu | affine_silu | affine_silu; affineSiluHalf2Kernel; g4693x1x1; b192x1x1; r17; s0 | 60 | 3.328 | 4.032 | 1.212x | 100.0% | 0.042 | library_gemm (30) | library_gemm (1.212x; 30) |
| 120 | outer_03.post_projection_c384_to_c768_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b256x1x1; r154; s73728 | 60 | 14.320 | 21.616 | 1.509x | 95.8% | 0.452 | affine_silu (30) | linear2_residual (1.617x; 30) |
| 121 | outer_04.pre_norm_silu | affine_silu | affine_silu; affineSiluHalf2Kernel; g4693x1x1; b384x1x1; r17; s0 | 60 | 5.056 | 7.984 | 1.579x | 100.0% | 0.182 | library_gemm (60) | library_gemm (1.579x; 60) |
| 122 | outer_04.pre_projection_c768_to_c384 | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 60 | 11.904 | 16.945 | 1.423x | 94.3% | 0.314 | library_gemm (30) | library_gemm (1.597x; 30) |
| 123 | outer_04.transformer_0.block_12.attention_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Half2Kernel; g1174x1x1; b128x1x1; r38; s0 | 60 | 2.656 | 4.768 | 1.795x | 100.0% | 0.145 | library_gemm (30) | wide_qkv (2.229x; 30) |
| 124 | outer_04.transformer_0.block_12.attention_qkv_projection | wide_qkv | wide_qkv; wide_qkv_kernel; g9x37x1; b128x1x1; r136; s65536 | 60 | 19.104 | 24.112 | 1.262x | 65.0% | 0.295 | qk_rope (30) | library_gemm (1.343x; 28) |
| 125 | outer_04.transformer_0.block_12.attention_qk_rope | qk_rope | qk_rope; fusedQKRoPE19HalfKernel; g361x13x1; b192x1x1; r16; s0 | 60 | 4.128 | 7.712 | 1.868x | 100.0% | 0.212 | fa4 (30) | wide_qkv (2.279x; 30) |
| 126 | outer_04.transformer_0.block_12.attention_fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 60 | 11.904 | 15.552 | 1.306x | 47.4% | 0.220 | library_gemm (30) | qk_rope (1.324x; 30) |
| 127 | outer_04.transformer_0.block_12.attention_out_projection_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 60 | 8.608 | 18.160 | 2.110x | 96.1% | 0.581 | library_gemm (41) | fused_ffn (2.476x; 19) |
| 128 | outer_04.transformer_0.block_12.ffn_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Half2Kernel; g1174x1x1; b128x1x1; r38; s0 | 60 | 2.704 | 4.736 | 1.751x | 100.0% | 0.130 | fused_ffn (30) | fused_ffn (2.296x; 30) |
| 129 | outer_04.transformer_0.block_12.ffn_linear1_gate_swiglu | fused_ffn | fused_ffn; fused_ffn_kernel; g18x37x1; b128x1x1; r146; s32768 | 60 | 41.184 | 44.672 | 1.085x | 76.9% | 0.237 | fused_ffn (30) | fused_ffn (1.138x; 30) |
| 130 | outer_04.transformer_0.block_12.ffn_linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 60 | 21.121 | 34.864 | 1.651x | 98.0% | 0.836 | fused_ffn (30) | wide_qkv (1.744x; 30) |
| 131 | outer_04.transformer_1.block_13.attention_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Half2Kernel; g1174x1x1; b128x1x1; r38; s0 | 60 | 2.688 | 4.465 | 1.661x | 100.0% | 0.127 | linear2_residual (30) | wide_qkv (2.131x; 30) |
| 132 | outer_04.transformer_1.block_13.attention_qkv_projection | wide_qkv | wide_qkv; wide_qkv_kernel; g9x37x1; b128x1x1; r136; s65536 | 60 | 19.504 | 26.529 | 1.360x | 83.3% | 0.449 | linear2_residual (30) | linear2_residual (1.596x; 30) |
| 133 | outer_04.transformer_1.block_13.attention_qk_rope | qk_rope | qk_rope; fusedQKRoPE19HalfKernel; g361x13x1; b192x1x1; r16; s0 | 60 | 4.128 | 7.585 | 1.837x | 98.3% | 0.211 | fa4 (30) | wide_qkv (2.306x; 30) |
| 134 | outer_04.transformer_1.block_13.attention_fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 60 | 12.032 | 15.168 | 1.261x | 47.1% | 0.199 | library_gemm (28) | fa4 (1.400x; 4) |
| 135 | outer_04.transformer_1.block_13.attention_out_projection_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 60 | 8.672 | 16.496 | 1.902x | 95.9% | 0.540 | library_gemm (33) | fused_ffn (2.410x; 27) |
| 136 | outer_04.transformer_1.block_13.ffn_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Half2Kernel; g1174x1x1; b128x1x1; r38; s0 | 60 | 2.688 | 4.448 | 1.655x | 100.0% | 0.123 | fused_ffn (30) | fused_ffn (2.274x; 30) |
| 137 | outer_04.transformer_1.block_13.ffn_linear1_gate_swiglu | fused_ffn | fused_ffn; fused_ffn_kernel; g18x37x1; b128x1x1; r146; s32768 | 60 | 43.248 | 46.257 | 1.070x | 80.1% | 0.173 | fused_ffn (30) | fused_ffn (1.103x; 30) |
| 138 | outer_04.transformer_1.block_13.ffn_linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 60 | 21.073 | 34.977 | 1.660x | 98.3% | 0.872 | fused_ffn (30) | wide_qkv (1.768x; 30) |
| 139 | outer_04.transformer_2.block_14.attention_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Half2Kernel; g1174x1x1; b128x1x1; r38; s0 | 60 | 2.688 | 4.704 | 1.750x | 100.0% | 0.131 | linear2_residual (30) | wide_qkv (2.167x; 30) |
| 140 | outer_04.transformer_2.block_14.attention_qkv_projection | wide_qkv | wide_qkv; wide_qkv_kernel; g9x37x1; b128x1x1; r136; s65536 | 60 | 19.488 | 27.456 | 1.409x | 82.9% | 0.490 | linear2_residual (30) | linear2_residual (1.665x; 30) |
| 141 | outer_04.transformer_2.block_14.attention_qk_rope | qk_rope | qk_rope; fusedQKRoPE19HalfKernel; g361x13x1; b192x1x1; r16; s0 | 60 | 4.192 | 7.504 | 1.790x | 100.0% | 0.207 | fa4 (30) | wide_qkv (2.225x; 30) |
| 142 | outer_04.transformer_2.block_14.attention_fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 60 | 12.049 | 15.296 | 1.270x | 47.1% | 0.200 | library_gemm (30) | qk_rope (1.293x; 30) |
| 143 | outer_04.transformer_2.block_14.attention_out_projection_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 60 | 8.704 | 18.081 | 2.077x | 95.9% | 0.564 | library_gemm (33) | fused_ffn (2.423x; 27) |
| 144 | outer_04.transformer_2.block_14.ffn_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Half2Kernel; g1174x1x1; b128x1x1; r38; s0 | 60 | 2.688 | 4.656 | 1.732x | 100.0% | 0.124 | fused_ffn (30) | fused_ffn (2.292x; 30) |
| 145 | outer_04.transformer_2.block_14.ffn_linear1_gate_swiglu | fused_ffn | fused_ffn; fused_ffn_kernel; g18x37x1; b128x1x1; r146; s32768 | 60 | 37.552 | 42.800 | 1.140x | 81.3% | 0.318 | fused_ffn (30) | fused_ffn (1.162x; 30) |
| 146 | outer_04.transformer_2.block_14.ffn_linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 60 | 21.152 | 36.512 | 1.726x | 98.3% | 0.956 | fused_ffn (30) | library_gemm (1.830x; 30) |
| 147 | outer_04.post_norm_silu | affine_silu | affine_silu; affineSiluHalf2Kernel; g4693x1x1; b192x1x1; r17; s0 | 60 | 3.424 | 4.448 | 1.299x | 100.0% | 0.062 | library_gemm (30) | linear2_residual (1.336x; 30) |
| 148 | outer_04.post_projection_c384_to_c768_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b256x1x1; r154; s73728 | 60 | 14.736 | 24.608 | 1.670x | 96.0% | 0.616 | affine_silu (30) | linear2_residual (1.862x; 30) |
| 149 | outer_05.pre_norm_silu | affine_silu | affine_silu; affineSiluHalf2Kernel; g4693x1x1; b384x1x1; r17; s0 | 60 | 5.184 | 8.592 | 1.657x | 100.0% | 0.212 | library_gemm (60) | library_gemm (1.657x; 60) |
| 150 | outer_05.pre_projection_c768_to_c384 | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 60 | 12.161 | 18.209 | 1.497x | 94.6% | 0.387 | wide_qkv (30) | library_gemm (1.722x; 26) |
| 151 | outer_05.transformer_0.block_15.attention_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Half2Kernel; g1174x1x1; b128x1x1; r38; s0 | 60 | 2.720 | 5.040 | 1.853x | 100.0% | 0.148 | library_gemm (30) | wide_qkv (2.247x; 30) |
| 152 | outer_05.transformer_0.block_15.attention_qkv_projection | wide_qkv | wide_qkv; wide_qkv_kernel; g9x37x1; b128x1x1; r136; s65536 | 60 | 19.569 | 25.712 | 1.314x | 64.9% | 0.358 | qk_rope (30) | library_gemm (1.401x; 29) |
| 153 | outer_05.transformer_0.block_15.attention_qk_rope | qk_rope | qk_rope; fusedQKRoPE19HalfKernel; g361x13x1; b192x1x1; r16; s0 | 60 | 4.192 | 7.744 | 1.847x | 100.0% | 0.220 | fa4 (30) | wide_qkv (2.328x; 30) |
| 154 | outer_05.transformer_0.block_15.attention_fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 60 | 12.128 | 16.128 | 1.330x | 47.7% | 0.242 | library_gemm (30) | qk_rope (1.354x; 30) |
| 155 | outer_05.transformer_0.block_15.attention_out_projection_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 60 | 8.640 | 18.320 | 2.120x | 95.9% | 0.596 | library_gemm (42) | fused_ffn (2.491x; 18) |
| 156 | outer_05.transformer_0.block_15.ffn_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Half2Kernel; g1174x1x1; b128x1x1; r38; s0 | 60 | 2.736 | 4.864 | 1.778x | 100.0% | 0.131 | fused_ffn (30) | fused_ffn (2.292x; 30) |
| 157 | outer_05.transformer_0.block_15.ffn_linear1_gate_swiglu | fused_ffn | fused_ffn; fused_ffn_kernel; g18x37x1; b128x1x1; r146; s32768 | 60 | 43.153 | 48.688 | 1.128x | 76.0% | 0.347 | fused_ffn (30) | fused_ffn (1.195x; 30) |
| 158 | outer_05.transformer_0.block_15.ffn_linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 60 | 21.216 | 34.257 | 1.615x | 97.9% | 0.801 | fused_ffn (30) | wide_qkv (1.705x; 30) |
| 159 | outer_05.transformer_1.block_16.attention_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Half2Kernel; g1174x1x1; b128x1x1; r38; s0 | 60 | 2.720 | 4.705 | 1.730x | 100.0% | 0.126 | linear2_residual (30) | wide_qkv (2.106x; 30) |
| 160 | outer_05.transformer_1.block_16.attention_qkv_projection | wide_qkv | wide_qkv; wide_qkv_kernel; g9x37x1; b128x1x1; r136; s65536 | 60 | 19.648 | 26.321 | 1.340x | 82.9% | 0.429 | linear2_residual (30) | linear2_residual (1.593x; 30) |
| 161 | outer_05.transformer_1.block_16.attention_qk_rope | qk_rope | qk_rope; fusedQKRoPE19HalfKernel; g361x13x1; b192x1x1; r16; s0 | 60 | 4.192 | 7.633 | 1.821x | 100.0% | 0.211 | fa4 (30) | wide_qkv (2.260x; 30) |
| 162 | outer_05.transformer_1.block_16.attention_fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 60 | 12.065 | 15.136 | 1.255x | 44.2% | 0.185 | library_gemm (30) | library_gemm (1.259x; 30) |
| 163 | outer_05.transformer_1.block_16.attention_out_projection_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 60 | 8.577 | 18.096 | 2.110x | 95.9% | 0.566 | library_gemm (37) | fused_ffn (2.455x; 23) |
| 164 | outer_05.transformer_1.block_16.ffn_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Half2Kernel; g1174x1x1; b128x1x1; r38; s0 | 60 | 2.688 | 4.944 | 1.839x | 100.0% | 0.138 | fused_ffn (30) | fused_ffn (2.458x; 30) |
| 165 | outer_05.transformer_1.block_16.ffn_linear1_gate_swiglu | fused_ffn | fused_ffn; fused_ffn_kernel; g18x37x1; b128x1x1; r146; s32768 | 60 | 43.425 | 45.808 | 1.055x | 73.1% | 0.167 | fused_ffn (30) | fused_ffn (1.120x; 30) |
| 166 | outer_05.transformer_1.block_16.ffn_linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 60 | 21.025 | 33.792 | 1.607x | 98.3% | 0.788 | fused_ffn (30) | wide_qkv (1.743x; 30) |
| 167 | outer_05.transformer_2.block_17.attention_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Half2Kernel; g1174x1x1; b128x1x1; r38; s0 | 60 | 2.688 | 4.416 | 1.643x | 100.0% | 0.118 | linear2_residual (30) | wide_qkv (1.994x; 30) |
| 168 | outer_05.transformer_2.block_17.attention_qkv_projection | wide_qkv | wide_qkv; wide_qkv_kernel; g9x37x1; b128x1x1; r136; s65536 | 60 | 19.440 | 25.120 | 1.292x | 81.8% | 0.358 | linear2_residual (30) | linear2_residual (1.502x; 30) |
| 169 | outer_05.transformer_2.block_17.attention_qk_rope | qk_rope | qk_rope; fusedQKRoPE19HalfKernel; g361x13x1; b192x1x1; r16; s0 | 60 | 4.160 | 7.536 | 1.812x | 99.1% | 0.202 | fa4 (30) | wide_qkv (2.246x; 30) |
| 170 | outer_05.transformer_2.block_17.attention_fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 60 | 12.000 | 14.912 | 1.243x | 47.6% | 0.180 | library_gemm (29) | qk_rope (1.248x; 29) |
| 171 | outer_05.transformer_2.block_17.attention_out_projection_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 60 | 8.640 | 17.152 | 1.985x | 95.9% | 0.555 | fused_ffn (30) | fused_ffn (2.443x; 30) |
| 172 | outer_05.transformer_2.block_17.ffn_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Half2Kernel; g1174x1x1; b128x1x1; r38; s0 | 60 | 2.688 | 4.880 | 1.815x | 100.0% | 0.138 | fused_ffn (30) | fused_ffn (2.429x; 30) |
| 173 | outer_05.transformer_2.block_17.ffn_linear1_gate_swiglu | fused_ffn | fused_ffn; fused_ffn_kernel; g18x37x1; b128x1x1; r146; s32768 | 60 | 43.361 | 45.888 | 1.058x | 74.1% | 0.165 | fused_ffn (30) | fused_ffn (1.121x; 30) |
| 174 | outer_05.transformer_2.block_17.ffn_linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 60 | 20.992 | 31.904 | 1.520x | 98.1% | 0.688 | fused_ffn (30) | library_gemm (1.617x; 30) |
| 175 | outer_05.post_norm_silu | affine_silu | affine_silu; affineSiluHalf2Kernel; g4693x1x1; b192x1x1; r17; s0 | 60 | 3.328 | 4.096 | 1.231x | 100.0% | 0.046 | library_gemm (30) | linear2_residual (1.260x; 30) |
| 176 | outer_05.post_projection_c384_to_c768_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b256x1x1; r154; s73728 | 60 | 14.496 | 22.817 | 1.574x | 95.8% | 0.506 | affine_silu (30) | linear2_residual (1.687x; 30) |
| 177 | outer_06.pre_norm_silu | affine_silu | affine_silu; affineSiluHalf2Kernel; g4693x1x1; b384x1x1; r17; s0 | 60 | 5.088 | 8.161 | 1.604x | 100.0% | 0.185 | library_gemm (60) | library_gemm (1.604x; 60) |
| 178 | outer_06.pre_projection_c768_to_c384 | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 60 | 12.032 | 17.136 | 1.424x | 94.3% | 0.323 | wide_qkv (30) | library_gemm (1.590x; 29) |
| 179 | outer_06.transformer_0.block_18.attention_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Half2Kernel; g1174x1x1; b128x1x1; r38; s0 | 60 | 2.688 | 4.832 | 1.798x | 100.0% | 0.138 | library_gemm (30) | wide_qkv (2.095x; 30) |
| 180 | outer_06.transformer_0.block_18.attention_qkv_projection | wide_qkv | wide_qkv; wide_qkv_kernel; g9x37x1; b128x1x1; r136; s65536 | 60 | 19.297 | 25.296 | 1.311x | 65.6% | 0.350 | library_gemm (30) | library_gemm (1.399x; 30) |
| 181 | outer_06.transformer_0.block_18.attention_qk_rope | qk_rope | qk_rope; fusedQKRoPE19HalfKernel; g361x13x1; b192x1x1; r16; s0 | 60 | 4.144 | 7.712 | 1.861x | 100.0% | 0.214 | fa4 (30) | wide_qkv (2.293x; 30) |
| 182 | outer_06.transformer_0.block_18.attention_fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 60 | 12.032 | 15.968 | 1.327x | 47.7% | 0.236 | library_gemm (30) | qk_rope (1.348x; 30) |
| 183 | outer_06.transformer_0.block_18.attention_out_projection_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 60 | 8.544 | 18.240 | 2.135x | 96.0% | 0.593 | library_gemm (43) | fused_ffn (2.513x; 17) |
| 184 | outer_06.transformer_0.block_18.ffn_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Half2Kernel; g1174x1x1; b128x1x1; r38; s0 | 60 | 2.720 | 4.736 | 1.741x | 100.0% | 0.129 | fused_ffn (30) | fused_ffn (2.294x; 30) |
| 185 | outer_06.transformer_0.block_18.ffn_linear1_gate_swiglu | fused_ffn | fused_ffn; fused_ffn_kernel; g18x37x1; b128x1x1; r146; s32768 | 60 | 42.305 | 48.160 | 1.138x | 76.9% | 0.355 | fused_ffn (30) | fused_ffn (1.175x; 30) |
| 186 | outer_06.transformer_0.block_18.ffn_linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 60 | 21.200 | 33.601 | 1.585x | 97.8% | 0.748 | fused_ffn (30) | wide_qkv (1.606x; 30) |
| 187 | outer_06.transformer_1.block_19.attention_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Half2Kernel; g1174x1x1; b128x1x1; r38; s0 | 60 | 2.688 | 4.864 | 1.809x | 100.0% | 0.137 | linear2_residual (30) | wide_qkv (2.232x; 30) |
| 188 | outer_06.transformer_1.block_19.attention_qkv_projection | wide_qkv | wide_qkv; wide_qkv_kernel; g9x37x1; b128x1x1; r136; s65536 | 60 | 19.600 | 26.465 | 1.350x | 81.5% | 0.453 | linear2_residual (30) | linear2_residual (1.637x; 30) |
| 189 | outer_06.transformer_1.block_19.attention_qk_rope | qk_rope | qk_rope; fusedQKRoPE19HalfKernel; g361x13x1; b192x1x1; r16; s0 | 60 | 4.160 | 7.520 | 1.808x | 100.0% | 0.208 | fa4 (30) | wide_qkv (2.269x; 30) |
| 190 | outer_06.transformer_1.block_19.attention_fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 60 | 12.080 | 15.104 | 1.250x | 46.0% | 0.184 | library_gemm (30) | qk_rope (1.258x; 30) |
| 191 | outer_06.transformer_1.block_19.attention_out_projection_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 60 | 8.720 | 17.776 | 2.039x | 95.8% | 0.553 | library_gemm (33) | fused_ffn (2.422x; 27) |
| 192 | outer_06.transformer_1.block_19.ffn_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Half2Kernel; g1174x1x1; b128x1x1; r38; s0 | 60 | 2.720 | 4.545 | 1.671x | 100.0% | 0.118 | fused_ffn (30) | fused_ffn (2.200x; 30) |
| 193 | outer_06.transformer_1.block_19.ffn_linear1_gate_swiglu | fused_ffn | fused_ffn; fused_ffn_kernel; g18x37x1; b128x1x1; r146; s32768 | 60 | 41.984 | 47.089 | 1.122x | 80.3% | 0.321 | fused_ffn (30) | fused_ffn (1.168x; 30) |
| 194 | outer_06.transformer_1.block_19.ffn_linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 60 | 21.169 | 34.689 | 1.639x | 98.3% | 0.827 | fused_ffn (30) | wide_qkv (1.686x; 30) |
| 195 | outer_06.transformer_2.block_20.attention_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Half2Kernel; g1174x1x1; b128x1x1; r38; s0 | 60 | 2.720 | 4.848 | 1.782x | 100.0% | 0.135 | linear2_residual (30) | wide_qkv (2.212x; 30) |
| 196 | outer_06.transformer_2.block_20.attention_qkv_projection | wide_qkv | wide_qkv; wide_qkv_kernel; g9x37x1; b128x1x1; r136; s65536 | 60 | 19.488 | 27.088 | 1.390x | 81.8% | 0.480 | linear2_residual (30) | linear2_residual (1.662x; 30) |
| 197 | outer_06.transformer_2.block_20.attention_qk_rope | qk_rope | qk_rope; fusedQKRoPE19HalfKernel; g361x13x1; b192x1x1; r16; s0 | 60 | 4.176 | 7.648 | 1.831x | 98.4% | 0.213 | fa4 (30) | wide_qkv (2.322x; 30) |
| 198 | outer_06.transformer_2.block_20.attention_fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 60 | 12.096 | 15.328 | 1.267x | 47.4% | 0.201 | library_gemm (28) | fa4 (1.409x; 4) |
| 199 | outer_06.transformer_2.block_20.attention_out_projection_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 60 | 8.704 | 15.824 | 1.818x | 95.9% | 0.542 | library_gemm (37) | fused_ffn (2.393x; 23) |
| 200 | outer_06.transformer_2.block_20.ffn_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Half2Kernel; g1174x1x1; b128x1x1; r38; s0 | 60 | 2.688 | 4.464 | 1.660x | 100.0% | 0.112 | fused_ffn (30) | fused_ffn (2.166x; 30) |
| 201 | outer_06.transformer_2.block_20.ffn_linear1_gate_swiglu | fused_ffn | fused_ffn; fused_ffn_kernel; g18x37x1; b128x1x1; r146; s32768 | 60 | 38.065 | 42.417 | 1.114x | 81.6% | 0.263 | fused_ffn (30) | fused_ffn (1.126x; 30) |
| 202 | outer_06.transformer_2.block_20.ffn_linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 60 | 21.184 | 37.024 | 1.748x | 98.3% | 0.974 | fused_ffn (30) | library_gemm (1.829x; 30) |
| 203 | outer_06.post_norm_silu | affine_silu | affine_silu; affineSiluHalf2Kernel; g4693x1x1; b192x1x1; r17; s0 | 60 | 3.392 | 4.400 | 1.297x | 100.0% | 0.062 | library_gemm (30) | linear2_residual (1.349x; 30) |
| 204 | outer_06.post_projection_c384_to_c768_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b256x1x1; r154; s73728 | 60 | 14.688 | 25.200 | 1.716x | 96.1% | 0.635 | affine_silu (30) | linear2_residual (1.853x; 30) |
| 205 | outer_07.pre_norm_silu | affine_silu | affine_silu; affineSiluHalf2Kernel; g4693x1x1; b384x1x1; r17; s0 | 60 | 5.152 | 8.656 | 1.680x | 100.0% | 0.209 | library_gemm (60) | library_gemm (1.680x; 60) |
| 206 | outer_07.pre_projection_c768_to_c384 | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 60 | 12.129 | 17.440 | 1.438x | 94.6% | 0.400 | wide_qkv (30) | library_gemm (1.762x; 28) |
| 207 | outer_07.transformer_0.block_21.attention_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Half2Kernel; g1174x1x1; b128x1x1; r38; s0 | 60 | 2.720 | 4.944 | 1.818x | 100.0% | 0.151 | library_gemm (30) | wide_qkv (2.294x; 30) |
| 208 | outer_07.transformer_0.block_21.attention_qkv_projection | wide_qkv | wide_qkv; wide_qkv_kernel; g9x37x1; b128x1x1; r136; s65536 | 60 | 19.584 | 25.808 | 1.318x | 64.9% | 0.352 | qk_rope (30) | library_gemm (1.397x; 28) |
| 209 | outer_07.transformer_0.block_21.attention_qk_rope | qk_rope | qk_rope; fusedQKRoPE19HalfKernel; g361x13x1; b192x1x1; r16; s0 | 60 | 4.128 | 7.680 | 1.860x | 100.0% | 0.222 | fa4 (30) | wide_qkv (2.318x; 30) |
| 210 | outer_07.transformer_0.block_21.attention_fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 60 | 12.113 | 16.064 | 1.326x | 47.9% | 0.237 | library_gemm (30) | qk_rope (1.335x; 30) |
| 211 | outer_07.transformer_0.block_21.attention_out_projection_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 60 | 8.672 | 18.400 | 2.122x | 95.9% | 0.592 | library_gemm (48) | fused_ffn (2.509x; 12) |
| 212 | outer_07.transformer_0.block_21.ffn_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Half2Kernel; g1174x1x1; b128x1x1; r38; s0 | 60 | 2.720 | 4.816 | 1.771x | 100.0% | 0.131 | fused_ffn (30) | fused_ffn (2.294x; 30) |
| 213 | outer_07.transformer_0.block_21.ffn_linear1_gate_swiglu | fused_ffn | fused_ffn; fused_ffn_kernel; g18x37x1; b128x1x1; r146; s32768 | 60 | 40.688 | 44.193 | 1.086x | 82.0% | 0.206 | fused_ffn (30) | fused_ffn (1.101x; 30) |
| 214 | outer_07.transformer_0.block_21.ffn_linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 60 | 21.265 | 37.361 | 1.757x | 98.0% | 0.976 | fused_ffn (30) | wide_qkv (1.832x; 30) |
| 215 | outer_07.transformer_1.block_22.attention_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Half2Kernel; g1174x1x1; b128x1x1; r38; s0 | 60 | 2.720 | 4.800 | 1.765x | 100.0% | 0.136 | linear2_residual (30) | wide_qkv (2.176x; 30) |
| 216 | outer_07.transformer_1.block_22.attention_qkv_projection | wide_qkv | wide_qkv; wide_qkv_kernel; g9x37x1; b128x1x1; r136; s65536 | 60 | 19.697 | 27.601 | 1.401x | 83.0% | 0.491 | linear2_residual (30) | linear2_residual (1.649x; 30) |
| 217 | outer_07.transformer_1.block_22.attention_qk_rope | qk_rope | qk_rope; fusedQKRoPE19HalfKernel; g361x13x1; b192x1x1; r16; s0 | 60 | 4.160 | 7.712 | 1.854x | 100.0% | 0.215 | fa4 (30) | wide_qkv (2.292x; 30) |
| 218 | outer_07.transformer_1.block_22.attention_fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 60 | 12.128 | 15.536 | 1.281x | 47.1% | 0.207 | library_gemm (30) | qk_rope (1.303x; 30) |
| 219 | outer_07.transformer_1.block_22.attention_out_projection_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 60 | 8.608 | 18.016 | 2.093x | 95.8% | 0.569 | library_gemm (37) | fused_ffn (2.461x; 23) |
| 220 | outer_07.transformer_1.block_22.ffn_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Half2Kernel; g1174x1x1; b128x1x1; r38; s0 | 60 | 2.736 | 4.592 | 1.678x | 100.0% | 0.121 | fused_ffn (30) | fused_ffn (2.222x; 30) |
| 221 | outer_07.transformer_1.block_22.ffn_linear1_gate_swiglu | fused_ffn | fused_ffn; fused_ffn_kernel; g18x37x1; b128x1x1; r146; s32768 | 60 | 38.145 | 43.872 | 1.150x | 81.2% | 0.339 | fused_ffn (30) | fused_ffn (1.170x; 30) |
| 222 | outer_07.transformer_1.block_22.ffn_linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 60 | 21.265 | 37.456 | 1.761x | 98.4% | 0.977 | fused_ffn (30) | wide_qkv (1.836x; 30) |
| 223 | outer_07.transformer_2.block_23.attention_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Half2Kernel; g1174x1x1; b128x1x1; r38; s0 | 60 | 2.720 | 4.624 | 1.700x | 100.0% | 0.127 | linear2_residual (30) | wide_qkv (2.135x; 30) |
| 224 | outer_07.transformer_2.block_23.attention_qkv_projection | wide_qkv | wide_qkv; wide_qkv_kernel; g9x37x1; b128x1x1; r136; s65536 | 60 | 19.713 | 27.584 | 1.399x | 83.1% | 0.498 | linear2_residual (30) | linear2_residual (1.664x; 30) |
| 225 | outer_07.transformer_2.block_23.attention_qk_rope | qk_rope | qk_rope; fusedQKRoPE19HalfKernel; g361x13x1; b192x1x1; r16; s0 | 60 | 4.192 | 7.745 | 1.847x | 100.0% | 0.217 | fa4 (30) | wide_qkv (2.309x; 30) |
| 226 | outer_07.transformer_2.block_23.attention_fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 60 | 12.209 | 15.409 | 1.262x | 46.9% | 0.200 | library_gemm (30) | qk_rope (1.280x; 30) |
| 227 | outer_07.transformer_2.block_23.attention_out_projection_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 60 | 8.672 | 17.952 | 2.070x | 96.0% | 0.571 | library_gemm (40) | fused_ffn (2.441x; 20) |
| 228 | outer_07.transformer_2.block_23.ffn_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Half2Kernel; g1174x1x1; b128x1x1; r38; s0 | 60 | 2.720 | 4.704 | 1.729x | 100.0% | 0.123 | fused_ffn (30) | fused_ffn (2.229x; 30) |
| 229 | outer_07.transformer_2.block_23.ffn_linear1_gate_swiglu | fused_ffn | fused_ffn; fused_ffn_kernel; g18x37x1; b128x1x1; r146; s32768 | 60 | 37.665 | 43.568 | 1.157x | 82.9% | 0.351 | fused_ffn (30) | fused_ffn (1.165x; 30) |
| 230 | outer_07.transformer_2.block_23.ffn_linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 60 | 21.280 | 37.088 | 1.743x | 98.3% | 0.960 | fused_ffn (30) | library_gemm (1.797x; 30) |
| 231 | outer_07.post_norm_silu | affine_silu | affine_silu; affineSiluHalf2Kernel; g4693x1x1; b192x1x1; r17; s0 | 60 | 3.424 | 4.465 | 1.304x | 100.0% | 0.060 | library_gemm (30) | linear2_residual (1.332x; 30) |
| 232 | outer_07.post_projection_c384_to_c768_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b256x1x1; r154; s73728 | 60 | 14.720 | 24.241 | 1.647x | 96.0% | 0.619 | affine_silu (30) | linear2_residual (1.886x; 30) |
| 233 | outer_08.pre_norm_silu | affine_silu | affine_silu; affineSiluHalf2Kernel; g4693x1x1; b384x1x1; r17; s0 | 60 | 5.184 | 8.528 | 1.645x | 100.0% | 0.208 | library_gemm (60) | library_gemm (1.645x; 60) |
| 234 | outer_08.pre_projection_c768_to_c384 | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 60 | 12.209 | 18.160 | 1.487x | 94.7% | 0.385 | wide_qkv (30) | library_gemm (1.701x; 26) |
| 235 | outer_08.transformer_0.block_24.attention_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Half2Kernel; g1174x1x1; b128x1x1; r38; s0 | 60 | 2.720 | 4.944 | 1.818x | 100.0% | 0.148 | library_gemm (30) | wide_qkv (2.224x; 30) |
| 236 | outer_08.transformer_0.block_24.attention_qkv_projection | wide_qkv | wide_qkv; wide_qkv_kernel; g9x37x1; b128x1x1; r136; s65536 | 60 | 19.648 | 26.368 | 1.342x | 65.7% | 0.384 | library_gemm (30) | library_gemm (1.415x; 30) |
| 237 | outer_08.transformer_0.block_24.attention_qk_rope | qk_rope | qk_rope; fusedQKRoPE19HalfKernel; g361x13x1; b192x1x1; r16; s0 | 60 | 4.224 | 8.064 | 1.909x | 100.0% | 0.223 | fa4 (30) | wide_qkv (2.326x; 30) |
| 238 | outer_08.transformer_0.block_24.attention_fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 60 | 12.176 | 16.208 | 1.331x | 50.5% | 0.268 | library_gemm (30) | qk_rope (1.382x; 30) |
| 239 | outer_08.transformer_0.block_24.attention_out_projection_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 60 | 8.800 | 18.432 | 2.095x | 96.1% | 0.594 | library_gemm (42) | fused_ffn (2.475x; 18) |
| 240 | outer_08.transformer_0.block_24.ffn_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Half2Kernel; g1174x1x1; b128x1x1; r38; s0 | 60 | 2.752 | 4.992 | 1.814x | 100.0% | 0.139 | fused_ffn (30) | fused_ffn (2.395x; 30) |
| 241 | outer_08.transformer_0.block_24.ffn_linear1_gate_swiglu | fused_ffn | fused_ffn; fused_ffn_kernel; g18x37x1; b128x1x1; r146; s32768 | 60 | 42.833 | 48.385 | 1.130x | 78.5% | 0.355 | fused_ffn (30) | fused_ffn (1.189x; 30) |
| 242 | outer_08.transformer_0.block_24.ffn_linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 60 | 21.265 | 34.961 | 1.644x | 97.9% | 0.826 | fused_ffn (30) | wide_qkv (1.686x; 30) |
| 243 | outer_08.transformer_1.block_25.attention_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Half2Kernel; g1174x1x1; b128x1x1; r38; s0 | 60 | 2.720 | 4.688 | 1.724x | 100.0% | 0.132 | linear2_residual (30) | wide_qkv (2.194x; 30) |
| 244 | outer_08.transformer_1.block_25.attention_qkv_projection | wide_qkv | wide_qkv; wide_qkv_kernel; g9x37x1; b128x1x1; r136; s65536 | 60 | 19.648 | 27.505 | 1.400x | 83.1% | 0.479 | linear2_residual (30) | linear2_residual (1.649x; 30) |
| 245 | outer_08.transformer_1.block_25.attention_qk_rope | qk_rope | qk_rope; fusedQKRoPE19HalfKernel; g361x13x1; b192x1x1; r16; s0 | 60 | 4.192 | 7.616 | 1.817x | 99.2% | 0.211 | fa4 (30) | wide_qkv (2.271x; 30) |
| 246 | outer_08.transformer_1.block_25.attention_fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 60 | 12.160 | 15.296 | 1.258x | 46.4% | 0.195 | library_gemm (29) | qk_rope (1.276x; 29) |
| 247 | outer_08.transformer_1.block_25.attention_out_projection_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 60 | 8.672 | 15.697 | 1.810x | 95.9% | 0.562 | library_gemm (40) | fused_ffn (2.456x; 20) |
| 248 | outer_08.transformer_1.block_25.ffn_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Half2Kernel; g1174x1x1; b128x1x1; r38; s0 | 60 | 2.720 | 4.784 | 1.759x | 100.0% | 0.127 | fused_ffn (30) | fused_ffn (2.318x; 30) |
| 249 | outer_08.transformer_1.block_25.ffn_linear1_gate_swiglu | fused_ffn | fused_ffn; fused_ffn_kernel; g18x37x1; b128x1x1; r146; s32768 | 60 | 43.169 | 48.257 | 1.118x | 77.1% | 0.306 | fused_ffn (30) | fused_ffn (1.167x; 30) |
| 250 | outer_08.transformer_1.block_25.ffn_linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 60 | 21.216 | 34.240 | 1.614x | 98.3% | 0.811 | fused_ffn (30) | wide_qkv (1.717x; 30) |
| 251 | outer_08.transformer_2.block_26.attention_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Half2Kernel; g1174x1x1; b128x1x1; r38; s0 | 60 | 2.720 | 4.480 | 1.647x | 100.0% | 0.120 | linear2_residual (30) | wide_qkv (2.059x; 30) |
| 252 | outer_08.transformer_2.block_26.attention_qkv_projection | wide_qkv | wide_qkv; wide_qkv_kernel; g9x37x1; b128x1x1; r136; s65536 | 60 | 19.616 | 26.784 | 1.365x | 82.6% | 0.443 | linear2_residual (30) | linear2_residual (1.613x; 30) |
| 253 | outer_08.transformer_2.block_26.attention_qk_rope | qk_rope | qk_rope; fusedQKRoPE19HalfKernel; g361x13x1; b192x1x1; r16; s0 | 60 | 4.160 | 7.632 | 1.835x | 100.0% | 0.203 | fa4 (30) | wide_qkv (2.250x; 30) |
| 254 | outer_08.transformer_2.block_26.attention_fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 60 | 12.128 | 15.232 | 1.256x | 47.1% | 0.191 | library_gemm (30) | qk_rope (1.278x; 30) |
| 255 | outer_08.transformer_2.block_26.attention_out_projection_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 60 | 8.688 | 17.664 | 2.033x | 95.9% | 0.552 | library_gemm (34) | fused_ffn (2.411x; 26) |
| 256 | outer_08.transformer_2.block_26.ffn_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Half2Kernel; g1174x1x1; b128x1x1; r38; s0 | 60 | 2.752 | 4.705 | 1.709x | 100.0% | 0.122 | fused_ffn (30) | fused_ffn (2.256x; 30) |
| 257 | outer_08.transformer_2.block_26.ffn_linear1_gate_swiglu | fused_ffn | fused_ffn; fused_ffn_kernel; g18x37x1; b128x1x1; r146; s32768 | 60 | 41.969 | 47.537 | 1.133x | 77.8% | 0.321 | fused_ffn (30) | fused_ffn (1.162x; 30) |
| 258 | outer_08.transformer_2.block_26.ffn_linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 60 | 21.200 | 34.001 | 1.604x | 98.2% | 0.773 | fused_ffn (30) | library_gemm (1.643x; 30) |
| 259 | outer_08.post_norm_silu | affine_silu | affine_silu; affineSiluHalf2Kernel; g4693x1x1; b192x1x1; r17; s0 | 60 | 3.392 | 4.256 | 1.255x | 100.0% | 0.050 | library_gemm (30) | linear2_residual (1.311x; 30) |
| 260 | outer_08.post_projection_c384_to_c768_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b256x1x1; r154; s73728 | 60 | 14.624 | 24.304 | 1.662x | 96.0% | 0.618 | affine_silu (30) | linear2_residual (1.896x; 30) |
| 261 | outer_09.pre_norm_silu | affine_silu | affine_silu; affineSiluHalf2Kernel; g4693x1x1; b384x1x1; r17; s0 | 60 | 5.152 | 8.528 | 1.655x | 100.0% | 0.203 | library_gemm (60) | library_gemm (1.655x; 60) |
| 262 | outer_09.pre_projection_c768_to_c384 | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 60 | 12.160 | 17.248 | 1.418x | 94.4% | 0.335 | wide_qkv (30) | library_gemm (1.645x; 5) |
| 263 | outer_09.transformer_0.block_27.attention_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Half2Kernel; g1174x1x1; b128x1x1; r38; s0 | 60 | 2.688 | 4.896 | 1.822x | 100.0% | 0.141 | library_gemm (30) | wide_qkv (2.107x; 30) |
| 264 | outer_09.transformer_0.block_27.attention_qkv_projection | wide_qkv | wide_qkv; wide_qkv_kernel; g9x37x1; b128x1x1; r136; s65536 | 60 | 19.489 | 25.760 | 1.322x | 65.8% | 0.374 | library_gemm (30) | library_gemm (1.421x; 30) |
| 265 | outer_09.transformer_0.block_27.attention_qk_rope | qk_rope | qk_rope; fusedQKRoPE19HalfKernel; g361x13x1; b192x1x1; r16; s0 | 60 | 4.176 | 7.904 | 1.892x | 100.0% | 0.221 | fa4 (30) | wide_qkv (2.322x; 30) |
| 266 | outer_09.transformer_0.block_27.attention_fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 60 | 12.112 | 16.192 | 1.337x | 47.7% | 0.244 | library_gemm (30) | qk_rope (1.351x; 30) |
| 267 | outer_09.transformer_0.block_27.attention_out_projection_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 60 | 8.688 | 18.352 | 2.112x | 95.9% | 0.589 | library_gemm (44) | fused_ffn (2.460x; 16) |
| 268 | outer_09.transformer_0.block_27.ffn_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Half2Kernel; g1174x1x1; b128x1x1; r38; s0 | 60 | 2.720 | 4.976 | 1.829x | 100.0% | 0.144 | fused_ffn (30) | fused_ffn (2.476x; 30) |
| 269 | outer_09.transformer_0.block_27.ffn_linear1_gate_swiglu | fused_ffn | fused_ffn; fused_ffn_kernel; g18x37x1; b128x1x1; r146; s32768 | 60 | 43.488 | 45.825 | 1.054x | 71.2% | 0.179 | fused_ffn (34) | fused_ffn (1.123x; 34) |
| 270 | outer_09.transformer_0.block_27.ffn_linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 60 | 21.056 | 32.864 | 1.561x | 97.9% | 0.727 | fused_ffn (30) | wide_qkv (1.692x; 30) |
| 271 | outer_09.transformer_1.block_28.attention_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Half2Kernel; g1174x1x1; b128x1x1; r38; s0 | 60 | 2.688 | 4.576 | 1.702x | 100.0% | 0.117 | linear2_residual (30) | wide_qkv (2.012x; 30) |
| 272 | outer_09.transformer_1.block_28.attention_qkv_projection | wide_qkv | wide_qkv; wide_qkv_kernel; g9x37x1; b128x1x1; r136; s65536 | 60 | 19.441 | 24.448 | 1.258x | 81.8% | 0.323 | linear2_residual (30) | linear2_residual (1.451x; 30) |
| 273 | outer_09.transformer_1.block_28.attention_qk_rope | qk_rope | qk_rope; fusedQKRoPE19HalfKernel; g361x13x1; b192x1x1; r16; s0 | 60 | 4.160 | 7.360 | 1.769x | 100.0% | 0.201 | fa4 (30) | wide_qkv (2.250x; 30) |
| 274 | outer_09.transformer_1.block_28.attention_fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 60 | 12.000 | 14.864 | 1.239x | 47.4% | 0.176 | library_gemm (30) | qk_rope (1.259x; 30) |
| 275 | outer_09.transformer_1.block_28.attention_out_projection_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 60 | 8.624 | 17.505 | 2.030x | 95.8% | 0.553 | library_gemm (35) | fused_ffn (2.438x; 25) |
| 276 | outer_09.transformer_1.block_28.ffn_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Half2Kernel; g1174x1x1; b128x1x1; r38; s0 | 60 | 2.688 | 4.912 | 1.827x | 100.0% | 0.137 | fused_ffn (30) | fused_ffn (2.452x; 30) |
| 277 | outer_09.transformer_1.block_28.ffn_linear1_gate_swiglu | fused_ffn | fused_ffn; fused_ffn_kernel; g18x37x1; b128x1x1; r146; s32768 | 60 | 42.897 | 44.256 | 1.032x | 72.4% | 0.124 | fused_ffn (30) | fused_ffn (1.091x; 30) |
| 278 | outer_09.transformer_1.block_28.ffn_linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 60 | 20.848 | 32.160 | 1.543x | 98.2% | 0.699 | fused_ffn (30) | wide_qkv (1.665x; 30) |
| 279 | outer_09.transformer_2.block_29.attention_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Half2Kernel; g1174x1x1; b128x1x1; r38; s0 | 60 | 2.656 | 4.432 | 1.669x | 100.0% | 0.123 | linear2_residual (30) | wide_qkv (2.072x; 30) |
| 280 | outer_09.transformer_2.block_29.attention_qkv_projection | wide_qkv | wide_qkv; wide_qkv_kernel; g9x37x1; b128x1x1; r136; s65536 | 60 | 19.328 | 24.689 | 1.277x | 81.5% | 0.324 | linear2_residual (30) | linear2_residual (1.445x; 30) |
| 281 | outer_09.transformer_2.block_29.attention_qk_rope | qk_rope | qk_rope; fusedQKRoPE19HalfKernel; g361x13x1; b192x1x1; r16; s0 | 60 | 4.128 | 7.361 | 1.783x | 99.1% | 0.197 | fa4 (30) | wide_qkv (2.209x; 30) |
| 282 | outer_09.transformer_2.block_29.attention_fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 60 | 11.921 | 14.640 | 1.228x | 48.6% | 0.171 | library_gemm (29) | qk_rope (1.229x; 29) |
| 283 | outer_09.transformer_2.block_29.attention_out_projection_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 60 | 8.576 | 15.168 | 1.769x | 95.8% | 0.540 | library_gemm (32) | fused_ffn (2.424x; 28) |
| 284 | outer_09.transformer_2.block_29.ffn_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Half2Kernel; g1174x1x1; b128x1x1; r38; s0 | 60 | 2.704 | 4.864 | 1.799x | 100.0% | 0.132 | fused_ffn (30) | fused_ffn (2.373x; 30) |
| 285 | outer_09.transformer_2.block_29.ffn_linear1_gate_swiglu | fused_ffn | fused_ffn; fused_ffn_kernel; g18x37x1; b128x1x1; r146; s32768 | 60 | 42.160 | 44.465 | 1.055x | 74.1% | 0.167 | fused_ffn (31) | fused_ffn (1.135x; 31) |
| 286 | outer_09.transformer_2.block_29.ffn_linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 60 | 20.736 | 30.800 | 1.485x | 98.1% | 0.631 | fused_ffn (30) | library_gemm (1.576x; 30) |
| 287 | outer_09.post_norm_silu | affine_silu | affine_silu; affineSiluHalf2Kernel; g4693x1x1; b192x1x1; r17; s0 | 60 | 3.360 | 3.985 | 1.186x | 100.0% | 0.040 | library_gemm (30) | linear2_residual (1.219x; 30) |
| 288 | outer_09.post_projection_c384_to_c768_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b256x1x1; r154; s73728 | 60 | 14.400 | 22.000 | 1.528x | 95.7% | 0.470 | affine_silu (30) | linear2_residual (1.636x; 30) |
| 289 | outer_10.pre_norm_silu | affine_silu | affine_silu; affineSiluHalf2Kernel; g4693x1x1; b384x1x1; r17; s0 | 60 | 5.072 | 8.000 | 1.577x | 100.0% | 0.176 | library_gemm (60) | library_gemm (1.577x; 60) |
| 290 | outer_10.pre_projection_c768_to_c384 | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 60 | 11.921 | 16.976 | 1.424x | 94.3% | 0.308 | library_gemm (30) | library_gemm (1.561x; 30) |
| 291 | outer_10.transformer_0.block_30.attention_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Half2Kernel; g1174x1x1; b128x1x1; r38; s0 | 60 | 2.688 | 4.928 | 1.833x | 100.0% | 0.142 | library_gemm (30) | wide_qkv (2.161x; 30) |
| 292 | outer_10.transformer_0.block_30.attention_qkv_projection | wide_qkv | wide_qkv; wide_qkv_kernel; g9x37x1; b128x1x1; r136; s65536 | 60 | 19.232 | 24.113 | 1.254x | 65.4% | 0.309 | library_gemm (30) | library_gemm (1.354x; 30) |
| 293 | outer_10.transformer_0.block_30.attention_qk_rope | qk_rope | qk_rope; fusedQKRoPE19HalfKernel; g361x13x1; b192x1x1; r16; s0 | 60 | 4.096 | 7.664 | 1.871x | 100.0% | 0.213 | fa4 (30) | wide_qkv (2.320x; 30) |
| 294 | outer_10.transformer_0.block_30.attention_fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 60 | 11.968 | 15.776 | 1.318x | 47.3% | 0.227 | library_gemm (30) | qk_rope (1.338x; 30) |
| 295 | outer_10.transformer_0.block_30.attention_out_projection_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 60 | 8.544 | 18.288 | 2.140x | 96.1% | 0.593 | library_gemm (39) | fused_ffn (2.513x; 21) |
| 296 | outer_10.transformer_0.block_30.ffn_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Half2Kernel; g1174x1x1; b128x1x1; r38; s0 | 60 | 2.704 | 4.784 | 1.769x | 100.0% | 0.126 | fused_ffn (30) | fused_ffn (2.290x; 30) |
| 297 | outer_10.transformer_0.block_30.ffn_linear1_gate_swiglu | fused_ffn | fused_ffn; fused_ffn_kernel; g18x37x1; b128x1x1; r146; s32768 | 60 | 40.865 | 43.856 | 1.073x | 82.3% | 0.188 | fused_ffn (30) | fused_ffn (1.095x; 30) |
| 298 | outer_10.transformer_0.block_30.ffn_linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 60 | 21.152 | 36.304 | 1.716x | 98.0% | 0.911 | fused_ffn (30) | wide_qkv (1.756x; 30) |
| 299 | outer_10.transformer_1.block_31.attention_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Half2Kernel; g1174x1x1; b128x1x1; r38; s0 | 60 | 2.688 | 4.736 | 1.762x | 100.0% | 0.133 | linear2_residual (30) | wide_qkv (2.167x; 30) |
| 300 | outer_10.transformer_1.block_31.attention_qkv_projection | wide_qkv | wide_qkv; wide_qkv_kernel; g9x37x1; b128x1x1; r136; s65536 | 60 | 19.552 | 27.456 | 1.404x | 83.1% | 0.475 | linear2_residual (30) | linear2_residual (1.639x; 30) |
| 301 | outer_10.transformer_1.block_31.attention_qk_rope | qk_rope | qk_rope; fusedQKRoPE19HalfKernel; g361x13x1; b192x1x1; r16; s0 | 60 | 4.096 | 7.584 | 1.852x | 100.0% | 0.212 | fa4 (30) | wide_qkv (2.277x; 30) |
| 302 | outer_10.transformer_1.block_31.attention_fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 60 | 12.064 | 15.312 | 1.269x | 47.3% | 0.199 | library_gemm (30) | qk_rope (1.282x; 30) |
| 303 | outer_10.transformer_1.block_31.attention_out_projection_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 60 | 8.576 | 17.840 | 2.080x | 95.9% | 0.564 | library_gemm (36) | fused_ffn (2.446x; 24) |
| 304 | outer_10.transformer_1.block_31.ffn_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Half2Kernel; g1174x1x1; b128x1x1; r38; s0 | 60 | 2.688 | 4.736 | 1.762x | 100.0% | 0.127 | fused_ffn (30) | fused_ffn (2.286x; 30) |
| 305 | outer_10.transformer_1.block_31.ffn_linear1_gate_swiglu | fused_ffn | fused_ffn; fused_ffn_kernel; g18x37x1; b128x1x1; r146; s32768 | 60 | 39.600 | 45.505 | 1.149x | 78.9% | 0.341 | fused_ffn (30) | fused_ffn (1.173x; 30) |
| 306 | outer_10.transformer_1.block_31.ffn_linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 60 | 21.152 | 35.824 | 1.694x | 98.3% | 0.871 | fused_ffn (30) | wide_qkv (1.769x; 30) |
| 307 | outer_10.transformer_2.block_32.attention_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Half2Kernel; g1174x1x1; b128x1x1; r38; s0 | 60 | 2.688 | 4.688 | 1.744x | 100.0% | 0.133 | linear2_residual (30) | wide_qkv (2.179x; 30) |
| 308 | outer_10.transformer_2.block_32.attention_qkv_projection | wide_qkv | wide_qkv; wide_qkv_kernel; g9x37x1; b128x1x1; r136; s65536 | 60 | 19.616 | 27.376 | 1.396x | 83.0% | 0.481 | linear2_residual (30) | linear2_residual (1.661x; 30) |
| 309 | outer_10.transformer_2.block_32.attention_qk_rope | qk_rope | qk_rope; fusedQKRoPE19HalfKernel; g361x13x1; b192x1x1; r16; s0 | 60 | 4.192 | 7.488 | 1.786x | 100.0% | 0.203 | fa4 (30) | wide_qkv (2.225x; 30) |
| 310 | outer_10.transformer_2.block_32.attention_fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 60 | 12.096 | 15.408 | 1.274x | 47.9% | 0.203 | library_gemm (30) | qk_rope (1.310x; 30) |
| 311 | outer_10.transformer_2.block_32.attention_out_projection_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 60 | 8.704 | 17.840 | 2.050x | 95.9% | 0.557 | library_gemm (37) | fused_ffn (2.434x; 23) |
| 312 | outer_10.transformer_2.block_32.ffn_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Half2Kernel; g1174x1x1; b128x1x1; r38; s0 | 60 | 2.752 | 4.784 | 1.738x | 100.0% | 0.121 | fused_ffn (30) | fused_ffn (2.209x; 30) |
| 313 | outer_10.transformer_2.block_32.ffn_linear1_gate_swiglu | fused_ffn | fused_ffn; fused_ffn_kernel; g18x37x1; b128x1x1; r146; s32768 | 60 | 42.385 | 48.336 | 1.140x | 78.3% | 0.344 | fused_ffn (30) | fused_ffn (1.176x; 30) |
| 314 | outer_10.transformer_2.block_32.ffn_linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 60 | 21.120 | 34.320 | 1.625x | 98.2% | 0.793 | fused_ffn (30) | library_gemm (1.644x; 30) |
| 315 | outer_10.post_norm_silu | affine_silu | affine_silu; affineSiluHalf2Kernel; g4693x1x1; b192x1x1; r17; s0 | 60 | 3.424 | 4.272 | 1.248x | 100.0% | 0.048 | library_gemm (30) | linear2_residual (1.280x; 30) |
| 316 | outer_10.post_projection_c384_to_c768_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b256x1x1; r154; s73728 | 60 | 14.624 | 25.504 | 1.744x | 96.2% | 0.686 | affine_silu (30) | linear2_residual (1.945x; 30) |
| 317 | trunk.tip_norm_silu | affine_silu | affine_silu; affineSiluHalf2Kernel; g4693x1x1; b384x1x1; r17; s0 | 60 | 5.120 | 8.272 | 1.616x | 85.7% | 0.183 | library_gemm (60) | library_gemm (1.616x; 60) |
| 318 | policy.p1_conv | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r80; s98304 | 60 | 6.240 | 11.344 | 1.818x | 93.0% | 0.308 | library_gemm (60) | library_gemm (1.818x; 60) |
| 319 | policy.g1_conv | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r80; s98304 | 60 | 5.952 | 8.096 | 1.360x | 91.3% | 0.121 | head_elementwise (30) | library_gemm (1.487x; 30) |
| 320 | policy.g1_norm_silu | head_elementwise | head_elementwise; applyCScaleBiasNHWCSiluHalfKernel; g1x73x13; b96x5x1; r16; s0 | 60 | 2.112 | 2.672 | 1.265x | 99.7% | 0.034 | head_elementwise (30) | library_gemm (1.409x; 30) |
| 321 | policy.g1_half_to_float | copy_reformat | copy_reformat; copyFromHalfKernel; g880x1x1; b512x1x1; r16; s0 | 60 | 1.568 | 1.856 | 1.184x | 57.5% | 0.018 | library_gemm (40) | library_gemm (1.214x; 40) |
| 322 | policy.g1_global_pool | head_elementwise | head_elementwise; gPoolChannelsNHWCKernel; g2x1x13; b64x8x1; r22; s4096 | 60 | 4.512 | 6.848 | 1.518x | 95.5% | 0.173 | library_gemm (60) | library_gemm (1.518x; 60) |
| 323 | policy.gpool_to_bias_matmul | library_gemm | library_gemm; gemmSN_NN_kernel; g2x2x1; b256x1x1; r64; s21504 | 60 | 5.409 | 6.800 | 1.257x | 83.1% | 0.085 | head_elementwise (60) | head_elementwise (1.257x; 60) |
| 324 | policy.p1_half_to_float | copy_reformat | copy_reformat; copyFromHalfKernel; g880x1x1; b512x1x1; r16; s0 | 60 | 1.536 | 1.760 | 1.146x | 73.8% | 0.015 | library_gemm (54) | head_elementwise (1.198x; 6) |
| 325 | policy.gpool_bias_add | head_elementwise | head_elementwise; addNCBiasInplaceNHWCKernel; g1x73x13; b96x5x1; r16; s0 | 60 | 1.824 | 2.064 | 1.132x | 98.9% | 0.017 | library_gemm (60) | library_gemm (1.132x; 60) |
| 326 | policy.p1_norm_silu | head_elementwise | head_elementwise; applyCScaleBiasNHWCSiluKernel; g1x73x13; b96x5x1; r16; s0 | 60 | 2.208 | 2.464 | 1.116x | 82.1% | 0.016 | library_gemm (56) | copy_reformat (1.268x; 4) |
| 327 | policy.p2_conv | library_gemm | library_gemm; Kernel2; g74x1x1; b128x1x1; r90; s98304 | 60 | 3.968 | 4.624 | 1.165x | 89.9% | 0.053 | library_gemm (30) | head_elementwise (1.286x; 28) |
| 328 | policy.gpool_to_pass_matmul | library_gemm | library_gemm; gemmSN_NN_kernel; g2x2x1; b256x1x1; r64; s21504 | 60 | 5.376 | 6.000 | 1.116x | 83.9% | 0.046 | library_gemm (60) | library_gemm (1.116x; 60) |
| 329 | policy.pass_bias_silu | head_elementwise | head_elementwise; addCBiasInplaceNCKernelSilu; g1x3x1; b96x5x1; r16; s0 | 60 | 1.008 | 1.152 | 1.143x | 99.7% | 0.054 | library_gemm (60) | library_gemm (1.143x; 60) |
| 330 | policy.gpool_to_pass_matmul2 | library_gemm | library_gemm; gemmSN_NN_kernel; g1x2x1; b256x1x1; r64; s21504 | 60 | 2.336 | 2.704 | 1.158x | 97.4% | 0.034 | library_gemm (57) | library_gemm (1.137x; 57) |
| 331 | value.v1_conv | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r118; s98304 | 60 | 8.064 | 9.216 | 1.143x | 85.9% | 0.074 | head_elementwise (32) | head_elementwise (1.147x; 32) |
| 332 | value.v1_norm_silu | head_elementwise | head_elementwise; applyCScaleBiasNHWCSiluHalfKernel; g1x181x13; b192x2x1; r16; s0 | 60 | 3.136 | 3.392 | 1.082x | 77.6% | 0.081 | library_gemm (30) | library_gemm (1.842x; 30) |
| 333 | value.v1_half_to_float | copy_reformat | copy_reformat; copyFromHalfKernel; g1760x1x1; b512x1x1; r16; s0 | 60 | 2.176 | 2.496 | 1.147x | 91.8% | 0.045 | library_gemm (33) | library_gemm (1.309x; 33) |
| 334 | value.v1_global_pool | head_elementwise | head_elementwise; valueHeadPoolChannelsNHWCKernel; g3x1x13; b64x8x1; r22; s2048 | 60 | 3.264 | 3.680 | 1.127x | 88.5% | 0.032 | library_gemm (30) | head_elementwise (1.314x; 5) |
| 335 | value.v2_matmul | library_gemm | library_gemm; gemmSN_NN_kernel; g3x2x1; b256x1x1; r64; s21504 | 60 | 9.568 | 10.112 | 1.057x | 93.1% | 0.037 | library_gemm (57) | library_gemm (1.054x; 57) |
| 336 | value.v2_bias_silu | head_elementwise | head_elementwise; addCBiasInplaceNCKernelSilu; g1x7x1; b192x2x1; r16; s0 | 60 | 1.024 | 1.088 | 1.062x | 97.7% | 0.005 | library_gemm (59) | library_gemm (1.062x; 59) |
| 337 | value.v3_matmul | library_gemm | library_gemm; gemmSN_NN_kernel; g1x2x1; b256x1x1; r64; s21504 | 60 | 3.505 | 3.808 | 1.087x | 77.5% | 0.021 | library_gemm (45) | library_gemm (1.105x; 45) |
| 338 | value.v3_bias | head_elementwise | head_elementwise; addCBiasInplaceNCKernel; g1x1x1; b3x170x1; r16; s0 | 60 | 0.960 | 1.056 | 1.100x | 94.5% | 0.007 | library_gemm (58) | library_gemm (1.100x; 58) |
| 339 | value.score_matmul | library_gemm | library_gemm; gemmSN_NN_kernel; g1x2x1; b256x1x1; r64; s21504 | 60 | 3.520 | 3.808 | 1.082x | 78.1% | 0.020 | library_gemm (53) | library_gemm (1.091x; 53) |
| 340 | value.score_bias | head_elementwise | head_elementwise; addCBiasInplaceNCKernel; g1x1x1; b6x85x1; r16; s0 | 60 | 0.928 | 1.024 | 1.103x | 95.8% | 0.007 | library_gemm (56) | library_gemm (1.103x; 56) |
| 341 | value.ownership_conv | library_gemm | library_gemm; Kernel2; g8x19x3; b128x1x1; r118; s33792 | 60 | 4.032 | 4.561 | 1.131x | 60.8% | 0.032 | library_gemm (51) | head_elementwise (1.159x; 7) |
| 342 | value.ownership_conv_splitk_reduce | library_gemm | library_gemm; splitKreduce_kernel; g147x1x1; b32x16x1; r49; s0 | 60 | 1.377 | 1.440 | 1.046x | 70.3% | 0.007 | library_gemm (30) | library_gemm (1.046x; 30) |
| 343 | value.ownership_half_to_float | copy_reformat | copy_reformat; copyFromHalfKernel; g10x1x1; b512x1x1; r16; s0 | 60 | 0.928 | 1.120 | 1.206x | 37.0% | 0.019 | library_gemm (27) | idle (1.706x; 26) |
