# Nsys stream interference report

- Timed iterations: 30; streams: 65, 82
- Kernels per forward: 65=344, 82=344
- Iteration start offset stream 82 - 65: median 262.05 us, p10..p90 177.97..270.30 us, range 102.50..415.75 us.

## Kernel families

| family | instances | total ms | median us | p90 us | peer overlap | S2/S1 median | summed excess ms | matched |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| fused_ffn | 1980 | 88.766 | 43.872 | 50.724 | 70.8% | 1.151x | 12.207 | 1980 |
| library_gemm | 4140 | 68.862 | 18.880 | 23.200 | 96.0% | 1.726x | 30.971 | 4140 |
| linear2_residual | 1980 | 61.746 | 31.584 | 34.432 | 96.5% | 1.513x | 20.472 | 1980 |
| wide_qkv | 1980 | 47.104 | 22.720 | 27.553 | 72.3% | 1.177x | 8.976 | 1980 |
| fa4 | 1980 | 35.522 | 16.832 | 22.144 | 73.5% | 1.416x | 12.015 | 1980 |
| rmsnorm | 3960 | 14.542 | 3.328 | 5.152 | 99.2% | 1.368x | 4.894 | 3960 |
| qk_rope | 1980 | 10.675 | 4.928 | 8.544 | 89.0% | 1.278x | 3.060 | 1980 |
| affine_silu | 1380 | 9.746 | 7.104 | 10.979 | 92.8% | 1.448x | 3.904 | 1380 |
| head_elementwise | 720 | 2.647 | 2.784 | 8.899 | 90.3% | 1.344x | 0.866 | 720 |
| cudnn | 180 | 1.709 | 2.176 | 25.162 | 51.2% | 1.333x | 0.371 | 180 |
| copy_reformat | 300 | 0.701 | 2.144 | 3.716 | 94.4% | 1.478x | 0.278 | 300 |
| sumChannelsNCHWKernel | 60 | 0.157 | 2.192 | 3.619 | 94.2% | 1.292x | 0.056 | 60 |

## Dominant interference pairs

| running family | dominant peer | instances | median overlap | median us | S2/S1 median | matched |
|---|---|---:|---:|---:|---:|---:|
| library_gemm | fused_ffn | 1323 | 97.6% | 20.608 | 2.340x | 1323 |
| qk_rope | wide_qkv | 279 | 100.0% | 8.736 | 2.275x | 279 |
| library_gemm | wide_qkv | 664 | 94.9% | 19.041 | 2.249x | 664 |
| affine_silu | fused_ffn | 264 | 100.0% | 7.328 | 2.189x | 264 |
| affine_silu | wide_qkv | 286 | 97.2% | 10.976 | 2.173x | 286 |
| qk_rope | fused_ffn | 8 | 100.0% | 8.288 | 2.149x | 8 |
| head_elementwise | wide_qkv | 110 | 100.0% | 3.936 | 2.148x | 110 |
| rmsnorm | fused_ffn | 714 | 100.0% | 5.152 | 2.130x | 714 |
| copy_reformat | wide_qkv | 56 | 93.3% | 3.712 | 2.056x | 56 |
| copy_reformat | fa4 | 13 | 100.0% | 3.072 | 2.043x | 13 |
| sumChannelsNCHWKernel | fused_ffn | 19 | 92.9% | 3.424 | 2.019x | 19 |
| rmsnorm | wide_qkv | 284 | 100.0% | 4.448 | 1.829x | 284 |
| copy_reformat | rmsnorm | 17 | 90.6% | 1.696 | 1.828x | 17 |
| fa4 | linear2_residual | 437 | 96.7% | 21.536 | 1.818x | 437 |
| head_elementwise | rmsnorm | 21 | 85.9% | 2.560 | 1.768x | 21 |
| library_gemm | library_gemm | 832 | 96.6% | 14.496 | 1.700x | 832 |
| linear2_residual | linear2_residual | 15 | 96.9% | 34.496 | 1.660x | 15 |
| library_gemm | linear2_residual | 821 | 95.9% | 20.353 | 1.596x | 821 |
| linear2_residual | fused_ffn | 876 | 97.2% | 32.768 | 1.576x | 876 |
| linear2_residual | wide_qkv | 19 | 96.9% | 31.840 | 1.513x | 19 |
| wide_qkv | linear2_residual | 19 | 91.4% | 28.352 | 1.508x | 19 |
| rmsnorm | rmsnorm | 9 | 92.5% | 3.712 | 1.506x | 9 |
| copy_reformat | linear2_residual | 85 | 100.0% | 1.568 | 1.483x | 85 |
| linear2_residual | library_gemm | 582 | 96.7% | 30.689 | 1.478x | 582 |
| head_elementwise | fused_ffn | 171 | 100.0% | 1.856 | 1.438x | 171 |
| copy_reformat | library_gemm | 76 | 100.0% | 2.208 | 1.425x | 76 |
| head_elementwise | qk_rope | 17 | 82.5% | 6.400 | 1.421x | 17 |
| affine_silu | rmsnorm | 70 | 33.3% | 7.232 | 1.419x | 70 |
| linear2_residual | fa4 | 437 | 96.6% | 29.569 | 1.416x | 437 |
| affine_silu | library_gemm | 25 | 59.1% | 7.168 | 1.413x | 25 |
| wide_qkv | fused_ffn | 445 | 98.0% | 27.136 | 1.412x | 445 |
| fa4 | fused_ffn | 596 | 63.3% | 16.672 | 1.405x | 596 |
| rmsnorm | linear2_residual | 617 | 100.0% | 3.360 | 1.382x | 617 |
| cudnn | fused_ffn | 68 | 100.0% | 2.240 | 1.375x | 68 |
| fa4 | fa4 | 6 | 65.1% | 15.968 | 1.359x | 6 |
| qk_rope | fa4 | 279 | 100.0% | 5.184 | 1.347x | 279 |
| fused_ffn | wide_qkv | 96 | 69.6% | 53.105 | 1.344x | 96 |
| affine_silu | linear2_residual | 550 | 100.0% | 6.720 | 1.344x | 550 |
| rmsnorm | fa4 | 1194 | 100.0% | 3.264 | 1.338x | 1194 |
| head_elementwise | linear2_residual | 148 | 100.0% | 3.936 | 1.333x | 148 |

## Logical operation groups

Isolated reference total is the isolated median for each ordinal multiplied by its S2 call count; it is a normalized reference, not a second trace total.

| logical group | families | positions | calls | isolated reference ms | S2 total ms | total ratio | excess ms |
|---|---|---:|---:|---:|---:|---:|---:|
| transformer.ffn_linear2_residual | linear2_residual | 33 | 1980 | 41.274 | 61.746 | 1.496x | 20.472 |
| transformer.attention_out_projection_residual | library_gemm | 33 | 1980 | 16.790 | 37.014 | 2.205x | 20.224 |
| transformer.ffn_linear1_gate_swiglu | fused_ffn | 33 | 1980 | 76.639 | 88.766 | 1.158x | 12.207 |
| transformer.attention_fa4 | fa4 | 33 | 1980 | 23.508 | 35.522 | 1.511x | 12.015 |
| transformer.attention_qkv_projection | wide_qkv | 33 | 1980 | 38.128 | 47.104 | 1.235x | 8.976 |
| outer.post_projection_c384_to_c768_residual | library_gemm | 11 | 660 | 9.489 | 15.252 | 1.607x | 5.763 |
| transformer.attention_qk_rope | qk_rope | 33 | 1980 | 7.615 | 10.675 | 1.402x | 3.060 |
| outer.pre_projection_c768_to_c384 | library_gemm | 11 | 660 | 7.852 | 10.889 | 1.387x | 3.038 |
| transformer.attention_rmsnorm | rmsnorm | 33 | 1980 | 4.811 | 7.647 | 1.590x | 2.838 |
| outer.pre_norm_silu | affine_silu | 11 | 660 | 3.329 | 5.680 | 1.706x | 2.351 |
| transformer.ffn_rmsnorm | rmsnorm | 33 | 1980 | 4.839 | 6.895 | 1.425x | 2.056 |
| outer.post_norm_silu | affine_silu | 11 | 660 | 2.210 | 3.582 | 1.621x | 1.372 |
| value.v1_conv | library_gemm | 1 | 60 | 0.476 | 0.949 | 1.994x | 0.473 |
| policy.g1_conv | library_gemm | 1 | 60 | 0.354 | 0.730 | 2.060x | 0.376 |
| frontend.initial_conv | cudnn | 1 | 60 | 1.169 | 1.450 | 1.240x | 0.281 |
| policy.g1_global_pool | head_elementwise | 1 | 60 | 0.269 | 0.460 | 1.712x | 0.191 |
| trunk.tip_norm_silu | affine_silu | 1 | 60 | 0.303 | 0.484 | 1.595x | 0.181 |
| frontend.initial_global_broadcast_add | head_elementwise | 1 | 60 | 0.463 | 0.607 | 1.312x | 0.144 |
| frontend.initial_global_matmul | library_gemm | 1 | 60 | 0.157 | 0.301 | 1.914x | 0.144 |
| value.v2_matmul | library_gemm | 1 | 60 | 0.572 | 0.715 | 1.250x | 0.143 |
| policy.p1_conv | library_gemm | 1 | 60 | 0.372 | 0.511 | 1.374x | 0.139 |
| policy.p2_conv | library_gemm | 1 | 60 | 0.234 | 0.372 | 1.589x | 0.138 |
| value.v1_norm_silu | head_elementwise | 1 | 60 | 0.188 | 0.318 | 1.688x | 0.129 |
| policy.gpool_to_pass_matmul | library_gemm | 1 | 60 | 0.318 | 0.442 | 1.392x | 0.125 |
| value.ownership_conv | library_gemm | 1 | 60 | 0.240 | 0.341 | 1.422x | 0.101 |
| policy.g1_half_to_float | copy_reformat | 1 | 60 | 0.092 | 0.189 | 2.056x | 0.097 |
| value.v1_global_pool | head_elementwise | 1 | 60 | 0.194 | 0.277 | 1.431x | 0.084 |
| value.v3_matmul | library_gemm | 1 | 60 | 0.207 | 0.289 | 1.395x | 0.082 |
| policy.g1_norm_silu | head_elementwise | 1 | 60 | 0.126 | 0.199 | 1.585x | 0.074 |
| policy.gpool_to_bias_matmul | library_gemm | 1 | 60 | 0.325 | 0.395 | 1.215x | 0.070 |
| value.v1_half_to_float | copy_reformat | 1 | 60 | 0.129 | 0.195 | 1.513x | 0.066 |
| policy.p1_half_to_float | copy_reformat | 1 | 60 | 0.090 | 0.148 | 1.639x | 0.058 |
| input.mask_sum | sumChannelsNCHWKernel | 1 | 60 | 0.102 | 0.157 | 1.548x | 0.056 |
| policy.gpool_bias_add | head_elementwise | 1 | 60 | 0.108 | 0.163 | 1.513x | 0.055 |
| value.ownership_conv_splitk_reduce | library_gemm | 1 | 60 | 0.083 | 0.136 | 1.643x | 0.053 |
| value.v2_bias_silu | head_elementwise | 1 | 60 | 0.061 | 0.113 | 1.832x | 0.051 |
| value.score_matmul | library_gemm | 1 | 60 | 0.209 | 0.260 | 1.243x | 0.051 |
| frontend.initial_conv_nhwc_padding_0 | cudnn | 1 | 60 | 0.077 | 0.127 | 1.658x | 0.051 |
| policy.p1_norm_silu | head_elementwise | 1 | 60 | 0.129 | 0.177 | 1.375x | 0.048 |
| frontend.initial_conv_nhwc_padding_1 | cudnn | 1 | 60 | 0.092 | 0.131 | 1.427x | 0.039 |
| input.extract_mask | head_elementwise | 1 | 60 | 0.069 | 0.101 | 1.457x | 0.032 |
| input.mask_half_to_float | copy_reformat | 1 | 60 | 0.056 | 0.085 | 1.524x | 0.029 |
| policy.gpool_to_pass_matmul2 | library_gemm | 1 | 60 | 0.138 | 0.167 | 1.205x | 0.028 |
| value.ownership_half_to_float | copy_reformat | 1 | 60 | 0.056 | 0.084 | 1.508x | 0.028 |
| value.v3_bias | head_elementwise | 1 | 60 | 0.058 | 0.082 | 1.426x | 0.025 |
| frontend.initial_global_matmul_splitk_reduce | library_gemm | 1 | 60 | 0.075 | 0.098 | 1.308x | 0.023 |
| policy.pass_bias_silu | head_elementwise | 1 | 60 | 0.061 | 0.081 | 1.310x | 0.019 |
| value.score_bias | head_elementwise | 1 | 60 | 0.056 | 0.070 | 1.252x | 0.014 |

## `library_gemm` logical breakdown

| logical group | positions | calls | isolated reference ms | S2 total ms | total ratio | excess ms |
|---|---:|---:|---:|---:|---:|---:|
| transformer.attention_out_projection_residual | 33 | 1980 | 16.790 | 37.014 | 2.205x | 20.224 |
| outer.post_projection_c384_to_c768_residual | 11 | 660 | 9.489 | 15.252 | 1.607x | 5.763 |
| outer.pre_projection_c768_to_c384 | 11 | 660 | 7.852 | 10.889 | 1.387x | 3.038 |
| value.v1_conv | 1 | 60 | 0.476 | 0.949 | 1.994x | 0.473 |
| policy.g1_conv | 1 | 60 | 0.354 | 0.730 | 2.060x | 0.376 |
| frontend.initial_global_matmul | 1 | 60 | 0.157 | 0.301 | 1.914x | 0.144 |
| value.v2_matmul | 1 | 60 | 0.572 | 0.715 | 1.250x | 0.143 |
| policy.p1_conv | 1 | 60 | 0.372 | 0.511 | 1.374x | 0.139 |
| policy.p2_conv | 1 | 60 | 0.234 | 0.372 | 1.589x | 0.138 |
| policy.gpool_to_pass_matmul | 1 | 60 | 0.318 | 0.442 | 1.392x | 0.125 |
| value.ownership_conv | 1 | 60 | 0.240 | 0.341 | 1.422x | 0.101 |
| value.v3_matmul | 1 | 60 | 0.207 | 0.289 | 1.395x | 0.082 |
| policy.gpool_to_bias_matmul | 1 | 60 | 0.325 | 0.395 | 1.215x | 0.070 |
| value.ownership_conv_splitk_reduce | 1 | 60 | 0.083 | 0.136 | 1.643x | 0.053 |
| value.score_matmul | 1 | 60 | 0.209 | 0.260 | 1.243x | 0.051 |
| policy.gpool_to_pass_matmul2 | 1 | 60 | 0.138 | 0.167 | 1.205x | 0.028 |
| frontend.initial_global_matmul_splitk_reduce | 1 | 60 | 0.075 | 0.098 | 1.308x | 0.023 |

## Top ordinal hotspots by summed excess

The worst peer is the highest median S2/S1 slowdown among peer families observed at least four times for that ordinal.

| rank | ordinal | logical position | family | calls | isolated us | S2 us | S2/S1 | excess ms | common peer | worst peer |
|---:|---:|---|---|---:|---:|---:|---:|---:|---|---|
| 1 | 267 | outer_09.transformer_0.block_27.attention_out_projection_residual | library_gemm | 60 | 8.512 | 19.904 | 2.338x | 0.736 | fused_ffn (46) | fused_ffn (2.509x; 46) |
| 2 | 183 | outer_06.transformer_0.block_18.attention_out_projection_residual | library_gemm | 60 | 8.464 | 20.432 | 2.414x | 0.736 | fused_ffn (48) | fused_ffn (2.537x; 48) |
| 3 | 295 | outer_10.transformer_0.block_30.attention_out_projection_residual | library_gemm | 60 | 8.416 | 20.064 | 2.384x | 0.735 | fused_ffn (51) | fused_ffn (2.418x; 51) |
| 4 | 204 | outer_06.post_projection_c384_to_c768_residual | library_gemm | 60 | 14.496 | 27.536 | 1.899x | 0.732 | fused_ffn (27) | linear2_residual (1.939x; 26) |
| 5 | 202 | outer_06.transformer_2.block_20.ffn_linear2_residual | linear2_residual | 60 | 21.056 | 33.601 | 1.596x | 0.731 | fused_ffn (52) | fused_ffn (1.609x; 52) |
| 6 | 230 | outer_07.transformer_2.block_23.ffn_linear2_residual | linear2_residual | 60 | 21.088 | 33.441 | 1.586x | 0.727 | fused_ffn (52) | fused_ffn (1.599x; 52) |
| 7 | 79 | outer_02.transformer_1.block_07.attention_out_projection_residual | library_gemm | 60 | 8.448 | 19.392 | 2.295x | 0.726 | fused_ffn (27) | fused_ffn (2.625x; 27) |
| 8 | 127 | outer_04.transformer_0.block_12.attention_out_projection_residual | library_gemm | 60 | 8.432 | 20.480 | 2.429x | 0.725 | fused_ffn (48) | fused_ffn (2.486x; 48) |
| 9 | 211 | outer_07.transformer_0.block_21.attention_out_projection_residual | library_gemm | 60 | 8.496 | 20.448 | 2.407x | 0.723 | fused_ffn (52) | fused_ffn (2.422x; 52) |
| 10 | 155 | outer_05.transformer_0.block_15.attention_out_projection_residual | library_gemm | 60 | 8.512 | 20.160 | 2.368x | 0.722 | fused_ffn (46) | fused_ffn (2.479x; 46) |
| 11 | 99 | outer_03.transformer_0.block_09.attention_out_projection_residual | library_gemm | 60 | 8.480 | 20.256 | 2.389x | 0.720 | fused_ffn (47) | fused_ffn (2.521x; 47) |
| 12 | 214 | outer_07.transformer_0.block_21.ffn_linear2_residual | linear2_residual | 60 | 21.088 | 31.409 | 1.489x | 0.711 | fa4 (29) | library_gemm (1.731x; 26) |
| 13 | 43 | outer_01.transformer_0.block_03.attention_out_projection_residual | library_gemm | 60 | 8.496 | 19.872 | 2.339x | 0.710 | fused_ffn (42) | fused_ffn (2.537x; 42) |
| 14 | 191 | outer_06.transformer_1.block_19.attention_out_projection_residual | library_gemm | 60 | 8.496 | 19.409 | 2.284x | 0.710 | fused_ffn (29) | fused_ffn (2.595x; 29) |
| 15 | 71 | outer_02.transformer_0.block_06.attention_out_projection_residual | library_gemm | 60 | 8.448 | 20.320 | 2.405x | 0.706 | fused_ffn (45) | fused_ffn (2.527x; 45) |
| 16 | 135 | outer_04.transformer_1.block_13.attention_out_projection_residual | library_gemm | 60 | 8.480 | 19.424 | 2.291x | 0.706 | fused_ffn (28) | fused_ffn (2.553x; 28) |
| 17 | 275 | outer_09.transformer_1.block_28.attention_out_projection_residual | library_gemm | 60 | 8.432 | 19.264 | 2.285x | 0.704 | fused_ffn (29) | fused_ffn (2.554x; 29) |
| 18 | 158 | outer_05.transformer_0.block_15.ffn_linear2_residual | linear2_residual | 60 | 20.928 | 31.904 | 1.524x | 0.695 | fa4 (28) | linear2_residual (1.703x; 5) |
| 19 | 51 | outer_01.transformer_1.block_04.attention_out_projection_residual | library_gemm | 60 | 8.512 | 19.232 | 2.259x | 0.687 | fused_ffn (27) | fused_ffn (2.579x; 27) |
| 20 | 239 | outer_08.transformer_0.block_24.attention_out_projection_residual | library_gemm | 60 | 8.560 | 19.984 | 2.335x | 0.686 | fused_ffn (48) | wide_qkv (2.366x; 5) |
| 21 | 174 | outer_05.transformer_2.block_17.ffn_linear2_residual | linear2_residual | 60 | 20.720 | 32.513 | 1.569x | 0.684 | fused_ffn (50) | fused_ffn (1.575x; 50) |
| 22 | 258 | outer_08.transformer_2.block_26.ffn_linear2_residual | linear2_residual | 60 | 20.896 | 32.608 | 1.560x | 0.683 | fused_ffn (52) | fused_ffn (1.582x; 52) |
| 23 | 247 | outer_08.transformer_1.block_25.attention_out_projection_residual | library_gemm | 60 | 8.496 | 19.537 | 2.299x | 0.675 | fused_ffn (29) | fused_ffn (2.331x; 29) |
| 24 | 250 | outer_08.transformer_1.block_25.ffn_linear2_residual | linear2_residual | 60 | 20.960 | 32.320 | 1.542x | 0.672 | fused_ffn (29) | fused_ffn (1.602x; 29) |
| 25 | 107 | outer_03.transformer_1.block_10.attention_out_projection_residual | library_gemm | 60 | 8.416 | 19.120 | 2.272x | 0.672 | fused_ffn (27) | fused_ffn (2.570x; 27) |
| 26 | 62 | outer_01.transformer_2.block_05.ffn_linear2_residual | linear2_residual | 60 | 20.768 | 32.096 | 1.545x | 0.671 | fused_ffn (48) | fused_ffn (1.565x; 48) |
| 27 | 242 | outer_08.transformer_0.block_24.ffn_linear2_residual | linear2_residual | 60 | 21.024 | 32.112 | 1.527x | 0.669 | fa4 (29) | library_gemm (1.629x; 26) |
| 28 | 74 | outer_02.transformer_0.block_06.ffn_linear2_residual | linear2_residual | 60 | 20.864 | 30.929 | 1.482x | 0.665 | fa4 (27) | library_gemm (1.653x; 25) |
| 29 | 163 | outer_05.transformer_1.block_16.attention_out_projection_residual | library_gemm | 60 | 8.480 | 19.360 | 2.283x | 0.664 | fused_ffn (28) | fused_ffn (2.347x; 28) |
| 30 | 146 | outer_04.transformer_2.block_14.ffn_linear2_residual | linear2_residual | 60 | 20.976 | 32.736 | 1.561x | 0.661 | fused_ffn (50) | fused_ffn (1.564x; 50) |
| 31 | 278 | outer_09.transformer_1.block_28.ffn_linear2_residual | linear2_residual | 60 | 20.640 | 31.472 | 1.525x | 0.659 | fused_ffn (29) | fused_ffn (1.625x; 29) |
| 32 | 46 | outer_01.transformer_0.block_03.ffn_linear2_residual | linear2_residual | 60 | 20.849 | 32.481 | 1.558x | 0.658 | fa4 (27) | library_gemm (1.673x; 23) |
| 33 | 166 | outer_05.transformer_1.block_16.ffn_linear2_residual | linear2_residual | 60 | 20.816 | 32.193 | 1.547x | 0.653 | fused_ffn (28) | fused_ffn (1.596x; 28) |
| 34 | 90 | outer_02.transformer_2.block_08.ffn_linear2_residual | linear2_residual | 60 | 20.881 | 32.145 | 1.539x | 0.649 | fused_ffn (48) | wide_qkv (1.622x; 4) |
| 35 | 219 | outer_07.transformer_1.block_22.attention_out_projection_residual | library_gemm | 60 | 8.576 | 19.264 | 2.246x | 0.649 | fused_ffn (29) | fused_ffn (2.250x; 29) |
| 36 | 102 | outer_03.transformer_0.block_09.ffn_linear2_residual | linear2_residual | 60 | 20.736 | 32.048 | 1.546x | 0.646 | fa4 (27) | library_gemm (1.617x; 19) |
| 37 | 270 | outer_09.transformer_0.block_27.ffn_linear2_residual | linear2_residual | 60 | 20.784 | 31.936 | 1.537x | 0.645 | fa4 (29) | library_gemm (1.604x; 24) |
| 38 | 286 | outer_09.transformer_2.block_29.ffn_linear2_residual | linear2_residual | 60 | 20.592 | 31.280 | 1.519x | 0.638 | fused_ffn (52) | fused_ffn (1.569x; 52) |
| 39 | 118 | outer_03.transformer_2.block_11.ffn_linear2_residual | linear2_residual | 60 | 20.513 | 30.736 | 1.498x | 0.636 | fused_ffn (48) | fused_ffn (1.551x; 48) |
| 40 | 130 | outer_04.transformer_0.block_12.ffn_linear2_residual | linear2_residual | 60 | 20.849 | 30.913 | 1.483x | 0.636 | fa4 (28) | library_gemm (1.684x; 24) |

## Full fixed-forward ordinal map

| ordinal | logical position | family | resource signature | calls | isolated us | S2 us | S2/S1 | overlap | excess ms | common peer | worst peer |
|---:|---|---|---|---:|---:|---:|---:|---:|---:|---|---|
| 0 | input.extract_mask | head_elementwise | head_elementwise; extractChannel0KernelNHWC; g10x1x1; b512x1x1; r16; s0 | 60 | 1.152 | 1.536 | 1.333x | 92.1% | 0.032 | linear2_residual (29) | rmsnorm (1.944x; 11) |
| 1 | input.mask_half_to_float | copy_reformat | copy_reformat; copyFromHalfKernel; g10x1x1; b512x1x1; r16; s0 | 60 | 0.928 | 1.280 | 1.379x | 91.3% | 0.029 | linear2_residual (28) | fused_ffn (2.414x; 6) |
| 2 | input.mask_sum | sumChannelsNCHWKernel | sumChannelsNCHWKernel; sumChannelsNCHWKernel; g1x1x13; b256x2x1; r22; s2048 | 60 | 1.696 | 2.192 | 1.292x | 94.2% | 0.056 | linear2_residual (28) | fused_ffn (2.019x; 19) |
| 3 | frontend.initial_conv_nhwc_padding_0 | cudnn | cudnn; nhwcAddPaddingKernel; g340x1x1; b768x1x1; r34; s0 | 60 | 1.280 | 1.968 | 1.538x | 97.3% | 0.051 | linear2_residual (28) | fused_ffn (1.725x; 22) |
| 4 | frontend.initial_conv_nhwc_padding_1 | cudnn | cudnn; nhwcAddPaddingKernel; g340x1x1; b768x1x1; r34; s0 | 60 | 1.536 | 2.112 | 1.375x | 97.6% | 0.039 | linear2_residual (28) | fused_ffn (1.396x; 23) |
| 5 | frontend.initial_conv | cudnn | cudnn; Kernel; g296x3x1; b128x1x1; r94; s81920 | 60 | 19.488 | 24.177 | 1.241x | 43.0% | 0.281 | fused_ffn (23) | fused_ffn (1.317x; 23) |
| 6 | frontend.initial_global_matmul | library_gemm | library_gemm; Kernel2; g8x1x3; b128x1x1; r128; s24576 | 60 | 2.624 | 3.264 | 1.244x | 98.5% | 0.144 | library_gemm (29) | wide_qkv (3.146x; 5) |
| 7 | frontend.initial_global_matmul_splitk_reduce | library_gemm | library_gemm; splitKreduce_kernel; g24x1x1; b32x16x1; r49; s0 | 60 | 1.248 | 1.600 | 1.282x | 96.1% | 0.023 | library_gemm (28) | linear2_residual (1.308x; 23) |
| 8 | frontend.initial_global_broadcast_add | head_elementwise | head_elementwise; addNCBiasInplaceNHWCHalfKernel; g3x361x13; b256x1x1; r16; s0 | 60 | 7.712 | 10.448 | 1.355x | 82.8% | 0.144 | library_gemm (26) | linear2_residual (1.382x; 23) |
| 9 | outer_00.pre_norm_silu | affine_silu | affine_silu; affineSiluHalf2Kernel; g4693x1x1; b384x1x1; r17; s0 | 60 | 4.928 | 6.688 | 1.357x | 62.0% | 0.103 | linear2_residual (24) | linear2_residual (1.468x; 24) |
| 10 | outer_00.pre_projection_c768_to_c384 | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 60 | 11.584 | 15.681 | 1.354x | 95.7% | 0.292 | library_gemm (29) | fused_ffn (2.069x; 5) |
| 11 | outer_00.transformer_0.block_00.attention_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | 2.368 | 3.488 | 1.473x | 97.3% | 0.066 | library_gemm (27) | fused_ffn (2.162x; 5) |
| 12 | outer_00.transformer_0.block_00.attention_qkv_projection | wide_qkv | wide_qkv; wide_qkv_kernel; g9x37x1; b128x1x1; r136; s65536 | 60 | 18.800 | 20.992 | 1.117x | 63.2% | 0.181 | qk_rope (23) | fused_ffn (1.369x; 5) |
| 13 | outer_00.transformer_0.block_00.attention_qk_rope | qk_rope | qk_rope; fusedQKRoPE19Half2Kernel; g361x13x1; b192x1x1; r16; s0 | 60 | 3.776 | 4.496 | 1.191x | 85.0% | 0.048 | head_elementwise (25) | library_gemm (1.610x; 5) |
| 14 | outer_00.transformer_0.block_00.attention_fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 60 | 11.680 | 14.048 | 1.203x | 68.2% | 0.187 | library_gemm (49) | linear2_residual (1.674x; 5) |
| 15 | outer_00.transformer_0.block_00.attention_out_projection_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 60 | 8.416 | 12.544 | 1.490x | 92.9% | 0.361 | library_gemm (24) | fused_ffn (2.304x; 22) |
| 16 | outer_00.transformer_0.block_00.ffn_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | 2.432 | 3.104 | 1.276x | 91.1% | 0.079 | library_gemm (29) | fused_ffn (2.171x; 23) |
| 17 | outer_00.transformer_0.block_00.ffn_linear1_gate_swiglu | fused_ffn | fused_ffn; fused_ffn_candidate_a_reuse_kernel; g18x37x1; b128x1x1; r136; s32768 | 60 | 39.248 | 40.000 | 1.019x | 67.4% | 0.109 | library_gemm (31) | linear2_residual (1.023x; 11) |
| 18 | outer_00.transformer_0.block_00.ffn_linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 60 | 20.864 | 26.512 | 1.271x | 93.3% | 0.447 | library_gemm (48) | fa4 (1.296x; 4) |
| 19 | outer_00.transformer_1.block_01.attention_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | 2.464 | 2.608 | 1.058x | 99.0% | 0.019 | library_gemm (48) | fa4 (1.279x; 4) |
| 20 | outer_00.transformer_1.block_01.attention_qkv_projection | wide_qkv | wide_qkv; wide_qkv_kernel; g9x37x1; b128x1x1; r136; s65536 | 60 | 19.392 | 21.648 | 1.116x | 60.1% | 0.148 | library_gemm (31) | affine_silu (1.149x; 23) |
| 21 | outer_00.transformer_1.block_01.attention_qk_rope | qk_rope | qk_rope; fusedQKRoPE19Half2Kernel; g361x13x1; b192x1x1; r16; s0 | 60 | 3.872 | 4.368 | 1.128x | 39.1% | 0.036 | library_gemm (51) | library_gemm (1.140x; 51) |
| 22 | outer_00.transformer_1.block_01.attention_fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 60 | 11.936 | 15.905 | 1.332x | 78.2% | 0.324 | library_gemm (48) | library_gemm (1.393x; 48) |
| 23 | outer_00.transformer_1.block_01.attention_out_projection_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 60 | 8.496 | 18.416 | 2.168x | 85.1% | 0.364 | library_gemm (29) | fused_ffn (2.495x; 4) |
| 24 | outer_00.transformer_1.block_01.ffn_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | 2.464 | 3.008 | 1.221x | 91.9% | 0.032 | fa4 (24) | linear2_residual (1.383x; 4) |
| 25 | outer_00.transformer_1.block_01.ffn_linear1_gate_swiglu | fused_ffn | fused_ffn; fused_ffn_candidate_a_reuse_kernel; g18x37x1; b128x1x1; r136; s32768 | 60 | 37.856 | 41.456 | 1.095x | 48.3% | 0.291 | library_gemm (27) | linear2_residual (1.318x; 4) |
| 26 | outer_00.transformer_1.block_01.ffn_linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 60 | 20.817 | 26.529 | 1.274x | 94.4% | 0.467 | fused_ffn (24) | fused_ffn (1.582x; 24) |
| 27 | outer_00.transformer_2.block_02.attention_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | 2.400 | 2.912 | 1.213x | 92.2% | 0.059 | fused_ffn (24) | fused_ffn (1.893x; 24) |
| 28 | outer_00.transformer_2.block_02.attention_qkv_projection | wide_qkv | wide_qkv; wide_qkv_kernel; g9x37x1; b128x1x1; r136; s65536 | 60 | 19.200 | 26.785 | 1.395x | 82.4% | 0.442 | library_gemm (34) | library_gemm (1.531x; 34) |
| 29 | outer_00.transformer_2.block_02.attention_qk_rope | qk_rope | qk_rope; fusedQKRoPE19Half2Kernel; g361x13x1; b192x1x1; r16; s0 | 60 | 3.808 | 5.104 | 1.340x | 97.5% | 0.155 | linear2_residual (24) | wide_qkv (2.273x; 24) |
| 30 | outer_00.transformer_2.block_02.attention_fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 60 | 11.824 | 15.953 | 1.349x | 73.2% | 0.335 | linear2_residual (24) | linear2_residual (1.821x; 24) |
| 31 | outer_00.transformer_2.block_02.attention_out_projection_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 60 | 8.448 | 18.752 | 2.220x | 96.0% | 0.507 | library_gemm (26) | fused_ffn (2.561x; 9) |
| 32 | outer_00.transformer_2.block_02.ffn_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | 2.464 | 3.200 | 1.299x | 99.6% | 0.034 | library_gemm (26) | linear2_residual (1.351x; 9) |
| 33 | outer_00.transformer_2.block_02.ffn_linear1_gate_swiglu | fused_ffn | fused_ffn; fused_ffn_candidate_a_reuse_kernel; g18x37x1; b128x1x1; r136; s32768 | 60 | 38.224 | 45.728 | 1.196x | 61.1% | 0.413 | library_gemm (25) | linear2_residual (1.284x; 9) |
| 34 | outer_00.transformer_2.block_02.ffn_linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 60 | 20.768 | 31.232 | 1.504x | 97.4% | 0.626 | fused_ffn (48) | fused_ffn (1.527x; 48) |
| 35 | outer_00.post_norm_silu | affine_silu | affine_silu; affineSiluHalf2Kernel; g4693x1x1; b192x1x1; r17; s0 | 60 | 3.328 | 4.160 | 1.250x | 100.0% | 0.115 | fused_ffn (24) | fused_ffn (2.135x; 24) |
| 36 | outer_00.post_projection_c384_to_c768_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b256x1x1; r154; s73728 | 60 | 14.320 | 22.912 | 1.600x | 97.3% | 0.466 | linear2_residual (45) | linear2_residual (1.611x; 45) |
| 37 | outer_01.pre_norm_silu | affine_silu | affine_silu; affineSiluHalf2Kernel; g4693x1x1; b384x1x1; r17; s0 | 60 | 5.120 | 7.152 | 1.397x | 89.2% | 0.207 | linear2_residual (24) | wide_qkv (2.163x; 24) |
| 38 | outer_01.pre_projection_c768_to_c384 | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 60 | 11.904 | 14.640 | 1.230x | 95.3% | 0.220 | fa4 (24) | linear2_residual (1.431x; 24) |
| 39 | outer_01.transformer_0.block_03.attention_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | 2.432 | 4.496 | 1.849x | 100.0% | 0.129 | fa4 (24) | fused_ffn (2.099x; 12) |
| 40 | outer_01.transformer_0.block_03.attention_qkv_projection | wide_qkv | wide_qkv; wide_qkv_kernel; g9x37x1; b128x1x1; r136; s65536 | 60 | 19.152 | 22.561 | 1.178x | 71.3% | 0.251 | library_gemm (24) | fused_ffn (1.390x; 12) |
| 41 | outer_01.transformer_0.block_03.attention_qk_rope | qk_rope | qk_rope; fusedQKRoPE19Half2Kernel; g361x13x1; b192x1x1; r16; s0 | 60 | 3.808 | 4.880 | 1.282x | 100.0% | 0.067 | fa4 (24) | fa4 (1.361x; 24) |
| 42 | outer_01.transformer_0.block_03.attention_fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 60 | 11.809 | 15.953 | 1.351x | 67.0% | 0.289 | fused_ffn (24) | linear2_residual (1.726x; 12) |
| 43 | outer_01.transformer_0.block_03.attention_out_projection_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 60 | 8.496 | 19.872 | 2.339x | 96.4% | 0.710 | fused_ffn (42) | fused_ffn (2.537x; 42) |
| 44 | outer_01.transformer_0.block_03.ffn_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | 2.432 | 3.616 | 1.487x | 99.5% | 0.123 | fused_ffn (24) | fused_ffn (2.493x; 24) |
| 45 | outer_01.transformer_0.block_03.ffn_linear1_gate_swiglu | fused_ffn | fused_ffn; fused_ffn_candidate_a_reuse_kernel; g18x37x1; b128x1x1; r136; s32768 | 60 | 39.760 | 40.913 | 1.029x | 81.2% | 0.248 | linear2_residual (38) | linear2_residual (1.202x; 38) |
| 46 | outer_01.transformer_0.block_03.ffn_linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 60 | 20.849 | 32.481 | 1.558x | 96.8% | 0.658 | fa4 (27) | library_gemm (1.673x; 23) |
| 47 | outer_01.transformer_1.block_04.attention_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | 2.432 | 2.976 | 1.224x | 100.0% | 0.045 | fa4 (27) | fused_ffn (2.066x; 9) |
| 48 | outer_01.transformer_1.block_04.attention_qkv_projection | wide_qkv | wide_qkv; wide_qkv_kernel; g9x37x1; b128x1x1; r136; s65536 | 60 | 19.328 | 22.384 | 1.158x | 66.2% | 0.227 | library_gemm (27) | fused_ffn (1.407x; 9) |
| 49 | outer_01.transformer_1.block_04.attention_qk_rope | qk_rope | qk_rope; fusedQKRoPE19Half2Kernel; g361x13x1; b192x1x1; r16; s0 | 60 | 3.808 | 4.608 | 1.210x | 69.2% | 0.050 | library_gemm (51) | linear2_residual (1.311x; 9) |
| 50 | outer_01.transformer_1.block_04.attention_fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 60 | 11.888 | 20.112 | 1.692x | 81.3% | 0.450 | fused_ffn (27) | library_gemm (1.848x; 24) |
| 51 | outer_01.transformer_1.block_04.attention_out_projection_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 60 | 8.512 | 19.232 | 2.259x | 96.7% | 0.687 | fused_ffn (27) | fused_ffn (2.579x; 27) |
| 52 | outer_01.transformer_1.block_04.ffn_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | 2.448 | 3.264 | 1.333x | 98.9% | 0.049 | fa4 (27) | linear2_residual (1.359x; 27) |
| 53 | outer_01.transformer_1.block_04.ffn_linear1_gate_swiglu | fused_ffn | fused_ffn; fused_ffn_candidate_a_reuse_kernel; g18x37x1; b128x1x1; r136; s32768 | 60 | 38.560 | 47.057 | 1.220x | 65.3% | 0.432 | library_gemm (29) | wide_qkv (1.397x; 4) |
| 54 | outer_01.transformer_1.block_04.ffn_linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 60 | 20.721 | 30.176 | 1.456x | 97.0% | 0.607 | fused_ffn (27) | fused_ffn (1.583x; 27) |
| 55 | outer_01.transformer_2.block_05.attention_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | 2.432 | 3.360 | 1.382x | 100.0% | 0.088 | fused_ffn (27) | fused_ffn (1.987x; 27) |
| 56 | outer_01.transformer_2.block_05.attention_qkv_projection | wide_qkv | wide_qkv; wide_qkv_kernel; g9x37x1; b128x1x1; r136; s65536 | 60 | 19.200 | 22.768 | 1.186x | 81.1% | 0.319 | fused_ffn (27) | fused_ffn (1.423x; 27) |
| 57 | outer_01.transformer_2.block_05.attention_qk_rope | qk_rope | qk_rope; fusedQKRoPE19Half2Kernel; g361x13x1; b192x1x1; r16; s0 | 60 | 3.840 | 5.168 | 1.346x | 100.0% | 0.154 | linear2_residual (27) | wide_qkv (2.208x; 24) |
| 58 | outer_01.transformer_2.block_05.attention_fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 60 | 11.809 | 16.288 | 1.379x | 75.4% | 0.353 | linear2_residual (27) | linear2_residual (1.786x; 27) |
| 59 | outer_01.transformer_2.block_05.attention_out_projection_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 60 | 8.480 | 18.848 | 2.223x | 96.1% | 0.544 | library_gemm (28) | fused_ffn (2.547x; 8) |
| 60 | outer_01.transformer_2.block_05.ffn_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | 2.432 | 3.104 | 1.276x | 99.1% | 0.042 | fa4 (24) | linear2_residual (1.408x; 7) |
| 61 | outer_01.transformer_2.block_05.ffn_linear1_gate_swiglu | fused_ffn | fused_ffn; fused_ffn_candidate_a_reuse_kernel; g18x37x1; b128x1x1; r136; s32768 | 60 | 38.400 | 46.288 | 1.205x | 61.1% | 0.434 | fused_ffn (24) | linear2_residual (1.290x; 9) |
| 62 | outer_01.transformer_2.block_05.ffn_linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 60 | 20.768 | 32.096 | 1.545x | 97.6% | 0.671 | fused_ffn (48) | fused_ffn (1.565x; 48) |
| 63 | outer_01.post_norm_silu | affine_silu | affine_silu; affineSiluHalf2Kernel; g4693x1x1; b192x1x1; r17; s0 | 60 | 3.328 | 4.320 | 1.298x | 99.7% | 0.114 | fused_ffn (24) | fused_ffn (2.053x; 24) |
| 64 | outer_01.post_projection_c384_to_c768_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b256x1x1; r154; s73728 | 60 | 14.336 | 22.880 | 1.596x | 97.0% | 0.487 | linear2_residual (41) | fused_ffn (1.728x; 10) |
| 65 | outer_02.pre_norm_silu | affine_silu | affine_silu; affineSiluHalf2Kernel; g4693x1x1; b384x1x1; r17; s0 | 60 | 5.024 | 7.248 | 1.443x | 92.7% | 0.231 | linear2_residual (24) | wide_qkv (2.261x; 24) |
| 66 | outer_02.pre_projection_c768_to_c384 | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 60 | 11.920 | 15.248 | 1.279x | 95.6% | 0.263 | linear2_residual (26) | linear2_residual (1.481x; 26) |
| 67 | outer_02.transformer_0.block_06.attention_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | 2.432 | 4.448 | 1.829x | 100.0% | 0.116 | fa4 (24) | fused_ffn (2.007x; 6) |
| 68 | outer_02.transformer_0.block_06.attention_qkv_projection | wide_qkv | wide_qkv; wide_qkv_kernel; g9x37x1; b128x1x1; r136; s65536 | 60 | 19.201 | 22.640 | 1.179x | 69.6% | 0.260 | library_gemm (24) | linear2_residual (1.525x; 4) |
| 69 | outer_02.transformer_0.block_06.attention_qk_rope | qk_rope | qk_rope; fusedQKRoPE19Half2Kernel; g361x13x1; b192x1x1; r16; s0 | 60 | 3.872 | 4.928 | 1.273x | 100.0% | 0.079 | library_gemm (26) | linear2_residual (1.343x; 6) |
| 70 | outer_02.transformer_0.block_06.attention_fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 60 | 11.825 | 16.209 | 1.371x | 65.1% | 0.294 | fused_ffn (26) | linear2_residual (1.736x; 6) |
| 71 | outer_02.transformer_0.block_06.attention_out_projection_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 60 | 8.448 | 20.320 | 2.405x | 96.7% | 0.706 | fused_ffn (45) | fused_ffn (2.527x; 45) |
| 72 | outer_02.transformer_0.block_06.ffn_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | 2.432 | 3.520 | 1.447x | 99.8% | 0.101 | fused_ffn (24) | fused_ffn (2.152x; 24) |
| 73 | outer_02.transformer_0.block_06.ffn_linear1_gate_swiglu | fused_ffn | fused_ffn; fused_ffn_candidate_a_reuse_kernel; g18x37x1; b128x1x1; r136; s32768 | 60 | 40.801 | 44.912 | 1.101x | 80.9% | 0.303 | linear2_residual (29) | wide_qkv (1.236x; 4) |
| 74 | outer_02.transformer_0.block_06.ffn_linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 60 | 20.864 | 30.929 | 1.482x | 97.2% | 0.665 | fa4 (27) | library_gemm (1.653x; 25) |
| 75 | outer_02.transformer_1.block_07.attention_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | 2.432 | 2.880 | 1.184x | 99.5% | 0.038 | fa4 (27) | fused_ffn (2.092x; 5) |
| 76 | outer_02.transformer_1.block_07.attention_qkv_projection | wide_qkv | wide_qkv; wide_qkv_kernel; g9x37x1; b128x1x1; r136; s65536 | 60 | 19.201 | 22.384 | 1.166x | 66.2% | 0.230 | library_gemm (27) | fused_ffn (1.422x; 7) |
| 77 | outer_02.transformer_1.block_07.attention_qk_rope | qk_rope | qk_rope; fusedQKRoPE19Half2Kernel; g361x13x1; b192x1x1; r16; s0 | 60 | 3.840 | 4.672 | 1.217x | 70.7% | 0.061 | library_gemm (53) | linear2_residual (1.367x; 5) |
| 78 | outer_02.transformer_1.block_07.attention_fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 60 | 11.824 | 20.272 | 1.714x | 80.1% | 0.454 | fused_ffn (29) | library_gemm (1.851x; 26) |
| 79 | outer_02.transformer_1.block_07.attention_out_projection_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 60 | 8.448 | 19.392 | 2.295x | 96.6% | 0.726 | fused_ffn (27) | fused_ffn (2.625x; 27) |
| 80 | outer_02.transformer_1.block_07.ffn_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | 2.464 | 3.344 | 1.357x | 98.6% | 0.052 | fa4 (27) | linear2_residual (1.390x; 27) |
| 81 | outer_02.transformer_1.block_07.ffn_linear1_gate_swiglu | fused_ffn | fused_ffn; fused_ffn_candidate_a_reuse_kernel; g18x37x1; b128x1x1; r136; s32768 | 60 | 40.720 | 47.472 | 1.166x | 65.9% | 0.297 | library_gemm (29) | linear2_residual (1.180x; 27) |
| 82 | outer_02.transformer_1.block_07.ffn_linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 60 | 20.864 | 30.032 | 1.439x | 97.1% | 0.595 | fused_ffn (27) | fused_ffn (1.587x; 27) |
| 83 | outer_02.transformer_2.block_08.attention_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | 2.464 | 3.504 | 1.422x | 99.7% | 0.093 | fused_ffn (27) | fused_ffn (2.091x; 27) |
| 84 | outer_02.transformer_2.block_08.attention_qkv_projection | wide_qkv | wide_qkv; wide_qkv_kernel; g9x37x1; b128x1x1; r136; s65536 | 60 | 19.216 | 23.056 | 1.200x | 80.4% | 0.327 | fused_ffn (29) | fused_ffn (1.409x; 29) |
| 85 | outer_02.transformer_2.block_08.attention_qk_rope | qk_rope | qk_rope; fusedQKRoPE19Half2Kernel; g361x13x1; b192x1x1; r16; s0 | 60 | 3.840 | 5.216 | 1.358x | 100.0% | 0.162 | linear2_residual (27) | wide_qkv (2.246x; 24) |
| 86 | outer_02.transformer_2.block_08.attention_fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 60 | 11.888 | 16.640 | 1.400x | 76.4% | 0.355 | linear2_residual (27) | linear2_residual (1.787x; 27) |
| 87 | outer_02.transformer_2.block_08.attention_out_projection_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 60 | 8.512 | 18.785 | 2.207x | 96.0% | 0.525 | library_gemm (28) | fused_ffn (2.554x; 6) |
| 88 | outer_02.transformer_2.block_08.ffn_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | 2.432 | 3.120 | 1.283x | 99.5% | 0.045 | library_gemm (25) | linear2_residual (1.382x; 5) |
| 89 | outer_02.transformer_2.block_08.ffn_linear1_gate_swiglu | fused_ffn | fused_ffn; fused_ffn_candidate_a_reuse_kernel; g18x37x1; b128x1x1; r136; s32768 | 60 | 37.873 | 46.416 | 1.226x | 60.7% | 0.441 | library_gemm (26) | fused_ffn (1.283x; 24) |
| 90 | outer_02.transformer_2.block_08.ffn_linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 60 | 20.881 | 32.145 | 1.539x | 97.5% | 0.649 | fused_ffn (48) | wide_qkv (1.622x; 4) |
| 91 | outer_02.post_norm_silu | affine_silu | affine_silu; affineSiluHalf2Kernel; g4693x1x1; b192x1x1; r17; s0 | 60 | 3.360 | 4.368 | 1.300x | 99.6% | 0.126 | fused_ffn (24) | fused_ffn (2.143x; 24) |
| 92 | outer_02.post_projection_c384_to_c768_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b256x1x1; r154; s73728 | 60 | 14.433 | 22.977 | 1.592x | 97.1% | 0.473 | linear2_residual (41) | linear2_residual (1.610x; 41) |
| 93 | outer_03.pre_norm_silu | affine_silu | affine_silu; affineSiluHalf2Kernel; g4693x1x1; b384x1x1; r17; s0 | 60 | 5.056 | 7.056 | 1.396x | 92.7% | 0.225 | linear2_residual (24) | wide_qkv (2.209x; 24) |
| 94 | outer_03.pre_projection_c768_to_c384 | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 60 | 11.952 | 15.569 | 1.303x | 95.4% | 0.273 | fa4 (24) | library_gemm (2.096x; 4) |
| 95 | outer_03.transformer_0.block_09.attention_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | 2.432 | 4.576 | 1.882x | 100.0% | 0.130 | fa4 (24) | fused_ffn (2.178x; 10) |
| 96 | outer_03.transformer_0.block_09.attention_qkv_projection | wide_qkv | wide_qkv; wide_qkv_kernel; g9x37x1; b128x1x1; r136; s65536 | 60 | 19.264 | 22.529 | 1.169x | 70.9% | 0.260 | library_gemm (24) | fused_ffn (1.388x; 10) |
| 97 | outer_03.transformer_0.block_09.attention_qk_rope | qk_rope | qk_rope; fusedQKRoPE19Half2Kernel; g361x13x1; b192x1x1; r16; s0 | 60 | 3.840 | 4.880 | 1.271x | 100.0% | 0.070 | fa4 (24) | fa4 (1.346x; 24) |
| 98 | outer_03.transformer_0.block_09.attention_fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 60 | 11.921 | 16.224 | 1.361x | 66.2% | 0.297 | fused_ffn (24) | linear2_residual (1.689x; 10) |
| 99 | outer_03.transformer_0.block_09.attention_out_projection_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 60 | 8.480 | 20.256 | 2.389x | 96.6% | 0.720 | fused_ffn (47) | fused_ffn (2.521x; 47) |
| 100 | outer_03.transformer_0.block_09.ffn_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | 2.432 | 3.536 | 1.454x | 99.5% | 0.109 | fused_ffn (24) | fused_ffn (2.270x; 24) |
| 101 | outer_03.transformer_0.block_09.ffn_linear1_gate_swiglu | fused_ffn | fused_ffn; fused_ffn_candidate_a_reuse_kernel; g18x37x1; b128x1x1; r136; s32768 | 60 | 38.129 | 42.752 | 1.121x | 79.5% | 0.426 | linear2_residual (31) | linear2_residual (1.306x; 31) |
| 102 | outer_03.transformer_0.block_09.ffn_linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 60 | 20.736 | 32.048 | 1.546x | 97.1% | 0.646 | fa4 (27) | library_gemm (1.617x; 19) |
| 103 | outer_03.transformer_1.block_10.attention_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | 2.432 | 2.864 | 1.178x | 100.0% | 0.038 | fa4 (27) | fused_ffn (2.053x; 7) |
| 104 | outer_03.transformer_1.block_10.attention_qkv_projection | wide_qkv | wide_qkv; wide_qkv_kernel; g9x37x1; b128x1x1; r136; s65536 | 60 | 19.216 | 22.016 | 1.146x | 66.2% | 0.208 | library_gemm (27) | fused_ffn (1.400x; 7) |
| 105 | outer_03.transformer_1.block_10.attention_qk_rope | qk_rope | qk_rope; fusedQKRoPE19Half2Kernel; g361x13x1; b192x1x1; r16; s0 | 60 | 3.824 | 4.656 | 1.218x | 69.9% | 0.052 | library_gemm (53) | linear2_residual (1.331x; 7) |
| 106 | outer_03.transformer_1.block_10.attention_fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 60 | 11.776 | 19.056 | 1.618x | 80.0% | 0.427 | fused_ffn (27) | library_gemm (1.804x; 26) |
| 107 | outer_03.transformer_1.block_10.attention_out_projection_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 60 | 8.416 | 19.120 | 2.272x | 96.6% | 0.672 | fused_ffn (27) | fused_ffn (2.570x; 27) |
| 108 | outer_03.transformer_1.block_10.ffn_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | 2.401 | 3.360 | 1.399x | 99.1% | 0.057 | fa4 (27) | linear2_residual (1.439x; 27) |
| 109 | outer_03.transformer_1.block_10.ffn_linear1_gate_swiglu | fused_ffn | fused_ffn; fused_ffn_candidate_a_reuse_kernel; g18x37x1; b128x1x1; r136; s32768 | 60 | 40.096 | 47.089 | 1.174x | 65.4% | 0.310 | library_gemm (28) | wide_qkv (1.313x; 5) |
| 110 | outer_03.transformer_1.block_10.ffn_linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 60 | 20.592 | 30.177 | 1.465x | 97.0% | 0.606 | fused_ffn (27) | fused_ffn (1.591x; 27) |
| 111 | outer_03.transformer_2.block_11.attention_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | 2.368 | 3.408 | 1.439x | 100.0% | 0.100 | fused_ffn (27) | fused_ffn (2.149x; 27) |
| 112 | outer_03.transformer_2.block_11.attention_qkv_projection | wide_qkv | wide_qkv; wide_qkv_kernel; g9x37x1; b128x1x1; r136; s65536 | 60 | 19.104 | 22.673 | 1.187x | 81.2% | 0.288 | fused_ffn (27) | fused_ffn (1.350x; 27) |
| 113 | outer_03.transformer_2.block_11.attention_qk_rope | qk_rope | qk_rope; fusedQKRoPE19Half2Kernel; g361x13x1; b192x1x1; r16; s0 | 60 | 3.808 | 5.136 | 1.349x | 100.0% | 0.155 | linear2_residual (27) | wide_qkv (2.231x; 24) |
| 114 | outer_03.transformer_2.block_11.attention_fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 60 | 11.760 | 16.608 | 1.412x | 75.8% | 0.370 | linear2_residual (27) | linear2_residual (1.820x; 27) |
| 115 | outer_03.transformer_2.block_11.attention_out_projection_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 60 | 8.400 | 18.784 | 2.236x | 96.1% | 0.542 | library_gemm (28) | fused_ffn (2.556x; 8) |
| 116 | outer_03.transformer_2.block_11.ffn_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | 2.464 | 3.200 | 1.299x | 99.2% | 0.038 | fa4 (24) | linear2_residual (1.351x; 8) |
| 117 | outer_03.transformer_2.block_11.ffn_linear1_gate_swiglu | fused_ffn | fused_ffn; fused_ffn_candidate_a_reuse_kernel; g18x37x1; b128x1x1; r136; s32768 | 60 | 40.624 | 45.697 | 1.125x | 62.1% | 0.274 | fused_ffn (24) | linear2_residual (1.232x; 9) |
| 118 | outer_03.transformer_2.block_11.ffn_linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 60 | 20.513 | 30.736 | 1.498x | 97.5% | 0.636 | fused_ffn (48) | fused_ffn (1.551x; 48) |
| 119 | outer_03.post_norm_silu | affine_silu | affine_silu; affineSiluHalf2Kernel; g4693x1x1; b192x1x1; r17; s0 | 60 | 3.312 | 4.176 | 1.261x | 100.0% | 0.123 | fused_ffn (24) | fused_ffn (2.183x; 24) |
| 120 | outer_03.post_projection_c384_to_c768_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b256x1x1; r154; s73728 | 60 | 14.161 | 21.776 | 1.538x | 97.1% | 0.448 | fused_ffn (25) | fused_ffn (1.679x; 25) |
| 121 | outer_04.pre_norm_silu | affine_silu | affine_silu; affineSiluHalf2Kernel; g4693x1x1; b384x1x1; r17; s0 | 60 | 4.992 | 7.184 | 1.439x | 90.8% | 0.221 | linear2_residual (24) | wide_qkv (2.208x; 24) |
| 122 | outer_04.pre_projection_c768_to_c384 | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 60 | 11.744 | 14.992 | 1.277x | 95.5% | 0.276 | linear2_residual (25) | linear2_residual (1.649x; 25) |
| 123 | outer_04.transformer_0.block_12.attention_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | 2.432 | 4.448 | 1.829x | 100.0% | 0.125 | fa4 (24) | fused_ffn (2.132x; 10) |
| 124 | outer_04.transformer_0.block_12.attention_qkv_projection | wide_qkv | wide_qkv; wide_qkv_kernel; g9x37x1; b128x1x1; r136; s65536 | 60 | 18.928 | 21.856 | 1.155x | 70.2% | 0.241 | library_gemm (24) | fused_ffn (1.405x; 11) |
| 125 | outer_04.transformer_0.block_12.attention_qk_rope | qk_rope | qk_rope; fusedQKRoPE19Half2Kernel; g361x13x1; b192x1x1; r16; s0 | 60 | 3.808 | 4.736 | 1.244x | 100.0% | 0.071 | fa4 (24) | fa4 (1.361x; 24) |
| 126 | outer_04.transformer_0.block_12.attention_fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 60 | 11.744 | 16.032 | 1.365x | 66.6% | 0.296 | library_gemm (24) | linear2_residual (1.789x; 10) |
| 127 | outer_04.transformer_0.block_12.attention_out_projection_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 60 | 8.432 | 20.480 | 2.429x | 96.6% | 0.725 | fused_ffn (48) | fused_ffn (2.486x; 48) |
| 128 | outer_04.transformer_0.block_12.ffn_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | 2.432 | 3.584 | 1.474x | 99.3% | 0.089 | fused_ffn (24) | fused_ffn (1.908x; 24) |
| 129 | outer_04.transformer_0.block_12.ffn_linear1_gate_swiglu | fused_ffn | fused_ffn; fused_ffn_candidate_a_reuse_kernel; g18x37x1; b128x1x1; r136; s32768 | 60 | 38.209 | 43.984 | 1.151x | 82.9% | 0.382 | linear2_residual (48) | wide_qkv (1.340x; 4) |
| 130 | outer_04.transformer_0.block_12.ffn_linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 60 | 20.849 | 30.913 | 1.483x | 96.9% | 0.636 | fa4 (28) | library_gemm (1.684x; 24) |
| 131 | outer_04.transformer_1.block_13.attention_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | 2.400 | 2.832 | 1.180x | 100.0% | 0.047 | fa4 (28) | fused_ffn (2.413x; 7) |
| 132 | outer_04.transformer_1.block_13.attention_qkv_projection | wide_qkv | wide_qkv; wide_qkv_kernel; g9x37x1; b128x1x1; r136; s65536 | 60 | 19.328 | 22.896 | 1.185x | 70.3% | 0.243 | library_gemm (28) | fused_ffn (1.435x; 7) |
| 133 | outer_04.transformer_1.block_13.attention_qk_rope | qk_rope | qk_rope; fusedQKRoPE19Half2Kernel; g361x13x1; b192x1x1; r16; s0 | 60 | 3.840 | 4.641 | 1.208x | 69.2% | 0.052 | library_gemm (53) | linear2_residual (1.317x; 7) |
| 134 | outer_04.transformer_1.block_13.attention_fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 60 | 11.873 | 20.529 | 1.729x | 80.4% | 0.465 | fused_ffn (27) | library_gemm (1.871x; 25) |
| 135 | outer_04.transformer_1.block_13.attention_out_projection_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 60 | 8.480 | 19.424 | 2.291x | 96.6% | 0.706 | fused_ffn (28) | fused_ffn (2.553x; 28) |
| 136 | outer_04.transformer_1.block_13.ffn_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | 2.432 | 3.232 | 1.329x | 98.8% | 0.051 | linear2_residual (28) | fa4 (1.368x; 27) |
| 137 | outer_04.transformer_1.block_13.ffn_linear1_gate_swiglu | fused_ffn | fused_ffn; fused_ffn_candidate_a_reuse_kernel; g18x37x1; b128x1x1; r136; s32768 | 60 | 38.273 | 45.392 | 1.186x | 72.9% | 0.370 | linear2_residual (28) | wide_qkv (1.339x; 4) |
| 138 | outer_04.transformer_1.block_13.ffn_linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 60 | 20.832 | 29.169 | 1.400x | 96.7% | 0.516 | fused_ffn (27) | fused_ffn (1.456x; 27) |
| 139 | outer_04.transformer_2.block_14.attention_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | 2.432 | 3.600 | 1.480x | 99.9% | 0.126 | fused_ffn (27) | fused_ffn (2.474x; 27) |
| 140 | outer_04.transformer_2.block_14.attention_qkv_projection | wide_qkv | wide_qkv; wide_qkv_kernel; g9x37x1; b128x1x1; r136; s65536 | 60 | 19.328 | 23.233 | 1.202x | 80.7% | 0.409 | fused_ffn (28) | fused_ffn (1.546x; 28) |
| 141 | outer_04.transformer_2.block_14.attention_qk_rope | qk_rope | qk_rope; fusedQKRoPE19Half2Kernel; g361x13x1; b192x1x1; r16; s0 | 60 | 3.888 | 5.152 | 1.325x | 100.0% | 0.161 | linear2_residual (28) | wide_qkv (2.247x; 25) |
| 142 | outer_04.transformer_2.block_14.attention_fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 60 | 11.920 | 16.672 | 1.399x | 76.0% | 0.402 | linear2_residual (28) | linear2_residual (1.871x; 28) |
| 143 | outer_04.transformer_2.block_14.attention_out_projection_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 60 | 8.512 | 19.008 | 2.233x | 95.8% | 0.547 | library_gemm (28) | fused_ffn (2.541x; 7) |
| 144 | outer_04.transformer_2.block_14.ffn_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | 2.432 | 3.184 | 1.309x | 99.2% | 0.038 | fa4 (25) | linear2_residual (1.408x; 7) |
| 145 | outer_04.transformer_2.block_14.ffn_linear1_gate_swiglu | fused_ffn | fused_ffn; fused_ffn_candidate_a_reuse_kernel; g18x37x1; b128x1x1; r136; s32768 | 60 | 35.505 | 40.368 | 1.137x | 67.1% | 0.315 | fused_ffn (25) | linear2_residual (1.326x; 7) |
| 146 | outer_04.transformer_2.block_14.ffn_linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 60 | 20.976 | 32.736 | 1.561x | 97.3% | 0.661 | fused_ffn (50) | fused_ffn (1.564x; 50) |
| 147 | outer_04.post_norm_silu | affine_silu | affine_silu; affineSiluHalf2Kernel; g4693x1x1; b192x1x1; r17; s0 | 60 | 3.377 | 4.608 | 1.365x | 100.0% | 0.137 | fused_ffn (25) | fused_ffn (2.189x; 25) |
| 148 | outer_04.post_projection_c384_to_c768_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b256x1x1; r154; s73728 | 60 | 14.528 | 24.145 | 1.662x | 97.3% | 0.562 | linear2_residual (45) | linear2_residual (1.703x; 45) |
| 149 | outer_05.pre_norm_silu | affine_silu | affine_silu; affineSiluHalf2Kernel; g4693x1x1; b384x1x1; r17; s0 | 60 | 5.120 | 8.241 | 1.609x | 90.8% | 0.227 | linear2_residual (25) | wide_qkv (2.138x; 25) |
| 150 | outer_05.pre_projection_c768_to_c384 | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 60 | 12.032 | 15.280 | 1.270x | 95.2% | 0.230 | fa4 (25) | linear2_residual (1.439x; 25) |
| 151 | outer_05.transformer_0.block_15.attention_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | 2.464 | 4.624 | 1.877x | 100.0% | 0.130 | fa4 (25) | fused_ffn (2.084x; 10) |
| 152 | outer_05.transformer_0.block_15.attention_qkv_projection | wide_qkv | wide_qkv; wide_qkv_kernel; g9x37x1; b128x1x1; r136; s65536 | 60 | 19.360 | 22.928 | 1.184x | 69.9% | 0.264 | library_gemm (25) | fused_ffn (1.427x; 10) |
| 153 | outer_05.transformer_0.block_15.attention_qk_rope | qk_rope | qk_rope; fusedQKRoPE19Half2Kernel; g361x13x1; b192x1x1; r16; s0 | 60 | 3.872 | 4.768 | 1.231x | 100.0% | 0.063 | fa4 (25) | linear2_residual (1.364x; 10) |
| 154 | outer_05.transformer_0.block_15.attention_fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 60 | 11.952 | 16.497 | 1.380x | 66.3% | 0.298 | fused_ffn (25) | linear2_residual (1.685x; 10) |
| 155 | outer_05.transformer_0.block_15.attention_out_projection_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 60 | 8.512 | 20.160 | 2.368x | 96.5% | 0.722 | fused_ffn (46) | fused_ffn (2.479x; 46) |
| 156 | outer_05.transformer_0.block_15.ffn_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | 2.464 | 3.600 | 1.461x | 99.6% | 0.110 | fused_ffn (25) | fused_ffn (2.299x; 25) |
| 157 | outer_05.transformer_0.block_15.ffn_linear1_gate_swiglu | fused_ffn | fused_ffn; fused_ffn_candidate_a_reuse_kernel; g18x37x1; b128x1x1; r136; s32768 | 60 | 40.992 | 44.849 | 1.094x | 82.4% | 0.337 | linear2_residual (41) | linear2_residual (1.247x; 41) |
| 158 | outer_05.transformer_0.block_15.ffn_linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 60 | 20.928 | 31.904 | 1.524x | 97.0% | 0.695 | fa4 (28) | linear2_residual (1.703x; 5) |
| 159 | outer_05.transformer_1.block_16.attention_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | 2.432 | 2.976 | 1.224x | 100.0% | 0.047 | fa4 (28) | fused_ffn (2.237x; 7) |
| 160 | outer_05.transformer_1.block_16.attention_qkv_projection | wide_qkv | wide_qkv; wide_qkv_kernel; g9x37x1; b128x1x1; r136; s65536 | 60 | 19.345 | 22.528 | 1.165x | 67.3% | 0.220 | library_gemm (28) | fused_ffn (1.390x; 7) |
| 161 | outer_05.transformer_1.block_16.attention_qk_rope | qk_rope | qk_rope; fusedQKRoPE19Half2Kernel; g361x13x1; b192x1x1; r16; s0 | 60 | 3.856 | 4.720 | 1.224x | 68.7% | 0.054 | library_gemm (53) | linear2_residual (1.328x; 7) |
| 162 | outer_05.transformer_1.block_16.attention_fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 60 | 11.920 | 21.345 | 1.791x | 79.9% | 0.460 | fused_ffn (28) | linear2_residual (1.869x; 7) |
| 163 | outer_05.transformer_1.block_16.attention_out_projection_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 60 | 8.480 | 19.360 | 2.283x | 96.5% | 0.664 | fused_ffn (28) | fused_ffn (2.347x; 28) |
| 164 | outer_05.transformer_1.block_16.ffn_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | 2.432 | 3.360 | 1.382x | 99.1% | 0.055 | fa4 (28) | linear2_residual (1.421x; 28) |
| 165 | outer_05.transformer_1.block_16.ffn_linear1_gate_swiglu | fused_ffn | fused_ffn; fused_ffn_candidate_a_reuse_kernel; g18x37x1; b128x1x1; r136; s32768 | 60 | 40.481 | 47.553 | 1.175x | 65.0% | 0.391 | library_gemm (29) | linear2_residual (1.258x; 28) |
| 166 | outer_05.transformer_1.block_16.ffn_linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 60 | 20.816 | 32.193 | 1.547x | 97.0% | 0.653 | fused_ffn (28) | fused_ffn (1.596x; 28) |
| 167 | outer_05.transformer_2.block_17.attention_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | 2.400 | 3.536 | 1.473x | 100.0% | 0.100 | fused_ffn (28) | fused_ffn (2.087x; 28) |
| 168 | outer_05.transformer_2.block_17.attention_qkv_projection | wide_qkv | wide_qkv; wide_qkv_kernel; g9x37x1; b128x1x1; r136; s65536 | 60 | 19.233 | 23.169 | 1.205x | 80.5% | 0.324 | fused_ffn (28) | fused_ffn (1.398x; 28) |
| 169 | outer_05.transformer_2.block_17.attention_qk_rope | qk_rope | qk_rope; fusedQKRoPE19Half2Kernel; g361x13x1; b192x1x1; r16; s0 | 60 | 3.824 | 5.200 | 1.360x | 100.0% | 0.159 | linear2_residual (28) | wide_qkv (2.234x; 25) |
| 170 | outer_05.transformer_2.block_17.attention_fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 60 | 11.808 | 16.561 | 1.403x | 76.3% | 0.376 | linear2_residual (28) | linear2_residual (1.828x; 28) |
| 171 | outer_05.transformer_2.block_17.attention_out_projection_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 60 | 8.448 | 19.009 | 2.250x | 96.0% | 0.548 | library_gemm (28) | fused_ffn (2.580x; 7) |
| 172 | outer_05.transformer_2.block_17.ffn_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | 2.400 | 3.072 | 1.280x | 99.2% | 0.041 | fa4 (25) | linear2_residual (1.433x; 6) |
| 173 | outer_05.transformer_2.block_17.ffn_linear1_gate_swiglu | fused_ffn | fused_ffn; fused_ffn_candidate_a_reuse_kernel; g18x37x1; b128x1x1; r136; s32768 | 60 | 38.688 | 46.593 | 1.204x | 61.1% | 0.436 | fused_ffn (26) | linear2_residual (1.327x; 6) |
| 174 | outer_05.transformer_2.block_17.ffn_linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 60 | 20.720 | 32.513 | 1.569x | 97.4% | 0.684 | fused_ffn (50) | fused_ffn (1.575x; 50) |
| 175 | outer_05.post_norm_silu | affine_silu | affine_silu; affineSiluHalf2Kernel; g4693x1x1; b192x1x1; r17; s0 | 60 | 3.312 | 4.497 | 1.357x | 99.9% | 0.133 | fused_ffn (25) | fused_ffn (2.212x; 25) |
| 176 | outer_05.post_projection_c384_to_c768_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b256x1x1; r154; s73728 | 60 | 14.304 | 23.328 | 1.631x | 97.1% | 0.513 | fused_ffn (26) | fused_ffn (1.716x; 26) |
| 177 | outer_06.pre_norm_silu | affine_silu | affine_silu; affineSiluHalf2Kernel; g4693x1x1; b384x1x1; r17; s0 | 60 | 5.024 | 7.888 | 1.570x | 92.5% | 0.245 | linear2_residual (25) | wide_qkv (2.248x; 25) |
| 178 | outer_06.pre_projection_c768_to_c384 | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 60 | 11.872 | 15.344 | 1.292x | 95.4% | 0.276 | linear2_residual (26) | linear2_residual (1.590x; 26) |
| 179 | outer_06.transformer_0.block_18.attention_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | 2.432 | 4.608 | 1.895x | 100.0% | 0.128 | fa4 (25) | fused_ffn (2.026x; 7) |
| 180 | outer_06.transformer_0.block_18.attention_qkv_projection | wide_qkv | wide_qkv; wide_qkv_kernel; g9x37x1; b128x1x1; r136; s65536 | 60 | 19.201 | 22.256 | 1.159x | 69.1% | 0.237 | library_gemm (25) | fused_ffn (1.407x; 8) |
| 181 | outer_06.transformer_0.block_18.attention_qk_rope | qk_rope | qk_rope; fusedQKRoPE19Half2Kernel; g361x13x1; b192x1x1; r16; s0 | 60 | 3.840 | 4.672 | 1.217x | 100.0% | 0.069 | library_gemm (26) | linear2_residual (1.342x; 7) |
| 182 | outer_06.transformer_0.block_18.attention_fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 60 | 11.840 | 16.048 | 1.355x | 65.0% | 0.289 | fused_ffn (26) | linear2_residual (1.816x; 7) |
| 183 | outer_06.transformer_0.block_18.attention_out_projection_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 60 | 8.464 | 20.432 | 2.414x | 96.7% | 0.736 | fused_ffn (48) | fused_ffn (2.537x; 48) |
| 184 | outer_06.transformer_0.block_18.ffn_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | 2.496 | 3.520 | 1.410x | 99.5% | 0.088 | linear2_residual (26) | fused_ffn (1.923x; 25) |
| 185 | outer_06.transformer_0.block_18.ffn_linear1_gate_swiglu | fused_ffn | fused_ffn; fused_ffn_candidate_a_reuse_kernel; g18x37x1; b128x1x1; r136; s32768 | 60 | 40.432 | 48.032 | 1.188x | 82.9% | 0.453 | linear2_residual (50) | wide_qkv (1.315x; 4) |
| 186 | outer_06.transformer_0.block_18.ffn_linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 60 | 20.849 | 30.305 | 1.454x | 96.8% | 0.583 | fa4 (29) | library_gemm (1.527x; 25) |
| 187 | outer_06.transformer_1.block_19.attention_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | 2.464 | 2.736 | 1.110x | 100.0% | 0.035 | fa4 (29) | fused_ffn (2.442x; 5) |
| 188 | outer_06.transformer_1.block_19.attention_qkv_projection | wide_qkv | wide_qkv; wide_qkv_kernel; g9x37x1; b128x1x1; r136; s65536 | 60 | 19.328 | 23.648 | 1.224x | 68.8% | 0.280 | library_gemm (29) | fused_ffn (1.522x; 5) |
| 189 | outer_06.transformer_1.block_19.attention_qk_rope | qk_rope | qk_rope; fusedQKRoPE19Half2Kernel; g361x13x1; b192x1x1; r16; s0 | 60 | 3.872 | 4.672 | 1.207x | 76.8% | 0.060 | library_gemm (55) | linear2_residual (1.331x; 5) |
| 190 | outer_06.transformer_1.block_19.attention_fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 60 | 11.905 | 20.465 | 1.719x | 77.8% | 0.462 | fused_ffn (29) | library_gemm (1.848x; 26) |
| 191 | outer_06.transformer_1.block_19.attention_out_projection_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 60 | 8.496 | 19.409 | 2.284x | 96.6% | 0.710 | fused_ffn (29) | fused_ffn (2.595x; 29) |
| 192 | outer_06.transformer_1.block_19.ffn_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | 2.432 | 3.328 | 1.368x | 99.3% | 0.056 | linear2_residual (29) | linear2_residual (1.382x; 29) |
| 193 | outer_06.transformer_1.block_19.ffn_linear1_gate_swiglu | fused_ffn | fused_ffn; fused_ffn_candidate_a_reuse_kernel; g18x37x1; b128x1x1; r136; s32768 | 60 | 39.361 | 46.433 | 1.180x | 72.6% | 0.409 | linear2_residual (29) | linear2_residual (1.201x; 29) |
| 194 | outer_06.transformer_1.block_19.ffn_linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 60 | 20.928 | 29.409 | 1.405x | 96.8% | 0.549 | fused_ffn (28) | fused_ffn (1.547x; 28) |
| 195 | outer_06.transformer_2.block_20.attention_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | 2.400 | 3.456 | 1.440x | 99.8% | 0.100 | fused_ffn (28) | fused_ffn (2.100x; 28) |
| 196 | outer_06.transformer_2.block_20.attention_qkv_projection | wide_qkv | wide_qkv; wide_qkv_kernel; g9x37x1; b128x1x1; r136; s65536 | 60 | 19.312 | 24.736 | 1.281x | 82.8% | 0.397 | fused_ffn (29) | fused_ffn (1.471x; 29) |
| 197 | outer_06.transformer_2.block_20.attention_qk_rope | qk_rope | qk_rope; fusedQKRoPE19Half2Kernel; g361x13x1; b192x1x1; r16; s0 | 60 | 3.872 | 5.296 | 1.368x | 100.0% | 0.174 | linear2_residual (29) | wide_qkv (2.322x; 26) |
| 198 | outer_06.transformer_2.block_20.attention_fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 60 | 11.904 | 18.416 | 1.547x | 76.0% | 0.400 | linear2_residual (29) | linear2_residual (1.836x; 29) |
| 199 | outer_06.transformer_2.block_20.attention_out_projection_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 60 | 8.480 | 19.168 | 2.260x | 96.0% | 0.545 | library_gemm (29) | wide_qkv (2.343x; 26) |
| 200 | outer_06.transformer_2.block_20.ffn_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | 2.432 | 3.168 | 1.303x | 99.1% | 0.038 | fa4 (26) | linear2_residual (1.461x; 5) |
| 201 | outer_06.transformer_2.block_20.ffn_linear1_gate_swiglu | fused_ffn | fused_ffn; fused_ffn_candidate_a_reuse_kernel; g18x37x1; b128x1x1; r136; s32768 | 60 | 35.008 | 39.904 | 1.140x | 66.0% | 0.304 | library_gemm (27) | linear2_residual (1.424x; 5) |
| 202 | outer_06.transformer_2.block_20.ffn_linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 60 | 21.056 | 33.601 | 1.596x | 97.4% | 0.731 | fused_ffn (52) | fused_ffn (1.609x; 52) |
| 203 | outer_06.post_norm_silu | affine_silu | affine_silu; affineSiluHalf2Kernel; g4693x1x1; b192x1x1; r17; s0 | 60 | 3.361 | 4.560 | 1.357x | 100.0% | 0.147 | fused_ffn (26) | fused_ffn (2.262x; 26) |
| 204 | outer_06.post_projection_c384_to_c768_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b256x1x1; r154; s73728 | 60 | 14.496 | 27.536 | 1.899x | 97.4% | 0.732 | fused_ffn (27) | linear2_residual (1.939x; 26) |
| 205 | outer_07.pre_norm_silu | affine_silu | affine_silu; affineSiluHalf2Kernel; g4693x1x1; b384x1x1; r17; s0 | 60 | 5.088 | 7.664 | 1.506x | 93.9% | 0.211 | linear2_residual (26) | wide_qkv (1.937x; 26) |
| 206 | outer_07.pre_projection_c768_to_c384 | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 60 | 12.032 | 15.904 | 1.322x | 95.8% | 0.363 | linear2_residual (27) | linear2_residual (1.816x; 27) |
| 207 | outer_07.transformer_0.block_21.attention_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | 2.464 | 4.705 | 1.909x | 100.0% | 0.144 | fa4 (26) | wide_qkv (2.227x; 26) |
| 208 | outer_07.transformer_0.block_21.attention_qkv_projection | wide_qkv | wide_qkv; wide_qkv_kernel; g9x37x1; b128x1x1; r136; s65536 | 60 | 19.328 | 22.256 | 1.152x | 69.1% | 0.219 | library_gemm (26) | fused_ffn (1.416x; 7) |
| 209 | outer_07.transformer_0.block_21.attention_qk_rope | qk_rope | qk_rope; fusedQKRoPE19Half2Kernel; g361x13x1; b192x1x1; r16; s0 | 60 | 3.872 | 5.024 | 1.298x | 100.0% | 0.067 | library_gemm (27) | linear2_residual (1.380x; 7) |
| 210 | outer_07.transformer_0.block_21.attention_fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 60 | 11.968 | 16.160 | 1.350x | 64.8% | 0.292 | library_gemm (27) | linear2_residual (2.005x; 7) |
| 211 | outer_07.transformer_0.block_21.attention_out_projection_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 60 | 8.496 | 20.448 | 2.407x | 96.6% | 0.723 | fused_ffn (52) | fused_ffn (2.422x; 52) |
| 212 | outer_07.transformer_0.block_21.ffn_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | 2.432 | 3.568 | 1.467x | 99.6% | 0.090 | fused_ffn (26) | fused_ffn (1.888x; 26) |
| 213 | outer_07.transformer_0.block_21.ffn_linear1_gate_swiglu | fused_ffn | fused_ffn; fused_ffn_candidate_a_reuse_kernel; g18x37x1; b128x1x1; r136; s32768 | 60 | 35.616 | 43.553 | 1.223x | 91.8% | 0.622 | linear2_residual (52) | linear2_residual (1.286x; 52) |
| 214 | outer_07.transformer_0.block_21.ffn_linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 60 | 21.088 | 31.409 | 1.489x | 96.7% | 0.711 | fa4 (29) | library_gemm (1.731x; 26) |
| 215 | outer_07.transformer_1.block_22.attention_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | 2.400 | 2.944 | 1.227x | 100.0% | 0.049 | fa4 (29) | fused_ffn (2.947x; 5) |
| 216 | outer_07.transformer_1.block_22.attention_qkv_projection | wide_qkv | wide_qkv; wide_qkv_kernel; g9x37x1; b128x1x1; r136; s65536 | 60 | 19.488 | 23.297 | 1.195x | 66.8% | 0.257 | library_gemm (29) | fused_ffn (1.476x; 5) |
| 217 | outer_07.transformer_1.block_22.attention_qk_rope | qk_rope | qk_rope; fusedQKRoPE19Half2Kernel; g361x13x1; b192x1x1; r16; s0 | 60 | 3.888 | 4.608 | 1.185x | 66.4% | 0.047 | library_gemm (55) | linear2_residual (1.374x; 5) |
| 218 | outer_07.transformer_1.block_22.attention_fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 60 | 12.000 | 21.312 | 1.776x | 81.0% | 0.509 | fused_ffn (29) | linear2_residual (1.893x; 5) |
| 219 | outer_07.transformer_1.block_22.attention_out_projection_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 60 | 8.576 | 19.264 | 2.246x | 96.3% | 0.649 | fused_ffn (29) | fused_ffn (2.250x; 29) |
| 220 | outer_07.transformer_1.block_22.ffn_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | 2.465 | 3.328 | 1.350x | 99.4% | 0.053 | fa4 (29) | fa4 (1.350x; 29) |
| 221 | outer_07.transformer_1.block_22.ffn_linear1_gate_swiglu | fused_ffn | fused_ffn; fused_ffn_candidate_a_reuse_kernel; g18x37x1; b128x1x1; r136; s32768 | 60 | 35.824 | 47.360 | 1.322x | 77.7% | 0.476 | library_gemm (29) | linear2_residual (1.377x; 29) |
| 222 | outer_07.transformer_1.block_22.ffn_linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 60 | 20.992 | 30.369 | 1.447x | 96.9% | 0.627 | fused_ffn (29) | fused_ffn (1.605x; 29) |
| 223 | outer_07.transformer_2.block_23.attention_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | 2.465 | 3.664 | 1.487x | 100.0% | 0.117 | fused_ffn (29) | fused_ffn (2.220x; 29) |
| 224 | outer_07.transformer_2.block_23.attention_qkv_projection | wide_qkv | wide_qkv; wide_qkv_kernel; g9x37x1; b128x1x1; r136; s65536 | 60 | 19.424 | 26.496 | 1.364x | 83.3% | 0.476 | library_gemm (31) | fused_ffn (1.526x; 29) |
| 225 | outer_07.transformer_2.block_23.attention_qk_rope | qk_rope | qk_rope; fusedQKRoPE19Half2Kernel; g361x13x1; b192x1x1; r16; s0 | 60 | 3.872 | 5.280 | 1.364x | 100.0% | 0.184 | linear2_residual (29) | wide_qkv (2.413x; 26) |
| 226 | outer_07.transformer_2.block_23.attention_fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 60 | 11.985 | 19.473 | 1.625x | 77.1% | 0.433 | linear2_residual (29) | linear2_residual (1.930x; 29) |
| 227 | outer_07.transformer_2.block_23.attention_out_projection_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 60 | 8.544 | 19.313 | 2.260x | 96.0% | 0.538 | library_gemm (29) | fused_ffn (2.618x; 5) |
| 228 | outer_07.transformer_2.block_23.ffn_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | 2.432 | 3.168 | 1.303x | 99.4% | 0.037 | library_gemm (27) | linear2_residual (1.382x; 5) |
| 229 | outer_07.transformer_2.block_23.ffn_linear1_gate_swiglu | fused_ffn | fused_ffn; fused_ffn_candidate_a_reuse_kernel; g18x37x1; b128x1x1; r136; s32768 | 60 | 36.177 | 40.704 | 1.125x | 67.9% | 0.322 | library_gemm (27) | linear2_residual (1.341x; 5) |
| 230 | outer_07.transformer_2.block_23.ffn_linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 60 | 21.088 | 33.441 | 1.586x | 97.5% | 0.727 | fused_ffn (52) | fused_ffn (1.599x; 52) |
| 231 | outer_07.post_norm_silu | affine_silu | affine_silu; affineSiluHalf2Kernel; g4693x1x1; b192x1x1; r17; s0 | 60 | 3.376 | 4.608 | 1.365x | 100.0% | 0.149 | fused_ffn (26) | fused_ffn (2.251x; 26) |
| 232 | outer_07.post_projection_c384_to_c768_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b256x1x1; r154; s73728 | 60 | 14.528 | 25.216 | 1.736x | 97.3% | 0.601 | linear2_residual (29) | fused_ffn (1.751x; 23) |
| 233 | outer_08.pre_norm_silu | affine_silu | affine_silu; affineSiluHalf2Kernel; g4693x1x1; b384x1x1; r17; s0 | 60 | 5.088 | 7.504 | 1.475x | 93.7% | 0.234 | linear2_residual (26) | wide_qkv (2.233x; 26) |
| 234 | outer_08.pre_projection_c768_to_c384 | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 60 | 12.032 | 15.840 | 1.316x | 95.6% | 0.311 | fa4 (26) | linear2_residual (1.614x; 26) |
| 235 | outer_08.transformer_0.block_24.attention_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | 2.464 | 4.640 | 1.883x | 100.0% | 0.135 | fa4 (26) | fused_ffn (2.084x; 8) |
| 236 | outer_08.transformer_0.block_24.attention_qkv_projection | wide_qkv | wide_qkv; wide_qkv_kernel; g9x37x1; b128x1x1; r136; s65536 | 60 | 19.440 | 22.913 | 1.179x | 69.1% | 0.257 | library_gemm (26) | fused_ffn (1.427x; 8) |
| 237 | outer_08.transformer_0.block_24.attention_qk_rope | qk_rope | qk_rope; fusedQKRoPE19Half2Kernel; g361x13x1; b192x1x1; r16; s0 | 60 | 3.920 | 4.896 | 1.249x | 100.0% | 0.062 | fa4 (26) | linear2_residual (1.351x; 8) |
| 238 | outer_08.transformer_0.block_24.attention_fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 60 | 12.000 | 16.624 | 1.385x | 65.5% | 0.308 | fused_ffn (26) | linear2_residual (1.799x; 8) |
| 239 | outer_08.transformer_0.block_24.attention_out_projection_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 60 | 8.560 | 19.984 | 2.335x | 96.5% | 0.686 | fused_ffn (48) | wide_qkv (2.366x; 5) |
| 240 | outer_08.transformer_0.block_24.ffn_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | 2.464 | 3.504 | 1.422x | 99.6% | 0.105 | fused_ffn (26) | fused_ffn (2.188x; 26) |
| 241 | outer_08.transformer_0.block_24.ffn_linear1_gate_swiglu | fused_ffn | fused_ffn; fused_ffn_candidate_a_reuse_kernel; g18x37x1; b128x1x1; r136; s32768 | 60 | 40.336 | 46.529 | 1.154x | 87.9% | 0.498 | linear2_residual (52) | linear2_residual (1.198x; 52) |
| 242 | outer_08.transformer_0.block_24.ffn_linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 60 | 21.024 | 32.112 | 1.527x | 97.0% | 0.669 | fa4 (29) | library_gemm (1.629x; 26) |
| 243 | outer_08.transformer_1.block_25.attention_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | 2.432 | 2.784 | 1.145x | 100.0% | 0.035 | fa4 (29) | fused_ffn (2.118x; 5) |
| 244 | outer_08.transformer_1.block_25.attention_qkv_projection | wide_qkv | wide_qkv; wide_qkv_kernel; g9x37x1; b128x1x1; r136; s65536 | 60 | 19.456 | 23.041 | 1.184x | 69.1% | 0.252 | library_gemm (29) | fused_ffn (1.442x; 5) |
| 245 | outer_08.transformer_1.block_25.attention_qk_rope | qk_rope | qk_rope; fusedQKRoPE19Half2Kernel; g361x13x1; b192x1x1; r16; s0 | 60 | 3.872 | 4.640 | 1.198x | 70.1% | 0.054 | library_gemm (55) | linear2_residual (1.339x; 5) |
| 246 | outer_08.transformer_1.block_25.attention_fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 60 | 11.968 | 20.768 | 1.735x | 79.8% | 0.487 | fused_ffn (29) | library_gemm (1.866x; 26) |
| 247 | outer_08.transformer_1.block_25.attention_out_projection_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 60 | 8.496 | 19.537 | 2.299x | 96.5% | 0.675 | fused_ffn (29) | fused_ffn (2.331x; 29) |
| 248 | outer_08.transformer_1.block_25.ffn_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | 2.464 | 3.360 | 1.364x | 99.4% | 0.053 | fa4 (29) | linear2_residual (1.377x; 29) |
| 249 | outer_08.transformer_1.block_25.ffn_linear1_gate_swiglu | fused_ffn | fused_ffn; fused_ffn_candidate_a_reuse_kernel; g18x37x1; b128x1x1; r136; s32768 | 60 | 40.880 | 47.776 | 1.169x | 68.3% | 0.378 | library_gemm (29) | linear2_residual (1.268x; 29) |
| 250 | outer_08.transformer_1.block_25.ffn_linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 60 | 20.960 | 32.320 | 1.542x | 96.9% | 0.672 | fused_ffn (29) | fused_ffn (1.602x; 29) |
| 251 | outer_08.transformer_2.block_26.attention_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | 2.464 | 3.472 | 1.409x | 100.0% | 0.098 | fused_ffn (29) | fused_ffn (2.039x; 29) |
| 252 | outer_08.transformer_2.block_26.attention_qkv_projection | wide_qkv | wide_qkv; wide_qkv_kernel; g9x37x1; b128x1x1; r136; s65536 | 60 | 19.424 | 23.616 | 1.216x | 81.4% | 0.338 | fused_ffn (29) | fused_ffn (1.402x; 29) |
| 253 | outer_08.transformer_2.block_26.attention_qk_rope | qk_rope | qk_rope; fusedQKRoPE19Half2Kernel; g361x13x1; b192x1x1; r16; s0 | 60 | 3.840 | 5.344 | 1.392x | 100.0% | 0.166 | linear2_residual (29) | wide_qkv (2.234x; 26) |
| 254 | outer_08.transformer_2.block_26.attention_fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 60 | 11.920 | 16.529 | 1.387x | 76.9% | 0.383 | linear2_residual (29) | linear2_residual (1.801x; 29) |
| 255 | outer_08.transformer_2.block_26.attention_out_projection_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 60 | 8.512 | 18.912 | 2.222x | 95.9% | 0.535 | library_gemm (29) | fused_ffn (2.560x; 5) |
| 256 | outer_08.transformer_2.block_26.ffn_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | 2.464 | 3.056 | 1.240x | 99.2% | 0.033 | fa4 (26) | linear2_residual (1.429x; 5) |
| 257 | outer_08.transformer_2.block_26.ffn_linear1_gate_swiglu | fused_ffn | fused_ffn; fused_ffn_candidate_a_reuse_kernel; g18x37x1; b128x1x1; r136; s32768 | 60 | 40.224 | 46.448 | 1.155x | 63.2% | 0.300 | fused_ffn (26) | linear2_residual (1.217x; 5) |
| 258 | outer_08.transformer_2.block_26.ffn_linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 60 | 20.896 | 32.608 | 1.560x | 97.4% | 0.683 | fused_ffn (52) | fused_ffn (1.582x; 52) |
| 259 | outer_08.post_norm_silu | affine_silu | affine_silu; affineSiluHalf2Kernel; g4693x1x1; b192x1x1; r17; s0 | 60 | 3.360 | 4.416 | 1.314x | 100.0% | 0.133 | fused_ffn (26) | fused_ffn (2.176x; 26) |
| 260 | outer_08.post_projection_c384_to_c768_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b256x1x1; r154; s73728 | 60 | 14.400 | 23.920 | 1.661x | 97.4% | 0.584 | linear2_residual (49) | linear2_residual (1.735x; 49) |
| 261 | outer_09.pre_norm_silu | affine_silu | affine_silu; affineSiluHalf2Kernel; g4693x1x1; b384x1x1; r17; s0 | 60 | 5.024 | 8.512 | 1.694x | 91.5% | 0.224 | linear2_residual (26) | wide_qkv (2.067x; 26) |
| 262 | outer_09.pre_projection_c768_to_c384 | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 60 | 12.000 | 15.264 | 1.272x | 95.3% | 0.221 | fa4 (26) | linear2_residual (1.397x; 26) |
| 263 | outer_09.transformer_0.block_27.attention_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | 2.400 | 4.512 | 1.880x | 100.0% | 0.128 | fa4 (26) | fused_ffn (2.147x; 8) |
| 264 | outer_09.transformer_0.block_27.attention_qkv_projection | wide_qkv | wide_qkv; wide_qkv_kernel; g9x37x1; b128x1x1; r136; s65536 | 60 | 19.312 | 22.640 | 1.172x | 69.3% | 0.245 | library_gemm (26) | fused_ffn (1.446x; 8) |
| 265 | outer_09.transformer_0.block_27.attention_qk_rope | qk_rope | qk_rope; fusedQKRoPE19Half2Kernel; g361x13x1; b192x1x1; r16; s0 | 60 | 3.856 | 4.864 | 1.261x | 100.0% | 0.064 | fa4 (26) | linear2_residual (1.369x; 8) |
| 266 | outer_09.transformer_0.block_27.attention_fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 60 | 11.937 | 16.608 | 1.391x | 65.0% | 0.296 | fused_ffn (26) | linear2_residual (1.798x; 8) |
| 267 | outer_09.transformer_0.block_27.attention_out_projection_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 60 | 8.512 | 19.904 | 2.338x | 96.6% | 0.736 | fused_ffn (46) | fused_ffn (2.509x; 46) |
| 268 | outer_09.transformer_0.block_27.ffn_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | 2.464 | 3.520 | 1.429x | 99.6% | 0.121 | fused_ffn (26) | fused_ffn (2.455x; 26) |
| 269 | outer_09.transformer_0.block_27.ffn_linear1_gate_swiglu | fused_ffn | fused_ffn; fused_ffn_candidate_a_reuse_kernel; g18x37x1; b128x1x1; r136; s32768 | 60 | 39.328 | 43.824 | 1.114x | 80.8% | 0.396 | linear2_residual (39) | linear2_residual (1.270x; 39) |
| 270 | outer_09.transformer_0.block_27.ffn_linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 60 | 20.784 | 31.936 | 1.537x | 96.9% | 0.645 | fa4 (29) | library_gemm (1.604x; 24) |
| 271 | outer_09.transformer_1.block_28.attention_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | 2.432 | 2.800 | 1.151x | 100.0% | 0.033 | fa4 (29) | fused_ffn (2.013x; 5) |
| 272 | outer_09.transformer_1.block_28.attention_qkv_projection | wide_qkv | wide_qkv; wide_qkv_kernel; g9x37x1; b128x1x1; r136; s65536 | 60 | 19.200 | 22.064 | 1.149x | 64.8% | 0.194 | library_gemm (29) | fused_ffn (1.412x; 5) |
| 273 | outer_09.transformer_1.block_28.attention_qk_rope | qk_rope | qk_rope; fusedQKRoPE19Half2Kernel; g361x13x1; b192x1x1; r16; s0 | 60 | 3.840 | 4.544 | 1.183x | 67.1% | 0.044 | library_gemm (55) | linear2_residual (1.358x; 5) |
| 274 | outer_09.transformer_1.block_28.attention_fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 60 | 11.808 | 20.320 | 1.721x | 79.7% | 0.457 | fused_ffn (29) | library_gemm (1.824x; 26) |
| 275 | outer_09.transformer_1.block_28.attention_out_projection_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 60 | 8.432 | 19.264 | 2.285x | 96.9% | 0.704 | fused_ffn (29) | fused_ffn (2.554x; 29) |
| 276 | outer_09.transformer_1.block_28.ffn_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | 2.432 | 3.328 | 1.368x | 99.5% | 0.054 | fa4 (29) | linear2_residual (1.395x; 29) |
| 277 | outer_09.transformer_1.block_28.ffn_linear1_gate_swiglu | fused_ffn | fused_ffn; fused_ffn_candidate_a_reuse_kernel; g18x37x1; b128x1x1; r136; s32768 | 60 | 38.144 | 46.321 | 1.214x | 64.9% | 0.444 | library_gemm (29) | linear2_residual (1.276x; 29) |
| 278 | outer_09.transformer_1.block_28.ffn_linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 60 | 20.640 | 31.472 | 1.525x | 97.1% | 0.659 | fused_ffn (29) | fused_ffn (1.625x; 29) |
| 279 | outer_09.transformer_2.block_29.attention_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | 2.400 | 3.440 | 1.433x | 100.0% | 0.103 | fused_ffn (29) | fused_ffn (2.187x; 29) |
| 280 | outer_09.transformer_2.block_29.attention_qkv_projection | wide_qkv | wide_qkv; wide_qkv_kernel; g9x37x1; b128x1x1; r136; s65536 | 60 | 19.025 | 22.608 | 1.188x | 81.4% | 0.298 | fused_ffn (29) | fused_ffn (1.351x; 29) |
| 281 | outer_09.transformer_2.block_29.attention_qk_rope | qk_rope | qk_rope; fusedQKRoPE19Half2Kernel; g361x13x1; b192x1x1; r16; s0 | 60 | 3.808 | 5.184 | 1.361x | 100.0% | 0.168 | linear2_residual (29) | wide_qkv (2.294x; 26) |
| 282 | outer_09.transformer_2.block_29.attention_fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 60 | 11.760 | 16.288 | 1.385x | 76.5% | 0.394 | linear2_residual (29) | linear2_residual (1.856x; 29) |
| 283 | outer_09.transformer_2.block_29.attention_out_projection_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 60 | 8.464 | 18.704 | 2.210x | 96.0% | 0.522 | library_gemm (29) | fused_ffn (2.393x; 5) |
| 284 | outer_09.transformer_2.block_29.ffn_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | 2.464 | 3.136 | 1.273x | 99.2% | 0.033 | fa4 (26) | linear2_residual (1.364x; 5) |
| 285 | outer_09.transformer_2.block_29.ffn_linear1_gate_swiglu | fused_ffn | fused_ffn; fused_ffn_candidate_a_reuse_kernel; g18x37x1; b128x1x1; r136; s32768 | 60 | 39.456 | 45.793 | 1.161x | 59.6% | 0.333 | fused_ffn (26) | linear2_residual (1.255x; 5) |
| 286 | outer_09.transformer_2.block_29.ffn_linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 60 | 20.592 | 31.280 | 1.519x | 97.5% | 0.638 | fused_ffn (52) | fused_ffn (1.569x; 52) |
| 287 | outer_09.post_norm_silu | affine_silu | affine_silu; affineSiluHalf2Kernel; g4693x1x1; b192x1x1; r17; s0 | 60 | 3.328 | 4.224 | 1.269x | 100.0% | 0.142 | fused_ffn (26) | fused_ffn (2.356x; 26) |
| 288 | outer_09.post_projection_c384_to_c768_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b256x1x1; r154; s73728 | 60 | 14.208 | 22.880 | 1.610x | 97.3% | 0.478 | linear2_residual (27) | fused_ffn (1.669x; 25) |
| 289 | outer_10.pre_norm_silu | affine_silu | affine_silu; affineSiluHalf2Kernel; g4693x1x1; b384x1x1; r17; s0 | 60 | 5.024 | 7.232 | 1.439x | 92.1% | 0.224 | linear2_residual (26) | wide_qkv (2.172x; 26) |
| 290 | outer_10.pre_projection_c768_to_c384 | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 60 | 11.792 | 15.008 | 1.273x | 95.7% | 0.312 | fa4 (26) | linear2_residual (1.752x; 26) |
| 291 | outer_10.transformer_0.block_30.attention_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | 2.433 | 4.560 | 1.875x | 100.0% | 0.138 | fa4 (26) | fused_ffn (2.118x; 8) |
| 292 | outer_10.transformer_0.block_30.attention_qkv_projection | wide_qkv | wide_qkv; wide_qkv_kernel; g9x37x1; b128x1x1; r136; s65536 | 60 | 19.041 | 22.016 | 1.156x | 68.6% | 0.215 | library_gemm (26) | fused_ffn (1.417x; 8) |
| 293 | outer_10.transformer_0.block_30.attention_qk_rope | qk_rope | qk_rope; fusedQKRoPE19Half2Kernel; g361x13x1; b192x1x1; r16; s0 | 60 | 3.824 | 4.736 | 1.238x | 100.0% | 0.065 | fa4 (26) | linear2_residual (1.372x; 8) |
| 294 | outer_10.transformer_0.block_30.attention_fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 60 | 11.776 | 16.192 | 1.375x | 65.0% | 0.295 | fused_ffn (26) | linear2_residual (1.753x; 8) |
| 295 | outer_10.transformer_0.block_30.attention_out_projection_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 60 | 8.416 | 20.064 | 2.384x | 96.7% | 0.735 | fused_ffn (51) | fused_ffn (2.418x; 51) |
| 296 | outer_10.transformer_0.block_30.ffn_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | 2.432 | 3.536 | 1.454x | 99.4% | 0.103 | fused_ffn (26) | fused_ffn (2.099x; 26) |
| 297 | outer_10.transformer_0.block_30.ffn_linear1_gate_swiglu | fused_ffn | fused_ffn; fused_ffn_candidate_a_reuse_kernel; g18x37x1; b128x1x1; r136; s32768 | 60 | 35.872 | 42.865 | 1.195x | 92.3% | 0.473 | linear2_residual (52) | linear2_residual (1.236x; 52) |
| 298 | outer_10.transformer_0.block_30.ffn_linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 60 | 20.849 | 29.712 | 1.425x | 96.6% | 0.589 | library_gemm (29) | library_gemm (1.636x; 29) |
| 299 | outer_10.transformer_1.block_31.attention_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | 2.449 | 2.816 | 1.150x | 99.3% | 0.039 | fa4 (26) | fused_ffn (2.352x; 5) |
| 300 | outer_10.transformer_1.block_31.attention_qkv_projection | wide_qkv | wide_qkv; wide_qkv_kernel; g9x37x1; b128x1x1; r136; s65536 | 60 | 19.328 | 22.432 | 1.161x | 68.2% | 0.231 | library_gemm (29) | fused_ffn (1.513x; 5) |
| 301 | outer_10.transformer_1.block_31.attention_qk_rope | qk_rope | qk_rope; fusedQKRoPE19Half2Kernel; g361x13x1; b192x1x1; r16; s0 | 60 | 3.872 | 4.576 | 1.182x | 65.1% | 0.047 | library_gemm (55) | linear2_residual (1.322x; 5) |
| 302 | outer_10.transformer_1.block_31.attention_fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 60 | 11.936 | 19.152 | 1.605x | 70.3% | 0.408 | library_gemm (29) | linear2_residual (1.847x; 5) |
| 303 | outer_10.transformer_1.block_31.attention_out_projection_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 60 | 8.496 | 18.977 | 2.234x | 95.1% | 0.497 | library_gemm (28) | fused_ffn (2.540x; 26) |
| 304 | outer_10.transformer_1.block_31.ffn_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | 2.432 | 3.104 | 1.276x | 99.3% | 0.035 | head_elementwise (26) | linear2_residual (1.368x; 26) |
| 305 | outer_10.transformer_1.block_31.ffn_linear1_gate_swiglu | fused_ffn | fused_ffn; fused_ffn_candidate_a_reuse_kernel; g18x37x1; b128x1x1; r136; s32768 | 60 | 37.473 | 42.657 | 1.138x | 71.5% | 0.380 | library_gemm (33) | linear2_residual (1.245x; 26) |
| 306 | outer_10.transformer_1.block_31.ffn_linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 60 | 20.864 | 26.800 | 1.285x | 94.7% | 0.337 | library_gemm (53) | fused_ffn (1.477x; 6) |
| 307 | outer_10.transformer_2.block_32.attention_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | 2.432 | 3.168 | 1.303x | 97.5% | 0.060 | library_gemm (49) | fused_ffn (2.539x; 6) |
| 308 | outer_10.transformer_2.block_32.attention_qkv_projection | wide_qkv | wide_qkv; wide_qkv_kernel; g9x37x1; b128x1x1; r136; s65536 | 60 | 19.360 | 22.928 | 1.184x | 65.9% | 0.238 | head_elementwise (25) | fused_ffn (1.458x; 6) |
| 309 | outer_10.transformer_2.block_32.attention_qk_rope | qk_rope | qk_rope; fusedQKRoPE19Half2Kernel; g361x13x1; b192x1x1; r16; s0 | 60 | 3.841 | 4.608 | 1.200x | 73.9% | 0.140 | head_elementwise (26) | wide_qkv (2.400x; 23) |
| 310 | outer_10.transformer_2.block_32.attention_fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 60 | 11.936 | 14.768 | 1.237x | 60.1% | 0.171 | library_gemm (27) | linear2_residual (1.831x; 6) |
| 311 | outer_10.transformer_2.block_32.attention_out_projection_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 60 | 8.464 | 14.400 | 1.701x | 92.7% | 0.254 | library_gemm (50) | library_gemm (1.703x; 50) |
| 312 | outer_10.transformer_2.block_32.ffn_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | 2.464 | 2.688 | 1.091x | 96.0% | 0.014 | library_gemm (49) | library_gemm (1.091x; 49) |
| 313 | outer_10.transformer_2.block_32.ffn_linear1_gate_swiglu | fused_ffn | fused_ffn; fused_ffn_candidate_a_reuse_kernel; g18x37x1; b128x1x1; r136; s32768 | 60 | 40.736 | 41.649 | 1.022x | 54.7% | 0.213 | library_gemm (32) | fused_ffn (1.176x; 23) |
| 314 | outer_10.transformer_2.block_32.ffn_linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 60 | 20.928 | 27.968 | 1.336x | 85.5% | 0.534 | fused_ffn (26) | fused_ffn (1.669x; 26) |
| 315 | outer_10.post_norm_silu | affine_silu | affine_silu; affineSiluHalf2Kernel; g4693x1x1; b192x1x1; r17; s0 | 60 | 3.392 | 4.160 | 1.226x | 97.1% | 0.052 | cudnn (23) | fa4 (1.406x; 4) |
| 316 | outer_10.post_projection_c384_to_c768_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b256x1x1; r154; s73728 | 60 | 14.432 | 18.432 | 1.277x | 93.7% | 0.417 | linear2_residual (23) | linear2_residual (1.801x; 23) |
| 317 | trunk.tip_norm_silu | affine_silu | affine_silu; affineSiluHalf2Kernel; g4693x1x1; b384x1x1; r17; s0 | 60 | 5.056 | 6.928 | 1.370x | 76.0% | 0.181 | wide_qkv (23) | wide_qkv (2.114x; 23) |
| 318 | policy.p1_conv | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r80; s98304 | 60 | 6.192 | 7.601 | 1.227x | 87.6% | 0.139 | library_gemm (26) | fused_ffn (1.344x; 4) |
| 319 | policy.g1_conv | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r80; s98304 | 60 | 5.904 | 12.272 | 2.078x | 93.7% | 0.376 | fa4 (23) | fused_ffn (2.596x; 6) |
| 320 | policy.g1_norm_silu | head_elementwise | head_elementwise; applyCScaleBiasNHWCSiluHalfKernel; g1x73x13; b96x5x1; r16; s0 | 60 | 2.096 | 2.848 | 1.359x | 86.1% | 0.074 | wide_qkv (25) | wide_qkv (2.107x; 25) |
| 321 | policy.g1_half_to_float | copy_reformat | copy_reformat; copyFromHalfKernel; g880x1x1; b512x1x1; r16; s0 | 60 | 1.536 | 2.528 | 1.646x | 97.0% | 0.097 | wide_qkv (25) | wide_qkv (2.792x; 25) |
| 322 | policy.g1_global_pool | head_elementwise | head_elementwise; gPoolChannelsNHWCKernel; g2x1x13; b64x8x1; r22; s4096 | 60 | 4.480 | 8.608 | 1.921x | 90.2% | 0.191 | fused_ffn (22) | fused_ffn (2.079x; 22) |
| 323 | policy.gpool_to_bias_matmul | library_gemm | library_gemm; gemmSN_NN_kernel; g2x2x1; b256x1x1; r64; s21504 | 60 | 5.424 | 6.657 | 1.227x | 95.4% | 0.070 | fa4 (25) | fused_ffn (1.268x; 24) |
| 324 | policy.p1_half_to_float | copy_reformat | copy_reformat; copyFromHalfKernel; g880x1x1; b512x1x1; r16; s0 | 60 | 1.504 | 2.208 | 1.468x | 97.6% | 0.058 | fused_ffn (24) | fa4 (2.043x; 12) |
| 325 | policy.gpool_bias_add | head_elementwise | head_elementwise; addNCBiasInplaceNHWCKernel; g1x73x13; b96x5x1; r16; s0 | 60 | 1.792 | 2.560 | 1.429x | 96.1% | 0.055 | fused_ffn (24) | fused_ffn (1.607x; 24) |
| 326 | policy.p1_norm_silu | head_elementwise | head_elementwise; applyCScaleBiasNHWCSiluKernel; g1x73x13; b96x5x1; r16; s0 | 60 | 2.144 | 2.896 | 1.351x | 84.9% | 0.048 | library_gemm (25) | library_gemm (1.403x; 25) |
| 327 | policy.p2_conv | library_gemm | library_gemm; Kernel2; g74x1x1; b128x1x1; r90; s98304 | 60 | 3.904 | 5.984 | 1.533x | 92.0% | 0.138 | fused_ffn (34) | wide_qkv (2.098x; 5) |
| 328 | policy.gpool_to_pass_matmul | library_gemm | library_gemm; gemmSN_NN_kernel; g2x2x1; b256x1x1; r64; s21504 | 60 | 5.296 | 6.689 | 1.263x | 95.9% | 0.125 | fused_ffn (25) | fused_ffn (1.293x; 25) |
| 329 | policy.pass_bias_silu | head_elementwise | head_elementwise; addCBiasInplaceNCKernelSilu; g1x3x1; b96x5x1; r16; s0 | 60 | 1.024 | 1.280 | 1.250x | 98.2% | 0.019 | fused_ffn (25) | linear2_residual (1.281x; 25) |
| 330 | policy.gpool_to_pass_matmul2 | library_gemm | library_gemm; gemmSN_NN_kernel; g1x2x1; b256x1x1; r64; s21504 | 60 | 2.304 | 2.752 | 1.194x | 97.8% | 0.028 | fused_ffn (25) | fused_ffn (1.208x; 25) |
| 331 | value.v1_conv | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r118; s98304 | 60 | 7.936 | 17.216 | 2.169x | 94.7% | 0.473 | linear2_residual (33) | linear2_residual (2.363x; 33) |
| 332 | value.v1_norm_silu | head_elementwise | head_elementwise; applyCScaleBiasNHWCSiluHalfKernel; g1x181x13; b192x2x1; r16; s0 | 60 | 3.136 | 4.128 | 1.316x | 99.0% | 0.129 | wide_qkv (27) | wide_qkv (2.184x; 27) |
| 333 | value.v1_half_to_float | copy_reformat | copy_reformat; copyFromHalfKernel; g1760x1x1; b512x1x1; r16; s0 | 60 | 2.144 | 3.232 | 1.508x | 93.5% | 0.066 | wide_qkv (27) | wide_qkv (1.552x; 27) |
| 334 | value.v1_global_pool | head_elementwise | head_elementwise; valueHeadPoolChannelsNHWCKernel; g3x1x13; b64x8x1; r22; s2048 | 60 | 3.232 | 4.352 | 1.347x | 92.6% | 0.084 | fa4 (27) | fused_ffn (2.366x; 5) |
| 335 | value.v2_matmul | library_gemm | library_gemm; gemmSN_NN_kernel; g3x2x1; b256x1x1; r64; s21504 | 60 | 9.536 | 11.872 | 1.245x | 94.6% | 0.143 | library_gemm (27) | linear2_residual (1.295x; 26) |
| 336 | value.v2_bias_silu | head_elementwise | head_elementwise; addCBiasInplaceNCKernelSilu; g1x7x1; b192x2x1; r16; s0 | 60 | 1.024 | 1.344 | 1.312x | 98.2% | 0.051 | library_gemm (30) | wide_qkv (2.812x; 22) |
| 337 | value.v3_matmul | library_gemm | library_gemm; gemmSN_NN_kernel; g1x2x1; b256x1x1; r64; s21504 | 60 | 3.456 | 4.576 | 1.324x | 90.8% | 0.082 | rmsnorm (27) | wide_qkv (1.616x; 22) |
| 338 | value.v3_bias | head_elementwise | head_elementwise; addCBiasInplaceNCKernel; g1x1x1; b3x170x1; r16; s0 | 60 | 0.960 | 1.280 | 1.333x | 86.5% | 0.025 | fused_ffn (32) | qk_rope (1.900x; 5) |
| 339 | value.score_matmul | library_gemm | library_gemm; gemmSN_NN_kernel; g1x2x1; b256x1x1; r64; s21504 | 60 | 3.488 | 4.352 | 1.248x | 94.7% | 0.051 | fused_ffn (28) | fa4 (1.303x; 22) |
| 340 | value.score_bias | head_elementwise | head_elementwise; addCBiasInplaceNCKernel; g1x1x1; b6x85x1; r16; s0 | 60 | 0.928 | 1.168 | 1.259x | 98.6% | 0.014 | fused_ffn (28) | linear2_residual (1.397x; 6) |
| 341 | value.ownership_conv | library_gemm | library_gemm; Kernel2; g8x19x3; b128x1x1; r118; s33792 | 60 | 4.000 | 5.568 | 1.392x | 96.8% | 0.101 | fused_ffn (28) | linear2_residual (2.168x; 6) |
| 342 | value.ownership_conv_splitk_reduce | library_gemm | library_gemm; splitKreduce_kernel; g147x1x1; b32x16x1; r49; s0 | 60 | 1.376 | 1.952 | 1.419x | 96.5% | 0.053 | fused_ffn (27) | library_gemm (2.105x; 24) |
| 343 | value.ownership_half_to_float | copy_reformat | copy_reformat; copyFromHalfKernel; g10x1x1; b512x1x1; r16; s0 | 60 | 0.928 | 1.120 | 1.207x | 87.8% | 0.028 | library_gemm (25) | library_gemm (1.241x; 25) |
