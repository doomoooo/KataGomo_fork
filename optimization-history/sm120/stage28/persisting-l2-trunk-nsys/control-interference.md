# Nsys stream interference report

- Timed iterations: 30; streams: 65, 82
- Kernels per forward: 65=344, 82=344
- Iteration start offset stream 82 - 65: median -125.39 us, p10..p90 -127.26..-123.43 us, range -189.03..-99.81 us.

## Kernel families

| family | instances | total ms | median us | p90 us | peer overlap | S2/S1 median | summed excess ms | matched |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| fused_ffn | 1980 | 91.055 | 45.505 | 50.497 | 74.5% | n/a | 0.000 | 0 |
| library_gemm | 4140 | 70.721 | 19.232 | 25.312 | 95.7% | n/a | 0.000 | 0 |
| linear2_residual | 1980 | 63.400 | 32.224 | 35.072 | 97.6% | n/a | 0.000 | 0 |
| wide_qkv | 1980 | 48.589 | 23.328 | 30.112 | 72.6% | n/a | 0.000 | 0 |
| fa4 | 1980 | 32.475 | 15.361 | 21.280 | 57.1% | n/a | 0.000 | 0 |
| rmsnorm | 3960 | 15.125 | 3.392 | 5.600 | 99.7% | n/a | 0.000 | 0 |
| qk_rope | 1980 | 13.579 | 5.984 | 9.440 | 93.1% | n/a | 0.000 | 0 |
| affine_silu | 1380 | 9.522 | 7.088 | 9.664 | 97.9% | n/a | 0.000 | 0 |
| head_elementwise | 720 | 2.455 | 2.512 | 6.628 | 82.4% | n/a | 0.000 | 0 |
| cudnn | 180 | 1.575 | 2.144 | 23.092 | 45.7% | n/a | 0.000 | 0 |
| copy_reformat | 300 | 0.602 | 1.888 | 3.488 | 73.8% | n/a | 0.000 | 0 |
| sumChannelsNCHWKernel | 60 | 0.159 | 2.432 | 4.064 | 93.9% | n/a | 0.000 | 0 |

## Dominant interference pairs

| running family | dominant peer | instances | median overlap | median us | S2/S1 median | matched |
|---|---|---:|---:|---:|---:|---:|
| rmsnorm | library_gemm | 1350 | 100.0% | 2.784 | n/a | 0 |
| library_gemm | fused_ffn | 1240 | 96.8% | 21.376 | n/a | 0 |
| library_gemm | library_gemm | 1085 | 96.5% | 14.464 | n/a | 0 |
| fa4 | library_gemm | 960 | 51.3% | 15.168 | n/a | 0 |
| linear2_residual | fused_ffn | 960 | 98.3% | 32.128 | n/a | 0 |
| fused_ffn | linear2_residual | 951 | 87.1% | 45.025 | n/a | 0 |
| library_gemm | linear2_residual | 800 | 96.9% | 21.857 | n/a | 0 |
| fused_ffn | fused_ffn | 669 | 69.2% | 48.288 | n/a | 0 |
| linear2_residual | library_gemm | 661 | 97.4% | 31.232 | n/a | 0 |
| rmsnorm | wide_qkv | 660 | 100.0% | 4.673 | n/a | 0 |
| wide_qkv | qk_rope | 660 | 65.8% | 22.624 | n/a | 0 |
| qk_rope | fa4 | 660 | 100.0% | 5.856 | n/a | 0 |
| rmsnorm | fused_ffn | 660 | 100.0% | 5.632 | n/a | 0 |
| qk_rope | wide_qkv | 660 | 100.0% | 9.344 | n/a | 0 |
| affine_silu | linear2_residual | 658 | 100.0% | 4.752 | n/a | 0 |
| qk_rope | library_gemm | 655 | 100.0% | 5.248 | n/a | 0 |
| rmsnorm | linear2_residual | 630 | 100.0% | 3.328 | n/a | 0 |
| fa4 | qk_rope | 603 | 43.4% | 15.008 | n/a | 0 |
| rmsnorm | fa4 | 600 | 100.0% | 3.392 | n/a | 0 |
| wide_qkv | library_gemm | 584 | 65.8% | 22.848 | n/a | 0 |
| library_gemm | fa4 | 370 | 94.9% | 14.464 | n/a | 0 |
| library_gemm | wide_qkv | 369 | 96.0% | 19.424 | n/a | 0 |
| linear2_residual | wide_qkv | 330 | 97.5% | 33.921 | n/a | 0 |
| fused_ffn | library_gemm | 330 | 47.9% | 43.105 | n/a | 0 |
| wide_qkv | linear2_residual | 330 | 95.8% | 30.560 | n/a | 0 |
| affine_silu | wide_qkv | 330 | 100.0% | 9.600 | n/a | 0 |
| affine_silu | fused_ffn | 301 | 100.0% | 6.720 | n/a | 0 |
| fa4 | fused_ffn | 292 | 61.7% | 16.448 | n/a | 0 |
| wide_qkv | affine_silu | 270 | 70.3% | 24.192 | n/a | 0 |
| head_elementwise | library_gemm | 235 | 96.2% | 4.928 | n/a | 0 |
| head_elementwise | head_elementwise | 145 | 74.2% | 1.408 | n/a | 0 |
| wide_qkv | wide_qkv | 135 | 65.0% | 23.744 | n/a | 0 |
| copy_reformat | library_gemm | 131 | 100.0% | 1.408 | n/a | 0 |
| library_gemm | head_elementwise | 125 | 90.6% | 9.664 | n/a | 0 |
| fa4 | fa4 | 114 | 65.3% | 16.624 | n/a | 0 |
| cudnn | fused_ffn | 87 | 100.0% | 2.272 | n/a | 0 |
| cudnn | library_gemm | 87 | 83.7% | 1.728 | n/a | 0 |
| head_elementwise | fused_ffn | 61 | 100.0% | 2.752 | n/a | 0 |
| affine_silu | library_gemm | 60 | 100.0% | 4.625 | n/a | 0 |
| head_elementwise | linear2_residual | 59 | 100.0% | 1.440 | n/a | 0 |

## Logical operation groups

Isolated reference total is the isolated median for each ordinal multiplied by its S2 call count; it is a normalized reference, not a second trace total.

| logical group | families | positions | calls | isolated reference ms | S2 total ms | total ratio | excess ms |
|---|---|---:|---:|---:|---:|---:|---:|
| transformer.ffn_linear1_gate_swiglu | fused_ffn | 33 | 1980 | 0.000 | 91.055 | n/a | 0.000 |
| transformer.ffn_linear2_residual | linear2_residual | 33 | 1980 | 0.000 | 63.400 | n/a | 0.000 |
| transformer.attention_qkv_projection | wide_qkv | 33 | 1980 | 0.000 | 48.589 | n/a | 0.000 |
| transformer.attention_out_projection_residual | library_gemm | 33 | 1980 | 0.000 | 36.519 | n/a | 0.000 |
| transformer.attention_fa4 | fa4 | 33 | 1980 | 0.000 | 32.475 | n/a | 0.000 |
| outer.post_projection_c384_to_c768_residual | library_gemm | 11 | 660 | 0.000 | 17.387 | n/a | 0.000 |
| transformer.attention_qk_rope | qk_rope | 33 | 1980 | 0.000 | 13.579 | n/a | 0.000 |
| outer.pre_projection_c768_to_c384 | library_gemm | 11 | 660 | 0.000 | 11.499 | n/a | 0.000 |
| transformer.ffn_rmsnorm | rmsnorm | 33 | 1980 | 0.000 | 7.703 | n/a | 0.000 |
| transformer.attention_rmsnorm | rmsnorm | 33 | 1980 | 0.000 | 7.422 | n/a | 0.000 |
| outer.pre_norm_silu | affine_silu | 11 | 660 | 0.000 | 5.459 | n/a | 0.000 |
| outer.post_norm_silu | affine_silu | 11 | 660 | 0.000 | 3.624 | n/a | 0.000 |
| frontend.initial_conv | cudnn | 1 | 60 | 0.000 | 1.354 | n/a | 0.000 |
| value.v1_conv | library_gemm | 1 | 60 | 0.000 | 0.897 | n/a | 0.000 |
| value.v2_matmul | library_gemm | 1 | 60 | 0.000 | 0.874 | n/a | 0.000 |
| frontend.initial_global_broadcast_add | head_elementwise | 1 | 60 | 0.000 | 0.580 | n/a | 0.000 |
| trunk.tip_norm_silu | affine_silu | 1 | 60 | 0.000 | 0.439 | n/a | 0.000 |
| policy.p1_conv | library_gemm | 1 | 60 | 0.000 | 0.435 | n/a | 0.000 |
| policy.gpool_to_pass_matmul | library_gemm | 1 | 60 | 0.000 | 0.402 | n/a | 0.000 |
| policy.g1_conv | library_gemm | 1 | 60 | 0.000 | 0.401 | n/a | 0.000 |
| policy.p2_conv | library_gemm | 1 | 60 | 0.000 | 0.394 | n/a | 0.000 |
| policy.gpool_to_bias_matmul | library_gemm | 1 | 60 | 0.000 | 0.374 | n/a | 0.000 |
| policy.g1_global_pool | head_elementwise | 1 | 60 | 0.000 | 0.348 | n/a | 0.000 |
| value.v3_matmul | library_gemm | 1 | 60 | 0.000 | 0.312 | n/a | 0.000 |
| value.ownership_conv | library_gemm | 1 | 60 | 0.000 | 0.294 | n/a | 0.000 |
| value.v1_global_pool | head_elementwise | 1 | 60 | 0.000 | 0.288 | n/a | 0.000 |
| frontend.initial_global_matmul | library_gemm | 1 | 60 | 0.000 | 0.281 | n/a | 0.000 |
| value.v1_norm_silu | head_elementwise | 1 | 60 | 0.000 | 0.271 | n/a | 0.000 |
| value.score_matmul | library_gemm | 1 | 60 | 0.000 | 0.256 | n/a | 0.000 |
| policy.g1_norm_silu | head_elementwise | 1 | 60 | 0.000 | 0.187 | n/a | 0.000 |
| value.v1_half_to_float | copy_reformat | 1 | 60 | 0.000 | 0.182 | n/a | 0.000 |
| policy.gpool_bias_add | head_elementwise | 1 | 60 | 0.000 | 0.172 | n/a | 0.000 |
| policy.gpool_to_pass_matmul2 | library_gemm | 1 | 60 | 0.000 | 0.164 | n/a | 0.000 |
| input.mask_sum | sumChannelsNCHWKernel | 1 | 60 | 0.000 | 0.159 | n/a | 0.000 |
| policy.p1_norm_silu | head_elementwise | 1 | 60 | 0.000 | 0.156 | n/a | 0.000 |
| policy.p1_half_to_float | copy_reformat | 1 | 60 | 0.000 | 0.156 | n/a | 0.000 |
| value.v2_bias_silu | head_elementwise | 1 | 60 | 0.000 | 0.142 | n/a | 0.000 |
| value.ownership_conv_splitk_reduce | library_gemm | 1 | 60 | 0.000 | 0.130 | n/a | 0.000 |
| frontend.initial_conv_nhwc_padding_1 | cudnn | 1 | 60 | 0.000 | 0.121 | n/a | 0.000 |
| policy.g1_half_to_float | copy_reformat | 1 | 60 | 0.000 | 0.113 | n/a | 0.000 |
| frontend.initial_global_matmul_splitk_reduce | library_gemm | 1 | 60 | 0.000 | 0.102 | n/a | 0.000 |
| frontend.initial_conv_nhwc_padding_0 | cudnn | 1 | 60 | 0.000 | 0.100 | n/a | 0.000 |
| input.extract_mask | head_elementwise | 1 | 60 | 0.000 | 0.092 | n/a | 0.000 |
| input.mask_half_to_float | copy_reformat | 1 | 60 | 0.000 | 0.082 | n/a | 0.000 |
| policy.pass_bias_silu | head_elementwise | 1 | 60 | 0.000 | 0.077 | n/a | 0.000 |
| value.score_bias | head_elementwise | 1 | 60 | 0.000 | 0.071 | n/a | 0.000 |
| value.v3_bias | head_elementwise | 1 | 60 | 0.000 | 0.070 | n/a | 0.000 |
| value.ownership_half_to_float | copy_reformat | 1 | 60 | 0.000 | 0.069 | n/a | 0.000 |

## `library_gemm` logical breakdown

| logical group | positions | calls | isolated reference ms | S2 total ms | total ratio | excess ms |
|---|---:|---:|---:|---:|---:|---:|
| transformer.attention_out_projection_residual | 33 | 1980 | 0.000 | 36.519 | n/a | 0.000 |
| outer.post_projection_c384_to_c768_residual | 11 | 660 | 0.000 | 17.387 | n/a | 0.000 |
| outer.pre_projection_c768_to_c384 | 11 | 660 | 0.000 | 11.499 | n/a | 0.000 |
| value.v1_conv | 1 | 60 | 0.000 | 0.897 | n/a | 0.000 |
| value.v2_matmul | 1 | 60 | 0.000 | 0.874 | n/a | 0.000 |
| policy.p1_conv | 1 | 60 | 0.000 | 0.435 | n/a | 0.000 |
| policy.gpool_to_pass_matmul | 1 | 60 | 0.000 | 0.402 | n/a | 0.000 |
| policy.g1_conv | 1 | 60 | 0.000 | 0.401 | n/a | 0.000 |
| policy.p2_conv | 1 | 60 | 0.000 | 0.394 | n/a | 0.000 |
| policy.gpool_to_bias_matmul | 1 | 60 | 0.000 | 0.374 | n/a | 0.000 |
| value.v3_matmul | 1 | 60 | 0.000 | 0.312 | n/a | 0.000 |
| value.ownership_conv | 1 | 60 | 0.000 | 0.294 | n/a | 0.000 |
| frontend.initial_global_matmul | 1 | 60 | 0.000 | 0.281 | n/a | 0.000 |
| value.score_matmul | 1 | 60 | 0.000 | 0.256 | n/a | 0.000 |
| policy.gpool_to_pass_matmul2 | 1 | 60 | 0.000 | 0.164 | n/a | 0.000 |
| value.ownership_conv_splitk_reduce | 1 | 60 | 0.000 | 0.130 | n/a | 0.000 |
| frontend.initial_global_matmul_splitk_reduce | 1 | 60 | 0.000 | 0.102 | n/a | 0.000 |

## Top ordinal hotspots by summed excess

The worst peer is the highest median S2/S1 slowdown among peer families observed at least four times for that ordinal.

| rank | ordinal | logical position | family | calls | isolated us | S2 us | S2/S1 | excess ms | common peer | worst peer |
|---:|---:|---|---|---:|---:|---:|---:|---:|---|---|
| 1 | 241 | outer_08.transformer_0.block_24.ffn_linear1_gate_swiglu | fused_ffn | 60 | n/a | 49.745 | n/a | 0.000 | linear2_residual (60) | n/a |
| 2 | 157 | outer_05.transformer_0.block_15.ffn_linear1_gate_swiglu | fused_ffn | 60 | n/a | 49.072 | n/a | 0.000 | linear2_residual (60) | n/a |
| 3 | 249 | outer_08.transformer_1.block_25.ffn_linear1_gate_swiglu | fused_ffn | 60 | n/a | 47.825 | n/a | 0.000 | fused_ffn (30) | n/a |
| 4 | 73 | outer_02.transformer_0.block_06.ffn_linear1_gate_swiglu | fused_ffn | 60 | n/a | 48.337 | n/a | 0.000 | linear2_residual (58) | n/a |
| 5 | 269 | outer_09.transformer_0.block_27.ffn_linear1_gate_swiglu | fused_ffn | 60 | n/a | 47.745 | n/a | 0.000 | linear2_residual (60) | n/a |
| 6 | 185 | outer_06.transformer_0.block_18.ffn_linear1_gate_swiglu | fused_ffn | 60 | n/a | 47.728 | n/a | 0.000 | linear2_residual (60) | n/a |
| 7 | 193 | outer_06.transformer_1.block_19.ffn_linear1_gate_swiglu | fused_ffn | 60 | n/a | 47.489 | n/a | 0.000 | fused_ffn (30) | n/a |
| 8 | 81 | outer_02.transformer_1.block_07.ffn_linear1_gate_swiglu | fused_ffn | 60 | n/a | 47.729 | n/a | 0.000 | fused_ffn (31) | n/a |
| 9 | 313 | outer_10.transformer_2.block_32.ffn_linear1_gate_swiglu | fused_ffn | 60 | n/a | 46.961 | n/a | 0.000 | fused_ffn (30) | n/a |
| 10 | 89 | outer_02.transformer_2.block_08.ffn_linear1_gate_swiglu | fused_ffn | 60 | n/a | 47.825 | n/a | 0.000 | fused_ffn (30) | n/a |
| 11 | 257 | outer_08.transformer_2.block_26.ffn_linear1_gate_swiglu | fused_ffn | 60 | n/a | 46.752 | n/a | 0.000 | fused_ffn (30) | n/a |
| 12 | 165 | outer_05.transformer_1.block_16.ffn_linear1_gate_swiglu | fused_ffn | 60 | n/a | 46.816 | n/a | 0.000 | fused_ffn (30) | n/a |
| 13 | 53 | outer_01.transformer_1.block_04.ffn_linear1_gate_swiglu | fused_ffn | 60 | n/a | 46.689 | n/a | 0.000 | fused_ffn (30) | n/a |
| 14 | 61 | outer_01.transformer_2.block_05.ffn_linear1_gate_swiglu | fused_ffn | 60 | n/a | 46.880 | n/a | 0.000 | fused_ffn (30) | n/a |
| 15 | 101 | outer_03.transformer_0.block_09.ffn_linear1_gate_swiglu | fused_ffn | 60 | n/a | 45.648 | n/a | 0.000 | linear2_residual (59) | n/a |
| 16 | 173 | outer_05.transformer_2.block_17.ffn_linear1_gate_swiglu | fused_ffn | 60 | n/a | 46.544 | n/a | 0.000 | fused_ffn (30) | n/a |
| 17 | 213 | outer_07.transformer_0.block_21.ffn_linear1_gate_swiglu | fused_ffn | 60 | n/a | 45.904 | n/a | 0.000 | linear2_residual (60) | n/a |
| 18 | 45 | outer_01.transformer_0.block_03.ffn_linear1_gate_swiglu | fused_ffn | 60 | n/a | 46.640 | n/a | 0.000 | linear2_residual (60) | n/a |
| 19 | 33 | outer_00.transformer_2.block_02.ffn_linear1_gate_swiglu | fused_ffn | 60 | n/a | 46.544 | n/a | 0.000 | fused_ffn (30) | n/a |
| 20 | 137 | outer_04.transformer_1.block_13.ffn_linear1_gate_swiglu | fused_ffn | 60 | n/a | 45.985 | n/a | 0.000 | fused_ffn (30) | n/a |
| 21 | 305 | outer_10.transformer_1.block_31.ffn_linear1_gate_swiglu | fused_ffn | 60 | n/a | 45.841 | n/a | 0.000 | fused_ffn (30) | n/a |
| 22 | 285 | outer_09.transformer_2.block_29.ffn_linear1_gate_swiglu | fused_ffn | 60 | n/a | 46.209 | n/a | 0.000 | fused_ffn (30) | n/a |
| 23 | 277 | outer_09.transformer_1.block_28.ffn_linear1_gate_swiglu | fused_ffn | 60 | n/a | 45.584 | n/a | 0.000 | fused_ffn (32) | n/a |
| 24 | 25 | outer_00.transformer_1.block_01.ffn_linear1_gate_swiglu | fused_ffn | 60 | n/a | 45.296 | n/a | 0.000 | fused_ffn (31) | n/a |
| 25 | 129 | outer_04.transformer_0.block_12.ffn_linear1_gate_swiglu | fused_ffn | 60 | n/a | 45.537 | n/a | 0.000 | linear2_residual (58) | n/a |
| 26 | 117 | outer_03.transformer_2.block_11.ffn_linear1_gate_swiglu | fused_ffn | 60 | n/a | 45.841 | n/a | 0.000 | fused_ffn (30) | n/a |
| 27 | 109 | outer_03.transformer_1.block_10.ffn_linear1_gate_swiglu | fused_ffn | 60 | n/a | 45.072 | n/a | 0.000 | fused_ffn (30) | n/a |
| 28 | 221 | outer_07.transformer_1.block_22.ffn_linear1_gate_swiglu | fused_ffn | 60 | n/a | 43.904 | n/a | 0.000 | fused_ffn (30) | n/a |
| 29 | 297 | outer_10.transformer_0.block_30.ffn_linear1_gate_swiglu | fused_ffn | 60 | n/a | 43.728 | n/a | 0.000 | linear2_residual (60) | n/a |
| 30 | 229 | outer_07.transformer_2.block_23.ffn_linear1_gate_swiglu | fused_ffn | 60 | n/a | 41.968 | n/a | 0.000 | fused_ffn (30) | n/a |
| 31 | 17 | outer_00.transformer_0.block_00.ffn_linear1_gate_swiglu | fused_ffn | 60 | n/a | 42.065 | n/a | 0.000 | linear2_residual (30) | n/a |
| 32 | 145 | outer_04.transformer_2.block_14.ffn_linear1_gate_swiglu | fused_ffn | 60 | n/a | 41.393 | n/a | 0.000 | fused_ffn (30) | n/a |
| 33 | 201 | outer_06.transformer_2.block_20.ffn_linear1_gate_swiglu | fused_ffn | 60 | n/a | 40.577 | n/a | 0.000 | fused_ffn (30) | n/a |
| 34 | 222 | outer_07.transformer_1.block_22.ffn_linear2_residual | linear2_residual | 60 | n/a | 36.657 | n/a | 0.000 | fused_ffn (30) | n/a |
| 35 | 214 | outer_07.transformer_0.block_21.ffn_linear2_residual | linear2_residual | 60 | n/a | 34.432 | n/a | 0.000 | library_gemm (30) | n/a |
| 36 | 138 | outer_04.transformer_1.block_13.ffn_linear2_residual | linear2_residual | 60 | n/a | 34.209 | n/a | 0.000 | fused_ffn (30) | n/a |
| 37 | 306 | outer_10.transformer_1.block_31.ffn_linear2_residual | linear2_residual | 60 | n/a | 33.648 | n/a | 0.000 | fused_ffn (30) | n/a |
| 38 | 250 | outer_08.transformer_1.block_25.ffn_linear2_residual | linear2_residual | 60 | n/a | 33.344 | n/a | 0.000 | fused_ffn (30) | n/a |
| 39 | 230 | outer_07.transformer_2.block_23.ffn_linear2_residual | linear2_residual | 60 | n/a | 32.960 | n/a | 0.000 | fused_ffn (60) | n/a |
| 40 | 194 | outer_06.transformer_1.block_19.ffn_linear2_residual | linear2_residual | 60 | n/a | 33.568 | n/a | 0.000 | fused_ffn (30) | n/a |

## Full fixed-forward ordinal map

| ordinal | logical position | family | resource signature | calls | isolated us | S2 us | S2/S1 | overlap | excess ms | common peer | worst peer |
|---:|---|---|---|---:|---:|---:|---:|---:|---:|---|---|
| 0 | input.extract_mask | head_elementwise | head_elementwise; extractChannel0KernelNHWC; g10x1x1; b512x1x1; r16; s0 | 60 | n/a | 1.344 | n/a | 81.5% | 0.000 | head_elementwise (28) | n/a |
| 1 | input.mask_half_to_float | copy_reformat | copy_reformat; copyFromHalfKernel; g10x1x1; b512x1x1; r16; s0 | 60 | n/a | 1.136 | n/a | 87.1% | 0.000 | head_elementwise (20) | n/a |
| 2 | input.mask_sum | sumChannelsNCHWKernel | sumChannelsNCHWKernel; sumChannelsNCHWKernel; g1x1x13; b256x2x1; r22; s2048 | 60 | n/a | 2.432 | n/a | 93.9% | 0.000 | library_gemm (30) | n/a |
| 3 | frontend.initial_conv_nhwc_padding_0 | cudnn | cudnn; nhwcAddPaddingKernel; g340x1x1; b768x1x1; r34; s0 | 60 | n/a | 1.696 | n/a | 90.8% | 0.000 | fused_ffn (29) | n/a |
| 4 | frontend.initial_conv_nhwc_padding_1 | cudnn | cudnn; nhwcAddPaddingKernel; g340x1x1; b768x1x1; r34; s0 | 60 | n/a | 2.144 | n/a | 98.0% | 0.000 | library_gemm (30) | n/a |
| 5 | frontend.initial_conv | cudnn | cudnn; Kernel; g296x3x1; b128x1x1; r94; s81920 | 60 | n/a | 22.256 | n/a | 37.6% | 0.000 | fused_ffn (29) | n/a |
| 6 | frontend.initial_global_matmul | library_gemm | library_gemm; Kernel2; g8x1x3; b128x1x1; r128; s24576 | 60 | n/a | 3.616 | n/a | 95.1% | 0.000 | linear2_residual (29) | n/a |
| 7 | frontend.initial_global_matmul_splitk_reduce | library_gemm | library_gemm; splitKreduce_kernel; g24x1x1; b32x16x1; r49; s0 | 60 | n/a | 1.632 | n/a | 98.6% | 0.000 | library_gemm (29) | n/a |
| 8 | frontend.initial_global_broadcast_add | head_elementwise | head_elementwise; addNCBiasInplaceNHWCHalfKernel; g3x361x13; b256x1x1; r16; s0 | 60 | n/a | 9.360 | n/a | 66.8% | 0.000 | linear2_residual (29) | n/a |
| 9 | outer_00.pre_norm_silu | affine_silu | affine_silu; affineSiluHalf2Kernel; g4693x1x1; b384x1x1; r17; s0 | 60 | n/a | 5.840 | n/a | 59.8% | 0.000 | linear2_residual (28) | n/a |
| 10 | outer_00.pre_projection_c768_to_c384 | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 60 | n/a | 13.441 | n/a | 94.9% | 0.000 | wide_qkv (29) | n/a |
| 11 | outer_00.transformer_0.block_00.attention_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | n/a | 3.088 | n/a | 98.3% | 0.000 | wide_qkv (30) | n/a |
| 12 | outer_00.transformer_0.block_00.attention_qkv_projection | wide_qkv | wide_qkv; wide_qkv_kernel; g9x37x1; b128x1x1; r136; s65536 | 60 | n/a | 20.657 | n/a | 65.1% | 0.000 | qk_rope (30) | n/a |
| 13 | outer_00.transformer_0.block_00.attention_qk_rope | qk_rope | qk_rope; fusedQKRoPE19HalfKernel; g361x13x1; b192x1x1; r16; s0 | 60 | n/a | 4.497 | n/a | 43.3% | 0.000 | fa4 (30) | n/a |
| 14 | outer_00.transformer_0.block_00.attention_fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 60 | n/a | 14.001 | n/a | 63.4% | 0.000 | library_gemm (39) | n/a |
| 15 | outer_00.transformer_0.block_00.attention_out_projection_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 60 | n/a | 13.329 | n/a | 79.7% | 0.000 | library_gemm (37) | n/a |
| 16 | outer_00.transformer_0.block_00.ffn_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | n/a | 4.256 | n/a | 94.0% | 0.000 | fused_ffn (30) | n/a |
| 17 | outer_00.transformer_0.block_00.ffn_linear1_gate_swiglu | fused_ffn | fused_ffn; fused_ffn_kernel; g18x37x1; b128x1x1; r146; s32768 | 60 | n/a | 42.065 | n/a | 64.7% | 0.000 | linear2_residual (30) | n/a |
| 18 | outer_00.transformer_0.block_00.ffn_linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 60 | n/a | 28.000 | n/a | 95.2% | 0.000 | wide_qkv (30) | n/a |
| 19 | outer_00.transformer_1.block_01.attention_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | n/a | 3.504 | n/a | 92.8% | 0.000 | wide_qkv (30) | n/a |
| 20 | outer_00.transformer_1.block_01.attention_qkv_projection | wide_qkv | wide_qkv; wide_qkv_kernel; g9x37x1; b128x1x1; r136; s65536 | 60 | n/a | 22.896 | n/a | 71.4% | 0.000 | qk_rope (30) | n/a |
| 21 | outer_00.transformer_1.block_01.attention_qk_rope | qk_rope | qk_rope; fusedQKRoPE19HalfKernel; g361x13x1; b192x1x1; r16; s0 | 60 | n/a | 7.345 | n/a | 89.8% | 0.000 | fa4 (30) | n/a |
| 22 | outer_00.transformer_1.block_01.attention_fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 60 | n/a | 14.864 | n/a | 54.1% | 0.000 | fa4 (30) | n/a |
| 23 | outer_00.transformer_1.block_01.attention_out_projection_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 60 | n/a | 15.713 | n/a | 95.7% | 0.000 | library_gemm (32) | n/a |
| 24 | outer_00.transformer_1.block_01.ffn_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | n/a | 4.192 | n/a | 100.0% | 0.000 | fused_ffn (30) | n/a |
| 25 | outer_00.transformer_1.block_01.ffn_linear1_gate_swiglu | fused_ffn | fused_ffn; fused_ffn_kernel; g18x37x1; b128x1x1; r146; s32768 | 60 | n/a | 45.296 | n/a | 75.1% | 0.000 | fused_ffn (31) | n/a |
| 26 | outer_00.transformer_1.block_01.ffn_linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 60 | n/a | 30.928 | n/a | 98.0% | 0.000 | fused_ffn (30) | n/a |
| 27 | outer_00.transformer_2.block_02.attention_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | n/a | 2.992 | n/a | 100.0% | 0.000 | library_gemm (30) | n/a |
| 28 | outer_00.transformer_2.block_02.attention_qkv_projection | wide_qkv | wide_qkv; wide_qkv_kernel; g9x37x1; b128x1x1; r136; s65536 | 60 | n/a | 26.592 | n/a | 84.0% | 0.000 | linear2_residual (30) | n/a |
| 29 | outer_00.transformer_2.block_02.attention_qk_rope | qk_rope | qk_rope; fusedQKRoPE19HalfKernel; g361x13x1; b192x1x1; r16; s0 | 60 | n/a | 8.208 | n/a | 82.6% | 0.000 | library_gemm (30) | n/a |
| 30 | outer_00.transformer_2.block_02.attention_fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 60 | n/a | 17.568 | n/a | 69.6% | 0.000 | library_gemm (30) | n/a |
| 31 | outer_00.transformer_2.block_02.attention_out_projection_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 60 | n/a | 16.561 | n/a | 96.1% | 0.000 | library_gemm (30) | n/a |
| 32 | outer_00.transformer_2.block_02.ffn_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | n/a | 2.944 | n/a | 100.0% | 0.000 | fa4 (30) | n/a |
| 33 | outer_00.transformer_2.block_02.ffn_linear1_gate_swiglu | fused_ffn | fused_ffn; fused_ffn_kernel; g18x37x1; b128x1x1; r146; s32768 | 60 | n/a | 46.544 | n/a | 57.4% | 0.000 | fused_ffn (30) | n/a |
| 34 | outer_00.transformer_2.block_02.ffn_linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 60 | n/a | 30.640 | n/a | 97.5% | 0.000 | fused_ffn (60) | n/a |
| 35 | outer_00.post_norm_silu | affine_silu | affine_silu; affineSiluHalf2Kernel; g4693x1x1; b192x1x1; r17; s0 | 60 | n/a | 5.184 | n/a | 100.0% | 0.000 | fused_ffn (30) | n/a |
| 36 | outer_00.post_projection_c384_to_c768_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b256x1x1; r154; s73728 | 60 | n/a | 25.776 | n/a | 97.6% | 0.000 | fused_ffn (30) | n/a |
| 37 | outer_01.pre_norm_silu | affine_silu | affine_silu; affineSiluHalf2Kernel; g4693x1x1; b384x1x1; r17; s0 | 60 | n/a | 8.048 | n/a | 98.9% | 0.000 | linear2_residual (30) | n/a |
| 38 | outer_01.pre_projection_c768_to_c384 | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 60 | n/a | 17.312 | n/a | 95.3% | 0.000 | linear2_residual (30) | n/a |
| 39 | outer_01.transformer_0.block_03.attention_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | n/a | 4.544 | n/a | 100.0% | 0.000 | fa4 (30) | n/a |
| 40 | outer_01.transformer_0.block_03.attention_qkv_projection | wide_qkv | wide_qkv; wide_qkv_kernel; g9x37x1; b128x1x1; r136; s65536 | 60 | n/a | 22.064 | n/a | 64.4% | 0.000 | library_gemm (30) | n/a |
| 41 | outer_01.transformer_0.block_03.attention_qk_rope | qk_rope | qk_rope; fusedQKRoPE19HalfKernel; g361x13x1; b192x1x1; r16; s0 | 60 | n/a | 5.361 | n/a | 100.0% | 0.000 | fa4 (30) | n/a |
| 42 | outer_01.transformer_0.block_03.attention_fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 60 | n/a | 15.280 | n/a | 55.5% | 0.000 | library_gemm (30) | n/a |
| 43 | outer_01.transformer_0.block_03.attention_out_projection_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 60 | n/a | 22.160 | n/a | 97.1% | 0.000 | fused_ffn (60) | n/a |
| 44 | outer_01.transformer_0.block_03.ffn_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | n/a | 4.624 | n/a | 100.0% | 0.000 | fused_ffn (30) | n/a |
| 45 | outer_01.transformer_0.block_03.ffn_linear1_gate_swiglu | fused_ffn | fused_ffn; fused_ffn_kernel; g18x37x1; b128x1x1; r146; s32768 | 60 | n/a | 46.640 | n/a | 86.8% | 0.000 | linear2_residual (60) | n/a |
| 46 | outer_01.transformer_0.block_03.ffn_linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 60 | n/a | 29.793 | n/a | 97.3% | 0.000 | library_gemm (30) | n/a |
| 47 | outer_01.transformer_1.block_04.attention_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | n/a | 4.000 | n/a | 100.0% | 0.000 | library_gemm (30) | n/a |
| 48 | outer_01.transformer_1.block_04.attention_qkv_projection | wide_qkv | wide_qkv; wide_qkv_kernel; g9x37x1; b128x1x1; r136; s65536 | 60 | n/a | 23.328 | n/a | 66.9% | 0.000 | library_gemm (30) | n/a |
| 49 | outer_01.transformer_1.block_04.attention_qk_rope | qk_rope | qk_rope; fusedQKRoPE19HalfKernel; g361x13x1; b192x1x1; r16; s0 | 60 | n/a | 7.552 | n/a | 97.5% | 0.000 | fa4 (30) | n/a |
| 50 | outer_01.transformer_1.block_04.attention_fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 60 | n/a | 14.928 | n/a | 47.7% | 0.000 | qk_rope (30) | n/a |
| 51 | outer_01.transformer_1.block_04.attention_out_projection_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 60 | n/a | 16.064 | n/a | 95.9% | 0.000 | library_gemm (32) | n/a |
| 52 | outer_01.transformer_1.block_04.ffn_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | n/a | 4.176 | n/a | 100.0% | 0.000 | fused_ffn (30) | n/a |
| 53 | outer_01.transformer_1.block_04.ffn_linear1_gate_swiglu | fused_ffn | fused_ffn; fused_ffn_kernel; g18x37x1; b128x1x1; r146; s32768 | 60 | n/a | 46.689 | n/a | 76.1% | 0.000 | fused_ffn (30) | n/a |
| 54 | outer_01.transformer_1.block_04.ffn_linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 60 | n/a | 32.513 | n/a | 98.1% | 0.000 | fused_ffn (30) | n/a |
| 55 | outer_01.transformer_2.block_05.attention_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | n/a | 3.088 | n/a | 100.0% | 0.000 | library_gemm (30) | n/a |
| 56 | outer_01.transformer_2.block_05.attention_qkv_projection | wide_qkv | wide_qkv; wide_qkv_kernel; g9x37x1; b128x1x1; r136; s65536 | 60 | n/a | 26.576 | n/a | 84.1% | 0.000 | linear2_residual (30) | n/a |
| 57 | outer_01.transformer_2.block_05.attention_qk_rope | qk_rope | qk_rope; fusedQKRoPE19HalfKernel; g361x13x1; b192x1x1; r16; s0 | 60 | n/a | 8.273 | n/a | 86.6% | 0.000 | library_gemm (30) | n/a |
| 58 | outer_01.transformer_2.block_05.attention_fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 60 | n/a | 17.520 | n/a | 66.2% | 0.000 | library_gemm (30) | n/a |
| 59 | outer_01.transformer_2.block_05.attention_out_projection_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 60 | n/a | 16.752 | n/a | 96.1% | 0.000 | library_gemm (30) | n/a |
| 60 | outer_01.transformer_2.block_05.ffn_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | n/a | 3.040 | n/a | 100.0% | 0.000 | fa4 (30) | n/a |
| 61 | outer_01.transformer_2.block_05.ffn_linear1_gate_swiglu | fused_ffn | fused_ffn; fused_ffn_kernel; g18x37x1; b128x1x1; r146; s32768 | 60 | n/a | 46.880 | n/a | 57.1% | 0.000 | fused_ffn (30) | n/a |
| 62 | outer_01.transformer_2.block_05.ffn_linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 60 | n/a | 31.761 | n/a | 97.6% | 0.000 | fused_ffn (60) | n/a |
| 63 | outer_01.post_norm_silu | affine_silu | affine_silu; affineSiluHalf2Kernel; g4693x1x1; b192x1x1; r17; s0 | 60 | n/a | 5.296 | n/a | 100.0% | 0.000 | fused_ffn (30) | n/a |
| 64 | outer_01.post_projection_c384_to_c768_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b256x1x1; r154; s73728 | 60 | n/a | 26.720 | n/a | 97.6% | 0.000 | fused_ffn (30) | n/a |
| 65 | outer_02.pre_norm_silu | affine_silu | affine_silu; affineSiluHalf2Kernel; g4693x1x1; b384x1x1; r17; s0 | 60 | n/a | 8.176 | n/a | 99.1% | 0.000 | linear2_residual (30) | n/a |
| 66 | outer_02.pre_projection_c768_to_c384 | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 60 | n/a | 17.440 | n/a | 95.2% | 0.000 | linear2_residual (29) | n/a |
| 67 | outer_02.transformer_0.block_06.attention_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | n/a | 4.624 | n/a | 100.0% | 0.000 | fa4 (30) | n/a |
| 68 | outer_02.transformer_0.block_06.attention_qkv_projection | wide_qkv | wide_qkv; wide_qkv_kernel; g9x37x1; b128x1x1; r136; s65536 | 60 | n/a | 22.256 | n/a | 64.3% | 0.000 | library_gemm (30) | n/a |
| 69 | outer_02.transformer_0.block_06.attention_qk_rope | qk_rope | qk_rope; fusedQKRoPE19HalfKernel; g361x13x1; b192x1x1; r16; s0 | 60 | n/a | 5.344 | n/a | 100.0% | 0.000 | fa4 (30) | n/a |
| 70 | outer_02.transformer_0.block_06.attention_fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 60 | n/a | 15.393 | n/a | 54.9% | 0.000 | library_gemm (30) | n/a |
| 71 | outer_02.transformer_0.block_06.attention_out_projection_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 60 | n/a | 22.304 | n/a | 97.0% | 0.000 | fused_ffn (60) | n/a |
| 72 | outer_02.transformer_0.block_06.ffn_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | n/a | 4.624 | n/a | 100.0% | 0.000 | fused_ffn (30) | n/a |
| 73 | outer_02.transformer_0.block_06.ffn_linear1_gate_swiglu | fused_ffn | fused_ffn; fused_ffn_kernel; g18x37x1; b128x1x1; r146; s32768 | 60 | n/a | 48.337 | n/a | 85.8% | 0.000 | linear2_residual (58) | n/a |
| 74 | outer_02.transformer_0.block_06.ffn_linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 60 | n/a | 29.904 | n/a | 97.3% | 0.000 | library_gemm (30) | n/a |
| 75 | outer_02.transformer_1.block_07.attention_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | n/a | 3.872 | n/a | 100.0% | 0.000 | library_gemm (30) | n/a |
| 76 | outer_02.transformer_1.block_07.attention_qkv_projection | wide_qkv | wide_qkv; wide_qkv_kernel; g9x37x1; b128x1x1; r136; s65536 | 60 | n/a | 23.216 | n/a | 67.0% | 0.000 | qk_rope (30) | n/a |
| 77 | outer_02.transformer_1.block_07.attention_qk_rope | qk_rope | qk_rope; fusedQKRoPE19HalfKernel; g361x13x1; b192x1x1; r16; s0 | 60 | n/a | 7.329 | n/a | 99.2% | 0.000 | fa4 (30) | n/a |
| 78 | outer_02.transformer_1.block_07.attention_fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 60 | n/a | 14.880 | n/a | 47.0% | 0.000 | qk_rope (30) | n/a |
| 79 | outer_02.transformer_1.block_07.attention_out_projection_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 60 | n/a | 16.160 | n/a | 95.7% | 0.000 | library_gemm (31) | n/a |
| 80 | outer_02.transformer_1.block_07.ffn_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | n/a | 4.160 | n/a | 100.0% | 0.000 | fused_ffn (30) | n/a |
| 81 | outer_02.transformer_1.block_07.ffn_linear1_gate_swiglu | fused_ffn | fused_ffn; fused_ffn_kernel; g18x37x1; b128x1x1; r146; s32768 | 60 | n/a | 47.729 | n/a | 76.7% | 0.000 | fused_ffn (31) | n/a |
| 82 | outer_02.transformer_1.block_07.ffn_linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 60 | n/a | 32.913 | n/a | 98.1% | 0.000 | fused_ffn (30) | n/a |
| 83 | outer_02.transformer_2.block_08.attention_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | n/a | 3.120 | n/a | 100.0% | 0.000 | library_gemm (30) | n/a |
| 84 | outer_02.transformer_2.block_08.attention_qkv_projection | wide_qkv | wide_qkv; wide_qkv_kernel; g9x37x1; b128x1x1; r136; s65536 | 60 | n/a | 27.265 | n/a | 84.9% | 0.000 | linear2_residual (30) | n/a |
| 85 | outer_02.transformer_2.block_08.attention_qk_rope | qk_rope | qk_rope; fusedQKRoPE19HalfKernel; g361x13x1; b192x1x1; r16; s0 | 60 | n/a | 8.080 | n/a | 81.8% | 0.000 | library_gemm (30) | n/a |
| 86 | outer_02.transformer_2.block_08.attention_fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 60 | n/a | 17.088 | n/a | 68.0% | 0.000 | library_gemm (30) | n/a |
| 87 | outer_02.transformer_2.block_08.attention_out_projection_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 60 | n/a | 16.896 | n/a | 96.2% | 0.000 | library_gemm (30) | n/a |
| 88 | outer_02.transformer_2.block_08.ffn_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | n/a | 3.056 | n/a | 100.0% | 0.000 | fa4 (30) | n/a |
| 89 | outer_02.transformer_2.block_08.ffn_linear1_gate_swiglu | fused_ffn | fused_ffn; fused_ffn_kernel; g18x37x1; b128x1x1; r146; s32768 | 60 | n/a | 47.825 | n/a | 58.6% | 0.000 | fused_ffn (30) | n/a |
| 90 | outer_02.transformer_2.block_08.ffn_linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 60 | n/a | 31.744 | n/a | 97.5% | 0.000 | fused_ffn (60) | n/a |
| 91 | outer_02.post_norm_silu | affine_silu | affine_silu; affineSiluHalf2Kernel; g4693x1x1; b192x1x1; r17; s0 | 60 | n/a | 5.088 | n/a | 100.0% | 0.000 | fused_ffn (30) | n/a |
| 92 | outer_02.post_projection_c384_to_c768_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b256x1x1; r154; s73728 | 60 | n/a | 26.336 | n/a | 97.6% | 0.000 | linear2_residual (35) | n/a |
| 93 | outer_03.pre_norm_silu | affine_silu | affine_silu; affineSiluHalf2Kernel; g4693x1x1; b384x1x1; r17; s0 | 60 | n/a | 8.672 | n/a | 98.7% | 0.000 | linear2_residual (30) | n/a |
| 94 | outer_03.pre_projection_c768_to_c384 | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 60 | n/a | 15.441 | n/a | 95.0% | 0.000 | linear2_residual (30) | n/a |
| 95 | outer_03.transformer_0.block_09.attention_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | n/a | 4.496 | n/a | 100.0% | 0.000 | fa4 (30) | n/a |
| 96 | outer_03.transformer_0.block_09.attention_qkv_projection | wide_qkv | wide_qkv; wide_qkv_kernel; g9x37x1; b128x1x1; r136; s65536 | 60 | n/a | 22.336 | n/a | 63.9% | 0.000 | library_gemm (30) | n/a |
| 97 | outer_03.transformer_0.block_09.attention_qk_rope | qk_rope | qk_rope; fusedQKRoPE19HalfKernel; g361x13x1; b192x1x1; r16; s0 | 60 | n/a | 5.344 | n/a | 98.9% | 0.000 | fa4 (30) | n/a |
| 98 | outer_03.transformer_0.block_09.attention_fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 60 | n/a | 15.553 | n/a | 55.3% | 0.000 | fused_ffn (30) | n/a |
| 99 | outer_03.transformer_0.block_09.attention_out_projection_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 60 | n/a | 22.384 | n/a | 97.1% | 0.000 | fused_ffn (60) | n/a |
| 100 | outer_03.transformer_0.block_09.ffn_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | n/a | 4.592 | n/a | 100.0% | 0.000 | fused_ffn (30) | n/a |
| 101 | outer_03.transformer_0.block_09.ffn_linear1_gate_swiglu | fused_ffn | fused_ffn; fused_ffn_kernel; g18x37x1; b128x1x1; r146; s32768 | 60 | n/a | 45.648 | n/a | 82.1% | 0.000 | linear2_residual (59) | n/a |
| 102 | outer_03.transformer_0.block_09.ffn_linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 60 | n/a | 31.664 | n/a | 97.3% | 0.000 | library_gemm (30) | n/a |
| 103 | outer_03.transformer_1.block_10.attention_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | n/a | 3.824 | n/a | 100.0% | 0.000 | library_gemm (30) | n/a |
| 104 | outer_03.transformer_1.block_10.attention_qkv_projection | wide_qkv | wide_qkv; wide_qkv_kernel; g9x37x1; b128x1x1; r136; s65536 | 60 | n/a | 22.128 | n/a | 65.8% | 0.000 | qk_rope (30) | n/a |
| 105 | outer_03.transformer_1.block_10.attention_qk_rope | qk_rope | qk_rope; fusedQKRoPE19HalfKernel; g361x13x1; b192x1x1; r16; s0 | 60 | n/a | 7.393 | n/a | 95.1% | 0.000 | fa4 (30) | n/a |
| 106 | outer_03.transformer_1.block_10.attention_fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 60 | n/a | 14.592 | n/a | 47.5% | 0.000 | qk_rope (29) | n/a |
| 107 | outer_03.transformer_1.block_10.attention_out_projection_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 60 | n/a | 16.192 | n/a | 95.7% | 0.000 | library_gemm (31) | n/a |
| 108 | outer_03.transformer_1.block_10.ffn_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | n/a | 4.256 | n/a | 100.0% | 0.000 | fused_ffn (30) | n/a |
| 109 | outer_03.transformer_1.block_10.ffn_linear1_gate_swiglu | fused_ffn | fused_ffn; fused_ffn_kernel; g18x37x1; b128x1x1; r146; s32768 | 60 | n/a | 45.072 | n/a | 76.3% | 0.000 | fused_ffn (30) | n/a |
| 110 | outer_03.transformer_1.block_10.ffn_linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 60 | n/a | 30.576 | n/a | 98.0% | 0.000 | fused_ffn (30) | n/a |
| 111 | outer_03.transformer_2.block_11.attention_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | n/a | 3.024 | n/a | 100.0% | 0.000 | library_gemm (30) | n/a |
| 112 | outer_03.transformer_2.block_11.attention_qkv_projection | wide_qkv | wide_qkv; wide_qkv_kernel; g9x37x1; b128x1x1; r136; s65536 | 60 | n/a | 25.633 | n/a | 85.0% | 0.000 | affine_silu (30) | n/a |
| 113 | outer_03.transformer_2.block_11.attention_qk_rope | qk_rope | qk_rope; fusedQKRoPE19HalfKernel; g361x13x1; b192x1x1; r16; s0 | 60 | n/a | 7.872 | n/a | 75.3% | 0.000 | library_gemm (30) | n/a |
| 114 | outer_03.transformer_2.block_11.attention_fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 60 | n/a | 17.824 | n/a | 71.0% | 0.000 | library_gemm (30) | n/a |
| 115 | outer_03.transformer_2.block_11.attention_out_projection_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 60 | n/a | 16.528 | n/a | 96.1% | 0.000 | library_gemm (30) | n/a |
| 116 | outer_03.transformer_2.block_11.ffn_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | n/a | 3.008 | n/a | 100.0% | 0.000 | fa4 (30) | n/a |
| 117 | outer_03.transformer_2.block_11.ffn_linear1_gate_swiglu | fused_ffn | fused_ffn; fused_ffn_kernel; g18x37x1; b128x1x1; r146; s32768 | 60 | n/a | 45.841 | n/a | 57.4% | 0.000 | fused_ffn (30) | n/a |
| 118 | outer_03.transformer_2.block_11.ffn_linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 60 | n/a | 30.464 | n/a | 97.5% | 0.000 | fused_ffn (60) | n/a |
| 119 | outer_03.post_norm_silu | affine_silu | affine_silu; affineSiluHalf2Kernel; g4693x1x1; b192x1x1; r17; s0 | 60 | n/a | 5.200 | n/a | 100.0% | 0.000 | fused_ffn (30) | n/a |
| 120 | outer_03.post_projection_c384_to_c768_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b256x1x1; r154; s73728 | 60 | n/a | 25.392 | n/a | 97.5% | 0.000 | linear2_residual (32) | n/a |
| 121 | outer_04.pre_norm_silu | affine_silu | affine_silu; affineSiluHalf2Kernel; g4693x1x1; b384x1x1; r17; s0 | 60 | n/a | 8.080 | n/a | 98.3% | 0.000 | linear2_residual (30) | n/a |
| 122 | outer_04.pre_projection_c768_to_c384 | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 60 | n/a | 16.337 | n/a | 95.2% | 0.000 | fa4 (30) | n/a |
| 123 | outer_04.transformer_0.block_12.attention_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | n/a | 4.320 | n/a | 100.0% | 0.000 | fa4 (30) | n/a |
| 124 | outer_04.transformer_0.block_12.attention_qkv_projection | wide_qkv | wide_qkv; wide_qkv_kernel; g9x37x1; b128x1x1; r136; s65536 | 60 | n/a | 22.256 | n/a | 63.8% | 0.000 | library_gemm (30) | n/a |
| 125 | outer_04.transformer_0.block_12.attention_qk_rope | qk_rope | qk_rope; fusedQKRoPE19HalfKernel; g361x13x1; b192x1x1; r16; s0 | 60 | n/a | 5.344 | n/a | 100.0% | 0.000 | fa4 (30) | n/a |
| 126 | outer_04.transformer_0.block_12.attention_fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 60 | n/a | 15.552 | n/a | 55.1% | 0.000 | fused_ffn (30) | n/a |
| 127 | outer_04.transformer_0.block_12.attention_out_projection_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 60 | n/a | 22.304 | n/a | 97.0% | 0.000 | fused_ffn (60) | n/a |
| 128 | outer_04.transformer_0.block_12.ffn_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | n/a | 4.080 | n/a | 100.0% | 0.000 | fused_ffn (30) | n/a |
| 129 | outer_04.transformer_0.block_12.ffn_linear1_gate_swiglu | fused_ffn | fused_ffn; fused_ffn_kernel; g18x37x1; b128x1x1; r146; s32768 | 60 | n/a | 45.537 | n/a | 90.7% | 0.000 | linear2_residual (58) | n/a |
| 130 | outer_04.transformer_0.block_12.ffn_linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 60 | n/a | 30.640 | n/a | 97.4% | 0.000 | library_gemm (30) | n/a |
| 131 | outer_04.transformer_1.block_13.attention_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | n/a | 4.240 | n/a | 100.0% | 0.000 | library_gemm (30) | n/a |
| 132 | outer_04.transformer_1.block_13.attention_qkv_projection | wide_qkv | wide_qkv; wide_qkv_kernel; g9x37x1; b128x1x1; r136; s65536 | 60 | n/a | 23.073 | n/a | 65.5% | 0.000 | qk_rope (30) | n/a |
| 133 | outer_04.transformer_1.block_13.attention_qk_rope | qk_rope | qk_rope; fusedQKRoPE19HalfKernel; g361x13x1; b192x1x1; r16; s0 | 60 | n/a | 7.584 | n/a | 100.0% | 0.000 | fa4 (30) | n/a |
| 134 | outer_04.transformer_1.block_13.attention_fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 60 | n/a | 15.008 | n/a | 45.5% | 0.000 | library_gemm (30) | n/a |
| 135 | outer_04.transformer_1.block_13.attention_out_projection_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 60 | n/a | 17.888 | n/a | 95.9% | 0.000 | library_gemm (31) | n/a |
| 136 | outer_04.transformer_1.block_13.ffn_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | n/a | 4.112 | n/a | 100.0% | 0.000 | fused_ffn (30) | n/a |
| 137 | outer_04.transformer_1.block_13.ffn_linear1_gate_swiglu | fused_ffn | fused_ffn; fused_ffn_kernel; g18x37x1; b128x1x1; r146; s32768 | 60 | n/a | 45.985 | n/a | 79.0% | 0.000 | fused_ffn (30) | n/a |
| 138 | outer_04.transformer_1.block_13.ffn_linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 60 | n/a | 34.209 | n/a | 98.2% | 0.000 | fused_ffn (30) | n/a |
| 139 | outer_04.transformer_2.block_14.attention_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | n/a | 3.216 | n/a | 100.0% | 0.000 | library_gemm (30) | n/a |
| 140 | outer_04.transformer_2.block_14.attention_qkv_projection | wide_qkv | wide_qkv; wide_qkv_kernel; g9x37x1; b128x1x1; r136; s65536 | 60 | n/a | 28.224 | n/a | 84.4% | 0.000 | affine_silu (30) | n/a |
| 141 | outer_04.transformer_2.block_14.attention_qk_rope | qk_rope | qk_rope; fusedQKRoPE19HalfKernel; g361x13x1; b192x1x1; r16; s0 | 60 | n/a | 8.016 | n/a | 89.9% | 0.000 | library_gemm (30) | n/a |
| 142 | outer_04.transformer_2.block_14.attention_fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 60 | n/a | 18.128 | n/a | 64.7% | 0.000 | library_gemm (30) | n/a |
| 143 | outer_04.transformer_2.block_14.attention_out_projection_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 60 | n/a | 17.136 | n/a | 96.1% | 0.000 | library_gemm (30) | n/a |
| 144 | outer_04.transformer_2.block_14.ffn_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | n/a | 3.072 | n/a | 100.0% | 0.000 | fa4 (30) | n/a |
| 145 | outer_04.transformer_2.block_14.ffn_linear1_gate_swiglu | fused_ffn | fused_ffn; fused_ffn_kernel; g18x37x1; b128x1x1; r146; s32768 | 60 | n/a | 41.393 | n/a | 64.0% | 0.000 | fused_ffn (30) | n/a |
| 146 | outer_04.transformer_2.block_14.ffn_linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 60 | n/a | 32.160 | n/a | 97.6% | 0.000 | fused_ffn (60) | n/a |
| 147 | outer_04.post_norm_silu | affine_silu | affine_silu; affineSiluHalf2Kernel; g4693x1x1; b192x1x1; r17; s0 | 60 | n/a | 5.504 | n/a | 100.0% | 0.000 | fused_ffn (30) | n/a |
| 148 | outer_04.post_projection_c384_to_c768_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b256x1x1; r154; s73728 | 60 | n/a | 27.329 | n/a | 97.6% | 0.000 | linear2_residual (31) | n/a |
| 149 | outer_05.pre_norm_silu | affine_silu | affine_silu; affineSiluHalf2Kernel; g4693x1x1; b384x1x1; r17; s0 | 60 | n/a | 8.608 | n/a | 99.4% | 0.000 | linear2_residual (30) | n/a |
| 150 | outer_05.pre_projection_c768_to_c384 | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 60 | n/a | 18.496 | n/a | 95.4% | 0.000 | fa4 (30) | n/a |
| 151 | outer_05.transformer_0.block_15.attention_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | n/a | 4.336 | n/a | 100.0% | 0.000 | fa4 (30) | n/a |
| 152 | outer_05.transformer_0.block_15.attention_qkv_projection | wide_qkv | wide_qkv; wide_qkv_kernel; g9x37x1; b128x1x1; r136; s65536 | 60 | n/a | 22.992 | n/a | 64.7% | 0.000 | library_gemm (30) | n/a |
| 153 | outer_05.transformer_0.block_15.attention_qk_rope | qk_rope | qk_rope; fusedQKRoPE19HalfKernel; g361x13x1; b192x1x1; r16; s0 | 60 | n/a | 5.408 | n/a | 98.8% | 0.000 | fa4 (30) | n/a |
| 154 | outer_05.transformer_0.block_15.attention_fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 60 | n/a | 15.968 | n/a | 56.6% | 0.000 | fused_ffn (29) | n/a |
| 155 | outer_05.transformer_0.block_15.attention_out_projection_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 60 | n/a | 20.977 | n/a | 96.8% | 0.000 | fused_ffn (59) | n/a |
| 156 | outer_05.transformer_0.block_15.ffn_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | n/a | 4.592 | n/a | 100.0% | 0.000 | fused_ffn (30) | n/a |
| 157 | outer_05.transformer_0.block_15.ffn_linear1_gate_swiglu | fused_ffn | fused_ffn; fused_ffn_kernel; g18x37x1; b128x1x1; r146; s32768 | 60 | n/a | 49.072 | n/a | 84.0% | 0.000 | linear2_residual (60) | n/a |
| 158 | outer_05.transformer_0.block_15.ffn_linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 60 | n/a | 31.921 | n/a | 97.4% | 0.000 | library_gemm (30) | n/a |
| 159 | outer_05.transformer_1.block_16.attention_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | n/a | 3.968 | n/a | 100.0% | 0.000 | library_gemm (30) | n/a |
| 160 | outer_05.transformer_1.block_16.attention_qkv_projection | wide_qkv | wide_qkv; wide_qkv_kernel; g9x37x1; b128x1x1; r136; s65536 | 60 | n/a | 23.008 | n/a | 65.7% | 0.000 | qk_rope (30) | n/a |
| 161 | outer_05.transformer_1.block_16.attention_qk_rope | qk_rope | qk_rope; fusedQKRoPE19HalfKernel; g361x13x1; b192x1x1; r16; s0 | 60 | n/a | 7.584 | n/a | 97.5% | 0.000 | fa4 (30) | n/a |
| 162 | outer_05.transformer_1.block_16.attention_fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 60 | n/a | 15.040 | n/a | 47.4% | 0.000 | qk_rope (29) | n/a |
| 163 | outer_05.transformer_1.block_16.attention_out_projection_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 60 | n/a | 16.720 | n/a | 95.9% | 0.000 | library_gemm (31) | n/a |
| 164 | outer_05.transformer_1.block_16.ffn_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | n/a | 4.224 | n/a | 100.0% | 0.000 | fused_ffn (30) | n/a |
| 165 | outer_05.transformer_1.block_16.ffn_linear1_gate_swiglu | fused_ffn | fused_ffn; fused_ffn_kernel; g18x37x1; b128x1x1; r146; s32768 | 60 | n/a | 46.816 | n/a | 75.8% | 0.000 | fused_ffn (30) | n/a |
| 166 | outer_05.transformer_1.block_16.ffn_linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 60 | n/a | 33.025 | n/a | 98.1% | 0.000 | fused_ffn (30) | n/a |
| 167 | outer_05.transformer_2.block_17.attention_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | n/a | 3.040 | n/a | 100.0% | 0.000 | library_gemm (30) | n/a |
| 168 | outer_05.transformer_2.block_17.attention_qkv_projection | wide_qkv | wide_qkv; wide_qkv_kernel; g9x37x1; b128x1x1; r136; s65536 | 60 | n/a | 26.576 | n/a | 84.2% | 0.000 | affine_silu (30) | n/a |
| 169 | outer_05.transformer_2.block_17.attention_qk_rope | qk_rope | qk_rope; fusedQKRoPE19HalfKernel; g361x13x1; b192x1x1; r16; s0 | 60 | n/a | 8.065 | n/a | 79.9% | 0.000 | library_gemm (30) | n/a |
| 170 | outer_05.transformer_2.block_17.attention_fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 60 | n/a | 18.096 | n/a | 68.7% | 0.000 | library_gemm (30) | n/a |
| 171 | outer_05.transformer_2.block_17.attention_out_projection_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 60 | n/a | 16.912 | n/a | 96.1% | 0.000 | library_gemm (30) | n/a |
| 172 | outer_05.transformer_2.block_17.ffn_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | n/a | 3.120 | n/a | 100.0% | 0.000 | fa4 (30) | n/a |
| 173 | outer_05.transformer_2.block_17.ffn_linear1_gate_swiglu | fused_ffn | fused_ffn; fused_ffn_kernel; g18x37x1; b128x1x1; r146; s32768 | 60 | n/a | 46.544 | n/a | 57.2% | 0.000 | fused_ffn (30) | n/a |
| 174 | outer_05.transformer_2.block_17.ffn_linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 60 | n/a | 31.776 | n/a | 97.5% | 0.000 | fused_ffn (60) | n/a |
| 175 | outer_05.post_norm_silu | affine_silu | affine_silu; affineSiluHalf2Kernel; g4693x1x1; b192x1x1; r17; s0 | 60 | n/a | 5.312 | n/a | 100.0% | 0.000 | fused_ffn (30) | n/a |
| 176 | outer_05.post_projection_c384_to_c768_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b256x1x1; r154; s73728 | 60 | n/a | 26.688 | n/a | 97.6% | 0.000 | fused_ffn (30) | n/a |
| 177 | outer_06.pre_norm_silu | affine_silu | affine_silu; affineSiluHalf2Kernel; g4693x1x1; b384x1x1; r17; s0 | 60 | n/a | 8.208 | n/a | 98.8% | 0.000 | linear2_residual (30) | n/a |
| 178 | outer_06.pre_projection_c768_to_c384 | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 60 | n/a | 16.817 | n/a | 95.1% | 0.000 | fa4 (30) | n/a |
| 179 | outer_06.transformer_0.block_18.attention_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | n/a | 4.544 | n/a | 100.0% | 0.000 | fa4 (30) | n/a |
| 180 | outer_06.transformer_0.block_18.attention_qkv_projection | wide_qkv | wide_qkv; wide_qkv_kernel; g9x37x1; b128x1x1; r136; s65536 | 60 | n/a | 22.384 | n/a | 63.9% | 0.000 | library_gemm (30) | n/a |
| 181 | outer_06.transformer_0.block_18.attention_qk_rope | qk_rope | qk_rope; fusedQKRoPE19HalfKernel; g361x13x1; b192x1x1; r16; s0 | 60 | n/a | 5.312 | n/a | 97.8% | 0.000 | fa4 (30) | n/a |
| 182 | outer_06.transformer_0.block_18.attention_fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 60 | n/a | 15.712 | n/a | 56.4% | 0.000 | fused_ffn (30) | n/a |
| 183 | outer_06.transformer_0.block_18.attention_out_projection_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 60 | n/a | 22.032 | n/a | 97.0% | 0.000 | fused_ffn (58) | n/a |
| 184 | outer_06.transformer_0.block_18.ffn_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | n/a | 4.272 | n/a | 100.0% | 0.000 | fused_ffn (30) | n/a |
| 185 | outer_06.transformer_0.block_18.ffn_linear1_gate_swiglu | fused_ffn | fused_ffn; fused_ffn_kernel; g18x37x1; b128x1x1; r146; s32768 | 60 | n/a | 47.728 | n/a | 88.7% | 0.000 | linear2_residual (60) | n/a |
| 186 | outer_06.transformer_0.block_18.ffn_linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 60 | n/a | 30.689 | n/a | 97.3% | 0.000 | library_gemm (30) | n/a |
| 187 | outer_06.transformer_1.block_19.attention_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | n/a | 4.032 | n/a | 100.0% | 0.000 | library_gemm (30) | n/a |
| 188 | outer_06.transformer_1.block_19.attention_qkv_projection | wide_qkv | wide_qkv; wide_qkv_kernel; g9x37x1; b128x1x1; r136; s65536 | 60 | n/a | 23.457 | n/a | 66.0% | 0.000 | qk_rope (30) | n/a |
| 189 | outer_06.transformer_1.block_19.attention_qk_rope | qk_rope | qk_rope; fusedQKRoPE19HalfKernel; g361x13x1; b192x1x1; r16; s0 | 60 | n/a | 7.360 | n/a | 98.2% | 0.000 | fa4 (30) | n/a |
| 190 | outer_06.transformer_1.block_19.attention_fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 60 | n/a | 14.976 | n/a | 48.2% | 0.000 | library_gemm (28) | n/a |
| 191 | outer_06.transformer_1.block_19.attention_out_projection_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 60 | n/a | 16.785 | n/a | 95.9% | 0.000 | fused_ffn (30) | n/a |
| 192 | outer_06.transformer_1.block_19.ffn_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | n/a | 4.016 | n/a | 100.0% | 0.000 | fused_ffn (30) | n/a |
| 193 | outer_06.transformer_1.block_19.ffn_linear1_gate_swiglu | fused_ffn | fused_ffn; fused_ffn_kernel; g18x37x1; b128x1x1; r146; s32768 | 60 | n/a | 47.489 | n/a | 77.5% | 0.000 | fused_ffn (30) | n/a |
| 194 | outer_06.transformer_1.block_19.ffn_linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 60 | n/a | 33.568 | n/a | 98.1% | 0.000 | fused_ffn (30) | n/a |
| 195 | outer_06.transformer_2.block_20.attention_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | n/a | 3.168 | n/a | 100.0% | 0.000 | library_gemm (30) | n/a |
| 196 | outer_06.transformer_2.block_20.attention_qkv_projection | wide_qkv | wide_qkv; wide_qkv_kernel; g9x37x1; b128x1x1; r136; s65536 | 60 | n/a | 29.248 | n/a | 84.3% | 0.000 | linear2_residual (30) | n/a |
| 197 | outer_06.transformer_2.block_20.attention_qk_rope | qk_rope | qk_rope; fusedQKRoPE19HalfKernel; g361x13x1; b192x1x1; r16; s0 | 60 | n/a | 8.080 | n/a | 99.9% | 0.000 | library_gemm (30) | n/a |
| 198 | outer_06.transformer_2.block_20.attention_fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 60 | n/a | 17.328 | n/a | 58.1% | 0.000 | library_gemm (30) | n/a |
| 199 | outer_06.transformer_2.block_20.attention_out_projection_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 60 | n/a | 17.232 | n/a | 96.2% | 0.000 | library_gemm (30) | n/a |
| 200 | outer_06.transformer_2.block_20.ffn_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | n/a | 3.072 | n/a | 100.0% | 0.000 | fa4 (30) | n/a |
| 201 | outer_06.transformer_2.block_20.ffn_linear1_gate_swiglu | fused_ffn | fused_ffn; fused_ffn_kernel; g18x37x1; b128x1x1; r146; s32768 | 60 | n/a | 40.577 | n/a | 64.4% | 0.000 | fused_ffn (30) | n/a |
| 202 | outer_06.transformer_2.block_20.ffn_linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 60 | n/a | 32.641 | n/a | 97.5% | 0.000 | fused_ffn (60) | n/a |
| 203 | outer_06.post_norm_silu | affine_silu | affine_silu; affineSiluHalf2Kernel; g4693x1x1; b192x1x1; r17; s0 | 60 | n/a | 5.984 | n/a | 100.0% | 0.000 | fused_ffn (30) | n/a |
| 204 | outer_06.post_projection_c384_to_c768_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b256x1x1; r154; s73728 | 60 | n/a | 28.945 | n/a | 97.7% | 0.000 | linear2_residual (37) | n/a |
| 205 | outer_07.pre_norm_silu | affine_silu | affine_silu; affineSiluHalf2Kernel; g4693x1x1; b384x1x1; r17; s0 | 60 | n/a | 8.096 | n/a | 100.0% | 0.000 | linear2_residual (30) | n/a |
| 206 | outer_07.pre_projection_c768_to_c384 | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 60 | n/a | 18.048 | n/a | 94.4% | 0.000 | linear2_residual (30) | n/a |
| 207 | outer_07.transformer_0.block_21.attention_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | n/a | 3.776 | n/a | 100.0% | 0.000 | fa4 (30) | n/a |
| 208 | outer_07.transformer_0.block_21.attention_qkv_projection | wide_qkv | wide_qkv; wide_qkv_kernel; g9x37x1; b128x1x1; r136; s65536 | 60 | n/a | 23.201 | n/a | 65.6% | 0.000 | library_gemm (30) | n/a |
| 209 | outer_07.transformer_0.block_21.attention_qk_rope | qk_rope | qk_rope; fusedQKRoPE19HalfKernel; g361x13x1; b192x1x1; r16; s0 | 60 | n/a | 5.376 | n/a | 100.0% | 0.000 | fa4 (30) | n/a |
| 210 | outer_07.transformer_0.block_21.attention_fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 60 | n/a | 16.112 | n/a | 56.4% | 0.000 | library_gemm (30) | n/a |
| 211 | outer_07.transformer_0.block_21.attention_out_projection_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 60 | n/a | 20.864 | n/a | 96.8% | 0.000 | fused_ffn (58) | n/a |
| 212 | outer_07.transformer_0.block_21.ffn_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | n/a | 4.592 | n/a | 100.0% | 0.000 | fused_ffn (30) | n/a |
| 213 | outer_07.transformer_0.block_21.ffn_linear1_gate_swiglu | fused_ffn | fused_ffn; fused_ffn_kernel; g18x37x1; b128x1x1; r146; s32768 | 60 | n/a | 45.904 | n/a | 93.2% | 0.000 | linear2_residual (60) | n/a |
| 214 | outer_07.transformer_0.block_21.ffn_linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 60 | n/a | 34.432 | n/a | 97.6% | 0.000 | library_gemm (30) | n/a |
| 215 | outer_07.transformer_1.block_22.attention_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | n/a | 3.968 | n/a | 100.0% | 0.000 | library_gemm (30) | n/a |
| 216 | outer_07.transformer_1.block_22.attention_qkv_projection | wide_qkv | wide_qkv; wide_qkv_kernel; g9x37x1; b128x1x1; r136; s65536 | 60 | n/a | 24.032 | n/a | 65.3% | 0.000 | qk_rope (30) | n/a |
| 217 | outer_07.transformer_1.block_22.attention_qk_rope | qk_rope | qk_rope; fusedQKRoPE19HalfKernel; g361x13x1; b192x1x1; r16; s0 | 60 | n/a | 7.456 | n/a | 99.2% | 0.000 | fa4 (30) | n/a |
| 218 | outer_07.transformer_1.block_22.attention_fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 60 | n/a | 15.377 | n/a | 49.0% | 0.000 | qk_rope (30) | n/a |
| 219 | outer_07.transformer_1.block_22.attention_out_projection_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 60 | n/a | 17.793 | n/a | 96.0% | 0.000 | fused_ffn (30) | n/a |
| 220 | outer_07.transformer_1.block_22.ffn_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | n/a | 4.064 | n/a | 100.0% | 0.000 | fused_ffn (30) | n/a |
| 221 | outer_07.transformer_1.block_22.ffn_linear1_gate_swiglu | fused_ffn | fused_ffn; fused_ffn_kernel; g18x37x1; b128x1x1; r146; s32768 | 60 | n/a | 43.904 | n/a | 83.8% | 0.000 | fused_ffn (30) | n/a |
| 222 | outer_07.transformer_1.block_22.ffn_linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 60 | n/a | 36.657 | n/a | 98.2% | 0.000 | fused_ffn (30) | n/a |
| 223 | outer_07.transformer_2.block_23.attention_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | n/a | 3.136 | n/a | 100.0% | 0.000 | library_gemm (30) | n/a |
| 224 | outer_07.transformer_2.block_23.attention_qkv_projection | wide_qkv | wide_qkv; wide_qkv_kernel; g9x37x1; b128x1x1; r136; s65536 | 60 | n/a | 28.881 | n/a | 84.7% | 0.000 | linear2_residual (30) | n/a |
| 225 | outer_07.transformer_2.block_23.attention_qk_rope | qk_rope | qk_rope; fusedQKRoPE19HalfKernel; g361x13x1; b192x1x1; r16; s0 | 60 | n/a | 8.128 | n/a | 86.4% | 0.000 | library_gemm (30) | n/a |
| 226 | outer_07.transformer_2.block_23.attention_fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 60 | n/a | 18.528 | n/a | 67.5% | 0.000 | library_gemm (30) | n/a |
| 227 | outer_07.transformer_2.block_23.attention_out_projection_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 60 | n/a | 17.152 | n/a | 96.2% | 0.000 | library_gemm (30) | n/a |
| 228 | outer_07.transformer_2.block_23.ffn_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | n/a | 3.024 | n/a | 100.0% | 0.000 | fa4 (30) | n/a |
| 229 | outer_07.transformer_2.block_23.ffn_linear1_gate_swiglu | fused_ffn | fused_ffn; fused_ffn_kernel; g18x37x1; b128x1x1; r146; s32768 | 60 | n/a | 41.968 | n/a | 65.5% | 0.000 | fused_ffn (30) | n/a |
| 230 | outer_07.transformer_2.block_23.ffn_linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 60 | n/a | 32.960 | n/a | 97.6% | 0.000 | fused_ffn (60) | n/a |
| 231 | outer_07.post_norm_silu | affine_silu | affine_silu; affineSiluHalf2Kernel; g4693x1x1; b192x1x1; r17; s0 | 60 | n/a | 5.952 | n/a | 100.0% | 0.000 | fused_ffn (30) | n/a |
| 232 | outer_07.post_projection_c384_to_c768_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b256x1x1; r154; s73728 | 60 | n/a | 26.304 | n/a | 97.6% | 0.000 | linear2_residual (37) | n/a |
| 233 | outer_08.pre_norm_silu | affine_silu | affine_silu; affineSiluHalf2Kernel; g4693x1x1; b384x1x1; r17; s0 | 60 | n/a | 8.289 | n/a | 99.2% | 0.000 | linear2_residual (30) | n/a |
| 234 | outer_08.pre_projection_c768_to_c384 | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 60 | n/a | 18.448 | n/a | 95.4% | 0.000 | fa4 (30) | n/a |
| 235 | outer_08.transformer_0.block_24.attention_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | n/a | 4.320 | n/a | 100.0% | 0.000 | fa4 (30) | n/a |
| 236 | outer_08.transformer_0.block_24.attention_qkv_projection | wide_qkv | wide_qkv; wide_qkv_kernel; g9x37x1; b128x1x1; r136; s65536 | 60 | n/a | 23.249 | n/a | 65.9% | 0.000 | library_gemm (30) | n/a |
| 237 | outer_08.transformer_0.block_24.attention_qk_rope | qk_rope | qk_rope; fusedQKRoPE19HalfKernel; g361x13x1; b192x1x1; r16; s0 | 60 | n/a | 5.376 | n/a | 100.0% | 0.000 | fa4 (30) | n/a |
| 238 | outer_08.transformer_0.block_24.attention_fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 60 | n/a | 15.905 | n/a | 56.1% | 0.000 | library_gemm (30) | n/a |
| 239 | outer_08.transformer_0.block_24.attention_out_projection_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 60 | n/a | 21.056 | n/a | 96.8% | 0.000 | fused_ffn (58) | n/a |
| 240 | outer_08.transformer_0.block_24.ffn_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | n/a | 4.832 | n/a | 100.0% | 0.000 | fused_ffn (30) | n/a |
| 241 | outer_08.transformer_0.block_24.ffn_linear1_gate_swiglu | fused_ffn | fused_ffn; fused_ffn_kernel; g18x37x1; b128x1x1; r146; s32768 | 60 | n/a | 49.745 | n/a | 87.9% | 0.000 | linear2_residual (60) | n/a |
| 242 | outer_08.transformer_0.block_24.ffn_linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 60 | n/a | 33.120 | n/a | 97.4% | 0.000 | library_gemm (30) | n/a |
| 243 | outer_08.transformer_1.block_25.attention_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | n/a | 4.000 | n/a | 100.0% | 0.000 | library_gemm (30) | n/a |
| 244 | outer_08.transformer_1.block_25.attention_qkv_projection | wide_qkv | wide_qkv; wide_qkv_kernel; g9x37x1; b128x1x1; r136; s65536 | 60 | n/a | 23.649 | n/a | 65.0% | 0.000 | qk_rope (30) | n/a |
| 245 | outer_08.transformer_1.block_25.attention_qk_rope | qk_rope | qk_rope; fusedQKRoPE19HalfKernel; g361x13x1; b192x1x1; r16; s0 | 60 | n/a | 7.552 | n/a | 99.2% | 0.000 | fa4 (30) | n/a |
| 246 | outer_08.transformer_1.block_25.attention_fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 60 | n/a | 14.992 | n/a | 46.4% | 0.000 | qk_rope (30) | n/a |
| 247 | outer_08.transformer_1.block_25.attention_out_projection_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 60 | n/a | 16.656 | n/a | 95.9% | 0.000 | library_gemm (31) | n/a |
| 248 | outer_08.transformer_1.block_25.ffn_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | n/a | 4.144 | n/a | 100.0% | 0.000 | fused_ffn (30) | n/a |
| 249 | outer_08.transformer_1.block_25.ffn_linear1_gate_swiglu | fused_ffn | fused_ffn; fused_ffn_kernel; g18x37x1; b128x1x1; r146; s32768 | 60 | n/a | 47.825 | n/a | 78.6% | 0.000 | fused_ffn (30) | n/a |
| 250 | outer_08.transformer_1.block_25.ffn_linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 60 | n/a | 33.344 | n/a | 98.2% | 0.000 | fused_ffn (30) | n/a |
| 251 | outer_08.transformer_2.block_26.attention_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | n/a | 3.088 | n/a | 100.0% | 0.000 | library_gemm (30) | n/a |
| 252 | outer_08.transformer_2.block_26.attention_qkv_projection | wide_qkv | wide_qkv; wide_qkv_kernel; g9x37x1; b128x1x1; r136; s65536 | 60 | n/a | 28.337 | n/a | 83.2% | 0.000 | linear2_residual (30) | n/a |
| 253 | outer_08.transformer_2.block_26.attention_qk_rope | qk_rope | qk_rope; fusedQKRoPE19HalfKernel; g361x13x1; b192x1x1; r16; s0 | 60 | n/a | 8.272 | n/a | 99.2% | 0.000 | library_gemm (30) | n/a |
| 254 | outer_08.transformer_2.block_26.attention_fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 60 | n/a | 18.064 | n/a | 59.6% | 0.000 | library_gemm (30) | n/a |
| 255 | outer_08.transformer_2.block_26.attention_out_projection_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 60 | n/a | 17.136 | n/a | 96.2% | 0.000 | library_gemm (30) | n/a |
| 256 | outer_08.transformer_2.block_26.ffn_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | n/a | 3.040 | n/a | 100.0% | 0.000 | fa4 (30) | n/a |
| 257 | outer_08.transformer_2.block_26.ffn_linear1_gate_swiglu | fused_ffn | fused_ffn; fused_ffn_kernel; g18x37x1; b128x1x1; r146; s32768 | 60 | n/a | 46.752 | n/a | 60.6% | 0.000 | fused_ffn (30) | n/a |
| 258 | outer_08.transformer_2.block_26.ffn_linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 60 | n/a | 32.016 | n/a | 97.5% | 0.000 | fused_ffn (60) | n/a |
| 259 | outer_08.post_norm_silu | affine_silu | affine_silu; affineSiluHalf2Kernel; g4693x1x1; b192x1x1; r17; s0 | 60 | n/a | 5.120 | n/a | 100.0% | 0.000 | fused_ffn (30) | n/a |
| 260 | outer_08.post_projection_c384_to_c768_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b256x1x1; r154; s73728 | 60 | n/a | 26.768 | n/a | 97.6% | 0.000 | fused_ffn (30) | n/a |
| 261 | outer_09.pre_norm_silu | affine_silu | affine_silu; affineSiluHalf2Kernel; g4693x1x1; b384x1x1; r17; s0 | 60 | n/a | 8.097 | n/a | 100.0% | 0.000 | linear2_residual (30) | n/a |
| 262 | outer_09.pre_projection_c768_to_c384 | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 60 | n/a | 17.280 | n/a | 95.0% | 0.000 | linear2_residual (30) | n/a |
| 263 | outer_09.transformer_0.block_27.attention_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | n/a | 3.920 | n/a | 100.0% | 0.000 | fa4 (30) | n/a |
| 264 | outer_09.transformer_0.block_27.attention_qkv_projection | wide_qkv | wide_qkv; wide_qkv_kernel; g9x37x1; b128x1x1; r136; s65536 | 60 | n/a | 22.560 | n/a | 66.9% | 0.000 | library_gemm (30) | n/a |
| 265 | outer_09.transformer_0.block_27.attention_qk_rope | qk_rope | qk_rope; fusedQKRoPE19HalfKernel; g361x13x1; b192x1x1; r16; s0 | 60 | n/a | 5.312 | n/a | 97.8% | 0.000 | fa4 (30) | n/a |
| 266 | outer_09.transformer_0.block_27.attention_fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 60 | n/a | 16.224 | n/a | 57.1% | 0.000 | fused_ffn (30) | n/a |
| 267 | outer_09.transformer_0.block_27.attention_out_projection_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 60 | n/a | 21.953 | n/a | 97.0% | 0.000 | fused_ffn (58) | n/a |
| 268 | outer_09.transformer_0.block_27.ffn_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | n/a | 4.720 | n/a | 100.0% | 0.000 | fused_ffn (30) | n/a |
| 269 | outer_09.transformer_0.block_27.ffn_linear1_gate_swiglu | fused_ffn | fused_ffn; fused_ffn_kernel; g18x37x1; b128x1x1; r146; s32768 | 60 | n/a | 47.745 | n/a | 80.3% | 0.000 | linear2_residual (60) | n/a |
| 270 | outer_09.transformer_0.block_27.ffn_linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 60 | n/a | 30.465 | n/a | 97.3% | 0.000 | library_gemm (30) | n/a |
| 271 | outer_09.transformer_1.block_28.attention_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | n/a | 3.712 | n/a | 100.0% | 0.000 | library_gemm (30) | n/a |
| 272 | outer_09.transformer_1.block_28.attention_qkv_projection | wide_qkv | wide_qkv; wide_qkv_kernel; g9x37x1; b128x1x1; r136; s65536 | 60 | n/a | 22.369 | n/a | 65.9% | 0.000 | qk_rope (30) | n/a |
| 273 | outer_09.transformer_1.block_28.attention_qk_rope | qk_rope | qk_rope; fusedQKRoPE19HalfKernel; g361x13x1; b192x1x1; r16; s0 | 60 | n/a | 7.393 | n/a | 96.7% | 0.000 | fa4 (30) | n/a |
| 274 | outer_09.transformer_1.block_28.attention_fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 60 | n/a | 14.848 | n/a | 47.4% | 0.000 | qk_rope (28) | n/a |
| 275 | outer_09.transformer_1.block_28.attention_out_projection_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 60 | n/a | 16.384 | n/a | 95.9% | 0.000 | library_gemm (31) | n/a |
| 276 | outer_09.transformer_1.block_28.ffn_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | n/a | 4.368 | n/a | 100.0% | 0.000 | fused_ffn (30) | n/a |
| 277 | outer_09.transformer_1.block_28.ffn_linear1_gate_swiglu | fused_ffn | fused_ffn; fused_ffn_kernel; g18x37x1; b128x1x1; r146; s32768 | 60 | n/a | 45.584 | n/a | 75.8% | 0.000 | fused_ffn (32) | n/a |
| 278 | outer_09.transformer_1.block_28.ffn_linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 60 | n/a | 30.880 | n/a | 98.0% | 0.000 | fused_ffn (30) | n/a |
| 279 | outer_09.transformer_2.block_29.attention_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | n/a | 3.024 | n/a | 100.0% | 0.000 | library_gemm (30) | n/a |
| 280 | outer_09.transformer_2.block_29.attention_qkv_projection | wide_qkv | wide_qkv; wide_qkv_kernel; g9x37x1; b128x1x1; r136; s65536 | 60 | n/a | 25.632 | n/a | 84.6% | 0.000 | affine_silu (30) | n/a |
| 281 | outer_09.transformer_2.block_29.attention_qk_rope | qk_rope | qk_rope; fusedQKRoPE19HalfKernel; g361x13x1; b192x1x1; r16; s0 | 60 | n/a | 7.888 | n/a | 76.9% | 0.000 | library_gemm (30) | n/a |
| 282 | outer_09.transformer_2.block_29.attention_fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 60 | n/a | 17.441 | n/a | 69.9% | 0.000 | library_gemm (30) | n/a |
| 283 | outer_09.transformer_2.block_29.attention_out_projection_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 60 | n/a | 16.800 | n/a | 96.1% | 0.000 | library_gemm (30) | n/a |
| 284 | outer_09.transformer_2.block_29.ffn_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | n/a | 2.992 | n/a | 100.0% | 0.000 | fa4 (30) | n/a |
| 285 | outer_09.transformer_2.block_29.ffn_linear1_gate_swiglu | fused_ffn | fused_ffn; fused_ffn_kernel; g18x37x1; b128x1x1; r146; s32768 | 60 | n/a | 46.209 | n/a | 57.4% | 0.000 | fused_ffn (30) | n/a |
| 286 | outer_09.transformer_2.block_29.ffn_linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 60 | n/a | 30.817 | n/a | 97.6% | 0.000 | fused_ffn (60) | n/a |
| 287 | outer_09.post_norm_silu | affine_silu | affine_silu; affineSiluHalf2Kernel; g4693x1x1; b192x1x1; r17; s0 | 60 | n/a | 5.712 | n/a | 100.0% | 0.000 | fused_ffn (30) | n/a |
| 288 | outer_09.post_projection_c384_to_c768_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b256x1x1; r154; s73728 | 60 | n/a | 25.105 | n/a | 97.5% | 0.000 | linear2_residual (31) | n/a |
| 289 | outer_10.pre_norm_silu | affine_silu | affine_silu; affineSiluHalf2Kernel; g4693x1x1; b384x1x1; r17; s0 | 60 | n/a | 8.144 | n/a | 98.5% | 0.000 | linear2_residual (30) | n/a |
| 290 | outer_10.pre_projection_c768_to_c384 | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 60 | n/a | 16.448 | n/a | 95.3% | 0.000 | fa4 (30) | n/a |
| 291 | outer_10.transformer_0.block_30.attention_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | n/a | 4.448 | n/a | 100.0% | 0.000 | fa4 (30) | n/a |
| 292 | outer_10.transformer_0.block_30.attention_qkv_projection | wide_qkv | wide_qkv; wide_qkv_kernel; g9x37x1; b128x1x1; r136; s65536 | 60 | n/a | 22.352 | n/a | 63.8% | 0.000 | library_gemm (30) | n/a |
| 293 | outer_10.transformer_0.block_30.attention_qk_rope | qk_rope | qk_rope; fusedQKRoPE19HalfKernel; g361x13x1; b192x1x1; r16; s0 | 60 | n/a | 5.440 | n/a | 100.0% | 0.000 | fa4 (30) | n/a |
| 294 | outer_10.transformer_0.block_30.attention_fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 60 | n/a | 15.552 | n/a | 55.2% | 0.000 | fused_ffn (30) | n/a |
| 295 | outer_10.transformer_0.block_30.attention_out_projection_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 60 | n/a | 22.256 | n/a | 97.0% | 0.000 | fused_ffn (58) | n/a |
| 296 | outer_10.transformer_0.block_30.ffn_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | n/a | 4.384 | n/a | 100.0% | 0.000 | fused_ffn (30) | n/a |
| 297 | outer_10.transformer_0.block_30.ffn_linear1_gate_swiglu | fused_ffn | fused_ffn; fused_ffn_kernel; g18x37x1; b128x1x1; r146; s32768 | 60 | n/a | 43.728 | n/a | 94.9% | 0.000 | linear2_residual (60) | n/a |
| 298 | outer_10.transformer_0.block_30.ffn_linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 60 | n/a | 31.776 | n/a | 97.4% | 0.000 | library_gemm (30) | n/a |
| 299 | outer_10.transformer_1.block_31.attention_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | n/a | 4.016 | n/a | 100.0% | 0.000 | library_gemm (30) | n/a |
| 300 | outer_10.transformer_1.block_31.attention_qkv_projection | wide_qkv | wide_qkv; wide_qkv_kernel; g9x37x1; b128x1x1; r136; s65536 | 60 | n/a | 23.552 | n/a | 65.3% | 0.000 | qk_rope (30) | n/a |
| 301 | outer_10.transformer_1.block_31.attention_qk_rope | qk_rope | qk_rope; fusedQKRoPE19HalfKernel; g361x13x1; b192x1x1; r16; s0 | 60 | n/a | 7.520 | n/a | 100.0% | 0.000 | fa4 (30) | n/a |
| 302 | outer_10.transformer_1.block_31.attention_fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 60 | n/a | 15.072 | n/a | 45.7% | 0.000 | library_gemm (30) | n/a |
| 303 | outer_10.transformer_1.block_31.attention_out_projection_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 60 | n/a | 17.888 | n/a | 95.9% | 0.000 | fused_ffn (30) | n/a |
| 304 | outer_10.transformer_1.block_31.ffn_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | n/a | 4.080 | n/a | 100.0% | 0.000 | fused_ffn (30) | n/a |
| 305 | outer_10.transformer_1.block_31.ffn_linear1_gate_swiglu | fused_ffn | fused_ffn; fused_ffn_kernel; g18x37x1; b128x1x1; r146; s32768 | 60 | n/a | 45.841 | n/a | 83.4% | 0.000 | fused_ffn (30) | n/a |
| 306 | outer_10.transformer_1.block_31.ffn_linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 60 | n/a | 33.648 | n/a | 98.1% | 0.000 | fused_ffn (30) | n/a |
| 307 | outer_10.transformer_2.block_32.attention_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | n/a | 3.200 | n/a | 100.0% | 0.000 | library_gemm (30) | n/a |
| 308 | outer_10.transformer_2.block_32.attention_qkv_projection | wide_qkv | wide_qkv; wide_qkv_kernel; g9x37x1; b128x1x1; r136; s65536 | 60 | n/a | 29.073 | n/a | 84.6% | 0.000 | linear2_residual (30) | n/a |
| 309 | outer_10.transformer_2.block_32.attention_qk_rope | qk_rope | qk_rope; fusedQKRoPE19HalfKernel; g361x13x1; b192x1x1; r16; s0 | 60 | n/a | 8.960 | n/a | 96.8% | 0.000 | library_gemm (30) | n/a |
| 310 | outer_10.transformer_2.block_32.attention_fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 60 | n/a | 16.848 | n/a | 54.3% | 0.000 | qk_rope (30) | n/a |
| 311 | outer_10.transformer_2.block_32.attention_out_projection_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 60 | n/a | 12.177 | n/a | 96.5% | 0.000 | head_elementwise (30) | n/a |
| 312 | outer_10.transformer_2.block_32.ffn_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | n/a | 2.752 | n/a | 96.9% | 0.000 | library_gemm (60) | n/a |
| 313 | outer_10.transformer_2.block_32.ffn_linear1_gate_swiglu | fused_ffn | fused_ffn; fused_ffn_kernel; g18x37x1; b128x1x1; r146; s32768 | 60 | n/a | 46.961 | n/a | 57.5% | 0.000 | fused_ffn (30) | n/a |
| 314 | outer_10.transformer_2.block_32.ffn_linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 60 | n/a | 28.384 | n/a | 97.0% | 0.000 | fused_ffn (30) | n/a |
| 315 | outer_10.post_norm_silu | affine_silu | affine_silu; affineSiluHalf2Kernel; g4693x1x1; b192x1x1; r17; s0 | 60 | n/a | 4.000 | n/a | 100.0% | 0.000 | library_gemm (30) | n/a |
| 316 | outer_10.post_projection_c384_to_c768_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b256x1x1; r154; s73728 | 60 | n/a | 24.032 | n/a | 94.5% | 0.000 | head_elementwise (30) | n/a |
| 317 | trunk.tip_norm_silu | affine_silu | affine_silu; affineSiluHalf2Kernel; g4693x1x1; b384x1x1; r17; s0 | 60 | n/a | 7.056 | n/a | 100.0% | 0.000 | library_gemm (30) | n/a |
| 318 | policy.p1_conv | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r80; s98304 | 60 | n/a | 7.344 | n/a | 95.3% | 0.000 | library_gemm (30) | n/a |
| 319 | policy.g1_conv | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r80; s98304 | 60 | n/a | 6.528 | n/a | 89.7% | 0.000 | fa4 (29) | n/a |
| 320 | policy.g1_norm_silu | head_elementwise | head_elementwise; applyCScaleBiasNHWCSiluHalfKernel; g1x73x13; b96x5x1; r16; s0 | 60 | n/a | 3.120 | n/a | 88.2% | 0.000 | fa4 (30) | n/a |
| 321 | policy.g1_half_to_float | copy_reformat | copy_reformat; copyFromHalfKernel; g880x1x1; b512x1x1; r16; s0 | 60 | n/a | 1.920 | n/a | 73.4% | 0.000 | fa4 (30) | n/a |
| 322 | policy.g1_global_pool | head_elementwise | head_elementwise; gPoolChannelsNHWCKernel; g2x1x13; b64x8x1; r22; s4096 | 60 | n/a | 5.872 | n/a | 90.4% | 0.000 | library_gemm (60) | n/a |
| 323 | policy.gpool_to_bias_matmul | library_gemm | library_gemm; gemmSN_NN_kernel; g2x2x1; b256x1x1; r64; s21504 | 60 | n/a | 6.336 | n/a | 86.7% | 0.000 | library_gemm (59) | n/a |
| 324 | policy.p1_half_to_float | copy_reformat | copy_reformat; copyFromHalfKernel; g880x1x1; b512x1x1; r16; s0 | 60 | n/a | 2.016 | n/a | 48.4% | 0.000 | fused_ffn (30) | n/a |
| 325 | policy.gpool_bias_add | head_elementwise | head_elementwise; addNCBiasInplaceNHWCKernel; g1x73x13; b96x5x1; r16; s0 | 60 | n/a | 2.288 | n/a | 72.1% | 0.000 | fused_ffn (30) | n/a |
| 326 | policy.p1_norm_silu | head_elementwise | head_elementwise; applyCScaleBiasNHWCSiluKernel; g1x73x13; b96x5x1; r16; s0 | 60 | n/a | 2.688 | n/a | 77.2% | 0.000 | fused_ffn (30) | n/a |
| 327 | policy.p2_conv | library_gemm | library_gemm; Kernel2; g74x1x1; b128x1x1; r90; s98304 | 60 | n/a | 6.000 | n/a | 83.5% | 0.000 | fused_ffn (30) | n/a |
| 328 | policy.gpool_to_pass_matmul | library_gemm | library_gemm; gemmSN_NN_kernel; g2x2x1; b256x1x1; r64; s21504 | 60 | n/a | 6.896 | n/a | 79.5% | 0.000 | linear2_residual (29) | n/a |
| 329 | policy.pass_bias_silu | head_elementwise | head_elementwise; addCBiasInplaceNCKernelSilu; g1x3x1; b96x5x1; r16; s0 | 60 | n/a | 1.248 | n/a | 97.7% | 0.000 | linear2_residual (30) | n/a |
| 330 | policy.gpool_to_pass_matmul2 | library_gemm | library_gemm; gemmSN_NN_kernel; g1x2x1; b256x1x1; r64; s21504 | 60 | n/a | 2.752 | n/a | 91.0% | 0.000 | linear2_residual (30) | n/a |
| 331 | value.v1_conv | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r118; s98304 | 60 | n/a | 15.168 | n/a | 93.4% | 0.000 | linear2_residual (30) | n/a |
| 332 | value.v1_norm_silu | head_elementwise | head_elementwise; applyCScaleBiasNHWCSiluHalfKernel; g1x181x13; b192x2x1; r16; s0 | 60 | n/a | 4.544 | n/a | 82.3% | 0.000 | library_gemm (30) | n/a |
| 333 | value.v1_half_to_float | copy_reformat | copy_reformat; copyFromHalfKernel; g1760x1x1; b512x1x1; r16; s0 | 60 | n/a | 3.264 | n/a | 81.9% | 0.000 | library_gemm (33) | n/a |
| 334 | value.v1_global_pool | head_elementwise | head_elementwise; valueHeadPoolChannelsNHWCKernel; g3x1x13; b64x8x1; r22; s2048 | 60 | n/a | 3.969 | n/a | 98.2% | 0.000 | library_gemm (59) | n/a |
| 335 | value.v2_matmul | library_gemm | library_gemm; gemmSN_NN_kernel; g3x2x1; b256x1x1; r64; s21504 | 60 | n/a | 15.008 | n/a | 92.6% | 0.000 | library_gemm (48) | n/a |
| 336 | value.v2_bias_silu | head_elementwise | head_elementwise; addCBiasInplaceNCKernelSilu; g1x7x1; b192x2x1; r16; s0 | 60 | n/a | 2.544 | n/a | 94.5% | 0.000 | library_gemm (30) | n/a |
| 337 | value.v3_matmul | library_gemm | library_gemm; gemmSN_NN_kernel; g1x2x1; b256x1x1; r64; s21504 | 60 | n/a | 5.392 | n/a | 94.8% | 0.000 | library_gemm (30) | n/a |
| 338 | value.v3_bias | head_elementwise | head_elementwise; addCBiasInplaceNCKernel; g1x1x1; b3x170x1; r16; s0 | 60 | n/a | 1.040 | n/a | 74.1% | 0.000 | head_elementwise (27) | n/a |
| 339 | value.score_matmul | library_gemm | library_gemm; gemmSN_NN_kernel; g1x2x1; b256x1x1; r64; s21504 | 60 | n/a | 4.256 | n/a | 88.5% | 0.000 | head_elementwise (28) | n/a |
| 340 | value.score_bias | head_elementwise | head_elementwise; addCBiasInplaceNCKernel; g1x1x1; b6x85x1; r16; s0 | 60 | n/a | 1.152 | n/a | 96.7% | 0.000 | head_elementwise (30) | n/a |
| 341 | value.ownership_conv | library_gemm | library_gemm; Kernel2; g8x19x3; b128x1x1; r118; s33792 | 60 | n/a | 4.752 | n/a | 71.9% | 0.000 | library_gemm (30) | n/a |
| 342 | value.ownership_conv_splitk_reduce | library_gemm | library_gemm; splitKreduce_kernel; g147x1x1; b32x16x1; r49; s0 | 60 | n/a | 1.760 | n/a | 97.3% | 0.000 | library_gemm (57) | n/a |
| 343 | value.ownership_half_to_float | copy_reformat | copy_reformat; copyFromHalfKernel; g10x1x1; b512x1x1; r16; s0 | 60 | n/a | 1.120 | n/a | 94.8% | 0.000 | library_gemm (55) | n/a |
