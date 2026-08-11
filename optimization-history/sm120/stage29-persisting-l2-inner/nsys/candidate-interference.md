# Nsys stream interference report

- Timed iterations: 30; streams: 65, 82
- Kernels per forward: 65=344, 82=344
- Iteration start offset stream 82 - 65: median 3.76 us, p10..p90 3.45..5.00 us, range 3.33..35.23 us.

## Kernel families

| family | instances | total ms | median us | p90 us | peer overlap | S2/S1 median | summed excess ms | matched |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| fused_ffn | 1980 | 88.655 | 44.288 | 48.740 | 79.0% | n/a | 0.000 | 0 |
| library_gemm | 4140 | 65.173 | 15.808 | 21.536 | 94.7% | n/a | 0.000 | 0 |
| linear2_residual | 1980 | 64.618 | 32.672 | 35.233 | 98.2% | n/a | 0.000 | 0 |
| wide_qkv | 1980 | 50.034 | 23.665 | 31.012 | 77.6% | n/a | 0.000 | 0 |
| fa4 | 1980 | 29.553 | 14.784 | 15.552 | 46.0% | n/a | 0.000 | 0 |
| rmsnorm | 3960 | 16.347 | 4.032 | 5.600 | 100.0% | n/a | 0.000 | 0 |
| qk_rope | 1980 | 14.845 | 7.392 | 9.504 | 98.4% | n/a | 0.000 | 0 |
| affine_silu | 1380 | 8.563 | 6.096 | 9.472 | 97.5% | n/a | 0.000 | 0 |
| head_elementwise | 720 | 2.282 | 2.496 | 8.064 | 76.5% | n/a | 0.000 | 0 |
| cudnn | 180 | 1.497 | 1.728 | 21.799 | 22.4% | n/a | 0.000 | 0 |
| copy_reformat | 300 | 0.552 | 1.728 | 2.528 | 63.5% | n/a | 0.000 | 0 |
| sumChannelsNCHWKernel | 60 | 0.118 | 1.872 | 2.208 | 42.8% | n/a | 0.000 | 0 |

## Dominant interference pairs

| running family | dominant peer | instances | median overlap | median us | S2/S1 median | matched |
|---|---|---:|---:|---:|---:|---:|
| library_gemm | library_gemm | 1661 | 96.5% | 14.368 | n/a | 0 |
| rmsnorm | library_gemm | 1312 | 100.0% | 2.736 | n/a | 0 |
| affine_silu | library_gemm | 1023 | 100.0% | 7.104 | n/a | 0 |
| fused_ffn | fused_ffn | 1022 | 69.4% | 47.521 | n/a | 0 |
| qk_rope | wide_qkv | 990 | 100.0% | 9.280 | n/a | 0 |
| linear2_residual | fused_ffn | 990 | 98.5% | 31.584 | n/a | 0 |
| rmsnorm | wide_qkv | 990 | 100.0% | 4.928 | n/a | 0 |
| wide_qkv | qk_rope | 990 | 64.9% | 21.985 | n/a | 0 |
| qk_rope | fa4 | 990 | 100.0% | 5.792 | n/a | 0 |
| rmsnorm | fused_ffn | 990 | 100.0% | 5.504 | n/a | 0 |
| library_gemm | fused_ffn | 963 | 95.1% | 20.896 | n/a | 0 |
| fused_ffn | linear2_residual | 958 | 88.5% | 42.688 | n/a | 0 |
| fa4 | qk_rope | 928 | 42.4% | 14.688 | n/a | 0 |
| fa4 | library_gemm | 928 | 44.6% | 14.784 | n/a | 0 |
| rmsnorm | linear2_residual | 660 | 100.0% | 3.264 | n/a | 0 |
| wide_qkv | linear2_residual | 660 | 96.4% | 30.384 | n/a | 0 |
| linear2_residual | wide_qkv | 660 | 97.8% | 33.728 | n/a | 0 |
| library_gemm | affine_silu | 580 | 94.7% | 18.944 | n/a | 0 |
| head_elementwise | library_gemm | 533 | 100.0% | 2.208 | n/a | 0 |
| affine_silu | linear2_residual | 330 | 100.0% | 4.160 | n/a | 0 |
| library_gemm | linear2_residual | 330 | 95.6% | 23.424 | n/a | 0 |
| linear2_residual | library_gemm | 330 | 97.7% | 32.176 | n/a | 0 |
| wide_qkv | library_gemm | 329 | 66.4% | 25.344 | n/a | 0 |
| library_gemm | wide_qkv | 326 | 93.4% | 15.457 | n/a | 0 |
| library_gemm | head_elementwise | 201 | 84.7% | 6.528 | n/a | 0 |
| copy_reformat | library_gemm | 147 | 74.3% | 1.824 | n/a | 0 |
| fa4 | fa4 | 124 | 65.0% | 16.337 | n/a | 0 |
| cudnn | cudnn | 103 | 65.3% | 1.728 | n/a | 0 |
| copy_reformat | idle | 68 | 0.0% | 1.424 | n/a | 0 |
| copy_reformat | head_elementwise | 66 | 80.1% | 2.256 | n/a | 0 |
| head_elementwise | head_elementwise | 57 | 94.3% | 2.560 | n/a | 0 |
| head_elementwise | copy_reformat | 43 | 79.3% | 3.808 | n/a | 0 |
| library_gemm | cudnn | 38 | 93.5% | 4.208 | n/a | 0 |
| library_gemm | copy_reformat | 36 | 61.4% | 2.224 | n/a | 0 |
| cudnn | library_gemm | 32 | 15.4% | 21.601 | n/a | 0 |
| head_elementwise | idle | 29 | 0.0% | 1.504 | n/a | 0 |
| sumChannelsNCHWKernel | cudnn | 29 | 40.3% | 2.016 | n/a | 0 |
| cudnn | sumChannelsNCHWKernel | 27 | 62.8% | 1.440 | n/a | 0 |
| sumChannelsNCHWKernel | head_elementwise | 25 | 48.4% | 1.856 | n/a | 0 |
| head_elementwise | sumChannelsNCHWKernel | 25 | 75.0% | 1.312 | n/a | 0 |

## Logical operation groups

Isolated reference total is the isolated median for each ordinal multiplied by its S2 call count; it is a normalized reference, not a second trace total.

| logical group | families | positions | calls | isolated reference ms | S2 total ms | total ratio | excess ms |
|---|---|---:|---:|---:|---:|---:|---:|
| transformer.ffn_linear1_gate_swiglu | fused_ffn | 33 | 1980 | 0.000 | 88.655 | n/a | 0.000 |
| transformer.ffn_linear2_residual | linear2_residual | 33 | 1980 | 0.000 | 64.618 | n/a | 0.000 |
| transformer.attention_qkv_projection | wide_qkv | 33 | 1980 | 0.000 | 50.034 | n/a | 0.000 |
| transformer.attention_out_projection_residual | library_gemm | 33 | 1980 | 0.000 | 34.892 | n/a | 0.000 |
| transformer.attention_fa4 | fa4 | 33 | 1980 | 0.000 | 29.553 | n/a | 0.000 |
| transformer.attention_qk_rope | qk_rope | 33 | 1980 | 0.000 | 14.845 | n/a | 0.000 |
| outer.post_projection_c384_to_c768_residual | library_gemm | 11 | 660 | 0.000 | 14.270 | n/a | 0.000 |
| outer.pre_projection_c768_to_c384 | library_gemm | 11 | 660 | 0.000 | 11.355 | n/a | 0.000 |
| transformer.attention_rmsnorm | rmsnorm | 33 | 1980 | 0.000 | 8.252 | n/a | 0.000 |
| transformer.ffn_rmsnorm | rmsnorm | 33 | 1980 | 0.000 | 8.095 | n/a | 0.000 |
| outer.pre_norm_silu | affine_silu | 11 | 660 | 0.000 | 5.386 | n/a | 0.000 |
| outer.post_norm_silu | affine_silu | 11 | 660 | 0.000 | 2.698 | n/a | 0.000 |
| frontend.initial_conv | cudnn | 1 | 60 | 0.000 | 1.300 | n/a | 0.000 |
| value.v2_matmul | library_gemm | 1 | 60 | 0.000 | 0.626 | n/a | 0.000 |
| policy.p1_conv | library_gemm | 1 | 60 | 0.000 | 0.615 | n/a | 0.000 |
| value.v1_conv | library_gemm | 1 | 60 | 0.000 | 0.567 | n/a | 0.000 |
| frontend.initial_global_broadcast_add | head_elementwise | 1 | 60 | 0.000 | 0.513 | n/a | 0.000 |
| trunk.tip_norm_silu | affine_silu | 1 | 60 | 0.000 | 0.478 | n/a | 0.000 |
| policy.g1_conv | library_gemm | 1 | 60 | 0.000 | 0.466 | n/a | 0.000 |
| policy.g1_global_pool | head_elementwise | 1 | 60 | 0.000 | 0.440 | n/a | 0.000 |
| policy.gpool_to_bias_matmul | library_gemm | 1 | 60 | 0.000 | 0.401 | n/a | 0.000 |
| policy.gpool_to_pass_matmul | library_gemm | 1 | 60 | 0.000 | 0.372 | n/a | 0.000 |
| policy.p2_conv | library_gemm | 1 | 60 | 0.000 | 0.292 | n/a | 0.000 |
| value.ownership_conv | library_gemm | 1 | 60 | 0.000 | 0.269 | n/a | 0.000 |
| value.v1_norm_silu | head_elementwise | 1 | 60 | 0.000 | 0.262 | n/a | 0.000 |
| value.score_matmul | library_gemm | 1 | 60 | 0.000 | 0.235 | n/a | 0.000 |
| value.v1_global_pool | head_elementwise | 1 | 60 | 0.000 | 0.234 | n/a | 0.000 |
| frontend.initial_global_matmul | library_gemm | 1 | 60 | 0.000 | 0.231 | n/a | 0.000 |
| value.v3_matmul | library_gemm | 1 | 60 | 0.000 | 0.229 | n/a | 0.000 |
| value.v1_half_to_float | copy_reformat | 1 | 60 | 0.000 | 0.183 | n/a | 0.000 |
| policy.gpool_to_pass_matmul2 | library_gemm | 1 | 60 | 0.000 | 0.174 | n/a | 0.000 |
| policy.g1_norm_silu | head_elementwise | 1 | 60 | 0.000 | 0.160 | n/a | 0.000 |
| policy.p1_norm_silu | head_elementwise | 1 | 60 | 0.000 | 0.156 | n/a | 0.000 |
| policy.gpool_bias_add | head_elementwise | 1 | 60 | 0.000 | 0.122 | n/a | 0.000 |
| input.mask_sum | sumChannelsNCHWKernel | 1 | 60 | 0.000 | 0.118 | n/a | 0.000 |
| policy.g1_half_to_float | copy_reformat | 1 | 60 | 0.000 | 0.114 | n/a | 0.000 |
| policy.pass_bias_silu | head_elementwise | 1 | 60 | 0.000 | 0.110 | n/a | 0.000 |
| policy.p1_half_to_float | copy_reformat | 1 | 60 | 0.000 | 0.105 | n/a | 0.000 |
| frontend.initial_conv_nhwc_padding_1 | cudnn | 1 | 60 | 0.000 | 0.101 | n/a | 0.000 |
| value.ownership_conv_splitk_reduce | library_gemm | 1 | 60 | 0.000 | 0.096 | n/a | 0.000 |
| frontend.initial_conv_nhwc_padding_0 | cudnn | 1 | 60 | 0.000 | 0.095 | n/a | 0.000 |
| input.extract_mask | head_elementwise | 1 | 60 | 0.000 | 0.094 | n/a | 0.000 |
| frontend.initial_global_matmul_splitk_reduce | library_gemm | 1 | 60 | 0.000 | 0.083 | n/a | 0.000 |
| input.mask_half_to_float | copy_reformat | 1 | 60 | 0.000 | 0.075 | n/a | 0.000 |
| value.ownership_half_to_float | copy_reformat | 1 | 60 | 0.000 | 0.075 | n/a | 0.000 |
| value.v2_bias_silu | head_elementwise | 1 | 60 | 0.000 | 0.065 | n/a | 0.000 |
| value.v3_bias | head_elementwise | 1 | 60 | 0.000 | 0.063 | n/a | 0.000 |
| value.score_bias | head_elementwise | 1 | 60 | 0.000 | 0.062 | n/a | 0.000 |

## `library_gemm` logical breakdown

| logical group | positions | calls | isolated reference ms | S2 total ms | total ratio | excess ms |
|---|---:|---:|---:|---:|---:|---:|
| transformer.attention_out_projection_residual | 33 | 1980 | 0.000 | 34.892 | n/a | 0.000 |
| outer.post_projection_c384_to_c768_residual | 11 | 660 | 0.000 | 14.270 | n/a | 0.000 |
| outer.pre_projection_c768_to_c384 | 11 | 660 | 0.000 | 11.355 | n/a | 0.000 |
| value.v2_matmul | 1 | 60 | 0.000 | 0.626 | n/a | 0.000 |
| policy.p1_conv | 1 | 60 | 0.000 | 0.615 | n/a | 0.000 |
| value.v1_conv | 1 | 60 | 0.000 | 0.567 | n/a | 0.000 |
| policy.g1_conv | 1 | 60 | 0.000 | 0.466 | n/a | 0.000 |
| policy.gpool_to_bias_matmul | 1 | 60 | 0.000 | 0.401 | n/a | 0.000 |
| policy.gpool_to_pass_matmul | 1 | 60 | 0.000 | 0.372 | n/a | 0.000 |
| policy.p2_conv | 1 | 60 | 0.000 | 0.292 | n/a | 0.000 |
| value.ownership_conv | 1 | 60 | 0.000 | 0.269 | n/a | 0.000 |
| value.score_matmul | 1 | 60 | 0.000 | 0.235 | n/a | 0.000 |
| frontend.initial_global_matmul | 1 | 60 | 0.000 | 0.231 | n/a | 0.000 |
| value.v3_matmul | 1 | 60 | 0.000 | 0.229 | n/a | 0.000 |
| policy.gpool_to_pass_matmul2 | 1 | 60 | 0.000 | 0.174 | n/a | 0.000 |
| value.ownership_conv_splitk_reduce | 1 | 60 | 0.000 | 0.096 | n/a | 0.000 |
| frontend.initial_global_matmul_splitk_reduce | 1 | 60 | 0.000 | 0.083 | n/a | 0.000 |

## Top ordinal hotspots by summed excess

The worst peer is the highest median S2/S1 slowdown among peer families observed at least four times for that ordinal.

| rank | ordinal | logical position | family | calls | isolated us | S2 us | S2/S1 | excess ms | common peer | worst peer |
|---:|---:|---|---|---:|---:|---:|---:|---:|---|---|
| 1 | 157 | outer_05.transformer_0.block_15.ffn_linear1_gate_swiglu | fused_ffn | 60 | n/a | 47.425 | n/a | 0.000 | fused_ffn (30) | n/a |
| 2 | 241 | outer_08.transformer_0.block_24.ffn_linear1_gate_swiglu | fused_ffn | 60 | n/a | 46.753 | n/a | 0.000 | fused_ffn (33) | n/a |
| 3 | 249 | outer_08.transformer_1.block_25.ffn_linear1_gate_swiglu | fused_ffn | 60 | n/a | 46.992 | n/a | 0.000 | fused_ffn (30) | n/a |
| 4 | 81 | outer_02.transformer_1.block_07.ffn_linear1_gate_swiglu | fused_ffn | 60 | n/a | 46.480 | n/a | 0.000 | fused_ffn (31) | n/a |
| 5 | 313 | outer_10.transformer_2.block_32.ffn_linear1_gate_swiglu | fused_ffn | 60 | n/a | 47.184 | n/a | 0.000 | fused_ffn (30) | n/a |
| 6 | 73 | outer_02.transformer_0.block_06.ffn_linear1_gate_swiglu | fused_ffn | 60 | n/a | 46.977 | n/a | 0.000 | fused_ffn (30) | n/a |
| 7 | 185 | outer_06.transformer_0.block_18.ffn_linear1_gate_swiglu | fused_ffn | 60 | n/a | 46.448 | n/a | 0.000 | fused_ffn (30) | n/a |
| 8 | 269 | outer_09.transformer_0.block_27.ffn_linear1_gate_swiglu | fused_ffn | 60 | n/a | 46.225 | n/a | 0.000 | fused_ffn (30) | n/a |
| 9 | 193 | outer_06.transformer_1.block_19.ffn_linear1_gate_swiglu | fused_ffn | 60 | n/a | 46.305 | n/a | 0.000 | fused_ffn (30) | n/a |
| 10 | 173 | outer_05.transformer_2.block_17.ffn_linear1_gate_swiglu | fused_ffn | 60 | n/a | 45.600 | n/a | 0.000 | fused_ffn (32) | n/a |
| 11 | 89 | outer_02.transformer_2.block_08.ffn_linear1_gate_swiglu | fused_ffn | 60 | n/a | 46.065 | n/a | 0.000 | fused_ffn (30) | n/a |
| 12 | 257 | outer_08.transformer_2.block_26.ffn_linear1_gate_swiglu | fused_ffn | 60 | n/a | 45.792 | n/a | 0.000 | fused_ffn (30) | n/a |
| 13 | 165 | outer_05.transformer_1.block_16.ffn_linear1_gate_swiglu | fused_ffn | 60 | n/a | 45.536 | n/a | 0.000 | fused_ffn (33) | n/a |
| 14 | 61 | outer_01.transformer_2.block_05.ffn_linear1_gate_swiglu | fused_ffn | 60 | n/a | 45.825 | n/a | 0.000 | fused_ffn (32) | n/a |
| 15 | 53 | outer_01.transformer_1.block_04.ffn_linear1_gate_swiglu | fused_ffn | 60 | n/a | 45.584 | n/a | 0.000 | fused_ffn (31) | n/a |
| 16 | 33 | outer_00.transformer_2.block_02.ffn_linear1_gate_swiglu | fused_ffn | 60 | n/a | 45.056 | n/a | 0.000 | fused_ffn (34) | n/a |
| 17 | 101 | outer_03.transformer_0.block_09.ffn_linear1_gate_swiglu | fused_ffn | 60 | n/a | 45.409 | n/a | 0.000 | fused_ffn (36) | n/a |
| 18 | 277 | outer_09.transformer_1.block_28.ffn_linear1_gate_swiglu | fused_ffn | 60 | n/a | 45.425 | n/a | 0.000 | fused_ffn (31) | n/a |
| 19 | 25 | outer_00.transformer_1.block_01.ffn_linear1_gate_swiglu | fused_ffn | 60 | n/a | 45.089 | n/a | 0.000 | fused_ffn (31) | n/a |
| 20 | 285 | outer_09.transformer_2.block_29.ffn_linear1_gate_swiglu | fused_ffn | 60 | n/a | 44.865 | n/a | 0.000 | fused_ffn (34) | n/a |
| 21 | 45 | outer_01.transformer_0.block_03.ffn_linear1_gate_swiglu | fused_ffn | 60 | n/a | 45.184 | n/a | 0.000 | fused_ffn (30) | n/a |
| 22 | 17 | outer_00.transformer_0.block_00.ffn_linear1_gate_swiglu | fused_ffn | 60 | n/a | 44.608 | n/a | 0.000 | fused_ffn (30) | n/a |
| 23 | 117 | outer_03.transformer_2.block_11.ffn_linear1_gate_swiglu | fused_ffn | 60 | n/a | 44.257 | n/a | 0.000 | fused_ffn (32) | n/a |
| 24 | 109 | outer_03.transformer_1.block_10.ffn_linear1_gate_swiglu | fused_ffn | 60 | n/a | 44.913 | n/a | 0.000 | fused_ffn (31) | n/a |
| 25 | 137 | outer_04.transformer_1.block_13.ffn_linear1_gate_swiglu | fused_ffn | 60 | n/a | 44.449 | n/a | 0.000 | fused_ffn (30) | n/a |
| 26 | 305 | outer_10.transformer_1.block_31.ffn_linear1_gate_swiglu | fused_ffn | 60 | n/a | 43.984 | n/a | 0.000 | fused_ffn (30) | n/a |
| 27 | 129 | outer_04.transformer_0.block_12.ffn_linear1_gate_swiglu | fused_ffn | 60 | n/a | 43.505 | n/a | 0.000 | fused_ffn (31) | n/a |
| 28 | 221 | outer_07.transformer_1.block_22.ffn_linear1_gate_swiglu | fused_ffn | 60 | n/a | 42.176 | n/a | 0.000 | fused_ffn (30) | n/a |
| 29 | 213 | outer_07.transformer_0.block_21.ffn_linear1_gate_swiglu | fused_ffn | 60 | n/a | 42.049 | n/a | 0.000 | fused_ffn (30) | n/a |
| 30 | 297 | outer_10.transformer_0.block_30.ffn_linear1_gate_swiglu | fused_ffn | 60 | n/a | 42.145 | n/a | 0.000 | fused_ffn (30) | n/a |
| 31 | 229 | outer_07.transformer_2.block_23.ffn_linear1_gate_swiglu | fused_ffn | 60 | n/a | 41.809 | n/a | 0.000 | fused_ffn (30) | n/a |
| 32 | 145 | outer_04.transformer_2.block_14.ffn_linear1_gate_swiglu | fused_ffn | 60 | n/a | 41.281 | n/a | 0.000 | fused_ffn (30) | n/a |
| 33 | 201 | outer_06.transformer_2.block_20.ffn_linear1_gate_swiglu | fused_ffn | 60 | n/a | 40.688 | n/a | 0.000 | fused_ffn (30) | n/a |
| 34 | 214 | outer_07.transformer_0.block_21.ffn_linear2_residual | linear2_residual | 60 | n/a | 35.920 | n/a | 0.000 | fused_ffn (30) | n/a |
| 35 | 222 | outer_07.transformer_1.block_22.ffn_linear2_residual | linear2_residual | 60 | n/a | 35.216 | n/a | 0.000 | fused_ffn (30) | n/a |
| 36 | 230 | outer_07.transformer_2.block_23.ffn_linear2_residual | linear2_residual | 60 | n/a | 35.104 | n/a | 0.000 | fused_ffn (30) | n/a |
| 37 | 202 | outer_06.transformer_2.block_20.ffn_linear2_residual | linear2_residual | 60 | n/a | 34.240 | n/a | 0.000 | fused_ffn (30) | n/a |
| 38 | 146 | outer_04.transformer_2.block_14.ffn_linear2_residual | linear2_residual | 60 | n/a | 33.904 | n/a | 0.000 | fused_ffn (30) | n/a |
| 39 | 298 | outer_10.transformer_0.block_30.ffn_linear2_residual | linear2_residual | 60 | n/a | 34.081 | n/a | 0.000 | fused_ffn (30) | n/a |
| 40 | 242 | outer_08.transformer_0.block_24.ffn_linear2_residual | linear2_residual | 60 | n/a | 33.840 | n/a | 0.000 | fused_ffn (30) | n/a |

## Full fixed-forward ordinal map

| ordinal | logical position | family | resource signature | calls | isolated us | S2 us | S2/S1 | overlap | excess ms | common peer | worst peer |
|---:|---|---|---|---:|---:|---:|---:|---:|---:|---|---|
| 0 | input.extract_mask | head_elementwise | head_elementwise; extractChannel0KernelNHWC; g10x1x1; b512x1x1; r16; s0 | 60 | n/a | 1.312 | n/a | 29.0% | 0.000 | idle (29) | n/a |
| 1 | input.mask_half_to_float | copy_reformat | copy_reformat; copyFromHalfKernel; g10x1x1; b512x1x1; r16; s0 | 60 | n/a | 1.088 | n/a | 13.8% | 0.000 | idle (41) | n/a |
| 2 | input.mask_sum | sumChannelsNCHWKernel | sumChannelsNCHWKernel; sumChannelsNCHWKernel; g1x1x13; b256x2x1; r22; s2048 | 60 | n/a | 1.872 | n/a | 42.8% | 0.000 | cudnn (29) | n/a |
| 3 | frontend.initial_conv_nhwc_padding_0 | cudnn | cudnn; nhwcAddPaddingKernel; g340x1x1; b768x1x1; r34; s0 | 60 | n/a | 1.520 | n/a | 67.0% | 0.000 | cudnn (28) | n/a |
| 4 | frontend.initial_conv_nhwc_padding_1 | cudnn | cudnn; nhwcAddPaddingKernel; g340x1x1; b768x1x1; r34; s0 | 60 | n/a | 1.664 | n/a | 59.2% | 0.000 | cudnn (46) | n/a |
| 5 | frontend.initial_conv | cudnn | cudnn; Kernel; g296x3x1; b128x1x1; r94; s81920 | 60 | n/a | 21.633 | n/a | 16.3% | 0.000 | cudnn (29) | n/a |
| 6 | frontend.initial_global_matmul | library_gemm | library_gemm; Kernel2; g8x1x3; b128x1x1; r128; s24576 | 60 | n/a | 3.744 | n/a | 88.1% | 0.000 | cudnn (30) | n/a |
| 7 | frontend.initial_global_matmul_splitk_reduce | library_gemm | library_gemm; splitKreduce_kernel; g24x1x1; b32x16x1; r49; s0 | 60 | n/a | 1.280 | n/a | 84.3% | 0.000 | library_gemm (23) | n/a |
| 8 | frontend.initial_global_broadcast_add | head_elementwise | head_elementwise; addNCBiasInplaceNHWCHalfKernel; g3x361x13; b256x1x1; r16; s0 | 60 | n/a | 8.096 | n/a | 35.3% | 0.000 | library_gemm (29) | n/a |
| 9 | outer_00.pre_norm_silu | affine_silu | affine_silu; affineSiluHalf2Kernel; g4693x1x1; b384x1x1; r17; s0 | 60 | n/a | 6.096 | n/a | 61.5% | 0.000 | library_gemm (33) | n/a |
| 10 | outer_00.pre_projection_c768_to_c384 | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 60 | n/a | 14.656 | n/a | 94.3% | 0.000 | wide_qkv (26) | n/a |
| 11 | outer_00.transformer_0.block_00.attention_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | n/a | 3.728 | n/a | 98.4% | 0.000 | wide_qkv (30) | n/a |
| 12 | outer_00.transformer_0.block_00.attention_qkv_projection | wide_qkv | wide_qkv; wide_qkv_kernel; g9x37x1; b128x1x1; r136; s65536 | 60 | n/a | 22.336 | n/a | 69.1% | 0.000 | library_gemm (30) | n/a |
| 13 | outer_00.transformer_0.block_00.attention_qk_rope | qk_rope | qk_rope; fusedQKRoPE19HalfKernel; g361x13x1; b192x1x1; r16; s0 | 60 | n/a | 7.232 | n/a | 97.3% | 0.000 | fa4 (30) | n/a |
| 14 | outer_00.transformer_0.block_00.attention_fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 60 | n/a | 14.464 | n/a | 49.5% | 0.000 | library_gemm (27) | n/a |
| 15 | outer_00.transformer_0.block_00.attention_out_projection_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 60 | n/a | 16.320 | n/a | 95.8% | 0.000 | library_gemm (31) | n/a |
| 16 | outer_00.transformer_0.block_00.ffn_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | n/a | 3.840 | n/a | 100.0% | 0.000 | fused_ffn (30) | n/a |
| 17 | outer_00.transformer_0.block_00.ffn_linear1_gate_swiglu | fused_ffn | fused_ffn; fused_ffn_kernel; g18x37x1; b128x1x1; r146; s32768 | 60 | n/a | 44.608 | n/a | 77.8% | 0.000 | fused_ffn (30) | n/a |
| 18 | outer_00.transformer_0.block_00.ffn_linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 60 | n/a | 32.144 | n/a | 98.1% | 0.000 | fused_ffn (30) | n/a |
| 19 | outer_00.transformer_1.block_01.attention_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | n/a | 3.728 | n/a | 100.0% | 0.000 | linear2_residual (30) | n/a |
| 20 | outer_00.transformer_1.block_01.attention_qkv_projection | wide_qkv | wide_qkv; wide_qkv_kernel; g9x37x1; b128x1x1; r136; s65536 | 60 | n/a | 25.584 | n/a | 83.3% | 0.000 | linear2_residual (30) | n/a |
| 21 | outer_00.transformer_1.block_01.attention_qk_rope | qk_rope | qk_rope; fusedQKRoPE19HalfKernel; g361x13x1; b192x1x1; r16; s0 | 60 | n/a | 7.441 | n/a | 99.2% | 0.000 | fa4 (30) | n/a |
| 22 | outer_00.transformer_1.block_01.attention_fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 60 | n/a | 14.624 | n/a | 44.3% | 0.000 | library_gemm (29) | n/a |
| 23 | outer_00.transformer_1.block_01.attention_out_projection_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 60 | n/a | 15.920 | n/a | 95.8% | 0.000 | library_gemm (31) | n/a |
| 24 | outer_00.transformer_1.block_01.ffn_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | n/a | 4.000 | n/a | 100.0% | 0.000 | fused_ffn (30) | n/a |
| 25 | outer_00.transformer_1.block_01.ffn_linear1_gate_swiglu | fused_ffn | fused_ffn; fused_ffn_kernel; g18x37x1; b128x1x1; r146; s32768 | 60 | n/a | 45.089 | n/a | 76.4% | 0.000 | fused_ffn (31) | n/a |
| 26 | outer_00.transformer_1.block_01.ffn_linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 60 | n/a | 31.280 | n/a | 98.3% | 0.000 | fused_ffn (30) | n/a |
| 27 | outer_00.transformer_2.block_02.attention_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | n/a | 3.648 | n/a | 100.0% | 0.000 | linear2_residual (30) | n/a |
| 28 | outer_00.transformer_2.block_02.attention_qkv_projection | wide_qkv | wide_qkv; wide_qkv_kernel; g9x37x1; b128x1x1; r136; s65536 | 60 | n/a | 24.768 | n/a | 83.1% | 0.000 | linear2_residual (30) | n/a |
| 29 | outer_00.transformer_2.block_02.attention_qk_rope | qk_rope | qk_rope; fusedQKRoPE19HalfKernel; g361x13x1; b192x1x1; r16; s0 | 60 | n/a | 7.392 | n/a | 96.6% | 0.000 | fa4 (30) | n/a |
| 30 | outer_00.transformer_2.block_02.attention_fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 60 | n/a | 14.464 | n/a | 46.9% | 0.000 | library_gemm (26) | n/a |
| 31 | outer_00.transformer_2.block_02.attention_out_projection_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 60 | n/a | 15.920 | n/a | 95.8% | 0.000 | library_gemm (32) | n/a |
| 32 | outer_00.transformer_2.block_02.ffn_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | n/a | 4.048 | n/a | 100.0% | 0.000 | fused_ffn (30) | n/a |
| 33 | outer_00.transformer_2.block_02.ffn_linear1_gate_swiglu | fused_ffn | fused_ffn; fused_ffn_kernel; g18x37x1; b128x1x1; r146; s32768 | 60 | n/a | 45.056 | n/a | 76.3% | 0.000 | fused_ffn (34) | n/a |
| 34 | outer_00.transformer_2.block_02.ffn_linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 60 | n/a | 30.480 | n/a | 98.2% | 0.000 | fused_ffn (30) | n/a |
| 35 | outer_00.post_norm_silu | affine_silu | affine_silu; affineSiluHalf2Kernel; g4693x1x1; b192x1x1; r17; s0 | 60 | n/a | 3.984 | n/a | 100.0% | 0.000 | library_gemm (30) | n/a |
| 36 | outer_00.post_projection_c384_to_c768_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b256x1x1; r154; s73728 | 60 | n/a | 19.600 | n/a | 95.2% | 0.000 | affine_silu (30) | n/a |
| 37 | outer_01.pre_norm_silu | affine_silu | affine_silu; affineSiluHalf2Kernel; g4693x1x1; b384x1x1; r17; s0 | 60 | n/a | 8.080 | n/a | 100.0% | 0.000 | library_gemm (60) | n/a |
| 38 | outer_01.pre_projection_c768_to_c384 | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 60 | n/a | 16.368 | n/a | 94.0% | 0.000 | wide_qkv (30) | n/a |
| 39 | outer_01.transformer_0.block_03.attention_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | n/a | 4.369 | n/a | 100.0% | 0.000 | library_gemm (30) | n/a |
| 40 | outer_01.transformer_0.block_03.attention_qkv_projection | wide_qkv | wide_qkv; wide_qkv_kernel; g9x37x1; b128x1x1; r136; s65536 | 60 | n/a | 23.233 | n/a | 65.5% | 0.000 | library_gemm (30) | n/a |
| 41 | outer_01.transformer_0.block_03.attention_qk_rope | qk_rope | qk_rope; fusedQKRoPE19HalfKernel; g361x13x1; b192x1x1; r16; s0 | 60 | n/a | 7.568 | n/a | 99.2% | 0.000 | fa4 (30) | n/a |
| 42 | outer_01.transformer_0.block_03.attention_fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 60 | n/a | 14.720 | n/a | 44.7% | 0.000 | library_gemm (29) | n/a |
| 43 | outer_01.transformer_0.block_03.attention_out_projection_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 60 | n/a | 16.944 | n/a | 95.9% | 0.000 | fused_ffn (30) | n/a |
| 44 | outer_01.transformer_0.block_03.ffn_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | n/a | 4.096 | n/a | 100.0% | 0.000 | fused_ffn (30) | n/a |
| 45 | outer_01.transformer_0.block_03.ffn_linear1_gate_swiglu | fused_ffn | fused_ffn; fused_ffn_kernel; g18x37x1; b128x1x1; r146; s32768 | 60 | n/a | 45.184 | n/a | 77.6% | 0.000 | fused_ffn (30) | n/a |
| 46 | outer_01.transformer_0.block_03.ffn_linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 60 | n/a | 32.448 | n/a | 98.1% | 0.000 | fused_ffn (30) | n/a |
| 47 | outer_01.transformer_1.block_04.attention_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | n/a | 3.744 | n/a | 100.0% | 0.000 | linear2_residual (30) | n/a |
| 48 | outer_01.transformer_1.block_04.attention_qkv_projection | wide_qkv | wide_qkv; wide_qkv_kernel; g9x37x1; b128x1x1; r136; s65536 | 60 | n/a | 25.648 | n/a | 83.2% | 0.000 | linear2_residual (30) | n/a |
| 49 | outer_01.transformer_1.block_04.attention_qk_rope | qk_rope | qk_rope; fusedQKRoPE19HalfKernel; g361x13x1; b192x1x1; r16; s0 | 60 | n/a | 7.360 | n/a | 100.0% | 0.000 | fa4 (30) | n/a |
| 50 | outer_01.transformer_1.block_04.attention_fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 60 | n/a | 14.720 | n/a | 45.4% | 0.000 | library_gemm (30) | n/a |
| 51 | outer_01.transformer_1.block_04.attention_out_projection_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 60 | n/a | 17.136 | n/a | 95.8% | 0.000 | fused_ffn (30) | n/a |
| 52 | outer_01.transformer_1.block_04.ffn_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | n/a | 4.112 | n/a | 100.0% | 0.000 | fused_ffn (30) | n/a |
| 53 | outer_01.transformer_1.block_04.ffn_linear1_gate_swiglu | fused_ffn | fused_ffn; fused_ffn_kernel; g18x37x1; b128x1x1; r146; s32768 | 60 | n/a | 45.584 | n/a | 76.4% | 0.000 | fused_ffn (31) | n/a |
| 54 | outer_01.transformer_1.block_04.ffn_linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 60 | n/a | 31.312 | n/a | 98.4% | 0.000 | fused_ffn (30) | n/a |
| 55 | outer_01.transformer_2.block_05.attention_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | n/a | 4.016 | n/a | 100.0% | 0.000 | linear2_residual (30) | n/a |
| 56 | outer_01.transformer_2.block_05.attention_qkv_projection | wide_qkv | wide_qkv; wide_qkv_kernel; g9x37x1; b128x1x1; r136; s65536 | 60 | n/a | 24.881 | n/a | 82.9% | 0.000 | linear2_residual (30) | n/a |
| 57 | outer_01.transformer_2.block_05.attention_qk_rope | qk_rope | qk_rope; fusedQKRoPE19HalfKernel; g361x13x1; b192x1x1; r16; s0 | 60 | n/a | 7.504 | n/a | 94.9% | 0.000 | fa4 (30) | n/a |
| 58 | outer_01.transformer_2.block_05.attention_fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 60 | n/a | 14.592 | n/a | 47.9% | 0.000 | library_gemm (24) | n/a |
| 59 | outer_01.transformer_2.block_05.attention_out_projection_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 60 | n/a | 15.936 | n/a | 95.7% | 0.000 | library_gemm (32) | n/a |
| 60 | outer_01.transformer_2.block_05.ffn_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | n/a | 4.240 | n/a | 100.0% | 0.000 | fused_ffn (30) | n/a |
| 61 | outer_01.transformer_2.block_05.ffn_linear1_gate_swiglu | fused_ffn | fused_ffn; fused_ffn_kernel; g18x37x1; b128x1x1; r146; s32768 | 60 | n/a | 45.825 | n/a | 76.0% | 0.000 | fused_ffn (32) | n/a |
| 62 | outer_01.transformer_2.block_05.ffn_linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 60 | n/a | 30.928 | n/a | 98.2% | 0.000 | fused_ffn (30) | n/a |
| 63 | outer_01.post_norm_silu | affine_silu | affine_silu; affineSiluHalf2Kernel; g4693x1x1; b192x1x1; r17; s0 | 60 | n/a | 4.032 | n/a | 100.0% | 0.000 | library_gemm (30) | n/a |
| 64 | outer_01.post_projection_c384_to_c768_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b256x1x1; r154; s73728 | 60 | n/a | 20.497 | n/a | 95.2% | 0.000 | affine_silu (30) | n/a |
| 65 | outer_02.pre_norm_silu | affine_silu | affine_silu; affineSiluHalf2Kernel; g4693x1x1; b384x1x1; r17; s0 | 60 | n/a | 8.048 | n/a | 100.0% | 0.000 | library_gemm (60) | n/a |
| 66 | outer_02.pre_projection_c768_to_c384 | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 60 | n/a | 16.416 | n/a | 94.1% | 0.000 | wide_qkv (30) | n/a |
| 67 | outer_02.transformer_0.block_06.attention_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | n/a | 4.416 | n/a | 100.0% | 0.000 | library_gemm (30) | n/a |
| 68 | outer_02.transformer_0.block_06.attention_qkv_projection | wide_qkv | wide_qkv; wide_qkv_kernel; g9x37x1; b128x1x1; r136; s65536 | 60 | n/a | 23.456 | n/a | 65.7% | 0.000 | library_gemm (30) | n/a |
| 69 | outer_02.transformer_0.block_06.attention_qk_rope | qk_rope | qk_rope; fusedQKRoPE19HalfKernel; g361x13x1; b192x1x1; r16; s0 | 60 | n/a | 7.648 | n/a | 100.0% | 0.000 | fa4 (30) | n/a |
| 70 | outer_02.transformer_0.block_06.attention_fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 60 | n/a | 15.008 | n/a | 45.8% | 0.000 | library_gemm (30) | n/a |
| 71 | outer_02.transformer_0.block_06.attention_out_projection_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 60 | n/a | 17.552 | n/a | 95.9% | 0.000 | fused_ffn (30) | n/a |
| 72 | outer_02.transformer_0.block_06.ffn_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | n/a | 4.016 | n/a | 100.0% | 0.000 | fused_ffn (30) | n/a |
| 73 | outer_02.transformer_0.block_06.ffn_linear1_gate_swiglu | fused_ffn | fused_ffn; fused_ffn_kernel; g18x37x1; b128x1x1; r146; s32768 | 60 | n/a | 46.977 | n/a | 76.2% | 0.000 | fused_ffn (30) | n/a |
| 74 | outer_02.transformer_0.block_06.ffn_linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 60 | n/a | 32.496 | n/a | 98.1% | 0.000 | fused_ffn (30) | n/a |
| 75 | outer_02.transformer_1.block_07.attention_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | n/a | 4.016 | n/a | 100.0% | 0.000 | linear2_residual (30) | n/a |
| 76 | outer_02.transformer_1.block_07.attention_qkv_projection | wide_qkv | wide_qkv; wide_qkv_kernel; g9x37x1; b128x1x1; r136; s65536 | 60 | n/a | 25.456 | n/a | 83.3% | 0.000 | linear2_residual (30) | n/a |
| 77 | outer_02.transformer_1.block_07.attention_qk_rope | qk_rope | qk_rope; fusedQKRoPE19HalfKernel; g361x13x1; b192x1x1; r16; s0 | 60 | n/a | 7.489 | n/a | 99.2% | 0.000 | fa4 (30) | n/a |
| 78 | outer_02.transformer_1.block_07.attention_fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 60 | n/a | 14.656 | n/a | 44.8% | 0.000 | library_gemm (29) | n/a |
| 79 | outer_02.transformer_1.block_07.attention_out_projection_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 60 | n/a | 16.768 | n/a | 95.8% | 0.000 | fused_ffn (30) | n/a |
| 80 | outer_02.transformer_1.block_07.ffn_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | n/a | 3.888 | n/a | 100.0% | 0.000 | fused_ffn (30) | n/a |
| 81 | outer_02.transformer_1.block_07.ffn_linear1_gate_swiglu | fused_ffn | fused_ffn; fused_ffn_kernel; g18x37x1; b128x1x1; r146; s32768 | 60 | n/a | 46.480 | n/a | 78.5% | 0.000 | fused_ffn (31) | n/a |
| 82 | outer_02.transformer_1.block_07.ffn_linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 60 | n/a | 32.256 | n/a | 98.4% | 0.000 | fused_ffn (30) | n/a |
| 83 | outer_02.transformer_2.block_08.attention_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | n/a | 3.760 | n/a | 100.0% | 0.000 | linear2_residual (30) | n/a |
| 84 | outer_02.transformer_2.block_08.attention_qkv_projection | wide_qkv | wide_qkv; wide_qkv_kernel; g9x37x1; b128x1x1; r136; s65536 | 60 | n/a | 25.776 | n/a | 83.3% | 0.000 | linear2_residual (30) | n/a |
| 85 | outer_02.transformer_2.block_08.attention_qk_rope | qk_rope | qk_rope; fusedQKRoPE19HalfKernel; g361x13x1; b192x1x1; r16; s0 | 60 | n/a | 7.456 | n/a | 100.0% | 0.000 | fa4 (30) | n/a |
| 86 | outer_02.transformer_2.block_08.attention_fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 60 | n/a | 14.689 | n/a | 43.4% | 0.000 | library_gemm (30) | n/a |
| 87 | outer_02.transformer_2.block_08.attention_out_projection_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 60 | n/a | 17.536 | n/a | 95.8% | 0.000 | fused_ffn (30) | n/a |
| 88 | outer_02.transformer_2.block_08.ffn_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | n/a | 4.016 | n/a | 100.0% | 0.000 | fused_ffn (30) | n/a |
| 89 | outer_02.transformer_2.block_08.ffn_linear1_gate_swiglu | fused_ffn | fused_ffn; fused_ffn_kernel; g18x37x1; b128x1x1; r146; s32768 | 60 | n/a | 46.065 | n/a | 76.7% | 0.000 | fused_ffn (30) | n/a |
| 90 | outer_02.transformer_2.block_08.ffn_linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 60 | n/a | 31.648 | n/a | 98.2% | 0.000 | fused_ffn (30) | n/a |
| 91 | outer_02.post_norm_silu | affine_silu | affine_silu; affineSiluHalf2Kernel; g4693x1x1; b192x1x1; r17; s0 | 60 | n/a | 4.064 | n/a | 100.0% | 0.000 | library_gemm (30) | n/a |
| 92 | outer_02.post_projection_c384_to_c768_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b256x1x1; r154; s73728 | 60 | n/a | 20.689 | n/a | 95.1% | 0.000 | affine_silu (30) | n/a |
| 93 | outer_03.pre_norm_silu | affine_silu | affine_silu; affineSiluHalf2Kernel; g4693x1x1; b384x1x1; r17; s0 | 60 | n/a | 8.144 | n/a | 100.0% | 0.000 | library_gemm (60) | n/a |
| 94 | outer_03.pre_projection_c768_to_c384 | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 60 | n/a | 17.040 | n/a | 94.1% | 0.000 | wide_qkv (30) | n/a |
| 95 | outer_03.transformer_0.block_09.attention_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | n/a | 4.416 | n/a | 100.0% | 0.000 | library_gemm (30) | n/a |
| 96 | outer_03.transformer_0.block_09.attention_qkv_projection | wide_qkv | wide_qkv; wide_qkv_kernel; g9x37x1; b128x1x1; r136; s65536 | 60 | n/a | 23.872 | n/a | 65.5% | 0.000 | library_gemm (30) | n/a |
| 97 | outer_03.transformer_0.block_09.attention_qk_rope | qk_rope | qk_rope; fusedQKRoPE19HalfKernel; g361x13x1; b192x1x1; r16; s0 | 60 | n/a | 7.552 | n/a | 100.0% | 0.000 | fa4 (30) | n/a |
| 98 | outer_03.transformer_0.block_09.attention_fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 60 | n/a | 15.216 | n/a | 46.1% | 0.000 | library_gemm (30) | n/a |
| 99 | outer_03.transformer_0.block_09.attention_out_projection_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 60 | n/a | 17.968 | n/a | 95.9% | 0.000 | fused_ffn (30) | n/a |
| 100 | outer_03.transformer_0.block_09.ffn_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | n/a | 4.160 | n/a | 100.0% | 0.000 | fused_ffn (30) | n/a |
| 101 | outer_03.transformer_0.block_09.ffn_linear1_gate_swiglu | fused_ffn | fused_ffn; fused_ffn_kernel; g18x37x1; b128x1x1; r146; s32768 | 60 | n/a | 45.409 | n/a | 75.7% | 0.000 | fused_ffn (36) | n/a |
| 102 | outer_03.transformer_0.block_09.ffn_linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 60 | n/a | 31.360 | n/a | 98.0% | 0.000 | fused_ffn (30) | n/a |
| 103 | outer_03.transformer_1.block_10.attention_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | n/a | 4.016 | n/a | 100.0% | 0.000 | linear2_residual (30) | n/a |
| 104 | outer_03.transformer_1.block_10.attention_qkv_projection | wide_qkv | wide_qkv; wide_qkv_kernel; g9x37x1; b128x1x1; r136; s65536 | 60 | n/a | 24.401 | n/a | 83.2% | 0.000 | linear2_residual (30) | n/a |
| 105 | outer_03.transformer_1.block_10.attention_qk_rope | qk_rope | qk_rope; fusedQKRoPE19HalfKernel; g361x13x1; b192x1x1; r16; s0 | 60 | n/a | 7.296 | n/a | 97.5% | 0.000 | fa4 (30) | n/a |
| 106 | outer_03.transformer_1.block_10.attention_fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 60 | n/a | 14.336 | n/a | 46.2% | 0.000 | library_gemm (27) | n/a |
| 107 | outer_03.transformer_1.block_10.attention_out_projection_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 60 | n/a | 16.208 | n/a | 95.7% | 0.000 | library_gemm (31) | n/a |
| 108 | outer_03.transformer_1.block_10.ffn_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | n/a | 4.144 | n/a | 100.0% | 0.000 | fused_ffn (30) | n/a |
| 109 | outer_03.transformer_1.block_10.ffn_linear1_gate_swiglu | fused_ffn | fused_ffn; fused_ffn_kernel; g18x37x1; b128x1x1; r146; s32768 | 60 | n/a | 44.913 | n/a | 75.9% | 0.000 | fused_ffn (31) | n/a |
| 110 | outer_03.transformer_1.block_10.ffn_linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 60 | n/a | 30.544 | n/a | 98.4% | 0.000 | fused_ffn (30) | n/a |
| 111 | outer_03.transformer_2.block_11.attention_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | n/a | 3.776 | n/a | 100.0% | 0.000 | linear2_residual (30) | n/a |
| 112 | outer_03.transformer_2.block_11.attention_qkv_projection | wide_qkv | wide_qkv; wide_qkv_kernel; g9x37x1; b128x1x1; r136; s65536 | 60 | n/a | 24.209 | n/a | 82.9% | 0.000 | linear2_residual (30) | n/a |
| 113 | outer_03.transformer_2.block_11.attention_qk_rope | qk_rope | qk_rope; fusedQKRoPE19HalfKernel; g361x13x1; b192x1x1; r16; s0 | 60 | n/a | 7.296 | n/a | 94.8% | 0.000 | fa4 (30) | n/a |
| 114 | outer_03.transformer_2.block_11.attention_fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 60 | n/a | 14.368 | n/a | 48.8% | 0.000 | library_gemm (24) | n/a |
| 115 | outer_03.transformer_2.block_11.attention_out_projection_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 60 | n/a | 15.776 | n/a | 95.7% | 0.000 | library_gemm (32) | n/a |
| 116 | outer_03.transformer_2.block_11.ffn_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | n/a | 4.016 | n/a | 100.0% | 0.000 | fused_ffn (30) | n/a |
| 117 | outer_03.transformer_2.block_11.ffn_linear1_gate_swiglu | fused_ffn | fused_ffn; fused_ffn_kernel; g18x37x1; b128x1x1; r146; s32768 | 60 | n/a | 44.257 | n/a | 76.5% | 0.000 | fused_ffn (32) | n/a |
| 118 | outer_03.transformer_2.block_11.ffn_linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 60 | n/a | 29.888 | n/a | 98.2% | 0.000 | fused_ffn (30) | n/a |
| 119 | outer_03.post_norm_silu | affine_silu | affine_silu; affineSiluHalf2Kernel; g4693x1x1; b192x1x1; r17; s0 | 60 | n/a | 3.968 | n/a | 100.0% | 0.000 | library_gemm (30) | n/a |
| 120 | outer_03.post_projection_c384_to_c768_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b256x1x1; r154; s73728 | 60 | n/a | 19.488 | n/a | 95.1% | 0.000 | affine_silu (30) | n/a |
| 121 | outer_04.pre_norm_silu | affine_silu | affine_silu; affineSiluHalf2Kernel; g4693x1x1; b384x1x1; r17; s0 | 60 | n/a | 7.936 | n/a | 100.0% | 0.000 | library_gemm (60) | n/a |
| 122 | outer_04.pre_projection_c768_to_c384 | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 60 | n/a | 16.112 | n/a | 94.1% | 0.000 | wide_qkv (30) | n/a |
| 123 | outer_04.transformer_0.block_12.attention_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | n/a | 4.352 | n/a | 100.0% | 0.000 | library_gemm (30) | n/a |
| 124 | outer_04.transformer_0.block_12.attention_qkv_projection | wide_qkv | wide_qkv; wide_qkv_kernel; g9x37x1; b128x1x1; r136; s65536 | 60 | n/a | 22.704 | n/a | 65.4% | 0.000 | qk_rope (30) | n/a |
| 125 | outer_04.transformer_0.block_12.attention_qk_rope | qk_rope | qk_rope; fusedQKRoPE19HalfKernel; g361x13x1; b192x1x1; r16; s0 | 60 | n/a | 7.536 | n/a | 99.2% | 0.000 | fa4 (30) | n/a |
| 126 | outer_04.transformer_0.block_12.attention_fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 60 | n/a | 14.592 | n/a | 44.8% | 0.000 | library_gemm (29) | n/a |
| 127 | outer_04.transformer_0.block_12.attention_out_projection_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 60 | n/a | 16.128 | n/a | 95.9% | 0.000 | library_gemm (31) | n/a |
| 128 | outer_04.transformer_0.block_12.ffn_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | n/a | 4.256 | n/a | 100.0% | 0.000 | fused_ffn (30) | n/a |
| 129 | outer_04.transformer_0.block_12.ffn_linear1_gate_swiglu | fused_ffn | fused_ffn; fused_ffn_kernel; g18x37x1; b128x1x1; r146; s32768 | 60 | n/a | 43.505 | n/a | 79.2% | 0.000 | fused_ffn (31) | n/a |
| 130 | outer_04.transformer_0.block_12.ffn_linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 60 | n/a | 32.864 | n/a | 98.1% | 0.000 | fused_ffn (30) | n/a |
| 131 | outer_04.transformer_1.block_13.attention_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | n/a | 3.888 | n/a | 100.0% | 0.000 | linear2_residual (30) | n/a |
| 132 | outer_04.transformer_1.block_13.attention_qkv_projection | wide_qkv | wide_qkv; wide_qkv_kernel; g9x37x1; b128x1x1; r136; s65536 | 60 | n/a | 25.793 | n/a | 83.3% | 0.000 | linear2_residual (30) | n/a |
| 133 | outer_04.transformer_1.block_13.attention_qk_rope | qk_rope | qk_rope; fusedQKRoPE19HalfKernel; g361x13x1; b192x1x1; r16; s0 | 60 | n/a | 7.568 | n/a | 99.2% | 0.000 | fa4 (30) | n/a |
| 134 | outer_04.transformer_1.block_13.attention_fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 60 | n/a | 14.720 | n/a | 44.5% | 0.000 | library_gemm (29) | n/a |
| 135 | outer_04.transformer_1.block_13.attention_out_projection_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 60 | n/a | 17.504 | n/a | 95.8% | 0.000 | library_gemm (31) | n/a |
| 136 | outer_04.transformer_1.block_13.ffn_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | n/a | 3.632 | n/a | 100.0% | 0.000 | fused_ffn (30) | n/a |
| 137 | outer_04.transformer_1.block_13.ffn_linear1_gate_swiglu | fused_ffn | fused_ffn; fused_ffn_kernel; g18x37x1; b128x1x1; r146; s32768 | 60 | n/a | 44.449 | n/a | 82.3% | 0.000 | fused_ffn (30) | n/a |
| 138 | outer_04.transformer_1.block_13.ffn_linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 60 | n/a | 33.600 | n/a | 98.4% | 0.000 | fused_ffn (30) | n/a |
| 139 | outer_04.transformer_2.block_14.attention_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | n/a | 4.016 | n/a | 100.0% | 0.000 | linear2_residual (30) | n/a |
| 140 | outer_04.transformer_2.block_14.attention_qkv_projection | wide_qkv | wide_qkv; wide_qkv_kernel; g9x37x1; b128x1x1; r136; s65536 | 60 | n/a | 26.529 | n/a | 82.3% | 0.000 | linear2_residual (30) | n/a |
| 141 | outer_04.transformer_2.block_14.attention_qk_rope | qk_rope | qk_rope; fusedQKRoPE19HalfKernel; g361x13x1; b192x1x1; r16; s0 | 60 | n/a | 7.440 | n/a | 99.2% | 0.000 | fa4 (30) | n/a |
| 142 | outer_04.transformer_2.block_14.attention_fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 60 | n/a | 14.784 | n/a | 44.8% | 0.000 | library_gemm (29) | n/a |
| 143 | outer_04.transformer_2.block_14.attention_out_projection_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 60 | n/a | 16.992 | n/a | 95.7% | 0.000 | fused_ffn (30) | n/a |
| 144 | outer_04.transformer_2.block_14.ffn_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | n/a | 4.048 | n/a | 100.0% | 0.000 | fused_ffn (30) | n/a |
| 145 | outer_04.transformer_2.block_14.ffn_linear1_gate_swiglu | fused_ffn | fused_ffn; fused_ffn_kernel; g18x37x1; b128x1x1; r146; s32768 | 60 | n/a | 41.281 | n/a | 82.5% | 0.000 | fused_ffn (30) | n/a |
| 146 | outer_04.transformer_2.block_14.ffn_linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 60 | n/a | 33.904 | n/a | 98.3% | 0.000 | fused_ffn (30) | n/a |
| 147 | outer_04.post_norm_silu | affine_silu | affine_silu; affineSiluHalf2Kernel; g4693x1x1; b192x1x1; r17; s0 | 60 | n/a | 4.320 | n/a | 100.0% | 0.000 | library_gemm (30) | n/a |
| 148 | outer_04.post_projection_c384_to_c768_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b256x1x1; r154; s73728 | 60 | n/a | 21.281 | n/a | 95.3% | 0.000 | affine_silu (30) | n/a |
| 149 | outer_05.pre_norm_silu | affine_silu | affine_silu; affineSiluHalf2Kernel; g4693x1x1; b384x1x1; r17; s0 | 60 | n/a | 8.320 | n/a | 100.0% | 0.000 | library_gemm (60) | n/a |
| 150 | outer_05.pre_projection_c768_to_c384 | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 60 | n/a | 17.280 | n/a | 94.5% | 0.000 | wide_qkv (30) | n/a |
| 151 | outer_05.transformer_0.block_15.attention_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | n/a | 4.384 | n/a | 100.0% | 0.000 | library_gemm (30) | n/a |
| 152 | outer_05.transformer_0.block_15.attention_qkv_projection | wide_qkv | wide_qkv; wide_qkv_kernel; g9x37x1; b128x1x1; r136; s65536 | 60 | n/a | 24.432 | n/a | 65.4% | 0.000 | library_gemm (30) | n/a |
| 153 | outer_05.transformer_0.block_15.attention_qk_rope | qk_rope | qk_rope; fusedQKRoPE19HalfKernel; g361x13x1; b192x1x1; r16; s0 | 60 | n/a | 7.648 | n/a | 99.2% | 0.000 | fa4 (30) | n/a |
| 154 | outer_05.transformer_0.block_15.attention_fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 60 | n/a | 15.296 | n/a | 47.5% | 0.000 | library_gemm (29) | n/a |
| 155 | outer_05.transformer_0.block_15.attention_out_projection_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 60 | n/a | 17.776 | n/a | 95.9% | 0.000 | library_gemm (31) | n/a |
| 156 | outer_05.transformer_0.block_15.ffn_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | n/a | 4.144 | n/a | 100.0% | 0.000 | fused_ffn (30) | n/a |
| 157 | outer_05.transformer_0.block_15.ffn_linear1_gate_swiglu | fused_ffn | fused_ffn; fused_ffn_kernel; g18x37x1; b128x1x1; r146; s32768 | 60 | n/a | 47.425 | n/a | 78.4% | 0.000 | fused_ffn (30) | n/a |
| 158 | outer_05.transformer_0.block_15.ffn_linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 60 | n/a | 32.576 | n/a | 98.0% | 0.000 | fused_ffn (30) | n/a |
| 159 | outer_05.transformer_1.block_16.attention_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | n/a | 4.048 | n/a | 100.0% | 0.000 | linear2_residual (30) | n/a |
| 160 | outer_05.transformer_1.block_16.attention_qkv_projection | wide_qkv | wide_qkv; wide_qkv_kernel; g9x37x1; b128x1x1; r136; s65536 | 60 | n/a | 25.808 | n/a | 82.8% | 0.000 | linear2_residual (30) | n/a |
| 161 | outer_05.transformer_1.block_16.attention_qk_rope | qk_rope | qk_rope; fusedQKRoPE19HalfKernel; g361x13x1; b192x1x1; r16; s0 | 60 | n/a | 7.520 | n/a | 99.2% | 0.000 | fa4 (30) | n/a |
| 162 | outer_05.transformer_1.block_16.attention_fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 60 | n/a | 14.720 | n/a | 44.6% | 0.000 | library_gemm (29) | n/a |
| 163 | outer_05.transformer_1.block_16.attention_out_projection_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 60 | n/a | 17.056 | n/a | 95.7% | 0.000 | fused_ffn (30) | n/a |
| 164 | outer_05.transformer_1.block_16.ffn_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | n/a | 4.048 | n/a | 100.0% | 0.000 | fused_ffn (30) | n/a |
| 165 | outer_05.transformer_1.block_16.ffn_linear1_gate_swiglu | fused_ffn | fused_ffn; fused_ffn_kernel; g18x37x1; b128x1x1; r146; s32768 | 60 | n/a | 45.536 | n/a | 76.5% | 0.000 | fused_ffn (33) | n/a |
| 166 | outer_05.transformer_1.block_16.ffn_linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 60 | n/a | 32.096 | n/a | 98.4% | 0.000 | fused_ffn (30) | n/a |
| 167 | outer_05.transformer_2.block_17.attention_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | n/a | 3.984 | n/a | 100.0% | 0.000 | linear2_residual (30) | n/a |
| 168 | outer_05.transformer_2.block_17.attention_qkv_projection | wide_qkv | wide_qkv; wide_qkv_kernel; g9x37x1; b128x1x1; r136; s65536 | 60 | n/a | 24.960 | n/a | 83.4% | 0.000 | linear2_residual (30) | n/a |
| 169 | outer_05.transformer_2.block_17.attention_qk_rope | qk_rope | qk_rope; fusedQKRoPE19HalfKernel; g361x13x1; b192x1x1; r16; s0 | 60 | n/a | 7.408 | n/a | 93.1% | 0.000 | fa4 (30) | n/a |
| 170 | outer_05.transformer_2.block_17.attention_fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 60 | n/a | 14.625 | n/a | 49.5% | 0.000 | library_gemm (22) | n/a |
| 171 | outer_05.transformer_2.block_17.attention_out_projection_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 60 | n/a | 15.344 | n/a | 95.7% | 0.000 | library_gemm (34) | n/a |
| 172 | outer_05.transformer_2.block_17.ffn_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | n/a | 4.096 | n/a | 100.0% | 0.000 | fused_ffn (30) | n/a |
| 173 | outer_05.transformer_2.block_17.ffn_linear1_gate_swiglu | fused_ffn | fused_ffn; fused_ffn_kernel; g18x37x1; b128x1x1; r146; s32768 | 60 | n/a | 45.600 | n/a | 76.3% | 0.000 | fused_ffn (32) | n/a |
| 174 | outer_05.transformer_2.block_17.ffn_linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 60 | n/a | 30.624 | n/a | 98.2% | 0.000 | fused_ffn (30) | n/a |
| 175 | outer_05.post_norm_silu | affine_silu | affine_silu; affineSiluHalf2Kernel; g4693x1x1; b192x1x1; r17; s0 | 60 | n/a | 4.032 | n/a | 100.0% | 0.000 | library_gemm (30) | n/a |
| 176 | outer_05.post_projection_c384_to_c768_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b256x1x1; r154; s73728 | 60 | n/a | 20.241 | n/a | 95.2% | 0.000 | affine_silu (30) | n/a |
| 177 | outer_06.pre_norm_silu | affine_silu | affine_silu; affineSiluHalf2Kernel; g4693x1x1; b384x1x1; r17; s0 | 60 | n/a | 8.320 | n/a | 100.0% | 0.000 | library_gemm (60) | n/a |
| 178 | outer_06.pre_projection_c768_to_c384 | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 60 | n/a | 16.128 | n/a | 94.1% | 0.000 | wide_qkv (30) | n/a |
| 179 | outer_06.transformer_0.block_18.attention_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | n/a | 4.048 | n/a | 100.0% | 0.000 | library_gemm (30) | n/a |
| 180 | outer_06.transformer_0.block_18.attention_qkv_projection | wide_qkv | wide_qkv; wide_qkv_kernel; g9x37x1; b128x1x1; r136; s65536 | 60 | n/a | 24.208 | n/a | 65.7% | 0.000 | library_gemm (30) | n/a |
| 181 | outer_06.transformer_0.block_18.attention_qk_rope | qk_rope | qk_rope; fusedQKRoPE19HalfKernel; g361x13x1; b192x1x1; r16; s0 | 60 | n/a | 7.296 | n/a | 100.0% | 0.000 | fa4 (30) | n/a |
| 182 | outer_06.transformer_0.block_18.attention_fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 60 | n/a | 14.896 | n/a | 44.7% | 0.000 | library_gemm (30) | n/a |
| 183 | outer_06.transformer_0.block_18.attention_out_projection_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 60 | n/a | 17.504 | n/a | 95.8% | 0.000 | fused_ffn (30) | n/a |
| 184 | outer_06.transformer_0.block_18.ffn_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | n/a | 3.840 | n/a | 100.0% | 0.000 | fused_ffn (30) | n/a |
| 185 | outer_06.transformer_0.block_18.ffn_linear1_gate_swiglu | fused_ffn | fused_ffn; fused_ffn_kernel; g18x37x1; b128x1x1; r146; s32768 | 60 | n/a | 46.448 | n/a | 79.2% | 0.000 | fused_ffn (30) | n/a |
| 186 | outer_06.transformer_0.block_18.ffn_linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 60 | n/a | 32.880 | n/a | 98.1% | 0.000 | fused_ffn (30) | n/a |
| 187 | outer_06.transformer_1.block_19.attention_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | n/a | 3.936 | n/a | 100.0% | 0.000 | linear2_residual (30) | n/a |
| 188 | outer_06.transformer_1.block_19.attention_qkv_projection | wide_qkv | wide_qkv; wide_qkv_kernel; g9x37x1; b128x1x1; r136; s65536 | 60 | n/a | 26.209 | n/a | 82.8% | 0.000 | linear2_residual (30) | n/a |
| 189 | outer_06.transformer_1.block_19.attention_qk_rope | qk_rope | qk_rope; fusedQKRoPE19HalfKernel; g361x13x1; b192x1x1; r16; s0 | 60 | n/a | 7.520 | n/a | 100.0% | 0.000 | fa4 (30) | n/a |
| 190 | outer_06.transformer_1.block_19.attention_fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 60 | n/a | 14.704 | n/a | 44.5% | 0.000 | library_gemm (30) | n/a |
| 191 | outer_06.transformer_1.block_19.attention_out_projection_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 60 | n/a | 17.488 | n/a | 95.8% | 0.000 | fused_ffn (30) | n/a |
| 192 | outer_06.transformer_1.block_19.ffn_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | n/a | 3.760 | n/a | 100.0% | 0.000 | fused_ffn (30) | n/a |
| 193 | outer_06.transformer_1.block_19.ffn_linear1_gate_swiglu | fused_ffn | fused_ffn; fused_ffn_kernel; g18x37x1; b128x1x1; r146; s32768 | 60 | n/a | 46.305 | n/a | 82.6% | 0.000 | fused_ffn (30) | n/a |
| 194 | outer_06.transformer_1.block_19.ffn_linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 60 | n/a | 33.120 | n/a | 98.4% | 0.000 | fused_ffn (30) | n/a |
| 195 | outer_06.transformer_2.block_20.attention_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | n/a | 4.160 | n/a | 100.0% | 0.000 | linear2_residual (30) | n/a |
| 196 | outer_06.transformer_2.block_20.attention_qkv_projection | wide_qkv | wide_qkv; wide_qkv_kernel; g9x37x1; b128x1x1; r136; s65536 | 60 | n/a | 26.768 | n/a | 80.3% | 0.000 | linear2_residual (30) | n/a |
| 197 | outer_06.transformer_2.block_20.attention_qk_rope | qk_rope | qk_rope; fusedQKRoPE19HalfKernel; g361x13x1; b192x1x1; r16; s0 | 60 | n/a | 7.504 | n/a | 100.0% | 0.000 | fa4 (30) | n/a |
| 198 | outer_06.transformer_2.block_20.attention_fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 60 | n/a | 14.752 | n/a | 43.4% | 0.000 | library_gemm (30) | n/a |
| 199 | outer_06.transformer_2.block_20.attention_out_projection_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 60 | n/a | 17.632 | n/a | 95.8% | 0.000 | fused_ffn (30) | n/a |
| 200 | outer_06.transformer_2.block_20.ffn_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | n/a | 3.856 | n/a | 100.0% | 0.000 | fused_ffn (30) | n/a |
| 201 | outer_06.transformer_2.block_20.ffn_linear1_gate_swiglu | fused_ffn | fused_ffn; fused_ffn_kernel; g18x37x1; b128x1x1; r146; s32768 | 60 | n/a | 40.688 | n/a | 83.3% | 0.000 | fused_ffn (30) | n/a |
| 202 | outer_06.transformer_2.block_20.ffn_linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 60 | n/a | 34.240 | n/a | 98.3% | 0.000 | fused_ffn (30) | n/a |
| 203 | outer_06.post_norm_silu | affine_silu | affine_silu; affineSiluHalf2Kernel; g4693x1x1; b192x1x1; r17; s0 | 60 | n/a | 4.336 | n/a | 100.0% | 0.000 | library_gemm (30) | n/a |
| 204 | outer_06.post_projection_c384_to_c768_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b256x1x1; r154; s73728 | 60 | n/a | 21.808 | n/a | 95.3% | 0.000 | affine_silu (30) | n/a |
| 205 | outer_07.pre_norm_silu | affine_silu | affine_silu; affineSiluHalf2Kernel; g4693x1x1; b384x1x1; r17; s0 | 60 | n/a | 8.464 | n/a | 100.0% | 0.000 | library_gemm (60) | n/a |
| 206 | outer_07.pre_projection_c768_to_c384 | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 60 | n/a | 17.856 | n/a | 94.5% | 0.000 | wide_qkv (30) | n/a |
| 207 | outer_07.transformer_0.block_21.attention_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | n/a | 4.448 | n/a | 100.0% | 0.000 | library_gemm (30) | n/a |
| 208 | outer_07.transformer_0.block_21.attention_qkv_projection | wide_qkv | wide_qkv; wide_qkv_kernel; g9x37x1; b128x1x1; r136; s65536 | 60 | n/a | 24.224 | n/a | 65.2% | 0.000 | library_gemm (30) | n/a |
| 209 | outer_07.transformer_0.block_21.attention_qk_rope | qk_rope | qk_rope; fusedQKRoPE19HalfKernel; g361x13x1; b192x1x1; r16; s0 | 60 | n/a | 7.632 | n/a | 100.0% | 0.000 | fa4 (30) | n/a |
| 210 | outer_07.transformer_0.block_21.attention_fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 60 | n/a | 15.280 | n/a | 45.8% | 0.000 | library_gemm (30) | n/a |
| 211 | outer_07.transformer_0.block_21.attention_out_projection_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 60 | n/a | 17.936 | n/a | 95.7% | 0.000 | library_gemm (32) | n/a |
| 212 | outer_07.transformer_0.block_21.ffn_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | n/a | 3.985 | n/a | 100.0% | 0.000 | fused_ffn (30) | n/a |
| 213 | outer_07.transformer_0.block_21.ffn_linear1_gate_swiglu | fused_ffn | fused_ffn; fused_ffn_kernel; g18x37x1; b128x1x1; r146; s32768 | 60 | n/a | 42.049 | n/a | 83.7% | 0.000 | fused_ffn (30) | n/a |
| 214 | outer_07.transformer_0.block_21.ffn_linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 60 | n/a | 35.920 | n/a | 98.2% | 0.000 | fused_ffn (30) | n/a |
| 215 | outer_07.transformer_1.block_22.attention_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | n/a | 3.664 | n/a | 100.0% | 0.000 | linear2_residual (30) | n/a |
| 216 | outer_07.transformer_1.block_22.attention_qkv_projection | wide_qkv | wide_qkv; wide_qkv_kernel; g9x37x1; b128x1x1; r136; s65536 | 60 | n/a | 27.041 | n/a | 83.8% | 0.000 | linear2_residual (30) | n/a |
| 217 | outer_07.transformer_1.block_22.attention_qk_rope | qk_rope | qk_rope; fusedQKRoPE19HalfKernel; g361x13x1; b192x1x1; r16; s0 | 60 | n/a | 7.696 | n/a | 99.1% | 0.000 | fa4 (30) | n/a |
| 218 | outer_07.transformer_1.block_22.attention_fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 60 | n/a | 14.944 | n/a | 45.5% | 0.000 | library_gemm (29) | n/a |
| 219 | outer_07.transformer_1.block_22.attention_out_projection_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 60 | n/a | 17.392 | n/a | 95.7% | 0.000 | fused_ffn (30) | n/a |
| 220 | outer_07.transformer_1.block_22.ffn_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | n/a | 3.840 | n/a | 100.0% | 0.000 | fused_ffn (30) | n/a |
| 221 | outer_07.transformer_1.block_22.ffn_linear1_gate_swiglu | fused_ffn | fused_ffn; fused_ffn_kernel; g18x37x1; b128x1x1; r146; s32768 | 60 | n/a | 42.176 | n/a | 84.3% | 0.000 | fused_ffn (30) | n/a |
| 222 | outer_07.transformer_1.block_22.ffn_linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 60 | n/a | 35.216 | n/a | 98.5% | 0.000 | fused_ffn (30) | n/a |
| 223 | outer_07.transformer_2.block_23.attention_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | n/a | 4.000 | n/a | 100.0% | 0.000 | linear2_residual (30) | n/a |
| 224 | outer_07.transformer_2.block_23.attention_qkv_projection | wide_qkv | wide_qkv; wide_qkv_kernel; g9x37x1; b128x1x1; r136; s65536 | 60 | n/a | 26.928 | n/a | 83.1% | 0.000 | linear2_residual (30) | n/a |
| 225 | outer_07.transformer_2.block_23.attention_qk_rope | qk_rope | qk_rope; fusedQKRoPE19HalfKernel; g361x13x1; b192x1x1; r16; s0 | 60 | n/a | 7.553 | n/a | 99.1% | 0.000 | fa4 (30) | n/a |
| 226 | outer_07.transformer_2.block_23.attention_fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 60 | n/a | 14.976 | n/a | 45.5% | 0.000 | library_gemm (29) | n/a |
| 227 | outer_07.transformer_2.block_23.attention_out_projection_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 60 | n/a | 17.377 | n/a | 95.8% | 0.000 | fused_ffn (30) | n/a |
| 228 | outer_07.transformer_2.block_23.ffn_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | n/a | 4.000 | n/a | 100.0% | 0.000 | fused_ffn (30) | n/a |
| 229 | outer_07.transformer_2.block_23.ffn_linear1_gate_swiglu | fused_ffn | fused_ffn; fused_ffn_kernel; g18x37x1; b128x1x1; r146; s32768 | 60 | n/a | 41.809 | n/a | 84.4% | 0.000 | fused_ffn (30) | n/a |
| 230 | outer_07.transformer_2.block_23.ffn_linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 60 | n/a | 35.104 | n/a | 98.4% | 0.000 | fused_ffn (30) | n/a |
| 231 | outer_07.post_norm_silu | affine_silu | affine_silu; affineSiluHalf2Kernel; g4693x1x1; b192x1x1; r17; s0 | 60 | n/a | 4.320 | n/a | 100.0% | 0.000 | library_gemm (30) | n/a |
| 232 | outer_07.post_projection_c384_to_c768_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b256x1x1; r154; s73728 | 60 | n/a | 21.904 | n/a | 95.5% | 0.000 | affine_silu (30) | n/a |
| 233 | outer_08.pre_norm_silu | affine_silu | affine_silu; affineSiluHalf2Kernel; g4693x1x1; b384x1x1; r17; s0 | 60 | n/a | 8.512 | n/a | 100.0% | 0.000 | library_gemm (60) | n/a |
| 234 | outer_08.pre_projection_c768_to_c384 | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 60 | n/a | 17.760 | n/a | 94.4% | 0.000 | wide_qkv (30) | n/a |
| 235 | outer_08.transformer_0.block_24.attention_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | n/a | 4.384 | n/a | 100.0% | 0.000 | library_gemm (30) | n/a |
| 236 | outer_08.transformer_0.block_24.attention_qkv_projection | wide_qkv | wide_qkv; wide_qkv_kernel; g9x37x1; b128x1x1; r136; s65536 | 60 | n/a | 24.529 | n/a | 65.3% | 0.000 | library_gemm (30) | n/a |
| 237 | outer_08.transformer_0.block_24.attention_qk_rope | qk_rope | qk_rope; fusedQKRoPE19HalfKernel; g361x13x1; b192x1x1; r16; s0 | 60 | n/a | 7.664 | n/a | 100.0% | 0.000 | fa4 (30) | n/a |
| 238 | outer_08.transformer_0.block_24.attention_fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 60 | n/a | 15.392 | n/a | 47.1% | 0.000 | library_gemm (30) | n/a |
| 239 | outer_08.transformer_0.block_24.attention_out_projection_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 60 | n/a | 18.241 | n/a | 95.9% | 0.000 | library_gemm (31) | n/a |
| 240 | outer_08.transformer_0.block_24.ffn_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | n/a | 4.032 | n/a | 100.0% | 0.000 | fused_ffn (30) | n/a |
| 241 | outer_08.transformer_0.block_24.ffn_linear1_gate_swiglu | fused_ffn | fused_ffn; fused_ffn_kernel; g18x37x1; b128x1x1; r146; s32768 | 60 | n/a | 46.753 | n/a | 81.9% | 0.000 | fused_ffn (33) | n/a |
| 242 | outer_08.transformer_0.block_24.ffn_linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 60 | n/a | 33.840 | n/a | 98.1% | 0.000 | fused_ffn (30) | n/a |
| 243 | outer_08.transformer_1.block_25.attention_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | n/a | 3.888 | n/a | 100.0% | 0.000 | linear2_residual (30) | n/a |
| 244 | outer_08.transformer_1.block_25.attention_qkv_projection | wide_qkv | wide_qkv; wide_qkv_kernel; g9x37x1; b128x1x1; r136; s65536 | 60 | n/a | 26.896 | n/a | 82.7% | 0.000 | linear2_residual (30) | n/a |
| 245 | outer_08.transformer_1.block_25.attention_qk_rope | qk_rope | qk_rope; fusedQKRoPE19HalfKernel; g361x13x1; b192x1x1; r16; s0 | 60 | n/a | 7.504 | n/a | 99.2% | 0.000 | fa4 (30) | n/a |
| 246 | outer_08.transformer_1.block_25.attention_fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 60 | n/a | 14.816 | n/a | 45.0% | 0.000 | library_gemm (29) | n/a |
| 247 | outer_08.transformer_1.block_25.attention_out_projection_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 60 | n/a | 16.928 | n/a | 95.8% | 0.000 | fused_ffn (30) | n/a |
| 248 | outer_08.transformer_1.block_25.ffn_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | n/a | 4.000 | n/a | 100.0% | 0.000 | fused_ffn (30) | n/a |
| 249 | outer_08.transformer_1.block_25.ffn_linear1_gate_swiglu | fused_ffn | fused_ffn; fused_ffn_kernel; g18x37x1; b128x1x1; r146; s32768 | 60 | n/a | 46.992 | n/a | 79.3% | 0.000 | fused_ffn (30) | n/a |
| 250 | outer_08.transformer_1.block_25.ffn_linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 60 | n/a | 32.736 | n/a | 98.4% | 0.000 | fused_ffn (30) | n/a |
| 251 | outer_08.transformer_2.block_26.attention_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | n/a | 3.856 | n/a | 100.0% | 0.000 | linear2_residual (30) | n/a |
| 252 | outer_08.transformer_2.block_26.attention_qkv_projection | wide_qkv | wide_qkv; wide_qkv_kernel; g9x37x1; b128x1x1; r136; s65536 | 60 | n/a | 26.241 | n/a | 83.3% | 0.000 | linear2_residual (30) | n/a |
| 253 | outer_08.transformer_2.block_26.attention_qk_rope | qk_rope | qk_rope; fusedQKRoPE19HalfKernel; g361x13x1; b192x1x1; r16; s0 | 60 | n/a | 7.408 | n/a | 100.0% | 0.000 | fa4 (30) | n/a |
| 254 | outer_08.transformer_2.block_26.attention_fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 60 | n/a | 14.752 | n/a | 45.0% | 0.000 | library_gemm (30) | n/a |
| 255 | outer_08.transformer_2.block_26.attention_out_projection_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 60 | n/a | 17.280 | n/a | 95.8% | 0.000 | fused_ffn (30) | n/a |
| 256 | outer_08.transformer_2.block_26.ffn_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | n/a | 3.872 | n/a | 100.0% | 0.000 | fused_ffn (30) | n/a |
| 257 | outer_08.transformer_2.block_26.ffn_linear1_gate_swiglu | fused_ffn | fused_ffn; fused_ffn_kernel; g18x37x1; b128x1x1; r146; s32768 | 60 | n/a | 45.792 | n/a | 80.0% | 0.000 | fused_ffn (30) | n/a |
| 258 | outer_08.transformer_2.block_26.ffn_linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 60 | n/a | 32.608 | n/a | 98.3% | 0.000 | fused_ffn (30) | n/a |
| 259 | outer_08.post_norm_silu | affine_silu | affine_silu; affineSiluHalf2Kernel; g4693x1x1; b192x1x1; r17; s0 | 60 | n/a | 4.096 | n/a | 100.0% | 0.000 | library_gemm (30) | n/a |
| 260 | outer_08.post_projection_c384_to_c768_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b256x1x1; r154; s73728 | 60 | n/a | 22.544 | n/a | 95.6% | 0.000 | affine_silu (30) | n/a |
| 261 | outer_09.pre_norm_silu | affine_silu | affine_silu; affineSiluHalf2Kernel; g4693x1x1; b384x1x1; r17; s0 | 60 | n/a | 8.240 | n/a | 100.0% | 0.000 | library_gemm (60) | n/a |
| 262 | outer_09.pre_projection_c768_to_c384 | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 60 | n/a | 17.184 | n/a | 94.1% | 0.000 | affine_silu (30) | n/a |
| 263 | outer_09.transformer_0.block_27.attention_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | n/a | 4.624 | n/a | 100.0% | 0.000 | library_gemm (30) | n/a |
| 264 | outer_09.transformer_0.block_27.attention_qkv_projection | wide_qkv | wide_qkv; wide_qkv_kernel; g9x37x1; b128x1x1; r136; s65536 | 60 | n/a | 24.592 | n/a | 65.5% | 0.000 | library_gemm (30) | n/a |
| 265 | outer_09.transformer_0.block_27.attention_qk_rope | qk_rope | qk_rope; fusedQKRoPE19HalfKernel; g361x13x1; b192x1x1; r16; s0 | 60 | n/a | 7.616 | n/a | 99.2% | 0.000 | fa4 (30) | n/a |
| 266 | outer_09.transformer_0.block_27.attention_fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 60 | n/a | 15.264 | n/a | 47.0% | 0.000 | library_gemm (29) | n/a |
| 267 | outer_09.transformer_0.block_27.attention_out_projection_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 60 | n/a | 17.601 | n/a | 95.7% | 0.000 | library_gemm (31) | n/a |
| 268 | outer_09.transformer_0.block_27.ffn_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | n/a | 4.144 | n/a | 100.0% | 0.000 | fused_ffn (30) | n/a |
| 269 | outer_09.transformer_0.block_27.ffn_linear1_gate_swiglu | fused_ffn | fused_ffn; fused_ffn_kernel; g18x37x1; b128x1x1; r146; s32768 | 60 | n/a | 46.225 | n/a | 75.1% | 0.000 | fused_ffn (30) | n/a |
| 270 | outer_09.transformer_0.block_27.ffn_linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 60 | n/a | 31.248 | n/a | 98.0% | 0.000 | fused_ffn (30) | n/a |
| 271 | outer_09.transformer_1.block_28.attention_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | n/a | 3.904 | n/a | 100.0% | 0.000 | linear2_residual (30) | n/a |
| 272 | outer_09.transformer_1.block_28.attention_qkv_projection | wide_qkv | wide_qkv; wide_qkv_kernel; g9x37x1; b128x1x1; r136; s65536 | 60 | n/a | 24.673 | n/a | 83.3% | 0.000 | linear2_residual (30) | n/a |
| 273 | outer_09.transformer_1.block_28.attention_qk_rope | qk_rope | qk_rope; fusedQKRoPE19HalfKernel; g361x13x1; b192x1x1; r16; s0 | 60 | n/a | 7.536 | n/a | 94.1% | 0.000 | fa4 (30) | n/a |
| 274 | outer_09.transformer_1.block_28.attention_fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 60 | n/a | 14.528 | n/a | 48.7% | 0.000 | library_gemm (23) | n/a |
| 275 | outer_09.transformer_1.block_28.attention_out_projection_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 60 | n/a | 15.440 | n/a | 95.7% | 0.000 | library_gemm (34) | n/a |
| 276 | outer_09.transformer_1.block_28.ffn_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | n/a | 4.096 | n/a | 100.0% | 0.000 | fused_ffn (30) | n/a |
| 277 | outer_09.transformer_1.block_28.ffn_linear1_gate_swiglu | fused_ffn | fused_ffn; fused_ffn_kernel; g18x37x1; b128x1x1; r146; s32768 | 60 | n/a | 45.425 | n/a | 76.1% | 0.000 | fused_ffn (31) | n/a |
| 278 | outer_09.transformer_1.block_28.ffn_linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 60 | n/a | 30.672 | n/a | 98.4% | 0.000 | fused_ffn (30) | n/a |
| 279 | outer_09.transformer_2.block_29.attention_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | n/a | 3.889 | n/a | 100.0% | 0.000 | linear2_residual (30) | n/a |
| 280 | outer_09.transformer_2.block_29.attention_qkv_projection | wide_qkv | wide_qkv; wide_qkv_kernel; g9x37x1; b128x1x1; r136; s65536 | 60 | n/a | 24.416 | n/a | 82.8% | 0.000 | linear2_residual (30) | n/a |
| 281 | outer_09.transformer_2.block_29.attention_qk_rope | qk_rope | qk_rope; fusedQKRoPE19HalfKernel; g361x13x1; b192x1x1; r16; s0 | 60 | n/a | 7.296 | n/a | 94.9% | 0.000 | fa4 (30) | n/a |
| 282 | outer_09.transformer_2.block_29.attention_fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 60 | n/a | 14.400 | n/a | 48.7% | 0.000 | library_gemm (24) | n/a |
| 283 | outer_09.transformer_2.block_29.attention_out_projection_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 60 | n/a | 15.952 | n/a | 95.7% | 0.000 | library_gemm (31) | n/a |
| 284 | outer_09.transformer_2.block_29.ffn_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | n/a | 4.032 | n/a | 100.0% | 0.000 | fused_ffn (30) | n/a |
| 285 | outer_09.transformer_2.block_29.ffn_linear1_gate_swiglu | fused_ffn | fused_ffn; fused_ffn_kernel; g18x37x1; b128x1x1; r146; s32768 | 60 | n/a | 44.865 | n/a | 76.7% | 0.000 | fused_ffn (34) | n/a |
| 286 | outer_09.transformer_2.block_29.ffn_linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 60 | n/a | 30.144 | n/a | 98.2% | 0.000 | fused_ffn (30) | n/a |
| 287 | outer_09.post_norm_silu | affine_silu | affine_silu; affineSiluHalf2Kernel; g4693x1x1; b192x1x1; r17; s0 | 60 | n/a | 3.968 | n/a | 100.0% | 0.000 | library_gemm (30) | n/a |
| 288 | outer_09.post_projection_c384_to_c768_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b256x1x1; r154; s73728 | 60 | n/a | 19.761 | n/a | 95.3% | 0.000 | affine_silu (30) | n/a |
| 289 | outer_10.pre_norm_silu | affine_silu | affine_silu; affineSiluHalf2Kernel; g4693x1x1; b384x1x1; r17; s0 | 60 | n/a | 7.872 | n/a | 100.0% | 0.000 | library_gemm (60) | n/a |
| 290 | outer_10.pre_projection_c768_to_c384 | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 60 | n/a | 16.113 | n/a | 94.1% | 0.000 | wide_qkv (30) | n/a |
| 291 | outer_10.transformer_0.block_30.attention_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | n/a | 4.224 | n/a | 100.0% | 0.000 | library_gemm (30) | n/a |
| 292 | outer_10.transformer_0.block_30.attention_qkv_projection | wide_qkv | wide_qkv; wide_qkv_kernel; g9x37x1; b128x1x1; r136; s65536 | 60 | n/a | 23.392 | n/a | 65.7% | 0.000 | library_gemm (30) | n/a |
| 293 | outer_10.transformer_0.block_30.attention_qk_rope | qk_rope | qk_rope; fusedQKRoPE19HalfKernel; g361x13x1; b192x1x1; r16; s0 | 60 | n/a | 7.408 | n/a | 98.3% | 0.000 | fa4 (30) | n/a |
| 294 | outer_10.transformer_0.block_30.attention_fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 60 | n/a | 14.672 | n/a | 45.4% | 0.000 | library_gemm (28) | n/a |
| 295 | outer_10.transformer_0.block_30.attention_out_projection_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 60 | n/a | 16.288 | n/a | 95.9% | 0.000 | library_gemm (31) | n/a |
| 296 | outer_10.transformer_0.block_30.ffn_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | n/a | 3.968 | n/a | 100.0% | 0.000 | fused_ffn (30) | n/a |
| 297 | outer_10.transformer_0.block_30.ffn_linear1_gate_swiglu | fused_ffn | fused_ffn; fused_ffn_kernel; g18x37x1; b128x1x1; r146; s32768 | 60 | n/a | 42.145 | n/a | 83.8% | 0.000 | fused_ffn (30) | n/a |
| 298 | outer_10.transformer_0.block_30.ffn_linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 60 | n/a | 34.081 | n/a | 98.1% | 0.000 | fused_ffn (30) | n/a |
| 299 | outer_10.transformer_1.block_31.attention_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | n/a | 4.192 | n/a | 100.0% | 0.000 | linear2_residual (30) | n/a |
| 300 | outer_10.transformer_1.block_31.attention_qkv_projection | wide_qkv | wide_qkv; wide_qkv_kernel; g9x37x1; b128x1x1; r136; s65536 | 60 | n/a | 26.576 | n/a | 82.7% | 0.000 | linear2_residual (30) | n/a |
| 301 | outer_10.transformer_1.block_31.attention_qk_rope | qk_rope | qk_rope; fusedQKRoPE19HalfKernel; g361x13x1; b192x1x1; r16; s0 | 60 | n/a | 7.505 | n/a | 98.3% | 0.000 | fa4 (30) | n/a |
| 302 | outer_10.transformer_1.block_31.attention_fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 60 | n/a | 14.800 | n/a | 45.9% | 0.000 | library_gemm (28) | n/a |
| 303 | outer_10.transformer_1.block_31.attention_out_projection_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 60 | n/a | 17.248 | n/a | 95.8% | 0.000 | fused_ffn (30) | n/a |
| 304 | outer_10.transformer_1.block_31.ffn_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | n/a | 3.984 | n/a | 100.0% | 0.000 | fused_ffn (30) | n/a |
| 305 | outer_10.transformer_1.block_31.ffn_linear1_gate_swiglu | fused_ffn | fused_ffn; fused_ffn_kernel; g18x37x1; b128x1x1; r146; s32768 | 60 | n/a | 43.984 | n/a | 81.2% | 0.000 | fused_ffn (30) | n/a |
| 306 | outer_10.transformer_1.block_31.ffn_linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 60 | n/a | 32.913 | n/a | 98.4% | 0.000 | fused_ffn (30) | n/a |
| 307 | outer_10.transformer_2.block_32.attention_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | n/a | 4.176 | n/a | 100.0% | 0.000 | linear2_residual (30) | n/a |
| 308 | outer_10.transformer_2.block_32.attention_qkv_projection | wide_qkv | wide_qkv; wide_qkv_kernel; g9x37x1; b128x1x1; r136; s65536 | 60 | n/a | 26.800 | n/a | 83.1% | 0.000 | linear2_residual (30) | n/a |
| 309 | outer_10.transformer_2.block_32.attention_qk_rope | qk_rope | qk_rope; fusedQKRoPE19HalfKernel; g361x13x1; b192x1x1; r16; s0 | 60 | n/a | 7.616 | n/a | 97.5% | 0.000 | fa4 (30) | n/a |
| 310 | outer_10.transformer_2.block_32.attention_fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 60 | n/a | 14.816 | n/a | 46.3% | 0.000 | library_gemm (27) | n/a |
| 311 | outer_10.transformer_2.block_32.attention_out_projection_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 60 | n/a | 17.136 | n/a | 95.8% | 0.000 | library_gemm (31) | n/a |
| 312 | outer_10.transformer_2.block_32.ffn_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | n/a | 3.920 | n/a | 100.0% | 0.000 | fused_ffn (30) | n/a |
| 313 | outer_10.transformer_2.block_32.ffn_linear1_gate_swiglu | fused_ffn | fused_ffn; fused_ffn_kernel; g18x37x1; b128x1x1; r146; s32768 | 60 | n/a | 47.184 | n/a | 80.8% | 0.000 | fused_ffn (30) | n/a |
| 314 | outer_10.transformer_2.block_32.ffn_linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 60 | n/a | 32.689 | n/a | 98.2% | 0.000 | fused_ffn (30) | n/a |
| 315 | outer_10.post_norm_silu | affine_silu | affine_silu; affineSiluHalf2Kernel; g4693x1x1; b192x1x1; r17; s0 | 60 | n/a | 4.128 | n/a | 100.0% | 0.000 | library_gemm (30) | n/a |
| 316 | outer_10.post_projection_c384_to_c768_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b256x1x1; r154; s73728 | 60 | n/a | 22.704 | n/a | 95.5% | 0.000 | affine_silu (30) | n/a |
| 317 | trunk.tip_norm_silu | affine_silu | affine_silu; affineSiluHalf2Kernel; g4693x1x1; b384x1x1; r17; s0 | 60 | n/a | 8.240 | n/a | 86.7% | 0.000 | library_gemm (60) | n/a |
| 318 | policy.p1_conv | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r80; s98304 | 60 | n/a | 10.368 | n/a | 93.4% | 0.000 | library_gemm (60) | n/a |
| 319 | policy.g1_conv | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r80; s98304 | 60 | n/a | 7.840 | n/a | 91.3% | 0.000 | head_elementwise (30) | n/a |
| 320 | policy.g1_norm_silu | head_elementwise | head_elementwise; applyCScaleBiasNHWCSiluHalfKernel; g1x73x13; b96x5x1; r16; s0 | 60 | n/a | 2.656 | n/a | 99.7% | 0.000 | head_elementwise (30) | n/a |
| 321 | policy.g1_half_to_float | copy_reformat | copy_reformat; copyFromHalfKernel; g880x1x1; b512x1x1; r16; s0 | 60 | n/a | 1.872 | n/a | 57.7% | 0.000 | library_gemm (40) | n/a |
| 322 | policy.g1_global_pool | head_elementwise | head_elementwise; gPoolChannelsNHWCKernel; g2x1x13; b64x8x1; r22; s4096 | 60 | n/a | 6.608 | n/a | 94.5% | 0.000 | library_gemm (60) | n/a |
| 323 | policy.gpool_to_bias_matmul | library_gemm | library_gemm; gemmSN_NN_kernel; g2x2x1; b256x1x1; r64; s21504 | 60 | n/a | 6.592 | n/a | 82.1% | 0.000 | head_elementwise (60) | n/a |
| 324 | policy.p1_half_to_float | copy_reformat | copy_reformat; copyFromHalfKernel; g880x1x1; b512x1x1; r16; s0 | 60 | n/a | 1.696 | n/a | 77.0% | 0.000 | library_gemm (50) | n/a |
| 325 | policy.gpool_bias_add | head_elementwise | head_elementwise; addNCBiasInplaceNHWCKernel; g1x73x13; b96x5x1; r16; s0 | 60 | n/a | 1.984 | n/a | 97.9% | 0.000 | library_gemm (60) | n/a |
| 326 | policy.p1_norm_silu | head_elementwise | head_elementwise; applyCScaleBiasNHWCSiluKernel; g1x73x13; b96x5x1; r16; s0 | 60 | n/a | 2.688 | n/a | 86.9% | 0.000 | library_gemm (58) | n/a |
| 327 | policy.p2_conv | library_gemm | library_gemm; Kernel2; g74x1x1; b128x1x1; r90; s98304 | 60 | n/a | 4.528 | n/a | 89.2% | 0.000 | library_gemm (30) | n/a |
| 328 | policy.gpool_to_pass_matmul | library_gemm | library_gemm; gemmSN_NN_kernel; g2x2x1; b256x1x1; r64; s21504 | 60 | n/a | 6.048 | n/a | 84.2% | 0.000 | library_gemm (56) | n/a |
| 329 | policy.pass_bias_silu | head_elementwise | head_elementwise; addCBiasInplaceNCKernelSilu; g1x3x1; b96x5x1; r16; s0 | 60 | n/a | 1.200 | n/a | 99.3% | 0.000 | library_gemm (60) | n/a |
| 330 | policy.gpool_to_pass_matmul2 | library_gemm | library_gemm; gemmSN_NN_kernel; g1x2x1; b256x1x1; r64; s21504 | 60 | n/a | 2.640 | n/a | 97.3% | 0.000 | library_gemm (56) | n/a |
| 331 | value.v1_conv | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r118; s98304 | 60 | n/a | 9.312 | n/a | 85.2% | 0.000 | head_elementwise (28) | n/a |
| 332 | value.v1_norm_silu | head_elementwise | head_elementwise; applyCScaleBiasNHWCSiluHalfKernel; g1x181x13; b192x2x1; r16; s0 | 60 | n/a | 3.680 | n/a | 77.3% | 0.000 | library_gemm (30) | n/a |
| 333 | value.v1_half_to_float | copy_reformat | copy_reformat; copyFromHalfKernel; g1760x1x1; b512x1x1; r16; s0 | 60 | n/a | 2.528 | n/a | 92.8% | 0.000 | head_elementwise (33) | n/a |
| 334 | value.v1_global_pool | head_elementwise | head_elementwise; valueHeadPoolChannelsNHWCKernel; g3x1x13; b64x8x1; r22; s2048 | 60 | n/a | 3.712 | n/a | 89.7% | 0.000 | library_gemm (30) | n/a |
| 335 | value.v2_matmul | library_gemm | library_gemm; gemmSN_NN_kernel; g3x2x1; b256x1x1; r64; s21504 | 60 | n/a | 10.016 | n/a | 93.0% | 0.000 | library_gemm (59) | n/a |
| 336 | value.v2_bias_silu | head_elementwise | head_elementwise; addCBiasInplaceNCKernelSilu; g1x7x1; b192x2x1; r16; s0 | 60 | n/a | 1.056 | n/a | 97.9% | 0.000 | library_gemm (59) | n/a |
| 337 | value.v3_matmul | library_gemm | library_gemm; gemmSN_NN_kernel; g1x2x1; b256x1x1; r64; s21504 | 60 | n/a | 3.808 | n/a | 76.0% | 0.000 | library_gemm (51) | n/a |
| 338 | value.v3_bias | head_elementwise | head_elementwise; addCBiasInplaceNCKernel; g1x1x1; b3x170x1; r16; s0 | 60 | n/a | 1.024 | n/a | 96.9% | 0.000 | library_gemm (58) | n/a |
| 339 | value.score_matmul | library_gemm | library_gemm; gemmSN_NN_kernel; g1x2x1; b256x1x1; r64; s21504 | 60 | n/a | 3.872 | n/a | 77.9% | 0.000 | library_gemm (56) | n/a |
| 340 | value.score_bias | head_elementwise | head_elementwise; addCBiasInplaceNCKernel; g1x1x1; b6x85x1; r16; s0 | 60 | n/a | 1.024 | n/a | 96.9% | 0.000 | library_gemm (57) | n/a |
| 341 | value.ownership_conv | library_gemm | library_gemm; Kernel2; g8x19x3; b128x1x1; r118; s33792 | 60 | n/a | 4.544 | n/a | 64.2% | 0.000 | library_gemm (56) | n/a |
| 342 | value.ownership_conv_splitk_reduce | library_gemm | library_gemm; splitKreduce_kernel; g147x1x1; b32x16x1; r49; s0 | 60 | n/a | 1.440 | n/a | 61.4% | 0.000 | library_gemm (30) | n/a |
| 343 | value.ownership_half_to_float | copy_reformat | copy_reformat; copyFromHalfKernel; g10x1x1; b512x1x1; r16; s0 | 60 | n/a | 1.168 | n/a | 31.8% | 0.000 | library_gemm (28) | n/a |
