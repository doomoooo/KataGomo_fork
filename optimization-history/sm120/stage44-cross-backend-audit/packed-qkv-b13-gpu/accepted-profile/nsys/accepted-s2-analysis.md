# Nsys stream interference report

- Timed iterations: 20; streams: 65, 82
- Kernels per forward: 65=344, 82=344
- Iteration start offset stream 82 - 65: median -68.83 us, p10..p90 -70.79..-63.78 us, range -71.78..-38.59 us.

## Kernel families

| family | instances | total ms | median us | p90 us | peer overlap | S2/S1 median | summed excess ms | matched |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| fused_ffn | 1320 | 63.240 | 49.089 | 54.369 | 66.6% | 1.269x | 12.178 | 1320 |
| library_gemm | 2760 | 41.996 | 16.176 | 21.600 | 96.0% | 1.539x | 16.663 | 2760 |
| linear2_residual | 1320 | 39.445 | 30.113 | 33.792 | 96.8% | 1.446x | 11.906 | 1320 |
| wide_qkv | 1320 | 30.022 | 23.905 | 28.192 | 89.3% | 1.690x | 11.331 | 1320 |
| fa4 | 1320 | 23.161 | 17.504 | 19.329 | 74.4% | 1.464x | 7.435 | 1320 |
| qk_rope | 1320 | 10.539 | 8.384 | 8.896 | 99.2% | 1.492x | 3.112 | 1320 |
| rmsnorm | 2640 | 10.185 | 3.553 | 5.184 | 96.1% | 1.460x | 3.735 | 2640 |
| affine_silu | 920 | 5.543 | 5.856 | 7.296 | 83.8% | 1.414x | 1.643 | 920 |
| head_elementwise | 480 | 1.391 | 2.336 | 5.664 | 65.4% | 1.134x | 0.200 | 480 |
| cudnn | 120 | 1.098 | 2.448 | 21.668 | 54.6% | 1.125x | 0.202 | 120 |
| copy_reformat | 200 | 0.404 | 2.048 | 2.976 | 78.7% | 1.276x | 0.120 | 200 |
| sumChannelsNCHWKernel | 40 | 0.094 | 2.336 | 2.790 | 80.4% | 1.377x | 0.026 | 40 |

## Dominant interference pairs

| running family | dominant peer | instances | median overlap | median us | S2/S1 median | matched |
|---|---|---:|---:|---:|---:|---:|
| cudnn | wide_qkv | 18 | 97.0% | 6.864 | 5.362x | 18 |
| library_gemm | fused_ffn | 896 | 97.3% | 18.848 | 2.243x | 896 |
| affine_silu | fused_ffn | 11 | 100.0% | 7.232 | 2.171x | 11 |
| copy_reformat | rmsnorm | 4 | 82.1% | 2.048 | 2.133x | 4 |
| rmsnorm | fused_ffn | 839 | 100.0% | 5.056 | 2.065x | 839 |
| affine_silu | wide_qkv | 10 | 100.0% | 9.968 | 1.984x | 10 |
| wide_qkv | fused_ffn | 839 | 98.1% | 26.400 | 1.870x | 839 |
| library_gemm | wide_qkv | 253 | 95.4% | 14.720 | 1.722x | 253 |
| library_gemm | qk_rope | 220 | 95.4% | 13.969 | 1.623x | 220 |
| linear2_residual | fused_ffn | 451 | 96.9% | 33.280 | 1.598x | 451 |
| sumChannelsNCHWKernel | rmsnorm | 13 | 83.1% | 2.656 | 1.566x | 13 |
| copy_reformat | linear2_residual | 38 | 100.0% | 2.448 | 1.562x | 38 |
| fa4 | wide_qkv | 84 | 71.8% | 18.368 | 1.549x | 84 |
| cudnn | qk_rope | 18 | 100.0% | 2.352 | 1.531x | 18 |
| qk_rope | linear2_residual | 839 | 100.0% | 8.577 | 1.528x | 839 |
| fa4 | linear2_residual | 657 | 94.4% | 17.825 | 1.495x | 657 |
| affine_silu | fa4 | 400 | 100.0% | 4.864 | 1.455x | 400 |
| rmsnorm | fa4 | 898 | 100.0% | 3.552 | 1.453x | 898 |
| rmsnorm | library_gemm | 58 | 96.8% | 3.552 | 1.447x | 58 |
| linear2_residual | fa4 | 294 | 97.2% | 30.112 | 1.446x | 294 |
| fa4 | fused_ffn | 396 | 49.7% | 17.088 | 1.428x | 396 |
| affine_silu | rmsnorm | 8 | 47.6% | 7.200 | 1.424x | 8 |
| head_elementwise | linear2_residual | 38 | 100.0% | 4.608 | 1.406x | 38 |
| affine_silu | library_gemm | 459 | 76.4% | 7.072 | 1.405x | 459 |
| affine_silu | linear2_residual | 11 | 100.0% | 7.041 | 1.401x | 11 |
| fused_ffn | wide_qkv | 348 | 63.9% | 53.761 | 1.385x | 348 |
| sumChannelsNCHWKernel | head_elementwise | 11 | 79.5% | 2.336 | 1.377x | 11 |
| copy_reformat | head_elementwise | 36 | 69.9% | 2.656 | 1.351x | 36 |
| rmsnorm | linear2_residual | 451 | 100.0% | 3.296 | 1.346x | 451 |
| fa4 | affine_silu | 108 | 64.5% | 15.792 | 1.328x | 108 |
| fa4 | rmsnorm | 55 | 45.7% | 15.744 | 1.326x | 55 |
| cudnn | head_elementwise | 9 | 100.0% | 1.696 | 1.325x | 9 |
| fa4 | library_gemm | 19 | 63.0% | 15.489 | 1.322x | 19 |
| library_gemm | affine_silu | 39 | 92.6% | 7.456 | 1.319x | 39 |
| fused_ffn | linear2_residual | 451 | 79.3% | 49.632 | 1.278x | 451 |
| copy_reformat | library_gemm | 78 | 100.0% | 1.568 | 1.276x | 78 |
| linear2_residual | wide_qkv | 543 | 96.8% | 26.592 | 1.270x | 543 |
| library_gemm | cudnn | 40 | 87.3% | 5.424 | 1.240x | 40 |
| sumChannelsNCHWKernel | copy_reformat | 7 | 76.9% | 2.080 | 1.226x | 7 |
| library_gemm | fa4 | 37 | 100.0% | 3.648 | 1.225x | 37 |

## Logical operation groups

Isolated reference total is the isolated median for each ordinal multiplied by its S2 call count; it is a normalized reference, not a second trace total.

| logical group | families | positions | calls | isolated reference ms | S2 total ms | total ratio | excess ms |
|---|---|---:|---:|---:|---:|---:|---:|
| transformer.attention_out_projection_residual | library_gemm | 33 | 1320 | 11.228 | 24.134 | 2.149x | 12.905 |
| transformer.ffn_linear1_gate_swiglu | fused_ffn | 33 | 1320 | 51.067 | 63.240 | 1.238x | 12.178 |
| transformer.ffn_linear2_residual | linear2_residual | 33 | 1320 | 27.539 | 39.445 | 1.432x | 11.906 |
| transformer.attention_qkv_projection | wide_qkv | 33 | 1320 | 18.691 | 30.022 | 1.606x | 11.331 |
| transformer.attention_fa4 | fa4 | 33 | 1320 | 15.726 | 23.161 | 1.473x | 7.435 |
| transformer.attention_qk_rope | qk_rope | 33 | 1320 | 7.427 | 10.539 | 1.419x | 3.112 |
| transformer.attention_rmsnorm | rmsnorm | 33 | 1320 | 3.221 | 5.852 | 1.817x | 2.632 |
| outer.pre_projection_c768_to_c384 | library_gemm | 11 | 440 | 5.245 | 6.787 | 1.294x | 1.542 |
| outer.post_projection_c384_to_c768_residual | library_gemm | 11 | 440 | 6.350 | 7.611 | 1.199x | 1.261 |
| transformer.ffn_rmsnorm | rmsnorm | 33 | 1320 | 3.230 | 4.333 | 1.341x | 1.103 |
| outer.pre_norm_silu | affine_silu | 11 | 440 | 2.225 | 3.105 | 1.396x | 0.880 |
| outer.post_norm_silu | affine_silu | 11 | 440 | 1.473 | 2.178 | 1.479x | 0.705 |
| policy.g1_conv | library_gemm | 1 | 40 | 0.236 | 0.592 | 2.513x | 0.356 |
| value.v2_matmul | library_gemm | 1 | 40 | 0.380 | 0.494 | 1.298x | 0.113 |
| frontend.initial_conv_nhwc_padding_0 | cudnn | 1 | 40 | 0.051 | 0.155 | 3.022x | 0.104 |
| policy.p2_conv | library_gemm | 1 | 40 | 0.157 | 0.248 | 1.575x | 0.091 |
| frontend.initial_conv | cudnn | 1 | 40 | 0.783 | 0.865 | 1.104x | 0.082 |
| value.v1_conv | library_gemm | 1 | 40 | 0.318 | 0.387 | 1.215x | 0.069 |
| trunk.tip_norm_silu | affine_silu | 1 | 40 | 0.202 | 0.260 | 1.285x | 0.058 |
| policy.gpool_to_pass_matmul2 | library_gemm | 1 | 40 | 0.092 | 0.145 | 1.572x | 0.053 |
| frontend.initial_global_matmul | library_gemm | 1 | 40 | 0.105 | 0.153 | 1.455x | 0.048 |
| policy.p1_conv | library_gemm | 1 | 40 | 0.250 | 0.297 | 1.190x | 0.047 |
| policy.gpool_to_pass_matmul | library_gemm | 1 | 40 | 0.211 | 0.252 | 1.198x | 0.042 |
| value.ownership_conv | library_gemm | 1 | 40 | 0.161 | 0.198 | 1.227x | 0.037 |
| policy.p1_norm_silu | head_elementwise | 1 | 40 | 0.086 | 0.122 | 1.422x | 0.036 |
| value.ownership_half_to_float | copy_reformat | 1 | 40 | 0.037 | 0.073 | 1.972x | 0.036 |
| policy.g1_norm_silu | head_elementwise | 1 | 40 | 0.085 | 0.119 | 1.397x | 0.034 |
| policy.p1_half_to_float | copy_reformat | 1 | 40 | 0.060 | 0.093 | 1.545x | 0.033 |
| policy.g1_global_pool | head_elementwise | 1 | 40 | 0.178 | 0.209 | 1.175x | 0.031 |
| policy.gpool_to_bias_matmul | library_gemm | 1 | 40 | 0.215 | 0.242 | 1.126x | 0.027 |
| input.mask_sum | sumChannelsNCHWKernel | 1 | 40 | 0.068 | 0.094 | 1.379x | 0.026 |
| value.ownership_conv_splitk_reduce | library_gemm | 1 | 40 | 0.055 | 0.079 | 1.430x | 0.024 |
| value.v1_half_to_float | copy_reformat | 1 | 40 | 0.086 | 0.108 | 1.263x | 0.023 |
| value.score_matmul | library_gemm | 1 | 40 | 0.140 | 0.162 | 1.156x | 0.022 |
| policy.g1_half_to_float | copy_reformat | 1 | 40 | 0.063 | 0.083 | 1.316x | 0.020 |
| frontend.initial_global_broadcast_add | head_elementwise | 1 | 40 | 0.310 | 0.329 | 1.063x | 0.020 |
| value.v1_global_pool | head_elementwise | 1 | 40 | 0.130 | 0.149 | 1.144x | 0.019 |
| value.v3_matmul | library_gemm | 1 | 40 | 0.139 | 0.157 | 1.133x | 0.018 |
| frontend.initial_conv_nhwc_padding_1 | cudnn | 1 | 40 | 0.061 | 0.078 | 1.278x | 0.017 |
| value.v1_norm_silu | head_elementwise | 1 | 40 | 0.127 | 0.140 | 1.108x | 0.014 |
| policy.gpool_bias_add | head_elementwise | 1 | 40 | 0.072 | 0.084 | 1.170x | 0.012 |
| value.v2_bias_silu | head_elementwise | 1 | 40 | 0.041 | 0.053 | 1.291x | 0.012 |
| input.mask_half_to_float | copy_reformat | 1 | 40 | 0.038 | 0.047 | 1.224x | 0.009 |
| frontend.initial_global_matmul_splitk_reduce | library_gemm | 1 | 40 | 0.051 | 0.059 | 1.158x | 0.008 |
| input.extract_mask | head_elementwise | 1 | 40 | 0.047 | 0.054 | 1.130x | 0.006 |
| value.v3_bias | head_elementwise | 1 | 40 | 0.038 | 0.044 | 1.147x | 0.006 |
| policy.pass_bias_silu | head_elementwise | 1 | 40 | 0.041 | 0.046 | 1.132x | 0.006 |
| value.score_bias | head_elementwise | 1 | 40 | 0.037 | 0.042 | 1.141x | 0.005 |

## `library_gemm` logical breakdown

| logical group | positions | calls | isolated reference ms | S2 total ms | total ratio | excess ms |
|---|---:|---:|---:|---:|---:|---:|
| transformer.attention_out_projection_residual | 33 | 1320 | 11.228 | 24.134 | 2.149x | 12.905 |
| outer.pre_projection_c768_to_c384 | 11 | 440 | 5.245 | 6.787 | 1.294x | 1.542 |
| outer.post_projection_c384_to_c768_residual | 11 | 440 | 6.350 | 7.611 | 1.199x | 1.261 |
| policy.g1_conv | 1 | 40 | 0.236 | 0.592 | 2.513x | 0.356 |
| value.v2_matmul | 1 | 40 | 0.380 | 0.494 | 1.298x | 0.113 |
| policy.p2_conv | 1 | 40 | 0.157 | 0.248 | 1.575x | 0.091 |
| value.v1_conv | 1 | 40 | 0.318 | 0.387 | 1.215x | 0.069 |
| policy.gpool_to_pass_matmul2 | 1 | 40 | 0.092 | 0.145 | 1.572x | 0.053 |
| frontend.initial_global_matmul | 1 | 40 | 0.105 | 0.153 | 1.455x | 0.048 |
| policy.p1_conv | 1 | 40 | 0.250 | 0.297 | 1.190x | 0.047 |
| policy.gpool_to_pass_matmul | 1 | 40 | 0.211 | 0.252 | 1.198x | 0.042 |
| value.ownership_conv | 1 | 40 | 0.161 | 0.198 | 1.227x | 0.037 |
| policy.gpool_to_bias_matmul | 1 | 40 | 0.215 | 0.242 | 1.126x | 0.027 |
| value.ownership_conv_splitk_reduce | 1 | 40 | 0.055 | 0.079 | 1.430x | 0.024 |
| value.score_matmul | 1 | 40 | 0.140 | 0.162 | 1.156x | 0.022 |
| value.v3_matmul | 1 | 40 | 0.139 | 0.157 | 1.133x | 0.018 |
| frontend.initial_global_matmul_splitk_reduce | 1 | 40 | 0.051 | 0.059 | 1.158x | 0.008 |

## Top ordinal hotspots by summed excess

The worst peer is the highest median S2/S1 slowdown among peer families observed at least four times for that ordinal.

| rank | ordinal | logical position | family | calls | isolated us | S2 us | S2/S1 | excess ms | common peer | worst peer |
|---:|---:|---|---|---:|---:|---:|---:|---:|---|---|
| 1 | 201 | outer_06.transformer_2.block_20.ffn_linear1_gate_swiglu | fused_ffn | 40 | 33.697 | 50.352 | 1.494x | 0.643 | linear2_residual (20) | wide_qkv (1.515x; 10) |
| 2 | 229 | outer_07.transformer_2.block_23.ffn_linear1_gate_swiglu | fused_ffn | 40 | 35.840 | 51.136 | 1.427x | 0.602 | linear2_residual (20) | wide_qkv (1.429x; 17) |
| 3 | 145 | outer_04.transformer_2.block_14.ffn_linear1_gate_swiglu | fused_ffn | 40 | 35.441 | 50.880 | 1.436x | 0.597 | linear2_residual (20) | wide_qkv (1.454x; 9) |
| 4 | 89 | outer_02.transformer_2.block_08.ffn_linear1_gate_swiglu | fused_ffn | 40 | 37.920 | 50.881 | 1.342x | 0.571 | linear2_residual (20) | wide_qkv (1.462x; 17) |
| 5 | 264 | outer_09.transformer_0.block_27.attention_qkv_projection | wide_qkv | 40 | 14.209 | 27.808 | 1.957x | 0.538 | fused_ffn (38) | fused_ffn (1.963x; 38) |
| 6 | 180 | outer_06.transformer_0.block_18.attention_qkv_projection | wide_qkv | 40 | 14.048 | 27.552 | 1.961x | 0.533 | fused_ffn (38) | fused_ffn (1.965x; 38) |
| 7 | 152 | outer_05.transformer_0.block_15.attention_qkv_projection | wide_qkv | 40 | 14.113 | 26.833 | 1.901x | 0.522 | fused_ffn (38) | fused_ffn (1.949x; 38) |
| 8 | 61 | outer_01.transformer_2.block_05.ffn_linear1_gate_swiglu | fused_ffn | 40 | 38.529 | 49.761 | 1.292x | 0.519 | linear2_residual (20) | wide_qkv (1.419x; 18) |
| 9 | 68 | outer_02.transformer_0.block_06.attention_qkv_projection | wide_qkv | 40 | 14.064 | 27.201 | 1.934x | 0.517 | fused_ffn (38) | fused_ffn (1.936x; 38) |
| 10 | 236 | outer_08.transformer_0.block_24.attention_qkv_projection | wide_qkv | 40 | 14.240 | 27.408 | 1.925x | 0.515 | fused_ffn (38) | fused_ffn (1.926x; 38) |
| 11 | 173 | outer_05.transformer_2.block_17.ffn_linear1_gate_swiglu | fused_ffn | 40 | 38.849 | 50.432 | 1.298x | 0.512 | linear2_residual (20) | wide_qkv (1.405x; 19) |
| 12 | 199 | outer_06.transformer_2.block_20.attention_out_projection_residual | library_gemm | 40 | 8.480 | 20.736 | 2.445x | 0.508 | fused_ffn (20) | fused_ffn (2.677x; 20) |
| 13 | 33 | outer_00.transformer_2.block_02.ffn_linear1_gate_swiglu | fused_ffn | 40 | 38.304 | 49.441 | 1.291x | 0.506 | linear2_residual (20) | wide_qkv (1.407x; 17) |
| 14 | 96 | outer_03.transformer_0.block_09.attention_qkv_projection | wide_qkv | 40 | 14.257 | 27.088 | 1.900x | 0.506 | fused_ffn (38) | fused_ffn (1.901x; 38) |
| 15 | 255 | outer_08.transformer_2.block_26.attention_out_projection_residual | library_gemm | 40 | 8.480 | 20.864 | 2.460x | 0.500 | fused_ffn (20) | fused_ffn (2.657x; 20) |
| 16 | 40 | outer_01.transformer_0.block_03.attention_qkv_projection | wide_qkv | 40 | 14.080 | 26.801 | 1.903x | 0.498 | fused_ffn (38) | fused_ffn (1.906x; 38) |
| 17 | 285 | outer_09.transformer_2.block_29.ffn_linear1_gate_swiglu | fused_ffn | 40 | 38.288 | 48.961 | 1.279x | 0.497 | linear2_residual (20) | wide_qkv (1.398x; 17) |
| 18 | 87 | outer_02.transformer_2.block_08.attention_out_projection_residual | library_gemm | 40 | 8.544 | 20.785 | 2.433x | 0.492 | fused_ffn (20) | fused_ffn (2.646x; 20) |
| 19 | 59 | outer_01.transformer_2.block_05.attention_out_projection_residual | library_gemm | 40 | 8.464 | 20.704 | 2.446x | 0.486 | fused_ffn (20) | fused_ffn (2.620x; 20) |
| 20 | 292 | outer_10.transformer_0.block_30.attention_qkv_projection | wide_qkv | 40 | 14.016 | 26.288 | 1.876x | 0.485 | fused_ffn (38) | fused_ffn (1.879x; 38) |
| 21 | 143 | outer_04.transformer_2.block_14.attention_out_projection_residual | library_gemm | 40 | 8.512 | 20.721 | 2.434x | 0.485 | fused_ffn (20) | fused_ffn (2.532x; 20) |
| 22 | 171 | outer_05.transformer_2.block_17.attention_out_projection_residual | library_gemm | 40 | 8.480 | 19.808 | 2.336x | 0.485 | fused_ffn (20) | fused_ffn (2.651x; 20) |
| 23 | 208 | outer_07.transformer_0.block_21.attention_qkv_projection | wide_qkv | 40 | 14.273 | 26.624 | 1.865x | 0.483 | fused_ffn (38) | fused_ffn (1.869x; 38) |
| 24 | 222 | outer_07.transformer_1.block_22.ffn_linear2_residual | linear2_residual | 40 | 21.056 | 32.897 | 1.562x | 0.480 | fused_ffn (20) | fused_ffn (1.684x; 20) |
| 25 | 257 | outer_08.transformer_2.block_26.ffn_linear1_gate_swiglu | fused_ffn | 40 | 40.273 | 51.953 | 1.290x | 0.479 | linear2_residual (20) | wide_qkv (1.373x; 19) |
| 26 | 115 | outer_03.transformer_2.block_11.attention_out_projection_residual | library_gemm | 40 | 8.416 | 19.473 | 2.314x | 0.467 | fused_ffn (20) | fused_ffn (2.572x; 20) |
| 27 | 311 | outer_10.transformer_2.block_32.attention_out_projection_residual | library_gemm | 40 | 8.464 | 20.000 | 2.363x | 0.466 | fused_ffn (20) | fused_ffn (2.437x; 20) |
| 28 | 250 | outer_08.transformer_1.block_25.ffn_linear2_residual | linear2_residual | 40 | 20.960 | 32.961 | 1.573x | 0.466 | fused_ffn (20) | fused_ffn (1.623x; 20) |
| 29 | 31 | outer_00.transformer_2.block_02.attention_out_projection_residual | library_gemm | 40 | 8.512 | 19.601 | 2.303x | 0.465 | fused_ffn (20) | fused_ffn (2.558x; 20) |
| 30 | 283 | outer_09.transformer_2.block_29.attention_out_projection_residual | library_gemm | 40 | 8.480 | 19.328 | 2.279x | 0.463 | fused_ffn (20) | fused_ffn (2.555x; 20) |
| 31 | 124 | outer_04.transformer_0.block_12.attention_qkv_projection | wide_qkv | 40 | 14.033 | 25.824 | 1.840x | 0.460 | fused_ffn (38) | fused_ffn (1.846x; 38) |
| 32 | 227 | outer_07.transformer_2.block_23.attention_out_projection_residual | library_gemm | 40 | 8.608 | 20.289 | 2.357x | 0.458 | fused_ffn (20) | library_gemm (2.364x; 19) |
| 33 | 82 | outer_02.transformer_1.block_07.ffn_linear2_residual | linear2_residual | 40 | 20.832 | 31.888 | 1.531x | 0.447 | fused_ffn (20) | fused_ffn (1.617x; 20) |
| 34 | 166 | outer_05.transformer_1.block_16.ffn_linear2_residual | linear2_residual | 40 | 20.848 | 31.776 | 1.524x | 0.441 | fused_ffn (20) | fused_ffn (1.605x; 20) |
| 35 | 54 | outer_01.transformer_1.block_04.ffn_linear2_residual | linear2_residual | 40 | 20.736 | 31.665 | 1.527x | 0.437 | fused_ffn (20) | fused_ffn (1.593x; 20) |
| 36 | 194 | outer_06.transformer_1.block_19.ffn_linear2_residual | linear2_residual | 40 | 20.992 | 31.232 | 1.488x | 0.432 | fused_ffn (20) | fused_ffn (1.617x; 20) |
| 37 | 138 | outer_04.transformer_1.block_13.ffn_linear2_residual | linear2_residual | 40 | 20.832 | 32.065 | 1.539x | 0.431 | fused_ffn (20) | fused_ffn (1.596x; 20) |
| 38 | 26 | outer_00.transformer_1.block_01.ffn_linear2_residual | linear2_residual | 40 | 20.881 | 31.536 | 1.510x | 0.429 | fused_ffn (20) | fused_ffn (1.584x; 20) |
| 39 | 278 | outer_09.transformer_1.block_28.ffn_linear2_residual | linear2_residual | 40 | 20.641 | 31.488 | 1.526x | 0.425 | fused_ffn (20) | fused_ffn (1.591x; 20) |
| 40 | 110 | outer_03.transformer_1.block_10.ffn_linear2_residual | linear2_residual | 40 | 20.640 | 31.073 | 1.505x | 0.421 | fused_ffn (20) | fused_ffn (1.593x; 20) |

## Full fixed-forward ordinal map

| ordinal | logical position | family | resource signature | calls | isolated us | S2 us | S2/S1 | overlap | excess ms | common peer | worst peer |
|---:|---|---|---|---:|---:|---:|---:|---:|---:|---|---|
| 0 | input.extract_mask | head_elementwise | head_elementwise; extractChannel0KernelNHWC; g10x1x1; b512x1x1; r16; s0 | 40 | 1.184 | 1.312 | 1.108x | 87.3% | 0.006 | library_gemm (32) | library_gemm (1.135x; 32) |
| 1 | input.mask_half_to_float | copy_reformat | copy_reformat; copyFromHalfKernel; g10x1x1; b512x1x1; r16; s0 | 40 | 0.960 | 1.056 | 1.100x | 75.2% | 0.009 | library_gemm (16) | rmsnorm (2.133x; 4) |
| 2 | input.mask_sum | sumChannelsNCHWKernel | sumChannelsNCHWKernel; sumChannelsNCHWKernel; g1x1x13; b256x2x1; r22; s2048 | 40 | 1.696 | 2.336 | 1.377x | 80.4% | 0.026 | rmsnorm (13) | rmsnorm (1.566x; 13) |
| 3 | frontend.initial_conv_nhwc_padding_0 | cudnn | cudnn; nhwcAddPaddingKernel; g340x1x1; b768x1x1; r34; s0 | 40 | 1.280 | 1.824 | 1.425x | 94.2% | 0.104 | wide_qkv (18) | wide_qkv (5.362x; 18) |
| 4 | frontend.initial_conv_nhwc_padding_1 | cudnn | cudnn; nhwcAddPaddingKernel; g340x1x1; b768x1x1; r34; s0 | 40 | 1.536 | 1.856 | 1.208x | 86.1% | 0.017 | library_gemm (21) | qk_rope (1.531x; 18) |
| 5 | frontend.initial_conv | cudnn | cudnn; Kernel; g296x3x1; b128x1x1; r94; s81920 | 40 | 19.584 | 21.520 | 1.099x | 44.6% | 0.082 | library_gemm (21) | library_gemm (1.100x; 21) |
| 6 | frontend.initial_global_matmul | library_gemm | library_gemm; Kernel2; g8x1x3; b128x1x1; r128; s24576 | 40 | 2.624 | 3.408 | 1.299x | 91.9% | 0.048 | fa4 (18) | fa4 (1.762x; 18) |
| 7 | frontend.initial_global_matmul_splitk_reduce | library_gemm | library_gemm; splitKreduce_kernel; g24x1x1; b32x16x1; r49; s0 | 40 | 1.280 | 1.440 | 1.125x | 96.9% | 0.008 | fa4 (18) | fa4 (1.150x; 18) |
| 8 | frontend.initial_global_broadcast_add | head_elementwise | head_elementwise; addNCBiasInplaceNHWCHalfKernel; g3x361x13; b256x1x1; r16; s0 | 40 | 7.744 | 8.145 | 1.052x | 25.7% | 0.020 | library_gemm (34) | library_gemm (1.050x; 34) |
| 9 | outer_00.pre_norm_silu | affine_silu | affine_silu; affineSiluHalf2Kernel; g4693x1x1; b384x1x1; r17; s0 | 40 | 4.960 | 6.256 | 1.261x | 60.5% | 0.054 | library_gemm (38) | library_gemm (1.261x; 38) |
| 10 | outer_00.pre_projection_c768_to_c384 | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 40 | 11.664 | 14.352 | 1.230x | 82.2% | 0.121 | library_gemm (19) | fused_ffn (1.343x; 18) |
| 11 | outer_00.transformer_0.block_00.attention_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 40 | 2.368 | 3.488 | 1.473x | 89.8% | 0.041 | fused_ffn (19) | fused_ffn (1.703x; 19) |
| 12 | outer_00.transformer_0.block_00.attention_qkv_projection | wide_qkv | wide_qkv; kernel_cutlass_kernel___main__FixedAtomLayoutGemmmakelocals_Kernel_object_at__CopyAtom_ThrID10_TVLayoutSrc1819201_TVLayoutDst1819201_Valuetypef16_tensor00o4693384111100_CopyAtom_ThrID10_T_0; g1x1x170; b288x1x1; r107; s99328 | 40 | 13.920 | 23.056 | 1.656x | 80.3% | 0.268 | fused_ffn (19) | fused_ffn (1.860x; 19) |
| 13 | outer_00.transformer_0.block_00.attention_qk_rope | qk_rope | qk_rope; batchSharedPackedFusedQKRoPE19Half2Kernel; g361x1x1; b192x1x1; r28; s0 | 40 | 5.504 | 7.744 | 1.407x | 83.9% | 0.083 | linear2_residual (19) | linear2_residual (1.488x; 19) |
| 14 | outer_00.transformer_0.block_00.attention_fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 40 | 11.712 | 17.280 | 1.475x | 78.5% | 0.208 | library_gemm (19) | linear2_residual (1.533x; 19) |
| 15 | outer_00.transformer_0.block_00.attention_out_projection_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 40 | 8.464 | 12.145 | 1.435x | 94.3% | 0.186 | affine_silu (18) | wide_qkv (1.750x; 17) |
| 16 | outer_00.transformer_0.block_00.ffn_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 40 | 2.432 | 3.520 | 1.447x | 98.3% | 0.043 | fa4 (20) | fa4 (1.507x; 20) |
| 17 | outer_00.transformer_0.block_00.ffn_linear1_gate_swiglu | fused_ffn | fused_ffn; fused_ffn_candidate_a_reuse_kernel; g18x37x1; b128x1x1; r136; s32768 | 40 | 39.312 | 41.936 | 1.067x | 54.8% | 0.137 | library_gemm (20) | wide_qkv (1.147x; 19) |
| 18 | outer_00.transformer_0.block_00.ffn_linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 40 | 20.960 | 31.856 | 1.520x | 97.2% | 0.417 | fused_ffn (20) | fused_ffn (1.572x; 20) |
| 19 | outer_00.transformer_1.block_01.attention_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 40 | 2.464 | 4.096 | 1.662x | 100.0% | 0.057 | fa4 (20) | fused_ffn (1.935x; 20) |
| 20 | outer_00.transformer_1.block_01.attention_qkv_projection | wide_qkv | wide_qkv; kernel_cutlass_kernel___main__FixedAtomLayoutGemmmakelocals_Kernel_object_at__CopyAtom_ThrID10_TVLayoutSrc1819201_TVLayoutDst1819201_Valuetypef16_tensor00o4693384111100_CopyAtom_ThrID10_T_0; g1x1x170; b288x1x1; r107; s99328 | 40 | 14.240 | 19.776 | 1.389x | 82.1% | 0.219 | fused_ffn (20) | fused_ffn (1.630x; 20) |
| 21 | outer_00.transformer_1.block_01.attention_qk_rope | qk_rope | qk_rope; batchSharedPackedFusedQKRoPE19Half2Kernel; g361x1x1; b192x1x1; r28; s0 | 40 | 5.696 | 7.664 | 1.346x | 99.8% | 0.079 | library_gemm (20) | linear2_residual (1.492x; 20) |
| 22 | outer_00.transformer_1.block_01.attention_fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 40 | 12.000 | 17.328 | 1.444x | 73.8% | 0.202 | linear2_residual (20) | linear2_residual (1.464x; 20) |
| 23 | outer_00.transformer_1.block_01.attention_out_projection_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 40 | 8.528 | 17.216 | 2.019x | 96.5% | 0.365 | fused_ffn (20) | fused_ffn (2.447x; 20) |
| 24 | outer_00.transformer_1.block_01.ffn_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 40 | 2.464 | 3.408 | 1.383x | 98.0% | 0.040 | fa4 (20) | fa4 (1.468x; 20) |
| 25 | outer_00.transformer_1.block_01.ffn_linear1_gate_swiglu | fused_ffn | fused_ffn; fused_ffn_candidate_a_reuse_kernel; g18x37x1; b128x1x1; r136; s32768 | 40 | 38.192 | 45.057 | 1.180x | 61.4% | 0.283 | linear2_residual (20) | linear2_residual (1.285x; 20) |
| 26 | outer_00.transformer_1.block_01.ffn_linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 40 | 20.881 | 31.536 | 1.510x | 97.0% | 0.429 | fused_ffn (20) | fused_ffn (1.584x; 20) |
| 27 | outer_00.transformer_2.block_02.attention_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 40 | 2.400 | 3.792 | 1.580x | 100.0% | 0.059 | fa4 (20) | fused_ffn (1.973x; 20) |
| 28 | outer_00.transformer_2.block_02.attention_qkv_projection | wide_qkv | wide_qkv; kernel_cutlass_kernel___main__FixedAtomLayoutGemmmakelocals_Kernel_object_at__CopyAtom_ThrID10_TVLayoutSrc1819201_TVLayoutDst1819201_Valuetypef16_tensor00o4693384111100_CopyAtom_ThrID10_T_0; g1x1x170; b288x1x1; r107; s99328 | 40 | 14.112 | 19.344 | 1.371x | 83.7% | 0.229 | fused_ffn (20) | fused_ffn (1.692x; 20) |
| 29 | outer_00.transformer_2.block_02.attention_qk_rope | qk_rope | qk_rope; batchSharedPackedFusedQKRoPE19Half2Kernel; g361x1x1; b192x1x1; r28; s0 | 40 | 5.616 | 7.600 | 1.353x | 99.9% | 0.082 | library_gemm (20) | linear2_residual (1.499x; 20) |
| 30 | outer_00.transformer_2.block_02.attention_fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 40 | 11.920 | 16.369 | 1.373x | 72.4% | 0.180 | linear2_residual (20) | linear2_residual (1.389x; 20) |
| 31 | outer_00.transformer_2.block_02.attention_out_projection_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 40 | 8.512 | 19.601 | 2.303x | 97.9% | 0.465 | fused_ffn (20) | fused_ffn (2.558x; 20) |
| 32 | outer_00.transformer_2.block_02.ffn_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 40 | 2.465 | 3.168 | 1.285x | 91.7% | 0.024 | linear2_residual (20) | linear2_residual (1.337x; 20) |
| 33 | outer_00.transformer_2.block_02.ffn_linear1_gate_swiglu | fused_ffn | fused_ffn; fused_ffn_candidate_a_reuse_kernel; g18x37x1; b128x1x1; r136; s32768 | 40 | 38.304 | 49.441 | 1.291x | 70.3% | 0.506 | linear2_residual (20) | wide_qkv (1.407x; 17) |
| 34 | outer_00.transformer_2.block_02.ffn_linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 40 | 20.753 | 29.008 | 1.398x | 96.6% | 0.266 | wide_qkv (31) | fa4 (1.454x; 7) |
| 35 | outer_00.post_norm_silu | affine_silu | affine_silu; affineSiluHalf2Kernel; g4693x1x1; b192x1x1; r17; s0 | 40 | 3.344 | 4.656 | 1.392x | 100.0% | 0.056 | fa4 (38) | fa4 (1.392x; 38) |
| 36 | outer_00.post_projection_c384_to_c768_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b256x1x1; r154; s73728 | 40 | 14.368 | 16.721 | 1.164x | 97.5% | 0.101 | library_gemm (38) | library_gemm (1.161x; 38) |
| 37 | outer_01.pre_norm_silu | affine_silu | affine_silu; affineSiluHalf2Kernel; g4693x1x1; b384x1x1; r17; s0 | 40 | 5.088 | 7.008 | 1.377x | 76.6% | 0.080 | library_gemm (38) | library_gemm (1.377x; 38) |
| 38 | outer_01.pre_projection_c768_to_c384 | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 40 | 11.920 | 14.880 | 1.248x | 96.6% | 0.122 | fused_ffn (38) | fused_ffn (1.248x; 38) |
| 39 | outer_01.transformer_0.block_03.attention_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 40 | 2.432 | 4.768 | 1.961x | 100.0% | 0.092 | fused_ffn (38) | fused_ffn (1.967x; 38) |
| 40 | outer_01.transformer_0.block_03.attention_qkv_projection | wide_qkv | wide_qkv; kernel_cutlass_kernel___main__FixedAtomLayoutGemmmakelocals_Kernel_object_at__CopyAtom_ThrID10_TVLayoutSrc1819201_TVLayoutDst1819201_Valuetypef16_tensor00o4693384111100_CopyAtom_ThrID10_T_0; g1x1x170; b288x1x1; r107; s99328 | 40 | 14.080 | 26.801 | 1.903x | 97.8% | 0.498 | fused_ffn (38) | fused_ffn (1.906x; 38) |
| 41 | outer_01.transformer_0.block_03.attention_qk_rope | qk_rope | qk_rope; batchSharedPackedFusedQKRoPE19Half2Kernel; g361x1x1; b192x1x1; r28; s0 | 40 | 5.568 | 8.512 | 1.529x | 99.9% | 0.114 | linear2_residual (38) | linear2_residual (1.534x; 38) |
| 42 | outer_01.transformer_0.block_03.attention_fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 40 | 11.840 | 16.800 | 1.419x | 68.6% | 0.203 | affine_silu (19) | wide_qkv (1.559x; 15) |
| 43 | outer_01.transformer_0.block_03.attention_out_projection_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 40 | 8.528 | 16.688 | 1.957x | 96.7% | 0.318 | library_gemm (19) | library_gemm (2.214x; 19) |
| 44 | outer_01.transformer_0.block_03.ffn_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 40 | 2.432 | 3.392 | 1.395x | 90.2% | 0.036 | fa4 (20) | fa4 (1.526x; 20) |
| 45 | outer_01.transformer_0.block_03.ffn_linear1_gate_swiglu | fused_ffn | fused_ffn; fused_ffn_candidate_a_reuse_kernel; g18x37x1; b128x1x1; r136; s32768 | 40 | 39.776 | 45.648 | 1.148x | 57.4% | 0.254 | library_gemm (21) | wide_qkv (1.304x; 18) |
| 46 | outer_01.transformer_0.block_03.ffn_linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 40 | 20.912 | 30.480 | 1.458x | 96.8% | 0.340 | fused_ffn (20) | fused_ffn (1.585x; 20) |
| 47 | outer_01.transformer_1.block_04.attention_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 40 | 2.448 | 4.240 | 1.732x | 100.0% | 0.074 | fa4 (20) | fused_ffn (2.092x; 20) |
| 48 | outer_01.transformer_1.block_04.attention_qkv_projection | wide_qkv | wide_qkv; kernel_cutlass_kernel___main__FixedAtomLayoutGemmmakelocals_Kernel_object_at__CopyAtom_ThrID10_TVLayoutSrc1819201_TVLayoutDst1819201_Valuetypef16_tensor00o4693384111100_CopyAtom_ThrID10_T_0; g1x1x170; b288x1x1; r107; s99328 | 40 | 14.208 | 19.952 | 1.404x | 83.8% | 0.233 | fused_ffn (20) | fused_ffn (1.633x; 20) |
| 49 | outer_01.transformer_1.block_04.attention_qk_rope | qk_rope | qk_rope; batchSharedPackedFusedQKRoPE19Half2Kernel; g361x1x1; b192x1x1; r28; s0 | 40 | 5.601 | 7.712 | 1.377x | 99.7% | 0.085 | library_gemm (20) | linear2_residual (1.534x; 20) |
| 50 | outer_01.transformer_1.block_04.attention_fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 40 | 11.889 | 17.536 | 1.475x | 74.7% | 0.225 | fused_ffn (20) | linear2_residual (1.497x; 20) |
| 51 | outer_01.transformer_1.block_04.attention_out_projection_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 40 | 8.528 | 16.992 | 1.992x | 96.1% | 0.381 | fused_ffn (20) | fused_ffn (2.563x; 20) |
| 52 | outer_01.transformer_1.block_04.ffn_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 40 | 2.432 | 3.392 | 1.395x | 99.0% | 0.041 | fa4 (20) | fa4 (1.513x; 20) |
| 53 | outer_01.transformer_1.block_04.ffn_linear1_gate_swiglu | fused_ffn | fused_ffn; fused_ffn_candidate_a_reuse_kernel; g18x37x1; b128x1x1; r136; s32768 | 40 | 38.577 | 45.264 | 1.173x | 61.4% | 0.306 | linear2_residual (20) | linear2_residual (1.302x; 20) |
| 54 | outer_01.transformer_1.block_04.ffn_linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 40 | 20.736 | 31.665 | 1.527x | 97.0% | 0.437 | fused_ffn (20) | fused_ffn (1.593x; 20) |
| 55 | outer_01.transformer_2.block_05.attention_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 40 | 2.432 | 3.984 | 1.638x | 100.0% | 0.063 | fa4 (20) | fused_ffn (2.039x; 20) |
| 56 | outer_01.transformer_2.block_05.attention_qkv_projection | wide_qkv | wide_qkv; kernel_cutlass_kernel___main__FixedAtomLayoutGemmmakelocals_Kernel_object_at__CopyAtom_ThrID10_TVLayoutSrc1819201_TVLayoutDst1819201_Valuetypef16_tensor00o4693384111100_CopyAtom_ThrID10_T_0; g1x1x170; b288x1x1; r107; s99328 | 40 | 14.177 | 19.440 | 1.371x | 81.9% | 0.217 | fused_ffn (20) | fused_ffn (1.651x; 20) |
| 57 | outer_01.transformer_2.block_05.attention_qk_rope | qk_rope | qk_rope; batchSharedPackedFusedQKRoPE19Half2Kernel; g361x1x1; b192x1x1; r28; s0 | 40 | 5.616 | 7.616 | 1.356x | 100.0% | 0.084 | library_gemm (20) | linear2_residual (1.530x; 20) |
| 58 | outer_01.transformer_2.block_05.attention_fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 40 | 11.872 | 16.608 | 1.399x | 71.8% | 0.188 | linear2_residual (20) | linear2_residual (1.403x; 20) |
| 59 | outer_01.transformer_2.block_05.attention_out_projection_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 40 | 8.464 | 20.704 | 2.446x | 97.8% | 0.486 | fused_ffn (20) | fused_ffn (2.620x; 20) |
| 60 | outer_01.transformer_2.block_05.ffn_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 40 | 2.432 | 3.248 | 1.336x | 92.4% | 0.028 | linear2_residual (20) | linear2_residual (1.375x; 20) |
| 61 | outer_01.transformer_2.block_05.ffn_linear1_gate_swiglu | fused_ffn | fused_ffn; fused_ffn_candidate_a_reuse_kernel; g18x37x1; b128x1x1; r136; s32768 | 40 | 38.529 | 49.761 | 1.292x | 69.1% | 0.519 | linear2_residual (20) | wide_qkv (1.419x; 18) |
| 62 | outer_01.transformer_2.block_05.ffn_linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 40 | 20.736 | 29.297 | 1.413x | 96.7% | 0.281 | wide_qkv (22) | fa4 (1.461x; 16) |
| 63 | outer_01.post_norm_silu | affine_silu | affine_silu; affineSiluHalf2Kernel; g4693x1x1; b192x1x1; r17; s0 | 40 | 3.328 | 4.672 | 1.404x | 100.0% | 0.056 | fa4 (38) | fa4 (1.404x; 38) |
| 64 | outer_01.post_projection_c384_to_c768_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b256x1x1; r154; s73728 | 40 | 14.384 | 16.881 | 1.174x | 97.5% | 0.107 | library_gemm (38) | library_gemm (1.172x; 38) |
| 65 | outer_02.pre_norm_silu | affine_silu | affine_silu; affineSiluHalf2Kernel; g4693x1x1; b384x1x1; r17; s0 | 40 | 5.024 | 7.008 | 1.395x | 73.3% | 0.080 | library_gemm (38) | library_gemm (1.395x; 38) |
| 66 | outer_02.pre_projection_c768_to_c384 | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 40 | 11.921 | 15.136 | 1.270x | 96.3% | 0.130 | fused_ffn (38) | fused_ffn (1.270x; 38) |
| 67 | outer_02.transformer_0.block_06.attention_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 40 | 2.416 | 5.024 | 2.079x | 100.0% | 0.098 | fused_ffn (38) | fused_ffn (2.086x; 38) |
| 68 | outer_02.transformer_0.block_06.attention_qkv_projection | wide_qkv | wide_qkv; kernel_cutlass_kernel___main__FixedAtomLayoutGemmmakelocals_Kernel_object_at__CopyAtom_ThrID10_TVLayoutSrc1819201_TVLayoutDst1819201_Valuetypef16_tensor00o4693384111100_CopyAtom_ThrID10_T_0; g1x1x170; b288x1x1; r107; s99328 | 40 | 14.064 | 27.201 | 1.934x | 97.9% | 0.517 | fused_ffn (38) | fused_ffn (1.936x; 38) |
| 69 | outer_02.transformer_0.block_06.attention_qk_rope | qk_rope | qk_rope; batchSharedPackedFusedQKRoPE19Half2Kernel; g361x1x1; b192x1x1; r28; s0 | 40 | 5.600 | 8.480 | 1.514x | 99.9% | 0.113 | linear2_residual (38) | linear2_residual (1.517x; 38) |
| 70 | outer_02.transformer_0.block_06.attention_fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 40 | 11.856 | 17.168 | 1.448x | 67.4% | 0.211 | affine_silu (18) | wide_qkv (1.565x; 17) |
| 71 | outer_02.transformer_0.block_06.attention_out_projection_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 40 | 8.480 | 16.832 | 1.985x | 96.9% | 0.323 | library_gemm (19) | library_gemm (2.257x; 19) |
| 72 | outer_02.transformer_0.block_06.ffn_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 40 | 2.432 | 3.280 | 1.349x | 87.8% | 0.034 | fa4 (20) | fa4 (1.513x; 20) |
| 73 | outer_02.transformer_0.block_06.ffn_linear1_gate_swiglu | fused_ffn | fused_ffn; fused_ffn_candidate_a_reuse_kernel; g18x37x1; b128x1x1; r136; s32768 | 40 | 40.257 | 46.385 | 1.152x | 55.8% | 0.314 | library_gemm (19) | wide_qkv (1.346x; 18) |
| 74 | outer_02.transformer_0.block_06.ffn_linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 40 | 20.880 | 31.408 | 1.504x | 96.7% | 0.345 | fused_ffn (20) | fused_ffn (1.603x; 20) |
| 75 | outer_02.transformer_1.block_07.attention_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 40 | 2.464 | 4.144 | 1.682x | 100.0% | 0.064 | fa4 (20) | fused_ffn (1.935x; 20) |
| 76 | outer_02.transformer_1.block_07.attention_qkv_projection | wide_qkv | wide_qkv; kernel_cutlass_kernel___main__FixedAtomLayoutGemmmakelocals_Kernel_object_at__CopyAtom_ThrID10_TVLayoutSrc1819201_TVLayoutDst1819201_Valuetypef16_tensor00o4693384111100_CopyAtom_ThrID10_T_0; g1x1x170; b288x1x1; r107; s99328 | 40 | 14.113 | 20.497 | 1.452x | 84.9% | 0.256 | fused_ffn (20) | fused_ffn (1.703x; 20) |
| 77 | outer_02.transformer_1.block_07.attention_qk_rope | qk_rope | qk_rope; batchSharedPackedFusedQKRoPE19Half2Kernel; g361x1x1; b192x1x1; r28; s0 | 40 | 5.632 | 7.729 | 1.372x | 99.4% | 0.085 | library_gemm (20) | linear2_residual (1.548x; 20) |
| 78 | outer_02.transformer_1.block_07.attention_fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 40 | 11.904 | 17.792 | 1.495x | 75.2% | 0.230 | linear2_residual (20) | linear2_residual (1.513x; 20) |
| 79 | outer_02.transformer_1.block_07.attention_out_projection_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 40 | 8.496 | 17.360 | 2.043x | 96.4% | 0.386 | fused_ffn (20) | fused_ffn (2.606x; 20) |
| 80 | outer_02.transformer_1.block_07.ffn_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 40 | 2.464 | 3.488 | 1.416x | 95.9% | 0.041 | fa4 (20) | fa4 (1.506x; 20) |
| 81 | outer_02.transformer_1.block_07.ffn_linear1_gate_swiglu | fused_ffn | fused_ffn; fused_ffn_candidate_a_reuse_kernel; g18x37x1; b128x1x1; r136; s32768 | 40 | 40.977 | 46.145 | 1.126x | 62.7% | 0.210 | library_gemm (20) | linear2_residual (1.213x; 20) |
| 82 | outer_02.transformer_1.block_07.ffn_linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 40 | 20.832 | 31.888 | 1.531x | 97.0% | 0.447 | fused_ffn (20) | fused_ffn (1.617x; 20) |
| 83 | outer_02.transformer_2.block_08.attention_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 40 | 2.449 | 3.920 | 1.601x | 100.0% | 0.058 | fa4 (20) | fused_ffn (1.941x; 20) |
| 84 | outer_02.transformer_2.block_08.attention_qkv_projection | wide_qkv | wide_qkv; kernel_cutlass_kernel___main__FixedAtomLayoutGemmmakelocals_Kernel_object_at__CopyAtom_ThrID10_TVLayoutSrc1819201_TVLayoutDst1819201_Valuetypef16_tensor00o4693384111100_CopyAtom_ThrID10_T_0; g1x1x170; b288x1x1; r107; s99328 | 40 | 14.113 | 19.857 | 1.407x | 83.4% | 0.236 | fused_ffn (20) | fused_ffn (1.681x; 20) |
| 85 | outer_02.transformer_2.block_08.attention_qk_rope | qk_rope | qk_rope; batchSharedPackedFusedQKRoPE19Half2Kernel; g361x1x1; b192x1x1; r28; s0 | 40 | 5.664 | 7.824 | 1.381x | 99.8% | 0.084 | library_gemm (20) | linear2_residual (1.506x; 20) |
| 86 | outer_02.transformer_2.block_08.attention_fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 40 | 11.888 | 16.704 | 1.405x | 72.8% | 0.200 | linear2_residual (20) | linear2_residual (1.413x; 20) |
| 87 | outer_02.transformer_2.block_08.attention_out_projection_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 40 | 8.544 | 20.785 | 2.433x | 97.9% | 0.492 | fused_ffn (20) | fused_ffn (2.646x; 20) |
| 88 | outer_02.transformer_2.block_08.ffn_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 40 | 2.448 | 3.152 | 1.288x | 92.2% | 0.025 | linear2_residual (20) | linear2_residual (1.333x; 20) |
| 89 | outer_02.transformer_2.block_08.ffn_linear1_gate_swiglu | fused_ffn | fused_ffn; fused_ffn_candidate_a_reuse_kernel; g18x37x1; b128x1x1; r136; s32768 | 40 | 37.920 | 50.881 | 1.342x | 69.7% | 0.571 | linear2_residual (20) | wide_qkv (1.462x; 17) |
| 90 | outer_02.transformer_2.block_08.ffn_linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 40 | 20.817 | 29.680 | 1.426x | 96.7% | 0.302 | wide_qkv (24) | fa4 (1.477x; 14) |
| 91 | outer_02.post_norm_silu | affine_silu | affine_silu; affineSiluHalf2Kernel; g4693x1x1; b192x1x1; r17; s0 | 40 | 3.392 | 4.800 | 1.415x | 100.0% | 0.059 | fa4 (38) | fa4 (1.415x; 38) |
| 92 | outer_02.post_projection_c384_to_c768_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b256x1x1; r154; s73728 | 40 | 14.480 | 17.136 | 1.183x | 97.7% | 0.113 | library_gemm (38) | library_gemm (1.181x; 38) |
| 93 | outer_03.pre_norm_silu | affine_silu | affine_silu; affineSiluHalf2Kernel; g4693x1x1; b384x1x1; r17; s0 | 40 | 5.056 | 7.104 | 1.405x | 73.1% | 0.083 | library_gemm (35) | library_gemm (1.405x; 35) |
| 94 | outer_03.pre_projection_c768_to_c384 | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 40 | 11.952 | 15.360 | 1.285x | 96.5% | 0.142 | fused_ffn (38) | fused_ffn (1.285x; 38) |
| 95 | outer_03.transformer_0.block_09.attention_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 40 | 2.417 | 5.152 | 2.132x | 100.0% | 0.105 | fused_ffn (38) | fused_ffn (2.132x; 38) |
| 96 | outer_03.transformer_0.block_09.attention_qkv_projection | wide_qkv | wide_qkv; kernel_cutlass_kernel___main__FixedAtomLayoutGemmmakelocals_Kernel_object_at__CopyAtom_ThrID10_TVLayoutSrc1819201_TVLayoutDst1819201_Valuetypef16_tensor00o4693384111100_CopyAtom_ThrID10_T_0; g1x1x170; b288x1x1; r107; s99328 | 40 | 14.257 | 27.088 | 1.900x | 97.9% | 0.506 | fused_ffn (38) | fused_ffn (1.901x; 38) |
| 97 | outer_03.transformer_0.block_09.attention_qk_rope | qk_rope | qk_rope; batchSharedPackedFusedQKRoPE19Half2Kernel; g361x1x1; b192x1x1; r28; s0 | 40 | 5.616 | 8.544 | 1.521x | 99.9% | 0.114 | linear2_residual (38) | linear2_residual (1.521x; 38) |
| 98 | outer_03.transformer_0.block_09.attention_fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 40 | 11.920 | 17.456 | 1.464x | 69.1% | 0.219 | wide_qkv (19) | wide_qkv (1.522x; 19) |
| 99 | outer_03.transformer_0.block_09.attention_out_projection_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 40 | 8.512 | 16.800 | 1.974x | 97.2% | 0.309 | library_gemm (19) | library_gemm (2.229x; 19) |
| 100 | outer_03.transformer_0.block_09.ffn_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 40 | 2.432 | 3.152 | 1.296x | 79.1% | 0.027 | fa4 (19) | fa4 (1.368x; 19) |
| 101 | outer_03.transformer_0.block_09.ffn_linear1_gate_swiglu | fused_ffn | fused_ffn; fused_ffn_candidate_a_reuse_kernel; g18x37x1; b128x1x1; r136; s32768 | 40 | 38.032 | 45.905 | 1.207x | 58.8% | 0.392 | library_gemm (23) | wide_qkv (1.430x; 12) |
| 102 | outer_03.transformer_0.block_09.ffn_linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 40 | 20.768 | 30.272 | 1.458x | 96.9% | 0.321 | fused_ffn (20) | fused_ffn (1.582x; 20) |
| 103 | outer_03.transformer_1.block_10.attention_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 40 | 2.464 | 4.128 | 1.675x | 100.0% | 0.063 | fa4 (20) | fused_ffn (1.935x; 20) |
| 104 | outer_03.transformer_1.block_10.attention_qkv_projection | wide_qkv | wide_qkv; kernel_cutlass_kernel___main__FixedAtomLayoutGemmmakelocals_Kernel_object_at__CopyAtom_ThrID10_TVLayoutSrc1819201_TVLayoutDst1819201_Valuetypef16_tensor00o4693384111100_CopyAtom_ThrID10_T_0; g1x1x170; b288x1x1; r107; s99328 | 40 | 14.081 | 19.825 | 1.408x | 84.2% | 0.232 | fused_ffn (20) | fused_ffn (1.632x; 20) |
| 105 | outer_03.transformer_1.block_10.attention_qk_rope | qk_rope | qk_rope; batchSharedPackedFusedQKRoPE19Half2Kernel; g361x1x1; b192x1x1; r28; s0 | 40 | 5.600 | 7.856 | 1.403x | 98.6% | 0.079 | library_gemm (20) | linear2_residual (1.503x; 20) |
| 106 | outer_03.transformer_1.block_10.attention_fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 40 | 11.824 | 17.664 | 1.494x | 77.4% | 0.234 | linear2_residual (20) | fused_ffn (1.604x; 18) |
| 107 | outer_03.transformer_1.block_10.attention_out_projection_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 40 | 8.400 | 16.512 | 1.966x | 96.2% | 0.367 | fused_ffn (20) | fused_ffn (2.566x; 20) |
| 108 | outer_03.transformer_1.block_10.ffn_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 40 | 2.432 | 3.360 | 1.382x | 95.8% | 0.039 | fa4 (20) | fa4 (1.461x; 20) |
| 109 | outer_03.transformer_1.block_10.ffn_linear1_gate_swiglu | fused_ffn | fused_ffn; fused_ffn_candidate_a_reuse_kernel; g18x37x1; b128x1x1; r136; s32768 | 40 | 40.193 | 44.592 | 1.109x | 62.7% | 0.185 | linear2_residual (20) | linear2_residual (1.211x; 20) |
| 110 | outer_03.transformer_1.block_10.ffn_linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 40 | 20.640 | 31.073 | 1.505x | 96.8% | 0.421 | fused_ffn (20) | fused_ffn (1.593x; 20) |
| 111 | outer_03.transformer_2.block_11.attention_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 40 | 2.368 | 3.968 | 1.676x | 100.0% | 0.064 | fa4 (20) | fused_ffn (2.054x; 20) |
| 112 | outer_03.transformer_2.block_11.attention_qkv_projection | wide_qkv | wide_qkv; kernel_cutlass_kernel___main__FixedAtomLayoutGemmmakelocals_Kernel_object_at__CopyAtom_ThrID10_TVLayoutSrc1819201_TVLayoutDst1819201_Valuetypef16_tensor00o4693384111100_CopyAtom_ThrID10_T_0; g1x1x170; b288x1x1; r107; s99328 | 40 | 14.208 | 19.168 | 1.349x | 83.3% | 0.207 | fused_ffn (20) | fused_ffn (1.630x; 20) |
| 113 | outer_03.transformer_2.block_11.attention_qk_rope | qk_rope | qk_rope; batchSharedPackedFusedQKRoPE19Half2Kernel; g361x1x1; b192x1x1; r28; s0 | 40 | 5.568 | 7.456 | 1.339x | 99.7% | 0.078 | library_gemm (20) | linear2_residual (1.489x; 20) |
| 114 | outer_03.transformer_2.block_11.attention_fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 40 | 11.776 | 16.256 | 1.380x | 73.3% | 0.187 | linear2_residual (20) | linear2_residual (1.383x; 20) |
| 115 | outer_03.transformer_2.block_11.attention_out_projection_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 40 | 8.416 | 19.473 | 2.314x | 97.9% | 0.467 | fused_ffn (20) | fused_ffn (2.572x; 20) |
| 116 | outer_03.transformer_2.block_11.ffn_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 40 | 2.464 | 3.120 | 1.266x | 91.8% | 0.022 | linear2_residual (20) | linear2_residual (1.312x; 20) |
| 117 | outer_03.transformer_2.block_11.ffn_linear1_gate_swiglu | fused_ffn | fused_ffn; fused_ffn_candidate_a_reuse_kernel; g18x37x1; b128x1x1; r136; s32768 | 40 | 40.608 | 49.169 | 1.211x | 69.8% | 0.390 | linear2_residual (20) | wide_qkv (1.314x; 15) |
| 118 | outer_03.transformer_2.block_11.ffn_linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 40 | 20.529 | 28.273 | 1.377x | 96.6% | 0.264 | wide_qkv (28) | fa4 (1.439x; 10) |
| 119 | outer_03.post_norm_silu | affine_silu | affine_silu; affineSiluHalf2Kernel; g4693x1x1; b192x1x1; r17; s0 | 40 | 3.296 | 4.576 | 1.388x | 100.0% | 0.055 | fa4 (38) | fa4 (1.388x; 38) |
| 120 | outer_03.post_projection_c384_to_c768_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b256x1x1; r154; s73728 | 40 | 14.192 | 16.480 | 1.161x | 97.4% | 0.099 | library_gemm (38) | library_gemm (1.161x; 38) |
| 121 | outer_04.pre_norm_silu | affine_silu | affine_silu; affineSiluHalf2Kernel; g4693x1x1; b384x1x1; r17; s0 | 40 | 5.008 | 6.992 | 1.396x | 74.3% | 0.082 | library_gemm (38) | library_gemm (1.396x; 38) |
| 122 | outer_04.pre_projection_c768_to_c384 | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 40 | 11.745 | 14.704 | 1.252x | 96.2% | 0.125 | fused_ffn (38) | fused_ffn (1.252x; 38) |
| 123 | outer_04.transformer_0.block_12.attention_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 40 | 2.432 | 4.976 | 2.046x | 100.0% | 0.096 | fused_ffn (38) | fused_ffn (2.059x; 38) |
| 124 | outer_04.transformer_0.block_12.attention_qkv_projection | wide_qkv | wide_qkv; kernel_cutlass_kernel___main__FixedAtomLayoutGemmmakelocals_Kernel_object_at__CopyAtom_ThrID10_TVLayoutSrc1819201_TVLayoutDst1819201_Valuetypef16_tensor00o4693384111100_CopyAtom_ThrID10_T_0; g1x1x170; b288x1x1; r107; s99328 | 40 | 14.033 | 25.824 | 1.840x | 97.8% | 0.460 | fused_ffn (38) | fused_ffn (1.846x; 38) |
| 125 | outer_04.transformer_0.block_12.attention_qk_rope | qk_rope | qk_rope; batchSharedPackedFusedQKRoPE19Half2Kernel; g361x1x1; b192x1x1; r28; s0 | 40 | 5.568 | 8.480 | 1.523x | 99.9% | 0.114 | linear2_residual (38) | linear2_residual (1.526x; 38) |
| 126 | outer_04.transformer_0.block_12.attention_fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 40 | 11.792 | 17.568 | 1.490x | 73.3% | 0.202 | linear2_residual (20) | linear2_residual (1.554x; 20) |
| 127 | outer_04.transformer_0.block_12.attention_out_projection_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 40 | 8.480 | 16.624 | 1.960x | 96.8% | 0.316 | library_gemm (19) | library_gemm (2.219x; 19) |
| 128 | outer_04.transformer_0.block_12.ffn_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 40 | 2.448 | 3.184 | 1.301x | 84.3% | 0.027 | fa4 (20) | fa4 (1.477x; 20) |
| 129 | outer_04.transformer_0.block_12.ffn_linear1_gate_swiglu | fused_ffn | fused_ffn; fused_ffn_candidate_a_reuse_kernel; g18x37x1; b128x1x1; r136; s32768 | 40 | 38.624 | 45.281 | 1.172x | 59.8% | 0.249 | library_gemm (22) | wide_qkv (1.284x; 15) |
| 130 | outer_04.transformer_0.block_12.ffn_linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 40 | 20.832 | 30.032 | 1.442x | 96.8% | 0.359 | fused_ffn (20) | fused_ffn (1.598x; 20) |
| 131 | outer_04.transformer_1.block_13.attention_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 40 | 2.416 | 4.544 | 1.881x | 100.0% | 0.094 | fa4 (20) | fused_ffn (2.391x; 20) |
| 132 | outer_04.transformer_1.block_13.attention_qkv_projection | wide_qkv | wide_qkv; kernel_cutlass_kernel___main__FixedAtomLayoutGemmmakelocals_Kernel_object_at__CopyAtom_ThrID10_TVLayoutSrc1819201_TVLayoutDst1819201_Valuetypef16_tensor00o4693384111100_CopyAtom_ThrID10_T_0; g1x1x170; b288x1x1; r107; s99328 | 40 | 14.144 | 21.328 | 1.508x | 86.8% | 0.294 | fused_ffn (20) | fused_ffn (1.838x; 20) |
| 133 | outer_04.transformer_1.block_13.attention_qk_rope | qk_rope | qk_rope; batchSharedPackedFusedQKRoPE19Half2Kernel; g361x1x1; b192x1x1; r28; s0 | 40 | 5.632 | 7.712 | 1.369x | 99.3% | 0.085 | library_gemm (20) | linear2_residual (1.526x; 20) |
| 134 | outer_04.transformer_1.block_13.attention_fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 40 | 11.952 | 18.224 | 1.525x | 75.7% | 0.248 | linear2_residual (20) | linear2_residual (1.540x; 20) |
| 135 | outer_04.transformer_1.block_13.attention_out_projection_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 40 | 8.512 | 16.736 | 1.966x | 96.3% | 0.360 | fused_ffn (20) | fused_ffn (2.472x; 20) |
| 136 | outer_04.transformer_1.block_13.ffn_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 40 | 2.433 | 3.424 | 1.408x | 93.9% | 0.042 | fa4 (20) | fa4 (1.480x; 20) |
| 137 | outer_04.transformer_1.block_13.ffn_linear1_gate_swiglu | fused_ffn | fused_ffn; fused_ffn_candidate_a_reuse_kernel; g18x37x1; b128x1x1; r136; s32768 | 40 | 38.416 | 46.369 | 1.207x | 71.7% | 0.298 | library_gemm (20) | linear2_residual (1.292x; 20) |
| 138 | outer_04.transformer_1.block_13.ffn_linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 40 | 20.832 | 32.065 | 1.539x | 97.0% | 0.431 | fused_ffn (20) | fused_ffn (1.596x; 20) |
| 139 | outer_04.transformer_2.block_14.attention_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 40 | 2.432 | 4.496 | 1.849x | 100.0% | 0.082 | fa4 (20) | fused_ffn (2.243x; 20) |
| 140 | outer_04.transformer_2.block_14.attention_qkv_projection | wide_qkv | wide_qkv; kernel_cutlass_kernel___main__FixedAtomLayoutGemmmakelocals_Kernel_object_at__CopyAtom_ThrID10_TVLayoutSrc1819201_TVLayoutDst1819201_Valuetypef16_tensor00o4693384111100_CopyAtom_ThrID10_T_0; g1x1x170; b288x1x1; r107; s99328 | 40 | 14.144 | 21.632 | 1.529x | 86.3% | 0.359 | fused_ffn (20) | fused_ffn (2.101x; 20) |
| 141 | outer_04.transformer_2.block_14.attention_qk_rope | qk_rope | qk_rope; batchSharedPackedFusedQKRoPE19Half2Kernel; g361x1x1; b192x1x1; r28; s0 | 40 | 5.633 | 7.776 | 1.381x | 99.4% | 0.090 | library_gemm (20) | linear2_residual (1.574x; 20) |
| 142 | outer_04.transformer_2.block_14.attention_fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 40 | 11.936 | 17.633 | 1.477x | 74.0% | 0.234 | linear2_residual (20) | linear2_residual (1.491x; 20) |
| 143 | outer_04.transformer_2.block_14.attention_out_projection_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 40 | 8.512 | 20.721 | 2.434x | 97.8% | 0.485 | fused_ffn (20) | fused_ffn (2.532x; 20) |
| 144 | outer_04.transformer_2.block_14.ffn_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 40 | 2.432 | 3.056 | 1.257x | 91.9% | 0.025 | linear2_residual (20) | linear2_residual (1.316x; 20) |
| 145 | outer_04.transformer_2.block_14.ffn_linear1_gate_swiglu | fused_ffn | fused_ffn; fused_ffn_candidate_a_reuse_kernel; g18x37x1; b128x1x1; r136; s32768 | 40 | 35.441 | 50.880 | 1.436x | 82.6% | 0.597 | linear2_residual (20) | wide_qkv (1.454x; 9) |
| 146 | outer_04.transformer_2.block_14.ffn_linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 40 | 20.960 | 28.385 | 1.354x | 96.9% | 0.306 | wide_qkv (28) | fa4 (1.370x; 10) |
| 147 | outer_04.post_norm_silu | affine_silu | affine_silu; affineSiluHalf2Kernel; g4693x1x1; b192x1x1; r17; s0 | 40 | 3.361 | 5.472 | 1.628x | 100.0% | 0.085 | fa4 (38) | fa4 (1.628x; 38) |
| 148 | outer_04.post_projection_c384_to_c768_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b256x1x1; r154; s73728 | 40 | 14.528 | 17.680 | 1.217x | 97.5% | 0.135 | library_gemm (38) | library_gemm (1.215x; 38) |
| 149 | outer_05.pre_norm_silu | affine_silu | affine_silu; affineSiluHalf2Kernel; g4693x1x1; b384x1x1; r17; s0 | 40 | 5.120 | 7.296 | 1.425x | 83.0% | 0.089 | library_gemm (38) | library_gemm (1.425x; 38) |
| 150 | outer_05.pre_projection_c768_to_c384 | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 40 | 12.049 | 16.048 | 1.332x | 96.3% | 0.164 | fused_ffn (38) | fused_ffn (1.332x; 38) |
| 151 | outer_05.transformer_0.block_15.attention_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 40 | 2.465 | 5.152 | 2.090x | 100.0% | 0.103 | fused_ffn (38) | fused_ffn (2.090x; 38) |
| 152 | outer_05.transformer_0.block_15.attention_qkv_projection | wide_qkv | wide_qkv; kernel_cutlass_kernel___main__FixedAtomLayoutGemmmakelocals_Kernel_object_at__CopyAtom_ThrID10_TVLayoutSrc1819201_TVLayoutDst1819201_Valuetypef16_tensor00o4693384111100_CopyAtom_ThrID10_T_0; g1x1x170; b288x1x1; r107; s99328 | 40 | 14.113 | 26.833 | 1.901x | 97.9% | 0.522 | fused_ffn (38) | fused_ffn (1.949x; 38) |
| 153 | outer_05.transformer_0.block_15.attention_qk_rope | qk_rope | qk_rope; batchSharedPackedFusedQKRoPE19Half2Kernel; g361x1x1; b192x1x1; r28; s0 | 40 | 5.680 | 8.720 | 1.535x | 99.9% | 0.121 | linear2_residual (38) | linear2_residual (1.538x; 38) |
| 154 | outer_05.transformer_0.block_15.attention_fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 40 | 11.952 | 18.304 | 1.531x | 78.7% | 0.256 | linear2_residual (30) | wide_qkv (1.582x; 9) |
| 155 | outer_05.transformer_0.block_15.attention_out_projection_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 40 | 8.512 | 17.456 | 2.051x | 97.1% | 0.336 | library_gemm (19) | library_gemm (2.346x; 19) |
| 156 | outer_05.transformer_0.block_15.ffn_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 40 | 2.433 | 3.392 | 1.394x | 84.9% | 0.035 | fa4 (20) | fa4 (1.506x; 20) |
| 157 | outer_05.transformer_0.block_15.ffn_linear1_gate_swiglu | fused_ffn | fused_ffn; fused_ffn_candidate_a_reuse_kernel; g18x37x1; b128x1x1; r136; s32768 | 40 | 41.152 | 47.408 | 1.152x | 58.8% | 0.333 | library_gemm (19) | wide_qkv (1.378x; 19) |
| 158 | outer_05.transformer_0.block_15.ffn_linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 40 | 20.976 | 30.641 | 1.461x | 97.0% | 0.365 | fused_ffn (20) | fused_ffn (1.614x; 20) |
| 159 | outer_05.transformer_1.block_16.attention_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 40 | 2.432 | 4.208 | 1.730x | 100.0% | 0.072 | fa4 (20) | fused_ffn (1.980x; 20) |
| 160 | outer_05.transformer_1.block_16.attention_qkv_projection | wide_qkv | wide_qkv; kernel_cutlass_kernel___main__FixedAtomLayoutGemmmakelocals_Kernel_object_at__CopyAtom_ThrID10_TVLayoutSrc1819201_TVLayoutDst1819201_Valuetypef16_tensor00o4693384111100_CopyAtom_ThrID10_T_0; g1x1x170; b288x1x1; r107; s99328 | 40 | 14.240 | 19.952 | 1.401x | 83.8% | 0.243 | fused_ffn (20) | fused_ffn (1.662x; 20) |
| 161 | outer_05.transformer_1.block_16.attention_qk_rope | qk_rope | qk_rope; batchSharedPackedFusedQKRoPE19Half2Kernel; g361x1x1; b192x1x1; r28; s0 | 40 | 5.600 | 7.745 | 1.383x | 99.1% | 0.084 | library_gemm (20) | linear2_residual (1.529x; 20) |
| 162 | outer_05.transformer_1.block_16.attention_fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 40 | 11.921 | 17.729 | 1.487x | 75.8% | 0.239 | linear2_residual (20) | linear2_residual (1.491x; 20) |
| 163 | outer_05.transformer_1.block_16.attention_out_projection_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 40 | 8.496 | 17.456 | 2.054x | 96.7% | 0.402 | fused_ffn (20) | fused_ffn (2.642x; 20) |
| 164 | outer_05.transformer_1.block_16.ffn_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 40 | 2.432 | 3.312 | 1.362x | 95.8% | 0.040 | fa4 (20) | fa4 (1.526x; 20) |
| 165 | outer_05.transformer_1.block_16.ffn_linear1_gate_swiglu | fused_ffn | fused_ffn; fused_ffn_candidate_a_reuse_kernel; g18x37x1; b128x1x1; r136; s32768 | 40 | 40.816 | 46.048 | 1.128x | 61.8% | 0.231 | linear2_residual (20) | linear2_residual (1.235x; 20) |
| 166 | outer_05.transformer_1.block_16.ffn_linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 40 | 20.848 | 31.776 | 1.524x | 97.1% | 0.441 | fused_ffn (20) | fused_ffn (1.605x; 20) |
| 167 | outer_05.transformer_2.block_17.attention_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 40 | 2.432 | 3.952 | 1.625x | 100.0% | 0.060 | fa4 (20) | fused_ffn (1.987x; 20) |
| 168 | outer_05.transformer_2.block_17.attention_qkv_projection | wide_qkv | wide_qkv; kernel_cutlass_kernel___main__FixedAtomLayoutGemmmakelocals_Kernel_object_at__CopyAtom_ThrID10_TVLayoutSrc1819201_TVLayoutDst1819201_Valuetypef16_tensor00o4693384111100_CopyAtom_ThrID10_T_0; g1x1x170; b288x1x1; r107; s99328 | 40 | 14.145 | 19.840 | 1.403x | 82.6% | 0.226 | fused_ffn (20) | fused_ffn (1.666x; 20) |
| 169 | outer_05.transformer_2.block_17.attention_qk_rope | qk_rope | qk_rope; batchSharedPackedFusedQKRoPE19Half2Kernel; g361x1x1; b192x1x1; r28; s0 | 40 | 5.568 | 7.680 | 1.379x | 99.7% | 0.086 | library_gemm (20) | linear2_residual (1.532x; 20) |
| 170 | outer_05.transformer_2.block_17.attention_fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 40 | 11.872 | 16.576 | 1.396x | 73.3% | 0.197 | linear2_residual (20) | fused_ffn (1.402x; 13) |
| 171 | outer_05.transformer_2.block_17.attention_out_projection_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 40 | 8.480 | 19.808 | 2.336x | 97.8% | 0.485 | fused_ffn (20) | fused_ffn (2.651x; 20) |
| 172 | outer_05.transformer_2.block_17.ffn_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 40 | 2.401 | 3.136 | 1.306x | 90.5% | 0.026 | linear2_residual (20) | linear2_residual (1.360x; 20) |
| 173 | outer_05.transformer_2.block_17.ffn_linear1_gate_swiglu | fused_ffn | fused_ffn; fused_ffn_candidate_a_reuse_kernel; g18x37x1; b128x1x1; r136; s32768 | 40 | 38.849 | 50.432 | 1.298x | 69.9% | 0.512 | linear2_residual (20) | wide_qkv (1.405x; 19) |
| 174 | outer_05.transformer_2.block_17.ffn_linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 40 | 20.753 | 29.169 | 1.406x | 96.6% | 0.274 | wide_qkv (28) | fa4 (1.456x; 10) |
| 175 | outer_05.post_norm_silu | affine_silu | affine_silu; affineSiluHalf2Kernel; g4693x1x1; b192x1x1; r17; s0 | 40 | 3.296 | 4.688 | 1.422x | 100.0% | 0.058 | fa4 (38) | fa4 (1.422x; 38) |
| 176 | outer_05.post_projection_c384_to_c768_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b256x1x1; r154; s73728 | 40 | 14.336 | 16.736 | 1.167x | 97.5% | 0.104 | library_gemm (38) | library_gemm (1.167x; 38) |
| 177 | outer_06.pre_norm_silu | affine_silu | affine_silu; affineSiluHalf2Kernel; g4693x1x1; b384x1x1; r17; s0 | 40 | 5.024 | 7.104 | 1.414x | 77.3% | 0.085 | library_gemm (38) | library_gemm (1.414x; 38) |
| 178 | outer_06.pre_projection_c768_to_c384 | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 40 | 11.872 | 14.960 | 1.260x | 96.2% | 0.126 | fused_ffn (38) | fused_ffn (1.260x; 38) |
| 179 | outer_06.transformer_0.block_18.attention_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 40 | 2.432 | 4.992 | 2.053x | 100.0% | 0.098 | fused_ffn (38) | fused_ffn (2.066x; 38) |
| 180 | outer_06.transformer_0.block_18.attention_qkv_projection | wide_qkv | wide_qkv; kernel_cutlass_kernel___main__FixedAtomLayoutGemmmakelocals_Kernel_object_at__CopyAtom_ThrID10_TVLayoutSrc1819201_TVLayoutDst1819201_Valuetypef16_tensor00o4693384111100_CopyAtom_ThrID10_T_0; g1x1x170; b288x1x1; r107; s99328 | 40 | 14.048 | 27.552 | 1.961x | 98.0% | 0.533 | fused_ffn (38) | fused_ffn (1.965x; 38) |
| 181 | outer_06.transformer_0.block_18.attention_qk_rope | qk_rope | qk_rope; batchSharedPackedFusedQKRoPE19Half2Kernel; g361x1x1; b192x1x1; r28; s0 | 40 | 5.600 | 8.448 | 1.509x | 99.9% | 0.113 | linear2_residual (38) | linear2_residual (1.509x; 38) |
| 182 | outer_06.transformer_0.block_18.attention_fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 40 | 11.873 | 17.168 | 1.446x | 72.0% | 0.217 | linear2_residual (19) | linear2_residual (1.563x; 19) |
| 183 | outer_06.transformer_0.block_18.attention_out_projection_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 40 | 8.480 | 16.976 | 2.002x | 97.1% | 0.320 | library_gemm (19) | library_gemm (2.208x; 19) |
| 184 | outer_06.transformer_0.block_18.ffn_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 40 | 2.496 | 3.232 | 1.295x | 85.5% | 0.029 | fa4 (20) | fa4 (1.481x; 20) |
| 185 | outer_06.transformer_0.block_18.ffn_linear1_gate_swiglu | fused_ffn | fused_ffn; fused_ffn_candidate_a_reuse_kernel; g18x37x1; b128x1x1; r136; s32768 | 40 | 40.560 | 48.224 | 1.189x | 62.4% | 0.348 | library_gemm (20) | wide_qkv (1.315x; 19) |
| 186 | outer_06.transformer_0.block_18.ffn_linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 40 | 20.896 | 29.728 | 1.423x | 96.9% | 0.307 | fused_ffn (20) | fused_ffn (1.466x; 20) |
| 187 | outer_06.transformer_1.block_19.attention_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 40 | 2.496 | 4.576 | 1.833x | 100.0% | 0.089 | fa4 (20) | fused_ffn (2.314x; 20) |
| 188 | outer_06.transformer_1.block_19.attention_qkv_projection | wide_qkv | wide_qkv; kernel_cutlass_kernel___main__FixedAtomLayoutGemmmakelocals_Kernel_object_at__CopyAtom_ThrID10_TVLayoutSrc1819201_TVLayoutDst1819201_Valuetypef16_tensor00o4693384111100_CopyAtom_ThrID10_T_0; g1x1x170; b288x1x1; r107; s99328 | 40 | 14.112 | 21.921 | 1.553x | 86.7% | 0.336 | fused_ffn (20) | fused_ffn (1.988x; 20) |
| 189 | outer_06.transformer_1.block_19.attention_qk_rope | qk_rope | qk_rope; batchSharedPackedFusedQKRoPE19Half2Kernel; g361x1x1; b192x1x1; r28; s0 | 40 | 5.665 | 7.632 | 1.347x | 99.3% | 0.084 | library_gemm (20) | linear2_residual (1.520x; 20) |
| 190 | outer_06.transformer_1.block_19.attention_fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 40 | 11.968 | 18.656 | 1.559x | 75.9% | 0.256 | linear2_residual (20) | linear2_residual (1.570x; 20) |
| 191 | outer_06.transformer_1.block_19.attention_out_projection_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 40 | 8.496 | 18.145 | 2.136x | 96.9% | 0.395 | fused_ffn (20) | fused_ffn (2.704x; 20) |
| 192 | outer_06.transformer_1.block_19.ffn_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 40 | 2.432 | 3.328 | 1.368x | 88.5% | 0.038 | linear2_residual (20) | fa4 (1.428x; 18) |
| 193 | outer_06.transformer_1.block_19.ffn_linear1_gate_swiglu | fused_ffn | fused_ffn; fused_ffn_candidate_a_reuse_kernel; g18x37x1; b128x1x1; r136; s32768 | 40 | 39.328 | 47.969 | 1.220x | 75.6% | 0.310 | library_gemm (20) | linear2_residual (1.285x; 20) |
| 194 | outer_06.transformer_1.block_19.ffn_linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 40 | 20.992 | 31.232 | 1.488x | 97.0% | 0.432 | fused_ffn (20) | fused_ffn (1.617x; 20) |
| 195 | outer_06.transformer_2.block_20.attention_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 40 | 2.432 | 4.240 | 1.744x | 100.0% | 0.078 | fa4 (20) | fused_ffn (2.197x; 20) |
| 196 | outer_06.transformer_2.block_20.attention_qkv_projection | wide_qkv | wide_qkv; kernel_cutlass_kernel___main__FixedAtomLayoutGemmmakelocals_Kernel_object_at__CopyAtom_ThrID10_TVLayoutSrc1819201_TVLayoutDst1819201_Valuetypef16_tensor00o4693384111100_CopyAtom_ThrID10_T_0; g1x1x170; b288x1x1; r107; s99328 | 40 | 14.272 | 22.304 | 1.563x | 86.1% | 0.348 | fused_ffn (20) | fused_ffn (2.030x; 20) |
| 197 | outer_06.transformer_2.block_20.attention_qk_rope | qk_rope | qk_rope; batchSharedPackedFusedQKRoPE19Half2Kernel; g361x1x1; b192x1x1; r28; s0 | 40 | 5.616 | 8.128 | 1.447x | 98.6% | 0.089 | library_gemm (20) | linear2_residual (1.564x; 20) |
| 198 | outer_06.transformer_2.block_20.attention_fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 40 | 11.968 | 18.273 | 1.527x | 77.5% | 0.267 | fused_ffn (20) | fused_ffn (1.668x; 20) |
| 199 | outer_06.transformer_2.block_20.attention_out_projection_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 40 | 8.480 | 20.736 | 2.445x | 97.8% | 0.508 | fused_ffn (20) | fused_ffn (2.677x; 20) |
| 200 | outer_06.transformer_2.block_20.ffn_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 40 | 2.433 | 3.152 | 1.296x | 90.4% | 0.029 | linear2_residual (20) | linear2_residual (1.375x; 20) |
| 201 | outer_06.transformer_2.block_20.ffn_linear1_gate_swiglu | fused_ffn | fused_ffn; fused_ffn_candidate_a_reuse_kernel; g18x37x1; b128x1x1; r136; s32768 | 40 | 33.697 | 50.352 | 1.494x | 83.0% | 0.643 | linear2_residual (20) | wide_qkv (1.515x; 10) |
| 202 | outer_06.transformer_2.block_20.ffn_linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 40 | 21.088 | 28.384 | 1.346x | 96.9% | 0.305 | wide_qkv (21) | fa4 (1.407x; 17) |
| 203 | outer_06.post_norm_silu | affine_silu | affine_silu; affineSiluHalf2Kernel; g4693x1x1; b192x1x1; r17; s0 | 40 | 3.376 | 5.457 | 1.616x | 100.0% | 0.084 | fa4 (38) | fa4 (1.616x; 38) |
| 204 | outer_06.post_projection_c384_to_c768_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b256x1x1; r154; s73728 | 40 | 14.624 | 17.888 | 1.223x | 97.6% | 0.137 | library_gemm (38) | library_gemm (1.223x; 38) |
| 205 | outer_07.pre_norm_silu | affine_silu | affine_silu; affineSiluHalf2Kernel; g4693x1x1; b384x1x1; r17; s0 | 40 | 5.121 | 7.360 | 1.437x | 78.1% | 0.088 | library_gemm (38) | library_gemm (1.437x; 38) |
| 206 | outer_07.pre_projection_c768_to_c384 | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 40 | 12.049 | 15.969 | 1.325x | 96.3% | 0.167 | fused_ffn (38) | fused_ffn (1.325x; 38) |
| 207 | outer_07.transformer_0.block_21.attention_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 40 | 2.464 | 5.152 | 2.091x | 100.0% | 0.103 | fused_ffn (38) | fused_ffn (2.091x; 38) |
| 208 | outer_07.transformer_0.block_21.attention_qkv_projection | wide_qkv | wide_qkv; kernel_cutlass_kernel___main__FixedAtomLayoutGemmmakelocals_Kernel_object_at__CopyAtom_ThrID10_TVLayoutSrc1819201_TVLayoutDst1819201_Valuetypef16_tensor00o4693384111100_CopyAtom_ThrID10_T_0; g1x1x170; b288x1x1; r107; s99328 | 40 | 14.273 | 26.624 | 1.865x | 97.8% | 0.483 | fused_ffn (38) | fused_ffn (1.869x; 38) |
| 209 | outer_07.transformer_0.block_21.attention_qk_rope | qk_rope | qk_rope; batchSharedPackedFusedQKRoPE19Half2Kernel; g361x1x1; b192x1x1; r28; s0 | 40 | 5.648 | 8.816 | 1.561x | 99.9% | 0.126 | linear2_residual (38) | linear2_residual (1.564x; 38) |
| 210 | outer_07.transformer_0.block_21.attention_fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 40 | 12.000 | 18.560 | 1.547x | 80.6% | 0.265 | linear2_residual (36) | linear2_residual (1.520x; 36) |
| 211 | outer_07.transformer_0.block_21.attention_out_projection_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 40 | 8.544 | 17.712 | 2.073x | 96.9% | 0.354 | library_gemm (19) | library_gemm (2.360x; 19) |
| 212 | outer_07.transformer_0.block_21.ffn_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 40 | 2.432 | 3.296 | 1.355x | 83.3% | 0.036 | fa4 (20) | fa4 (1.533x; 20) |
| 213 | outer_07.transformer_0.block_21.ffn_linear1_gate_swiglu | fused_ffn | fused_ffn; fused_ffn_candidate_a_reuse_kernel; g18x37x1; b128x1x1; r136; s32768 | 40 | 35.776 | 45.681 | 1.277x | 65.0% | 0.375 | library_gemm (25) | wide_qkv (1.431x; 14) |
| 214 | outer_07.transformer_0.block_21.ffn_linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 40 | 21.041 | 30.833 | 1.465x | 96.9% | 0.399 | fused_ffn (20) | fused_ffn (1.641x; 20) |
| 215 | outer_07.transformer_1.block_22.attention_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 40 | 2.432 | 4.401 | 1.809x | 100.0% | 0.088 | fa4 (20) | fused_ffn (2.191x; 20) |
| 216 | outer_07.transformer_1.block_22.attention_qkv_projection | wide_qkv | wide_qkv; kernel_cutlass_kernel___main__FixedAtomLayoutGemmmakelocals_Kernel_object_at__CopyAtom_ThrID10_TVLayoutSrc1819201_TVLayoutDst1819201_Valuetypef16_tensor00o4693384111100_CopyAtom_ThrID10_T_0; g1x1x170; b288x1x1; r107; s99328 | 40 | 14.304 | 23.536 | 1.645x | 86.9% | 0.388 | fused_ffn (20) | fused_ffn (2.123x; 20) |
| 217 | outer_07.transformer_1.block_22.attention_qk_rope | qk_rope | qk_rope; batchSharedPackedFusedQKRoPE19Half2Kernel; g361x1x1; b192x1x1; r28; s0 | 40 | 5.696 | 7.968 | 1.399x | 99.5% | 0.092 | library_gemm (20) | linear2_residual (1.556x; 20) |
| 218 | outer_07.transformer_1.block_22.attention_fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 40 | 12.048 | 18.896 | 1.568x | 75.6% | 0.267 | linear2_residual (20) | linear2_residual (1.590x; 20) |
| 219 | outer_07.transformer_1.block_22.attention_out_projection_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 40 | 8.624 | 17.152 | 1.989x | 96.1% | 0.370 | fused_ffn (20) | fused_ffn (2.399x; 20) |
| 220 | outer_07.transformer_1.block_22.ffn_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 40 | 2.496 | 3.632 | 1.455x | 99.0% | 0.044 | fa4 (20) | fa4 (1.538x; 20) |
| 221 | outer_07.transformer_1.block_22.ffn_linear1_gate_swiglu | fused_ffn | fused_ffn; fused_ffn_candidate_a_reuse_kernel; g18x37x1; b128x1x1; r136; s32768 | 40 | 35.904 | 44.480 | 1.239x | 75.8% | 0.344 | library_gemm (20) | linear2_residual (1.425x; 20) |
| 222 | outer_07.transformer_1.block_22.ffn_linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 40 | 21.056 | 32.897 | 1.562x | 97.1% | 0.480 | fused_ffn (20) | fused_ffn (1.684x; 20) |
| 223 | outer_07.transformer_2.block_23.attention_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 40 | 2.464 | 4.480 | 1.818x | 100.0% | 0.073 | fa4 (20) | fused_ffn (2.156x; 20) |
| 224 | outer_07.transformer_2.block_23.attention_qkv_projection | wide_qkv | wide_qkv; kernel_cutlass_kernel___main__FixedAtomLayoutGemmmakelocals_Kernel_object_at__CopyAtom_ThrID10_TVLayoutSrc1819201_TVLayoutDst1819201_Valuetypef16_tensor00o4693384111100_CopyAtom_ThrID10_T_0; g1x1x170; b288x1x1; r107; s99328 | 40 | 14.257 | 23.136 | 1.623x | 85.9% | 0.370 | fused_ffn (20) | fused_ffn (2.111x; 20) |
| 225 | outer_07.transformer_2.block_23.attention_qk_rope | qk_rope | qk_rope; batchSharedPackedFusedQKRoPE19Half2Kernel; g361x1x1; b192x1x1; r28; s0 | 40 | 5.744 | 7.856 | 1.368x | 99.9% | 0.089 | library_gemm (20) | linear2_residual (1.546x; 20) |
| 226 | outer_07.transformer_2.block_23.attention_fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 40 | 12.064 | 17.793 | 1.475x | 72.8% | 0.229 | linear2_residual (20) | linear2_residual (1.507x; 20) |
| 227 | outer_07.transformer_2.block_23.attention_out_projection_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 40 | 8.608 | 20.289 | 2.357x | 97.7% | 0.458 | fused_ffn (20) | library_gemm (2.364x; 19) |
| 228 | outer_07.transformer_2.block_23.ffn_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 40 | 2.449 | 3.216 | 1.313x | 91.7% | 0.028 | linear2_residual (20) | linear2_residual (1.385x; 20) |
| 229 | outer_07.transformer_2.block_23.ffn_linear1_gate_swiglu | fused_ffn | fused_ffn; fused_ffn_candidate_a_reuse_kernel; g18x37x1; b128x1x1; r136; s32768 | 40 | 35.840 | 51.136 | 1.427x | 83.1% | 0.602 | linear2_residual (20) | wide_qkv (1.429x; 17) |
| 230 | outer_07.transformer_2.block_23.ffn_linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 40 | 21.104 | 28.657 | 1.358x | 97.0% | 0.312 | wide_qkv (20) | fa4 (1.427x; 17) |
| 231 | outer_07.post_norm_silu | affine_silu | affine_silu; affineSiluHalf2Kernel; g4693x1x1; b192x1x1; r17; s0 | 40 | 3.376 | 5.504 | 1.630x | 100.0% | 0.089 | fa4 (38) | fa4 (1.630x; 38) |
| 232 | outer_07.post_projection_c384_to_c768_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b256x1x1; r154; s73728 | 40 | 14.672 | 17.904 | 1.220x | 97.5% | 0.139 | library_gemm (38) | library_gemm (1.219x; 38) |
| 233 | outer_08.pre_norm_silu | affine_silu | affine_silu; affineSiluHalf2Kernel; g4693x1x1; b384x1x1; r17; s0 | 40 | 5.120 | 7.296 | 1.425x | 78.8% | 0.086 | library_gemm (38) | library_gemm (1.422x; 38) |
| 234 | outer_08.pre_projection_c768_to_c384 | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 40 | 12.096 | 16.113 | 1.332x | 96.4% | 0.168 | fused_ffn (38) | fused_ffn (1.332x; 38) |
| 235 | outer_08.transformer_0.block_24.attention_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 40 | 2.464 | 5.104 | 2.071x | 100.0% | 0.100 | fused_ffn (38) | fused_ffn (2.078x; 38) |
| 236 | outer_08.transformer_0.block_24.attention_qkv_projection | wide_qkv | wide_qkv; kernel_cutlass_kernel___main__FixedAtomLayoutGemmmakelocals_Kernel_object_at__CopyAtom_ThrID10_TVLayoutSrc1819201_TVLayoutDst1819201_Valuetypef16_tensor00o4693384111100_CopyAtom_ThrID10_T_0; g1x1x170; b288x1x1; r107; s99328 | 40 | 14.240 | 27.408 | 1.925x | 97.8% | 0.515 | fused_ffn (38) | fused_ffn (1.926x; 38) |
| 237 | outer_08.transformer_0.block_24.attention_qk_rope | qk_rope | qk_rope; batchSharedPackedFusedQKRoPE19Half2Kernel; g361x1x1; b192x1x1; r28; s0 | 40 | 5.696 | 8.720 | 1.531x | 99.9% | 0.120 | linear2_residual (38) | linear2_residual (1.537x; 38) |
| 238 | outer_08.transformer_0.block_24.attention_fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 40 | 12.001 | 18.592 | 1.549x | 81.0% | 0.265 | linear2_residual (37) | linear2_residual (1.544x; 37) |
| 239 | outer_08.transformer_0.block_24.attention_out_projection_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 40 | 8.624 | 17.792 | 2.063x | 97.3% | 0.339 | library_gemm (19) | library_gemm (2.323x; 19) |
| 240 | outer_08.transformer_0.block_24.ffn_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 40 | 2.464 | 3.296 | 1.338x | 84.1% | 0.034 | fa4 (20) | fa4 (1.526x; 20) |
| 241 | outer_08.transformer_0.block_24.ffn_linear1_gate_swiglu | fused_ffn | fused_ffn; fused_ffn_candidate_a_reuse_kernel; g18x37x1; b128x1x1; r136; s32768 | 40 | 40.769 | 48.240 | 1.183x | 61.5% | 0.324 | library_gemm (21) | wide_qkv (1.352x; 18) |
| 242 | outer_08.transformer_0.block_24.ffn_linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 40 | 21.088 | 30.465 | 1.445x | 97.0% | 0.377 | fused_ffn (20) | fused_ffn (1.605x; 20) |
| 243 | outer_08.transformer_1.block_25.attention_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 40 | 2.464 | 4.400 | 1.786x | 100.0% | 0.082 | fa4 (20) | fused_ffn (2.059x; 20) |
| 244 | outer_08.transformer_1.block_25.attention_qkv_projection | wide_qkv | wide_qkv; kernel_cutlass_kernel___main__FixedAtomLayoutGemmmakelocals_Kernel_object_at__CopyAtom_ThrID10_TVLayoutSrc1819201_TVLayoutDst1819201_Valuetypef16_tensor00o4693384111100_CopyAtom_ThrID10_T_0; g1x1x170; b288x1x1; r107; s99328 | 40 | 14.240 | 21.184 | 1.488x | 84.8% | 0.284 | fused_ffn (20) | fused_ffn (1.760x; 20) |
| 245 | outer_08.transformer_1.block_25.attention_qk_rope | qk_rope | qk_rope; batchSharedPackedFusedQKRoPE19Half2Kernel; g361x1x1; b192x1x1; r28; s0 | 40 | 5.665 | 7.728 | 1.364x | 99.4% | 0.088 | library_gemm (20) | linear2_residual (1.545x; 20) |
| 246 | outer_08.transformer_1.block_25.attention_fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 40 | 12.032 | 18.448 | 1.533x | 76.0% | 0.257 | linear2_residual (20) | linear2_residual (1.540x; 20) |
| 247 | outer_08.transformer_1.block_25.attention_out_projection_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 40 | 8.528 | 18.544 | 2.174x | 96.6% | 0.413 | fused_ffn (20) | fused_ffn (2.681x; 20) |
| 248 | outer_08.transformer_1.block_25.ffn_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 40 | 2.448 | 3.568 | 1.458x | 100.0% | 0.044 | fa4 (20) | fa4 (1.575x; 20) |
| 249 | outer_08.transformer_1.block_25.ffn_linear1_gate_swiglu | fused_ffn | fused_ffn; fused_ffn_candidate_a_reuse_kernel; g18x37x1; b128x1x1; r136; s32768 | 40 | 40.800 | 46.353 | 1.136x | 63.1% | 0.268 | library_gemm (20) | linear2_residual (1.279x; 20) |
| 250 | outer_08.transformer_1.block_25.ffn_linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 40 | 20.960 | 32.961 | 1.573x | 97.1% | 0.466 | fused_ffn (20) | fused_ffn (1.623x; 20) |
| 251 | outer_08.transformer_2.block_26.attention_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 40 | 2.481 | 4.096 | 1.651x | 100.0% | 0.064 | fa4 (20) | fused_ffn (2.019x; 20) |
| 252 | outer_08.transformer_2.block_26.attention_qkv_projection | wide_qkv | wide_qkv; kernel_cutlass_kernel___main__FixedAtomLayoutGemmmakelocals_Kernel_object_at__CopyAtom_ThrID10_TVLayoutSrc1819201_TVLayoutDst1819201_Valuetypef16_tensor00o4693384111100_CopyAtom_ThrID10_T_0; g1x1x170; b288x1x1; r107; s99328 | 40 | 14.177 | 20.769 | 1.465x | 84.6% | 0.274 | fused_ffn (20) | fused_ffn (1.805x; 20) |
| 253 | outer_08.transformer_2.block_26.attention_qk_rope | qk_rope | qk_rope; batchSharedPackedFusedQKRoPE19Half2Kernel; g361x1x1; b192x1x1; r28; s0 | 40 | 5.648 | 7.712 | 1.365x | 100.0% | 0.086 | library_gemm (20) | linear2_residual (1.530x; 20) |
| 254 | outer_08.transformer_2.block_26.attention_fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 40 | 12.000 | 17.120 | 1.427x | 71.9% | 0.202 | linear2_residual (20) | linear2_residual (1.433x; 20) |
| 255 | outer_08.transformer_2.block_26.attention_out_projection_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 40 | 8.480 | 20.864 | 2.460x | 97.7% | 0.500 | fused_ffn (20) | fused_ffn (2.657x; 20) |
| 256 | outer_08.transformer_2.block_26.ffn_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 40 | 2.464 | 3.088 | 1.253x | 91.7% | 0.025 | linear2_residual (20) | linear2_residual (1.318x; 20) |
| 257 | outer_08.transformer_2.block_26.ffn_linear1_gate_swiglu | fused_ffn | fused_ffn; fused_ffn_candidate_a_reuse_kernel; g18x37x1; b128x1x1; r136; s32768 | 40 | 40.273 | 51.953 | 1.290x | 72.9% | 0.479 | linear2_residual (20) | wide_qkv (1.373x; 19) |
| 258 | outer_08.transformer_2.block_26.ffn_linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 40 | 20.928 | 29.328 | 1.401x | 97.0% | 0.308 | wide_qkv (29) | fa4 (1.443x; 9) |
| 259 | outer_08.post_norm_silu | affine_silu | affine_silu; affineSiluHalf2Kernel; g4693x1x1; b192x1x1; r17; s0 | 40 | 3.360 | 4.928 | 1.467x | 100.0% | 0.064 | fa4 (38) | fa4 (1.467x; 38) |
| 260 | outer_08.post_projection_c384_to_c768_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b256x1x1; r154; s73728 | 40 | 14.480 | 17.216 | 1.189x | 97.5% | 0.117 | library_gemm (38) | library_gemm (1.189x; 38) |
| 261 | outer_09.pre_norm_silu | affine_silu | affine_silu; affineSiluHalf2Kernel; g4693x1x1; b384x1x1; r17; s0 | 40 | 5.056 | 7.200 | 1.424x | 79.8% | 0.077 | library_gemm (37) | library_gemm (1.424x; 37) |
| 262 | outer_09.pre_projection_c768_to_c384 | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 40 | 12.049 | 15.712 | 1.304x | 96.2% | 0.152 | fused_ffn (38) | fused_ffn (1.304x; 38) |
| 263 | outer_09.transformer_0.block_27.attention_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 40 | 2.432 | 4.960 | 2.039x | 100.0% | 0.095 | fused_ffn (38) | fused_ffn (2.040x; 38) |
| 264 | outer_09.transformer_0.block_27.attention_qkv_projection | wide_qkv | wide_qkv; kernel_cutlass_kernel___main__FixedAtomLayoutGemmmakelocals_Kernel_object_at__CopyAtom_ThrID10_TVLayoutSrc1819201_TVLayoutDst1819201_Valuetypef16_tensor00o4693384111100_CopyAtom_ThrID10_T_0; g1x1x170; b288x1x1; r107; s99328 | 40 | 14.209 | 27.808 | 1.957x | 98.0% | 0.538 | fused_ffn (38) | fused_ffn (1.963x; 38) |
| 265 | outer_09.transformer_0.block_27.attention_qk_rope | qk_rope | qk_rope; batchSharedPackedFusedQKRoPE19Half2Kernel; g361x1x1; b192x1x1; r28; s0 | 40 | 5.632 | 8.464 | 1.503x | 99.9% | 0.113 | linear2_residual (38) | linear2_residual (1.509x; 38) |
| 266 | outer_09.transformer_0.block_27.attention_fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 40 | 11.936 | 17.664 | 1.480x | 72.4% | 0.230 | linear2_residual (21) | wide_qkv (1.530x; 12) |
| 267 | outer_09.transformer_0.block_27.attention_out_projection_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 40 | 8.560 | 16.992 | 1.985x | 97.3% | 0.320 | library_gemm (19) | library_gemm (2.243x; 19) |
| 268 | outer_09.transformer_0.block_27.ffn_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 40 | 2.464 | 3.328 | 1.351x | 83.4% | 0.035 | fa4 (20) | library_gemm (1.442x; 9) |
| 269 | outer_09.transformer_0.block_27.ffn_linear1_gate_swiglu | fused_ffn | fused_ffn; fused_ffn_candidate_a_reuse_kernel; g18x37x1; b128x1x1; r136; s32768 | 40 | 38.752 | 46.544 | 1.201x | 56.6% | 0.399 | wide_qkv (19) | wide_qkv (1.432x; 19) |
| 270 | outer_09.transformer_0.block_27.ffn_linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 40 | 20.768 | 31.041 | 1.495x | 96.9% | 0.336 | fused_ffn (20) | fused_ffn (1.605x; 20) |
| 271 | outer_09.transformer_1.block_28.attention_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 40 | 2.432 | 4.192 | 1.724x | 100.0% | 0.070 | fa4 (20) | fused_ffn (1.994x; 20) |
| 272 | outer_09.transformer_1.block_28.attention_qkv_projection | wide_qkv | wide_qkv; kernel_cutlass_kernel___main__FixedAtomLayoutGemmmakelocals_Kernel_object_at__CopyAtom_ThrID10_TVLayoutSrc1819201_TVLayoutDst1819201_Valuetypef16_tensor00o4693384111100_CopyAtom_ThrID10_T_0; g1x1x170; b288x1x1; r107; s99328 | 40 | 14.208 | 20.064 | 1.412x | 84.6% | 0.228 | fused_ffn (20) | fused_ffn (1.628x; 20) |
| 273 | outer_09.transformer_1.block_28.attention_qk_rope | qk_rope | qk_rope; batchSharedPackedFusedQKRoPE19Half2Kernel; g361x1x1; b192x1x1; r28; s0 | 40 | 5.600 | 7.568 | 1.351x | 99.4% | 0.083 | library_gemm (20) | linear2_residual (1.523x; 20) |
| 274 | outer_09.transformer_1.block_28.attention_fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 40 | 11.873 | 17.360 | 1.462x | 75.8% | 0.227 | linear2_residual (20) | linear2_residual (1.474x; 20) |
| 275 | outer_09.transformer_1.block_28.attention_out_projection_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 40 | 8.496 | 16.848 | 1.983x | 96.3% | 0.372 | fused_ffn (20) | fused_ffn (2.572x; 20) |
| 276 | outer_09.transformer_1.block_28.ffn_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 40 | 2.448 | 3.424 | 1.399x | 96.7% | 0.040 | fa4 (20) | fa4 (1.477x; 20) |
| 277 | outer_09.transformer_1.block_28.ffn_linear1_gate_swiglu | fused_ffn | fused_ffn; fused_ffn_candidate_a_reuse_kernel; g18x37x1; b128x1x1; r136; s32768 | 40 | 38.449 | 44.688 | 1.162x | 62.3% | 0.271 | linear2_residual (20) | linear2_residual (1.276x; 20) |
| 278 | outer_09.transformer_1.block_28.ffn_linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 40 | 20.641 | 31.488 | 1.526x | 97.1% | 0.425 | fused_ffn (20) | fused_ffn (1.591x; 20) |
| 279 | outer_09.transformer_2.block_29.attention_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 40 | 2.400 | 3.984 | 1.660x | 100.0% | 0.061 | fa4 (20) | fused_ffn (2.000x; 20) |
| 280 | outer_09.transformer_2.block_29.attention_qkv_projection | wide_qkv | wide_qkv; kernel_cutlass_kernel___main__FixedAtomLayoutGemmmakelocals_Kernel_object_at__CopyAtom_ThrID10_TVLayoutSrc1819201_TVLayoutDst1819201_Valuetypef16_tensor00o4693384111100_CopyAtom_ThrID10_T_0; g1x1x170; b288x1x1; r107; s99328 | 40 | 14.033 | 19.568 | 1.394x | 83.4% | 0.214 | fused_ffn (20) | fused_ffn (1.653x; 20) |
| 281 | outer_09.transformer_2.block_29.attention_qk_rope | qk_rope | qk_rope; batchSharedPackedFusedQKRoPE19Half2Kernel; g361x1x1; b192x1x1; r28; s0 | 40 | 5.568 | 7.553 | 1.356x | 99.5% | 0.081 | library_gemm (20) | linear2_residual (1.503x; 20) |
| 282 | outer_09.transformer_2.block_29.attention_fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 40 | 11.808 | 16.224 | 1.374x | 73.2% | 0.187 | linear2_residual (20) | linear2_residual (1.382x; 20) |
| 283 | outer_09.transformer_2.block_29.attention_out_projection_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 40 | 8.480 | 19.328 | 2.279x | 97.7% | 0.463 | fused_ffn (20) | fused_ffn (2.555x; 20) |
| 284 | outer_09.transformer_2.block_29.ffn_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 40 | 2.464 | 3.088 | 1.253x | 89.8% | 0.022 | linear2_residual (20) | linear2_residual (1.318x; 20) |
| 285 | outer_09.transformer_2.block_29.ffn_linear1_gate_swiglu | fused_ffn | fused_ffn; fused_ffn_candidate_a_reuse_kernel; g18x37x1; b128x1x1; r136; s32768 | 40 | 38.288 | 48.961 | 1.279x | 69.7% | 0.497 | linear2_residual (20) | wide_qkv (1.398x; 17) |
| 286 | outer_09.transformer_2.block_29.ffn_linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 40 | 20.561 | 29.089 | 1.415x | 96.9% | 0.266 | wide_qkv (27) | fa4 (1.440x; 11) |
| 287 | outer_09.post_norm_silu | affine_silu | affine_silu; affineSiluHalf2Kernel; g4693x1x1; b192x1x1; r17; s0 | 40 | 3.328 | 4.608 | 1.385x | 100.0% | 0.054 | fa4 (38) | fa4 (1.385x; 38) |
| 288 | outer_09.post_projection_c384_to_c768_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b256x1x1; r154; s73728 | 40 | 14.288 | 16.576 | 1.160x | 97.5% | 0.097 | library_gemm (38) | library_gemm (1.159x; 38) |
| 289 | outer_10.pre_norm_silu | affine_silu | affine_silu; affineSiluHalf2Kernel; g4693x1x1; b384x1x1; r17; s0 | 40 | 5.040 | 6.976 | 1.384x | 74.3% | 0.076 | library_gemm (37) | library_gemm (1.384x; 37) |
| 290 | outer_10.pre_projection_c768_to_c384 | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 40 | 11.809 | 14.832 | 1.256x | 95.9% | 0.125 | fused_ffn (38) | fused_ffn (1.256x; 38) |
| 291 | outer_10.transformer_0.block_30.attention_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 40 | 2.464 | 5.088 | 2.065x | 100.0% | 0.102 | fused_ffn (38) | fused_ffn (2.072x; 38) |
| 292 | outer_10.transformer_0.block_30.attention_qkv_projection | wide_qkv | wide_qkv; kernel_cutlass_kernel___main__FixedAtomLayoutGemmmakelocals_Kernel_object_at__CopyAtom_ThrID10_TVLayoutSrc1819201_TVLayoutDst1819201_Valuetypef16_tensor00o4693384111100_CopyAtom_ThrID10_T_0; g1x1x170; b288x1x1; r107; s99328 | 40 | 14.016 | 26.288 | 1.876x | 97.9% | 0.485 | fused_ffn (38) | fused_ffn (1.879x; 38) |
| 293 | outer_10.transformer_0.block_30.attention_qk_rope | qk_rope | qk_rope; batchSharedPackedFusedQKRoPE19Half2Kernel; g361x1x1; b192x1x1; r28; s0 | 40 | 5.584 | 8.528 | 1.527x | 99.9% | 0.115 | linear2_residual (38) | linear2_residual (1.530x; 38) |
| 294 | outer_10.transformer_0.block_30.attention_fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 40 | 11.808 | 17.344 | 1.469x | 71.6% | 0.214 | affine_silu (18) | linear2_residual (1.572x; 18) |
| 295 | outer_10.transformer_0.block_30.attention_out_projection_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 40 | 8.464 | 16.848 | 1.990x | 97.2% | 0.325 | library_gemm (19) | library_gemm (2.212x; 19) |
| 296 | outer_10.transformer_0.block_30.ffn_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 40 | 2.432 | 3.280 | 1.349x | 86.3% | 0.033 | fa4 (20) | fa4 (1.526x; 20) |
| 297 | outer_10.transformer_0.block_30.ffn_linear1_gate_swiglu | fused_ffn | fused_ffn; fused_ffn_candidate_a_reuse_kernel; g18x37x1; b128x1x1; r136; s32768 | 40 | 36.065 | 44.688 | 1.239x | 65.0% | 0.312 | library_gemm (20) | wide_qkv (1.332x; 19) |
| 298 | outer_10.transformer_0.block_30.ffn_linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 40 | 20.864 | 30.256 | 1.450x | 96.9% | 0.343 | fused_ffn (20) | fused_ffn (1.557x; 20) |
| 299 | outer_10.transformer_1.block_31.attention_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 40 | 2.496 | 4.577 | 1.834x | 100.0% | 0.100 | fa4 (20) | fused_ffn (2.583x; 20) |
| 300 | outer_10.transformer_1.block_31.attention_qkv_projection | wide_qkv | wide_qkv; kernel_cutlass_kernel___main__FixedAtomLayoutGemmmakelocals_Kernel_object_at__CopyAtom_ThrID10_TVLayoutSrc1819201_TVLayoutDst1819201_Valuetypef16_tensor00o4693384111100_CopyAtom_ThrID10_T_0; g1x1x170; b288x1x1; r107; s99328 | 40 | 14.273 | 21.504 | 1.507x | 87.1% | 0.325 | fused_ffn (20) | fused_ffn (1.932x; 20) |
| 301 | outer_10.transformer_1.block_31.attention_qk_rope | qk_rope | qk_rope; batchSharedPackedFusedQKRoPE19Half2Kernel; g361x1x1; b192x1x1; r28; s0 | 40 | 5.664 | 7.712 | 1.362x | 99.6% | 0.087 | library_gemm (20) | linear2_residual (1.540x; 20) |
| 302 | outer_10.transformer_1.block_31.attention_fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 40 | 11.952 | 18.816 | 1.574x | 75.8% | 0.257 | linear2_residual (20) | linear2_residual (1.582x; 20) |
| 303 | outer_10.transformer_1.block_31.attention_out_projection_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 40 | 8.512 | 17.761 | 2.087x | 96.8% | 0.372 | fused_ffn (20) | fused_ffn (2.492x; 20) |
| 304 | outer_10.transformer_1.block_31.ffn_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 40 | 2.432 | 3.424 | 1.408x | 89.6% | 0.042 | fa4 (20) | fa4 (1.434x; 20) |
| 305 | outer_10.transformer_1.block_31.ffn_linear1_gate_swiglu | fused_ffn | fused_ffn; fused_ffn_candidate_a_reuse_kernel; g18x37x1; b128x1x1; r136; s32768 | 40 | 37.361 | 46.128 | 1.235x | 73.6% | 0.365 | linear2_residual (20) | linear2_residual (1.369x; 20) |
| 306 | outer_10.transformer_1.block_31.ffn_linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 40 | 20.881 | 30.960 | 1.483x | 97.0% | 0.412 | fused_ffn (20) | fused_ffn (1.538x; 20) |
| 307 | outer_10.transformer_2.block_32.attention_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 40 | 2.432 | 4.560 | 1.875x | 100.0% | 0.084 | fa4 (20) | fused_ffn (2.329x; 20) |
| 308 | outer_10.transformer_2.block_32.attention_qkv_projection | wide_qkv | wide_qkv; kernel_cutlass_kernel___main__FixedAtomLayoutGemmmakelocals_Kernel_object_at__CopyAtom_ThrID10_TVLayoutSrc1819201_TVLayoutDst1819201_Valuetypef16_tensor00o4693384111100_CopyAtom_ThrID10_T_0; g1x1x170; b288x1x1; r107; s99328 | 40 | 14.225 | 20.849 | 1.466x | 85.0% | 0.286 | fused_ffn (20) | fused_ffn (1.800x; 20) |
| 309 | outer_10.transformer_2.block_32.attention_qk_rope | qk_rope | qk_rope; batchSharedPackedFusedQKRoPE19Half2Kernel; g361x1x1; b192x1x1; r28; s0 | 40 | 5.680 | 7.760 | 1.366x | 99.2% | 0.087 | library_gemm (20) | linear2_residual (1.532x; 20) |
| 310 | outer_10.transformer_2.block_32.attention_fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 40 | 12.000 | 17.392 | 1.449x | 75.6% | 0.234 | linear2_residual (20) | fused_ffn (1.631x; 16) |
| 311 | outer_10.transformer_2.block_32.attention_out_projection_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 40 | 8.464 | 20.000 | 2.363x | 97.6% | 0.466 | fused_ffn (20) | fused_ffn (2.437x; 20) |
| 312 | outer_10.transformer_2.block_32.ffn_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 40 | 2.496 | 3.248 | 1.301x | 91.4% | 0.028 | linear2_residual (20) | linear2_residual (1.333x; 20) |
| 313 | outer_10.transformer_2.block_32.ffn_linear1_gate_swiglu | fused_ffn | fused_ffn; fused_ffn_candidate_a_reuse_kernel; g18x37x1; b128x1x1; r136; s32768 | 40 | 40.833 | 48.913 | 1.198x | 64.7% | 0.354 | library_gemm (20) | linear2_residual (1.267x; 20) |
| 314 | outer_10.transformer_2.block_32.ffn_linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 40 | 20.960 | 29.584 | 1.411x | 94.3% | 0.291 | library_gemm (19) | wide_qkv (1.534x; 6) |
| 315 | outer_10.post_norm_silu | affine_silu | affine_silu; affineSiluHalf2Kernel; g4693x1x1; b192x1x1; r17; s0 | 40 | 3.376 | 4.208 | 1.247x | 66.1% | 0.045 | fa4 (19) | fa4 (1.450x; 19) |
| 316 | outer_10.post_projection_c384_to_c768_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b256x1x1; r154; s73728 | 40 | 14.400 | 17.120 | 1.189x | 94.7% | 0.111 | library_gemm (39) | library_gemm (1.185x; 39) |
| 317 | trunk.tip_norm_silu | affine_silu | affine_silu; affineSiluHalf2Kernel; g4693x1x1; b384x1x1; r17; s0 | 40 | 5.056 | 6.496 | 1.285x | 69.5% | 0.058 | library_gemm (35) | library_gemm (1.171x; 35) |
| 318 | policy.p1_conv | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r80; s98304 | 40 | 6.240 | 7.440 | 1.192x | 89.7% | 0.047 | fused_ffn (19) | fused_ffn (1.215x; 19) |
| 319 | policy.g1_conv | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r80; s98304 | 40 | 5.888 | 13.857 | 2.353x | 96.6% | 0.356 | library_gemm (21) | fused_ffn (3.021x; 19) |
| 320 | policy.g1_norm_silu | head_elementwise | head_elementwise; applyCScaleBiasNHWCSiluHalfKernel; g1x73x13; b96x5x1; r16; s0 | 40 | 2.128 | 2.896 | 1.361x | 77.5% | 0.034 | linear2_residual (19) | linear2_residual (1.564x; 19) |
| 321 | policy.g1_half_to_float | copy_reformat | copy_reformat; copyFromHalfKernel; g880x1x1; b512x1x1; r16; s0 | 40 | 1.568 | 2.144 | 1.367x | 87.2% | 0.020 | linear2_residual (19) | linear2_residual (1.450x; 19) |
| 322 | policy.g1_global_pool | head_elementwise | head_elementwise; gPoolChannelsNHWCKernel; g2x1x13; b64x8x1; r22; s4096 | 40 | 4.448 | 5.120 | 1.151x | 95.7% | 0.031 | linear2_residual (19) | linear2_residual (1.259x; 19) |
| 323 | policy.gpool_to_bias_matmul | library_gemm | library_gemm; gemmSN_NN_kernel; g2x2x1; b256x1x1; r64; s21504 | 40 | 5.377 | 5.920 | 1.101x | 99.4% | 0.027 | library_gemm (20) | linear2_residual (1.161x; 19) |
| 324 | policy.p1_half_to_float | copy_reformat | copy_reformat; copyFromHalfKernel; g880x1x1; b512x1x1; r16; s0 | 40 | 1.504 | 1.904 | 1.266x | 93.0% | 0.033 | library_gemm (21) | linear2_residual (1.894x; 19) |
| 325 | policy.gpool_bias_add | head_elementwise | head_elementwise; addNCBiasInplaceNHWCKernel; g1x73x13; b96x5x1; r16; s0 | 40 | 1.792 | 2.033 | 1.134x | 62.8% | 0.012 | affine_silu (18) | library_gemm (1.188x; 4) |
| 326 | policy.p1_norm_silu | head_elementwise | head_elementwise; applyCScaleBiasNHWCSiluKernel; g1x73x13; b96x5x1; r16; s0 | 40 | 2.144 | 2.672 | 1.246x | 76.0% | 0.036 | library_gemm (39) | library_gemm (1.194x; 39) |
| 327 | policy.p2_conv | library_gemm | library_gemm; Kernel2; g74x1x1; b128x1x1; r90; s98304 | 40 | 3.936 | 5.856 | 1.488x | 95.0% | 0.091 | library_gemm (39) | library_gemm (1.837x; 39) |
| 328 | policy.gpool_to_pass_matmul | library_gemm | library_gemm; gemmSN_NN_kernel; g2x2x1; b256x1x1; r64; s21504 | 40 | 5.264 | 5.984 | 1.137x | 90.7% | 0.042 | library_gemm (21) | affine_silu (1.313x; 18) |
| 329 | policy.pass_bias_silu | head_elementwise | head_elementwise; addCBiasInplaceNCKernelSilu; g1x3x1; b96x5x1; r16; s0 | 40 | 1.024 | 1.089 | 1.063x | 86.2% | 0.006 | library_gemm (22) | affine_silu (1.094x; 18) |
| 330 | policy.gpool_to_pass_matmul2 | library_gemm | library_gemm; gemmSN_NN_kernel; g1x2x1; b256x1x1; r64; s21504 | 40 | 2.304 | 2.672 | 1.160x | 93.5% | 0.053 | library_gemm (40) | library_gemm (1.160x; 40) |
| 331 | value.v1_conv | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r118; s98304 | 40 | 7.952 | 9.728 | 1.223x | 71.2% | 0.069 | library_gemm (21) | library_gemm (1.364x; 21) |
| 332 | value.v1_norm_silu | head_elementwise | head_elementwise; applyCScaleBiasNHWCSiluHalfKernel; g1x181x13; b192x2x1; r16; s0 | 40 | 3.168 | 3.456 | 1.091x | 62.6% | 0.014 | library_gemm (19) | library_gemm (1.182x; 19) |
| 333 | value.v1_half_to_float | copy_reformat | copy_reformat; copyFromHalfKernel; g1760x1x1; b512x1x1; r16; s0 | 40 | 2.144 | 2.592 | 1.209x | 48.9% | 0.023 | head_elementwise (20) | head_elementwise (1.381x; 20) |
| 334 | value.v1_global_pool | head_elementwise | head_elementwise; valueHeadPoolChannelsNHWCKernel; g3x1x13; b64x8x1; r22; s2048 | 40 | 3.248 | 3.648 | 1.123x | 67.3% | 0.019 | head_elementwise (19) | head_elementwise (1.133x; 19) |
| 335 | value.v2_matmul | library_gemm | library_gemm; gemmSN_NN_kernel; g3x2x1; b256x1x1; r64; s21504 | 40 | 9.504 | 10.448 | 1.099x | 87.4% | 0.113 | library_gemm (21) | cudnn (1.596x; 18) |
| 336 | value.v2_bias_silu | head_elementwise | head_elementwise; addCBiasInplaceNCKernelSilu; g1x7x1; b192x2x1; r16; s0 | 40 | 1.024 | 1.216 | 1.188x | 79.0% | 0.012 | head_elementwise (19) | head_elementwise (1.219x; 19) |
| 337 | value.v3_matmul | library_gemm | library_gemm; gemmSN_NN_kernel; g1x2x1; b256x1x1; r64; s21504 | 40 | 3.472 | 3.873 | 1.115x | 84.5% | 0.018 | library_gemm (19) | library_gemm (1.161x; 19) |
| 338 | value.v3_bias | head_elementwise | head_elementwise; addCBiasInplaceNCKernel; g1x1x1; b3x170x1; r16; s0 | 40 | 0.960 | 1.088 | 1.133x | 88.4% | 0.006 | library_gemm (37) | library_gemm (1.133x; 37) |
| 339 | value.score_matmul | library_gemm | library_gemm; gemmSN_NN_kernel; g1x2x1; b256x1x1; r64; s21504 | 40 | 3.504 | 4.048 | 1.155x | 88.1% | 0.022 | head_elementwise (19) | head_elementwise (1.224x; 19) |
| 340 | value.score_bias | head_elementwise | head_elementwise; addCBiasInplaceNCKernel; g1x1x1; b6x85x1; r16; s0 | 40 | 0.928 | 1.056 | 1.138x | 76.6% | 0.005 | library_gemm (20) | library_gemm (1.207x; 20) |
| 341 | value.ownership_conv | library_gemm | library_gemm; Kernel2; g8x19x3; b128x1x1; r118; s33792 | 40 | 4.032 | 5.024 | 1.246x | 57.1% | 0.037 | library_gemm (26) | library_gemm (1.262x; 26) |
| 342 | value.ownership_conv_splitk_reduce | library_gemm | library_gemm; splitKreduce_kernel; g147x1x1; b32x16x1; r49; s0 | 40 | 1.376 | 1.744 | 1.267x | 93.7% | 0.024 | library_gemm (38) | library_gemm (1.279x; 38) |
| 343 | value.ownership_half_to_float | copy_reformat | copy_reformat; copyFromHalfKernel; g10x1x1; b512x1x1; r16; s0 | 40 | 0.928 | 1.344 | 1.448x | 97.1% | 0.036 | library_gemm (38) | library_gemm (1.500x; 38) |
