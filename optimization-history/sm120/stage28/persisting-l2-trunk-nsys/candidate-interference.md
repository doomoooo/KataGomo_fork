# Nsys stream interference report

- Timed iterations: 30; streams: 65, 82
- Kernels per forward: 65=344, 82=344
- Iteration start offset stream 82 - 65: median 124.13 us, p10..p90 118.37..126.09 us, range 0.86..154.85 us.

## Kernel families

| family | instances | total ms | median us | p90 us | peer overlap | S2/S1 median | summed excess ms | matched |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| fused_ffn | 1980 | 90.535 | 45.281 | 50.560 | 74.3% | n/a | 0.000 | 0 |
| library_gemm | 4140 | 69.284 | 18.912 | 23.809 | 95.7% | n/a | 0.000 | 0 |
| linear2_residual | 1980 | 62.693 | 31.872 | 34.912 | 97.6% | n/a | 0.000 | 0 |
| wide_qkv | 1980 | 48.019 | 23.072 | 30.048 | 72.8% | n/a | 0.000 | 0 |
| fa4 | 1980 | 32.592 | 15.264 | 21.636 | 60.5% | n/a | 0.000 | 0 |
| rmsnorm | 3960 | 15.062 | 3.360 | 5.536 | 99.7% | n/a | 0.000 | 0 |
| qk_rope | 1980 | 13.124 | 5.824 | 9.472 | 89.7% | n/a | 0.000 | 0 |
| affine_silu | 1380 | 9.600 | 7.040 | 10.080 | 97.1% | n/a | 0.000 | 0 |
| head_elementwise | 720 | 2.454 | 2.496 | 7.971 | 81.5% | n/a | 0.000 | 0 |
| cudnn | 180 | 1.576 | 2.080 | 23.424 | 44.8% | n/a | 0.000 | 0 |
| copy_reformat | 300 | 0.596 | 1.856 | 3.552 | 81.6% | n/a | 0.000 | 0 |
| sumChannelsNCHWKernel | 60 | 0.153 | 1.936 | 4.032 | 93.8% | n/a | 0.000 | 0 |

## Dominant interference pairs

| running family | dominant peer | instances | median overlap | median us | S2/S1 median | matched |
|---|---|---:|---:|---:|---:|---:|
| library_gemm | fused_ffn | 1267 | 97.2% | 21.120 | n/a | 0 |
| rmsnorm | library_gemm | 1264 | 100.0% | 2.720 | n/a | 0 |
| library_gemm | library_gemm | 1047 | 96.5% | 14.400 | n/a | 0 |
| linear2_residual | fused_ffn | 951 | 98.3% | 31.841 | n/a | 0 |
| fused_ffn | linear2_residual | 944 | 87.2% | 45.056 | n/a | 0 |
| fa4 | library_gemm | 889 | 50.5% | 14.945 | n/a | 0 |
| library_gemm | linear2_residual | 754 | 96.4% | 22.976 | n/a | 0 |
| rmsnorm | fused_ffn | 681 | 100.0% | 5.537 | n/a | 0 |
| qk_rope | library_gemm | 673 | 100.0% | 5.152 | n/a | 0 |
| rmsnorm | fa4 | 670 | 100.0% | 3.424 | n/a | 0 |
| rmsnorm | linear2_residual | 643 | 100.0% | 3.328 | n/a | 0 |
| linear2_residual | library_gemm | 628 | 97.4% | 31.104 | n/a | 0 |
| affine_silu | linear2_residual | 626 | 100.0% | 6.816 | n/a | 0 |
| fused_ffn | fused_ffn | 623 | 69.1% | 48.000 | n/a | 0 |
| qk_rope | wide_qkv | 616 | 100.0% | 9.376 | n/a | 0 |
| rmsnorm | wide_qkv | 616 | 100.0% | 4.736 | n/a | 0 |
| wide_qkv | qk_rope | 616 | 65.8% | 22.177 | n/a | 0 |
| qk_rope | fa4 | 616 | 100.0% | 5.824 | n/a | 0 |
| wide_qkv | library_gemm | 578 | 63.3% | 22.304 | n/a | 0 |
| fa4 | qk_rope | 544 | 42.9% | 14.816 | n/a | 0 |
| library_gemm | wide_qkv | 414 | 97.0% | 19.040 | n/a | 0 |
| library_gemm | fa4 | 375 | 95.3% | 14.240 | n/a | 0 |
| fused_ffn | library_gemm | 364 | 49.0% | 42.480 | n/a | 0 |
| fa4 | fused_ffn | 329 | 61.7% | 16.257 | n/a | 0 |
| affine_silu | wide_qkv | 318 | 97.1% | 10.016 | n/a | 0 |
| wide_qkv | linear2_residual | 308 | 96.1% | 30.800 | n/a | 0 |
| linear2_residual | wide_qkv | 308 | 98.2% | 33.936 | n/a | 0 |
| wide_qkv | affine_silu | 304 | 69.0% | 23.776 | n/a | 0 |
| affine_silu | fused_ffn | 292 | 100.0% | 6.688 | n/a | 0 |
| head_elementwise | library_gemm | 261 | 100.0% | 4.512 | n/a | 0 |
| library_gemm | head_elementwise | 145 | 90.8% | 7.617 | n/a | 0 |
| fa4 | fa4 | 144 | 65.4% | 16.448 | n/a | 0 |
| copy_reformat | library_gemm | 119 | 100.0% | 1.600 | n/a | 0 |
| wide_qkv | wide_qkv | 108 | 64.2% | 23.441 | n/a | 0 |
| head_elementwise | head_elementwise | 94 | 50.7% | 1.984 | n/a | 0 |
| affine_silu | library_gemm | 90 | 100.0% | 5.440 | n/a | 0 |
| cudnn | library_gemm | 82 | 83.3% | 1.664 | n/a | 0 |
| cudnn | fused_ffn | 80 | 100.0% | 2.272 | n/a | 0 |
| head_elementwise | fused_ffn | 68 | 100.0% | 2.752 | n/a | 0 |
| wide_qkv | fused_ffn | 65 | 95.9% | 26.560 | n/a | 0 |

## Logical operation groups

Isolated reference total is the isolated median for each ordinal multiplied by its S2 call count; it is a normalized reference, not a second trace total.

| logical group | families | positions | calls | isolated reference ms | S2 total ms | total ratio | excess ms |
|---|---|---:|---:|---:|---:|---:|---:|
| transformer.ffn_linear1_gate_swiglu | fused_ffn | 33 | 1980 | 0.000 | 90.535 | n/a | 0.000 |
| transformer.ffn_linear2_residual | linear2_residual | 33 | 1980 | 0.000 | 62.693 | n/a | 0.000 |
| transformer.attention_qkv_projection | wide_qkv | 33 | 1980 | 0.000 | 48.019 | n/a | 0.000 |
| transformer.attention_out_projection_residual | library_gemm | 33 | 1980 | 0.000 | 36.355 | n/a | 0.000 |
| transformer.attention_fa4 | fa4 | 33 | 1980 | 0.000 | 32.592 | n/a | 0.000 |
| outer.post_projection_c384_to_c768_residual | library_gemm | 11 | 660 | 0.000 | 16.025 | n/a | 0.000 |
| transformer.attention_qk_rope | qk_rope | 33 | 1980 | 0.000 | 13.124 | n/a | 0.000 |
| outer.pre_projection_c768_to_c384 | library_gemm | 11 | 660 | 0.000 | 11.711 | n/a | 0.000 |
| transformer.ffn_rmsnorm | rmsnorm | 33 | 1980 | 0.000 | 7.537 | n/a | 0.000 |
| transformer.attention_rmsnorm | rmsnorm | 33 | 1980 | 0.000 | 7.525 | n/a | 0.000 |
| outer.pre_norm_silu | affine_silu | 11 | 660 | 0.000 | 5.534 | n/a | 0.000 |
| outer.post_norm_silu | affine_silu | 11 | 660 | 0.000 | 3.609 | n/a | 0.000 |
| frontend.initial_conv | cudnn | 1 | 60 | 0.000 | 1.358 | n/a | 0.000 |
| value.v1_conv | library_gemm | 1 | 60 | 0.000 | 0.857 | n/a | 0.000 |
| value.v2_matmul | library_gemm | 1 | 60 | 0.000 | 0.839 | n/a | 0.000 |
| frontend.initial_global_broadcast_add | head_elementwise | 1 | 60 | 0.000 | 0.566 | n/a | 0.000 |
| trunk.tip_norm_silu | affine_silu | 1 | 60 | 0.000 | 0.456 | n/a | 0.000 |
| policy.g1_conv | library_gemm | 1 | 60 | 0.000 | 0.455 | n/a | 0.000 |
| policy.p1_conv | library_gemm | 1 | 60 | 0.000 | 0.426 | n/a | 0.000 |
| policy.gpool_to_pass_matmul | library_gemm | 1 | 60 | 0.000 | 0.403 | n/a | 0.000 |
| policy.gpool_to_bias_matmul | library_gemm | 1 | 60 | 0.000 | 0.395 | n/a | 0.000 |
| policy.g1_global_pool | head_elementwise | 1 | 60 | 0.000 | 0.374 | n/a | 0.000 |
| policy.p2_conv | library_gemm | 1 | 60 | 0.000 | 0.345 | n/a | 0.000 |
| value.v1_global_pool | head_elementwise | 1 | 60 | 0.000 | 0.301 | n/a | 0.000 |
| value.v3_matmul | library_gemm | 1 | 60 | 0.000 | 0.293 | n/a | 0.000 |
| value.ownership_conv | library_gemm | 1 | 60 | 0.000 | 0.288 | n/a | 0.000 |
| value.v1_norm_silu | head_elementwise | 1 | 60 | 0.000 | 0.257 | n/a | 0.000 |
| value.score_matmul | library_gemm | 1 | 60 | 0.000 | 0.253 | n/a | 0.000 |
| frontend.initial_global_matmul | library_gemm | 1 | 60 | 0.000 | 0.243 | n/a | 0.000 |
| value.v1_half_to_float | copy_reformat | 1 | 60 | 0.000 | 0.182 | n/a | 0.000 |
| policy.g1_norm_silu | head_elementwise | 1 | 60 | 0.000 | 0.171 | n/a | 0.000 |
| policy.gpool_to_pass_matmul2 | library_gemm | 1 | 60 | 0.000 | 0.163 | n/a | 0.000 |
| policy.p1_norm_silu | head_elementwise | 1 | 60 | 0.000 | 0.161 | n/a | 0.000 |
| policy.gpool_bias_add | head_elementwise | 1 | 60 | 0.000 | 0.158 | n/a | 0.000 |
| input.mask_sum | sumChannelsNCHWKernel | 1 | 60 | 0.000 | 0.153 | n/a | 0.000 |
| policy.p1_half_to_float | copy_reformat | 1 | 60 | 0.000 | 0.149 | n/a | 0.000 |
| value.v2_bias_silu | head_elementwise | 1 | 60 | 0.000 | 0.146 | n/a | 0.000 |
| value.ownership_conv_splitk_reduce | library_gemm | 1 | 60 | 0.000 | 0.129 | n/a | 0.000 |
| frontend.initial_conv_nhwc_padding_1 | cudnn | 1 | 60 | 0.000 | 0.120 | n/a | 0.000 |
| policy.g1_half_to_float | copy_reformat | 1 | 60 | 0.000 | 0.114 | n/a | 0.000 |
| frontend.initial_global_matmul_splitk_reduce | library_gemm | 1 | 60 | 0.000 | 0.104 | n/a | 0.000 |
| frontend.initial_conv_nhwc_padding_0 | cudnn | 1 | 60 | 0.000 | 0.098 | n/a | 0.000 |
| input.extract_mask | head_elementwise | 1 | 60 | 0.000 | 0.094 | n/a | 0.000 |
| input.mask_half_to_float | copy_reformat | 1 | 60 | 0.000 | 0.080 | n/a | 0.000 |
| value.v3_bias | head_elementwise | 1 | 60 | 0.000 | 0.079 | n/a | 0.000 |
| policy.pass_bias_silu | head_elementwise | 1 | 60 | 0.000 | 0.076 | n/a | 0.000 |
| value.ownership_half_to_float | copy_reformat | 1 | 60 | 0.000 | 0.071 | n/a | 0.000 |
| value.score_bias | head_elementwise | 1 | 60 | 0.000 | 0.071 | n/a | 0.000 |

## `library_gemm` logical breakdown

| logical group | positions | calls | isolated reference ms | S2 total ms | total ratio | excess ms |
|---|---:|---:|---:|---:|---:|---:|
| transformer.attention_out_projection_residual | 33 | 1980 | 0.000 | 36.355 | n/a | 0.000 |
| outer.post_projection_c384_to_c768_residual | 11 | 660 | 0.000 | 16.025 | n/a | 0.000 |
| outer.pre_projection_c768_to_c384 | 11 | 660 | 0.000 | 11.711 | n/a | 0.000 |
| value.v1_conv | 1 | 60 | 0.000 | 0.857 | n/a | 0.000 |
| value.v2_matmul | 1 | 60 | 0.000 | 0.839 | n/a | 0.000 |
| policy.g1_conv | 1 | 60 | 0.000 | 0.455 | n/a | 0.000 |
| policy.p1_conv | 1 | 60 | 0.000 | 0.426 | n/a | 0.000 |
| policy.gpool_to_pass_matmul | 1 | 60 | 0.000 | 0.403 | n/a | 0.000 |
| policy.gpool_to_bias_matmul | 1 | 60 | 0.000 | 0.395 | n/a | 0.000 |
| policy.p2_conv | 1 | 60 | 0.000 | 0.345 | n/a | 0.000 |
| value.v3_matmul | 1 | 60 | 0.000 | 0.293 | n/a | 0.000 |
| value.ownership_conv | 1 | 60 | 0.000 | 0.288 | n/a | 0.000 |
| value.score_matmul | 1 | 60 | 0.000 | 0.253 | n/a | 0.000 |
| frontend.initial_global_matmul | 1 | 60 | 0.000 | 0.243 | n/a | 0.000 |
| policy.gpool_to_pass_matmul2 | 1 | 60 | 0.000 | 0.163 | n/a | 0.000 |
| value.ownership_conv_splitk_reduce | 1 | 60 | 0.000 | 0.129 | n/a | 0.000 |
| frontend.initial_global_matmul_splitk_reduce | 1 | 60 | 0.000 | 0.104 | n/a | 0.000 |

## Top ordinal hotspots by summed excess

The worst peer is the highest median S2/S1 slowdown among peer families observed at least four times for that ordinal.

| rank | ordinal | logical position | family | calls | isolated us | S2 us | S2/S1 | excess ms | common peer | worst peer |
|---:|---:|---|---|---:|---:|---:|---:|---:|---|---|
| 1 | 241 | outer_08.transformer_0.block_24.ffn_linear1_gate_swiglu | fused_ffn | 60 | n/a | 49.264 | n/a | 0.000 | linear2_residual (57) | n/a |
| 2 | 157 | outer_05.transformer_0.block_15.ffn_linear1_gate_swiglu | fused_ffn | 60 | n/a | 48.624 | n/a | 0.000 | linear2_residual (57) | n/a |
| 3 | 249 | outer_08.transformer_1.block_25.ffn_linear1_gate_swiglu | fused_ffn | 60 | n/a | 47.745 | n/a | 0.000 | linear2_residual (30) | n/a |
| 4 | 73 | outer_02.transformer_0.block_06.ffn_linear1_gate_swiglu | fused_ffn | 60 | n/a | 46.416 | n/a | 0.000 | linear2_residual (57) | n/a |
| 5 | 185 | outer_06.transformer_0.block_18.ffn_linear1_gate_swiglu | fused_ffn | 60 | n/a | 47.760 | n/a | 0.000 | linear2_residual (57) | n/a |
| 6 | 269 | outer_09.transformer_0.block_27.ffn_linear1_gate_swiglu | fused_ffn | 60 | n/a | 47.344 | n/a | 0.000 | linear2_residual (56) | n/a |
| 7 | 313 | outer_10.transformer_2.block_32.ffn_linear1_gate_swiglu | fused_ffn | 60 | n/a | 47.937 | n/a | 0.000 | library_gemm (30) | n/a |
| 8 | 81 | outer_02.transformer_1.block_07.ffn_linear1_gate_swiglu | fused_ffn | 60 | n/a | 47.248 | n/a | 0.000 | linear2_residual (30) | n/a |
| 9 | 165 | outer_05.transformer_1.block_16.ffn_linear1_gate_swiglu | fused_ffn | 60 | n/a | 46.608 | n/a | 0.000 | fused_ffn (29) | n/a |
| 10 | 89 | outer_02.transformer_2.block_08.ffn_linear1_gate_swiglu | fused_ffn | 60 | n/a | 46.880 | n/a | 0.000 | library_gemm (29) | n/a |
| 11 | 193 | outer_06.transformer_1.block_19.ffn_linear1_gate_swiglu | fused_ffn | 60 | n/a | 46.800 | n/a | 0.000 | linear2_residual (30) | n/a |
| 12 | 257 | outer_08.transformer_2.block_26.ffn_linear1_gate_swiglu | fused_ffn | 60 | n/a | 46.417 | n/a | 0.000 | library_gemm (29) | n/a |
| 13 | 173 | outer_05.transformer_2.block_17.ffn_linear1_gate_swiglu | fused_ffn | 60 | n/a | 47.056 | n/a | 0.000 | library_gemm (29) | n/a |
| 14 | 61 | outer_01.transformer_2.block_05.ffn_linear1_gate_swiglu | fused_ffn | 60 | n/a | 47.248 | n/a | 0.000 | library_gemm (29) | n/a |
| 15 | 53 | outer_01.transformer_1.block_04.ffn_linear1_gate_swiglu | fused_ffn | 60 | n/a | 45.168 | n/a | 0.000 | linear2_residual (30) | n/a |
| 16 | 101 | outer_03.transformer_0.block_09.ffn_linear1_gate_swiglu | fused_ffn | 60 | n/a | 44.209 | n/a | 0.000 | linear2_residual (57) | n/a |
| 17 | 45 | outer_01.transformer_0.block_03.ffn_linear1_gate_swiglu | fused_ffn | 60 | n/a | 46.304 | n/a | 0.000 | linear2_residual (57) | n/a |
| 18 | 213 | outer_07.transformer_0.block_21.ffn_linear1_gate_swiglu | fused_ffn | 60 | n/a | 45.025 | n/a | 0.000 | linear2_residual (57) | n/a |
| 19 | 285 | outer_09.transformer_2.block_29.ffn_linear1_gate_swiglu | fused_ffn | 60 | n/a | 45.761 | n/a | 0.000 | library_gemm (29) | n/a |
| 20 | 137 | outer_04.transformer_1.block_13.ffn_linear1_gate_swiglu | fused_ffn | 60 | n/a | 45.168 | n/a | 0.000 | linear2_residual (30) | n/a |
| 21 | 33 | outer_00.transformer_2.block_02.ffn_linear1_gate_swiglu | fused_ffn | 60 | n/a | 46.288 | n/a | 0.000 | library_gemm (29) | n/a |
| 22 | 305 | outer_10.transformer_1.block_31.ffn_linear1_gate_swiglu | fused_ffn | 60 | n/a | 45.696 | n/a | 0.000 | linear2_residual (30) | n/a |
| 23 | 277 | outer_09.transformer_1.block_28.ffn_linear1_gate_swiglu | fused_ffn | 60 | n/a | 45.281 | n/a | 0.000 | fused_ffn (29) | n/a |
| 24 | 25 | outer_00.transformer_1.block_01.ffn_linear1_gate_swiglu | fused_ffn | 60 | n/a | 44.577 | n/a | 0.000 | fused_ffn (31) | n/a |
| 25 | 117 | outer_03.transformer_2.block_11.ffn_linear1_gate_swiglu | fused_ffn | 60 | n/a | 46.593 | n/a | 0.000 | library_gemm (29) | n/a |
| 26 | 109 | outer_03.transformer_1.block_10.ffn_linear1_gate_swiglu | fused_ffn | 60 | n/a | 44.529 | n/a | 0.000 | fused_ffn (29) | n/a |
| 27 | 129 | outer_04.transformer_0.block_12.ffn_linear1_gate_swiglu | fused_ffn | 60 | n/a | 44.736 | n/a | 0.000 | linear2_residual (57) | n/a |
| 28 | 297 | outer_10.transformer_0.block_30.ffn_linear1_gate_swiglu | fused_ffn | 60 | n/a | 43.392 | n/a | 0.000 | linear2_residual (57) | n/a |
| 29 | 221 | outer_07.transformer_1.block_22.ffn_linear1_gate_swiglu | fused_ffn | 60 | n/a | 43.553 | n/a | 0.000 | linear2_residual (30) | n/a |
| 30 | 229 | outer_07.transformer_2.block_23.ffn_linear1_gate_swiglu | fused_ffn | 60 | n/a | 42.160 | n/a | 0.000 | library_gemm (29) | n/a |
| 31 | 17 | outer_00.transformer_0.block_00.ffn_linear1_gate_swiglu | fused_ffn | 60 | n/a | 42.016 | n/a | 0.000 | linear2_residual (29) | n/a |
| 32 | 145 | outer_04.transformer_2.block_14.ffn_linear1_gate_swiglu | fused_ffn | 60 | n/a | 41.361 | n/a | 0.000 | library_gemm (29) | n/a |
| 33 | 201 | outer_06.transformer_2.block_20.ffn_linear1_gate_swiglu | fused_ffn | 60 | n/a | 40.849 | n/a | 0.000 | library_gemm (29) | n/a |
| 34 | 222 | outer_07.transformer_1.block_22.ffn_linear2_residual | linear2_residual | 60 | n/a | 36.081 | n/a | 0.000 | fused_ffn (30) | n/a |
| 35 | 214 | outer_07.transformer_0.block_21.ffn_linear2_residual | linear2_residual | 60 | n/a | 34.385 | n/a | 0.000 | library_gemm (28) | n/a |
| 36 | 138 | outer_04.transformer_1.block_13.ffn_linear2_residual | linear2_residual | 60 | n/a | 33.857 | n/a | 0.000 | fused_ffn (30) | n/a |
| 37 | 230 | outer_07.transformer_2.block_23.ffn_linear2_residual | linear2_residual | 60 | n/a | 32.288 | n/a | 0.000 | fused_ffn (57) | n/a |
| 38 | 306 | outer_10.transformer_1.block_31.ffn_linear2_residual | linear2_residual | 60 | n/a | 33.504 | n/a | 0.000 | fused_ffn (30) | n/a |
| 39 | 250 | outer_08.transformer_1.block_25.ffn_linear2_residual | linear2_residual | 60 | n/a | 33.488 | n/a | 0.000 | fused_ffn (30) | n/a |
| 40 | 194 | outer_06.transformer_1.block_19.ffn_linear2_residual | linear2_residual | 60 | n/a | 33.153 | n/a | 0.000 | fused_ffn (30) | n/a |

## Full fixed-forward ordinal map

| ordinal | logical position | family | resource signature | calls | isolated us | S2 us | S2/S1 | overlap | excess ms | common peer | worst peer |
|---:|---|---|---|---:|---:|---:|---:|---:|---:|---|---|
| 0 | input.extract_mask | head_elementwise | head_elementwise; extractChannel0KernelNHWC; g10x1x1; b512x1x1; r16; s0 | 60 | n/a | 1.392 | n/a | 80.4% | 0.000 | head_elementwise (22) | n/a |
| 1 | input.mask_half_to_float | copy_reformat | copy_reformat; copyFromHalfKernel; g10x1x1; b512x1x1; r16; s0 | 60 | n/a | 1.120 | n/a | 87.6% | 0.000 | library_gemm (20) | n/a |
| 2 | input.mask_sum | sumChannelsNCHWKernel | sumChannelsNCHWKernel; sumChannelsNCHWKernel; g1x1x13; b256x2x1; r22; s2048 | 60 | n/a | 1.936 | n/a | 93.8% | 0.000 | library_gemm (29) | n/a |
| 3 | frontend.initial_conv_nhwc_padding_0 | cudnn | cudnn; nhwcAddPaddingKernel; g340x1x1; b768x1x1; r34; s0 | 60 | n/a | 1.633 | n/a | 92.2% | 0.000 | library_gemm (29) | n/a |
| 4 | frontend.initial_conv_nhwc_padding_1 | cudnn | cudnn; nhwcAddPaddingKernel; g340x1x1; b768x1x1; r34; s0 | 60 | n/a | 1.920 | n/a | 93.2% | 0.000 | fused_ffn (27) | n/a |
| 5 | frontend.initial_conv | cudnn | cudnn; Kernel; g296x3x1; b128x1x1; r94; s81920 | 60 | n/a | 22.272 | n/a | 37.1% | 0.000 | library_gemm (28) | n/a |
| 6 | frontend.initial_global_matmul | library_gemm | library_gemm; Kernel2; g8x1x3; b128x1x1; r128; s24576 | 60 | n/a | 3.360 | n/a | 96.4% | 0.000 | library_gemm (29) | n/a |
| 7 | frontend.initial_global_matmul_splitk_reduce | library_gemm | library_gemm; splitKreduce_kernel; g24x1x1; b32x16x1; r49; s0 | 60 | n/a | 1.600 | n/a | 97.1% | 0.000 | library_gemm (29) | n/a |
| 8 | frontend.initial_global_broadcast_add | head_elementwise | head_elementwise; addNCBiasInplaceNHWCHalfKernel; g3x361x13; b256x1x1; r16; s0 | 60 | n/a | 8.784 | n/a | 64.3% | 0.000 | linear2_residual (27) | n/a |
| 9 | outer_00.pre_norm_silu | affine_silu | affine_silu; affineSiluHalf2Kernel; g4693x1x1; b384x1x1; r17; s0 | 60 | n/a | 5.968 | n/a | 63.0% | 0.000 | linear2_residual (27) | n/a |
| 10 | outer_00.pre_projection_c768_to_c384 | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 60 | n/a | 13.552 | n/a | 95.2% | 0.000 | wide_qkv (28) | n/a |
| 11 | outer_00.transformer_0.block_00.attention_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | n/a | 3.280 | n/a | 99.5% | 0.000 | library_gemm (28) | n/a |
| 12 | outer_00.transformer_0.block_00.attention_qkv_projection | wide_qkv | wide_qkv; wide_qkv_kernel; g9x37x1; b128x1x1; r136; s65536 | 60 | n/a | 20.945 | n/a | 66.8% | 0.000 | library_gemm (31) | n/a |
| 13 | outer_00.transformer_0.block_00.attention_qk_rope | qk_rope | qk_rope; fusedQKRoPE19HalfKernel; g361x13x1; b192x1x1; r16; s0 | 60 | n/a | 4.448 | n/a | 52.7% | 0.000 | fa4 (28) | n/a |
| 14 | outer_00.transformer_0.block_00.attention_fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 60 | n/a | 13.984 | n/a | 63.6% | 0.000 | library_gemm (38) | n/a |
| 15 | outer_00.transformer_0.block_00.attention_out_projection_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 60 | n/a | 15.744 | n/a | 81.7% | 0.000 | library_gemm (34) | n/a |
| 16 | outer_00.transformer_0.block_00.ffn_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | n/a | 3.216 | n/a | 94.1% | 0.000 | fused_ffn (28) | n/a |
| 17 | outer_00.transformer_0.block_00.ffn_linear1_gate_swiglu | fused_ffn | fused_ffn; fused_ffn_kernel; g18x37x1; b128x1x1; r146; s32768 | 60 | n/a | 42.016 | n/a | 65.5% | 0.000 | linear2_residual (29) | n/a |
| 18 | outer_00.transformer_0.block_00.ffn_linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 60 | n/a | 27.729 | n/a | 95.5% | 0.000 | wide_qkv (28) | n/a |
| 19 | outer_00.transformer_1.block_01.attention_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | n/a | 3.632 | n/a | 93.8% | 0.000 | wide_qkv (28) | n/a |
| 20 | outer_00.transformer_1.block_01.attention_qkv_projection | wide_qkv | wide_qkv; wide_qkv_kernel; g9x37x1; b128x1x1; r136; s65536 | 60 | n/a | 22.544 | n/a | 71.7% | 0.000 | library_gemm (29) | n/a |
| 21 | outer_00.transformer_1.block_01.attention_qk_rope | qk_rope | qk_rope; fusedQKRoPE19HalfKernel; g361x13x1; b192x1x1; r16; s0 | 60 | n/a | 5.920 | n/a | 96.6% | 0.000 | fa4 (28) | n/a |
| 22 | outer_00.transformer_1.block_01.attention_fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 60 | n/a | 14.672 | n/a | 52.5% | 0.000 | library_gemm (25) | n/a |
| 23 | outer_00.transformer_1.block_01.attention_out_projection_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 60 | n/a | 18.976 | n/a | 95.9% | 0.000 | fused_ffn (30) | n/a |
| 24 | outer_00.transformer_1.block_01.ffn_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | n/a | 3.376 | n/a | 100.0% | 0.000 | fused_ffn (28) | n/a |
| 25 | outer_00.transformer_1.block_01.ffn_linear1_gate_swiglu | fused_ffn | fused_ffn; fused_ffn_kernel; g18x37x1; b128x1x1; r146; s32768 | 60 | n/a | 44.577 | n/a | 74.2% | 0.000 | fused_ffn (31) | n/a |
| 26 | outer_00.transformer_1.block_01.ffn_linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 60 | n/a | 31.168 | n/a | 98.1% | 0.000 | fused_ffn (30) | n/a |
| 27 | outer_00.transformer_2.block_02.attention_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | n/a | 2.976 | n/a | 100.0% | 0.000 | library_gemm (28) | n/a |
| 28 | outer_00.transformer_2.block_02.attention_qkv_projection | wide_qkv | wide_qkv; wide_qkv_kernel; g9x37x1; b128x1x1; r136; s65536 | 60 | n/a | 24.496 | n/a | 82.6% | 0.000 | affine_silu (28) | n/a |
| 29 | outer_00.transformer_2.block_02.attention_qk_rope | qk_rope | qk_rope; fusedQKRoPE19HalfKernel; g361x13x1; b192x1x1; r16; s0 | 60 | n/a | 5.648 | n/a | 74.9% | 0.000 | library_gemm (30) | n/a |
| 30 | outer_00.transformer_2.block_02.attention_fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 60 | n/a | 17.153 | n/a | 72.2% | 0.000 | library_gemm (28) | n/a |
| 31 | outer_00.transformer_2.block_02.attention_out_projection_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 60 | n/a | 18.544 | n/a | 96.6% | 0.000 | library_gemm (29) | n/a |
| 32 | outer_00.transformer_2.block_02.ffn_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | n/a | 3.072 | n/a | 99.8% | 0.000 | fa4 (29) | n/a |
| 33 | outer_00.transformer_2.block_02.ffn_linear1_gate_swiglu | fused_ffn | fused_ffn; fused_ffn_kernel; g18x37x1; b128x1x1; r146; s32768 | 60 | n/a | 46.288 | n/a | 58.4% | 0.000 | library_gemm (29) | n/a |
| 34 | outer_00.transformer_2.block_02.ffn_linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 60 | n/a | 30.432 | n/a | 97.6% | 0.000 | fused_ffn (57) | n/a |
| 35 | outer_00.post_norm_silu | affine_silu | affine_silu; affineSiluHalf2Kernel; g4693x1x1; b192x1x1; r17; s0 | 60 | n/a | 4.512 | n/a | 100.0% | 0.000 | fused_ffn (29) | n/a |
| 36 | outer_00.post_projection_c384_to_c768_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b256x1x1; r154; s73728 | 60 | n/a | 23.921 | n/a | 97.1% | 0.000 | fused_ffn (29) | n/a |
| 37 | outer_01.pre_norm_silu | affine_silu | affine_silu; affineSiluHalf2Kernel; g4693x1x1; b384x1x1; r17; s0 | 60 | n/a | 7.344 | n/a | 97.5% | 0.000 | linear2_residual (29) | n/a |
| 38 | outer_01.pre_projection_c768_to_c384 | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 60 | n/a | 14.672 | n/a | 95.7% | 0.000 | fa4 (29) | n/a |
| 39 | outer_01.transformer_0.block_03.attention_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | n/a | 4.640 | n/a | 100.0% | 0.000 | fa4 (29) | n/a |
| 40 | outer_01.transformer_0.block_03.attention_qkv_projection | wide_qkv | wide_qkv; wide_qkv_kernel; g9x37x1; b128x1x1; r136; s65536 | 60 | n/a | 21.776 | n/a | 65.0% | 0.000 | library_gemm (30) | n/a |
| 41 | outer_01.transformer_0.block_03.attention_qk_rope | qk_rope | qk_rope; fusedQKRoPE19HalfKernel; g361x13x1; b192x1x1; r16; s0 | 60 | n/a | 5.312 | n/a | 100.0% | 0.000 | library_gemm (30) | n/a |
| 42 | outer_01.transformer_0.block_03.attention_fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 60 | n/a | 15.633 | n/a | 57.4% | 0.000 | fused_ffn (29) | n/a |
| 43 | outer_01.transformer_0.block_03.attention_out_projection_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 60 | n/a | 21.857 | n/a | 96.8% | 0.000 | fused_ffn (57) | n/a |
| 44 | outer_01.transformer_0.block_03.ffn_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | n/a | 3.472 | n/a | 99.8% | 0.000 | linear2_residual (29) | n/a |
| 45 | outer_01.transformer_0.block_03.ffn_linear1_gate_swiglu | fused_ffn | fused_ffn; fused_ffn_kernel; g18x37x1; b128x1x1; r146; s32768 | 60 | n/a | 46.304 | n/a | 85.6% | 0.000 | linear2_residual (57) | n/a |
| 46 | outer_01.transformer_0.block_03.ffn_linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 60 | n/a | 28.256 | n/a | 97.3% | 0.000 | library_gemm (28) | n/a |
| 47 | outer_01.transformer_1.block_04.attention_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | n/a | 3.872 | n/a | 100.0% | 0.000 | library_gemm (28) | n/a |
| 48 | outer_01.transformer_1.block_04.attention_qkv_projection | wide_qkv | wide_qkv; wide_qkv_kernel; g9x37x1; b128x1x1; r136; s65536 | 60 | n/a | 23.392 | n/a | 68.2% | 0.000 | library_gemm (30) | n/a |
| 49 | outer_01.transformer_1.block_04.attention_qk_rope | qk_rope | qk_rope; fusedQKRoPE19HalfKernel; g361x13x1; b192x1x1; r16; s0 | 60 | n/a | 6.096 | n/a | 93.3% | 0.000 | fa4 (28) | n/a |
| 50 | outer_01.transformer_1.block_04.attention_fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 60 | n/a | 14.848 | n/a | 52.9% | 0.000 | qk_rope (28) | n/a |
| 51 | outer_01.transformer_1.block_04.attention_out_projection_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 60 | n/a | 18.081 | n/a | 96.0% | 0.000 | library_gemm (30) | n/a |
| 52 | outer_01.transformer_1.block_04.ffn_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | n/a | 3.184 | n/a | 100.0% | 0.000 | fused_ffn (28) | n/a |
| 53 | outer_01.transformer_1.block_04.ffn_linear1_gate_swiglu | fused_ffn | fused_ffn; fused_ffn_kernel; g18x37x1; b128x1x1; r146; s32768 | 60 | n/a | 45.168 | n/a | 75.6% | 0.000 | linear2_residual (30) | n/a |
| 54 | outer_01.transformer_1.block_04.ffn_linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 60 | n/a | 32.337 | n/a | 98.1% | 0.000 | fused_ffn (30) | n/a |
| 55 | outer_01.transformer_2.block_05.attention_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | n/a | 3.216 | n/a | 100.0% | 0.000 | library_gemm (28) | n/a |
| 56 | outer_01.transformer_2.block_05.attention_qkv_projection | wide_qkv | wide_qkv; wide_qkv_kernel; g9x37x1; b128x1x1; r136; s65536 | 60 | n/a | 24.752 | n/a | 83.8% | 0.000 | affine_silu (28) | n/a |
| 57 | outer_01.transformer_2.block_05.attention_qk_rope | qk_rope | qk_rope; fusedQKRoPE19HalfKernel; g361x13x1; b192x1x1; r16; s0 | 60 | n/a | 5.584 | n/a | 75.4% | 0.000 | library_gemm (30) | n/a |
| 58 | outer_01.transformer_2.block_05.attention_fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 60 | n/a | 17.152 | n/a | 73.5% | 0.000 | library_gemm (28) | n/a |
| 59 | outer_01.transformer_2.block_05.attention_out_projection_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 60 | n/a | 18.608 | n/a | 96.5% | 0.000 | library_gemm (29) | n/a |
| 60 | outer_01.transformer_2.block_05.ffn_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | n/a | 3.088 | n/a | 99.7% | 0.000 | fa4 (29) | n/a |
| 61 | outer_01.transformer_2.block_05.ffn_linear1_gate_swiglu | fused_ffn | fused_ffn; fused_ffn_kernel; g18x37x1; b128x1x1; r146; s32768 | 60 | n/a | 47.248 | n/a | 57.9% | 0.000 | library_gemm (29) | n/a |
| 62 | outer_01.transformer_2.block_05.ffn_linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 60 | n/a | 31.569 | n/a | 97.6% | 0.000 | fused_ffn (57) | n/a |
| 63 | outer_01.post_norm_silu | affine_silu | affine_silu; affineSiluHalf2Kernel; g4693x1x1; b192x1x1; r17; s0 | 60 | n/a | 4.512 | n/a | 100.0% | 0.000 | fused_ffn (29) | n/a |
| 64 | outer_01.post_projection_c384_to_c768_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b256x1x1; r154; s73728 | 60 | n/a | 24.720 | n/a | 97.1% | 0.000 | linear2_residual (31) | n/a |
| 65 | outer_02.pre_norm_silu | affine_silu | affine_silu; affineSiluHalf2Kernel; g4693x1x1; b384x1x1; r17; s0 | 60 | n/a | 7.312 | n/a | 97.2% | 0.000 | linear2_residual (29) | n/a |
| 66 | outer_02.pre_projection_c768_to_c384 | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 60 | n/a | 15.216 | n/a | 95.8% | 0.000 | linear2_residual (29) | n/a |
| 67 | outer_02.transformer_0.block_06.attention_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | n/a | 4.640 | n/a | 100.0% | 0.000 | fa4 (29) | n/a |
| 68 | outer_02.transformer_0.block_06.attention_qkv_projection | wide_qkv | wide_qkv; wide_qkv_kernel; g9x37x1; b128x1x1; r136; s65536 | 60 | n/a | 21.921 | n/a | 65.2% | 0.000 | library_gemm (30) | n/a |
| 69 | outer_02.transformer_0.block_06.attention_qk_rope | qk_rope | qk_rope; fusedQKRoPE19HalfKernel; g361x13x1; b192x1x1; r16; s0 | 60 | n/a | 5.328 | n/a | 100.0% | 0.000 | library_gemm (30) | n/a |
| 70 | outer_02.transformer_0.block_06.attention_fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 60 | n/a | 15.872 | n/a | 57.2% | 0.000 | fused_ffn (29) | n/a |
| 71 | outer_02.transformer_0.block_06.attention_out_projection_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 60 | n/a | 21.616 | n/a | 96.9% | 0.000 | fused_ffn (57) | n/a |
| 72 | outer_02.transformer_0.block_06.ffn_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | n/a | 3.504 | n/a | 99.8% | 0.000 | linear2_residual (29) | n/a |
| 73 | outer_02.transformer_0.block_06.ffn_linear1_gate_swiglu | fused_ffn | fused_ffn; fused_ffn_kernel; g18x37x1; b128x1x1; r146; s32768 | 60 | n/a | 46.416 | n/a | 84.4% | 0.000 | linear2_residual (57) | n/a |
| 74 | outer_02.transformer_0.block_06.ffn_linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 60 | n/a | 30.528 | n/a | 97.3% | 0.000 | library_gemm (28) | n/a |
| 75 | outer_02.transformer_1.block_07.attention_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | n/a | 3.584 | n/a | 100.0% | 0.000 | library_gemm (28) | n/a |
| 76 | outer_02.transformer_1.block_07.attention_qkv_projection | wide_qkv | wide_qkv; wide_qkv_kernel; g9x37x1; b128x1x1; r136; s65536 | 60 | n/a | 23.024 | n/a | 67.8% | 0.000 | qk_rope (28) | n/a |
| 77 | outer_02.transformer_1.block_07.attention_qk_rope | qk_rope | qk_rope; fusedQKRoPE19HalfKernel; g361x13x1; b192x1x1; r16; s0 | 60 | n/a | 6.064 | n/a | 95.1% | 0.000 | fa4 (28) | n/a |
| 78 | outer_02.transformer_1.block_07.attention_fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 60 | n/a | 14.832 | n/a | 52.4% | 0.000 | qk_rope (28) | n/a |
| 79 | outer_02.transformer_1.block_07.attention_out_projection_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 60 | n/a | 17.360 | n/a | 96.0% | 0.000 | library_gemm (32) | n/a |
| 80 | outer_02.transformer_1.block_07.ffn_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | n/a | 3.296 | n/a | 100.0% | 0.000 | fused_ffn (28) | n/a |
| 81 | outer_02.transformer_1.block_07.ffn_linear1_gate_swiglu | fused_ffn | fused_ffn; fused_ffn_kernel; g18x37x1; b128x1x1; r146; s32768 | 60 | n/a | 47.248 | n/a | 76.7% | 0.000 | linear2_residual (30) | n/a |
| 82 | outer_02.transformer_1.block_07.ffn_linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 60 | n/a | 32.464 | n/a | 98.1% | 0.000 | fused_ffn (30) | n/a |
| 83 | outer_02.transformer_2.block_08.attention_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | n/a | 3.152 | n/a | 100.0% | 0.000 | library_gemm (28) | n/a |
| 84 | outer_02.transformer_2.block_08.attention_qkv_projection | wide_qkv | wide_qkv; wide_qkv_kernel; g9x37x1; b128x1x1; r136; s65536 | 60 | n/a | 25.424 | n/a | 84.1% | 0.000 | affine_silu (28) | n/a |
| 85 | outer_02.transformer_2.block_08.attention_qk_rope | qk_rope | qk_rope; fusedQKRoPE19HalfKernel; g361x13x1; b192x1x1; r16; s0 | 60 | n/a | 5.776 | n/a | 75.6% | 0.000 | library_gemm (30) | n/a |
| 86 | outer_02.transformer_2.block_08.attention_fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 60 | n/a | 17.280 | n/a | 72.6% | 0.000 | library_gemm (28) | n/a |
| 87 | outer_02.transformer_2.block_08.attention_out_projection_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 60 | n/a | 18.752 | n/a | 96.6% | 0.000 | library_gemm (29) | n/a |
| 88 | outer_02.transformer_2.block_08.ffn_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | n/a | 3.104 | n/a | 99.8% | 0.000 | fa4 (29) | n/a |
| 89 | outer_02.transformer_2.block_08.ffn_linear1_gate_swiglu | fused_ffn | fused_ffn; fused_ffn_kernel; g18x37x1; b128x1x1; r146; s32768 | 60 | n/a | 46.880 | n/a | 58.8% | 0.000 | library_gemm (29) | n/a |
| 90 | outer_02.transformer_2.block_08.ffn_linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 60 | n/a | 31.633 | n/a | 97.5% | 0.000 | fused_ffn (57) | n/a |
| 91 | outer_02.post_norm_silu | affine_silu | affine_silu; affineSiluHalf2Kernel; g4693x1x1; b192x1x1; r17; s0 | 60 | n/a | 4.656 | n/a | 100.0% | 0.000 | fused_ffn (29) | n/a |
| 92 | outer_02.post_projection_c384_to_c768_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b256x1x1; r154; s73728 | 60 | n/a | 23.825 | n/a | 97.1% | 0.000 | linear2_residual (36) | n/a |
| 93 | outer_03.pre_norm_silu | affine_silu | affine_silu; affineSiluHalf2Kernel; g4693x1x1; b384x1x1; r17; s0 | 60 | n/a | 7.232 | n/a | 97.5% | 0.000 | linear2_residual (29) | n/a |
| 94 | outer_03.pre_projection_c768_to_c384 | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 60 | n/a | 15.184 | n/a | 95.7% | 0.000 | fa4 (29) | n/a |
| 95 | outer_03.transformer_0.block_09.attention_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | n/a | 4.497 | n/a | 100.0% | 0.000 | fa4 (29) | n/a |
| 96 | outer_03.transformer_0.block_09.attention_qkv_projection | wide_qkv | wide_qkv; wide_qkv_kernel; g9x37x1; b128x1x1; r136; s65536 | 60 | n/a | 22.096 | n/a | 65.3% | 0.000 | library_gemm (30) | n/a |
| 97 | outer_03.transformer_0.block_09.attention_qk_rope | qk_rope | qk_rope; fusedQKRoPE19HalfKernel; g361x13x1; b192x1x1; r16; s0 | 60 | n/a | 5.360 | n/a | 98.9% | 0.000 | library_gemm (30) | n/a |
| 98 | outer_03.transformer_0.block_09.attention_fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 60 | n/a | 15.824 | n/a | 57.1% | 0.000 | fused_ffn (29) | n/a |
| 99 | outer_03.transformer_0.block_09.attention_out_projection_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 60 | n/a | 21.440 | n/a | 96.6% | 0.000 | fused_ffn (56) | n/a |
| 100 | outer_03.transformer_0.block_09.ffn_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | n/a | 3.504 | n/a | 99.8% | 0.000 | linear2_residual (29) | n/a |
| 101 | outer_03.transformer_0.block_09.ffn_linear1_gate_swiglu | fused_ffn | fused_ffn; fused_ffn_kernel; g18x37x1; b128x1x1; r146; s32768 | 60 | n/a | 44.209 | n/a | 80.7% | 0.000 | linear2_residual (57) | n/a |
| 102 | outer_03.transformer_0.block_09.ffn_linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 60 | n/a | 31.104 | n/a | 97.3% | 0.000 | library_gemm (28) | n/a |
| 103 | outer_03.transformer_1.block_10.attention_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | n/a | 3.632 | n/a | 100.0% | 0.000 | library_gemm (28) | n/a |
| 104 | outer_03.transformer_1.block_10.attention_qkv_projection | wide_qkv | wide_qkv; wide_qkv_kernel; g9x37x1; b128x1x1; r136; s65536 | 60 | n/a | 22.096 | n/a | 66.0% | 0.000 | qk_rope (28) | n/a |
| 105 | outer_03.transformer_1.block_10.attention_qk_rope | qk_rope | qk_rope; fusedQKRoPE19HalfKernel; g361x13x1; b192x1x1; r16; s0 | 60 | n/a | 5.952 | n/a | 92.3% | 0.000 | fa4 (28) | n/a |
| 106 | outer_03.transformer_1.block_10.attention_fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 60 | n/a | 14.592 | n/a | 51.7% | 0.000 | qk_rope (27) | n/a |
| 107 | outer_03.transformer_1.block_10.attention_out_projection_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 60 | n/a | 18.080 | n/a | 95.8% | 0.000 | fused_ffn (29) | n/a |
| 108 | outer_03.transformer_1.block_10.ffn_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | n/a | 3.232 | n/a | 100.0% | 0.000 | fused_ffn (28) | n/a |
| 109 | outer_03.transformer_1.block_10.ffn_linear1_gate_swiglu | fused_ffn | fused_ffn; fused_ffn_kernel; g18x37x1; b128x1x1; r146; s32768 | 60 | n/a | 44.529 | n/a | 75.3% | 0.000 | fused_ffn (29) | n/a |
| 110 | outer_03.transformer_1.block_10.ffn_linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 60 | n/a | 30.544 | n/a | 98.1% | 0.000 | fused_ffn (30) | n/a |
| 111 | outer_03.transformer_2.block_11.attention_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | n/a | 2.992 | n/a | 100.0% | 0.000 | library_gemm (28) | n/a |
| 112 | outer_03.transformer_2.block_11.attention_qkv_projection | wide_qkv | wide_qkv; wide_qkv_kernel; g9x37x1; b128x1x1; r136; s65536 | 60 | n/a | 24.256 | n/a | 83.9% | 0.000 | affine_silu (28) | n/a |
| 113 | outer_03.transformer_2.block_11.attention_qk_rope | qk_rope | qk_rope; fusedQKRoPE19HalfKernel; g361x13x1; b192x1x1; r16; s0 | 60 | n/a | 5.248 | n/a | 74.5% | 0.000 | library_gemm (30) | n/a |
| 114 | outer_03.transformer_2.block_11.attention_fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 60 | n/a | 17.024 | n/a | 73.3% | 0.000 | library_gemm (28) | n/a |
| 115 | outer_03.transformer_2.block_11.attention_out_projection_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 60 | n/a | 18.320 | n/a | 96.6% | 0.000 | library_gemm (29) | n/a |
| 116 | outer_03.transformer_2.block_11.ffn_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | n/a | 3.088 | n/a | 99.5% | 0.000 | fa4 (29) | n/a |
| 117 | outer_03.transformer_2.block_11.ffn_linear1_gate_swiglu | fused_ffn | fused_ffn; fused_ffn_kernel; g18x37x1; b128x1x1; r146; s32768 | 60 | n/a | 46.593 | n/a | 58.2% | 0.000 | library_gemm (29) | n/a |
| 118 | outer_03.transformer_2.block_11.ffn_linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 60 | n/a | 30.177 | n/a | 97.5% | 0.000 | fused_ffn (57) | n/a |
| 119 | outer_03.post_norm_silu | affine_silu | affine_silu; affineSiluHalf2Kernel; g4693x1x1; b192x1x1; r17; s0 | 60 | n/a | 4.624 | n/a | 100.0% | 0.000 | fused_ffn (29) | n/a |
| 120 | outer_03.post_projection_c384_to_c768_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b256x1x1; r154; s73728 | 60 | n/a | 23.328 | n/a | 97.0% | 0.000 | fused_ffn (29) | n/a |
| 121 | outer_04.pre_norm_silu | affine_silu | affine_silu; affineSiluHalf2Kernel; g4693x1x1; b384x1x1; r17; s0 | 60 | n/a | 7.280 | n/a | 97.5% | 0.000 | linear2_residual (29) | n/a |
| 122 | outer_04.pre_projection_c768_to_c384 | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 60 | n/a | 14.528 | n/a | 95.5% | 0.000 | fa4 (29) | n/a |
| 123 | outer_04.transformer_0.block_12.attention_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | n/a | 4.384 | n/a | 100.0% | 0.000 | fa4 (29) | n/a |
| 124 | outer_04.transformer_0.block_12.attention_qkv_projection | wide_qkv | wide_qkv; wide_qkv_kernel; g9x37x1; b128x1x1; r136; s65536 | 60 | n/a | 21.825 | n/a | 65.1% | 0.000 | library_gemm (30) | n/a |
| 125 | outer_04.transformer_0.block_12.attention_qk_rope | qk_rope | qk_rope; fusedQKRoPE19HalfKernel; g361x13x1; b192x1x1; r16; s0 | 60 | n/a | 5.312 | n/a | 98.9% | 0.000 | library_gemm (30) | n/a |
| 126 | outer_04.transformer_0.block_12.attention_fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 60 | n/a | 15.585 | n/a | 56.6% | 0.000 | fused_ffn (28) | n/a |
| 127 | outer_04.transformer_0.block_12.attention_out_projection_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 60 | n/a | 21.408 | n/a | 96.4% | 0.000 | fused_ffn (56) | n/a |
| 128 | outer_04.transformer_0.block_12.ffn_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | n/a | 3.536 | n/a | 99.8% | 0.000 | linear2_residual (29) | n/a |
| 129 | outer_04.transformer_0.block_12.ffn_linear1_gate_swiglu | fused_ffn | fused_ffn; fused_ffn_kernel; g18x37x1; b128x1x1; r146; s32768 | 60 | n/a | 44.736 | n/a | 89.5% | 0.000 | linear2_residual (57) | n/a |
| 130 | outer_04.transformer_0.block_12.ffn_linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 60 | n/a | 28.880 | n/a | 97.3% | 0.000 | library_gemm (28) | n/a |
| 131 | outer_04.transformer_1.block_13.attention_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | n/a | 3.968 | n/a | 100.0% | 0.000 | library_gemm (28) | n/a |
| 132 | outer_04.transformer_1.block_13.attention_qkv_projection | wide_qkv | wide_qkv; wide_qkv_kernel; g9x37x1; b128x1x1; r136; s65536 | 60 | n/a | 23.041 | n/a | 66.5% | 0.000 | qk_rope (28) | n/a |
| 133 | outer_04.transformer_1.block_13.attention_qk_rope | qk_rope | qk_rope; fusedQKRoPE19HalfKernel; g361x13x1; b192x1x1; r16; s0 | 60 | n/a | 6.144 | n/a | 100.0% | 0.000 | fa4 (28) | n/a |
| 134 | outer_04.transformer_1.block_13.attention_fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 60 | n/a | 14.816 | n/a | 47.4% | 0.000 | library_gemm (28) | n/a |
| 135 | outer_04.transformer_1.block_13.attention_out_projection_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 60 | n/a | 19.408 | n/a | 95.8% | 0.000 | fused_ffn (30) | n/a |
| 136 | outer_04.transformer_1.block_13.ffn_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | n/a | 3.408 | n/a | 100.0% | 0.000 | fused_ffn (28) | n/a |
| 137 | outer_04.transformer_1.block_13.ffn_linear1_gate_swiglu | fused_ffn | fused_ffn; fused_ffn_kernel; g18x37x1; b128x1x1; r146; s32768 | 60 | n/a | 45.168 | n/a | 78.8% | 0.000 | linear2_residual (30) | n/a |
| 138 | outer_04.transformer_1.block_13.ffn_linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 60 | n/a | 33.857 | n/a | 98.1% | 0.000 | fused_ffn (30) | n/a |
| 139 | outer_04.transformer_2.block_14.attention_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | n/a | 3.200 | n/a | 100.0% | 0.000 | library_gemm (28) | n/a |
| 140 | outer_04.transformer_2.block_14.attention_qkv_projection | wide_qkv | wide_qkv; wide_qkv_kernel; g9x37x1; b128x1x1; r136; s65536 | 60 | n/a | 25.761 | n/a | 83.8% | 0.000 | affine_silu (28) | n/a |
| 141 | outer_04.transformer_2.block_14.attention_qk_rope | qk_rope | qk_rope; fusedQKRoPE19HalfKernel; g361x13x1; b192x1x1; r16; s0 | 60 | n/a | 7.024 | n/a | 78.2% | 0.000 | library_gemm (30) | n/a |
| 142 | outer_04.transformer_2.block_14.attention_fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 60 | n/a | 17.568 | n/a | 71.1% | 0.000 | library_gemm (28) | n/a |
| 143 | outer_04.transformer_2.block_14.attention_out_projection_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 60 | n/a | 18.753 | n/a | 96.6% | 0.000 | library_gemm (29) | n/a |
| 144 | outer_04.transformer_2.block_14.ffn_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | n/a | 3.168 | n/a | 99.8% | 0.000 | fa4 (29) | n/a |
| 145 | outer_04.transformer_2.block_14.ffn_linear1_gate_swiglu | fused_ffn | fused_ffn; fused_ffn_kernel; g18x37x1; b128x1x1; r146; s32768 | 60 | n/a | 41.361 | n/a | 65.2% | 0.000 | library_gemm (29) | n/a |
| 146 | outer_04.transformer_2.block_14.ffn_linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 60 | n/a | 31.857 | n/a | 97.5% | 0.000 | fused_ffn (57) | n/a |
| 147 | outer_04.post_norm_silu | affine_silu | affine_silu; affineSiluHalf2Kernel; g4693x1x1; b192x1x1; r17; s0 | 60 | n/a | 4.800 | n/a | 100.0% | 0.000 | fused_ffn (29) | n/a |
| 148 | outer_04.post_projection_c384_to_c768_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b256x1x1; r154; s73728 | 60 | n/a | 24.049 | n/a | 97.1% | 0.000 | linear2_residual (29) | n/a |
| 149 | outer_05.pre_norm_silu | affine_silu | affine_silu; affineSiluHalf2Kernel; g4693x1x1; b384x1x1; r17; s0 | 60 | n/a | 7.552 | n/a | 98.2% | 0.000 | linear2_residual (29) | n/a |
| 150 | outer_05.pre_projection_c768_to_c384 | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 60 | n/a | 15.712 | n/a | 96.0% | 0.000 | linear2_residual (29) | n/a |
| 151 | outer_05.transformer_0.block_15.attention_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | n/a | 4.480 | n/a | 100.0% | 0.000 | fa4 (29) | n/a |
| 152 | outer_05.transformer_0.block_15.attention_qkv_projection | wide_qkv | wide_qkv; wide_qkv_kernel; g9x37x1; b128x1x1; r136; s65536 | 60 | n/a | 22.528 | n/a | 65.5% | 0.000 | library_gemm (30) | n/a |
| 153 | outer_05.transformer_0.block_15.attention_qk_rope | qk_rope | qk_rope; fusedQKRoPE19HalfKernel; g361x13x1; b192x1x1; r16; s0 | 60 | n/a | 5.328 | n/a | 98.9% | 0.000 | library_gemm (30) | n/a |
| 154 | outer_05.transformer_0.block_15.attention_fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 60 | n/a | 15.841 | n/a | 58.3% | 0.000 | fused_ffn (28) | n/a |
| 155 | outer_05.transformer_0.block_15.attention_out_projection_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 60 | n/a | 20.736 | n/a | 96.5% | 0.000 | fused_ffn (55) | n/a |
| 156 | outer_05.transformer_0.block_15.ffn_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | n/a | 3.840 | n/a | 99.8% | 0.000 | linear2_residual (29) | n/a |
| 157 | outer_05.transformer_0.block_15.ffn_linear1_gate_swiglu | fused_ffn | fused_ffn; fused_ffn_kernel; g18x37x1; b128x1x1; r146; s32768 | 60 | n/a | 48.624 | n/a | 82.8% | 0.000 | linear2_residual (57) | n/a |
| 158 | outer_05.transformer_0.block_15.ffn_linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 60 | n/a | 31.808 | n/a | 97.4% | 0.000 | library_gemm (28) | n/a |
| 159 | outer_05.transformer_1.block_16.attention_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | n/a | 3.728 | n/a | 100.0% | 0.000 | library_gemm (28) | n/a |
| 160 | outer_05.transformer_1.block_16.attention_qkv_projection | wide_qkv | wide_qkv; wide_qkv_kernel; g9x37x1; b128x1x1; r136; s65536 | 60 | n/a | 22.753 | n/a | 66.5% | 0.000 | qk_rope (28) | n/a |
| 161 | outer_05.transformer_1.block_16.attention_qk_rope | qk_rope | qk_rope; fusedQKRoPE19HalfKernel; g361x13x1; b192x1x1; r16; s0 | 60 | n/a | 6.016 | n/a | 90.7% | 0.000 | fa4 (28) | n/a |
| 162 | outer_05.transformer_1.block_16.attention_fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 60 | n/a | 15.200 | n/a | 53.5% | 0.000 | qk_rope (27) | n/a |
| 163 | outer_05.transformer_1.block_16.attention_out_projection_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 60 | n/a | 18.128 | n/a | 95.8% | 0.000 | library_gemm (30) | n/a |
| 164 | outer_05.transformer_1.block_16.ffn_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | n/a | 3.344 | n/a | 100.0% | 0.000 | fused_ffn (28) | n/a |
| 165 | outer_05.transformer_1.block_16.ffn_linear1_gate_swiglu | fused_ffn | fused_ffn; fused_ffn_kernel; g18x37x1; b128x1x1; r146; s32768 | 60 | n/a | 46.608 | n/a | 75.4% | 0.000 | fused_ffn (29) | n/a |
| 166 | outer_05.transformer_1.block_16.ffn_linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 60 | n/a | 32.544 | n/a | 98.1% | 0.000 | fused_ffn (30) | n/a |
| 167 | outer_05.transformer_2.block_17.attention_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | n/a | 3.200 | n/a | 100.0% | 0.000 | library_gemm (28) | n/a |
| 168 | outer_05.transformer_2.block_17.attention_qkv_projection | wide_qkv | wide_qkv; wide_qkv_kernel; g9x37x1; b128x1x1; r136; s65536 | 60 | n/a | 24.800 | n/a | 83.6% | 0.000 | affine_silu (28) | n/a |
| 169 | outer_05.transformer_2.block_17.attention_qk_rope | qk_rope | qk_rope; fusedQKRoPE19HalfKernel; g361x13x1; b192x1x1; r16; s0 | 60 | n/a | 5.600 | n/a | 74.6% | 0.000 | library_gemm (30) | n/a |
| 170 | outer_05.transformer_2.block_17.attention_fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 60 | n/a | 16.944 | n/a | 73.6% | 0.000 | library_gemm (28) | n/a |
| 171 | outer_05.transformer_2.block_17.attention_out_projection_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 60 | n/a | 18.624 | n/a | 96.6% | 0.000 | library_gemm (29) | n/a |
| 172 | outer_05.transformer_2.block_17.ffn_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | n/a | 3.184 | n/a | 99.8% | 0.000 | fa4 (29) | n/a |
| 173 | outer_05.transformer_2.block_17.ffn_linear1_gate_swiglu | fused_ffn | fused_ffn; fused_ffn_kernel; g18x37x1; b128x1x1; r146; s32768 | 60 | n/a | 47.056 | n/a | 58.3% | 0.000 | library_gemm (29) | n/a |
| 174 | outer_05.transformer_2.block_17.ffn_linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 60 | n/a | 31.408 | n/a | 97.2% | 0.000 | fused_ffn (57) | n/a |
| 175 | outer_05.post_norm_silu | affine_silu | affine_silu; affineSiluHalf2Kernel; g4693x1x1; b192x1x1; r17; s0 | 60 | n/a | 4.576 | n/a | 100.0% | 0.000 | fused_ffn (29) | n/a |
| 176 | outer_05.post_projection_c384_to_c768_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b256x1x1; r154; s73728 | 60 | n/a | 24.593 | n/a | 97.0% | 0.000 | fused_ffn (29) | n/a |
| 177 | outer_06.pre_norm_silu | affine_silu | affine_silu; affineSiluHalf2Kernel; g4693x1x1; b384x1x1; r17; s0 | 60 | n/a | 7.584 | n/a | 97.0% | 0.000 | linear2_residual (29) | n/a |
| 178 | outer_06.pre_projection_c768_to_c384 | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 60 | n/a | 15.536 | n/a | 95.8% | 0.000 | fa4 (29) | n/a |
| 179 | outer_06.transformer_0.block_18.attention_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | n/a | 4.608 | n/a | 100.0% | 0.000 | fa4 (29) | n/a |
| 180 | outer_06.transformer_0.block_18.attention_qkv_projection | wide_qkv | wide_qkv; wide_qkv_kernel; g9x37x1; b128x1x1; r136; s65536 | 60 | n/a | 21.889 | n/a | 64.7% | 0.000 | library_gemm (29) | n/a |
| 181 | outer_06.transformer_0.block_18.attention_qk_rope | qk_rope | qk_rope; fusedQKRoPE19HalfKernel; g361x13x1; b192x1x1; r16; s0 | 60 | n/a | 5.312 | n/a | 97.7% | 0.000 | library_gemm (30) | n/a |
| 182 | outer_06.transformer_0.block_18.attention_fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 60 | n/a | 15.744 | n/a | 58.5% | 0.000 | fused_ffn (29) | n/a |
| 183 | outer_06.transformer_0.block_18.attention_out_projection_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 60 | n/a | 21.520 | n/a | 96.5% | 0.000 | fused_ffn (56) | n/a |
| 184 | outer_06.transformer_0.block_18.ffn_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | n/a | 3.456 | n/a | 99.8% | 0.000 | linear2_residual (29) | n/a |
| 185 | outer_06.transformer_0.block_18.ffn_linear1_gate_swiglu | fused_ffn | fused_ffn; fused_ffn_kernel; g18x37x1; b128x1x1; r146; s32768 | 60 | n/a | 47.760 | n/a | 87.2% | 0.000 | linear2_residual (57) | n/a |
| 186 | outer_06.transformer_0.block_18.ffn_linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 60 | n/a | 28.528 | n/a | 97.2% | 0.000 | library_gemm (28) | n/a |
| 187 | outer_06.transformer_1.block_19.attention_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | n/a | 4.000 | n/a | 100.0% | 0.000 | library_gemm (28) | n/a |
| 188 | outer_06.transformer_1.block_19.attention_qkv_projection | wide_qkv | wide_qkv; wide_qkv_kernel; g9x37x1; b128x1x1; r136; s65536 | 60 | n/a | 23.424 | n/a | 68.5% | 0.000 | library_gemm (29) | n/a |
| 189 | outer_06.transformer_1.block_19.attention_qk_rope | qk_rope | qk_rope; fusedQKRoPE19HalfKernel; g361x13x1; b192x1x1; r16; s0 | 60 | n/a | 6.176 | n/a | 99.1% | 0.000 | fa4 (28) | n/a |
| 190 | outer_06.transformer_1.block_19.attention_fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 60 | n/a | 14.912 | n/a | 48.3% | 0.000 | library_gemm (27) | n/a |
| 191 | outer_06.transformer_1.block_19.attention_out_projection_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 60 | n/a | 19.265 | n/a | 95.8% | 0.000 | fused_ffn (30) | n/a |
| 192 | outer_06.transformer_1.block_19.ffn_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | n/a | 3.248 | n/a | 100.0% | 0.000 | fused_ffn (28) | n/a |
| 193 | outer_06.transformer_1.block_19.ffn_linear1_gate_swiglu | fused_ffn | fused_ffn; fused_ffn_kernel; g18x37x1; b128x1x1; r146; s32768 | 60 | n/a | 46.800 | n/a | 77.4% | 0.000 | linear2_residual (30) | n/a |
| 194 | outer_06.transformer_1.block_19.ffn_linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 60 | n/a | 33.153 | n/a | 98.1% | 0.000 | fused_ffn (30) | n/a |
| 195 | outer_06.transformer_2.block_20.attention_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | n/a | 3.136 | n/a | 100.0% | 0.000 | library_gemm (28) | n/a |
| 196 | outer_06.transformer_2.block_20.attention_qkv_projection | wide_qkv | wide_qkv; wide_qkv_kernel; g9x37x1; b128x1x1; r136; s65536 | 60 | n/a | 27.040 | n/a | 83.3% | 0.000 | linear2_residual (28) | n/a |
| 197 | outer_06.transformer_2.block_20.attention_qk_rope | qk_rope | qk_rope; fusedQKRoPE19HalfKernel; g361x13x1; b192x1x1; r16; s0 | 60 | n/a | 6.960 | n/a | 88.0% | 0.000 | library_gemm (30) | n/a |
| 198 | outer_06.transformer_2.block_20.attention_fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 60 | n/a | 17.217 | n/a | 66.5% | 0.000 | library_gemm (28) | n/a |
| 199 | outer_06.transformer_2.block_20.attention_out_projection_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 60 | n/a | 18.945 | n/a | 96.6% | 0.000 | library_gemm (29) | n/a |
| 200 | outer_06.transformer_2.block_20.ffn_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | n/a | 3.136 | n/a | 99.8% | 0.000 | fa4 (29) | n/a |
| 201 | outer_06.transformer_2.block_20.ffn_linear1_gate_swiglu | fused_ffn | fused_ffn; fused_ffn_kernel; g18x37x1; b128x1x1; r146; s32768 | 60 | n/a | 40.849 | n/a | 65.5% | 0.000 | library_gemm (29) | n/a |
| 202 | outer_06.transformer_2.block_20.ffn_linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 60 | n/a | 32.145 | n/a | 97.5% | 0.000 | fused_ffn (57) | n/a |
| 203 | outer_06.post_norm_silu | affine_silu | affine_silu; affineSiluHalf2Kernel; g4693x1x1; b192x1x1; r17; s0 | 60 | n/a | 4.720 | n/a | 100.0% | 0.000 | fused_ffn (29) | n/a |
| 204 | outer_06.post_projection_c384_to_c768_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b256x1x1; r154; s73728 | 60 | n/a | 25.169 | n/a | 97.1% | 0.000 | fused_ffn (29) | n/a |
| 205 | outer_07.pre_norm_silu | affine_silu | affine_silu; affineSiluHalf2Kernel; g4693x1x1; b384x1x1; r17; s0 | 60 | n/a | 7.600 | n/a | 98.3% | 0.000 | linear2_residual (29) | n/a |
| 206 | outer_07.pre_projection_c768_to_c384 | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 60 | n/a | 15.825 | n/a | 95.9% | 0.000 | fa4 (29) | n/a |
| 207 | outer_07.transformer_0.block_21.attention_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | n/a | 4.320 | n/a | 100.0% | 0.000 | fa4 (29) | n/a |
| 208 | outer_07.transformer_0.block_21.attention_qkv_projection | wide_qkv | wide_qkv; wide_qkv_kernel; g9x37x1; b128x1x1; r136; s65536 | 60 | n/a | 22.849 | n/a | 66.6% | 0.000 | library_gemm (30) | n/a |
| 209 | outer_07.transformer_0.block_21.attention_qk_rope | qk_rope | qk_rope; fusedQKRoPE19HalfKernel; g361x13x1; b192x1x1; r16; s0 | 60 | n/a | 5.440 | n/a | 98.8% | 0.000 | library_gemm (30) | n/a |
| 210 | outer_07.transformer_0.block_21.attention_fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 60 | n/a | 15.968 | n/a | 58.1% | 0.000 | fused_ffn (29) | n/a |
| 211 | outer_07.transformer_0.block_21.attention_out_projection_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 60 | n/a | 20.288 | n/a | 96.4% | 0.000 | fused_ffn (57) | n/a |
| 212 | outer_07.transformer_0.block_21.ffn_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | n/a | 3.792 | n/a | 99.2% | 0.000 | linear2_residual (29) | n/a |
| 213 | outer_07.transformer_0.block_21.ffn_linear1_gate_swiglu | fused_ffn | fused_ffn; fused_ffn_kernel; g18x37x1; b128x1x1; r146; s32768 | 60 | n/a | 45.025 | n/a | 92.0% | 0.000 | linear2_residual (57) | n/a |
| 214 | outer_07.transformer_0.block_21.ffn_linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 60 | n/a | 34.385 | n/a | 97.6% | 0.000 | library_gemm (28) | n/a |
| 215 | outer_07.transformer_1.block_22.attention_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | n/a | 3.904 | n/a | 100.0% | 0.000 | library_gemm (28) | n/a |
| 216 | outer_07.transformer_1.block_22.attention_qkv_projection | wide_qkv | wide_qkv; wide_qkv_kernel; g9x37x1; b128x1x1; r136; s65536 | 60 | n/a | 23.728 | n/a | 65.6% | 0.000 | qk_rope (28) | n/a |
| 217 | outer_07.transformer_1.block_22.attention_qk_rope | qk_rope | qk_rope; fusedQKRoPE19HalfKernel; g361x13x1; b192x1x1; r16; s0 | 60 | n/a | 6.176 | n/a | 100.0% | 0.000 | fa4 (28) | n/a |
| 218 | outer_07.transformer_1.block_22.attention_fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 60 | n/a | 15.168 | n/a | 49.1% | 0.000 | library_gemm (28) | n/a |
| 219 | outer_07.transformer_1.block_22.attention_out_projection_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 60 | n/a | 19.105 | n/a | 96.0% | 0.000 | fused_ffn (30) | n/a |
| 220 | outer_07.transformer_1.block_22.ffn_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | n/a | 3.376 | n/a | 100.0% | 0.000 | fused_ffn (28) | n/a |
| 221 | outer_07.transformer_1.block_22.ffn_linear1_gate_swiglu | fused_ffn | fused_ffn; fused_ffn_kernel; g18x37x1; b128x1x1; r146; s32768 | 60 | n/a | 43.553 | n/a | 83.3% | 0.000 | linear2_residual (30) | n/a |
| 222 | outer_07.transformer_1.block_22.ffn_linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 60 | n/a | 36.081 | n/a | 98.3% | 0.000 | fused_ffn (30) | n/a |
| 223 | outer_07.transformer_2.block_23.attention_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | n/a | 3.072 | n/a | 100.0% | 0.000 | library_gemm (28) | n/a |
| 224 | outer_07.transformer_2.block_23.attention_qkv_projection | wide_qkv | wide_qkv; wide_qkv_kernel; g9x37x1; b128x1x1; r136; s65536 | 60 | n/a | 26.032 | n/a | 84.5% | 0.000 | affine_silu (28) | n/a |
| 225 | outer_07.transformer_2.block_23.attention_qk_rope | qk_rope | qk_rope; fusedQKRoPE19HalfKernel; g361x13x1; b192x1x1; r16; s0 | 60 | n/a | 6.208 | n/a | 76.2% | 0.000 | library_gemm (30) | n/a |
| 226 | outer_07.transformer_2.block_23.attention_fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 60 | n/a | 17.904 | n/a | 72.4% | 0.000 | library_gemm (28) | n/a |
| 227 | outer_07.transformer_2.block_23.attention_out_projection_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 60 | n/a | 18.768 | n/a | 96.6% | 0.000 | library_gemm (29) | n/a |
| 228 | outer_07.transformer_2.block_23.ffn_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | n/a | 3.104 | n/a | 99.8% | 0.000 | fa4 (29) | n/a |
| 229 | outer_07.transformer_2.block_23.ffn_linear1_gate_swiglu | fused_ffn | fused_ffn; fused_ffn_kernel; g18x37x1; b128x1x1; r146; s32768 | 60 | n/a | 42.160 | n/a | 67.3% | 0.000 | library_gemm (29) | n/a |
| 230 | outer_07.transformer_2.block_23.ffn_linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 60 | n/a | 32.288 | n/a | 97.7% | 0.000 | fused_ffn (57) | n/a |
| 231 | outer_07.post_norm_silu | affine_silu | affine_silu; affineSiluHalf2Kernel; g4693x1x1; b192x1x1; r17; s0 | 60 | n/a | 4.928 | n/a | 100.0% | 0.000 | fused_ffn (29) | n/a |
| 232 | outer_07.post_projection_c384_to_c768_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b256x1x1; r154; s73728 | 60 | n/a | 23.152 | n/a | 97.1% | 0.000 | linear2_residual (30) | n/a |
| 233 | outer_08.pre_norm_silu | affine_silu | affine_silu; affineSiluHalf2Kernel; g4693x1x1; b384x1x1; r17; s0 | 60 | n/a | 7.632 | n/a | 97.7% | 0.000 | linear2_residual (29) | n/a |
| 234 | outer_08.pre_projection_c768_to_c384 | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 60 | n/a | 16.144 | n/a | 95.9% | 0.000 | linear2_residual (29) | n/a |
| 235 | outer_08.transformer_0.block_24.attention_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | n/a | 4.464 | n/a | 100.0% | 0.000 | fa4 (29) | n/a |
| 236 | outer_08.transformer_0.block_24.attention_qkv_projection | wide_qkv | wide_qkv; wide_qkv_kernel; g9x37x1; b128x1x1; r136; s65536 | 60 | n/a | 22.913 | n/a | 65.6% | 0.000 | library_gemm (30) | n/a |
| 237 | outer_08.transformer_0.block_24.attention_qk_rope | qk_rope | qk_rope; fusedQKRoPE19HalfKernel; g361x13x1; b192x1x1; r16; s0 | 60 | n/a | 5.280 | n/a | 97.8% | 0.000 | library_gemm (30) | n/a |
| 238 | outer_08.transformer_0.block_24.attention_fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 60 | n/a | 16.272 | n/a | 59.4% | 0.000 | fused_ffn (28) | n/a |
| 239 | outer_08.transformer_0.block_24.attention_out_projection_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 60 | n/a | 20.753 | n/a | 96.6% | 0.000 | fused_ffn (56) | n/a |
| 240 | outer_08.transformer_0.block_24.ffn_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | n/a | 3.888 | n/a | 99.8% | 0.000 | linear2_residual (29) | n/a |
| 241 | outer_08.transformer_0.block_24.ffn_linear1_gate_swiglu | fused_ffn | fused_ffn; fused_ffn_kernel; g18x37x1; b128x1x1; r146; s32768 | 60 | n/a | 49.264 | n/a | 87.4% | 0.000 | linear2_residual (57) | n/a |
| 242 | outer_08.transformer_0.block_24.ffn_linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 60 | n/a | 32.800 | n/a | 97.4% | 0.000 | library_gemm (28) | n/a |
| 243 | outer_08.transformer_1.block_25.attention_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | n/a | 3.920 | n/a | 100.0% | 0.000 | library_gemm (28) | n/a |
| 244 | outer_08.transformer_1.block_25.attention_qkv_projection | wide_qkv | wide_qkv; wide_qkv_kernel; g9x37x1; b128x1x1; r136; s65536 | 60 | n/a | 23.264 | n/a | 65.4% | 0.000 | qk_rope (28) | n/a |
| 245 | outer_08.transformer_1.block_25.attention_qk_rope | qk_rope | qk_rope; fusedQKRoPE19HalfKernel; g361x13x1; b192x1x1; r16; s0 | 60 | n/a | 6.128 | n/a | 96.6% | 0.000 | fa4 (28) | n/a |
| 246 | outer_08.transformer_1.block_25.attention_fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 60 | n/a | 15.088 | n/a | 51.7% | 0.000 | qk_rope (26) | n/a |
| 247 | outer_08.transformer_1.block_25.attention_out_projection_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 60 | n/a | 18.401 | n/a | 96.0% | 0.000 | fused_ffn (29) | n/a |
| 248 | outer_08.transformer_1.block_25.ffn_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | n/a | 3.408 | n/a | 100.0% | 0.000 | fused_ffn (28) | n/a |
| 249 | outer_08.transformer_1.block_25.ffn_linear1_gate_swiglu | fused_ffn | fused_ffn; fused_ffn_kernel; g18x37x1; b128x1x1; r146; s32768 | 60 | n/a | 47.745 | n/a | 77.4% | 0.000 | linear2_residual (30) | n/a |
| 250 | outer_08.transformer_1.block_25.ffn_linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 60 | n/a | 33.488 | n/a | 98.2% | 0.000 | fused_ffn (30) | n/a |
| 251 | outer_08.transformer_2.block_26.attention_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | n/a | 3.200 | n/a | 100.0% | 0.000 | library_gemm (28) | n/a |
| 252 | outer_08.transformer_2.block_26.attention_qkv_projection | wide_qkv | wide_qkv; wide_qkv_kernel; g9x37x1; b128x1x1; r136; s65536 | 60 | n/a | 26.113 | n/a | 82.3% | 0.000 | affine_silu (28) | n/a |
| 253 | outer_08.transformer_2.block_26.attention_qk_rope | qk_rope | qk_rope; fusedQKRoPE19HalfKernel; g361x13x1; b192x1x1; r16; s0 | 60 | n/a | 6.128 | n/a | 76.5% | 0.000 | library_gemm (30) | n/a |
| 254 | outer_08.transformer_2.block_26.attention_fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 60 | n/a | 17.344 | n/a | 72.0% | 0.000 | library_gemm (28) | n/a |
| 255 | outer_08.transformer_2.block_26.attention_out_projection_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 60 | n/a | 18.816 | n/a | 96.6% | 0.000 | library_gemm (29) | n/a |
| 256 | outer_08.transformer_2.block_26.ffn_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | n/a | 3.168 | n/a | 99.7% | 0.000 | fa4 (29) | n/a |
| 257 | outer_08.transformer_2.block_26.ffn_linear1_gate_swiglu | fused_ffn | fused_ffn; fused_ffn_kernel; g18x37x1; b128x1x1; r146; s32768 | 60 | n/a | 46.417 | n/a | 61.1% | 0.000 | library_gemm (29) | n/a |
| 258 | outer_08.transformer_2.block_26.ffn_linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 60 | n/a | 31.392 | n/a | 97.5% | 0.000 | fused_ffn (57) | n/a |
| 259 | outer_08.post_norm_silu | affine_silu | affine_silu; affineSiluHalf2Kernel; g4693x1x1; b192x1x1; r17; s0 | 60 | n/a | 4.752 | n/a | 100.0% | 0.000 | fused_ffn (29) | n/a |
| 260 | outer_08.post_projection_c384_to_c768_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b256x1x1; r154; s73728 | 60 | n/a | 22.352 | n/a | 97.1% | 0.000 | fused_ffn (29) | n/a |
| 261 | outer_09.pre_norm_silu | affine_silu | affine_silu; affineSiluHalf2Kernel; g4693x1x1; b384x1x1; r17; s0 | 60 | n/a | 7.280 | n/a | 97.8% | 0.000 | linear2_residual (29) | n/a |
| 262 | outer_09.pre_projection_c768_to_c384 | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 60 | n/a | 15.729 | n/a | 95.9% | 0.000 | fa4 (29) | n/a |
| 263 | outer_09.transformer_0.block_27.attention_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | n/a | 4.528 | n/a | 100.0% | 0.000 | fa4 (29) | n/a |
| 264 | outer_09.transformer_0.block_27.attention_qkv_projection | wide_qkv | wide_qkv; wide_qkv_kernel; g9x37x1; b128x1x1; r136; s65536 | 60 | n/a | 22.080 | n/a | 65.6% | 0.000 | library_gemm (29) | n/a |
| 265 | outer_09.transformer_0.block_27.attention_qk_rope | qk_rope | qk_rope; fusedQKRoPE19HalfKernel; g361x13x1; b192x1x1; r16; s0 | 60 | n/a | 5.345 | n/a | 98.9% | 0.000 | library_gemm (30) | n/a |
| 266 | outer_09.transformer_0.block_27.attention_fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 60 | n/a | 15.824 | n/a | 57.4% | 0.000 | fused_ffn (28) | n/a |
| 267 | outer_09.transformer_0.block_27.attention_out_projection_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 60 | n/a | 21.872 | n/a | 96.9% | 0.000 | fused_ffn (56) | n/a |
| 268 | outer_09.transformer_0.block_27.ffn_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | n/a | 3.968 | n/a | 99.8% | 0.000 | linear2_residual (29) | n/a |
| 269 | outer_09.transformer_0.block_27.ffn_linear1_gate_swiglu | fused_ffn | fused_ffn; fused_ffn_kernel; g18x37x1; b128x1x1; r146; s32768 | 60 | n/a | 47.344 | n/a | 79.6% | 0.000 | linear2_residual (56) | n/a |
| 270 | outer_09.transformer_0.block_27.ffn_linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 60 | n/a | 29.265 | n/a | 97.2% | 0.000 | library_gemm (28) | n/a |
| 271 | outer_09.transformer_1.block_28.attention_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | n/a | 3.505 | n/a | 100.0% | 0.000 | library_gemm (28) | n/a |
| 272 | outer_09.transformer_1.block_28.attention_qkv_projection | wide_qkv | wide_qkv; wide_qkv_kernel; g9x37x1; b128x1x1; r136; s65536 | 60 | n/a | 22.224 | n/a | 66.9% | 0.000 | qk_rope (28) | n/a |
| 273 | outer_09.transformer_1.block_28.attention_qk_rope | qk_rope | qk_rope; fusedQKRoPE19HalfKernel; g361x13x1; b192x1x1; r16; s0 | 60 | n/a | 5.968 | n/a | 95.7% | 0.000 | fa4 (28) | n/a |
| 274 | outer_09.transformer_1.block_28.attention_fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 60 | n/a | 14.688 | n/a | 49.6% | 0.000 | qk_rope (27) | n/a |
| 275 | outer_09.transformer_1.block_28.attention_out_projection_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 60 | n/a | 17.728 | n/a | 95.9% | 0.000 | library_gemm (30) | n/a |
| 276 | outer_09.transformer_1.block_28.ffn_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | n/a | 3.136 | n/a | 100.0% | 0.000 | fused_ffn (28) | n/a |
| 277 | outer_09.transformer_1.block_28.ffn_linear1_gate_swiglu | fused_ffn | fused_ffn; fused_ffn_kernel; g18x37x1; b128x1x1; r146; s32768 | 60 | n/a | 45.281 | n/a | 75.6% | 0.000 | fused_ffn (29) | n/a |
| 278 | outer_09.transformer_1.block_28.ffn_linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 60 | n/a | 30.865 | n/a | 98.1% | 0.000 | fused_ffn (30) | n/a |
| 279 | outer_09.transformer_2.block_29.attention_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | n/a | 3.152 | n/a | 100.0% | 0.000 | library_gemm (28) | n/a |
| 280 | outer_09.transformer_2.block_29.attention_qkv_projection | wide_qkv | wide_qkv; wide_qkv_kernel; g9x37x1; b128x1x1; r136; s65536 | 60 | n/a | 24.704 | n/a | 83.5% | 0.000 | affine_silu (28) | n/a |
| 281 | outer_09.transformer_2.block_29.attention_qk_rope | qk_rope | qk_rope; fusedQKRoPE19HalfKernel; g361x13x1; b192x1x1; r16; s0 | 60 | n/a | 5.808 | n/a | 75.7% | 0.000 | library_gemm (30) | n/a |
| 282 | outer_09.transformer_2.block_29.attention_fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 60 | n/a | 16.817 | n/a | 71.7% | 0.000 | library_gemm (28) | n/a |
| 283 | outer_09.transformer_2.block_29.attention_out_projection_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 60 | n/a | 18.576 | n/a | 96.5% | 0.000 | library_gemm (29) | n/a |
| 284 | outer_09.transformer_2.block_29.ffn_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | n/a | 3.056 | n/a | 99.5% | 0.000 | fa4 (29) | n/a |
| 285 | outer_09.transformer_2.block_29.ffn_linear1_gate_swiglu | fused_ffn | fused_ffn; fused_ffn_kernel; g18x37x1; b128x1x1; r146; s32768 | 60 | n/a | 45.761 | n/a | 58.5% | 0.000 | library_gemm (29) | n/a |
| 286 | outer_09.transformer_2.block_29.ffn_linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 60 | n/a | 30.368 | n/a | 97.4% | 0.000 | fused_ffn (57) | n/a |
| 287 | outer_09.post_norm_silu | affine_silu | affine_silu; affineSiluHalf2Kernel; g4693x1x1; b192x1x1; r17; s0 | 60 | n/a | 4.880 | n/a | 100.0% | 0.000 | fused_ffn (29) | n/a |
| 288 | outer_09.post_projection_c384_to_c768_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b256x1x1; r154; s73728 | 60 | n/a | 23.024 | n/a | 96.9% | 0.000 | linear2_residual (29) | n/a |
| 289 | outer_10.pre_norm_silu | affine_silu | affine_silu; affineSiluHalf2Kernel; g4693x1x1; b384x1x1; r17; s0 | 60 | n/a | 7.408 | n/a | 97.6% | 0.000 | linear2_residual (29) | n/a |
| 290 | outer_10.pre_projection_c768_to_c384 | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 60 | n/a | 14.960 | n/a | 95.7% | 0.000 | linear2_residual (29) | n/a |
| 291 | outer_10.transformer_0.block_30.attention_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | n/a | 4.544 | n/a | 100.0% | 0.000 | fa4 (29) | n/a |
| 292 | outer_10.transformer_0.block_30.attention_qkv_projection | wide_qkv | wide_qkv; wide_qkv_kernel; g9x37x1; b128x1x1; r136; s65536 | 60 | n/a | 21.936 | n/a | 65.1% | 0.000 | library_gemm (30) | n/a |
| 293 | outer_10.transformer_0.block_30.attention_qk_rope | qk_rope | qk_rope; fusedQKRoPE19HalfKernel; g361x13x1; b192x1x1; r16; s0 | 60 | n/a | 5.328 | n/a | 98.9% | 0.000 | library_gemm (30) | n/a |
| 294 | outer_10.transformer_0.block_30.attention_fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 60 | n/a | 15.440 | n/a | 57.3% | 0.000 | fused_ffn (29) | n/a |
| 295 | outer_10.transformer_0.block_30.attention_out_projection_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 60 | n/a | 21.441 | n/a | 96.8% | 0.000 | fused_ffn (56) | n/a |
| 296 | outer_10.transformer_0.block_30.ffn_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | n/a | 3.456 | n/a | 99.8% | 0.000 | linear2_residual (29) | n/a |
| 297 | outer_10.transformer_0.block_30.ffn_linear1_gate_swiglu | fused_ffn | fused_ffn; fused_ffn_kernel; g18x37x1; b128x1x1; r146; s32768 | 60 | n/a | 43.392 | n/a | 93.0% | 0.000 | linear2_residual (57) | n/a |
| 298 | outer_10.transformer_0.block_30.ffn_linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 60 | n/a | 29.665 | n/a | 97.4% | 0.000 | library_gemm (28) | n/a |
| 299 | outer_10.transformer_1.block_31.attention_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | n/a | 3.936 | n/a | 100.0% | 0.000 | library_gemm (28) | n/a |
| 300 | outer_10.transformer_1.block_31.attention_qkv_projection | wide_qkv | wide_qkv; wide_qkv_kernel; g9x37x1; b128x1x1; r136; s65536 | 60 | n/a | 23.488 | n/a | 66.8% | 0.000 | qk_rope (28) | n/a |
| 301 | outer_10.transformer_1.block_31.attention_qk_rope | qk_rope | qk_rope; fusedQKRoPE19HalfKernel; g361x13x1; b192x1x1; r16; s0 | 60 | n/a | 6.128 | n/a | 100.0% | 0.000 | fa4 (28) | n/a |
| 302 | outer_10.transformer_1.block_31.attention_fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 60 | n/a | 15.024 | n/a | 49.3% | 0.000 | library_gemm (28) | n/a |
| 303 | outer_10.transformer_1.block_31.attention_out_projection_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 60 | n/a | 19.600 | n/a | 96.1% | 0.000 | fused_ffn (30) | n/a |
| 304 | outer_10.transformer_1.block_31.ffn_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | n/a | 3.248 | n/a | 100.0% | 0.000 | fused_ffn (28) | n/a |
| 305 | outer_10.transformer_1.block_31.ffn_linear1_gate_swiglu | fused_ffn | fused_ffn; fused_ffn_kernel; g18x37x1; b128x1x1; r146; s32768 | 60 | n/a | 45.696 | n/a | 82.8% | 0.000 | linear2_residual (30) | n/a |
| 306 | outer_10.transformer_1.block_31.ffn_linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 60 | n/a | 33.504 | n/a | 98.1% | 0.000 | fused_ffn (30) | n/a |
| 307 | outer_10.transformer_2.block_32.attention_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | n/a | 3.216 | n/a | 100.0% | 0.000 | library_gemm (28) | n/a |
| 308 | outer_10.transformer_2.block_32.attention_qkv_projection | wide_qkv | wide_qkv; wide_qkv_kernel; g9x37x1; b128x1x1; r136; s65536 | 60 | n/a | 26.817 | n/a | 84.2% | 0.000 | linear2_residual (28) | n/a |
| 309 | outer_10.transformer_2.block_32.attention_qk_rope | qk_rope | qk_rope; fusedQKRoPE19HalfKernel; g361x13x1; b192x1x1; r16; s0 | 60 | n/a | 8.672 | n/a | 90.8% | 0.000 | library_gemm (30) | n/a |
| 310 | outer_10.transformer_2.block_32.attention_fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 60 | n/a | 17.169 | n/a | 57.0% | 0.000 | library_gemm (28) | n/a |
| 311 | outer_10.transformer_2.block_32.attention_out_projection_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 60 | n/a | 14.352 | n/a | 95.7% | 0.000 | library_gemm (36) | n/a |
| 312 | outer_10.transformer_2.block_32.ffn_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 60 | n/a | 2.704 | n/a | 97.9% | 0.000 | library_gemm (47) | n/a |
| 313 | outer_10.transformer_2.block_32.ffn_linear1_gate_swiglu | fused_ffn | fused_ffn; fused_ffn_kernel; g18x37x1; b128x1x1; r146; s32768 | 60 | n/a | 47.937 | n/a | 60.3% | 0.000 | library_gemm (30) | n/a |
| 314 | outer_10.transformer_2.block_32.ffn_linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 60 | n/a | 27.697 | n/a | 97.0% | 0.000 | fused_ffn (29) | n/a |
| 315 | outer_10.post_norm_silu | affine_silu | affine_silu; affineSiluHalf2Kernel; g4693x1x1; b192x1x1; r17; s0 | 60 | n/a | 4.192 | n/a | 97.6% | 0.000 | library_gemm (30) | n/a |
| 316 | outer_10.post_projection_c384_to_c768_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b256x1x1; r154; s73728 | 60 | n/a | 19.952 | n/a | 94.1% | 0.000 | head_elementwise (28) | n/a |
| 317 | trunk.tip_norm_silu | affine_silu | affine_silu; affineSiluHalf2Kernel; g4693x1x1; b384x1x1; r17; s0 | 60 | n/a | 6.816 | n/a | 98.4% | 0.000 | library_gemm (31) | n/a |
| 318 | policy.p1_conv | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r80; s98304 | 60 | n/a | 7.152 | n/a | 95.9% | 0.000 | library_gemm (28) | n/a |
| 319 | policy.g1_conv | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r80; s98304 | 60 | n/a | 6.752 | n/a | 92.6% | 0.000 | fa4 (28) | n/a |
| 320 | policy.g1_norm_silu | head_elementwise | head_elementwise; applyCScaleBiasNHWCSiluHalfKernel; g1x73x13; b96x5x1; r16; s0 | 60 | n/a | 2.544 | n/a | 95.9% | 0.000 | library_gemm (40) | n/a |
| 321 | policy.g1_half_to_float | copy_reformat | copy_reformat; copyFromHalfKernel; g880x1x1; b512x1x1; r16; s0 | 60 | n/a | 1.920 | n/a | 81.6% | 0.000 | head_elementwise (24) | n/a |
| 322 | policy.g1_global_pool | head_elementwise | head_elementwise; gPoolChannelsNHWCKernel; g2x1x13; b64x8x1; r22; s4096 | 60 | n/a | 5.184 | n/a | 90.8% | 0.000 | library_gemm (50) | n/a |
| 323 | policy.gpool_to_bias_matmul | library_gemm | library_gemm; gemmSN_NN_kernel; g2x2x1; b256x1x1; r64; s21504 | 60 | n/a | 6.368 | n/a | 89.8% | 0.000 | library_gemm (44) | n/a |
| 324 | policy.p1_half_to_float | copy_reformat | copy_reformat; copyFromHalfKernel; g880x1x1; b512x1x1; r16; s0 | 60 | n/a | 1.920 | n/a | 73.5% | 0.000 | fused_ffn (28) | n/a |
| 325 | policy.gpool_bias_add | head_elementwise | head_elementwise; addNCBiasInplaceNHWCKernel; g1x73x13; b96x5x1; r16; s0 | 60 | n/a | 2.336 | n/a | 70.2% | 0.000 | fused_ffn (28) | n/a |
| 326 | policy.p1_norm_silu | head_elementwise | head_elementwise; applyCScaleBiasNHWCSiluKernel; g1x73x13; b96x5x1; r16; s0 | 60 | n/a | 2.688 | n/a | 69.7% | 0.000 | fused_ffn (28) | n/a |
| 327 | policy.p2_conv | library_gemm | library_gemm; Kernel2; g74x1x1; b128x1x1; r90; s98304 | 60 | n/a | 4.640 | n/a | 85.7% | 0.000 | fused_ffn (28) | n/a |
| 328 | policy.gpool_to_pass_matmul | library_gemm | library_gemm; gemmSN_NN_kernel; g2x2x1; b256x1x1; r64; s21504 | 60 | n/a | 6.784 | n/a | 76.2% | 0.000 | linear2_residual (28) | n/a |
| 329 | policy.pass_bias_silu | head_elementwise | head_elementwise; addCBiasInplaceNCKernelSilu; g1x3x1; b96x5x1; r16; s0 | 60 | n/a | 1.248 | n/a | 93.8% | 0.000 | linear2_residual (28) | n/a |
| 330 | policy.gpool_to_pass_matmul2 | library_gemm | library_gemm; gemmSN_NN_kernel; g1x2x1; b256x1x1; r64; s21504 | 60 | n/a | 2.736 | n/a | 94.2% | 0.000 | linear2_residual (28) | n/a |
| 331 | value.v1_conv | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r118; s98304 | 60 | n/a | 9.920 | n/a | 91.8% | 0.000 | linear2_residual (28) | n/a |
| 332 | value.v1_norm_silu | head_elementwise | head_elementwise; applyCScaleBiasNHWCSiluHalfKernel; g1x181x13; b192x2x1; r16; s0 | 60 | n/a | 4.112 | n/a | 77.8% | 0.000 | library_gemm (28) | n/a |
| 333 | value.v1_half_to_float | copy_reformat | copy_reformat; copyFromHalfKernel; g1760x1x1; b512x1x1; r16; s0 | 60 | n/a | 3.056 | n/a | 81.9% | 0.000 | library_gemm (32) | n/a |
| 334 | value.v1_global_pool | head_elementwise | head_elementwise; valueHeadPoolChannelsNHWCKernel; g3x1x13; b64x8x1; r22; s2048 | 60 | n/a | 4.608 | n/a | 97.8% | 0.000 | library_gemm (58) | n/a |
| 335 | value.v2_matmul | library_gemm | library_gemm; gemmSN_NN_kernel; g3x2x1; b256x1x1; r64; s21504 | 60 | n/a | 14.049 | n/a | 90.8% | 0.000 | library_gemm (45) | n/a |
| 336 | value.v2_bias_silu | head_elementwise | head_elementwise; addCBiasInplaceNCKernelSilu; g1x7x1; b192x2x1; r16; s0 | 60 | n/a | 2.528 | n/a | 98.6% | 0.000 | library_gemm (29) | n/a |
| 337 | value.v3_matmul | library_gemm | library_gemm; gemmSN_NN_kernel; g1x2x1; b256x1x1; r64; s21504 | 60 | n/a | 4.768 | n/a | 91.9% | 0.000 | wide_qkv (27) | n/a |
| 338 | value.v3_bias | head_elementwise | head_elementwise; addCBiasInplaceNCKernel; g1x1x1; b3x170x1; r16; s0 | 60 | n/a | 1.280 | n/a | 78.3% | 0.000 | copy_reformat (26) | n/a |
| 339 | value.score_matmul | library_gemm | library_gemm; gemmSN_NN_kernel; g1x2x1; b256x1x1; r64; s21504 | 60 | n/a | 4.160 | n/a | 90.7% | 0.000 | head_elementwise (29) | n/a |
| 340 | value.score_bias | head_elementwise | head_elementwise; addCBiasInplaceNCKernel; g1x1x1; b6x85x1; r16; s0 | 60 | n/a | 1.152 | n/a | 87.0% | 0.000 | fa4 (27) | n/a |
| 341 | value.ownership_conv | library_gemm | library_gemm; Kernel2; g8x19x3; b128x1x1; r118; s33792 | 60 | n/a | 4.673 | n/a | 84.7% | 0.000 | library_gemm (32) | n/a |
| 342 | value.ownership_conv_splitk_reduce | library_gemm | library_gemm; splitKreduce_kernel; g147x1x1; b32x16x1; r49; s0 | 60 | n/a | 1.856 | n/a | 92.2% | 0.000 | library_gemm (45) | n/a |
| 343 | value.ownership_half_to_float | copy_reformat | copy_reformat; copyFromHalfKernel; g10x1x1; b512x1x1; r16; s0 | 60 | n/a | 1.152 | n/a | 91.0% | 0.000 | library_gemm (39) | n/a |
