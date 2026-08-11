# Nsys stream interference report

- Timed iterations: 20; streams: 48
- Kernels per forward: 48=344

## Kernel families

| family | instances | total ms | median us | p90 us | peer overlap | S2/S1 median | summed excess ms | matched |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| fused_ffn | 660 | 25.417 | 38.529 | 40.929 | 0.0% | n/a | 0.000 | 0 |
| linear2_residual | 660 | 13.733 | 20.832 | 21.120 | 0.0% | n/a | 0.000 | 0 |
| wide_qkv | 660 | 12.690 | 19.232 | 19.585 | 0.0% | n/a | 0.000 | 0 |
| library_gemm | 1380 | 12.634 | 8.512 | 14.304 | 0.0% | n/a | 0.000 | 0 |
| fa4 | 660 | 7.832 | 11.872 | 12.064 | 0.0% | n/a | 0.000 | 0 |
| rmsnorm | 1320 | 3.218 | 2.432 | 2.496 | 0.0% | n/a | 0.000 | 0 |
| qk_rope | 660 | 2.538 | 3.840 | 3.936 | 0.0% | n/a | 0.000 | 0 |
| affine_silu | 460 | 1.948 | 4.960 | 5.120 | 0.0% | n/a | 0.000 | 0 |
| head_elementwise | 240 | 0.597 | 1.984 | 4.512 | 0.0% | n/a | 0.000 | 0 |
| cudnn | 60 | 0.448 | 1.536 | 19.553 | 0.0% | n/a | 0.000 | 0 |
| copy_reformat | 100 | 0.142 | 1.504 | 2.176 | 0.0% | n/a | 0.000 | 0 |
| sumChannelsNCHWKernel | 20 | 0.034 | 1.664 | 1.696 | 0.0% | n/a | 0.000 | 0 |

## Dominant interference pairs

| running family | dominant peer | instances | median overlap | median us | S2/S1 median | matched |
|---|---|---:|---:|---:|---:|---:|
| library_gemm | idle | 1380 | 0.0% | 8.512 | n/a | 0 |
| rmsnorm | idle | 1320 | 0.0% | 2.432 | n/a | 0 |
| wide_qkv | idle | 660 | 0.0% | 19.232 | n/a | 0 |
| qk_rope | idle | 660 | 0.0% | 3.840 | n/a | 0 |
| fa4 | idle | 660 | 0.0% | 11.872 | n/a | 0 |
| fused_ffn | idle | 660 | 0.0% | 38.529 | n/a | 0 |
| linear2_residual | idle | 660 | 0.0% | 20.832 | n/a | 0 |
| affine_silu | idle | 460 | 0.0% | 4.960 | n/a | 0 |
| head_elementwise | idle | 240 | 0.0% | 1.984 | n/a | 0 |
| copy_reformat | idle | 100 | 0.0% | 1.504 | n/a | 0 |
| cudnn | idle | 60 | 0.0% | 1.536 | n/a | 0 |
| sumChannelsNCHWKernel | idle | 20 | 0.0% | 1.664 | n/a | 0 |

## Logical operation groups

Isolated reference total is the isolated median for each ordinal multiplied by its S2 call count; it is a normalized reference, not a second trace total.

| logical group | families | positions | calls | isolated reference ms | S2 total ms | total ratio | excess ms |
|---|---|---:|---:|---:|---:|---:|---:|
| transformer.ffn_linear1_gate_swiglu | fused_ffn | 33 | 660 | 0.000 | 25.417 | n/a | 0.000 |
| transformer.ffn_linear2_residual | linear2_residual | 33 | 660 | 0.000 | 13.733 | n/a | 0.000 |
| transformer.attention_qkv_projection | wide_qkv | 33 | 660 | 0.000 | 12.690 | n/a | 0.000 |
| transformer.attention_fa4 | fa4 | 33 | 660 | 0.000 | 7.832 | n/a | 0.000 |
| transformer.attention_out_projection_residual | library_gemm | 33 | 660 | 0.000 | 5.595 | n/a | 0.000 |
| outer.post_projection_c384_to_c768_residual | library_gemm | 11 | 220 | 0.000 | 3.164 | n/a | 0.000 |
| outer.pre_projection_c768_to_c384 | library_gemm | 11 | 220 | 0.000 | 2.619 | n/a | 0.000 |
| transformer.attention_qk_rope | qk_rope | 33 | 660 | 0.000 | 2.538 | n/a | 0.000 |
| transformer.ffn_rmsnorm | rmsnorm | 33 | 660 | 0.000 | 1.614 | n/a | 0.000 |
| transformer.attention_rmsnorm | rmsnorm | 33 | 660 | 0.000 | 1.604 | n/a | 0.000 |
| outer.pre_norm_silu | affine_silu | 11 | 220 | 0.000 | 1.111 | n/a | 0.000 |
| outer.post_norm_silu | affine_silu | 11 | 220 | 0.000 | 0.736 | n/a | 0.000 |
| frontend.initial_conv | cudnn | 1 | 20 | 0.000 | 0.391 | n/a | 0.000 |
| value.v2_matmul | library_gemm | 1 | 20 | 0.000 | 0.190 | n/a | 0.000 |
| value.v1_conv | library_gemm | 1 | 20 | 0.000 | 0.159 | n/a | 0.000 |
| frontend.initial_global_broadcast_add | head_elementwise | 1 | 20 | 0.000 | 0.155 | n/a | 0.000 |
| policy.p1_conv | library_gemm | 1 | 20 | 0.000 | 0.123 | n/a | 0.000 |
| policy.g1_conv | library_gemm | 1 | 20 | 0.000 | 0.118 | n/a | 0.000 |
| policy.gpool_to_bias_matmul | library_gemm | 1 | 20 | 0.000 | 0.109 | n/a | 0.000 |
| policy.gpool_to_pass_matmul | library_gemm | 1 | 20 | 0.000 | 0.106 | n/a | 0.000 |
| trunk.tip_norm_silu | affine_silu | 1 | 20 | 0.000 | 0.101 | n/a | 0.000 |
| policy.g1_global_pool | head_elementwise | 1 | 20 | 0.000 | 0.089 | n/a | 0.000 |
| value.ownership_conv | library_gemm | 1 | 20 | 0.000 | 0.081 | n/a | 0.000 |
| policy.p2_conv | library_gemm | 1 | 20 | 0.000 | 0.078 | n/a | 0.000 |
| value.score_matmul | library_gemm | 1 | 20 | 0.000 | 0.070 | n/a | 0.000 |
| value.v3_matmul | library_gemm | 1 | 20 | 0.000 | 0.070 | n/a | 0.000 |
| value.v1_global_pool | head_elementwise | 1 | 20 | 0.000 | 0.065 | n/a | 0.000 |
| value.v1_norm_silu | head_elementwise | 1 | 20 | 0.000 | 0.063 | n/a | 0.000 |
| frontend.initial_global_matmul | library_gemm | 1 | 20 | 0.000 | 0.053 | n/a | 0.000 |
| policy.gpool_to_pass_matmul2 | library_gemm | 1 | 20 | 0.000 | 0.047 | n/a | 0.000 |
| policy.p1_norm_silu | head_elementwise | 1 | 20 | 0.000 | 0.043 | n/a | 0.000 |
| value.v1_half_to_float | copy_reformat | 1 | 20 | 0.000 | 0.043 | n/a | 0.000 |
| policy.g1_norm_silu | head_elementwise | 1 | 20 | 0.000 | 0.042 | n/a | 0.000 |
| policy.gpool_bias_add | head_elementwise | 1 | 20 | 0.000 | 0.036 | n/a | 0.000 |
| input.mask_sum | sumChannelsNCHWKernel | 1 | 20 | 0.000 | 0.034 | n/a | 0.000 |
| frontend.initial_conv_nhwc_padding_1 | cudnn | 1 | 20 | 0.000 | 0.031 | n/a | 0.000 |
| policy.g1_half_to_float | copy_reformat | 1 | 20 | 0.000 | 0.031 | n/a | 0.000 |
| policy.p1_half_to_float | copy_reformat | 1 | 20 | 0.000 | 0.030 | n/a | 0.000 |
| value.ownership_conv_splitk_reduce | library_gemm | 1 | 20 | 0.000 | 0.028 | n/a | 0.000 |
| frontend.initial_conv_nhwc_padding_0 | cudnn | 1 | 20 | 0.000 | 0.026 | n/a | 0.000 |
| frontend.initial_global_matmul_splitk_reduce | library_gemm | 1 | 20 | 0.000 | 0.025 | n/a | 0.000 |
| input.extract_mask | head_elementwise | 1 | 20 | 0.000 | 0.024 | n/a | 0.000 |
| value.v2_bias_silu | head_elementwise | 1 | 20 | 0.000 | 0.021 | n/a | 0.000 |
| policy.pass_bias_silu | head_elementwise | 1 | 20 | 0.000 | 0.020 | n/a | 0.000 |
| value.v3_bias | head_elementwise | 1 | 20 | 0.000 | 0.019 | n/a | 0.000 |
| value.ownership_half_to_float | copy_reformat | 1 | 20 | 0.000 | 0.019 | n/a | 0.000 |
| value.score_bias | head_elementwise | 1 | 20 | 0.000 | 0.019 | n/a | 0.000 |
| input.mask_half_to_float | copy_reformat | 1 | 20 | 0.000 | 0.019 | n/a | 0.000 |

## `library_gemm` logical breakdown

| logical group | positions | calls | isolated reference ms | S2 total ms | total ratio | excess ms |
|---|---:|---:|---:|---:|---:|---:|
| transformer.attention_out_projection_residual | 33 | 660 | 0.000 | 5.595 | n/a | 0.000 |
| outer.post_projection_c384_to_c768_residual | 11 | 220 | 0.000 | 3.164 | n/a | 0.000 |
| outer.pre_projection_c768_to_c384 | 11 | 220 | 0.000 | 2.619 | n/a | 0.000 |
| value.v2_matmul | 1 | 20 | 0.000 | 0.190 | n/a | 0.000 |
| value.v1_conv | 1 | 20 | 0.000 | 0.159 | n/a | 0.000 |
| policy.p1_conv | 1 | 20 | 0.000 | 0.123 | n/a | 0.000 |
| policy.g1_conv | 1 | 20 | 0.000 | 0.118 | n/a | 0.000 |
| policy.gpool_to_bias_matmul | 1 | 20 | 0.000 | 0.109 | n/a | 0.000 |
| policy.gpool_to_pass_matmul | 1 | 20 | 0.000 | 0.106 | n/a | 0.000 |
| value.ownership_conv | 1 | 20 | 0.000 | 0.081 | n/a | 0.000 |
| policy.p2_conv | 1 | 20 | 0.000 | 0.078 | n/a | 0.000 |
| value.score_matmul | 1 | 20 | 0.000 | 0.070 | n/a | 0.000 |
| value.v3_matmul | 1 | 20 | 0.000 | 0.070 | n/a | 0.000 |
| frontend.initial_global_matmul | 1 | 20 | 0.000 | 0.053 | n/a | 0.000 |
| policy.gpool_to_pass_matmul2 | 1 | 20 | 0.000 | 0.047 | n/a | 0.000 |
| value.ownership_conv_splitk_reduce | 1 | 20 | 0.000 | 0.028 | n/a | 0.000 |
| frontend.initial_global_matmul_splitk_reduce | 1 | 20 | 0.000 | 0.025 | n/a | 0.000 |

## Top ordinal hotspots by summed excess

The worst peer is the highest median S2/S1 slowdown among peer families observed at least four times for that ordinal.

| rank | ordinal | logical position | family | calls | isolated us | S2 us | S2/S1 | excess ms | common peer | worst peer |
|---:|---:|---|---|---:|---:|---:|---:|---:|---|---|
| 1 | 157 | outer_05.transformer_0.block_15.ffn_linear1_gate_swiglu | fused_ffn | 20 | n/a | 40.977 | n/a | 0.000 | idle (20) | n/a |
| 2 | 249 | outer_08.transformer_1.block_25.ffn_linear1_gate_swiglu | fused_ffn | 20 | n/a | 40.721 | n/a | 0.000 | idle (20) | n/a |
| 3 | 117 | outer_03.transformer_2.block_11.ffn_linear1_gate_swiglu | fused_ffn | 20 | n/a | 40.544 | n/a | 0.000 | idle (20) | n/a |
| 4 | 313 | outer_10.transformer_2.block_32.ffn_linear1_gate_swiglu | fused_ffn | 20 | n/a | 40.465 | n/a | 0.000 | idle (20) | n/a |
| 5 | 185 | outer_06.transformer_0.block_18.ffn_linear1_gate_swiglu | fused_ffn | 20 | n/a | 40.368 | n/a | 0.000 | idle (20) | n/a |
| 6 | 241 | outer_08.transformer_0.block_24.ffn_linear1_gate_swiglu | fused_ffn | 20 | n/a | 40.401 | n/a | 0.000 | idle (20) | n/a |
| 7 | 257 | outer_08.transformer_2.block_26.ffn_linear1_gate_swiglu | fused_ffn | 20 | n/a | 40.289 | n/a | 0.000 | idle (20) | n/a |
| 8 | 81 | outer_02.transformer_1.block_07.ffn_linear1_gate_swiglu | fused_ffn | 20 | n/a | 40.992 | n/a | 0.000 | idle (20) | n/a |
| 9 | 73 | outer_02.transformer_0.block_06.ffn_linear1_gate_swiglu | fused_ffn | 20 | n/a | 40.544 | n/a | 0.000 | idle (20) | n/a |
| 10 | 109 | outer_03.transformer_1.block_10.ffn_linear1_gate_swiglu | fused_ffn | 20 | n/a | 40.193 | n/a | 0.000 | idle (20) | n/a |
| 11 | 165 | outer_05.transformer_1.block_16.ffn_linear1_gate_swiglu | fused_ffn | 20 | n/a | 39.553 | n/a | 0.000 | idle (20) | n/a |
| 12 | 45 | outer_01.transformer_0.block_03.ffn_linear1_gate_swiglu | fused_ffn | 20 | n/a | 39.392 | n/a | 0.000 | idle (20) | n/a |
| 13 | 17 | outer_00.transformer_0.block_00.ffn_linear1_gate_swiglu | fused_ffn | 20 | n/a | 39.344 | n/a | 0.000 | idle (20) | n/a |
| 14 | 285 | outer_09.transformer_2.block_29.ffn_linear1_gate_swiglu | fused_ffn | 20 | n/a | 39.889 | n/a | 0.000 | idle (20) | n/a |
| 15 | 193 | outer_06.transformer_1.block_19.ffn_linear1_gate_swiglu | fused_ffn | 20 | n/a | 39.120 | n/a | 0.000 | idle (20) | n/a |
| 16 | 269 | outer_09.transformer_0.block_27.ffn_linear1_gate_swiglu | fused_ffn | 20 | n/a | 38.737 | n/a | 0.000 | idle (20) | n/a |
| 17 | 173 | outer_05.transformer_2.block_17.ffn_linear1_gate_swiglu | fused_ffn | 20 | n/a | 38.672 | n/a | 0.000 | idle (20) | n/a |
| 18 | 53 | outer_01.transformer_1.block_04.ffn_linear1_gate_swiglu | fused_ffn | 20 | n/a | 38.528 | n/a | 0.000 | idle (20) | n/a |
| 19 | 33 | outer_00.transformer_2.block_02.ffn_linear1_gate_swiglu | fused_ffn | 20 | n/a | 38.273 | n/a | 0.000 | idle (20) | n/a |
| 20 | 61 | outer_01.transformer_2.block_05.ffn_linear1_gate_swiglu | fused_ffn | 20 | n/a | 38.432 | n/a | 0.000 | idle (20) | n/a |
| 21 | 277 | outer_09.transformer_1.block_28.ffn_linear1_gate_swiglu | fused_ffn | 20 | n/a | 37.937 | n/a | 0.000 | idle (20) | n/a |
| 22 | 137 | outer_04.transformer_1.block_13.ffn_linear1_gate_swiglu | fused_ffn | 20 | n/a | 38.192 | n/a | 0.000 | idle (20) | n/a |
| 23 | 129 | outer_04.transformer_0.block_12.ffn_linear1_gate_swiglu | fused_ffn | 20 | n/a | 38.305 | n/a | 0.000 | idle (20) | n/a |
| 24 | 101 | outer_03.transformer_0.block_09.ffn_linear1_gate_swiglu | fused_ffn | 20 | n/a | 37.984 | n/a | 0.000 | idle (20) | n/a |
| 25 | 25 | outer_00.transformer_1.block_01.ffn_linear1_gate_swiglu | fused_ffn | 20 | n/a | 37.984 | n/a | 0.000 | idle (20) | n/a |
| 26 | 89 | outer_02.transformer_2.block_08.ffn_linear1_gate_swiglu | fused_ffn | 20 | n/a | 37.681 | n/a | 0.000 | idle (20) | n/a |
| 27 | 305 | outer_10.transformer_1.block_31.ffn_linear1_gate_swiglu | fused_ffn | 20 | n/a | 37.457 | n/a | 0.000 | idle (20) | n/a |
| 28 | 221 | outer_07.transformer_1.block_22.ffn_linear1_gate_swiglu | fused_ffn | 20 | n/a | 35.809 | n/a | 0.000 | idle (20) | n/a |
| 29 | 297 | outer_10.transformer_0.block_30.ffn_linear1_gate_swiglu | fused_ffn | 20 | n/a | 35.920 | n/a | 0.000 | idle (20) | n/a |
| 30 | 229 | outer_07.transformer_2.block_23.ffn_linear1_gate_swiglu | fused_ffn | 20 | n/a | 35.585 | n/a | 0.000 | idle (20) | n/a |
| 31 | 213 | outer_07.transformer_0.block_21.ffn_linear1_gate_swiglu | fused_ffn | 20 | n/a | 35.408 | n/a | 0.000 | idle (20) | n/a |
| 32 | 145 | outer_04.transformer_2.block_14.ffn_linear1_gate_swiglu | fused_ffn | 20 | n/a | 34.688 | n/a | 0.000 | idle (20) | n/a |
| 33 | 201 | outer_06.transformer_2.block_20.ffn_linear1_gate_swiglu | fused_ffn | 20 | n/a | 34.641 | n/a | 0.000 | idle (20) | n/a |
| 34 | 214 | outer_07.transformer_0.block_21.ffn_linear2_residual | linear2_residual | 20 | n/a | 21.056 | n/a | 0.000 | idle (20) | n/a |
| 35 | 230 | outer_07.transformer_2.block_23.ffn_linear2_residual | linear2_residual | 20 | n/a | 21.072 | n/a | 0.000 | idle (20) | n/a |
| 36 | 242 | outer_08.transformer_0.block_24.ffn_linear2_residual | linear2_residual | 20 | n/a | 21.041 | n/a | 0.000 | idle (20) | n/a |
| 37 | 222 | outer_07.transformer_1.block_22.ffn_linear2_residual | linear2_residual | 20 | n/a | 21.024 | n/a | 0.000 | idle (20) | n/a |
| 38 | 202 | outer_06.transformer_2.block_20.ffn_linear2_residual | linear2_residual | 20 | n/a | 20.960 | n/a | 0.000 | idle (20) | n/a |
| 39 | 18 | outer_00.transformer_0.block_00.ffn_linear2_residual | linear2_residual | 20 | n/a | 20.960 | n/a | 0.000 | idle (20) | n/a |
| 40 | 250 | outer_08.transformer_1.block_25.ffn_linear2_residual | linear2_residual | 20 | n/a | 20.960 | n/a | 0.000 | idle (20) | n/a |

## Full fixed-forward ordinal map

| ordinal | logical position | family | resource signature | calls | isolated us | S2 us | S2/S1 | overlap | excess ms | common peer | worst peer |
|---:|---|---|---|---:|---:|---:|---:|---:|---:|---|---|
| 0 | input.extract_mask | head_elementwise | head_elementwise; extractChannel0KernelNHWC; g10x1x1; b512x1x1; r16; s0 | 20 | n/a | 1.184 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 1 | input.mask_half_to_float | copy_reformat | copy_reformat; copyFromHalfKernel; g10x1x1; b512x1x1; r16; s0 | 20 | n/a | 0.928 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 2 | input.mask_sum | sumChannelsNCHWKernel | sumChannelsNCHWKernel; sumChannelsNCHWKernel; g1x1x13; b256x2x1; r22; s2048 | 20 | n/a | 1.664 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 3 | frontend.initial_conv_nhwc_padding_0 | cudnn | cudnn; nhwcAddPaddingKernel; g340x1x1; b768x1x1; r34; s0 | 20 | n/a | 1.280 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 4 | frontend.initial_conv_nhwc_padding_1 | cudnn | cudnn; nhwcAddPaddingKernel; g340x1x1; b768x1x1; r34; s0 | 20 | n/a | 1.536 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 5 | frontend.initial_conv | cudnn | cudnn; Kernel; g296x3x1; b128x1x1; r94; s81920 | 20 | n/a | 19.552 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 6 | frontend.initial_global_matmul | library_gemm | library_gemm; Kernel2; g8x1x3; b128x1x1; r128; s24576 | 20 | n/a | 2.624 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 7 | frontend.initial_global_matmul_splitk_reduce | library_gemm | library_gemm; splitKreduce_kernel; g24x1x1; b32x16x1; r49; s0 | 20 | n/a | 1.280 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 8 | frontend.initial_global_broadcast_add | head_elementwise | head_elementwise; addNCBiasInplaceNHWCHalfKernel; g3x361x13; b256x1x1; r16; s0 | 20 | n/a | 7.713 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 9 | outer_00.pre_norm_silu | affine_silu | affine_silu; affineSiluHalf2Kernel; g4693x1x1; b384x1x1; r17; s0 | 20 | n/a | 4.960 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 10 | outer_00.pre_projection_c768_to_c384 | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 20 | n/a | 11.648 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 11 | outer_00.transformer_0.block_00.attention_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 20 | n/a | 2.368 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 12 | outer_00.transformer_0.block_00.attention_qkv_projection | wide_qkv | wide_qkv; wide_qkv_kernel; g9x37x1; b128x1x1; r136; s65536 | 20 | n/a | 18.849 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 13 | outer_00.transformer_0.block_00.attention_qk_rope | qk_rope | qk_rope; fusedQKRoPE19Half2Kernel; g361x13x1; b192x1x1; r16; s0 | 20 | n/a | 3.808 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 14 | outer_00.transformer_0.block_00.attention_fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 20 | n/a | 11.712 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 15 | outer_00.transformer_0.block_00.attention_out_projection_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 20 | n/a | 8.449 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 16 | outer_00.transformer_0.block_00.ffn_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 20 | n/a | 2.432 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 17 | outer_00.transformer_0.block_00.ffn_linear1_gate_swiglu | fused_ffn | fused_ffn; fused_ffn_candidate_a_reuse_kernel; g18x37x1; b128x1x1; r136; s32768 | 20 | n/a | 39.344 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 18 | outer_00.transformer_0.block_00.ffn_linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 20 | n/a | 20.960 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 19 | outer_00.transformer_1.block_01.attention_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 20 | n/a | 2.464 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 20 | outer_00.transformer_1.block_01.attention_qkv_projection | wide_qkv | wide_qkv; wide_qkv_kernel; g9x37x1; b128x1x1; r136; s65536 | 20 | n/a | 19.424 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 21 | outer_00.transformer_1.block_01.attention_qk_rope | qk_rope | qk_rope; fusedQKRoPE19Half2Kernel; g361x13x1; b192x1x1; r16; s0 | 20 | n/a | 3.872 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 22 | outer_00.transformer_1.block_01.attention_fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 20 | n/a | 11.968 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 23 | outer_00.transformer_1.block_01.attention_out_projection_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 20 | n/a | 8.496 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 24 | outer_00.transformer_1.block_01.ffn_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 20 | n/a | 2.464 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 25 | outer_00.transformer_1.block_01.ffn_linear1_gate_swiglu | fused_ffn | fused_ffn; fused_ffn_candidate_a_reuse_kernel; g18x37x1; b128x1x1; r136; s32768 | 20 | n/a | 37.984 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 26 | outer_00.transformer_1.block_01.ffn_linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 20 | n/a | 20.848 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 27 | outer_00.transformer_2.block_02.attention_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 20 | n/a | 2.400 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 28 | outer_00.transformer_2.block_02.attention_qkv_projection | wide_qkv | wide_qkv; wide_qkv_kernel; g9x37x1; b128x1x1; r136; s65536 | 20 | n/a | 19.248 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 29 | outer_00.transformer_2.block_02.attention_qk_rope | qk_rope | qk_rope; fusedQKRoPE19Half2Kernel; g361x13x1; b192x1x1; r16; s0 | 20 | n/a | 3.840 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 30 | outer_00.transformer_2.block_02.attention_fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 20 | n/a | 11.872 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 31 | outer_00.transformer_2.block_02.attention_out_projection_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 20 | n/a | 8.480 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 32 | outer_00.transformer_2.block_02.ffn_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 20 | n/a | 2.464 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 33 | outer_00.transformer_2.block_02.ffn_linear1_gate_swiglu | fused_ffn | fused_ffn; fused_ffn_candidate_a_reuse_kernel; g18x37x1; b128x1x1; r136; s32768 | 20 | n/a | 38.273 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 34 | outer_00.transformer_2.block_02.ffn_linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 20 | n/a | 20.720 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 35 | outer_00.post_norm_silu | affine_silu | affine_silu; affineSiluHalf2Kernel; g4693x1x1; b192x1x1; r17; s0 | 20 | n/a | 3.328 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 36 | outer_00.post_projection_c384_to_c768_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b256x1x1; r154; s73728 | 20 | n/a | 14.336 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 37 | outer_01.pre_norm_silu | affine_silu | affine_silu; affineSiluHalf2Kernel; g4693x1x1; b384x1x1; r17; s0 | 20 | n/a | 5.088 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 38 | outer_01.pre_projection_c768_to_c384 | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 20 | n/a | 11.904 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 39 | outer_01.transformer_0.block_03.attention_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 20 | n/a | 2.432 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 40 | outer_01.transformer_0.block_03.attention_qkv_projection | wide_qkv | wide_qkv; wide_qkv_kernel; g9x37x1; b128x1x1; r136; s65536 | 20 | n/a | 19.056 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 41 | outer_01.transformer_0.block_03.attention_qk_rope | qk_rope | qk_rope; fusedQKRoPE19Half2Kernel; g361x13x1; b192x1x1; r16; s0 | 20 | n/a | 3.808 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 42 | outer_01.transformer_0.block_03.attention_fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 20 | n/a | 11.840 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 43 | outer_01.transformer_0.block_03.attention_out_projection_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 20 | n/a | 8.448 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 44 | outer_01.transformer_0.block_03.ffn_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 20 | n/a | 2.416 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 45 | outer_01.transformer_0.block_03.ffn_linear1_gate_swiglu | fused_ffn | fused_ffn; fused_ffn_candidate_a_reuse_kernel; g18x37x1; b128x1x1; r136; s32768 | 20 | n/a | 39.392 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 46 | outer_01.transformer_0.block_03.ffn_linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 20 | n/a | 20.864 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 47 | outer_01.transformer_1.block_04.attention_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 20 | n/a | 2.416 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 48 | outer_01.transformer_1.block_04.attention_qkv_projection | wide_qkv | wide_qkv; wide_qkv_kernel; g9x37x1; b128x1x1; r136; s65536 | 20 | n/a | 19.313 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 49 | outer_01.transformer_1.block_04.attention_qk_rope | qk_rope | qk_rope; fusedQKRoPE19Half2Kernel; g361x13x1; b192x1x1; r16; s0 | 20 | n/a | 3.840 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 50 | outer_01.transformer_1.block_04.attention_fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 20 | n/a | 11.888 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 51 | outer_01.transformer_1.block_04.attention_out_projection_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 20 | n/a | 8.496 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 52 | outer_01.transformer_1.block_04.ffn_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 20 | n/a | 2.448 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 53 | outer_01.transformer_1.block_04.ffn_linear1_gate_swiglu | fused_ffn | fused_ffn; fused_ffn_candidate_a_reuse_kernel; g18x37x1; b128x1x1; r136; s32768 | 20 | n/a | 38.528 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 54 | outer_01.transformer_1.block_04.ffn_linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 20 | n/a | 20.753 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 55 | outer_01.transformer_2.block_05.attention_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 20 | n/a | 2.432 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 56 | outer_01.transformer_2.block_05.attention_qkv_projection | wide_qkv | wide_qkv; wide_qkv_kernel; g9x37x1; b128x1x1; r136; s65536 | 20 | n/a | 19.152 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 57 | outer_01.transformer_2.block_05.attention_qk_rope | qk_rope | qk_rope; fusedQKRoPE19Half2Kernel; g361x13x1; b192x1x1; r16; s0 | 20 | n/a | 3.840 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 58 | outer_01.transformer_2.block_05.attention_fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 20 | n/a | 11.856 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 59 | outer_01.transformer_2.block_05.attention_out_projection_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 20 | n/a | 8.432 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 60 | outer_01.transformer_2.block_05.ffn_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 20 | n/a | 2.448 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 61 | outer_01.transformer_2.block_05.ffn_linear1_gate_swiglu | fused_ffn | fused_ffn; fused_ffn_candidate_a_reuse_kernel; g18x37x1; b128x1x1; r136; s32768 | 20 | n/a | 38.432 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 62 | outer_01.transformer_2.block_05.ffn_linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 20 | n/a | 20.784 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 63 | outer_01.post_norm_silu | affine_silu | affine_silu; affineSiluHalf2Kernel; g4693x1x1; b192x1x1; r17; s0 | 20 | n/a | 3.328 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 64 | outer_01.post_projection_c384_to_c768_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b256x1x1; r154; s73728 | 20 | n/a | 14.336 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 65 | outer_02.pre_norm_silu | affine_silu | affine_silu; affineSiluHalf2Kernel; g4693x1x1; b384x1x1; r17; s0 | 20 | n/a | 5.024 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 66 | outer_02.pre_projection_c768_to_c384 | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 20 | n/a | 11.920 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 67 | outer_02.transformer_0.block_06.attention_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 20 | n/a | 2.432 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 68 | outer_02.transformer_0.block_06.attention_qkv_projection | wide_qkv | wide_qkv; wide_qkv_kernel; g9x37x1; b128x1x1; r136; s65536 | 20 | n/a | 19.152 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 69 | outer_02.transformer_0.block_06.attention_qk_rope | qk_rope | qk_rope; fusedQKRoPE19Half2Kernel; g361x13x1; b192x1x1; r16; s0 | 20 | n/a | 3.856 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 70 | outer_02.transformer_0.block_06.attention_fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 20 | n/a | 11.840 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 71 | outer_02.transformer_0.block_06.attention_out_projection_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 20 | n/a | 8.480 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 72 | outer_02.transformer_0.block_06.ffn_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 20 | n/a | 2.432 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 73 | outer_02.transformer_0.block_06.ffn_linear1_gate_swiglu | fused_ffn | fused_ffn; fused_ffn_candidate_a_reuse_kernel; g18x37x1; b128x1x1; r136; s32768 | 20 | n/a | 40.544 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 74 | outer_02.transformer_0.block_06.ffn_linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 20 | n/a | 20.800 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 75 | outer_02.transformer_1.block_07.attention_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 20 | n/a | 2.432 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 76 | outer_02.transformer_1.block_07.attention_qkv_projection | wide_qkv | wide_qkv; wide_qkv_kernel; g9x37x1; b128x1x1; r136; s65536 | 20 | n/a | 19.280 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 77 | outer_02.transformer_1.block_07.attention_qk_rope | qk_rope | qk_rope; fusedQKRoPE19Half2Kernel; g361x13x1; b192x1x1; r16; s0 | 20 | n/a | 3.824 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 78 | outer_02.transformer_1.block_07.attention_fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 20 | n/a | 11.840 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 79 | outer_02.transformer_1.block_07.attention_out_projection_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 20 | n/a | 8.448 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 80 | outer_02.transformer_1.block_07.ffn_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 20 | n/a | 2.464 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 81 | outer_02.transformer_1.block_07.ffn_linear1_gate_swiglu | fused_ffn | fused_ffn; fused_ffn_candidate_a_reuse_kernel; g18x37x1; b128x1x1; r136; s32768 | 20 | n/a | 40.992 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 82 | outer_02.transformer_1.block_07.ffn_linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 20 | n/a | 20.800 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 83 | outer_02.transformer_2.block_08.attention_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 20 | n/a | 2.448 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 84 | outer_02.transformer_2.block_08.attention_qkv_projection | wide_qkv | wide_qkv; wide_qkv_kernel; g9x37x1; b128x1x1; r136; s65536 | 20 | n/a | 19.168 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 85 | outer_02.transformer_2.block_08.attention_qk_rope | qk_rope | qk_rope; fusedQKRoPE19Half2Kernel; g361x13x1; b192x1x1; r16; s0 | 20 | n/a | 3.840 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 86 | outer_02.transformer_2.block_08.attention_fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 20 | n/a | 11.840 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 87 | outer_02.transformer_2.block_08.attention_out_projection_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 20 | n/a | 8.528 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 88 | outer_02.transformer_2.block_08.ffn_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 20 | n/a | 2.448 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 89 | outer_02.transformer_2.block_08.ffn_linear1_gate_swiglu | fused_ffn | fused_ffn; fused_ffn_candidate_a_reuse_kernel; g18x37x1; b128x1x1; r136; s32768 | 20 | n/a | 37.681 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 90 | outer_02.transformer_2.block_08.ffn_linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 20 | n/a | 20.880 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 91 | outer_02.post_norm_silu | affine_silu | affine_silu; affineSiluHalf2Kernel; g4693x1x1; b192x1x1; r17; s0 | 20 | n/a | 3.360 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 92 | outer_02.post_projection_c384_to_c768_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b256x1x1; r154; s73728 | 20 | n/a | 14.416 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 93 | outer_03.pre_norm_silu | affine_silu | affine_silu; affineSiluHalf2Kernel; g4693x1x1; b384x1x1; r17; s0 | 20 | n/a | 5.088 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 94 | outer_03.pre_projection_c768_to_c384 | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 20 | n/a | 11.921 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 95 | outer_03.transformer_0.block_09.attention_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 20 | n/a | 2.432 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 96 | outer_03.transformer_0.block_09.attention_qkv_projection | wide_qkv | wide_qkv; wide_qkv_kernel; g9x37x1; b128x1x1; r136; s65536 | 20 | n/a | 19.280 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 97 | outer_03.transformer_0.block_09.attention_qk_rope | qk_rope | qk_rope; fusedQKRoPE19Half2Kernel; g361x13x1; b192x1x1; r16; s0 | 20 | n/a | 3.840 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 98 | outer_03.transformer_0.block_09.attention_fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 20 | n/a | 11.920 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 99 | outer_03.transformer_0.block_09.attention_out_projection_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 20 | n/a | 8.496 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 100 | outer_03.transformer_0.block_09.ffn_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 20 | n/a | 2.432 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 101 | outer_03.transformer_0.block_09.ffn_linear1_gate_swiglu | fused_ffn | fused_ffn; fused_ffn_candidate_a_reuse_kernel; g18x37x1; b128x1x1; r136; s32768 | 20 | n/a | 37.984 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 102 | outer_03.transformer_0.block_09.ffn_linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 20 | n/a | 20.688 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 103 | outer_03.transformer_1.block_10.attention_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 20 | n/a | 2.432 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 104 | outer_03.transformer_1.block_10.attention_qkv_projection | wide_qkv | wide_qkv; wide_qkv_kernel; g9x37x1; b128x1x1; r136; s65536 | 20 | n/a | 19.072 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 105 | outer_03.transformer_1.block_10.attention_qk_rope | qk_rope | qk_rope; fusedQKRoPE19Half2Kernel; g361x13x1; b192x1x1; r16; s0 | 20 | n/a | 3.824 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 106 | outer_03.transformer_1.block_10.attention_fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 20 | n/a | 11.808 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 107 | outer_03.transformer_1.block_10.attention_out_projection_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 20 | n/a | 8.416 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 108 | outer_03.transformer_1.block_10.ffn_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 20 | n/a | 2.432 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 109 | outer_03.transformer_1.block_10.ffn_linear1_gate_swiglu | fused_ffn | fused_ffn; fused_ffn_candidate_a_reuse_kernel; g18x37x1; b128x1x1; r136; s32768 | 20 | n/a | 40.193 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 110 | outer_03.transformer_1.block_10.ffn_linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 20 | n/a | 20.640 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 111 | outer_03.transformer_2.block_11.attention_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 20 | n/a | 2.368 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 112 | outer_03.transformer_2.block_11.attention_qkv_projection | wide_qkv | wide_qkv; wide_qkv_kernel; g9x37x1; b128x1x1; r136; s65536 | 20 | n/a | 19.041 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 113 | outer_03.transformer_2.block_11.attention_qk_rope | qk_rope | qk_rope; fusedQKRoPE19Half2Kernel; g361x13x1; b192x1x1; r16; s0 | 20 | n/a | 3.840 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 114 | outer_03.transformer_2.block_11.attention_fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 20 | n/a | 11.760 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 115 | outer_03.transformer_2.block_11.attention_out_projection_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 20 | n/a | 8.400 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 116 | outer_03.transformer_2.block_11.ffn_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 20 | n/a | 2.448 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 117 | outer_03.transformer_2.block_11.ffn_linear1_gate_swiglu | fused_ffn | fused_ffn; fused_ffn_candidate_a_reuse_kernel; g18x37x1; b128x1x1; r136; s32768 | 20 | n/a | 40.544 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 118 | outer_03.transformer_2.block_11.ffn_linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 20 | n/a | 20.480 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 119 | outer_03.post_norm_silu | affine_silu | affine_silu; affineSiluHalf2Kernel; g4693x1x1; b192x1x1; r17; s0 | 20 | n/a | 3.296 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 120 | outer_03.post_projection_c384_to_c768_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b256x1x1; r154; s73728 | 20 | n/a | 14.176 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 121 | outer_04.pre_norm_silu | affine_silu | affine_silu; affineSiluHalf2Kernel; g4693x1x1; b384x1x1; r17; s0 | 20 | n/a | 4.992 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 122 | outer_04.pre_projection_c768_to_c384 | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 20 | n/a | 11.760 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 123 | outer_04.transformer_0.block_12.attention_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 20 | n/a | 2.432 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 124 | outer_04.transformer_0.block_12.attention_qkv_projection | wide_qkv | wide_qkv; wide_qkv_kernel; g9x37x1; b128x1x1; r136; s65536 | 20 | n/a | 18.912 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 125 | outer_04.transformer_0.block_12.attention_qk_rope | qk_rope | qk_rope; fusedQKRoPE19Half2Kernel; g361x13x1; b192x1x1; r16; s0 | 20 | n/a | 3.824 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 126 | outer_04.transformer_0.block_12.attention_fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 20 | n/a | 11.745 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 127 | outer_04.transformer_0.block_12.attention_out_projection_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 20 | n/a | 8.448 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 128 | outer_04.transformer_0.block_12.ffn_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 20 | n/a | 2.432 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 129 | outer_04.transformer_0.block_12.ffn_linear1_gate_swiglu | fused_ffn | fused_ffn; fused_ffn_candidate_a_reuse_kernel; g18x37x1; b128x1x1; r136; s32768 | 20 | n/a | 38.305 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 130 | outer_04.transformer_0.block_12.ffn_linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 20 | n/a | 20.768 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 131 | outer_04.transformer_1.block_13.attention_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 20 | n/a | 2.400 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 132 | outer_04.transformer_1.block_13.attention_qkv_projection | wide_qkv | wide_qkv; wide_qkv_kernel; g9x37x1; b128x1x1; r136; s65536 | 20 | n/a | 19.296 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 133 | outer_04.transformer_1.block_13.attention_qk_rope | qk_rope | qk_rope; fusedQKRoPE19Half2Kernel; g361x13x1; b192x1x1; r16; s0 | 20 | n/a | 3.808 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 134 | outer_04.transformer_1.block_13.attention_fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 20 | n/a | 11.889 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 135 | outer_04.transformer_1.block_13.attention_out_projection_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 20 | n/a | 8.480 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 136 | outer_04.transformer_1.block_13.ffn_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 20 | n/a | 2.432 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 137 | outer_04.transformer_1.block_13.ffn_linear1_gate_swiglu | fused_ffn | fused_ffn; fused_ffn_candidate_a_reuse_kernel; g18x37x1; b128x1x1; r136; s32768 | 20 | n/a | 38.192 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 138 | outer_04.transformer_1.block_13.ffn_linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 20 | n/a | 20.913 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 139 | outer_04.transformer_2.block_14.attention_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 20 | n/a | 2.432 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 140 | outer_04.transformer_2.block_14.attention_qkv_projection | wide_qkv | wide_qkv; wide_qkv_kernel; g9x37x1; b128x1x1; r136; s65536 | 20 | n/a | 19.328 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 141 | outer_04.transformer_2.block_14.attention_qk_rope | qk_rope | qk_rope; fusedQKRoPE19Half2Kernel; g361x13x1; b192x1x1; r16; s0 | 20 | n/a | 3.872 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 142 | outer_04.transformer_2.block_14.attention_fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 20 | n/a | 11.904 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 143 | outer_04.transformer_2.block_14.attention_out_projection_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 20 | n/a | 8.448 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 144 | outer_04.transformer_2.block_14.ffn_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 20 | n/a | 2.432 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 145 | outer_04.transformer_2.block_14.ffn_linear1_gate_swiglu | fused_ffn | fused_ffn; fused_ffn_candidate_a_reuse_kernel; g18x37x1; b128x1x1; r136; s32768 | 20 | n/a | 34.688 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 146 | outer_04.transformer_2.block_14.ffn_linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 20 | n/a | 20.880 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 147 | outer_04.post_norm_silu | affine_silu | affine_silu; affineSiluHalf2Kernel; g4693x1x1; b192x1x1; r17; s0 | 20 | n/a | 3.345 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 148 | outer_04.post_projection_c384_to_c768_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b256x1x1; r154; s73728 | 20 | n/a | 14.528 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 149 | outer_05.pre_norm_silu | affine_silu | affine_silu; affineSiluHalf2Kernel; g4693x1x1; b384x1x1; r17; s0 | 20 | n/a | 5.088 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 150 | outer_05.pre_projection_c768_to_c384 | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 20 | n/a | 12.081 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 151 | outer_05.transformer_0.block_15.attention_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 20 | n/a | 2.464 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 152 | outer_05.transformer_0.block_15.attention_qkv_projection | wide_qkv | wide_qkv; wide_qkv_kernel; g9x37x1; b128x1x1; r136; s65536 | 20 | n/a | 19.377 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 153 | outer_05.transformer_0.block_15.attention_qk_rope | qk_rope | qk_rope; fusedQKRoPE19Half2Kernel; g361x13x1; b192x1x1; r16; s0 | 20 | n/a | 3.888 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 154 | outer_05.transformer_0.block_15.attention_fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 20 | n/a | 11.968 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 155 | outer_05.transformer_0.block_15.attention_out_projection_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 20 | n/a | 8.528 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 156 | outer_05.transformer_0.block_15.ffn_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 20 | n/a | 2.432 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 157 | outer_05.transformer_0.block_15.ffn_linear1_gate_swiglu | fused_ffn | fused_ffn; fused_ffn_candidate_a_reuse_kernel; g18x37x1; b128x1x1; r136; s32768 | 20 | n/a | 40.977 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 158 | outer_05.transformer_0.block_15.ffn_linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 20 | n/a | 20.944 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 159 | outer_05.transformer_1.block_16.attention_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 20 | n/a | 2.400 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 160 | outer_05.transformer_1.block_16.attention_qkv_projection | wide_qkv | wide_qkv; wide_qkv_kernel; g9x37x1; b128x1x1; r136; s65536 | 20 | n/a | 19.312 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 161 | outer_05.transformer_1.block_16.attention_qk_rope | qk_rope | qk_rope; fusedQKRoPE19Half2Kernel; g361x13x1; b192x1x1; r16; s0 | 20 | n/a | 3.840 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 162 | outer_05.transformer_1.block_16.attention_fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 20 | n/a | 11.904 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 163 | outer_05.transformer_1.block_16.attention_out_projection_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 20 | n/a | 8.464 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 164 | outer_05.transformer_1.block_16.ffn_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 20 | n/a | 2.432 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 165 | outer_05.transformer_1.block_16.ffn_linear1_gate_swiglu | fused_ffn | fused_ffn; fused_ffn_candidate_a_reuse_kernel; g18x37x1; b128x1x1; r136; s32768 | 20 | n/a | 39.553 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 166 | outer_05.transformer_1.block_16.ffn_linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 20 | n/a | 20.736 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 167 | outer_05.transformer_2.block_17.attention_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 20 | n/a | 2.400 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 168 | outer_05.transformer_2.block_17.attention_qkv_projection | wide_qkv | wide_qkv; wide_qkv_kernel; g9x37x1; b128x1x1; r136; s65536 | 20 | n/a | 19.184 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 169 | outer_05.transformer_2.block_17.attention_qk_rope | qk_rope | qk_rope; fusedQKRoPE19Half2Kernel; g361x13x1; b192x1x1; r16; s0 | 20 | n/a | 3.824 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 170 | outer_05.transformer_2.block_17.attention_fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 20 | n/a | 11.872 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 171 | outer_05.transformer_2.block_17.attention_out_projection_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 20 | n/a | 8.480 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 172 | outer_05.transformer_2.block_17.ffn_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 20 | n/a | 2.400 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 173 | outer_05.transformer_2.block_17.ffn_linear1_gate_swiglu | fused_ffn | fused_ffn; fused_ffn_candidate_a_reuse_kernel; g18x37x1; b128x1x1; r136; s32768 | 20 | n/a | 38.672 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 174 | outer_05.transformer_2.block_17.ffn_linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 20 | n/a | 20.688 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 175 | outer_05.post_norm_silu | affine_silu | affine_silu; affineSiluHalf2Kernel; g4693x1x1; b192x1x1; r17; s0 | 20 | n/a | 3.328 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 176 | outer_05.post_projection_c384_to_c768_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b256x1x1; r154; s73728 | 20 | n/a | 14.256 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 177 | outer_06.pre_norm_silu | affine_silu | affine_silu; affineSiluHalf2Kernel; g4693x1x1; b384x1x1; r17; s0 | 20 | n/a | 5.040 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 178 | outer_06.pre_projection_c768_to_c384 | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 20 | n/a | 11.905 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 179 | outer_06.transformer_0.block_18.attention_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 20 | n/a | 2.432 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 180 | outer_06.transformer_0.block_18.attention_qkv_projection | wide_qkv | wide_qkv; wide_qkv_kernel; g9x37x1; b128x1x1; r136; s65536 | 20 | n/a | 19.153 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 181 | outer_06.transformer_0.block_18.attention_qk_rope | qk_rope | qk_rope; fusedQKRoPE19Half2Kernel; g361x13x1; b192x1x1; r16; s0 | 20 | n/a | 3.856 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 182 | outer_06.transformer_0.block_18.attention_fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 20 | n/a | 11.840 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 183 | outer_06.transformer_0.block_18.attention_out_projection_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 20 | n/a | 8.432 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 184 | outer_06.transformer_0.block_18.ffn_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 20 | n/a | 2.480 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 185 | outer_06.transformer_0.block_18.ffn_linear1_gate_swiglu | fused_ffn | fused_ffn; fused_ffn_candidate_a_reuse_kernel; g18x37x1; b128x1x1; r136; s32768 | 20 | n/a | 40.368 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 186 | outer_06.transformer_0.block_18.ffn_linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 20 | n/a | 20.832 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 187 | outer_06.transformer_1.block_19.attention_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 20 | n/a | 2.432 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 188 | outer_06.transformer_1.block_19.attention_qkv_projection | wide_qkv | wide_qkv; wide_qkv_kernel; g9x37x1; b128x1x1; r136; s65536 | 20 | n/a | 19.297 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 189 | outer_06.transformer_1.block_19.attention_qk_rope | qk_rope | qk_rope; fusedQKRoPE19Half2Kernel; g361x13x1; b192x1x1; r16; s0 | 20 | n/a | 3.840 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 190 | outer_06.transformer_1.block_19.attention_fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 20 | n/a | 11.872 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 191 | outer_06.transformer_1.block_19.attention_out_projection_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 20 | n/a | 8.480 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 192 | outer_06.transformer_1.block_19.ffn_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 20 | n/a | 2.432 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 193 | outer_06.transformer_1.block_19.ffn_linear1_gate_swiglu | fused_ffn | fused_ffn; fused_ffn_candidate_a_reuse_kernel; g18x37x1; b128x1x1; r136; s32768 | 20 | n/a | 39.120 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 194 | outer_06.transformer_1.block_19.ffn_linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 20 | n/a | 20.880 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 195 | outer_06.transformer_2.block_20.attention_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 20 | n/a | 2.400 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 196 | outer_06.transformer_2.block_20.attention_qkv_projection | wide_qkv | wide_qkv; wide_qkv_kernel; g9x37x1; b128x1x1; r136; s65536 | 20 | n/a | 19.312 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 197 | outer_06.transformer_2.block_20.attention_qk_rope | qk_rope | qk_rope; fusedQKRoPE19Half2Kernel; g361x13x1; b192x1x1; r16; s0 | 20 | n/a | 3.872 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 198 | outer_06.transformer_2.block_20.attention_fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 20 | n/a | 11.920 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 199 | outer_06.transformer_2.block_20.attention_out_projection_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 20 | n/a | 8.480 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 200 | outer_06.transformer_2.block_20.ffn_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 20 | n/a | 2.432 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 201 | outer_06.transformer_2.block_20.ffn_linear1_gate_swiglu | fused_ffn | fused_ffn; fused_ffn_candidate_a_reuse_kernel; g18x37x1; b128x1x1; r136; s32768 | 20 | n/a | 34.641 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 202 | outer_06.transformer_2.block_20.ffn_linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 20 | n/a | 20.960 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 203 | outer_06.post_norm_silu | affine_silu | affine_silu; affineSiluHalf2Kernel; g4693x1x1; b192x1x1; r17; s0 | 20 | n/a | 3.360 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 204 | outer_06.post_projection_c384_to_c768_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b256x1x1; r154; s73728 | 20 | n/a | 14.560 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 205 | outer_07.pre_norm_silu | affine_silu | affine_silu; affineSiluHalf2Kernel; g4693x1x1; b384x1x1; r17; s0 | 20 | n/a | 5.088 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 206 | outer_07.pre_projection_c768_to_c384 | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 20 | n/a | 12.033 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 207 | outer_07.transformer_0.block_21.attention_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 20 | n/a | 2.464 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 208 | outer_07.transformer_0.block_21.attention_qkv_projection | wide_qkv | wide_qkv; wide_qkv_kernel; g9x37x1; b128x1x1; r136; s65536 | 20 | n/a | 19.328 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 209 | outer_07.transformer_0.block_21.attention_qk_rope | qk_rope | qk_rope; fusedQKRoPE19Half2Kernel; g361x13x1; b192x1x1; r16; s0 | 20 | n/a | 3.840 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 210 | outer_07.transformer_0.block_21.attention_fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 20 | n/a | 11.968 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 211 | outer_07.transformer_0.block_21.attention_out_projection_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 20 | n/a | 8.528 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 212 | outer_07.transformer_0.block_21.ffn_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 20 | n/a | 2.432 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 213 | outer_07.transformer_0.block_21.ffn_linear1_gate_swiglu | fused_ffn | fused_ffn; fused_ffn_candidate_a_reuse_kernel; g18x37x1; b128x1x1; r136; s32768 | 20 | n/a | 35.408 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 214 | outer_07.transformer_0.block_21.ffn_linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 20 | n/a | 21.056 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 215 | outer_07.transformer_1.block_22.attention_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 20 | n/a | 2.400 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 216 | outer_07.transformer_1.block_22.attention_qkv_projection | wide_qkv | wide_qkv; wide_qkv_kernel; g9x37x1; b128x1x1; r136; s65536 | 20 | n/a | 19.552 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 217 | outer_07.transformer_1.block_22.attention_qk_rope | qk_rope | qk_rope; fusedQKRoPE19Half2Kernel; g361x13x1; b192x1x1; r16; s0 | 20 | n/a | 3.872 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 218 | outer_07.transformer_1.block_22.attention_fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 20 | n/a | 12.016 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 219 | outer_07.transformer_1.block_22.attention_out_projection_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 20 | n/a | 8.576 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 220 | outer_07.transformer_1.block_22.ffn_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 20 | n/a | 2.464 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 221 | outer_07.transformer_1.block_22.ffn_linear1_gate_swiglu | fused_ffn | fused_ffn; fused_ffn_candidate_a_reuse_kernel; g18x37x1; b128x1x1; r136; s32768 | 20 | n/a | 35.809 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 222 | outer_07.transformer_1.block_22.ffn_linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 20 | n/a | 21.024 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 223 | outer_07.transformer_2.block_23.attention_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 20 | n/a | 2.464 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 224 | outer_07.transformer_2.block_23.attention_qkv_projection | wide_qkv | wide_qkv; wide_qkv_kernel; g9x37x1; b128x1x1; r136; s65536 | 20 | n/a | 19.505 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 225 | outer_07.transformer_2.block_23.attention_qk_rope | qk_rope | qk_rope; fusedQKRoPE19Half2Kernel; g361x13x1; b192x1x1; r16; s0 | 20 | n/a | 3.904 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 226 | outer_07.transformer_2.block_23.attention_fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 20 | n/a | 12.000 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 227 | outer_07.transformer_2.block_23.attention_out_projection_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 20 | n/a | 8.576 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 228 | outer_07.transformer_2.block_23.ffn_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 20 | n/a | 2.432 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 229 | outer_07.transformer_2.block_23.ffn_linear1_gate_swiglu | fused_ffn | fused_ffn; fused_ffn_candidate_a_reuse_kernel; g18x37x1; b128x1x1; r136; s32768 | 20 | n/a | 35.585 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 230 | outer_07.transformer_2.block_23.ffn_linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 20 | n/a | 21.072 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 231 | outer_07.post_norm_silu | affine_silu | affine_silu; affineSiluHalf2Kernel; g4693x1x1; b192x1x1; r17; s0 | 20 | n/a | 3.360 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 232 | outer_07.post_projection_c384_to_c768_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b256x1x1; r154; s73728 | 20 | n/a | 14.560 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 233 | outer_08.pre_norm_silu | affine_silu | affine_silu; affineSiluHalf2Kernel; g4693x1x1; b384x1x1; r17; s0 | 20 | n/a | 5.088 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 234 | outer_08.pre_projection_c768_to_c384 | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 20 | n/a | 12.049 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 235 | outer_08.transformer_0.block_24.attention_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 20 | n/a | 2.432 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 236 | outer_08.transformer_0.block_24.attention_qkv_projection | wide_qkv | wide_qkv; wide_qkv_kernel; g9x37x1; b128x1x1; r136; s65536 | 20 | n/a | 19.392 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 237 | outer_08.transformer_0.block_24.attention_qk_rope | qk_rope | qk_rope; fusedQKRoPE19Half2Kernel; g361x13x1; b192x1x1; r16; s0 | 20 | n/a | 3.904 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 238 | outer_08.transformer_0.block_24.attention_fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 20 | n/a | 11.984 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 239 | outer_08.transformer_0.block_24.attention_out_projection_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 20 | n/a | 8.560 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 240 | outer_08.transformer_0.block_24.ffn_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 20 | n/a | 2.464 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 241 | outer_08.transformer_0.block_24.ffn_linear1_gate_swiglu | fused_ffn | fused_ffn; fused_ffn_candidate_a_reuse_kernel; g18x37x1; b128x1x1; r136; s32768 | 20 | n/a | 40.401 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 242 | outer_08.transformer_0.block_24.ffn_linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 20 | n/a | 21.041 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 243 | outer_08.transformer_1.block_25.attention_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 20 | n/a | 2.432 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 244 | outer_08.transformer_1.block_25.attention_qkv_projection | wide_qkv | wide_qkv; wide_qkv_kernel; g9x37x1; b128x1x1; r136; s65536 | 20 | n/a | 19.456 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 245 | outer_08.transformer_1.block_25.attention_qk_rope | qk_rope | qk_rope; fusedQKRoPE19Half2Kernel; g361x13x1; b192x1x1; r16; s0 | 20 | n/a | 3.840 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 246 | outer_08.transformer_1.block_25.attention_fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 20 | n/a | 12.000 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 247 | outer_08.transformer_1.block_25.attention_out_projection_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 20 | n/a | 8.480 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 248 | outer_08.transformer_1.block_25.ffn_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 20 | n/a | 2.464 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 249 | outer_08.transformer_1.block_25.ffn_linear1_gate_swiglu | fused_ffn | fused_ffn; fused_ffn_candidate_a_reuse_kernel; g18x37x1; b128x1x1; r136; s32768 | 20 | n/a | 40.721 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 250 | outer_08.transformer_1.block_25.ffn_linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 20 | n/a | 20.960 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 251 | outer_08.transformer_2.block_26.attention_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 20 | n/a | 2.464 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 252 | outer_08.transformer_2.block_26.attention_qkv_projection | wide_qkv | wide_qkv; wide_qkv_kernel; g9x37x1; b128x1x1; r136; s65536 | 20 | n/a | 19.440 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 253 | outer_08.transformer_2.block_26.attention_qk_rope | qk_rope | qk_rope; fusedQKRoPE19Half2Kernel; g361x13x1; b192x1x1; r16; s0 | 20 | n/a | 3.840 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 254 | outer_08.transformer_2.block_26.attention_fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 20 | n/a | 11.904 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 255 | outer_08.transformer_2.block_26.attention_out_projection_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 20 | n/a | 8.496 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 256 | outer_08.transformer_2.block_26.ffn_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 20 | n/a | 2.432 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 257 | outer_08.transformer_2.block_26.ffn_linear1_gate_swiglu | fused_ffn | fused_ffn; fused_ffn_candidate_a_reuse_kernel; g18x37x1; b128x1x1; r136; s32768 | 20 | n/a | 40.289 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 258 | outer_08.transformer_2.block_26.ffn_linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 20 | n/a | 20.897 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 259 | outer_08.post_norm_silu | affine_silu | affine_silu; affineSiluHalf2Kernel; g4693x1x1; b192x1x1; r17; s0 | 20 | n/a | 3.360 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 260 | outer_08.post_projection_c384_to_c768_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b256x1x1; r154; s73728 | 20 | n/a | 14.448 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 261 | outer_09.pre_norm_silu | affine_silu | affine_silu; affineSiluHalf2Kernel; g4693x1x1; b384x1x1; r17; s0 | 20 | n/a | 5.024 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 262 | outer_09.pre_projection_c768_to_c384 | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 20 | n/a | 12.032 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 263 | outer_09.transformer_0.block_27.attention_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 20 | n/a | 2.400 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 264 | outer_09.transformer_0.block_27.attention_qkv_projection | wide_qkv | wide_qkv; wide_qkv_kernel; g9x37x1; b128x1x1; r136; s65536 | 20 | n/a | 19.216 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 265 | outer_09.transformer_0.block_27.attention_qk_rope | qk_rope | qk_rope; fusedQKRoPE19Half2Kernel; g361x13x1; b192x1x1; r16; s0 | 20 | n/a | 3.840 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 266 | outer_09.transformer_0.block_27.attention_fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 20 | n/a | 11.936 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 267 | outer_09.transformer_0.block_27.attention_out_projection_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 20 | n/a | 8.480 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 268 | outer_09.transformer_0.block_27.ffn_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 20 | n/a | 2.432 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 269 | outer_09.transformer_0.block_27.ffn_linear1_gate_swiglu | fused_ffn | fused_ffn; fused_ffn_candidate_a_reuse_kernel; g18x37x1; b128x1x1; r136; s32768 | 20 | n/a | 38.737 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 270 | outer_09.transformer_0.block_27.ffn_linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 20 | n/a | 20.817 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 271 | outer_09.transformer_1.block_28.attention_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 20 | n/a | 2.432 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 272 | outer_09.transformer_1.block_28.attention_qkv_projection | wide_qkv | wide_qkv; wide_qkv_kernel; g9x37x1; b128x1x1; r136; s65536 | 20 | n/a | 19.168 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 273 | outer_09.transformer_1.block_28.attention_qk_rope | qk_rope | qk_rope; fusedQKRoPE19Half2Kernel; g361x13x1; b192x1x1; r16; s0 | 20 | n/a | 3.824 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 274 | outer_09.transformer_1.block_28.attention_fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 20 | n/a | 11.824 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 275 | outer_09.transformer_1.block_28.attention_out_projection_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 20 | n/a | 8.432 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 276 | outer_09.transformer_1.block_28.ffn_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 20 | n/a | 2.432 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 277 | outer_09.transformer_1.block_28.ffn_linear1_gate_swiglu | fused_ffn | fused_ffn; fused_ffn_candidate_a_reuse_kernel; g18x37x1; b128x1x1; r136; s32768 | 20 | n/a | 37.937 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 278 | outer_09.transformer_1.block_28.ffn_linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 20 | n/a | 20.593 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 279 | outer_09.transformer_2.block_29.attention_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 20 | n/a | 2.368 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 280 | outer_09.transformer_2.block_29.attention_qkv_projection | wide_qkv | wide_qkv; wide_qkv_kernel; g9x37x1; b128x1x1; r136; s65536 | 20 | n/a | 19.024 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 281 | outer_09.transformer_2.block_29.attention_qk_rope | qk_rope | qk_rope; fusedQKRoPE19Half2Kernel; g361x13x1; b192x1x1; r16; s0 | 20 | n/a | 3.808 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 282 | outer_09.transformer_2.block_29.attention_fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 20 | n/a | 11.808 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 283 | outer_09.transformer_2.block_29.attention_out_projection_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 20 | n/a | 8.464 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 284 | outer_09.transformer_2.block_29.ffn_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 20 | n/a | 2.432 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 285 | outer_09.transformer_2.block_29.ffn_linear1_gate_swiglu | fused_ffn | fused_ffn; fused_ffn_candidate_a_reuse_kernel; g18x37x1; b128x1x1; r136; s32768 | 20 | n/a | 39.889 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 286 | outer_09.transformer_2.block_29.ffn_linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 20 | n/a | 20.512 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 287 | outer_09.post_norm_silu | affine_silu | affine_silu; affineSiluHalf2Kernel; g4693x1x1; b192x1x1; r17; s0 | 20 | n/a | 3.312 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 288 | outer_09.post_projection_c384_to_c768_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b256x1x1; r154; s73728 | 20 | n/a | 14.192 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 289 | outer_10.pre_norm_silu | affine_silu | affine_silu; affineSiluHalf2Kernel; g4693x1x1; b384x1x1; r17; s0 | 20 | n/a | 5.024 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 290 | outer_10.pre_projection_c768_to_c384 | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 20 | n/a | 11.808 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 291 | outer_10.transformer_0.block_30.attention_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 20 | n/a | 2.432 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 292 | outer_10.transformer_0.block_30.attention_qkv_projection | wide_qkv | wide_qkv; wide_qkv_kernel; g9x37x1; b128x1x1; r136; s65536 | 20 | n/a | 19.152 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 293 | outer_10.transformer_0.block_30.attention_qk_rope | qk_rope | qk_rope; fusedQKRoPE19Half2Kernel; g361x13x1; b192x1x1; r16; s0 | 20 | n/a | 3.840 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 294 | outer_10.transformer_0.block_30.attention_fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 20 | n/a | 11.776 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 295 | outer_10.transformer_0.block_30.attention_out_projection_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 20 | n/a | 8.400 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 296 | outer_10.transformer_0.block_30.ffn_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 20 | n/a | 2.432 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 297 | outer_10.transformer_0.block_30.ffn_linear1_gate_swiglu | fused_ffn | fused_ffn; fused_ffn_candidate_a_reuse_kernel; g18x37x1; b128x1x1; r136; s32768 | 20 | n/a | 35.920 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 298 | outer_10.transformer_0.block_30.ffn_linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 20 | n/a | 20.880 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 299 | outer_10.transformer_1.block_31.attention_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 20 | n/a | 2.464 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 300 | outer_10.transformer_1.block_31.attention_qkv_projection | wide_qkv | wide_qkv; wide_qkv_kernel; g9x37x1; b128x1x1; r136; s65536 | 20 | n/a | 19.297 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 301 | outer_10.transformer_1.block_31.attention_qk_rope | qk_rope | qk_rope; fusedQKRoPE19Half2Kernel; g361x13x1; b192x1x1; r16; s0 | 20 | n/a | 3.872 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 302 | outer_10.transformer_1.block_31.attention_fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 20 | n/a | 11.920 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 303 | outer_10.transformer_1.block_31.attention_out_projection_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 20 | n/a | 8.496 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 304 | outer_10.transformer_1.block_31.ffn_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 20 | n/a | 2.432 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 305 | outer_10.transformer_1.block_31.ffn_linear1_gate_swiglu | fused_ffn | fused_ffn; fused_ffn_candidate_a_reuse_kernel; g18x37x1; b128x1x1; r136; s32768 | 20 | n/a | 37.457 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 306 | outer_10.transformer_1.block_31.ffn_linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 20 | n/a | 20.945 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 307 | outer_10.transformer_2.block_32.attention_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 20 | n/a | 2.432 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 308 | outer_10.transformer_2.block_32.attention_qkv_projection | wide_qkv | wide_qkv; wide_qkv_kernel; g9x37x1; b128x1x1; r136; s65536 | 20 | n/a | 19.328 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 309 | outer_10.transformer_2.block_32.attention_qk_rope | qk_rope | qk_rope; fusedQKRoPE19Half2Kernel; g361x13x1; b192x1x1; r16; s0 | 20 | n/a | 3.872 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 310 | outer_10.transformer_2.block_32.attention_fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 20 | n/a | 11.937 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 311 | outer_10.transformer_2.block_32.attention_out_projection_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 20 | n/a | 8.464 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 312 | outer_10.transformer_2.block_32.ffn_rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 20 | n/a | 2.464 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 313 | outer_10.transformer_2.block_32.ffn_linear1_gate_swiglu | fused_ffn | fused_ffn; fused_ffn_candidate_a_reuse_kernel; g18x37x1; b128x1x1; r136; s32768 | 20 | n/a | 40.465 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 314 | outer_10.transformer_2.block_32.ffn_linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 20 | n/a | 20.960 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 315 | outer_10.post_norm_silu | affine_silu | affine_silu; affineSiluHalf2Kernel; g4693x1x1; b192x1x1; r17; s0 | 20 | n/a | 3.377 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 316 | outer_10.post_projection_c384_to_c768_residual | library_gemm | library_gemm; Kernel2; g148x1x1; b256x1x1; r154; s73728 | 20 | n/a | 14.432 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 317 | trunk.tip_norm_silu | affine_silu | affine_silu; affineSiluHalf2Kernel; g4693x1x1; b384x1x1; r17; s0 | 20 | n/a | 5.056 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 318 | policy.p1_conv | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r80; s98304 | 20 | n/a | 6.160 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 319 | policy.g1_conv | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r80; s98304 | 20 | n/a | 5.904 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 320 | policy.g1_norm_silu | head_elementwise | head_elementwise; applyCScaleBiasNHWCSiluHalfKernel; g1x73x13; b96x5x1; r16; s0 | 20 | n/a | 2.112 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 321 | policy.g1_half_to_float | copy_reformat | copy_reformat; copyFromHalfKernel; g880x1x1; b512x1x1; r16; s0 | 20 | n/a | 1.536 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 322 | policy.g1_global_pool | head_elementwise | head_elementwise; gPoolChannelsNHWCKernel; g2x1x13; b64x8x1; r22; s4096 | 20 | n/a | 4.448 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 323 | policy.gpool_to_bias_matmul | library_gemm | library_gemm; gemmSN_NN_kernel; g2x2x1; b256x1x1; r64; s21504 | 20 | n/a | 5.425 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 324 | policy.p1_half_to_float | copy_reformat | copy_reformat; copyFromHalfKernel; g880x1x1; b512x1x1; r16; s0 | 20 | n/a | 1.504 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 325 | policy.gpool_bias_add | head_elementwise | head_elementwise; addNCBiasInplaceNHWCKernel; g1x73x13; b96x5x1; r16; s0 | 20 | n/a | 1.792 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 326 | policy.p1_norm_silu | head_elementwise | head_elementwise; applyCScaleBiasNHWCSiluKernel; g1x73x13; b96x5x1; r16; s0 | 20 | n/a | 2.176 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 327 | policy.p2_conv | library_gemm | library_gemm; Kernel2; g74x1x1; b128x1x1; r90; s98304 | 20 | n/a | 3.936 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 328 | policy.gpool_to_pass_matmul | library_gemm | library_gemm; gemmSN_NN_kernel; g2x2x1; b256x1x1; r64; s21504 | 20 | n/a | 5.280 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 329 | policy.pass_bias_silu | head_elementwise | head_elementwise; addCBiasInplaceNCKernelSilu; g1x3x1; b96x5x1; r16; s0 | 20 | n/a | 1.024 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 330 | policy.gpool_to_pass_matmul2 | library_gemm | library_gemm; gemmSN_NN_kernel; g1x2x1; b256x1x1; r64; s21504 | 20 | n/a | 2.336 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 331 | value.v1_conv | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r118; s98304 | 20 | n/a | 7.968 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 332 | value.v1_norm_silu | head_elementwise | head_elementwise; applyCScaleBiasNHWCSiluHalfKernel; g1x181x13; b192x2x1; r16; s0 | 20 | n/a | 3.168 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 333 | value.v1_half_to_float | copy_reformat | copy_reformat; copyFromHalfKernel; g1760x1x1; b512x1x1; r16; s0 | 20 | n/a | 2.176 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 334 | value.v1_global_pool | head_elementwise | head_elementwise; valueHeadPoolChannelsNHWCKernel; g3x1x13; b64x8x1; r22; s2048 | 20 | n/a | 3.232 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 335 | value.v2_matmul | library_gemm | library_gemm; gemmSN_NN_kernel; g3x2x1; b256x1x1; r64; s21504 | 20 | n/a | 9.488 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 336 | value.v2_bias_silu | head_elementwise | head_elementwise; addCBiasInplaceNCKernelSilu; g1x7x1; b192x2x1; r16; s0 | 20 | n/a | 1.024 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 337 | value.v3_matmul | library_gemm | library_gemm; gemmSN_NN_kernel; g1x2x1; b256x1x1; r64; s21504 | 20 | n/a | 3.456 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 338 | value.v3_bias | head_elementwise | head_elementwise; addCBiasInplaceNCKernel; g1x1x1; b3x170x1; r16; s0 | 20 | n/a | 0.960 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 339 | value.score_matmul | library_gemm | library_gemm; gemmSN_NN_kernel; g1x2x1; b256x1x1; r64; s21504 | 20 | n/a | 3.488 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 340 | value.score_bias | head_elementwise | head_elementwise; addCBiasInplaceNCKernel; g1x1x1; b6x85x1; r16; s0 | 20 | n/a | 0.928 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 341 | value.ownership_conv | library_gemm | library_gemm; Kernel2; g8x19x3; b128x1x1; r118; s33792 | 20 | n/a | 4.064 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 342 | value.ownership_conv_splitk_reduce | library_gemm | library_gemm; splitKreduce_kernel; g147x1x1; b32x16x1; r49; s0 | 20 | n/a | 1.376 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 343 | value.ownership_half_to_float | copy_reformat | copy_reformat; copyFromHalfKernel; g10x1x1; b512x1x1; r16; s0 | 20 | n/a | 0.960 | n/a | 0.0% | 0.000 | idle (20) | n/a |
