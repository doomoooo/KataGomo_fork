# Nsys stream interference report

- Timed iterations: 20; streams: 48
- Kernels per forward: 48=410

## Kernel families

| family | instances | total ms | median us | p90 us | peer overlap | S2/S1 median | summed excess ms | matched |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| library_gemm | 3360 | 27.060 | 7.424 | 11.748 | 0.0% | n/a | 0.000 | 0 |
| fused_ffn | 660 | 25.234 | 38.224 | 40.577 | 0.0% | n/a | 0.000 | 0 |
| linear2_residual | 660 | 13.581 | 20.576 | 20.865 | 0.0% | n/a | 0.000 | 0 |
| fa4 | 660 | 7.734 | 11.712 | 11.904 | 0.0% | n/a | 0.000 | 0 |
| rmsnorm | 1320 | 3.193 | 2.400 | 2.464 | 0.0% | n/a | 0.000 | 0 |
| qk_rope | 660 | 2.527 | 3.809 | 3.904 | 0.0% | n/a | 0.000 | 0 |
| affine_silu | 460 | 1.934 | 4.928 | 5.056 | 0.0% | n/a | 0.000 | 0 |
| head_elementwise | 240 | 0.593 | 1.921 | 4.448 | 0.0% | n/a | 0.000 | 0 |
| cudnn | 60 | 0.445 | 1.520 | 19.428 | 0.0% | n/a | 0.000 | 0 |
| copy_reformat | 100 | 0.141 | 1.504 | 2.144 | 0.0% | n/a | 0.000 | 0 |
| sumChannelsNCHWKernel | 20 | 0.034 | 1.664 | 1.709 | 0.0% | n/a | 0.000 | 0 |

## Dominant interference pairs

| running family | dominant peer | instances | median overlap | median us | S2/S1 median | matched |
|---|---|---:|---:|---:|---:|---:|
| library_gemm | idle | 3360 | 0.0% | 7.424 | n/a | 0 |
| rmsnorm | idle | 1320 | 0.0% | 2.400 | n/a | 0 |
| qk_rope | idle | 660 | 0.0% | 3.809 | n/a | 0 |
| fa4 | idle | 660 | 0.0% | 11.712 | n/a | 0 |
| fused_ffn | idle | 660 | 0.0% | 38.224 | n/a | 0 |
| linear2_residual | idle | 660 | 0.0% | 20.576 | n/a | 0 |
| affine_silu | idle | 460 | 0.0% | 4.928 | n/a | 0 |
| head_elementwise | idle | 240 | 0.0% | 1.921 | n/a | 0 |
| copy_reformat | idle | 100 | 0.0% | 1.504 | n/a | 0 |
| cudnn | idle | 60 | 0.0% | 1.520 | n/a | 0 |
| sumChannelsNCHWKernel | idle | 20 | 0.0% | 1.664 | n/a | 0 |

## Logical operation groups

Isolated reference total is the isolated median for each ordinal multiplied by its S2 call count; it is a normalized reference, not a second trace total.

| logical group | families | positions | calls | isolated reference ms | S2 total ms | total ratio | excess ms |
|---|---|---:|---:|---:|---:|---:|---:|
| unclassified.ordinal_189.fused_ffn | fused_ffn | 1 | 20 | 0.000 | 0.809 | n/a | 0.000 |
| unclassified.ordinal_97.fused_ffn | fused_ffn | 1 | 20 | 0.000 | 0.807 | n/a | 0.000 |
| unclassified.ordinal_141.fused_ffn | fused_ffn | 1 | 20 | 0.000 | 0.802 | n/a | 0.000 |
| unclassified.ordinal_301.fused_ffn | fused_ffn | 1 | 20 | 0.000 | 0.802 | n/a | 0.000 |
| unclassified.ordinal_223.fused_ffn | fused_ffn | 1 | 20 | 0.000 | 0.800 | n/a | 0.000 |
| unclassified.ordinal_291.fused_ffn | fused_ffn | 1 | 20 | 0.000 | 0.799 | n/a | 0.000 |
| unclassified.ordinal_379.fused_ffn | fused_ffn | 1 | 20 | 0.000 | 0.797 | n/a | 0.000 |
| unclassified.ordinal_311.fused_ffn | fused_ffn | 1 | 20 | 0.000 | 0.793 | n/a | 0.000 |
| unclassified.ordinal_199.fused_ffn | fused_ffn | 1 | 20 | 0.000 | 0.791 | n/a | 0.000 |
| unclassified.ordinal_131.fused_ffn | fused_ffn | 1 | 20 | 0.000 | 0.789 | n/a | 0.000 |
| unclassified.ordinal_87.fused_ffn | fused_ffn | 1 | 20 | 0.000 | 0.787 | n/a | 0.000 |
| unclassified.ordinal_19.fused_ffn | fused_ffn | 1 | 20 | 0.000 | 0.786 | n/a | 0.000 |
| unclassified.ordinal_345.fused_ffn | fused_ffn | 1 | 20 | 0.000 | 0.786 | n/a | 0.000 |
| unclassified.ordinal_53.fused_ffn | fused_ffn | 1 | 20 | 0.000 | 0.782 | n/a | 0.000 |
| value.v2_matmul | fused_ffn | 1 | 20 | 0.000 | 0.782 | n/a | 0.000 |
| policy.gpool_bias_add | fused_ffn | 1 | 20 | 0.000 | 0.779 | n/a | 0.000 |
| unclassified.ordinal_233.fused_ffn | fused_ffn | 1 | 20 | 0.000 | 0.777 | n/a | 0.000 |
| unclassified.ordinal_209.fused_ffn | fused_ffn | 1 | 20 | 0.000 | 0.770 | n/a | 0.000 |
| unclassified.ordinal_63.fused_ffn | fused_ffn | 1 | 20 | 0.000 | 0.767 | n/a | 0.000 |
| unclassified.ordinal_73.fused_ffn | fused_ffn | 1 | 20 | 0.000 | 0.758 | n/a | 0.000 |
| unclassified.ordinal_165.fused_ffn | fused_ffn | 1 | 20 | 0.000 | 0.757 | n/a | 0.000 |
| unclassified.ordinal_39.fused_ffn | fused_ffn | 1 | 20 | 0.000 | 0.757 | n/a | 0.000 |
| unclassified.ordinal_155.fused_ffn | fused_ffn | 1 | 20 | 0.000 | 0.755 | n/a | 0.000 |
| unclassified.ordinal_29.fused_ffn | fused_ffn | 1 | 20 | 0.000 | 0.752 | n/a | 0.000 |
| unclassified.ordinal_121.fused_ffn | fused_ffn | 1 | 20 | 0.000 | 0.751 | n/a | 0.000 |
| unclassified.ordinal_107.fused_ffn | fused_ffn | 1 | 20 | 0.000 | 0.747 | n/a | 0.000 |
| unclassified.ordinal_369.fused_ffn | fused_ffn | 1 | 20 | 0.000 | 0.741 | n/a | 0.000 |
| unclassified.ordinal_359.fused_ffn | fused_ffn | 1 | 20 | 0.000 | 0.714 | n/a | 0.000 |
| unclassified.ordinal_267.fused_ffn | fused_ffn | 1 | 20 | 0.000 | 0.707 | n/a | 0.000 |
| unclassified.ordinal_257.fused_ffn | fused_ffn | 1 | 20 | 0.000 | 0.705 | n/a | 0.000 |
| unclassified.ordinal_277.fused_ffn | fused_ffn | 1 | 20 | 0.000 | 0.704 | n/a | 0.000 |
| unclassified.ordinal_175.fused_ffn | fused_ffn | 1 | 20 | 0.000 | 0.690 | n/a | 0.000 |
| unclassified.ordinal_243.fused_ffn | fused_ffn | 1 | 20 | 0.000 | 0.689 | n/a | 0.000 |
| unclassified.ordinal_20.linear2_residual | linear2_residual | 1 | 20 | 0.000 | 0.416 | n/a | 0.000 |
| unclassified.ordinal_278.linear2_residual | linear2_residual | 1 | 20 | 0.000 | 0.415 | n/a | 0.000 |
| unclassified.ordinal_258.linear2_residual | linear2_residual | 1 | 20 | 0.000 | 0.415 | n/a | 0.000 |
| unclassified.ordinal_292.linear2_residual | linear2_residual | 1 | 20 | 0.000 | 0.415 | n/a | 0.000 |
| unclassified.ordinal_244.linear2_residual | linear2_residual | 1 | 20 | 0.000 | 0.414 | n/a | 0.000 |
| unclassified.ordinal_302.linear2_residual | linear2_residual | 1 | 20 | 0.000 | 0.414 | n/a | 0.000 |
| unclassified.ordinal_268.linear2_residual | linear2_residual | 1 | 20 | 0.000 | 0.414 | n/a | 0.000 |
| unclassified.ordinal_176.linear2_residual | linear2_residual | 1 | 20 | 0.000 | 0.414 | n/a | 0.000 |
| unclassified.ordinal_30.linear2_residual | linear2_residual | 1 | 20 | 0.000 | 0.414 | n/a | 0.000 |
| unclassified.ordinal_370.linear2_residual | linear2_residual | 1 | 20 | 0.000 | 0.413 | n/a | 0.000 |
| unclassified.ordinal_380.linear2_residual | linear2_residual | 1 | 20 | 0.000 | 0.413 | n/a | 0.000 |
| unclassified.ordinal_190.linear2_residual | linear2_residual | 1 | 20 | 0.000 | 0.413 | n/a | 0.000 |
| unclassified.ordinal_312.linear2_residual | linear2_residual | 1 | 20 | 0.000 | 0.413 | n/a | 0.000 |
| unclassified.ordinal_234.linear2_residual | linear2_residual | 1 | 20 | 0.000 | 0.412 | n/a | 0.000 |
| unclassified.ordinal_166.linear2_residual | linear2_residual | 1 | 20 | 0.000 | 0.412 | n/a | 0.000 |
| unclassified.ordinal_360.linear2_residual | linear2_residual | 1 | 20 | 0.000 | 0.412 | n/a | 0.000 |
| unclassified.ordinal_156.linear2_residual | linear2_residual | 1 | 20 | 0.000 | 0.412 | n/a | 0.000 |
| unclassified.ordinal_54.linear2_residual | linear2_residual | 1 | 20 | 0.000 | 0.412 | n/a | 0.000 |
| unclassified.ordinal_98.linear2_residual | linear2_residual | 1 | 20 | 0.000 | 0.412 | n/a | 0.000 |
| unclassified.ordinal_224.linear2_residual | linear2_residual | 1 | 20 | 0.000 | 0.411 | n/a | 0.000 |
| unclassified.ordinal_88.linear2_residual | linear2_residual | 1 | 20 | 0.000 | 0.411 | n/a | 0.000 |
| unclassified.ordinal_108.linear2_residual | linear2_residual | 1 | 20 | 0.000 | 0.411 | n/a | 0.000 |
| unclassified.ordinal_40.linear2_residual | linear2_residual | 1 | 20 | 0.000 | 0.410 | n/a | 0.000 |
| policy.p1_norm_silu | linear2_residual | 1 | 20 | 0.000 | 0.410 | n/a | 0.000 |
| unclassified.ordinal_200.linear2_residual | linear2_residual | 1 | 20 | 0.000 | 0.410 | n/a | 0.000 |
| unclassified.ordinal_64.linear2_residual | linear2_residual | 1 | 20 | 0.000 | 0.410 | n/a | 0.000 |
| unclassified.ordinal_74.linear2_residual | linear2_residual | 1 | 20 | 0.000 | 0.410 | n/a | 0.000 |
| unclassified.ordinal_122.linear2_residual | linear2_residual | 1 | 20 | 0.000 | 0.409 | n/a | 0.000 |
| unclassified.ordinal_210.linear2_residual | linear2_residual | 1 | 20 | 0.000 | 0.408 | n/a | 0.000 |
| value.v2_bias_silu | linear2_residual | 1 | 20 | 0.000 | 0.408 | n/a | 0.000 |
| unclassified.ordinal_132.linear2_residual | linear2_residual | 1 | 20 | 0.000 | 0.408 | n/a | 0.000 |
| unclassified.ordinal_346.linear2_residual | linear2_residual | 1 | 20 | 0.000 | 0.406 | n/a | 0.000 |
| unclassified.ordinal_142.linear2_residual | linear2_residual | 1 | 20 | 0.000 | 0.405 | n/a | 0.000 |
| frontend.initial_conv | cudnn | 1 | 20 | 0.000 | 0.389 | n/a | 0.000 |
| unclassified.ordinal_280.library_gemm | library_gemm | 1 | 20 | 0.000 | 0.288 | n/a | 0.000 |
| unclassified.ordinal_178.library_gemm | library_gemm | 1 | 20 | 0.000 | 0.288 | n/a | 0.000 |
| unclassified.ordinal_246.library_gemm | library_gemm | 1 | 20 | 0.000 | 0.287 | n/a | 0.000 |
| unclassified.ordinal_110.library_gemm | library_gemm | 1 | 20 | 0.000 | 0.286 | n/a | 0.000 |
| unclassified.ordinal_314.library_gemm | library_gemm | 1 | 20 | 0.000 | 0.286 | n/a | 0.000 |
| unclassified.ordinal_382.library_gemm | library_gemm | 1 | 20 | 0.000 | 0.285 | n/a | 0.000 |
| unclassified.ordinal_76.library_gemm | library_gemm | 1 | 20 | 0.000 | 0.284 | n/a | 0.000 |
| unclassified.ordinal_212.library_gemm | library_gemm | 1 | 20 | 0.000 | 0.284 | n/a | 0.000 |
| unclassified.ordinal_42.library_gemm | library_gemm | 1 | 20 | 0.000 | 0.283 | n/a | 0.000 |
| unclassified.ordinal_348.library_gemm | library_gemm | 1 | 20 | 0.000 | 0.282 | n/a | 0.000 |
| unclassified.ordinal_144.library_gemm | library_gemm | 1 | 20 | 0.000 | 0.281 | n/a | 0.000 |
| unclassified.ordinal_282.library_gemm | library_gemm | 1 | 20 | 0.000 | 0.238 | n/a | 0.000 |
| unclassified.ordinal_180.library_gemm | library_gemm | 1 | 20 | 0.000 | 0.238 | n/a | 0.000 |
| unclassified.ordinal_248.library_gemm | library_gemm | 1 | 20 | 0.000 | 0.237 | n/a | 0.000 |
| unclassified.ordinal_26.fa4 | fa4 | 1 | 20 | 0.000 | 0.237 | n/a | 0.000 |
| unclassified.ordinal_316.library_gemm | library_gemm | 1 | 20 | 0.000 | 0.237 | n/a | 0.000 |
| unclassified.ordinal_264.fa4 | fa4 | 1 | 20 | 0.000 | 0.237 | n/a | 0.000 |
| unclassified.ordinal_44.library_gemm | library_gemm | 1 | 20 | 0.000 | 0.236 | n/a | 0.000 |
| unclassified.ordinal_288.fa4 | fa4 | 1 | 20 | 0.000 | 0.236 | n/a | 0.000 |
| unclassified.ordinal_274.fa4 | fa4 | 1 | 20 | 0.000 | 0.236 | n/a | 0.000 |
| unclassified.ordinal_112.library_gemm | library_gemm | 1 | 20 | 0.000 | 0.236 | n/a | 0.000 |
| unclassified.ordinal_186.fa4 | fa4 | 1 | 20 | 0.000 | 0.236 | n/a | 0.000 |
| policy.g1_global_pool | fa4 | 1 | 20 | 0.000 | 0.236 | n/a | 0.000 |
| unclassified.ordinal_254.fa4 | fa4 | 1 | 20 | 0.000 | 0.236 | n/a | 0.000 |
| unclassified.ordinal_298.fa4 | fa4 | 1 | 20 | 0.000 | 0.235 | n/a | 0.000 |
| unclassified.ordinal_376.fa4 | fa4 | 1 | 20 | 0.000 | 0.235 | n/a | 0.000 |
| unclassified.ordinal_214.library_gemm | library_gemm | 1 | 20 | 0.000 | 0.235 | n/a | 0.000 |
| unclassified.ordinal_78.library_gemm | library_gemm | 1 | 20 | 0.000 | 0.235 | n/a | 0.000 |
| unclassified.ordinal_172.fa4 | fa4 | 1 | 20 | 0.000 | 0.235 | n/a | 0.000 |
| unclassified.ordinal_366.fa4 | fa4 | 1 | 20 | 0.000 | 0.235 | n/a | 0.000 |
| unclassified.ordinal_162.fa4 | fa4 | 1 | 20 | 0.000 | 0.235 | n/a | 0.000 |
| unclassified.ordinal_240.fa4 | fa4 | 1 | 20 | 0.000 | 0.235 | n/a | 0.000 |
| unclassified.ordinal_308.fa4 | fa4 | 1 | 20 | 0.000 | 0.235 | n/a | 0.000 |
| unclassified.ordinal_118.fa4 | fa4 | 1 | 20 | 0.000 | 0.235 | n/a | 0.000 |
| unclassified.ordinal_36.fa4 | fa4 | 1 | 20 | 0.000 | 0.234 | n/a | 0.000 |
| unclassified.ordinal_196.fa4 | fa4 | 1 | 20 | 0.000 | 0.234 | n/a | 0.000 |
| unclassified.ordinal_60.fa4 | fa4 | 1 | 20 | 0.000 | 0.234 | n/a | 0.000 |
| unclassified.ordinal_230.fa4 | fa4 | 1 | 20 | 0.000 | 0.234 | n/a | 0.000 |
| unclassified.ordinal_50.fa4 | fa4 | 1 | 20 | 0.000 | 0.234 | n/a | 0.000 |
| unclassified.ordinal_146.library_gemm | library_gemm | 1 | 20 | 0.000 | 0.234 | n/a | 0.000 |
| unclassified.ordinal_94.fa4 | fa4 | 1 | 20 | 0.000 | 0.234 | n/a | 0.000 |
| unclassified.ordinal_104.fa4 | fa4 | 1 | 20 | 0.000 | 0.234 | n/a | 0.000 |
| unclassified.ordinal_84.fa4 | fa4 | 1 | 20 | 0.000 | 0.234 | n/a | 0.000 |
| unclassified.ordinal_350.library_gemm | library_gemm | 1 | 20 | 0.000 | 0.233 | n/a | 0.000 |
| unclassified.ordinal_70.fa4 | fa4 | 1 | 20 | 0.000 | 0.233 | n/a | 0.000 |
| unclassified.ordinal_220.fa4 | fa4 | 1 | 20 | 0.000 | 0.233 | n/a | 0.000 |
| value.v1_norm_silu | fa4 | 1 | 20 | 0.000 | 0.233 | n/a | 0.000 |
| unclassified.ordinal_152.fa4 | fa4 | 1 | 20 | 0.000 | 0.233 | n/a | 0.000 |
| unclassified.ordinal_16.fa4 | fa4 | 1 | 20 | 0.000 | 0.233 | n/a | 0.000 |
| unclassified.ordinal_206.fa4 | fa4 | 1 | 20 | 0.000 | 0.233 | n/a | 0.000 |
| unclassified.ordinal_356.fa4 | fa4 | 1 | 20 | 0.000 | 0.233 | n/a | 0.000 |
| unclassified.ordinal_128.fa4 | fa4 | 1 | 20 | 0.000 | 0.233 | n/a | 0.000 |
| unclassified.ordinal_138.fa4 | fa4 | 1 | 20 | 0.000 | 0.233 | n/a | 0.000 |
| value.ownership_conv_splitk_reduce | fa4 | 1 | 20 | 0.000 | 0.232 | n/a | 0.000 |
| unclassified.ordinal_10.library_gemm | library_gemm | 1 | 20 | 0.000 | 0.232 | n/a | 0.000 |
| unclassified.ordinal_401.library_gemm | library_gemm | 1 | 20 | 0.000 | 0.190 | n/a | 0.000 |
| unclassified.ordinal_197.library_gemm | library_gemm | 1 | 20 | 0.000 | 0.170 | n/a | 0.000 |
| unclassified.ordinal_187.library_gemm | library_gemm | 1 | 20 | 0.000 | 0.170 | n/a | 0.000 |
| unclassified.ordinal_265.library_gemm | library_gemm | 1 | 20 | 0.000 | 0.169 | n/a | 0.000 |
| unclassified.ordinal_275.library_gemm | library_gemm | 1 | 20 | 0.000 | 0.169 | n/a | 0.000 |
| unclassified.ordinal_95.library_gemm | library_gemm | 1 | 20 | 0.000 | 0.169 | n/a | 0.000 |
| unclassified.ordinal_119.library_gemm | library_gemm | 1 | 20 | 0.000 | 0.169 | n/a | 0.000 |
| unclassified.ordinal_61.library_gemm | library_gemm | 1 | 20 | 0.000 | 0.168 | n/a | 0.000 |
| unclassified.ordinal_163.library_gemm | library_gemm | 1 | 20 | 0.000 | 0.168 | n/a | 0.000 |
| unclassified.ordinal_241.library_gemm | library_gemm | 1 | 20 | 0.000 | 0.168 | n/a | 0.000 |
| unclassified.ordinal_129.library_gemm | library_gemm | 1 | 20 | 0.000 | 0.168 | n/a | 0.000 |
| unclassified.ordinal_367.library_gemm | library_gemm | 1 | 20 | 0.000 | 0.168 | n/a | 0.000 |
| unclassified.ordinal_299.library_gemm | library_gemm | 1 | 20 | 0.000 | 0.168 | n/a | 0.000 |
| unclassified.ordinal_27.library_gemm | library_gemm | 1 | 20 | 0.000 | 0.168 | n/a | 0.000 |
| unclassified.ordinal_289.library_gemm | library_gemm | 1 | 20 | 0.000 | 0.168 | n/a | 0.000 |
| unclassified.ordinal_377.library_gemm | library_gemm | 1 | 20 | 0.000 | 0.168 | n/a | 0.000 |
| unclassified.ordinal_51.library_gemm | library_gemm | 1 | 20 | 0.000 | 0.168 | n/a | 0.000 |
| policy.gpool_to_bias_matmul | library_gemm | 1 | 20 | 0.000 | 0.168 | n/a | 0.000 |
| unclassified.ordinal_71.library_gemm | library_gemm | 1 | 20 | 0.000 | 0.168 | n/a | 0.000 |
| unclassified.ordinal_173.library_gemm | library_gemm | 1 | 20 | 0.000 | 0.168 | n/a | 0.000 |
| unclassified.ordinal_309.library_gemm | library_gemm | 1 | 20 | 0.000 | 0.167 | n/a | 0.000 |
| unclassified.ordinal_221.library_gemm | library_gemm | 1 | 20 | 0.000 | 0.167 | n/a | 0.000 |
| unclassified.ordinal_357.library_gemm | library_gemm | 1 | 20 | 0.000 | 0.167 | n/a | 0.000 |
| unclassified.ordinal_255.library_gemm | library_gemm | 1 | 20 | 0.000 | 0.167 | n/a | 0.000 |
| value.v1_half_to_float | library_gemm | 1 | 20 | 0.000 | 0.167 | n/a | 0.000 |
| unclassified.ordinal_37.library_gemm | library_gemm | 1 | 20 | 0.000 | 0.167 | n/a | 0.000 |
| unclassified.ordinal_139.library_gemm | library_gemm | 1 | 20 | 0.000 | 0.167 | n/a | 0.000 |
| unclassified.ordinal_231.library_gemm | library_gemm | 1 | 20 | 0.000 | 0.166 | n/a | 0.000 |
| unclassified.ordinal_105.library_gemm | library_gemm | 1 | 20 | 0.000 | 0.166 | n/a | 0.000 |
| unclassified.ordinal_207.library_gemm | library_gemm | 1 | 20 | 0.000 | 0.166 | n/a | 0.000 |
| unclassified.ordinal_17.library_gemm | library_gemm | 1 | 20 | 0.000 | 0.166 | n/a | 0.000 |
| unclassified.ordinal_85.library_gemm | library_gemm | 1 | 20 | 0.000 | 0.166 | n/a | 0.000 |
| unclassified.ordinal_153.library_gemm | library_gemm | 1 | 20 | 0.000 | 0.165 | n/a | 0.000 |
| value.ownership_half_to_float | library_gemm | 1 | 20 | 0.000 | 0.165 | n/a | 0.000 |
| unclassified.ordinal_397.library_gemm | library_gemm | 1 | 20 | 0.000 | 0.158 | n/a | 0.000 |
| frontend.initial_global_broadcast_add | head_elementwise | 1 | 20 | 0.000 | 0.154 | n/a | 0.000 |
| unclassified.ordinal_260.library_gemm | library_gemm | 1 | 20 | 0.000 | 0.151 | n/a | 0.000 |
| unclassified.ordinal_22.library_gemm | library_gemm | 1 | 20 | 0.000 | 0.151 | n/a | 0.000 |
| unclassified.ordinal_270.library_gemm | library_gemm | 1 | 20 | 0.000 | 0.150 | n/a | 0.000 |
| unclassified.ordinal_304.library_gemm | library_gemm | 1 | 20 | 0.000 | 0.150 | n/a | 0.000 |
| unclassified.ordinal_372.library_gemm | library_gemm | 1 | 20 | 0.000 | 0.150 | n/a | 0.000 |
| unclassified.ordinal_294.library_gemm | library_gemm | 1 | 20 | 0.000 | 0.150 | n/a | 0.000 |
| unclassified.ordinal_158.library_gemm | library_gemm | 1 | 20 | 0.000 | 0.150 | n/a | 0.000 |
| unclassified.ordinal_226.library_gemm | library_gemm | 1 | 20 | 0.000 | 0.149 | n/a | 0.000 |
| unclassified.ordinal_32.library_gemm | library_gemm | 1 | 20 | 0.000 | 0.149 | n/a | 0.000 |
| unclassified.ordinal_236.library_gemm | library_gemm | 1 | 20 | 0.000 | 0.149 | n/a | 0.000 |
| unclassified.ordinal_168.library_gemm | library_gemm | 1 | 20 | 0.000 | 0.149 | n/a | 0.000 |
| unclassified.ordinal_261.library_gemm | library_gemm | 1 | 20 | 0.000 | 0.149 | n/a | 0.000 |
| unclassified.ordinal_24.library_gemm | library_gemm | 1 | 20 | 0.000 | 0.149 | n/a | 0.000 |
| unclassified.ordinal_193.library_gemm | library_gemm | 1 | 20 | 0.000 | 0.149 | n/a | 0.000 |
| unclassified.ordinal_23.library_gemm | library_gemm | 1 | 20 | 0.000 | 0.149 | n/a | 0.000 |
| unclassified.ordinal_192.library_gemm | library_gemm | 1 | 20 | 0.000 | 0.149 | n/a | 0.000 |
| unclassified.ordinal_306.library_gemm | library_gemm | 1 | 20 | 0.000 | 0.149 | n/a | 0.000 |
| unclassified.ordinal_100.library_gemm | library_gemm | 1 | 20 | 0.000 | 0.149 | n/a | 0.000 |
| unclassified.ordinal_362.library_gemm | library_gemm | 1 | 20 | 0.000 | 0.149 | n/a | 0.000 |
| unclassified.ordinal_202.library_gemm | library_gemm | 1 | 20 | 0.000 | 0.148 | n/a | 0.000 |
| unclassified.ordinal_305.library_gemm | library_gemm | 1 | 20 | 0.000 | 0.148 | n/a | 0.000 |
| unclassified.ordinal_33.library_gemm | library_gemm | 1 | 20 | 0.000 | 0.148 | n/a | 0.000 |
| unclassified.ordinal_66.library_gemm | library_gemm | 1 | 20 | 0.000 | 0.148 | n/a | 0.000 |
| unclassified.ordinal_284.library_gemm | library_gemm | 1 | 20 | 0.000 | 0.148 | n/a | 0.000 |
| unclassified.ordinal_262.library_gemm | library_gemm | 1 | 20 | 0.000 | 0.148 | n/a | 0.000 |
| unclassified.ordinal_91.library_gemm | library_gemm | 1 | 20 | 0.000 | 0.148 | n/a | 0.000 |
| unclassified.ordinal_56.library_gemm | library_gemm | 1 | 20 | 0.000 | 0.148 | n/a | 0.000 |
| unclassified.ordinal_227.library_gemm | library_gemm | 1 | 20 | 0.000 | 0.148 | n/a | 0.000 |
| policy.p1_conv | library_gemm | 1 | 20 | 0.000 | 0.148 | n/a | 0.000 |
| unclassified.ordinal_159.library_gemm | library_gemm | 1 | 20 | 0.000 | 0.148 | n/a | 0.000 |
| policy.gpool_to_pass_matmul | library_gemm | 1 | 20 | 0.000 | 0.148 | n/a | 0.000 |
| unclassified.ordinal_90.library_gemm | library_gemm | 1 | 20 | 0.000 | 0.148 | n/a | 0.000 |
| unclassified.ordinal_251.library_gemm | library_gemm | 1 | 20 | 0.000 | 0.148 | n/a | 0.000 |
| unclassified.ordinal_67.library_gemm | library_gemm | 1 | 20 | 0.000 | 0.148 | n/a | 0.000 |
| unclassified.ordinal_250.library_gemm | library_gemm | 1 | 20 | 0.000 | 0.147 | n/a | 0.000 |
| unclassified.ordinal_124.library_gemm | library_gemm | 1 | 20 | 0.000 | 0.147 | n/a | 0.000 |
| unclassified.ordinal_272.library_gemm | library_gemm | 1 | 20 | 0.000 | 0.147 | n/a | 0.000 |
| unclassified.ordinal_203.library_gemm | library_gemm | 1 | 20 | 0.000 | 0.147 | n/a | 0.000 |
| unclassified.ordinal_363.library_gemm | library_gemm | 1 | 20 | 0.000 | 0.147 | n/a | 0.000 |
| unclassified.ordinal_116.library_gemm | library_gemm | 1 | 20 | 0.000 | 0.147 | n/a | 0.000 |
| unclassified.ordinal_170.library_gemm | library_gemm | 1 | 20 | 0.000 | 0.147 | n/a | 0.000 |
| unclassified.ordinal_114.library_gemm | library_gemm | 1 | 20 | 0.000 | 0.147 | n/a | 0.000 |
| unclassified.ordinal_295.library_gemm | library_gemm | 1 | 20 | 0.000 | 0.147 | n/a | 0.000 |
| unclassified.ordinal_286.library_gemm | library_gemm | 1 | 20 | 0.000 | 0.147 | n/a | 0.000 |
| unclassified.ordinal_169.library_gemm | library_gemm | 1 | 20 | 0.000 | 0.147 | n/a | 0.000 |
| unclassified.ordinal_81.library_gemm | library_gemm | 1 | 20 | 0.000 | 0.147 | n/a | 0.000 |
| unclassified.ordinal_101.library_gemm | library_gemm | 1 | 20 | 0.000 | 0.147 | n/a | 0.000 |
| unclassified.ordinal_216.library_gemm | library_gemm | 1 | 20 | 0.000 | 0.147 | n/a | 0.000 |
| value.v3_bias | library_gemm | 1 | 20 | 0.000 | 0.147 | n/a | 0.000 |
| policy.pass_bias_silu | library_gemm | 1 | 20 | 0.000 | 0.147 | n/a | 0.000 |
| unclassified.ordinal_13.library_gemm | library_gemm | 1 | 20 | 0.000 | 0.147 | n/a | 0.000 |
| unclassified.ordinal_374.library_gemm | library_gemm | 1 | 20 | 0.000 | 0.147 | n/a | 0.000 |
| unclassified.ordinal_46.library_gemm | library_gemm | 1 | 20 | 0.000 | 0.147 | n/a | 0.000 |
| unclassified.ordinal_48.library_gemm | library_gemm | 1 | 20 | 0.000 | 0.147 | n/a | 0.000 |
| unclassified.ordinal_373.library_gemm | library_gemm | 1 | 20 | 0.000 | 0.147 | n/a | 0.000 |
| unclassified.ordinal_126.library_gemm | library_gemm | 1 | 20 | 0.000 | 0.147 | n/a | 0.000 |
| unclassified.ordinal_34.library_gemm | library_gemm | 1 | 20 | 0.000 | 0.147 | n/a | 0.000 |
| unclassified.ordinal_271.library_gemm | library_gemm | 1 | 20 | 0.000 | 0.147 | n/a | 0.000 |
| unclassified.ordinal_238.library_gemm | library_gemm | 1 | 20 | 0.000 | 0.147 | n/a | 0.000 |
| unclassified.ordinal_364.library_gemm | library_gemm | 1 | 20 | 0.000 | 0.147 | n/a | 0.000 |
| unclassified.ordinal_58.library_gemm | library_gemm | 1 | 20 | 0.000 | 0.147 | n/a | 0.000 |
| value.score_matmul | library_gemm | 1 | 20 | 0.000 | 0.146 | n/a | 0.000 |
| unclassified.ordinal_228.library_gemm | library_gemm | 1 | 20 | 0.000 | 0.146 | n/a | 0.000 |
| unclassified.ordinal_285.library_gemm | library_gemm | 1 | 20 | 0.000 | 0.146 | n/a | 0.000 |
| unclassified.ordinal_57.library_gemm | library_gemm | 1 | 20 | 0.000 | 0.146 | n/a | 0.000 |
| unclassified.ordinal_134.library_gemm | library_gemm | 1 | 20 | 0.000 | 0.146 | n/a | 0.000 |
| unclassified.ordinal_125.library_gemm | library_gemm | 1 | 20 | 0.000 | 0.146 | n/a | 0.000 |
| unclassified.ordinal_252.library_gemm | library_gemm | 1 | 20 | 0.000 | 0.146 | n/a | 0.000 |
| policy.g1_conv | library_gemm | 1 | 20 | 0.000 | 0.146 | n/a | 0.000 |
| unclassified.ordinal_182.library_gemm | library_gemm | 1 | 20 | 0.000 | 0.146 | n/a | 0.000 |
| policy.g1_norm_silu | library_gemm | 1 | 20 | 0.000 | 0.146 | n/a | 0.000 |
| unclassified.ordinal_204.library_gemm | library_gemm | 1 | 20 | 0.000 | 0.146 | n/a | 0.000 |
| unclassified.ordinal_296.library_gemm | library_gemm | 1 | 20 | 0.000 | 0.146 | n/a | 0.000 |
| unclassified.ordinal_354.library_gemm | library_gemm | 1 | 20 | 0.000 | 0.146 | n/a | 0.000 |
| unclassified.ordinal_160.library_gemm | library_gemm | 1 | 20 | 0.000 | 0.146 | n/a | 0.000 |
| unclassified.ordinal_237.library_gemm | library_gemm | 1 | 20 | 0.000 | 0.146 | n/a | 0.000 |
| unclassified.ordinal_217.library_gemm | library_gemm | 1 | 20 | 0.000 | 0.146 | n/a | 0.000 |
| unclassified.ordinal_194.library_gemm | library_gemm | 1 | 20 | 0.000 | 0.146 | n/a | 0.000 |
| unclassified.ordinal_68.library_gemm | library_gemm | 1 | 20 | 0.000 | 0.146 | n/a | 0.000 |
| unclassified.ordinal_183.library_gemm | library_gemm | 1 | 20 | 0.000 | 0.146 | n/a | 0.000 |
| unclassified.ordinal_12.library_gemm | library_gemm | 1 | 20 | 0.000 | 0.146 | n/a | 0.000 |
| unclassified.ordinal_102.library_gemm | library_gemm | 1 | 20 | 0.000 | 0.145 | n/a | 0.000 |
| unclassified.ordinal_80.library_gemm | library_gemm | 1 | 20 | 0.000 | 0.145 | n/a | 0.000 |
| unclassified.ordinal_184.library_gemm | library_gemm | 1 | 20 | 0.000 | 0.145 | n/a | 0.000 |
| unclassified.ordinal_47.library_gemm | library_gemm | 1 | 20 | 0.000 | 0.145 | n/a | 0.000 |
| unclassified.ordinal_115.library_gemm | library_gemm | 1 | 20 | 0.000 | 0.145 | n/a | 0.000 |
| unclassified.ordinal_150.library_gemm | library_gemm | 1 | 20 | 0.000 | 0.145 | n/a | 0.000 |
| unclassified.ordinal_218.library_gemm | library_gemm | 1 | 20 | 0.000 | 0.145 | n/a | 0.000 |
| unclassified.ordinal_135.library_gemm | library_gemm | 1 | 20 | 0.000 | 0.145 | n/a | 0.000 |
| unclassified.ordinal_82.library_gemm | library_gemm | 1 | 20 | 0.000 | 0.145 | n/a | 0.000 |
| unclassified.ordinal_148.library_gemm | library_gemm | 1 | 20 | 0.000 | 0.145 | n/a | 0.000 |
| unclassified.ordinal_92.library_gemm | library_gemm | 1 | 20 | 0.000 | 0.145 | n/a | 0.000 |
| unclassified.ordinal_149.library_gemm | library_gemm | 1 | 20 | 0.000 | 0.145 | n/a | 0.000 |
| unclassified.ordinal_136.library_gemm | library_gemm | 1 | 20 | 0.000 | 0.145 | n/a | 0.000 |
| unclassified.ordinal_14.library_gemm | library_gemm | 1 | 20 | 0.000 | 0.145 | n/a | 0.000 |
| unclassified.ordinal_353.library_gemm | library_gemm | 1 | 20 | 0.000 | 0.145 | n/a | 0.000 |
| policy.gpool_to_pass_matmul2 | library_gemm | 1 | 20 | 0.000 | 0.145 | n/a | 0.000 |
| value.score_bias | library_gemm | 1 | 20 | 0.000 | 0.145 | n/a | 0.000 |
| unclassified.ordinal_352.library_gemm | library_gemm | 1 | 20 | 0.000 | 0.144 | n/a | 0.000 |
| unclassified.ordinal_384.library_gemm | library_gemm | 1 | 20 | 0.000 | 0.123 | n/a | 0.000 |
| unclassified.ordinal_385.library_gemm | library_gemm | 1 | 20 | 0.000 | 0.117 | n/a | 0.000 |
| unclassified.ordinal_389.library_gemm | library_gemm | 1 | 20 | 0.000 | 0.107 | n/a | 0.000 |
| unclassified.ordinal_394.library_gemm | library_gemm | 1 | 20 | 0.000 | 0.106 | n/a | 0.000 |
| unclassified.ordinal_43.affine_silu | affine_silu | 1 | 20 | 0.000 | 0.101 | n/a | 0.000 |
| unclassified.ordinal_281.affine_silu | affine_silu | 1 | 20 | 0.000 | 0.101 | n/a | 0.000 |
| unclassified.ordinal_179.affine_silu | affine_silu | 1 | 20 | 0.000 | 0.101 | n/a | 0.000 |
| unclassified.ordinal_247.affine_silu | affine_silu | 1 | 20 | 0.000 | 0.101 | n/a | 0.000 |
| unclassified.ordinal_111.affine_silu | affine_silu | 1 | 20 | 0.000 | 0.100 | n/a | 0.000 |
| unclassified.ordinal_383.affine_silu | affine_silu | 1 | 20 | 0.000 | 0.100 | n/a | 0.000 |
| unclassified.ordinal_315.affine_silu | affine_silu | 1 | 20 | 0.000 | 0.100 | n/a | 0.000 |
| unclassified.ordinal_145.affine_silu | affine_silu | 1 | 20 | 0.000 | 0.100 | n/a | 0.000 |
| unclassified.ordinal_213.affine_silu | affine_silu | 1 | 20 | 0.000 | 0.100 | n/a | 0.000 |
| unclassified.ordinal_349.affine_silu | affine_silu | 1 | 20 | 0.000 | 0.100 | n/a | 0.000 |
| unclassified.ordinal_77.affine_silu | affine_silu | 1 | 20 | 0.000 | 0.099 | n/a | 0.000 |
| unclassified.ordinal_9.affine_silu | affine_silu | 1 | 20 | 0.000 | 0.099 | n/a | 0.000 |
| unclassified.ordinal_388.head_elementwise | head_elementwise | 1 | 20 | 0.000 | 0.088 | n/a | 0.000 |
| unclassified.ordinal_407.library_gemm | library_gemm | 1 | 20 | 0.000 | 0.081 | n/a | 0.000 |
| unclassified.ordinal_393.library_gemm | library_gemm | 1 | 20 | 0.000 | 0.078 | n/a | 0.000 |
| unclassified.ordinal_25.qk_rope | qk_rope | 1 | 20 | 0.000 | 0.078 | n/a | 0.000 |
| unclassified.ordinal_273.qk_rope | qk_rope | 1 | 20 | 0.000 | 0.078 | n/a | 0.000 |
| unclassified.ordinal_287.qk_rope | qk_rope | 1 | 20 | 0.000 | 0.077 | n/a | 0.000 |
| unclassified.ordinal_185.qk_rope | qk_rope | 1 | 20 | 0.000 | 0.077 | n/a | 0.000 |
| unclassified.ordinal_263.qk_rope | qk_rope | 1 | 20 | 0.000 | 0.077 | n/a | 0.000 |
| unclassified.ordinal_229.qk_rope | qk_rope | 1 | 20 | 0.000 | 0.077 | n/a | 0.000 |
| unclassified.ordinal_35.qk_rope | qk_rope | 1 | 20 | 0.000 | 0.077 | n/a | 0.000 |
| unclassified.ordinal_171.qk_rope | qk_rope | 1 | 20 | 0.000 | 0.077 | n/a | 0.000 |
| policy.g1_half_to_float | qk_rope | 1 | 20 | 0.000 | 0.077 | n/a | 0.000 |
| unclassified.ordinal_297.qk_rope | qk_rope | 1 | 20 | 0.000 | 0.077 | n/a | 0.000 |
| unclassified.ordinal_375.qk_rope | qk_rope | 1 | 20 | 0.000 | 0.077 | n/a | 0.000 |
| unclassified.ordinal_365.qk_rope | qk_rope | 1 | 20 | 0.000 | 0.077 | n/a | 0.000 |
| unclassified.ordinal_195.qk_rope | qk_rope | 1 | 20 | 0.000 | 0.077 | n/a | 0.000 |
| unclassified.ordinal_219.qk_rope | qk_rope | 1 | 20 | 0.000 | 0.077 | n/a | 0.000 |
| unclassified.ordinal_239.qk_rope | qk_rope | 1 | 20 | 0.000 | 0.077 | n/a | 0.000 |
| unclassified.ordinal_83.qk_rope | qk_rope | 1 | 20 | 0.000 | 0.077 | n/a | 0.000 |
| unclassified.ordinal_15.qk_rope | qk_rope | 1 | 20 | 0.000 | 0.077 | n/a | 0.000 |
| unclassified.ordinal_117.qk_rope | qk_rope | 1 | 20 | 0.000 | 0.077 | n/a | 0.000 |
| unclassified.ordinal_103.qk_rope | qk_rope | 1 | 20 | 0.000 | 0.077 | n/a | 0.000 |
| unclassified.ordinal_151.qk_rope | qk_rope | 1 | 20 | 0.000 | 0.076 | n/a | 0.000 |
| value.v1_conv | qk_rope | 1 | 20 | 0.000 | 0.076 | n/a | 0.000 |
| unclassified.ordinal_161.qk_rope | qk_rope | 1 | 20 | 0.000 | 0.076 | n/a | 0.000 |
| unclassified.ordinal_93.qk_rope | qk_rope | 1 | 20 | 0.000 | 0.076 | n/a | 0.000 |
| value.ownership_conv | qk_rope | 1 | 20 | 0.000 | 0.076 | n/a | 0.000 |
| unclassified.ordinal_355.qk_rope | qk_rope | 1 | 20 | 0.000 | 0.076 | n/a | 0.000 |
| unclassified.ordinal_69.qk_rope | qk_rope | 1 | 20 | 0.000 | 0.076 | n/a | 0.000 |
| unclassified.ordinal_253.qk_rope | qk_rope | 1 | 20 | 0.000 | 0.076 | n/a | 0.000 |
| unclassified.ordinal_307.qk_rope | qk_rope | 1 | 20 | 0.000 | 0.076 | n/a | 0.000 |
| unclassified.ordinal_127.qk_rope | qk_rope | 1 | 20 | 0.000 | 0.076 | n/a | 0.000 |
| unclassified.ordinal_49.qk_rope | qk_rope | 1 | 20 | 0.000 | 0.076 | n/a | 0.000 |
| unclassified.ordinal_59.qk_rope | qk_rope | 1 | 20 | 0.000 | 0.076 | n/a | 0.000 |
| unclassified.ordinal_205.qk_rope | qk_rope | 1 | 20 | 0.000 | 0.076 | n/a | 0.000 |
| unclassified.ordinal_137.qk_rope | qk_rope | 1 | 20 | 0.000 | 0.076 | n/a | 0.000 |
| unclassified.ordinal_405.library_gemm | library_gemm | 1 | 20 | 0.000 | 0.070 | n/a | 0.000 |
| unclassified.ordinal_403.library_gemm | library_gemm | 1 | 20 | 0.000 | 0.070 | n/a | 0.000 |
| unclassified.ordinal_177.affine_silu | affine_silu | 1 | 20 | 0.000 | 0.067 | n/a | 0.000 |
| unclassified.ordinal_279.affine_silu | affine_silu | 1 | 20 | 0.000 | 0.067 | n/a | 0.000 |
| unclassified.ordinal_381.affine_silu | affine_silu | 1 | 20 | 0.000 | 0.067 | n/a | 0.000 |
| unclassified.ordinal_245.affine_silu | affine_silu | 1 | 20 | 0.000 | 0.067 | n/a | 0.000 |
| unclassified.ordinal_313.affine_silu | affine_silu | 1 | 20 | 0.000 | 0.067 | n/a | 0.000 |
| unclassified.ordinal_109.affine_silu | affine_silu | 1 | 20 | 0.000 | 0.066 | n/a | 0.000 |
| unclassified.ordinal_75.affine_silu | affine_silu | 1 | 20 | 0.000 | 0.066 | n/a | 0.000 |
| unclassified.ordinal_347.affine_silu | affine_silu | 1 | 20 | 0.000 | 0.066 | n/a | 0.000 |
| unclassified.ordinal_41.affine_silu | affine_silu | 1 | 20 | 0.000 | 0.066 | n/a | 0.000 |
| unclassified.ordinal_211.affine_silu | affine_silu | 1 | 20 | 0.000 | 0.066 | n/a | 0.000 |
| unclassified.ordinal_143.affine_silu | affine_silu | 1 | 20 | 0.000 | 0.066 | n/a | 0.000 |
| unclassified.ordinal_400.head_elementwise | head_elementwise | 1 | 20 | 0.000 | 0.064 | n/a | 0.000 |
| unclassified.ordinal_398.head_elementwise | head_elementwise | 1 | 20 | 0.000 | 0.063 | n/a | 0.000 |
| frontend.initial_global_matmul | library_gemm | 1 | 20 | 0.000 | 0.053 | n/a | 0.000 |
| unclassified.ordinal_303.rmsnorm | rmsnorm | 1 | 20 | 0.000 | 0.049 | n/a | 0.000 |
| unclassified.ordinal_269.rmsnorm | rmsnorm | 1 | 20 | 0.000 | 0.049 | n/a | 0.000 |
| unclassified.ordinal_181.rmsnorm | rmsnorm | 1 | 20 | 0.000 | 0.049 | n/a | 0.000 |
| unclassified.ordinal_222.rmsnorm | rmsnorm | 1 | 20 | 0.000 | 0.049 | n/a | 0.000 |
| unclassified.ordinal_249.rmsnorm | rmsnorm | 1 | 20 | 0.000 | 0.049 | n/a | 0.000 |
| unclassified.ordinal_266.rmsnorm | rmsnorm | 1 | 20 | 0.000 | 0.049 | n/a | 0.000 |
| unclassified.ordinal_89.rmsnorm | rmsnorm | 1 | 20 | 0.000 | 0.049 | n/a | 0.000 |
| unclassified.ordinal_361.rmsnorm | rmsnorm | 1 | 20 | 0.000 | 0.049 | n/a | 0.000 |
| unclassified.ordinal_283.rmsnorm | rmsnorm | 1 | 20 | 0.000 | 0.049 | n/a | 0.000 |
| unclassified.ordinal_21.rmsnorm | rmsnorm | 1 | 20 | 0.000 | 0.049 | n/a | 0.000 |
| unclassified.ordinal_18.rmsnorm | rmsnorm | 1 | 20 | 0.000 | 0.049 | n/a | 0.000 |
| unclassified.ordinal_167.rmsnorm | rmsnorm | 1 | 20 | 0.000 | 0.049 | n/a | 0.000 |
| unclassified.ordinal_293.rmsnorm | rmsnorm | 1 | 20 | 0.000 | 0.049 | n/a | 0.000 |
| unclassified.ordinal_45.rmsnorm | rmsnorm | 1 | 20 | 0.000 | 0.049 | n/a | 0.000 |
| unclassified.ordinal_225.rmsnorm | rmsnorm | 1 | 20 | 0.000 | 0.049 | n/a | 0.000 |
| unclassified.ordinal_378.rmsnorm | rmsnorm | 1 | 20 | 0.000 | 0.049 | n/a | 0.000 |
| unclassified.ordinal_351.rmsnorm | rmsnorm | 1 | 20 | 0.000 | 0.049 | n/a | 0.000 |
| unclassified.ordinal_290.rmsnorm | rmsnorm | 1 | 20 | 0.000 | 0.049 | n/a | 0.000 |
| unclassified.ordinal_191.rmsnorm | rmsnorm | 1 | 20 | 0.000 | 0.049 | n/a | 0.000 |
| unclassified.ordinal_147.rmsnorm | rmsnorm | 1 | 20 | 0.000 | 0.049 | n/a | 0.000 |
| unclassified.ordinal_79.rmsnorm | rmsnorm | 1 | 20 | 0.000 | 0.049 | n/a | 0.000 |
| unclassified.ordinal_28.rmsnorm | rmsnorm | 1 | 20 | 0.000 | 0.049 | n/a | 0.000 |
| policy.p1_half_to_float | rmsnorm | 1 | 20 | 0.000 | 0.049 | n/a | 0.000 |
| unclassified.ordinal_55.rmsnorm | rmsnorm | 1 | 20 | 0.000 | 0.048 | n/a | 0.000 |
| unclassified.ordinal_38.rmsnorm | rmsnorm | 1 | 20 | 0.000 | 0.048 | n/a | 0.000 |
| unclassified.ordinal_65.rmsnorm | rmsnorm | 1 | 20 | 0.000 | 0.048 | n/a | 0.000 |
| unclassified.ordinal_201.rmsnorm | rmsnorm | 1 | 20 | 0.000 | 0.048 | n/a | 0.000 |
| unclassified.ordinal_215.rmsnorm | rmsnorm | 1 | 20 | 0.000 | 0.048 | n/a | 0.000 |
| unclassified.ordinal_164.rmsnorm | rmsnorm | 1 | 20 | 0.000 | 0.048 | n/a | 0.000 |
| unclassified.ordinal_140.rmsnorm | rmsnorm | 1 | 20 | 0.000 | 0.048 | n/a | 0.000 |
| value.v1_global_pool | rmsnorm | 1 | 20 | 0.000 | 0.048 | n/a | 0.000 |
| unclassified.ordinal_300.rmsnorm | rmsnorm | 1 | 20 | 0.000 | 0.048 | n/a | 0.000 |
| unclassified.ordinal_96.rmsnorm | rmsnorm | 1 | 20 | 0.000 | 0.048 | n/a | 0.000 |
| unclassified.ordinal_235.rmsnorm | rmsnorm | 1 | 20 | 0.000 | 0.048 | n/a | 0.000 |
| unclassified.ordinal_371.rmsnorm | rmsnorm | 1 | 20 | 0.000 | 0.048 | n/a | 0.000 |
| unclassified.ordinal_242.rmsnorm | rmsnorm | 1 | 20 | 0.000 | 0.048 | n/a | 0.000 |
| unclassified.ordinal_368.rmsnorm | rmsnorm | 1 | 20 | 0.000 | 0.048 | n/a | 0.000 |
| unclassified.ordinal_86.rmsnorm | rmsnorm | 1 | 20 | 0.000 | 0.048 | n/a | 0.000 |
| unclassified.ordinal_113.rmsnorm | rmsnorm | 1 | 20 | 0.000 | 0.048 | n/a | 0.000 |
| unclassified.ordinal_174.rmsnorm | rmsnorm | 1 | 20 | 0.000 | 0.048 | n/a | 0.000 |
| unclassified.ordinal_99.rmsnorm | rmsnorm | 1 | 20 | 0.000 | 0.048 | n/a | 0.000 |
| policy.p2_conv | rmsnorm | 1 | 20 | 0.000 | 0.048 | n/a | 0.000 |
| unclassified.ordinal_310.rmsnorm | rmsnorm | 1 | 20 | 0.000 | 0.048 | n/a | 0.000 |
| unclassified.ordinal_344.rmsnorm | rmsnorm | 1 | 20 | 0.000 | 0.048 | n/a | 0.000 |
| unclassified.ordinal_157.rmsnorm | rmsnorm | 1 | 20 | 0.000 | 0.048 | n/a | 0.000 |
| unclassified.ordinal_256.rmsnorm | rmsnorm | 1 | 20 | 0.000 | 0.048 | n/a | 0.000 |
| unclassified.ordinal_276.rmsnorm | rmsnorm | 1 | 20 | 0.000 | 0.048 | n/a | 0.000 |
| unclassified.ordinal_72.rmsnorm | rmsnorm | 1 | 20 | 0.000 | 0.048 | n/a | 0.000 |
| unclassified.ordinal_130.rmsnorm | rmsnorm | 1 | 20 | 0.000 | 0.048 | n/a | 0.000 |
| unclassified.ordinal_198.rmsnorm | rmsnorm | 1 | 20 | 0.000 | 0.048 | n/a | 0.000 |
| unclassified.ordinal_232.rmsnorm | rmsnorm | 1 | 20 | 0.000 | 0.048 | n/a | 0.000 |
| unclassified.ordinal_123.rmsnorm | rmsnorm | 1 | 20 | 0.000 | 0.048 | n/a | 0.000 |
| unclassified.ordinal_188.rmsnorm | rmsnorm | 1 | 20 | 0.000 | 0.048 | n/a | 0.000 |
| unclassified.ordinal_259.rmsnorm | rmsnorm | 1 | 20 | 0.000 | 0.048 | n/a | 0.000 |
| unclassified.ordinal_11.rmsnorm | rmsnorm | 1 | 20 | 0.000 | 0.048 | n/a | 0.000 |
| unclassified.ordinal_31.rmsnorm | rmsnorm | 1 | 20 | 0.000 | 0.048 | n/a | 0.000 |
| unclassified.ordinal_62.rmsnorm | rmsnorm | 1 | 20 | 0.000 | 0.048 | n/a | 0.000 |
| unclassified.ordinal_52.rmsnorm | rmsnorm | 1 | 20 | 0.000 | 0.048 | n/a | 0.000 |
| trunk.tip_norm_silu | rmsnorm | 1 | 20 | 0.000 | 0.048 | n/a | 0.000 |
| unclassified.ordinal_106.rmsnorm | rmsnorm | 1 | 20 | 0.000 | 0.048 | n/a | 0.000 |
| unclassified.ordinal_154.rmsnorm | rmsnorm | 1 | 20 | 0.000 | 0.048 | n/a | 0.000 |
| unclassified.ordinal_120.rmsnorm | rmsnorm | 1 | 20 | 0.000 | 0.048 | n/a | 0.000 |
| unclassified.ordinal_358.rmsnorm | rmsnorm | 1 | 20 | 0.000 | 0.048 | n/a | 0.000 |
| value.v3_matmul | rmsnorm | 1 | 20 | 0.000 | 0.048 | n/a | 0.000 |
| unclassified.ordinal_133.rmsnorm | rmsnorm | 1 | 20 | 0.000 | 0.047 | n/a | 0.000 |
| unclassified.ordinal_208.rmsnorm | rmsnorm | 1 | 20 | 0.000 | 0.047 | n/a | 0.000 |
| unclassified.ordinal_396.library_gemm | library_gemm | 1 | 20 | 0.000 | 0.046 | n/a | 0.000 |
| unclassified.ordinal_392.head_elementwise | head_elementwise | 1 | 20 | 0.000 | 0.043 | n/a | 0.000 |
| unclassified.ordinal_399.copy_reformat | copy_reformat | 1 | 20 | 0.000 | 0.043 | n/a | 0.000 |
| unclassified.ordinal_386.head_elementwise | head_elementwise | 1 | 20 | 0.000 | 0.042 | n/a | 0.000 |
| unclassified.ordinal_391.head_elementwise | head_elementwise | 1 | 20 | 0.000 | 0.035 | n/a | 0.000 |
| input.mask_sum | sumChannelsNCHWKernel | 1 | 20 | 0.000 | 0.034 | n/a | 0.000 |
| unclassified.ordinal_387.copy_reformat | copy_reformat | 1 | 20 | 0.000 | 0.031 | n/a | 0.000 |
| frontend.initial_conv_nhwc_padding_1 | cudnn | 1 | 20 | 0.000 | 0.030 | n/a | 0.000 |
| unclassified.ordinal_390.copy_reformat | copy_reformat | 1 | 20 | 0.000 | 0.030 | n/a | 0.000 |
| unclassified.ordinal_408.library_gemm | library_gemm | 1 | 20 | 0.000 | 0.027 | n/a | 0.000 |
| frontend.initial_conv_nhwc_padding_0 | cudnn | 1 | 20 | 0.000 | 0.026 | n/a | 0.000 |
| frontend.initial_global_matmul_splitk_reduce | library_gemm | 1 | 20 | 0.000 | 0.025 | n/a | 0.000 |
| input.extract_mask | head_elementwise | 1 | 20 | 0.000 | 0.024 | n/a | 0.000 |
| unclassified.ordinal_395.head_elementwise | head_elementwise | 1 | 20 | 0.000 | 0.021 | n/a | 0.000 |
| unclassified.ordinal_402.head_elementwise | head_elementwise | 1 | 20 | 0.000 | 0.020 | n/a | 0.000 |
| unclassified.ordinal_404.head_elementwise | head_elementwise | 1 | 20 | 0.000 | 0.019 | n/a | 0.000 |
| unclassified.ordinal_409.copy_reformat | copy_reformat | 1 | 20 | 0.000 | 0.019 | n/a | 0.000 |
| unclassified.ordinal_406.head_elementwise | head_elementwise | 1 | 20 | 0.000 | 0.019 | n/a | 0.000 |
| input.mask_half_to_float | copy_reformat | 1 | 20 | 0.000 | 0.019 | n/a | 0.000 |

## `library_gemm` logical breakdown

| logical group | positions | calls | isolated reference ms | S2 total ms | total ratio | excess ms |
|---|---:|---:|---:|---:|---:|---:|
| unclassified.ordinal_280.library_gemm | 1 | 20 | 0.000 | 0.288 | n/a | 0.000 |
| unclassified.ordinal_178.library_gemm | 1 | 20 | 0.000 | 0.288 | n/a | 0.000 |
| unclassified.ordinal_246.library_gemm | 1 | 20 | 0.000 | 0.287 | n/a | 0.000 |
| unclassified.ordinal_110.library_gemm | 1 | 20 | 0.000 | 0.286 | n/a | 0.000 |
| unclassified.ordinal_314.library_gemm | 1 | 20 | 0.000 | 0.286 | n/a | 0.000 |
| unclassified.ordinal_382.library_gemm | 1 | 20 | 0.000 | 0.285 | n/a | 0.000 |
| unclassified.ordinal_76.library_gemm | 1 | 20 | 0.000 | 0.284 | n/a | 0.000 |
| unclassified.ordinal_212.library_gemm | 1 | 20 | 0.000 | 0.284 | n/a | 0.000 |
| unclassified.ordinal_42.library_gemm | 1 | 20 | 0.000 | 0.283 | n/a | 0.000 |
| unclassified.ordinal_348.library_gemm | 1 | 20 | 0.000 | 0.282 | n/a | 0.000 |
| unclassified.ordinal_144.library_gemm | 1 | 20 | 0.000 | 0.281 | n/a | 0.000 |
| unclassified.ordinal_282.library_gemm | 1 | 20 | 0.000 | 0.238 | n/a | 0.000 |
| unclassified.ordinal_180.library_gemm | 1 | 20 | 0.000 | 0.238 | n/a | 0.000 |
| unclassified.ordinal_248.library_gemm | 1 | 20 | 0.000 | 0.237 | n/a | 0.000 |
| unclassified.ordinal_316.library_gemm | 1 | 20 | 0.000 | 0.237 | n/a | 0.000 |
| unclassified.ordinal_44.library_gemm | 1 | 20 | 0.000 | 0.236 | n/a | 0.000 |
| unclassified.ordinal_112.library_gemm | 1 | 20 | 0.000 | 0.236 | n/a | 0.000 |
| unclassified.ordinal_214.library_gemm | 1 | 20 | 0.000 | 0.235 | n/a | 0.000 |
| unclassified.ordinal_78.library_gemm | 1 | 20 | 0.000 | 0.235 | n/a | 0.000 |
| unclassified.ordinal_146.library_gemm | 1 | 20 | 0.000 | 0.234 | n/a | 0.000 |
| unclassified.ordinal_350.library_gemm | 1 | 20 | 0.000 | 0.233 | n/a | 0.000 |
| unclassified.ordinal_10.library_gemm | 1 | 20 | 0.000 | 0.232 | n/a | 0.000 |
| unclassified.ordinal_401.library_gemm | 1 | 20 | 0.000 | 0.190 | n/a | 0.000 |
| unclassified.ordinal_197.library_gemm | 1 | 20 | 0.000 | 0.170 | n/a | 0.000 |
| unclassified.ordinal_187.library_gemm | 1 | 20 | 0.000 | 0.170 | n/a | 0.000 |
| unclassified.ordinal_265.library_gemm | 1 | 20 | 0.000 | 0.169 | n/a | 0.000 |
| unclassified.ordinal_275.library_gemm | 1 | 20 | 0.000 | 0.169 | n/a | 0.000 |
| unclassified.ordinal_95.library_gemm | 1 | 20 | 0.000 | 0.169 | n/a | 0.000 |
| unclassified.ordinal_119.library_gemm | 1 | 20 | 0.000 | 0.169 | n/a | 0.000 |
| unclassified.ordinal_61.library_gemm | 1 | 20 | 0.000 | 0.168 | n/a | 0.000 |
| unclassified.ordinal_163.library_gemm | 1 | 20 | 0.000 | 0.168 | n/a | 0.000 |
| unclassified.ordinal_241.library_gemm | 1 | 20 | 0.000 | 0.168 | n/a | 0.000 |
| unclassified.ordinal_129.library_gemm | 1 | 20 | 0.000 | 0.168 | n/a | 0.000 |
| unclassified.ordinal_367.library_gemm | 1 | 20 | 0.000 | 0.168 | n/a | 0.000 |
| unclassified.ordinal_299.library_gemm | 1 | 20 | 0.000 | 0.168 | n/a | 0.000 |
| unclassified.ordinal_27.library_gemm | 1 | 20 | 0.000 | 0.168 | n/a | 0.000 |
| unclassified.ordinal_289.library_gemm | 1 | 20 | 0.000 | 0.168 | n/a | 0.000 |
| unclassified.ordinal_377.library_gemm | 1 | 20 | 0.000 | 0.168 | n/a | 0.000 |
| unclassified.ordinal_51.library_gemm | 1 | 20 | 0.000 | 0.168 | n/a | 0.000 |
| policy.gpool_to_bias_matmul | 1 | 20 | 0.000 | 0.168 | n/a | 0.000 |
| unclassified.ordinal_71.library_gemm | 1 | 20 | 0.000 | 0.168 | n/a | 0.000 |
| unclassified.ordinal_173.library_gemm | 1 | 20 | 0.000 | 0.168 | n/a | 0.000 |
| unclassified.ordinal_309.library_gemm | 1 | 20 | 0.000 | 0.167 | n/a | 0.000 |
| unclassified.ordinal_221.library_gemm | 1 | 20 | 0.000 | 0.167 | n/a | 0.000 |
| unclassified.ordinal_357.library_gemm | 1 | 20 | 0.000 | 0.167 | n/a | 0.000 |
| unclassified.ordinal_255.library_gemm | 1 | 20 | 0.000 | 0.167 | n/a | 0.000 |
| value.v1_half_to_float | 1 | 20 | 0.000 | 0.167 | n/a | 0.000 |
| unclassified.ordinal_37.library_gemm | 1 | 20 | 0.000 | 0.167 | n/a | 0.000 |
| unclassified.ordinal_139.library_gemm | 1 | 20 | 0.000 | 0.167 | n/a | 0.000 |
| unclassified.ordinal_231.library_gemm | 1 | 20 | 0.000 | 0.166 | n/a | 0.000 |
| unclassified.ordinal_105.library_gemm | 1 | 20 | 0.000 | 0.166 | n/a | 0.000 |
| unclassified.ordinal_207.library_gemm | 1 | 20 | 0.000 | 0.166 | n/a | 0.000 |
| unclassified.ordinal_17.library_gemm | 1 | 20 | 0.000 | 0.166 | n/a | 0.000 |
| unclassified.ordinal_85.library_gemm | 1 | 20 | 0.000 | 0.166 | n/a | 0.000 |
| unclassified.ordinal_153.library_gemm | 1 | 20 | 0.000 | 0.165 | n/a | 0.000 |
| value.ownership_half_to_float | 1 | 20 | 0.000 | 0.165 | n/a | 0.000 |
| unclassified.ordinal_397.library_gemm | 1 | 20 | 0.000 | 0.158 | n/a | 0.000 |
| unclassified.ordinal_260.library_gemm | 1 | 20 | 0.000 | 0.151 | n/a | 0.000 |
| unclassified.ordinal_22.library_gemm | 1 | 20 | 0.000 | 0.151 | n/a | 0.000 |
| unclassified.ordinal_270.library_gemm | 1 | 20 | 0.000 | 0.150 | n/a | 0.000 |
| unclassified.ordinal_304.library_gemm | 1 | 20 | 0.000 | 0.150 | n/a | 0.000 |
| unclassified.ordinal_372.library_gemm | 1 | 20 | 0.000 | 0.150 | n/a | 0.000 |
| unclassified.ordinal_294.library_gemm | 1 | 20 | 0.000 | 0.150 | n/a | 0.000 |
| unclassified.ordinal_158.library_gemm | 1 | 20 | 0.000 | 0.150 | n/a | 0.000 |
| unclassified.ordinal_226.library_gemm | 1 | 20 | 0.000 | 0.149 | n/a | 0.000 |
| unclassified.ordinal_32.library_gemm | 1 | 20 | 0.000 | 0.149 | n/a | 0.000 |
| unclassified.ordinal_236.library_gemm | 1 | 20 | 0.000 | 0.149 | n/a | 0.000 |
| unclassified.ordinal_168.library_gemm | 1 | 20 | 0.000 | 0.149 | n/a | 0.000 |
| unclassified.ordinal_261.library_gemm | 1 | 20 | 0.000 | 0.149 | n/a | 0.000 |
| unclassified.ordinal_24.library_gemm | 1 | 20 | 0.000 | 0.149 | n/a | 0.000 |
| unclassified.ordinal_193.library_gemm | 1 | 20 | 0.000 | 0.149 | n/a | 0.000 |
| unclassified.ordinal_23.library_gemm | 1 | 20 | 0.000 | 0.149 | n/a | 0.000 |
| unclassified.ordinal_192.library_gemm | 1 | 20 | 0.000 | 0.149 | n/a | 0.000 |
| unclassified.ordinal_306.library_gemm | 1 | 20 | 0.000 | 0.149 | n/a | 0.000 |
| unclassified.ordinal_100.library_gemm | 1 | 20 | 0.000 | 0.149 | n/a | 0.000 |
| unclassified.ordinal_362.library_gemm | 1 | 20 | 0.000 | 0.149 | n/a | 0.000 |
| unclassified.ordinal_202.library_gemm | 1 | 20 | 0.000 | 0.148 | n/a | 0.000 |
| unclassified.ordinal_305.library_gemm | 1 | 20 | 0.000 | 0.148 | n/a | 0.000 |
| unclassified.ordinal_33.library_gemm | 1 | 20 | 0.000 | 0.148 | n/a | 0.000 |
| unclassified.ordinal_66.library_gemm | 1 | 20 | 0.000 | 0.148 | n/a | 0.000 |
| unclassified.ordinal_284.library_gemm | 1 | 20 | 0.000 | 0.148 | n/a | 0.000 |
| unclassified.ordinal_262.library_gemm | 1 | 20 | 0.000 | 0.148 | n/a | 0.000 |
| unclassified.ordinal_91.library_gemm | 1 | 20 | 0.000 | 0.148 | n/a | 0.000 |
| unclassified.ordinal_56.library_gemm | 1 | 20 | 0.000 | 0.148 | n/a | 0.000 |
| unclassified.ordinal_227.library_gemm | 1 | 20 | 0.000 | 0.148 | n/a | 0.000 |
| policy.p1_conv | 1 | 20 | 0.000 | 0.148 | n/a | 0.000 |
| unclassified.ordinal_159.library_gemm | 1 | 20 | 0.000 | 0.148 | n/a | 0.000 |
| policy.gpool_to_pass_matmul | 1 | 20 | 0.000 | 0.148 | n/a | 0.000 |
| unclassified.ordinal_90.library_gemm | 1 | 20 | 0.000 | 0.148 | n/a | 0.000 |
| unclassified.ordinal_251.library_gemm | 1 | 20 | 0.000 | 0.148 | n/a | 0.000 |
| unclassified.ordinal_67.library_gemm | 1 | 20 | 0.000 | 0.148 | n/a | 0.000 |
| unclassified.ordinal_250.library_gemm | 1 | 20 | 0.000 | 0.147 | n/a | 0.000 |
| unclassified.ordinal_124.library_gemm | 1 | 20 | 0.000 | 0.147 | n/a | 0.000 |
| unclassified.ordinal_272.library_gemm | 1 | 20 | 0.000 | 0.147 | n/a | 0.000 |
| unclassified.ordinal_203.library_gemm | 1 | 20 | 0.000 | 0.147 | n/a | 0.000 |
| unclassified.ordinal_363.library_gemm | 1 | 20 | 0.000 | 0.147 | n/a | 0.000 |
| unclassified.ordinal_116.library_gemm | 1 | 20 | 0.000 | 0.147 | n/a | 0.000 |
| unclassified.ordinal_170.library_gemm | 1 | 20 | 0.000 | 0.147 | n/a | 0.000 |
| unclassified.ordinal_114.library_gemm | 1 | 20 | 0.000 | 0.147 | n/a | 0.000 |
| unclassified.ordinal_295.library_gemm | 1 | 20 | 0.000 | 0.147 | n/a | 0.000 |
| unclassified.ordinal_286.library_gemm | 1 | 20 | 0.000 | 0.147 | n/a | 0.000 |
| unclassified.ordinal_169.library_gemm | 1 | 20 | 0.000 | 0.147 | n/a | 0.000 |
| unclassified.ordinal_81.library_gemm | 1 | 20 | 0.000 | 0.147 | n/a | 0.000 |
| unclassified.ordinal_101.library_gemm | 1 | 20 | 0.000 | 0.147 | n/a | 0.000 |
| unclassified.ordinal_216.library_gemm | 1 | 20 | 0.000 | 0.147 | n/a | 0.000 |
| value.v3_bias | 1 | 20 | 0.000 | 0.147 | n/a | 0.000 |
| policy.pass_bias_silu | 1 | 20 | 0.000 | 0.147 | n/a | 0.000 |
| unclassified.ordinal_13.library_gemm | 1 | 20 | 0.000 | 0.147 | n/a | 0.000 |
| unclassified.ordinal_374.library_gemm | 1 | 20 | 0.000 | 0.147 | n/a | 0.000 |
| unclassified.ordinal_46.library_gemm | 1 | 20 | 0.000 | 0.147 | n/a | 0.000 |
| unclassified.ordinal_48.library_gemm | 1 | 20 | 0.000 | 0.147 | n/a | 0.000 |
| unclassified.ordinal_373.library_gemm | 1 | 20 | 0.000 | 0.147 | n/a | 0.000 |
| unclassified.ordinal_126.library_gemm | 1 | 20 | 0.000 | 0.147 | n/a | 0.000 |
| unclassified.ordinal_34.library_gemm | 1 | 20 | 0.000 | 0.147 | n/a | 0.000 |
| unclassified.ordinal_271.library_gemm | 1 | 20 | 0.000 | 0.147 | n/a | 0.000 |
| unclassified.ordinal_238.library_gemm | 1 | 20 | 0.000 | 0.147 | n/a | 0.000 |
| unclassified.ordinal_364.library_gemm | 1 | 20 | 0.000 | 0.147 | n/a | 0.000 |
| unclassified.ordinal_58.library_gemm | 1 | 20 | 0.000 | 0.147 | n/a | 0.000 |
| value.score_matmul | 1 | 20 | 0.000 | 0.146 | n/a | 0.000 |
| unclassified.ordinal_228.library_gemm | 1 | 20 | 0.000 | 0.146 | n/a | 0.000 |
| unclassified.ordinal_285.library_gemm | 1 | 20 | 0.000 | 0.146 | n/a | 0.000 |
| unclassified.ordinal_57.library_gemm | 1 | 20 | 0.000 | 0.146 | n/a | 0.000 |
| unclassified.ordinal_134.library_gemm | 1 | 20 | 0.000 | 0.146 | n/a | 0.000 |
| unclassified.ordinal_125.library_gemm | 1 | 20 | 0.000 | 0.146 | n/a | 0.000 |
| unclassified.ordinal_252.library_gemm | 1 | 20 | 0.000 | 0.146 | n/a | 0.000 |
| policy.g1_conv | 1 | 20 | 0.000 | 0.146 | n/a | 0.000 |
| unclassified.ordinal_182.library_gemm | 1 | 20 | 0.000 | 0.146 | n/a | 0.000 |
| policy.g1_norm_silu | 1 | 20 | 0.000 | 0.146 | n/a | 0.000 |
| unclassified.ordinal_204.library_gemm | 1 | 20 | 0.000 | 0.146 | n/a | 0.000 |
| unclassified.ordinal_296.library_gemm | 1 | 20 | 0.000 | 0.146 | n/a | 0.000 |
| unclassified.ordinal_354.library_gemm | 1 | 20 | 0.000 | 0.146 | n/a | 0.000 |
| unclassified.ordinal_160.library_gemm | 1 | 20 | 0.000 | 0.146 | n/a | 0.000 |
| unclassified.ordinal_237.library_gemm | 1 | 20 | 0.000 | 0.146 | n/a | 0.000 |
| unclassified.ordinal_217.library_gemm | 1 | 20 | 0.000 | 0.146 | n/a | 0.000 |
| unclassified.ordinal_194.library_gemm | 1 | 20 | 0.000 | 0.146 | n/a | 0.000 |
| unclassified.ordinal_68.library_gemm | 1 | 20 | 0.000 | 0.146 | n/a | 0.000 |
| unclassified.ordinal_183.library_gemm | 1 | 20 | 0.000 | 0.146 | n/a | 0.000 |
| unclassified.ordinal_12.library_gemm | 1 | 20 | 0.000 | 0.146 | n/a | 0.000 |
| unclassified.ordinal_102.library_gemm | 1 | 20 | 0.000 | 0.145 | n/a | 0.000 |
| unclassified.ordinal_80.library_gemm | 1 | 20 | 0.000 | 0.145 | n/a | 0.000 |
| unclassified.ordinal_184.library_gemm | 1 | 20 | 0.000 | 0.145 | n/a | 0.000 |
| unclassified.ordinal_47.library_gemm | 1 | 20 | 0.000 | 0.145 | n/a | 0.000 |
| unclassified.ordinal_115.library_gemm | 1 | 20 | 0.000 | 0.145 | n/a | 0.000 |
| unclassified.ordinal_150.library_gemm | 1 | 20 | 0.000 | 0.145 | n/a | 0.000 |
| unclassified.ordinal_218.library_gemm | 1 | 20 | 0.000 | 0.145 | n/a | 0.000 |
| unclassified.ordinal_135.library_gemm | 1 | 20 | 0.000 | 0.145 | n/a | 0.000 |
| unclassified.ordinal_82.library_gemm | 1 | 20 | 0.000 | 0.145 | n/a | 0.000 |
| unclassified.ordinal_148.library_gemm | 1 | 20 | 0.000 | 0.145 | n/a | 0.000 |
| unclassified.ordinal_92.library_gemm | 1 | 20 | 0.000 | 0.145 | n/a | 0.000 |
| unclassified.ordinal_149.library_gemm | 1 | 20 | 0.000 | 0.145 | n/a | 0.000 |
| unclassified.ordinal_136.library_gemm | 1 | 20 | 0.000 | 0.145 | n/a | 0.000 |
| unclassified.ordinal_14.library_gemm | 1 | 20 | 0.000 | 0.145 | n/a | 0.000 |
| unclassified.ordinal_353.library_gemm | 1 | 20 | 0.000 | 0.145 | n/a | 0.000 |
| policy.gpool_to_pass_matmul2 | 1 | 20 | 0.000 | 0.145 | n/a | 0.000 |
| value.score_bias | 1 | 20 | 0.000 | 0.145 | n/a | 0.000 |
| unclassified.ordinal_352.library_gemm | 1 | 20 | 0.000 | 0.144 | n/a | 0.000 |
| unclassified.ordinal_384.library_gemm | 1 | 20 | 0.000 | 0.123 | n/a | 0.000 |
| unclassified.ordinal_385.library_gemm | 1 | 20 | 0.000 | 0.117 | n/a | 0.000 |
| unclassified.ordinal_389.library_gemm | 1 | 20 | 0.000 | 0.107 | n/a | 0.000 |
| unclassified.ordinal_394.library_gemm | 1 | 20 | 0.000 | 0.106 | n/a | 0.000 |
| unclassified.ordinal_407.library_gemm | 1 | 20 | 0.000 | 0.081 | n/a | 0.000 |
| unclassified.ordinal_393.library_gemm | 1 | 20 | 0.000 | 0.078 | n/a | 0.000 |
| unclassified.ordinal_405.library_gemm | 1 | 20 | 0.000 | 0.070 | n/a | 0.000 |
| unclassified.ordinal_403.library_gemm | 1 | 20 | 0.000 | 0.070 | n/a | 0.000 |
| frontend.initial_global_matmul | 1 | 20 | 0.000 | 0.053 | n/a | 0.000 |
| unclassified.ordinal_396.library_gemm | 1 | 20 | 0.000 | 0.046 | n/a | 0.000 |
| unclassified.ordinal_408.library_gemm | 1 | 20 | 0.000 | 0.027 | n/a | 0.000 |
| frontend.initial_global_matmul_splitk_reduce | 1 | 20 | 0.000 | 0.025 | n/a | 0.000 |

## Top ordinal hotspots by summed excess

The worst peer is the highest median S2/S1 slowdown among peer families observed at least four times for that ordinal.

| rank | ordinal | logical position | family | calls | isolated us | S2 us | S2/S1 | excess ms | common peer | worst peer |
|---:|---:|---|---|---:|---:|---:|---:|---:|---|---|
| 1 | 189 | unclassified.ordinal_189.fused_ffn | fused_ffn | 20 | n/a | 40.544 | n/a | 0.000 | idle (20) | n/a |
| 2 | 97 | unclassified.ordinal_97.fused_ffn | fused_ffn | 20 | n/a | 40.481 | n/a | 0.000 | idle (20) | n/a |
| 3 | 141 | unclassified.ordinal_141.fused_ffn | fused_ffn | 20 | n/a | 40.096 | n/a | 0.000 | idle (20) | n/a |
| 4 | 301 | unclassified.ordinal_301.fused_ffn | fused_ffn | 20 | n/a | 40.065 | n/a | 0.000 | idle (20) | n/a |
| 5 | 223 | unclassified.ordinal_223.fused_ffn | fused_ffn | 20 | n/a | 39.920 | n/a | 0.000 | idle (20) | n/a |
| 6 | 291 | unclassified.ordinal_291.fused_ffn | fused_ffn | 20 | n/a | 39.905 | n/a | 0.000 | idle (20) | n/a |
| 7 | 379 | unclassified.ordinal_379.fused_ffn | fused_ffn | 20 | n/a | 39.776 | n/a | 0.000 | idle (20) | n/a |
| 8 | 311 | unclassified.ordinal_311.fused_ffn | fused_ffn | 20 | n/a | 39.601 | n/a | 0.000 | idle (20) | n/a |
| 9 | 199 | unclassified.ordinal_199.fused_ffn | fused_ffn | 20 | n/a | 39.728 | n/a | 0.000 | idle (20) | n/a |
| 10 | 131 | unclassified.ordinal_131.fused_ffn | fused_ffn | 20 | n/a | 39.632 | n/a | 0.000 | idle (20) | n/a |
| 11 | 87 | unclassified.ordinal_87.fused_ffn | fused_ffn | 20 | n/a | 40.081 | n/a | 0.000 | idle (20) | n/a |
| 12 | 19 | unclassified.ordinal_19.fused_ffn | fused_ffn | 20 | n/a | 39.312 | n/a | 0.000 | idle (20) | n/a |
| 13 | 345 | unclassified.ordinal_345.fused_ffn | fused_ffn | 20 | n/a | 39.665 | n/a | 0.000 | idle (20) | n/a |
| 14 | 53 | unclassified.ordinal_53.fused_ffn | fused_ffn | 20 | n/a | 38.977 | n/a | 0.000 | idle (20) | n/a |
| 15 | 335 | value.v2_matmul | fused_ffn | 20 | n/a | 39.680 | n/a | 0.000 | idle (20) | n/a |
| 16 | 325 | policy.gpool_bias_add | fused_ffn | 20 | n/a | 38.432 | n/a | 0.000 | idle (20) | n/a |
| 17 | 233 | unclassified.ordinal_233.fused_ffn | fused_ffn | 20 | n/a | 38.897 | n/a | 0.000 | idle (20) | n/a |
| 18 | 209 | unclassified.ordinal_209.fused_ffn | fused_ffn | 20 | n/a | 38.096 | n/a | 0.000 | idle (20) | n/a |
| 19 | 63 | unclassified.ordinal_63.fused_ffn | fused_ffn | 20 | n/a | 37.984 | n/a | 0.000 | idle (20) | n/a |
| 20 | 73 | unclassified.ordinal_73.fused_ffn | fused_ffn | 20 | n/a | 37.761 | n/a | 0.000 | idle (20) | n/a |
| 21 | 165 | unclassified.ordinal_165.fused_ffn | fused_ffn | 20 | n/a | 37.856 | n/a | 0.000 | idle (20) | n/a |
| 22 | 39 | unclassified.ordinal_39.fused_ffn | fused_ffn | 20 | n/a | 37.761 | n/a | 0.000 | idle (20) | n/a |
| 23 | 155 | unclassified.ordinal_155.fused_ffn | fused_ffn | 20 | n/a | 37.776 | n/a | 0.000 | idle (20) | n/a |
| 24 | 29 | unclassified.ordinal_29.fused_ffn | fused_ffn | 20 | n/a | 37.584 | n/a | 0.000 | idle (20) | n/a |
| 25 | 121 | unclassified.ordinal_121.fused_ffn | fused_ffn | 20 | n/a | 37.536 | n/a | 0.000 | idle (20) | n/a |
| 26 | 107 | unclassified.ordinal_107.fused_ffn | fused_ffn | 20 | n/a | 37.248 | n/a | 0.000 | idle (20) | n/a |
| 27 | 369 | unclassified.ordinal_369.fused_ffn | fused_ffn | 20 | n/a | 36.928 | n/a | 0.000 | idle (20) | n/a |
| 28 | 359 | unclassified.ordinal_359.fused_ffn | fused_ffn | 20 | n/a | 35.648 | n/a | 0.000 | idle (20) | n/a |
| 29 | 267 | unclassified.ordinal_267.fused_ffn | fused_ffn | 20 | n/a | 35.297 | n/a | 0.000 | idle (20) | n/a |
| 30 | 257 | unclassified.ordinal_257.fused_ffn | fused_ffn | 20 | n/a | 35.392 | n/a | 0.000 | idle (20) | n/a |
| 31 | 277 | unclassified.ordinal_277.fused_ffn | fused_ffn | 20 | n/a | 35.136 | n/a | 0.000 | idle (20) | n/a |
| 32 | 175 | unclassified.ordinal_175.fused_ffn | fused_ffn | 20 | n/a | 34.721 | n/a | 0.000 | idle (20) | n/a |
| 33 | 243 | unclassified.ordinal_243.fused_ffn | fused_ffn | 20 | n/a | 34.528 | n/a | 0.000 | idle (20) | n/a |
| 34 | 20 | unclassified.ordinal_20.linear2_residual | linear2_residual | 20 | n/a | 20.768 | n/a | 0.000 | idle (20) | n/a |
| 35 | 278 | unclassified.ordinal_278.linear2_residual | linear2_residual | 20 | n/a | 20.800 | n/a | 0.000 | idle (20) | n/a |
| 36 | 258 | unclassified.ordinal_258.linear2_residual | linear2_residual | 20 | n/a | 20.832 | n/a | 0.000 | idle (20) | n/a |
| 37 | 292 | unclassified.ordinal_292.linear2_residual | linear2_residual | 20 | n/a | 20.784 | n/a | 0.000 | idle (20) | n/a |
| 38 | 244 | unclassified.ordinal_244.linear2_residual | linear2_residual | 20 | n/a | 20.704 | n/a | 0.000 | idle (20) | n/a |
| 39 | 302 | unclassified.ordinal_302.linear2_residual | linear2_residual | 20 | n/a | 20.657 | n/a | 0.000 | idle (20) | n/a |
| 40 | 268 | unclassified.ordinal_268.linear2_residual | linear2_residual | 20 | n/a | 20.704 | n/a | 0.000 | idle (20) | n/a |

## Full fixed-forward ordinal map

| ordinal | logical position | family | resource signature | calls | isolated us | S2 us | S2/S1 | overlap | excess ms | common peer | worst peer |
|---:|---|---|---|---:|---:|---:|---:|---:|---:|---|---|
| 0 | input.extract_mask | head_elementwise | head_elementwise; extractChannel0KernelNHWC; g10x1x1; b512x1x1; r16; s0 | 20 | n/a | 1.168 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 1 | input.mask_half_to_float | copy_reformat | copy_reformat; copyFromHalfKernel; g10x1x1; b512x1x1; r16; s0 | 20 | n/a | 0.928 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 2 | input.mask_sum | sumChannelsNCHWKernel | sumChannelsNCHWKernel; sumChannelsNCHWKernel; g1x1x13; b256x2x1; r22; s2048 | 20 | n/a | 1.664 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 3 | frontend.initial_conv_nhwc_padding_0 | cudnn | cudnn; nhwcAddPaddingKernel; g340x1x1; b768x1x1; r34; s0 | 20 | n/a | 1.280 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 4 | frontend.initial_conv_nhwc_padding_1 | cudnn | cudnn; nhwcAddPaddingKernel; g340x1x1; b768x1x1; r34; s0 | 20 | n/a | 1.520 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 5 | frontend.initial_conv | cudnn | cudnn; Kernel; g296x3x1; b128x1x1; r94; s81920 | 20 | n/a | 19.424 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 6 | frontend.initial_global_matmul | library_gemm | library_gemm; Kernel2; g8x1x3; b128x1x1; r128; s24576 | 20 | n/a | 2.592 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 7 | frontend.initial_global_matmul_splitk_reduce | library_gemm | library_gemm; splitKreduce_kernel; g24x1x1; b32x16x1; r49; s0 | 20 | n/a | 1.264 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 8 | frontend.initial_global_broadcast_add | head_elementwise | head_elementwise; addNCBiasInplaceNHWCHalfKernel; g3x361x13; b256x1x1; r16; s0 | 20 | n/a | 7.712 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 9 | unclassified.ordinal_9.affine_silu | affine_silu | affine_silu; affineSiluHalf2Kernel; g4693x1x1; b384x1x1; r17; s0 | 20 | n/a | 4.928 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 10 | unclassified.ordinal_10.library_gemm | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 20 | n/a | 11.584 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 11 | unclassified.ordinal_11.rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 20 | n/a | 2.384 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 12 | unclassified.ordinal_12.library_gemm | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 20 | n/a | 7.264 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 13 | unclassified.ordinal_13.library_gemm | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 20 | n/a | 7.360 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 14 | unclassified.ordinal_14.library_gemm | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 20 | n/a | 7.232 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 15 | unclassified.ordinal_15.qk_rope | qk_rope | qk_rope; fusedQKRoPE19Half2Kernel; g361x13x1; b192x1x1; r16; s0 | 20 | n/a | 3.840 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 16 | unclassified.ordinal_16.fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 20 | n/a | 11.648 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 17 | unclassified.ordinal_17.library_gemm | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 20 | n/a | 8.288 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 18 | unclassified.ordinal_18.rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 20 | n/a | 2.432 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 19 | unclassified.ordinal_19.fused_ffn | fused_ffn | fused_ffn; fused_ffn_candidate_a_reuse_kernel; g18x37x1; b128x1x1; r136; s32768 | 20 | n/a | 39.312 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 20 | unclassified.ordinal_20.linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 20 | n/a | 20.768 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 21 | unclassified.ordinal_21.rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 20 | n/a | 2.433 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 22 | unclassified.ordinal_22.library_gemm | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 20 | n/a | 7.537 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 23 | unclassified.ordinal_23.library_gemm | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 20 | n/a | 7.424 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 24 | unclassified.ordinal_24.library_gemm | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 20 | n/a | 7.456 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 25 | unclassified.ordinal_25.qk_rope | qk_rope | qk_rope; fusedQKRoPE19Half2Kernel; g361x13x1; b192x1x1; r16; s0 | 20 | n/a | 3.888 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 26 | unclassified.ordinal_26.fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 20 | n/a | 11.872 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 27 | unclassified.ordinal_27.library_gemm | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 20 | n/a | 8.384 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 28 | unclassified.ordinal_28.rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 20 | n/a | 2.432 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 29 | unclassified.ordinal_29.fused_ffn | fused_ffn | fused_ffn; fused_ffn_candidate_a_reuse_kernel; g18x37x1; b128x1x1; r136; s32768 | 20 | n/a | 37.584 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 30 | unclassified.ordinal_30.linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 20 | n/a | 20.720 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 31 | unclassified.ordinal_31.rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 20 | n/a | 2.400 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 32 | unclassified.ordinal_32.library_gemm | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 20 | n/a | 7.488 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 33 | unclassified.ordinal_33.library_gemm | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 20 | n/a | 7.424 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 34 | unclassified.ordinal_34.library_gemm | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 20 | n/a | 7.328 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 35 | unclassified.ordinal_35.qk_rope | qk_rope | qk_rope; fusedQKRoPE19Half2Kernel; g361x13x1; b192x1x1; r16; s0 | 20 | n/a | 3.840 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 36 | unclassified.ordinal_36.fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 20 | n/a | 11.713 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 37 | unclassified.ordinal_37.library_gemm | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 20 | n/a | 8.320 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 38 | unclassified.ordinal_38.rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 20 | n/a | 2.400 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 39 | unclassified.ordinal_39.fused_ffn | fused_ffn | fused_ffn; fused_ffn_candidate_a_reuse_kernel; g18x37x1; b128x1x1; r136; s32768 | 20 | n/a | 37.761 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 40 | unclassified.ordinal_40.linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 20 | n/a | 20.480 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 41 | unclassified.ordinal_41.affine_silu | affine_silu | affine_silu; affineSiluHalf2Kernel; g4693x1x1; b192x1x1; r17; s0 | 20 | n/a | 3.296 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 42 | unclassified.ordinal_42.library_gemm | library_gemm | library_gemm; Kernel2; g148x1x1; b256x1x1; r154; s73728 | 20 | n/a | 14.176 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 43 | unclassified.ordinal_43.affine_silu | affine_silu | affine_silu; affineSiluHalf2Kernel; g4693x1x1; b384x1x1; r17; s0 | 20 | n/a | 5.088 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 44 | unclassified.ordinal_44.library_gemm | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 20 | n/a | 11.824 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 45 | unclassified.ordinal_45.rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 20 | n/a | 2.432 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 46 | unclassified.ordinal_46.library_gemm | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 20 | n/a | 7.328 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 47 | unclassified.ordinal_47.library_gemm | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 20 | n/a | 7.248 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 48 | unclassified.ordinal_48.library_gemm | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 20 | n/a | 7.344 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 49 | unclassified.ordinal_49.qk_rope | qk_rope | qk_rope; fusedQKRoPE19Half2Kernel; g361x13x1; b192x1x1; r16; s0 | 20 | n/a | 3.792 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 50 | unclassified.ordinal_50.fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 20 | n/a | 11.696 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 51 | unclassified.ordinal_51.library_gemm | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 20 | n/a | 8.368 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 52 | unclassified.ordinal_52.rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 20 | n/a | 2.400 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 53 | unclassified.ordinal_53.fused_ffn | fused_ffn | fused_ffn; fused_ffn_candidate_a_reuse_kernel; g18x37x1; b128x1x1; r136; s32768 | 20 | n/a | 38.977 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 54 | unclassified.ordinal_54.linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 20 | n/a | 20.592 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 55 | unclassified.ordinal_55.rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 20 | n/a | 2.432 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 56 | unclassified.ordinal_56.library_gemm | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 20 | n/a | 7.392 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 57 | unclassified.ordinal_57.library_gemm | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 20 | n/a | 7.296 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 58 | unclassified.ordinal_58.library_gemm | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 20 | n/a | 7.328 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 59 | unclassified.ordinal_59.qk_rope | qk_rope | qk_rope; fusedQKRoPE19Half2Kernel; g361x13x1; b192x1x1; r16; s0 | 20 | n/a | 3.776 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 60 | unclassified.ordinal_60.fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 20 | n/a | 11.713 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 61 | unclassified.ordinal_61.library_gemm | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 20 | n/a | 8.416 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 62 | unclassified.ordinal_62.rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 20 | n/a | 2.400 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 63 | unclassified.ordinal_63.fused_ffn | fused_ffn | fused_ffn; fused_ffn_candidate_a_reuse_kernel; g18x37x1; b128x1x1; r136; s32768 | 20 | n/a | 37.984 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 64 | unclassified.ordinal_64.linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 20 | n/a | 20.480 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 65 | unclassified.ordinal_65.rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 20 | n/a | 2.432 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 66 | unclassified.ordinal_66.library_gemm | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 20 | n/a | 7.392 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 67 | unclassified.ordinal_67.library_gemm | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 20 | n/a | 7.360 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 68 | unclassified.ordinal_68.library_gemm | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 20 | n/a | 7.232 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 69 | unclassified.ordinal_69.qk_rope | qk_rope | qk_rope; fusedQKRoPE19Half2Kernel; g361x13x1; b192x1x1; r16; s0 | 20 | n/a | 3.808 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 70 | unclassified.ordinal_70.fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 20 | n/a | 11.633 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 71 | unclassified.ordinal_71.library_gemm | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 20 | n/a | 8.352 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 72 | unclassified.ordinal_72.rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 20 | n/a | 2.400 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 73 | unclassified.ordinal_73.fused_ffn | fused_ffn | fused_ffn; fused_ffn_candidate_a_reuse_kernel; g18x37x1; b128x1x1; r136; s32768 | 20 | n/a | 37.761 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 74 | unclassified.ordinal_74.linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 20 | n/a | 20.497 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 75 | unclassified.ordinal_75.affine_silu | affine_silu | affine_silu; affineSiluHalf2Kernel; g4693x1x1; b192x1x1; r17; s0 | 20 | n/a | 3.296 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 76 | unclassified.ordinal_76.library_gemm | library_gemm | library_gemm; Kernel2; g148x1x1; b256x1x1; r154; s73728 | 20 | n/a | 14.144 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 77 | unclassified.ordinal_77.affine_silu | affine_silu | affine_silu; affineSiluHalf2Kernel; g4693x1x1; b384x1x1; r17; s0 | 20 | n/a | 4.976 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 78 | unclassified.ordinal_78.library_gemm | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 20 | n/a | 11.760 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 79 | unclassified.ordinal_79.rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 20 | n/a | 2.401 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 80 | unclassified.ordinal_80.library_gemm | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 20 | n/a | 7.264 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 81 | unclassified.ordinal_81.library_gemm | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 20 | n/a | 7.360 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 82 | unclassified.ordinal_82.library_gemm | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 20 | n/a | 7.248 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 83 | unclassified.ordinal_83.qk_rope | qk_rope | qk_rope; fusedQKRoPE19Half2Kernel; g361x13x1; b192x1x1; r16; s0 | 20 | n/a | 3.840 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 84 | unclassified.ordinal_84.fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 20 | n/a | 11.680 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 85 | unclassified.ordinal_85.library_gemm | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 20 | n/a | 8.256 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 86 | unclassified.ordinal_86.rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 20 | n/a | 2.432 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 87 | unclassified.ordinal_87.fused_ffn | fused_ffn | fused_ffn; fused_ffn_candidate_a_reuse_kernel; g18x37x1; b128x1x1; r136; s32768 | 20 | n/a | 40.081 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 88 | unclassified.ordinal_88.linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 20 | n/a | 20.480 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 89 | unclassified.ordinal_89.rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 20 | n/a | 2.432 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 90 | unclassified.ordinal_90.library_gemm | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 20 | n/a | 7.392 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 91 | unclassified.ordinal_91.library_gemm | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 20 | n/a | 7.392 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 92 | unclassified.ordinal_92.library_gemm | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 20 | n/a | 7.232 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 93 | unclassified.ordinal_93.qk_rope | qk_rope | qk_rope; fusedQKRoPE19Half2Kernel; g361x13x1; b192x1x1; r16; s0 | 20 | n/a | 3.808 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 94 | unclassified.ordinal_94.fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 20 | n/a | 11.680 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 95 | unclassified.ordinal_95.library_gemm | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 20 | n/a | 8.448 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 96 | unclassified.ordinal_96.rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 20 | n/a | 2.400 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 97 | unclassified.ordinal_97.fused_ffn | fused_ffn | fused_ffn; fused_ffn_candidate_a_reuse_kernel; g18x37x1; b128x1x1; r136; s32768 | 20 | n/a | 40.481 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 98 | unclassified.ordinal_98.linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 20 | n/a | 20.576 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 99 | unclassified.ordinal_99.rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 20 | n/a | 2.400 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 100 | unclassified.ordinal_100.library_gemm | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 20 | n/a | 7.424 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 101 | unclassified.ordinal_101.library_gemm | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 20 | n/a | 7.344 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 102 | unclassified.ordinal_102.library_gemm | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 20 | n/a | 7.264 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 103 | unclassified.ordinal_103.qk_rope | qk_rope | qk_rope; fusedQKRoPE19Half2Kernel; g361x13x1; b192x1x1; r16; s0 | 20 | n/a | 3.808 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 104 | unclassified.ordinal_104.fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 20 | n/a | 11.696 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 105 | unclassified.ordinal_105.library_gemm | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 20 | n/a | 8.288 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 106 | unclassified.ordinal_106.rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 20 | n/a | 2.400 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 107 | unclassified.ordinal_107.fused_ffn | fused_ffn | fused_ffn; fused_ffn_candidate_a_reuse_kernel; g18x37x1; b128x1x1; r136; s32768 | 20 | n/a | 37.248 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 108 | unclassified.ordinal_108.linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 20 | n/a | 20.512 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 109 | unclassified.ordinal_109.affine_silu | affine_silu | affine_silu; affineSiluHalf2Kernel; g4693x1x1; b192x1x1; r17; s0 | 20 | n/a | 3.328 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 110 | unclassified.ordinal_110.library_gemm | library_gemm | library_gemm; Kernel2; g148x1x1; b256x1x1; r154; s73728 | 20 | n/a | 14.272 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 111 | unclassified.ordinal_111.affine_silu | affine_silu | affine_silu; affineSiluHalf2Kernel; g4693x1x1; b384x1x1; r17; s0 | 20 | n/a | 5.008 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 112 | unclassified.ordinal_112.library_gemm | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 20 | n/a | 11.808 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 113 | unclassified.ordinal_113.rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 20 | n/a | 2.400 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 114 | unclassified.ordinal_114.library_gemm | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 20 | n/a | 7.345 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 115 | unclassified.ordinal_115.library_gemm | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 20 | n/a | 7.232 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 116 | unclassified.ordinal_116.library_gemm | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 20 | n/a | 7.328 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 117 | unclassified.ordinal_117.qk_rope | qk_rope | qk_rope; fusedQKRoPE19Half2Kernel; g361x13x1; b192x1x1; r16; s0 | 20 | n/a | 3.808 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 118 | unclassified.ordinal_118.fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 20 | n/a | 11.713 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 119 | unclassified.ordinal_119.library_gemm | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 20 | n/a | 8.432 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 120 | unclassified.ordinal_120.rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 20 | n/a | 2.400 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 121 | unclassified.ordinal_121.fused_ffn | fused_ffn | fused_ffn; fused_ffn_candidate_a_reuse_kernel; g18x37x1; b128x1x1; r136; s32768 | 20 | n/a | 37.536 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 122 | unclassified.ordinal_122.linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 20 | n/a | 20.416 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 123 | unclassified.ordinal_123.rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 20 | n/a | 2.400 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 124 | unclassified.ordinal_124.library_gemm | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 20 | n/a | 7.360 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 125 | unclassified.ordinal_125.library_gemm | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 20 | n/a | 7.312 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 126 | unclassified.ordinal_126.library_gemm | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 20 | n/a | 7.344 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 127 | unclassified.ordinal_127.qk_rope | qk_rope | qk_rope; fusedQKRoPE19Half2Kernel; g361x13x1; b192x1x1; r16; s0 | 20 | n/a | 3.792 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 128 | unclassified.ordinal_128.fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 20 | n/a | 11.616 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 129 | unclassified.ordinal_129.library_gemm | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 20 | n/a | 8.384 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 130 | unclassified.ordinal_130.rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 20 | n/a | 2.400 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 131 | unclassified.ordinal_131.fused_ffn | fused_ffn | fused_ffn; fused_ffn_candidate_a_reuse_kernel; g18x37x1; b128x1x1; r136; s32768 | 20 | n/a | 39.632 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 132 | unclassified.ordinal_132.linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 20 | n/a | 20.368 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 133 | unclassified.ordinal_133.rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 20 | n/a | 2.353 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 134 | unclassified.ordinal_134.library_gemm | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 20 | n/a | 7.312 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 135 | unclassified.ordinal_135.library_gemm | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 20 | n/a | 7.232 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 136 | unclassified.ordinal_136.library_gemm | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 20 | n/a | 7.248 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 137 | unclassified.ordinal_137.qk_rope | qk_rope | qk_rope; fusedQKRoPE19Half2Kernel; g361x13x1; b192x1x1; r16; s0 | 20 | n/a | 3.776 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 138 | unclassified.ordinal_138.fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 20 | n/a | 11.617 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 139 | unclassified.ordinal_139.library_gemm | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 20 | n/a | 8.320 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 140 | unclassified.ordinal_140.rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 20 | n/a | 2.417 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 141 | unclassified.ordinal_141.fused_ffn | fused_ffn | fused_ffn; fused_ffn_candidate_a_reuse_kernel; g18x37x1; b128x1x1; r136; s32768 | 20 | n/a | 40.096 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 142 | unclassified.ordinal_142.linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 20 | n/a | 20.288 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 143 | unclassified.ordinal_143.affine_silu | affine_silu | affine_silu; affineSiluHalf2Kernel; g4693x1x1; b192x1x1; r17; s0 | 20 | n/a | 3.280 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 144 | unclassified.ordinal_144.library_gemm | library_gemm | library_gemm; Kernel2; g148x1x1; b256x1x1; r154; s73728 | 20 | n/a | 14.032 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 145 | unclassified.ordinal_145.affine_silu | affine_silu | affine_silu; affineSiluHalf2Kernel; g4693x1x1; b384x1x1; r17; s0 | 20 | n/a | 4.992 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 146 | unclassified.ordinal_146.library_gemm | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 20 | n/a | 11.696 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 147 | unclassified.ordinal_147.rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 20 | n/a | 2.432 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 148 | unclassified.ordinal_148.library_gemm | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 20 | n/a | 7.232 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 149 | unclassified.ordinal_149.library_gemm | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 20 | n/a | 7.264 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 150 | unclassified.ordinal_150.library_gemm | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 20 | n/a | 7.264 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 151 | unclassified.ordinal_151.qk_rope | qk_rope | qk_rope; fusedQKRoPE19Half2Kernel; g361x13x1; b192x1x1; r16; s0 | 20 | n/a | 3.825 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 152 | unclassified.ordinal_152.fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 20 | n/a | 11.633 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 153 | unclassified.ordinal_153.library_gemm | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 20 | n/a | 8.256 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 154 | unclassified.ordinal_154.rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 20 | n/a | 2.400 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 155 | unclassified.ordinal_155.fused_ffn | fused_ffn | fused_ffn; fused_ffn_candidate_a_reuse_kernel; g18x37x1; b128x1x1; r136; s32768 | 20 | n/a | 37.776 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 156 | unclassified.ordinal_156.linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 20 | n/a | 20.512 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 157 | unclassified.ordinal_157.rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 20 | n/a | 2.400 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 158 | unclassified.ordinal_158.library_gemm | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 20 | n/a | 7.472 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 159 | unclassified.ordinal_159.library_gemm | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 20 | n/a | 7.360 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 160 | unclassified.ordinal_160.library_gemm | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 20 | n/a | 7.296 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 161 | unclassified.ordinal_161.qk_rope | qk_rope | qk_rope; fusedQKRoPE19Half2Kernel; g361x13x1; b192x1x1; r16; s0 | 20 | n/a | 3.824 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 162 | unclassified.ordinal_162.fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 20 | n/a | 11.744 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 163 | unclassified.ordinal_163.library_gemm | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 20 | n/a | 8.384 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 164 | unclassified.ordinal_164.rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 20 | n/a | 2.400 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 165 | unclassified.ordinal_165.fused_ffn | fused_ffn | fused_ffn; fused_ffn_candidate_a_reuse_kernel; g18x37x1; b128x1x1; r136; s32768 | 20 | n/a | 37.856 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 166 | unclassified.ordinal_166.linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 20 | n/a | 20.576 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 167 | unclassified.ordinal_167.rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 20 | n/a | 2.432 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 168 | unclassified.ordinal_168.library_gemm | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 20 | n/a | 7.440 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 169 | unclassified.ordinal_169.library_gemm | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 20 | n/a | 7.344 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 170 | unclassified.ordinal_170.library_gemm | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 20 | n/a | 7.360 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 171 | unclassified.ordinal_171.qk_rope | qk_rope | qk_rope; fusedQKRoPE19Half2Kernel; g361x13x1; b192x1x1; r16; s0 | 20 | n/a | 3.840 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 172 | unclassified.ordinal_172.fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 20 | n/a | 11.744 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 173 | unclassified.ordinal_173.library_gemm | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 20 | n/a | 8.384 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 174 | unclassified.ordinal_174.rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 20 | n/a | 2.400 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 175 | unclassified.ordinal_175.fused_ffn | fused_ffn | fused_ffn; fused_ffn_candidate_a_reuse_kernel; g18x37x1; b128x1x1; r136; s32768 | 20 | n/a | 34.721 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 176 | unclassified.ordinal_176.linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 20 | n/a | 20.704 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 177 | unclassified.ordinal_177.affine_silu | affine_silu | affine_silu; affineSiluHalf2Kernel; g4693x1x1; b192x1x1; r17; s0 | 20 | n/a | 3.360 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 178 | unclassified.ordinal_178.library_gemm | library_gemm | library_gemm; Kernel2; g148x1x1; b256x1x1; r154; s73728 | 20 | n/a | 14.400 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 179 | unclassified.ordinal_179.affine_silu | affine_silu | affine_silu; affineSiluHalf2Kernel; g4693x1x1; b384x1x1; r17; s0 | 20 | n/a | 5.056 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 180 | unclassified.ordinal_180.library_gemm | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 20 | n/a | 11.840 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 181 | unclassified.ordinal_181.rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 20 | n/a | 2.448 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 182 | unclassified.ordinal_182.library_gemm | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 20 | n/a | 7.296 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 183 | unclassified.ordinal_183.library_gemm | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 20 | n/a | 7.296 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 184 | unclassified.ordinal_184.library_gemm | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 20 | n/a | 7.264 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 185 | unclassified.ordinal_185.qk_rope | qk_rope | qk_rope; fusedQKRoPE19Half2Kernel; g361x13x1; b192x1x1; r16; s0 | 20 | n/a | 3.872 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 186 | unclassified.ordinal_186.fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 20 | n/a | 11.792 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 187 | unclassified.ordinal_187.library_gemm | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 20 | n/a | 8.448 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 188 | unclassified.ordinal_188.rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 20 | n/a | 2.400 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 189 | unclassified.ordinal_189.fused_ffn | fused_ffn | fused_ffn; fused_ffn_candidate_a_reuse_kernel; g18x37x1; b128x1x1; r136; s32768 | 20 | n/a | 40.544 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 190 | unclassified.ordinal_190.linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 20 | n/a | 20.593 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 191 | unclassified.ordinal_191.rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 20 | n/a | 2.432 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 192 | unclassified.ordinal_192.library_gemm | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 20 | n/a | 7.424 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 193 | unclassified.ordinal_193.library_gemm | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 20 | n/a | 7.440 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 194 | unclassified.ordinal_194.library_gemm | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 20 | n/a | 7.296 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 195 | unclassified.ordinal_195.qk_rope | qk_rope | qk_rope; fusedQKRoPE19Half2Kernel; g361x13x1; b192x1x1; r16; s0 | 20 | n/a | 3.840 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 196 | unclassified.ordinal_196.fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 20 | n/a | 11.728 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 197 | unclassified.ordinal_197.library_gemm | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 20 | n/a | 8.480 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 198 | unclassified.ordinal_198.rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 20 | n/a | 2.400 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 199 | unclassified.ordinal_199.fused_ffn | fused_ffn | fused_ffn; fused_ffn_candidate_a_reuse_kernel; g18x37x1; b128x1x1; r136; s32768 | 20 | n/a | 39.728 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 200 | unclassified.ordinal_200.linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 20 | n/a | 20.497 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 201 | unclassified.ordinal_201.rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 20 | n/a | 2.416 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 202 | unclassified.ordinal_202.library_gemm | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 20 | n/a | 7.424 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 203 | unclassified.ordinal_203.library_gemm | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 20 | n/a | 7.360 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 204 | unclassified.ordinal_204.library_gemm | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 20 | n/a | 7.296 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 205 | unclassified.ordinal_205.qk_rope | qk_rope | qk_rope; fusedQKRoPE19Half2Kernel; g361x13x1; b192x1x1; r16; s0 | 20 | n/a | 3.776 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 206 | unclassified.ordinal_206.fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 20 | n/a | 11.601 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 207 | unclassified.ordinal_207.library_gemm | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 20 | n/a | 8.288 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 208 | unclassified.ordinal_208.rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 20 | n/a | 2.368 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 209 | unclassified.ordinal_209.fused_ffn | fused_ffn | fused_ffn; fused_ffn_candidate_a_reuse_kernel; g18x37x1; b128x1x1; r136; s32768 | 20 | n/a | 38.096 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 210 | unclassified.ordinal_210.linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 20 | n/a | 20.416 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 211 | unclassified.ordinal_211.affine_silu | affine_silu | affine_silu; affineSiluHalf2Kernel; g4693x1x1; b192x1x1; r17; s0 | 20 | n/a | 3.280 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 212 | unclassified.ordinal_212.library_gemm | library_gemm | library_gemm; Kernel2; g148x1x1; b256x1x1; r154; s73728 | 20 | n/a | 14.192 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 213 | unclassified.ordinal_213.affine_silu | affine_silu | affine_silu; affineSiluHalf2Kernel; g4693x1x1; b384x1x1; r17; s0 | 20 | n/a | 4.976 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 214 | unclassified.ordinal_214.library_gemm | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 20 | n/a | 11.760 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 215 | unclassified.ordinal_215.rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 20 | n/a | 2.400 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 216 | unclassified.ordinal_216.library_gemm | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 20 | n/a | 7.360 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 217 | unclassified.ordinal_217.library_gemm | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 20 | n/a | 7.296 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 218 | unclassified.ordinal_218.library_gemm | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 20 | n/a | 7.264 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 219 | unclassified.ordinal_219.qk_rope | qk_rope | qk_rope; fusedQKRoPE19Half2Kernel; g361x13x1; b192x1x1; r16; s0 | 20 | n/a | 3.840 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 220 | unclassified.ordinal_220.fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 20 | n/a | 11.648 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 221 | unclassified.ordinal_221.library_gemm | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 20 | n/a | 8.368 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 222 | unclassified.ordinal_222.rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 20 | n/a | 2.432 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 223 | unclassified.ordinal_223.fused_ffn | fused_ffn | fused_ffn; fused_ffn_candidate_a_reuse_kernel; g18x37x1; b128x1x1; r136; s32768 | 20 | n/a | 39.920 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 224 | unclassified.ordinal_224.linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 20 | n/a | 20.576 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 225 | unclassified.ordinal_225.rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 20 | n/a | 2.432 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 226 | unclassified.ordinal_226.library_gemm | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 20 | n/a | 7.456 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 227 | unclassified.ordinal_227.library_gemm | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 20 | n/a | 7.376 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 228 | unclassified.ordinal_228.library_gemm | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 20 | n/a | 7.328 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 229 | unclassified.ordinal_229.qk_rope | qk_rope | qk_rope; fusedQKRoPE19Half2Kernel; g361x13x1; b192x1x1; r16; s0 | 20 | n/a | 3.840 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 230 | unclassified.ordinal_230.fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 20 | n/a | 11.696 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 231 | unclassified.ordinal_231.library_gemm | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 20 | n/a | 8.320 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 232 | unclassified.ordinal_232.rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 20 | n/a | 2.400 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 233 | unclassified.ordinal_233.fused_ffn | fused_ffn | fused_ffn; fused_ffn_candidate_a_reuse_kernel; g18x37x1; b128x1x1; r136; s32768 | 20 | n/a | 38.897 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 234 | unclassified.ordinal_234.linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 20 | n/a | 20.624 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 235 | unclassified.ordinal_235.rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 20 | n/a | 2.416 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 236 | unclassified.ordinal_236.library_gemm | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 20 | n/a | 7.456 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 237 | unclassified.ordinal_237.library_gemm | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 20 | n/a | 7.312 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 238 | unclassified.ordinal_238.library_gemm | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 20 | n/a | 7.312 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 239 | unclassified.ordinal_239.qk_rope | qk_rope | qk_rope; fusedQKRoPE19Half2Kernel; g361x13x1; b192x1x1; r16; s0 | 20 | n/a | 3.840 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 240 | unclassified.ordinal_240.fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 20 | n/a | 11.744 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 241 | unclassified.ordinal_241.library_gemm | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 20 | n/a | 8.400 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 242 | unclassified.ordinal_242.rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 20 | n/a | 2.401 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 243 | unclassified.ordinal_243.fused_ffn | fused_ffn | fused_ffn; fused_ffn_candidate_a_reuse_kernel; g18x37x1; b128x1x1; r136; s32768 | 20 | n/a | 34.528 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 244 | unclassified.ordinal_244.linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 20 | n/a | 20.704 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 245 | unclassified.ordinal_245.affine_silu | affine_silu | affine_silu; affineSiluHalf2Kernel; g4693x1x1; b192x1x1; r17; s0 | 20 | n/a | 3.328 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 246 | unclassified.ordinal_246.library_gemm | library_gemm | library_gemm; Kernel2; g148x1x1; b256x1x1; r154; s73728 | 20 | n/a | 14.384 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 247 | unclassified.ordinal_247.affine_silu | affine_silu | affine_silu; affineSiluHalf2Kernel; g4693x1x1; b384x1x1; r17; s0 | 20 | n/a | 5.024 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 248 | unclassified.ordinal_248.library_gemm | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 20 | n/a | 11.857 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 249 | unclassified.ordinal_249.rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 20 | n/a | 2.432 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 250 | unclassified.ordinal_250.library_gemm | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 20 | n/a | 7.360 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 251 | unclassified.ordinal_251.library_gemm | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 20 | n/a | 7.408 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 252 | unclassified.ordinal_252.library_gemm | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 20 | n/a | 7.328 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 253 | unclassified.ordinal_253.qk_rope | qk_rope | qk_rope; fusedQKRoPE19Half2Kernel; g361x13x1; b192x1x1; r16; s0 | 20 | n/a | 3.808 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 254 | unclassified.ordinal_254.fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 20 | n/a | 11.744 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 255 | unclassified.ordinal_255.library_gemm | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 20 | n/a | 8.336 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 256 | unclassified.ordinal_256.rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 20 | n/a | 2.400 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 257 | unclassified.ordinal_257.fused_ffn | fused_ffn | fused_ffn; fused_ffn_candidate_a_reuse_kernel; g18x37x1; b128x1x1; r136; s32768 | 20 | n/a | 35.392 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 258 | unclassified.ordinal_258.linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 20 | n/a | 20.832 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 259 | unclassified.ordinal_259.rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 20 | n/a | 2.400 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 260 | unclassified.ordinal_260.library_gemm | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 20 | n/a | 7.569 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 261 | unclassified.ordinal_261.library_gemm | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 20 | n/a | 7.456 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 262 | unclassified.ordinal_262.library_gemm | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 20 | n/a | 7.393 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 263 | unclassified.ordinal_263.qk_rope | qk_rope | qk_rope; fusedQKRoPE19Half2Kernel; g361x13x1; b192x1x1; r16; s0 | 20 | n/a | 3.840 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 264 | unclassified.ordinal_264.fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 20 | n/a | 11.825 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 265 | unclassified.ordinal_265.library_gemm | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 20 | n/a | 8.448 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 266 | unclassified.ordinal_266.rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 20 | n/a | 2.432 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 267 | unclassified.ordinal_267.fused_ffn | fused_ffn | fused_ffn; fused_ffn_candidate_a_reuse_kernel; g18x37x1; b128x1x1; r136; s32768 | 20 | n/a | 35.297 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 268 | unclassified.ordinal_268.linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 20 | n/a | 20.704 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 269 | unclassified.ordinal_269.rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 20 | n/a | 2.464 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 270 | unclassified.ordinal_270.library_gemm | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 20 | n/a | 7.520 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 271 | unclassified.ordinal_271.library_gemm | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 20 | n/a | 7.344 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 272 | unclassified.ordinal_272.library_gemm | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 20 | n/a | 7.361 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 273 | unclassified.ordinal_273.qk_rope | qk_rope | qk_rope; fusedQKRoPE19Half2Kernel; g361x13x1; b192x1x1; r16; s0 | 20 | n/a | 3.856 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 274 | unclassified.ordinal_274.fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 20 | n/a | 11.776 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 275 | unclassified.ordinal_275.library_gemm | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 20 | n/a | 8.432 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 276 | unclassified.ordinal_276.rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 20 | n/a | 2.400 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 277 | unclassified.ordinal_277.fused_ffn | fused_ffn | fused_ffn; fused_ffn_candidate_a_reuse_kernel; g18x37x1; b128x1x1; r136; s32768 | 20 | n/a | 35.136 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 278 | unclassified.ordinal_278.linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 20 | n/a | 20.800 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 279 | unclassified.ordinal_279.affine_silu | affine_silu | affine_silu; affineSiluHalf2Kernel; g4693x1x1; b192x1x1; r17; s0 | 20 | n/a | 3.360 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 280 | unclassified.ordinal_280.library_gemm | library_gemm | library_gemm; Kernel2; g148x1x1; b256x1x1; r154; s73728 | 20 | n/a | 14.400 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 281 | unclassified.ordinal_281.affine_silu | affine_silu | affine_silu; affineSiluHalf2Kernel; g4693x1x1; b384x1x1; r17; s0 | 20 | n/a | 5.056 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 282 | unclassified.ordinal_282.library_gemm | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 20 | n/a | 11.936 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 283 | unclassified.ordinal_283.rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 20 | n/a | 2.433 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 284 | unclassified.ordinal_284.library_gemm | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 20 | n/a | 7.409 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 285 | unclassified.ordinal_285.library_gemm | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 20 | n/a | 7.328 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 286 | unclassified.ordinal_286.library_gemm | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 20 | n/a | 7.328 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 287 | unclassified.ordinal_287.qk_rope | qk_rope | qk_rope; fusedQKRoPE19Half2Kernel; g361x13x1; b192x1x1; r16; s0 | 20 | n/a | 3.856 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 288 | unclassified.ordinal_288.fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 20 | n/a | 11.809 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 289 | unclassified.ordinal_289.library_gemm | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 20 | n/a | 8.368 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 290 | unclassified.ordinal_290.rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 20 | n/a | 2.432 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 291 | unclassified.ordinal_291.fused_ffn | fused_ffn | fused_ffn; fused_ffn_candidate_a_reuse_kernel; g18x37x1; b128x1x1; r136; s32768 | 20 | n/a | 39.905 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 292 | unclassified.ordinal_292.linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 20 | n/a | 20.784 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 293 | unclassified.ordinal_293.rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 20 | n/a | 2.432 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 294 | unclassified.ordinal_294.library_gemm | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 20 | n/a | 7.489 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 295 | unclassified.ordinal_295.library_gemm | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 20 | n/a | 7.392 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 296 | unclassified.ordinal_296.library_gemm | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 20 | n/a | 7.296 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 297 | unclassified.ordinal_297.qk_rope | qk_rope | qk_rope; fusedQKRoPE19Half2Kernel; g361x13x1; b192x1x1; r16; s0 | 20 | n/a | 3.840 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 298 | unclassified.ordinal_298.fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 20 | n/a | 11.792 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 299 | unclassified.ordinal_299.library_gemm | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 20 | n/a | 8.352 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 300 | unclassified.ordinal_300.rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 20 | n/a | 2.432 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 301 | unclassified.ordinal_301.fused_ffn | fused_ffn | fused_ffn; fused_ffn_candidate_a_reuse_kernel; g18x37x1; b128x1x1; r136; s32768 | 20 | n/a | 40.065 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 302 | unclassified.ordinal_302.linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 20 | n/a | 20.657 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 303 | unclassified.ordinal_303.rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 20 | n/a | 2.464 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 304 | unclassified.ordinal_304.library_gemm | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 20 | n/a | 7.505 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 305 | unclassified.ordinal_305.library_gemm | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 20 | n/a | 7.424 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 306 | unclassified.ordinal_306.library_gemm | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 20 | n/a | 7.440 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 307 | unclassified.ordinal_307.qk_rope | qk_rope | qk_rope; fusedQKRoPE19Half2Kernel; g361x13x1; b192x1x1; r16; s0 | 20 | n/a | 3.808 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 308 | unclassified.ordinal_308.fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 20 | n/a | 11.728 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 309 | unclassified.ordinal_309.library_gemm | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 20 | n/a | 8.352 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 310 | unclassified.ordinal_310.rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 20 | n/a | 2.400 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 311 | unclassified.ordinal_311.fused_ffn | fused_ffn | fused_ffn; fused_ffn_candidate_a_reuse_kernel; g18x37x1; b128x1x1; r136; s32768 | 20 | n/a | 39.601 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 312 | unclassified.ordinal_312.linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 20 | n/a | 20.624 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 313 | unclassified.ordinal_313.affine_silu | affine_silu | affine_silu; affineSiluHalf2Kernel; g4693x1x1; b192x1x1; r17; s0 | 20 | n/a | 3.328 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 314 | unclassified.ordinal_314.library_gemm | library_gemm | library_gemm; Kernel2; g148x1x1; b256x1x1; r154; s73728 | 20 | n/a | 14.256 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 315 | unclassified.ordinal_315.affine_silu | affine_silu | affine_silu; affineSiluHalf2Kernel; g4693x1x1; b384x1x1; r17; s0 | 20 | n/a | 4.976 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 316 | unclassified.ordinal_316.library_gemm | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 20 | n/a | 11.824 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 317 | trunk.tip_norm_silu | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 20 | n/a | 2.400 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 318 | policy.p1_conv | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 20 | n/a | 7.377 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 319 | policy.g1_conv | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 20 | n/a | 7.328 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 320 | policy.g1_norm_silu | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 20 | n/a | 7.312 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 321 | policy.g1_half_to_float | qk_rope | qk_rope; fusedQKRoPE19Half2Kernel; g361x13x1; b192x1x1; r16; s0 | 20 | n/a | 3.840 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 322 | policy.g1_global_pool | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 20 | n/a | 11.776 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 323 | policy.gpool_to_bias_matmul | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 20 | n/a | 8.368 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 324 | policy.p1_half_to_float | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 20 | n/a | 2.432 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 325 | policy.gpool_bias_add | fused_ffn | fused_ffn; fused_ffn_candidate_a_reuse_kernel; g18x37x1; b128x1x1; r136; s32768 | 20 | n/a | 38.432 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 326 | policy.p1_norm_silu | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 20 | n/a | 20.480 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 327 | policy.p2_conv | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 20 | n/a | 2.400 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 328 | policy.gpool_to_pass_matmul | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 20 | n/a | 7.392 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 329 | policy.pass_bias_silu | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 20 | n/a | 7.360 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 330 | policy.gpool_to_pass_matmul2 | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 20 | n/a | 7.232 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 331 | value.v1_conv | qk_rope | qk_rope; fusedQKRoPE19Half2Kernel; g361x13x1; b192x1x1; r16; s0 | 20 | n/a | 3.808 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 332 | value.v1_norm_silu | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 20 | n/a | 11.648 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 333 | value.v1_half_to_float | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 20 | n/a | 8.320 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 334 | value.v1_global_pool | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 20 | n/a | 2.401 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 335 | value.v2_matmul | fused_ffn | fused_ffn; fused_ffn_candidate_a_reuse_kernel; g18x37x1; b128x1x1; r136; s32768 | 20 | n/a | 39.680 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 336 | value.v2_bias_silu | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 20 | n/a | 20.368 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 337 | value.v3_matmul | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 20 | n/a | 2.368 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 338 | value.v3_bias | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 20 | n/a | 7.344 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 339 | value.score_matmul | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 20 | n/a | 7.328 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 340 | value.score_bias | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 20 | n/a | 7.232 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 341 | value.ownership_conv | qk_rope | qk_rope; fusedQKRoPE19Half2Kernel; g361x13x1; b192x1x1; r16; s0 | 20 | n/a | 3.808 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 342 | value.ownership_conv_splitk_reduce | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 20 | n/a | 11.584 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 343 | value.ownership_half_to_float | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 20 | n/a | 8.224 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 344 | unclassified.ordinal_344.rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 20 | n/a | 2.400 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 345 | unclassified.ordinal_345.fused_ffn | fused_ffn | fused_ffn; fused_ffn_candidate_a_reuse_kernel; g18x37x1; b128x1x1; r136; s32768 | 20 | n/a | 39.665 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 346 | unclassified.ordinal_346.linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 20 | n/a | 20.288 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 347 | unclassified.ordinal_347.affine_silu | affine_silu | affine_silu; affineSiluHalf2Kernel; g4693x1x1; b192x1x1; r17; s0 | 20 | n/a | 3.296 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 348 | unclassified.ordinal_348.library_gemm | library_gemm | library_gemm; Kernel2; g148x1x1; b256x1x1; r154; s73728 | 20 | n/a | 14.080 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 349 | unclassified.ordinal_349.affine_silu | affine_silu | affine_silu; affineSiluHalf2Kernel; g4693x1x1; b384x1x1; r17; s0 | 20 | n/a | 4.960 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 350 | unclassified.ordinal_350.library_gemm | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 20 | n/a | 11.665 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 351 | unclassified.ordinal_351.rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 20 | n/a | 2.432 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 352 | unclassified.ordinal_352.library_gemm | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 20 | n/a | 7.200 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 353 | unclassified.ordinal_353.library_gemm | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 20 | n/a | 7.200 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 354 | unclassified.ordinal_354.library_gemm | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 20 | n/a | 7.296 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 355 | unclassified.ordinal_355.qk_rope | qk_rope | qk_rope; fusedQKRoPE19Half2Kernel; g361x13x1; b192x1x1; r16; s0 | 20 | n/a | 3.808 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 356 | unclassified.ordinal_356.fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 20 | n/a | 11.600 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 357 | unclassified.ordinal_357.library_gemm | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 20 | n/a | 8.352 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 358 | unclassified.ordinal_358.rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 20 | n/a | 2.369 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 359 | unclassified.ordinal_359.fused_ffn | fused_ffn | fused_ffn; fused_ffn_candidate_a_reuse_kernel; g18x37x1; b128x1x1; r136; s32768 | 20 | n/a | 35.648 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 360 | unclassified.ordinal_360.linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 20 | n/a | 20.544 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 361 | unclassified.ordinal_361.rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 20 | n/a | 2.432 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 362 | unclassified.ordinal_362.library_gemm | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 20 | n/a | 7.408 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 363 | unclassified.ordinal_363.library_gemm | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 20 | n/a | 7.360 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 364 | unclassified.ordinal_364.library_gemm | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 20 | n/a | 7.328 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 365 | unclassified.ordinal_365.qk_rope | qk_rope | qk_rope; fusedQKRoPE19Half2Kernel; g361x13x1; b192x1x1; r16; s0 | 20 | n/a | 3.840 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 366 | unclassified.ordinal_366.fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 20 | n/a | 11.776 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 367 | unclassified.ordinal_367.library_gemm | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 20 | n/a | 8.400 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 368 | unclassified.ordinal_368.rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 20 | n/a | 2.400 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 369 | unclassified.ordinal_369.fused_ffn | fused_ffn | fused_ffn; fused_ffn_candidate_a_reuse_kernel; g18x37x1; b128x1x1; r136; s32768 | 20 | n/a | 36.928 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 370 | unclassified.ordinal_370.linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 20 | n/a | 20.640 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 371 | unclassified.ordinal_371.rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 20 | n/a | 2.400 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 372 | unclassified.ordinal_372.library_gemm | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 20 | n/a | 7.504 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 373 | unclassified.ordinal_373.library_gemm | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 20 | n/a | 7.328 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 374 | unclassified.ordinal_374.library_gemm | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 20 | n/a | 7.360 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 375 | unclassified.ordinal_375.qk_rope | qk_rope | qk_rope; fusedQKRoPE19Half2Kernel; g361x13x1; b192x1x1; r16; s0 | 20 | n/a | 3.824 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 376 | unclassified.ordinal_376.fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 20 | n/a | 11.744 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 377 | unclassified.ordinal_377.library_gemm | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 20 | n/a | 8.384 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 378 | unclassified.ordinal_378.rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 20 | n/a | 2.432 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 379 | unclassified.ordinal_379.fused_ffn | fused_ffn | fused_ffn; fused_ffn_candidate_a_reuse_kernel; g18x37x1; b128x1x1; r136; s32768 | 20 | n/a | 39.776 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 380 | unclassified.ordinal_380.linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 20 | n/a | 20.576 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 381 | unclassified.ordinal_381.affine_silu | affine_silu | affine_silu; affineSiluHalf2Kernel; g4693x1x1; b192x1x1; r17; s0 | 20 | n/a | 3.360 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 382 | unclassified.ordinal_382.library_gemm | library_gemm | library_gemm; Kernel2; g148x1x1; b256x1x1; r154; s73728 | 20 | n/a | 14.225 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 383 | unclassified.ordinal_383.affine_silu | affine_silu | affine_silu; affineSiluHalf2Kernel; g4693x1x1; b384x1x1; r17; s0 | 20 | n/a | 5.008 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 384 | unclassified.ordinal_384.library_gemm | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r80; s98304 | 20 | n/a | 6.113 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 385 | unclassified.ordinal_385.library_gemm | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r80; s98304 | 20 | n/a | 5.888 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 386 | unclassified.ordinal_386.head_elementwise | head_elementwise | head_elementwise; applyCScaleBiasNHWCSiluHalfKernel; g1x73x13; b96x5x1; r16; s0 | 20 | n/a | 2.080 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 387 | unclassified.ordinal_387.copy_reformat | copy_reformat | copy_reformat; copyFromHalfKernel; g880x1x1; b512x1x1; r16; s0 | 20 | n/a | 1.536 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 388 | unclassified.ordinal_388.head_elementwise | head_elementwise | head_elementwise; gPoolChannelsNHWCKernel; g2x1x13; b64x8x1; r22; s4096 | 20 | n/a | 4.416 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 389 | unclassified.ordinal_389.library_gemm | library_gemm | library_gemm; gemmSN_NN_kernel; g2x2x1; b256x1x1; r64; s21504 | 20 | n/a | 5.344 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 390 | unclassified.ordinal_390.copy_reformat | copy_reformat | copy_reformat; copyFromHalfKernel; g880x1x1; b512x1x1; r16; s0 | 20 | n/a | 1.504 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 391 | unclassified.ordinal_391.head_elementwise | head_elementwise | head_elementwise; addNCBiasInplaceNHWCKernel; g1x73x13; b96x5x1; r16; s0 | 20 | n/a | 1.760 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 392 | unclassified.ordinal_392.head_elementwise | head_elementwise | head_elementwise; applyCScaleBiasNHWCSiluKernel; g1x73x13; b96x5x1; r16; s0 | 20 | n/a | 2.144 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 393 | unclassified.ordinal_393.library_gemm | library_gemm | library_gemm; Kernel2; g74x1x1; b128x1x1; r90; s98304 | 20 | n/a | 3.904 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 394 | unclassified.ordinal_394.library_gemm | library_gemm | library_gemm; gemmSN_NN_kernel; g2x2x1; b256x1x1; r64; s21504 | 20 | n/a | 5.280 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 395 | unclassified.ordinal_395.head_elementwise | head_elementwise | head_elementwise; addCBiasInplaceNCKernelSilu; g1x3x1; b96x5x1; r16; s0 | 20 | n/a | 1.024 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 396 | unclassified.ordinal_396.library_gemm | library_gemm | library_gemm; gemmSN_NN_kernel; g1x2x1; b256x1x1; r64; s21504 | 20 | n/a | 2.304 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 397 | unclassified.ordinal_397.library_gemm | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r118; s98304 | 20 | n/a | 7.904 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 398 | unclassified.ordinal_398.head_elementwise | head_elementwise | head_elementwise; applyCScaleBiasNHWCSiluHalfKernel; g1x181x13; b192x2x1; r16; s0 | 20 | n/a | 3.136 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 399 | unclassified.ordinal_399.copy_reformat | copy_reformat | copy_reformat; copyFromHalfKernel; g1760x1x1; b512x1x1; r16; s0 | 20 | n/a | 2.144 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 400 | unclassified.ordinal_400.head_elementwise | head_elementwise | head_elementwise; valueHeadPoolChannelsNHWCKernel; g3x1x13; b64x8x1; r22; s2048 | 20 | n/a | 3.232 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 401 | unclassified.ordinal_401.library_gemm | library_gemm | library_gemm; gemmSN_NN_kernel; g3x2x1; b256x1x1; r64; s21504 | 20 | n/a | 9.488 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 402 | unclassified.ordinal_402.head_elementwise | head_elementwise | head_elementwise; addCBiasInplaceNCKernelSilu; g1x7x1; b192x2x1; r16; s0 | 20 | n/a | 1.024 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 403 | unclassified.ordinal_403.library_gemm | library_gemm | library_gemm; gemmSN_NN_kernel; g1x2x1; b256x1x1; r64; s21504 | 20 | n/a | 3.488 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 404 | unclassified.ordinal_404.head_elementwise | head_elementwise | head_elementwise; addCBiasInplaceNCKernel; g1x1x1; b3x170x1; r16; s0 | 20 | n/a | 0.960 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 405 | unclassified.ordinal_405.library_gemm | library_gemm | library_gemm; gemmSN_NN_kernel; g1x2x1; b256x1x1; r64; s21504 | 20 | n/a | 3.504 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 406 | unclassified.ordinal_406.head_elementwise | head_elementwise | head_elementwise; addCBiasInplaceNCKernel; g1x1x1; b6x85x1; r16; s0 | 20 | n/a | 0.928 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 407 | unclassified.ordinal_407.library_gemm | library_gemm | library_gemm; Kernel2; g8x19x3; b128x1x1; r118; s33792 | 20 | n/a | 4.064 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 408 | unclassified.ordinal_408.library_gemm | library_gemm | library_gemm; splitKreduce_kernel; g147x1x1; b32x16x1; r49; s0 | 20 | n/a | 1.376 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 409 | unclassified.ordinal_409.copy_reformat | copy_reformat | copy_reformat; copyFromHalfKernel; g10x1x1; b512x1x1; r16; s0 | 20 | n/a | 0.944 | n/a | 0.0% | 0.000 | idle (20) | n/a |
