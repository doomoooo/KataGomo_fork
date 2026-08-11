# Nsys stream interference report

- Timed iterations: 20; streams: 48
- Kernels per forward: 48=344

## Kernel families

| family | instances | total ms | median us | p90 us | peer overlap | S2/S1 median | summed excess ms | matched |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| fused_ffn | 660 | 25.543 | 38.784 | 41.152 | 0.0% | n/a | 0.000 | 0 |
| linear2_residual | 660 | 13.765 | 20.864 | 21.216 | 0.0% | n/a | 0.000 | 0 |
| library_gemm | 1380 | 12.666 | 8.544 | 14.336 | 0.0% | n/a | 0.000 | 0 |
| copy_reformat | 760 | 9.490 | 14.112 | 14.400 | 0.0% | n/a | 0.000 | 0 |
| fa4 | 660 | 7.856 | 11.904 | 12.096 | 0.0% | n/a | 0.000 | 0 |
| qk_rope | 660 | 3.712 | 5.632 | 5.728 | 0.0% | n/a | 0.000 | 0 |
| rmsnorm | 1320 | 3.233 | 2.432 | 2.496 | 0.0% | n/a | 0.000 | 0 |
| affine_silu | 460 | 1.950 | 4.960 | 5.120 | 0.0% | n/a | 0.000 | 0 |
| head_elementwise | 240 | 0.597 | 2.016 | 4.512 | 0.0% | n/a | 0.000 | 0 |
| cudnn | 60 | 0.448 | 1.536 | 19.585 | 0.0% | n/a | 0.000 | 0 |
| sumChannelsNCHWKernel | 20 | 0.034 | 1.696 | 1.696 | 0.0% | n/a | 0.000 | 0 |

## Dominant interference pairs

| running family | dominant peer | instances | median overlap | median us | S2/S1 median | matched |
|---|---|---:|---:|---:|---:|---:|
| library_gemm | idle | 1380 | 0.0% | 8.544 | n/a | 0 |
| rmsnorm | idle | 1320 | 0.0% | 2.432 | n/a | 0 |
| copy_reformat | idle | 760 | 0.0% | 14.112 | n/a | 0 |
| qk_rope | idle | 660 | 0.0% | 5.632 | n/a | 0 |
| fa4 | idle | 660 | 0.0% | 11.904 | n/a | 0 |
| fused_ffn | idle | 660 | 0.0% | 38.784 | n/a | 0 |
| linear2_residual | idle | 660 | 0.0% | 20.864 | n/a | 0 |
| affine_silu | idle | 460 | 0.0% | 4.960 | n/a | 0 |
| head_elementwise | idle | 240 | 0.0% | 2.016 | n/a | 0 |
| cudnn | idle | 60 | 0.0% | 1.536 | n/a | 0 |
| sumChannelsNCHWKernel | idle | 20 | 0.0% | 1.696 | n/a | 0 |

## Logical operation groups

Isolated reference total is the isolated median for each ordinal multiplied by its S2 call count; it is a normalized reference, not a second trace total.

| logical group | families | positions | calls | isolated reference ms | S2 total ms | total ratio | excess ms |
|---|---|---:|---:|---:|---:|---:|---:|
| unclassified.ordinal_157.fused_ffn | fused_ffn | 1 | 20 | 0.000 | 0.824 | n/a | 0.000 |
| unclassified.ordinal_81.fused_ffn | fused_ffn | 1 | 20 | 0.000 | 0.817 | n/a | 0.000 |
| unclassified.ordinal_249.fused_ffn | fused_ffn | 1 | 20 | 0.000 | 0.816 | n/a | 0.000 |
| unclassified.ordinal_241.fused_ffn | fused_ffn | 1 | 20 | 0.000 | 0.814 | n/a | 0.000 |
| unclassified.ordinal_313.fused_ffn | fused_ffn | 1 | 20 | 0.000 | 0.813 | n/a | 0.000 |
| unclassified.ordinal_185.fused_ffn | fused_ffn | 1 | 20 | 0.000 | 0.812 | n/a | 0.000 |
| unclassified.ordinal_117.fused_ffn | fused_ffn | 1 | 20 | 0.000 | 0.809 | n/a | 0.000 |
| unclassified.ordinal_257.fused_ffn | fused_ffn | 1 | 20 | 0.000 | 0.805 | n/a | 0.000 |
| unclassified.ordinal_109.fused_ffn | fused_ffn | 1 | 20 | 0.000 | 0.804 | n/a | 0.000 |
| unclassified.ordinal_165.fused_ffn | fused_ffn | 1 | 20 | 0.000 | 0.804 | n/a | 0.000 |
| unclassified.ordinal_73.fused_ffn | fused_ffn | 1 | 20 | 0.000 | 0.798 | n/a | 0.000 |
| unclassified.ordinal_45.fused_ffn | fused_ffn | 1 | 20 | 0.000 | 0.794 | n/a | 0.000 |
| unclassified.ordinal_193.fused_ffn | fused_ffn | 1 | 20 | 0.000 | 0.788 | n/a | 0.000 |
| unclassified.ordinal_17.fused_ffn | fused_ffn | 1 | 20 | 0.000 | 0.787 | n/a | 0.000 |
| unclassified.ordinal_53.fused_ffn | fused_ffn | 1 | 20 | 0.000 | 0.782 | n/a | 0.000 |
| unclassified.ordinal_285.fused_ffn | fused_ffn | 1 | 20 | 0.000 | 0.782 | n/a | 0.000 |
| unclassified.ordinal_173.fused_ffn | fused_ffn | 1 | 20 | 0.000 | 0.778 | n/a | 0.000 |
| unclassified.ordinal_269.fused_ffn | fused_ffn | 1 | 20 | 0.000 | 0.776 | n/a | 0.000 |
| unclassified.ordinal_277.fused_ffn | fused_ffn | 1 | 20 | 0.000 | 0.774 | n/a | 0.000 |
| unclassified.ordinal_137.fused_ffn | fused_ffn | 1 | 20 | 0.000 | 0.769 | n/a | 0.000 |
| unclassified.ordinal_61.fused_ffn | fused_ffn | 1 | 20 | 0.000 | 0.768 | n/a | 0.000 |
| unclassified.ordinal_129.fused_ffn | fused_ffn | 1 | 20 | 0.000 | 0.767 | n/a | 0.000 |
| unclassified.ordinal_25.fused_ffn | fused_ffn | 1 | 20 | 0.000 | 0.767 | n/a | 0.000 |
| unclassified.ordinal_33.fused_ffn | fused_ffn | 1 | 20 | 0.000 | 0.767 | n/a | 0.000 |
| unclassified.ordinal_101.fused_ffn | fused_ffn | 1 | 20 | 0.000 | 0.765 | n/a | 0.000 |
| unclassified.ordinal_89.fused_ffn | fused_ffn | 1 | 20 | 0.000 | 0.757 | n/a | 0.000 |
| unclassified.ordinal_305.fused_ffn | fused_ffn | 1 | 20 | 0.000 | 0.749 | n/a | 0.000 |
| unclassified.ordinal_297.fused_ffn | fused_ffn | 1 | 20 | 0.000 | 0.720 | n/a | 0.000 |
| unclassified.ordinal_229.fused_ffn | fused_ffn | 1 | 20 | 0.000 | 0.716 | n/a | 0.000 |
| unclassified.ordinal_213.fused_ffn | fused_ffn | 1 | 20 | 0.000 | 0.716 | n/a | 0.000 |
| unclassified.ordinal_221.fused_ffn | fused_ffn | 1 | 20 | 0.000 | 0.716 | n/a | 0.000 |
| unclassified.ordinal_145.fused_ffn | fused_ffn | 1 | 20 | 0.000 | 0.707 | n/a | 0.000 |
| unclassified.ordinal_201.fused_ffn | fused_ffn | 1 | 20 | 0.000 | 0.681 | n/a | 0.000 |
| unclassified.ordinal_230.linear2_residual | linear2_residual | 1 | 20 | 0.000 | 0.422 | n/a | 0.000 |
| unclassified.ordinal_214.linear2_residual | linear2_residual | 1 | 20 | 0.000 | 0.422 | n/a | 0.000 |
| unclassified.ordinal_202.linear2_residual | linear2_residual | 1 | 20 | 0.000 | 0.421 | n/a | 0.000 |
| unclassified.ordinal_222.linear2_residual | linear2_residual | 1 | 20 | 0.000 | 0.421 | n/a | 0.000 |
| unclassified.ordinal_242.linear2_residual | linear2_residual | 1 | 20 | 0.000 | 0.420 | n/a | 0.000 |
| unclassified.ordinal_18.linear2_residual | linear2_residual | 1 | 20 | 0.000 | 0.420 | n/a | 0.000 |
| unclassified.ordinal_146.linear2_residual | linear2_residual | 1 | 20 | 0.000 | 0.420 | n/a | 0.000 |
| unclassified.ordinal_250.linear2_residual | linear2_residual | 1 | 20 | 0.000 | 0.419 | n/a | 0.000 |
| unclassified.ordinal_314.linear2_residual | linear2_residual | 1 | 20 | 0.000 | 0.419 | n/a | 0.000 |
| unclassified.ordinal_194.linear2_residual | linear2_residual | 1 | 20 | 0.000 | 0.419 | n/a | 0.000 |
| unclassified.ordinal_158.linear2_residual | linear2_residual | 1 | 20 | 0.000 | 0.419 | n/a | 0.000 |
| unclassified.ordinal_258.linear2_residual | linear2_residual | 1 | 20 | 0.000 | 0.419 | n/a | 0.000 |
| unclassified.ordinal_306.linear2_residual | linear2_residual | 1 | 20 | 0.000 | 0.418 | n/a | 0.000 |
| unclassified.ordinal_298.linear2_residual | linear2_residual | 1 | 20 | 0.000 | 0.418 | n/a | 0.000 |
| unclassified.ordinal_26.linear2_residual | linear2_residual | 1 | 20 | 0.000 | 0.418 | n/a | 0.000 |
| unclassified.ordinal_130.linear2_residual | linear2_residual | 1 | 20 | 0.000 | 0.417 | n/a | 0.000 |
| unclassified.ordinal_186.linear2_residual | linear2_residual | 1 | 20 | 0.000 | 0.417 | n/a | 0.000 |
| unclassified.ordinal_46.linear2_residual | linear2_residual | 1 | 20 | 0.000 | 0.417 | n/a | 0.000 |
| unclassified.ordinal_138.linear2_residual | linear2_residual | 1 | 20 | 0.000 | 0.417 | n/a | 0.000 |
| unclassified.ordinal_90.linear2_residual | linear2_residual | 1 | 20 | 0.000 | 0.417 | n/a | 0.000 |
| unclassified.ordinal_82.linear2_residual | linear2_residual | 1 | 20 | 0.000 | 0.417 | n/a | 0.000 |
| unclassified.ordinal_166.linear2_residual | linear2_residual | 1 | 20 | 0.000 | 0.416 | n/a | 0.000 |
| unclassified.ordinal_74.linear2_residual | linear2_residual | 1 | 20 | 0.000 | 0.416 | n/a | 0.000 |
| unclassified.ordinal_270.linear2_residual | linear2_residual | 1 | 20 | 0.000 | 0.416 | n/a | 0.000 |
| unclassified.ordinal_34.linear2_residual | linear2_residual | 1 | 20 | 0.000 | 0.415 | n/a | 0.000 |
| unclassified.ordinal_102.linear2_residual | linear2_residual | 1 | 20 | 0.000 | 0.415 | n/a | 0.000 |
| unclassified.ordinal_62.linear2_residual | linear2_residual | 1 | 20 | 0.000 | 0.415 | n/a | 0.000 |
| unclassified.ordinal_174.linear2_residual | linear2_residual | 1 | 20 | 0.000 | 0.415 | n/a | 0.000 |
| unclassified.ordinal_54.linear2_residual | linear2_residual | 1 | 20 | 0.000 | 0.414 | n/a | 0.000 |
| unclassified.ordinal_278.linear2_residual | linear2_residual | 1 | 20 | 0.000 | 0.413 | n/a | 0.000 |
| unclassified.ordinal_110.linear2_residual | linear2_residual | 1 | 20 | 0.000 | 0.412 | n/a | 0.000 |
| unclassified.ordinal_286.linear2_residual | linear2_residual | 1 | 20 | 0.000 | 0.412 | n/a | 0.000 |
| unclassified.ordinal_118.linear2_residual | linear2_residual | 1 | 20 | 0.000 | 0.410 | n/a | 0.000 |
| frontend.initial_conv | cudnn | 1 | 20 | 0.000 | 0.392 | n/a | 0.000 |
| unclassified.ordinal_232.library_gemm | library_gemm | 1 | 20 | 0.000 | 0.293 | n/a | 0.000 |
| unclassified.ordinal_148.library_gemm | library_gemm | 1 | 20 | 0.000 | 0.291 | n/a | 0.000 |
| unclassified.ordinal_204.library_gemm | library_gemm | 1 | 20 | 0.000 | 0.291 | n/a | 0.000 |
| unclassified.ordinal_92.library_gemm | library_gemm | 1 | 20 | 0.000 | 0.290 | n/a | 0.000 |
| unclassified.ordinal_260.library_gemm | library_gemm | 1 | 20 | 0.000 | 0.289 | n/a | 0.000 |
| unclassified.ordinal_316.library_gemm | library_gemm | 1 | 20 | 0.000 | 0.289 | n/a | 0.000 |
| unclassified.ordinal_64.library_gemm | library_gemm | 1 | 20 | 0.000 | 0.287 | n/a | 0.000 |
| unclassified.ordinal_176.library_gemm | library_gemm | 1 | 20 | 0.000 | 0.287 | n/a | 0.000 |
| unclassified.ordinal_36.library_gemm | library_gemm | 1 | 20 | 0.000 | 0.287 | n/a | 0.000 |
| unclassified.ordinal_208.copy_reformat | copy_reformat | 1 | 20 | 0.000 | 0.286 | n/a | 0.000 |
| unclassified.ordinal_196.copy_reformat | copy_reformat | 1 | 20 | 0.000 | 0.286 | n/a | 0.000 |
| unclassified.ordinal_216.copy_reformat | copy_reformat | 1 | 20 | 0.000 | 0.286 | n/a | 0.000 |
| unclassified.ordinal_236.copy_reformat | copy_reformat | 1 | 20 | 0.000 | 0.285 | n/a | 0.000 |
| unclassified.ordinal_300.copy_reformat | copy_reformat | 1 | 20 | 0.000 | 0.285 | n/a | 0.000 |
| unclassified.ordinal_96.copy_reformat | copy_reformat | 1 | 20 | 0.000 | 0.285 | n/a | 0.000 |
| unclassified.ordinal_288.library_gemm | library_gemm | 1 | 20 | 0.000 | 0.285 | n/a | 0.000 |
| unclassified.ordinal_224.copy_reformat | copy_reformat | 1 | 20 | 0.000 | 0.285 | n/a | 0.000 |
| unclassified.ordinal_308.copy_reformat | copy_reformat | 1 | 20 | 0.000 | 0.285 | n/a | 0.000 |
| unclassified.ordinal_160.copy_reformat | copy_reformat | 1 | 20 | 0.000 | 0.285 | n/a | 0.000 |
| unclassified.ordinal_20.copy_reformat | copy_reformat | 1 | 20 | 0.000 | 0.285 | n/a | 0.000 |
| unclassified.ordinal_120.library_gemm | library_gemm | 1 | 20 | 0.000 | 0.285 | n/a | 0.000 |
| unclassified.ordinal_252.copy_reformat | copy_reformat | 1 | 20 | 0.000 | 0.285 | n/a | 0.000 |
| unclassified.ordinal_244.copy_reformat | copy_reformat | 1 | 20 | 0.000 | 0.284 | n/a | 0.000 |
| unclassified.ordinal_48.copy_reformat | copy_reformat | 1 | 20 | 0.000 | 0.284 | n/a | 0.000 |
| unclassified.ordinal_132.copy_reformat | copy_reformat | 1 | 20 | 0.000 | 0.284 | n/a | 0.000 |
| unclassified.ordinal_272.copy_reformat | copy_reformat | 1 | 20 | 0.000 | 0.284 | n/a | 0.000 |
| unclassified.ordinal_112.copy_reformat | copy_reformat | 1 | 20 | 0.000 | 0.284 | n/a | 0.000 |
| unclassified.ordinal_56.copy_reformat | copy_reformat | 1 | 20 | 0.000 | 0.284 | n/a | 0.000 |
| unclassified.ordinal_168.copy_reformat | copy_reformat | 1 | 20 | 0.000 | 0.283 | n/a | 0.000 |
| unclassified.ordinal_264.copy_reformat | copy_reformat | 1 | 20 | 0.000 | 0.283 | n/a | 0.000 |
| unclassified.ordinal_152.copy_reformat | copy_reformat | 1 | 20 | 0.000 | 0.283 | n/a | 0.000 |
| unclassified.ordinal_188.copy_reformat | copy_reformat | 1 | 20 | 0.000 | 0.282 | n/a | 0.000 |
| unclassified.ordinal_140.copy_reformat | copy_reformat | 1 | 20 | 0.000 | 0.282 | n/a | 0.000 |
| unclassified.ordinal_76.copy_reformat | copy_reformat | 1 | 20 | 0.000 | 0.282 | n/a | 0.000 |
| unclassified.ordinal_84.copy_reformat | copy_reformat | 1 | 20 | 0.000 | 0.282 | n/a | 0.000 |
| unclassified.ordinal_104.copy_reformat | copy_reformat | 1 | 20 | 0.000 | 0.282 | n/a | 0.000 |
| unclassified.ordinal_40.copy_reformat | copy_reformat | 1 | 20 | 0.000 | 0.282 | n/a | 0.000 |
| unclassified.ordinal_28.copy_reformat | copy_reformat | 1 | 20 | 0.000 | 0.282 | n/a | 0.000 |
| unclassified.ordinal_68.copy_reformat | copy_reformat | 1 | 20 | 0.000 | 0.281 | n/a | 0.000 |
| unclassified.ordinal_180.copy_reformat | copy_reformat | 1 | 20 | 0.000 | 0.281 | n/a | 0.000 |
| unclassified.ordinal_280.copy_reformat | copy_reformat | 1 | 20 | 0.000 | 0.281 | n/a | 0.000 |
| unclassified.ordinal_124.copy_reformat | copy_reformat | 1 | 20 | 0.000 | 0.281 | n/a | 0.000 |
| unclassified.ordinal_292.copy_reformat | copy_reformat | 1 | 20 | 0.000 | 0.280 | n/a | 0.000 |
| unclassified.ordinal_12.copy_reformat | copy_reformat | 1 | 20 | 0.000 | 0.279 | n/a | 0.000 |
| unclassified.ordinal_234.library_gemm | library_gemm | 1 | 20 | 0.000 | 0.241 | n/a | 0.000 |
| unclassified.ordinal_150.library_gemm | library_gemm | 1 | 20 | 0.000 | 0.241 | n/a | 0.000 |
| unclassified.ordinal_206.library_gemm | library_gemm | 1 | 20 | 0.000 | 0.241 | n/a | 0.000 |
| unclassified.ordinal_226.fa4 | fa4 | 1 | 20 | 0.000 | 0.241 | n/a | 0.000 |
| unclassified.ordinal_218.fa4 | fa4 | 1 | 20 | 0.000 | 0.241 | n/a | 0.000 |
| unclassified.ordinal_238.fa4 | fa4 | 1 | 20 | 0.000 | 0.240 | n/a | 0.000 |
| unclassified.ordinal_262.library_gemm | library_gemm | 1 | 20 | 0.000 | 0.240 | n/a | 0.000 |
| unclassified.ordinal_246.fa4 | fa4 | 1 | 20 | 0.000 | 0.240 | n/a | 0.000 |
| unclassified.ordinal_22.fa4 | fa4 | 1 | 20 | 0.000 | 0.240 | n/a | 0.000 |
| unclassified.ordinal_310.fa4 | fa4 | 1 | 20 | 0.000 | 0.240 | n/a | 0.000 |
| unclassified.ordinal_210.fa4 | fa4 | 1 | 20 | 0.000 | 0.239 | n/a | 0.000 |
| unclassified.ordinal_254.fa4 | fa4 | 1 | 20 | 0.000 | 0.239 | n/a | 0.000 |
| unclassified.ordinal_154.fa4 | fa4 | 1 | 20 | 0.000 | 0.239 | n/a | 0.000 |
| unclassified.ordinal_94.library_gemm | library_gemm | 1 | 20 | 0.000 | 0.239 | n/a | 0.000 |
| unclassified.ordinal_142.fa4 | fa4 | 1 | 20 | 0.000 | 0.239 | n/a | 0.000 |
| unclassified.ordinal_302.fa4 | fa4 | 1 | 20 | 0.000 | 0.239 | n/a | 0.000 |
| unclassified.ordinal_198.fa4 | fa4 | 1 | 20 | 0.000 | 0.239 | n/a | 0.000 |
| unclassified.ordinal_178.library_gemm | library_gemm | 1 | 20 | 0.000 | 0.239 | n/a | 0.000 |
| unclassified.ordinal_38.library_gemm | library_gemm | 1 | 20 | 0.000 | 0.239 | n/a | 0.000 |
| unclassified.ordinal_266.fa4 | fa4 | 1 | 20 | 0.000 | 0.239 | n/a | 0.000 |
| unclassified.ordinal_162.fa4 | fa4 | 1 | 20 | 0.000 | 0.238 | n/a | 0.000 |
| unclassified.ordinal_134.fa4 | fa4 | 1 | 20 | 0.000 | 0.238 | n/a | 0.000 |
| unclassified.ordinal_190.fa4 | fa4 | 1 | 20 | 0.000 | 0.238 | n/a | 0.000 |
| unclassified.ordinal_50.fa4 | fa4 | 1 | 20 | 0.000 | 0.238 | n/a | 0.000 |
| unclassified.ordinal_98.fa4 | fa4 | 1 | 20 | 0.000 | 0.238 | n/a | 0.000 |
| unclassified.ordinal_66.library_gemm | library_gemm | 1 | 20 | 0.000 | 0.238 | n/a | 0.000 |
| unclassified.ordinal_30.fa4 | fa4 | 1 | 20 | 0.000 | 0.238 | n/a | 0.000 |
| unclassified.ordinal_78.fa4 | fa4 | 1 | 20 | 0.000 | 0.238 | n/a | 0.000 |
| unclassified.ordinal_170.fa4 | fa4 | 1 | 20 | 0.000 | 0.238 | n/a | 0.000 |
| unclassified.ordinal_86.fa4 | fa4 | 1 | 20 | 0.000 | 0.238 | n/a | 0.000 |
| unclassified.ordinal_182.fa4 | fa4 | 1 | 20 | 0.000 | 0.238 | n/a | 0.000 |
| unclassified.ordinal_274.fa4 | fa4 | 1 | 20 | 0.000 | 0.238 | n/a | 0.000 |
| unclassified.ordinal_58.fa4 | fa4 | 1 | 20 | 0.000 | 0.237 | n/a | 0.000 |
| unclassified.ordinal_70.fa4 | fa4 | 1 | 20 | 0.000 | 0.237 | n/a | 0.000 |
| unclassified.ordinal_42.fa4 | fa4 | 1 | 20 | 0.000 | 0.237 | n/a | 0.000 |
| unclassified.ordinal_294.fa4 | fa4 | 1 | 20 | 0.000 | 0.236 | n/a | 0.000 |
| unclassified.ordinal_290.library_gemm | library_gemm | 1 | 20 | 0.000 | 0.236 | n/a | 0.000 |
| unclassified.ordinal_106.fa4 | fa4 | 1 | 20 | 0.000 | 0.236 | n/a | 0.000 |
| unclassified.ordinal_282.fa4 | fa4 | 1 | 20 | 0.000 | 0.236 | n/a | 0.000 |
| unclassified.ordinal_122.library_gemm | library_gemm | 1 | 20 | 0.000 | 0.235 | n/a | 0.000 |
| unclassified.ordinal_114.fa4 | fa4 | 1 | 20 | 0.000 | 0.235 | n/a | 0.000 |
| unclassified.ordinal_126.fa4 | fa4 | 1 | 20 | 0.000 | 0.235 | n/a | 0.000 |
| unclassified.ordinal_14.fa4 | fa4 | 1 | 20 | 0.000 | 0.234 | n/a | 0.000 |
| unclassified.ordinal_10.library_gemm | library_gemm | 1 | 20 | 0.000 | 0.234 | n/a | 0.000 |
| value.v2_matmul | library_gemm | 1 | 20 | 0.000 | 0.190 | n/a | 0.000 |
| unclassified.ordinal_239.library_gemm | library_gemm | 1 | 20 | 0.000 | 0.172 | n/a | 0.000 |
| unclassified.ordinal_219.library_gemm | library_gemm | 1 | 20 | 0.000 | 0.172 | n/a | 0.000 |
| unclassified.ordinal_227.library_gemm | library_gemm | 1 | 20 | 0.000 | 0.172 | n/a | 0.000 |
| unclassified.ordinal_211.library_gemm | library_gemm | 1 | 20 | 0.000 | 0.171 | n/a | 0.000 |
| unclassified.ordinal_155.library_gemm | library_gemm | 1 | 20 | 0.000 | 0.171 | n/a | 0.000 |
| unclassified.ordinal_23.library_gemm | library_gemm | 1 | 20 | 0.000 | 0.171 | n/a | 0.000 |
| unclassified.ordinal_87.library_gemm | library_gemm | 1 | 20 | 0.000 | 0.171 | n/a | 0.000 |
| unclassified.ordinal_43.library_gemm | library_gemm | 1 | 20 | 0.000 | 0.170 | n/a | 0.000 |
| unclassified.ordinal_247.library_gemm | library_gemm | 1 | 20 | 0.000 | 0.170 | n/a | 0.000 |
| unclassified.ordinal_267.library_gemm | library_gemm | 1 | 20 | 0.000 | 0.170 | n/a | 0.000 |
| unclassified.ordinal_99.library_gemm | library_gemm | 1 | 20 | 0.000 | 0.170 | n/a | 0.000 |
| unclassified.ordinal_303.library_gemm | library_gemm | 1 | 20 | 0.000 | 0.170 | n/a | 0.000 |
| unclassified.ordinal_51.library_gemm | library_gemm | 1 | 20 | 0.000 | 0.170 | n/a | 0.000 |
| unclassified.ordinal_183.library_gemm | library_gemm | 1 | 20 | 0.000 | 0.170 | n/a | 0.000 |
| unclassified.ordinal_143.library_gemm | library_gemm | 1 | 20 | 0.000 | 0.170 | n/a | 0.000 |
| unclassified.ordinal_191.library_gemm | library_gemm | 1 | 20 | 0.000 | 0.170 | n/a | 0.000 |
| unclassified.ordinal_163.library_gemm | library_gemm | 1 | 20 | 0.000 | 0.170 | n/a | 0.000 |
| unclassified.ordinal_255.library_gemm | library_gemm | 1 | 20 | 0.000 | 0.170 | n/a | 0.000 |
| unclassified.ordinal_79.library_gemm | library_gemm | 1 | 20 | 0.000 | 0.170 | n/a | 0.000 |
| unclassified.ordinal_31.library_gemm | library_gemm | 1 | 20 | 0.000 | 0.170 | n/a | 0.000 |
| unclassified.ordinal_311.library_gemm | library_gemm | 1 | 20 | 0.000 | 0.170 | n/a | 0.000 |
| unclassified.ordinal_135.library_gemm | library_gemm | 1 | 20 | 0.000 | 0.170 | n/a | 0.000 |
| unclassified.ordinal_275.library_gemm | library_gemm | 1 | 20 | 0.000 | 0.170 | n/a | 0.000 |
| unclassified.ordinal_283.library_gemm | library_gemm | 1 | 20 | 0.000 | 0.170 | n/a | 0.000 |
| unclassified.ordinal_15.library_gemm | library_gemm | 1 | 20 | 0.000 | 0.170 | n/a | 0.000 |
| unclassified.ordinal_171.library_gemm | library_gemm | 1 | 20 | 0.000 | 0.170 | n/a | 0.000 |
| unclassified.ordinal_127.library_gemm | library_gemm | 1 | 20 | 0.000 | 0.169 | n/a | 0.000 |
| unclassified.ordinal_71.library_gemm | library_gemm | 1 | 20 | 0.000 | 0.169 | n/a | 0.000 |
| unclassified.ordinal_199.library_gemm | library_gemm | 1 | 20 | 0.000 | 0.169 | n/a | 0.000 |
| unclassified.ordinal_295.library_gemm | library_gemm | 1 | 20 | 0.000 | 0.169 | n/a | 0.000 |
| unclassified.ordinal_59.library_gemm | library_gemm | 1 | 20 | 0.000 | 0.169 | n/a | 0.000 |
| unclassified.ordinal_107.library_gemm | library_gemm | 1 | 20 | 0.000 | 0.168 | n/a | 0.000 |
| unclassified.ordinal_115.library_gemm | library_gemm | 1 | 20 | 0.000 | 0.168 | n/a | 0.000 |
| value.v1_conv | library_gemm | 1 | 20 | 0.000 | 0.159 | n/a | 0.000 |
| frontend.initial_global_broadcast_add | head_elementwise | 1 | 20 | 0.000 | 0.155 | n/a | 0.000 |
| policy.p1_conv | library_gemm | 1 | 20 | 0.000 | 0.125 | n/a | 0.000 |
| policy.g1_conv | library_gemm | 1 | 20 | 0.000 | 0.118 | n/a | 0.000 |
| unclassified.ordinal_225.qk_rope | qk_rope | 1 | 20 | 0.000 | 0.115 | n/a | 0.000 |
| unclassified.ordinal_21.qk_rope | qk_rope | 1 | 20 | 0.000 | 0.114 | n/a | 0.000 |
| unclassified.ordinal_237.qk_rope | qk_rope | 1 | 20 | 0.000 | 0.114 | n/a | 0.000 |
| unclassified.ordinal_217.qk_rope | qk_rope | 1 | 20 | 0.000 | 0.114 | n/a | 0.000 |
| unclassified.ordinal_309.qk_rope | qk_rope | 1 | 20 | 0.000 | 0.113 | n/a | 0.000 |
| unclassified.ordinal_153.qk_rope | qk_rope | 1 | 20 | 0.000 | 0.113 | n/a | 0.000 |
| unclassified.ordinal_245.qk_rope | qk_rope | 1 | 20 | 0.000 | 0.113 | n/a | 0.000 |
| unclassified.ordinal_189.qk_rope | qk_rope | 1 | 20 | 0.000 | 0.113 | n/a | 0.000 |
| unclassified.ordinal_253.qk_rope | qk_rope | 1 | 20 | 0.000 | 0.113 | n/a | 0.000 |
| unclassified.ordinal_209.qk_rope | qk_rope | 1 | 20 | 0.000 | 0.113 | n/a | 0.000 |
| unclassified.ordinal_133.qk_rope | qk_rope | 1 | 20 | 0.000 | 0.113 | n/a | 0.000 |
| unclassified.ordinal_301.qk_rope | qk_rope | 1 | 20 | 0.000 | 0.113 | n/a | 0.000 |
| unclassified.ordinal_85.qk_rope | qk_rope | 1 | 20 | 0.000 | 0.113 | n/a | 0.000 |
| unclassified.ordinal_141.qk_rope | qk_rope | 1 | 20 | 0.000 | 0.113 | n/a | 0.000 |
| unclassified.ordinal_265.qk_rope | qk_rope | 1 | 20 | 0.000 | 0.113 | n/a | 0.000 |
| unclassified.ordinal_161.qk_rope | qk_rope | 1 | 20 | 0.000 | 0.113 | n/a | 0.000 |
| unclassified.ordinal_49.qk_rope | qk_rope | 1 | 20 | 0.000 | 0.113 | n/a | 0.000 |
| unclassified.ordinal_57.qk_rope | qk_rope | 1 | 20 | 0.000 | 0.112 | n/a | 0.000 |
| unclassified.ordinal_29.qk_rope | qk_rope | 1 | 20 | 0.000 | 0.112 | n/a | 0.000 |
| unclassified.ordinal_77.qk_rope | qk_rope | 1 | 20 | 0.000 | 0.112 | n/a | 0.000 |
| unclassified.ordinal_197.qk_rope | qk_rope | 1 | 20 | 0.000 | 0.112 | n/a | 0.000 |
| unclassified.ordinal_97.qk_rope | qk_rope | 1 | 20 | 0.000 | 0.112 | n/a | 0.000 |
| unclassified.ordinal_181.qk_rope | qk_rope | 1 | 20 | 0.000 | 0.112 | n/a | 0.000 |
| unclassified.ordinal_69.qk_rope | qk_rope | 1 | 20 | 0.000 | 0.112 | n/a | 0.000 |
| unclassified.ordinal_273.qk_rope | qk_rope | 1 | 20 | 0.000 | 0.112 | n/a | 0.000 |
| unclassified.ordinal_105.qk_rope | qk_rope | 1 | 20 | 0.000 | 0.112 | n/a | 0.000 |
| unclassified.ordinal_169.qk_rope | qk_rope | 1 | 20 | 0.000 | 0.112 | n/a | 0.000 |
| unclassified.ordinal_293.qk_rope | qk_rope | 1 | 20 | 0.000 | 0.112 | n/a | 0.000 |
| unclassified.ordinal_281.qk_rope | qk_rope | 1 | 20 | 0.000 | 0.112 | n/a | 0.000 |
| unclassified.ordinal_41.qk_rope | qk_rope | 1 | 20 | 0.000 | 0.111 | n/a | 0.000 |
| unclassified.ordinal_125.qk_rope | qk_rope | 1 | 20 | 0.000 | 0.111 | n/a | 0.000 |
| unclassified.ordinal_113.qk_rope | qk_rope | 1 | 20 | 0.000 | 0.111 | n/a | 0.000 |
| unclassified.ordinal_13.qk_rope | qk_rope | 1 | 20 | 0.000 | 0.110 | n/a | 0.000 |
| policy.gpool_to_bias_matmul | library_gemm | 1 | 20 | 0.000 | 0.108 | n/a | 0.000 |
| policy.gpool_to_pass_matmul | library_gemm | 1 | 20 | 0.000 | 0.106 | n/a | 0.000 |
| unclassified.ordinal_205.affine_silu | affine_silu | 1 | 20 | 0.000 | 0.102 | n/a | 0.000 |
| unclassified.ordinal_233.affine_silu | affine_silu | 1 | 20 | 0.000 | 0.102 | n/a | 0.000 |
| unclassified.ordinal_149.affine_silu | affine_silu | 1 | 20 | 0.000 | 0.102 | n/a | 0.000 |
| unclassified.ordinal_37.affine_silu | affine_silu | 1 | 20 | 0.000 | 0.102 | n/a | 0.000 |
| unclassified.ordinal_93.affine_silu | affine_silu | 1 | 20 | 0.000 | 0.101 | n/a | 0.000 |
| trunk.tip_norm_silu | affine_silu | 1 | 20 | 0.000 | 0.101 | n/a | 0.000 |
| unclassified.ordinal_261.affine_silu | affine_silu | 1 | 20 | 0.000 | 0.101 | n/a | 0.000 |
| unclassified.ordinal_177.affine_silu | affine_silu | 1 | 20 | 0.000 | 0.101 | n/a | 0.000 |
| unclassified.ordinal_65.affine_silu | affine_silu | 1 | 20 | 0.000 | 0.101 | n/a | 0.000 |
| unclassified.ordinal_289.affine_silu | affine_silu | 1 | 20 | 0.000 | 0.100 | n/a | 0.000 |
| unclassified.ordinal_121.affine_silu | affine_silu | 1 | 20 | 0.000 | 0.100 | n/a | 0.000 |
| unclassified.ordinal_9.affine_silu | affine_silu | 1 | 20 | 0.000 | 0.099 | n/a | 0.000 |
| policy.g1_global_pool | head_elementwise | 1 | 20 | 0.000 | 0.089 | n/a | 0.000 |
| value.ownership_conv | library_gemm | 1 | 20 | 0.000 | 0.081 | n/a | 0.000 |
| policy.p2_conv | library_gemm | 1 | 20 | 0.000 | 0.078 | n/a | 0.000 |
| value.score_matmul | library_gemm | 1 | 20 | 0.000 | 0.070 | n/a | 0.000 |
| value.v3_matmul | library_gemm | 1 | 20 | 0.000 | 0.070 | n/a | 0.000 |
| unclassified.ordinal_315.affine_silu | affine_silu | 1 | 20 | 0.000 | 0.068 | n/a | 0.000 |
| unclassified.ordinal_147.affine_silu | affine_silu | 1 | 20 | 0.000 | 0.068 | n/a | 0.000 |
| unclassified.ordinal_259.affine_silu | affine_silu | 1 | 20 | 0.000 | 0.067 | n/a | 0.000 |
| unclassified.ordinal_231.affine_silu | affine_silu | 1 | 20 | 0.000 | 0.067 | n/a | 0.000 |
| unclassified.ordinal_203.affine_silu | affine_silu | 1 | 20 | 0.000 | 0.067 | n/a | 0.000 |
| unclassified.ordinal_91.affine_silu | affine_silu | 1 | 20 | 0.000 | 0.067 | n/a | 0.000 |
| unclassified.ordinal_35.affine_silu | affine_silu | 1 | 20 | 0.000 | 0.067 | n/a | 0.000 |
| unclassified.ordinal_287.affine_silu | affine_silu | 1 | 20 | 0.000 | 0.067 | n/a | 0.000 |
| unclassified.ordinal_63.affine_silu | affine_silu | 1 | 20 | 0.000 | 0.066 | n/a | 0.000 |
| unclassified.ordinal_175.affine_silu | affine_silu | 1 | 20 | 0.000 | 0.066 | n/a | 0.000 |
| unclassified.ordinal_119.affine_silu | affine_silu | 1 | 20 | 0.000 | 0.066 | n/a | 0.000 |
| value.v1_global_pool | head_elementwise | 1 | 20 | 0.000 | 0.065 | n/a | 0.000 |
| value.v1_norm_silu | head_elementwise | 1 | 20 | 0.000 | 0.063 | n/a | 0.000 |
| frontend.initial_global_matmul | library_gemm | 1 | 20 | 0.000 | 0.053 | n/a | 0.000 |
| unclassified.ordinal_220.rmsnorm | rmsnorm | 1 | 20 | 0.000 | 0.050 | n/a | 0.000 |
| unclassified.ordinal_184.rmsnorm | rmsnorm | 1 | 20 | 0.000 | 0.050 | n/a | 0.000 |
| unclassified.ordinal_312.rmsnorm | rmsnorm | 1 | 20 | 0.000 | 0.050 | n/a | 0.000 |
| unclassified.ordinal_223.rmsnorm | rmsnorm | 1 | 20 | 0.000 | 0.050 | n/a | 0.000 |
| unclassified.ordinal_251.rmsnorm | rmsnorm | 1 | 20 | 0.000 | 0.050 | n/a | 0.000 |
| unclassified.ordinal_187.rmsnorm | rmsnorm | 1 | 20 | 0.000 | 0.050 | n/a | 0.000 |
| unclassified.ordinal_299.rmsnorm | rmsnorm | 1 | 20 | 0.000 | 0.050 | n/a | 0.000 |
| unclassified.ordinal_32.rmsnorm | rmsnorm | 1 | 20 | 0.000 | 0.050 | n/a | 0.000 |
| unclassified.ordinal_240.rmsnorm | rmsnorm | 1 | 20 | 0.000 | 0.050 | n/a | 0.000 |
| unclassified.ordinal_151.rmsnorm | rmsnorm | 1 | 20 | 0.000 | 0.049 | n/a | 0.000 |
| unclassified.ordinal_243.rmsnorm | rmsnorm | 1 | 20 | 0.000 | 0.049 | n/a | 0.000 |
| unclassified.ordinal_80.rmsnorm | rmsnorm | 1 | 20 | 0.000 | 0.049 | n/a | 0.000 |
| unclassified.ordinal_24.rmsnorm | rmsnorm | 1 | 20 | 0.000 | 0.049 | n/a | 0.000 |
| unclassified.ordinal_116.rmsnorm | rmsnorm | 1 | 20 | 0.000 | 0.049 | n/a | 0.000 |
| unclassified.ordinal_284.rmsnorm | rmsnorm | 1 | 20 | 0.000 | 0.049 | n/a | 0.000 |
| unclassified.ordinal_256.rmsnorm | rmsnorm | 1 | 20 | 0.000 | 0.049 | n/a | 0.000 |
| unclassified.ordinal_103.rmsnorm | rmsnorm | 1 | 20 | 0.000 | 0.049 | n/a | 0.000 |
| unclassified.ordinal_75.rmsnorm | rmsnorm | 1 | 20 | 0.000 | 0.049 | n/a | 0.000 |
| unclassified.ordinal_19.rmsnorm | rmsnorm | 1 | 20 | 0.000 | 0.049 | n/a | 0.000 |
| unclassified.ordinal_235.rmsnorm | rmsnorm | 1 | 20 | 0.000 | 0.049 | n/a | 0.000 |
| unclassified.ordinal_248.rmsnorm | rmsnorm | 1 | 20 | 0.000 | 0.049 | n/a | 0.000 |
| unclassified.ordinal_207.rmsnorm | rmsnorm | 1 | 20 | 0.000 | 0.049 | n/a | 0.000 |
| unclassified.ordinal_291.rmsnorm | rmsnorm | 1 | 20 | 0.000 | 0.049 | n/a | 0.000 |
| unclassified.ordinal_268.rmsnorm | rmsnorm | 1 | 20 | 0.000 | 0.049 | n/a | 0.000 |
| unclassified.ordinal_47.rmsnorm | rmsnorm | 1 | 20 | 0.000 | 0.049 | n/a | 0.000 |
| unclassified.ordinal_139.rmsnorm | rmsnorm | 1 | 20 | 0.000 | 0.049 | n/a | 0.000 |
| unclassified.ordinal_228.rmsnorm | rmsnorm | 1 | 20 | 0.000 | 0.049 | n/a | 0.000 |
| unclassified.ordinal_88.rmsnorm | rmsnorm | 1 | 20 | 0.000 | 0.049 | n/a | 0.000 |
| unclassified.ordinal_195.rmsnorm | rmsnorm | 1 | 20 | 0.000 | 0.049 | n/a | 0.000 |
| unclassified.ordinal_164.rmsnorm | rmsnorm | 1 | 20 | 0.000 | 0.049 | n/a | 0.000 |
| unclassified.ordinal_156.rmsnorm | rmsnorm | 1 | 20 | 0.000 | 0.049 | n/a | 0.000 |
| unclassified.ordinal_200.rmsnorm | rmsnorm | 1 | 20 | 0.000 | 0.049 | n/a | 0.000 |
| unclassified.ordinal_128.rmsnorm | rmsnorm | 1 | 20 | 0.000 | 0.049 | n/a | 0.000 |
| unclassified.ordinal_212.rmsnorm | rmsnorm | 1 | 20 | 0.000 | 0.049 | n/a | 0.000 |
| unclassified.ordinal_276.rmsnorm | rmsnorm | 1 | 20 | 0.000 | 0.049 | n/a | 0.000 |
| unclassified.ordinal_136.rmsnorm | rmsnorm | 1 | 20 | 0.000 | 0.049 | n/a | 0.000 |
| unclassified.ordinal_83.rmsnorm | rmsnorm | 1 | 20 | 0.000 | 0.049 | n/a | 0.000 |
| unclassified.ordinal_100.rmsnorm | rmsnorm | 1 | 20 | 0.000 | 0.049 | n/a | 0.000 |
| unclassified.ordinal_52.rmsnorm | rmsnorm | 1 | 20 | 0.000 | 0.049 | n/a | 0.000 |
| unclassified.ordinal_72.rmsnorm | rmsnorm | 1 | 20 | 0.000 | 0.049 | n/a | 0.000 |
| unclassified.ordinal_144.rmsnorm | rmsnorm | 1 | 20 | 0.000 | 0.049 | n/a | 0.000 |
| unclassified.ordinal_123.rmsnorm | rmsnorm | 1 | 20 | 0.000 | 0.049 | n/a | 0.000 |
| unclassified.ordinal_16.rmsnorm | rmsnorm | 1 | 20 | 0.000 | 0.049 | n/a | 0.000 |
| unclassified.ordinal_60.rmsnorm | rmsnorm | 1 | 20 | 0.000 | 0.049 | n/a | 0.000 |
| unclassified.ordinal_215.rmsnorm | rmsnorm | 1 | 20 | 0.000 | 0.049 | n/a | 0.000 |
| unclassified.ordinal_304.rmsnorm | rmsnorm | 1 | 20 | 0.000 | 0.049 | n/a | 0.000 |
| unclassified.ordinal_271.rmsnorm | rmsnorm | 1 | 20 | 0.000 | 0.049 | n/a | 0.000 |
| unclassified.ordinal_159.rmsnorm | rmsnorm | 1 | 20 | 0.000 | 0.049 | n/a | 0.000 |
| unclassified.ordinal_296.rmsnorm | rmsnorm | 1 | 20 | 0.000 | 0.049 | n/a | 0.000 |
| unclassified.ordinal_44.rmsnorm | rmsnorm | 1 | 20 | 0.000 | 0.049 | n/a | 0.000 |
| unclassified.ordinal_263.rmsnorm | rmsnorm | 1 | 20 | 0.000 | 0.049 | n/a | 0.000 |
| unclassified.ordinal_179.rmsnorm | rmsnorm | 1 | 20 | 0.000 | 0.049 | n/a | 0.000 |
| unclassified.ordinal_167.rmsnorm | rmsnorm | 1 | 20 | 0.000 | 0.049 | n/a | 0.000 |
| unclassified.ordinal_39.rmsnorm | rmsnorm | 1 | 20 | 0.000 | 0.049 | n/a | 0.000 |
| unclassified.ordinal_192.rmsnorm | rmsnorm | 1 | 20 | 0.000 | 0.049 | n/a | 0.000 |
| unclassified.ordinal_307.rmsnorm | rmsnorm | 1 | 20 | 0.000 | 0.049 | n/a | 0.000 |
| unclassified.ordinal_67.rmsnorm | rmsnorm | 1 | 20 | 0.000 | 0.049 | n/a | 0.000 |
| unclassified.ordinal_95.rmsnorm | rmsnorm | 1 | 20 | 0.000 | 0.049 | n/a | 0.000 |
| unclassified.ordinal_108.rmsnorm | rmsnorm | 1 | 20 | 0.000 | 0.049 | n/a | 0.000 |
| unclassified.ordinal_55.rmsnorm | rmsnorm | 1 | 20 | 0.000 | 0.048 | n/a | 0.000 |
| unclassified.ordinal_131.rmsnorm | rmsnorm | 1 | 20 | 0.000 | 0.048 | n/a | 0.000 |
| unclassified.ordinal_27.rmsnorm | rmsnorm | 1 | 20 | 0.000 | 0.048 | n/a | 0.000 |
| unclassified.ordinal_172.rmsnorm | rmsnorm | 1 | 20 | 0.000 | 0.048 | n/a | 0.000 |
| unclassified.ordinal_279.rmsnorm | rmsnorm | 1 | 20 | 0.000 | 0.048 | n/a | 0.000 |
| unclassified.ordinal_11.rmsnorm | rmsnorm | 1 | 20 | 0.000 | 0.048 | n/a | 0.000 |
| unclassified.ordinal_111.rmsnorm | rmsnorm | 1 | 20 | 0.000 | 0.048 | n/a | 0.000 |
| policy.gpool_to_pass_matmul2 | library_gemm | 1 | 20 | 0.000 | 0.046 | n/a | 0.000 |
| policy.p1_norm_silu | head_elementwise | 1 | 20 | 0.000 | 0.043 | n/a | 0.000 |
| value.v1_half_to_float | copy_reformat | 1 | 20 | 0.000 | 0.043 | n/a | 0.000 |
| policy.g1_norm_silu | head_elementwise | 1 | 20 | 0.000 | 0.043 | n/a | 0.000 |
| policy.gpool_bias_add | head_elementwise | 1 | 20 | 0.000 | 0.036 | n/a | 0.000 |
| input.mask_sum | sumChannelsNCHWKernel | 1 | 20 | 0.000 | 0.034 | n/a | 0.000 |
| policy.g1_half_to_float | copy_reformat | 1 | 20 | 0.000 | 0.031 | n/a | 0.000 |
| frontend.initial_conv_nhwc_padding_1 | cudnn | 1 | 20 | 0.000 | 0.031 | n/a | 0.000 |
| policy.p1_half_to_float | copy_reformat | 1 | 20 | 0.000 | 0.030 | n/a | 0.000 |
| value.ownership_conv_splitk_reduce | library_gemm | 1 | 20 | 0.000 | 0.028 | n/a | 0.000 |
| frontend.initial_conv_nhwc_padding_0 | cudnn | 1 | 20 | 0.000 | 0.026 | n/a | 0.000 |
| frontend.initial_global_matmul_splitk_reduce | library_gemm | 1 | 20 | 0.000 | 0.025 | n/a | 0.000 |
| input.extract_mask | head_elementwise | 1 | 20 | 0.000 | 0.024 | n/a | 0.000 |
| value.v2_bias_silu | head_elementwise | 1 | 20 | 0.000 | 0.021 | n/a | 0.000 |
| policy.pass_bias_silu | head_elementwise | 1 | 20 | 0.000 | 0.020 | n/a | 0.000 |
| value.v3_bias | head_elementwise | 1 | 20 | 0.000 | 0.019 | n/a | 0.000 |
| input.mask_half_to_float | copy_reformat | 1 | 20 | 0.000 | 0.019 | n/a | 0.000 |
| value.score_bias | head_elementwise | 1 | 20 | 0.000 | 0.019 | n/a | 0.000 |
| value.ownership_half_to_float | copy_reformat | 1 | 20 | 0.000 | 0.019 | n/a | 0.000 |

## `library_gemm` logical breakdown

| logical group | positions | calls | isolated reference ms | S2 total ms | total ratio | excess ms |
|---|---:|---:|---:|---:|---:|---:|
| unclassified.ordinal_232.library_gemm | 1 | 20 | 0.000 | 0.293 | n/a | 0.000 |
| unclassified.ordinal_148.library_gemm | 1 | 20 | 0.000 | 0.291 | n/a | 0.000 |
| unclassified.ordinal_204.library_gemm | 1 | 20 | 0.000 | 0.291 | n/a | 0.000 |
| unclassified.ordinal_92.library_gemm | 1 | 20 | 0.000 | 0.290 | n/a | 0.000 |
| unclassified.ordinal_260.library_gemm | 1 | 20 | 0.000 | 0.289 | n/a | 0.000 |
| unclassified.ordinal_316.library_gemm | 1 | 20 | 0.000 | 0.289 | n/a | 0.000 |
| unclassified.ordinal_64.library_gemm | 1 | 20 | 0.000 | 0.287 | n/a | 0.000 |
| unclassified.ordinal_176.library_gemm | 1 | 20 | 0.000 | 0.287 | n/a | 0.000 |
| unclassified.ordinal_36.library_gemm | 1 | 20 | 0.000 | 0.287 | n/a | 0.000 |
| unclassified.ordinal_288.library_gemm | 1 | 20 | 0.000 | 0.285 | n/a | 0.000 |
| unclassified.ordinal_120.library_gemm | 1 | 20 | 0.000 | 0.285 | n/a | 0.000 |
| unclassified.ordinal_234.library_gemm | 1 | 20 | 0.000 | 0.241 | n/a | 0.000 |
| unclassified.ordinal_150.library_gemm | 1 | 20 | 0.000 | 0.241 | n/a | 0.000 |
| unclassified.ordinal_206.library_gemm | 1 | 20 | 0.000 | 0.241 | n/a | 0.000 |
| unclassified.ordinal_262.library_gemm | 1 | 20 | 0.000 | 0.240 | n/a | 0.000 |
| unclassified.ordinal_94.library_gemm | 1 | 20 | 0.000 | 0.239 | n/a | 0.000 |
| unclassified.ordinal_178.library_gemm | 1 | 20 | 0.000 | 0.239 | n/a | 0.000 |
| unclassified.ordinal_38.library_gemm | 1 | 20 | 0.000 | 0.239 | n/a | 0.000 |
| unclassified.ordinal_66.library_gemm | 1 | 20 | 0.000 | 0.238 | n/a | 0.000 |
| unclassified.ordinal_290.library_gemm | 1 | 20 | 0.000 | 0.236 | n/a | 0.000 |
| unclassified.ordinal_122.library_gemm | 1 | 20 | 0.000 | 0.235 | n/a | 0.000 |
| unclassified.ordinal_10.library_gemm | 1 | 20 | 0.000 | 0.234 | n/a | 0.000 |
| value.v2_matmul | 1 | 20 | 0.000 | 0.190 | n/a | 0.000 |
| unclassified.ordinal_239.library_gemm | 1 | 20 | 0.000 | 0.172 | n/a | 0.000 |
| unclassified.ordinal_219.library_gemm | 1 | 20 | 0.000 | 0.172 | n/a | 0.000 |
| unclassified.ordinal_227.library_gemm | 1 | 20 | 0.000 | 0.172 | n/a | 0.000 |
| unclassified.ordinal_211.library_gemm | 1 | 20 | 0.000 | 0.171 | n/a | 0.000 |
| unclassified.ordinal_155.library_gemm | 1 | 20 | 0.000 | 0.171 | n/a | 0.000 |
| unclassified.ordinal_23.library_gemm | 1 | 20 | 0.000 | 0.171 | n/a | 0.000 |
| unclassified.ordinal_87.library_gemm | 1 | 20 | 0.000 | 0.171 | n/a | 0.000 |
| unclassified.ordinal_43.library_gemm | 1 | 20 | 0.000 | 0.170 | n/a | 0.000 |
| unclassified.ordinal_247.library_gemm | 1 | 20 | 0.000 | 0.170 | n/a | 0.000 |
| unclassified.ordinal_267.library_gemm | 1 | 20 | 0.000 | 0.170 | n/a | 0.000 |
| unclassified.ordinal_99.library_gemm | 1 | 20 | 0.000 | 0.170 | n/a | 0.000 |
| unclassified.ordinal_303.library_gemm | 1 | 20 | 0.000 | 0.170 | n/a | 0.000 |
| unclassified.ordinal_51.library_gemm | 1 | 20 | 0.000 | 0.170 | n/a | 0.000 |
| unclassified.ordinal_183.library_gemm | 1 | 20 | 0.000 | 0.170 | n/a | 0.000 |
| unclassified.ordinal_143.library_gemm | 1 | 20 | 0.000 | 0.170 | n/a | 0.000 |
| unclassified.ordinal_191.library_gemm | 1 | 20 | 0.000 | 0.170 | n/a | 0.000 |
| unclassified.ordinal_163.library_gemm | 1 | 20 | 0.000 | 0.170 | n/a | 0.000 |
| unclassified.ordinal_255.library_gemm | 1 | 20 | 0.000 | 0.170 | n/a | 0.000 |
| unclassified.ordinal_79.library_gemm | 1 | 20 | 0.000 | 0.170 | n/a | 0.000 |
| unclassified.ordinal_31.library_gemm | 1 | 20 | 0.000 | 0.170 | n/a | 0.000 |
| unclassified.ordinal_311.library_gemm | 1 | 20 | 0.000 | 0.170 | n/a | 0.000 |
| unclassified.ordinal_135.library_gemm | 1 | 20 | 0.000 | 0.170 | n/a | 0.000 |
| unclassified.ordinal_275.library_gemm | 1 | 20 | 0.000 | 0.170 | n/a | 0.000 |
| unclassified.ordinal_283.library_gemm | 1 | 20 | 0.000 | 0.170 | n/a | 0.000 |
| unclassified.ordinal_15.library_gemm | 1 | 20 | 0.000 | 0.170 | n/a | 0.000 |
| unclassified.ordinal_171.library_gemm | 1 | 20 | 0.000 | 0.170 | n/a | 0.000 |
| unclassified.ordinal_127.library_gemm | 1 | 20 | 0.000 | 0.169 | n/a | 0.000 |
| unclassified.ordinal_71.library_gemm | 1 | 20 | 0.000 | 0.169 | n/a | 0.000 |
| unclassified.ordinal_199.library_gemm | 1 | 20 | 0.000 | 0.169 | n/a | 0.000 |
| unclassified.ordinal_295.library_gemm | 1 | 20 | 0.000 | 0.169 | n/a | 0.000 |
| unclassified.ordinal_59.library_gemm | 1 | 20 | 0.000 | 0.169 | n/a | 0.000 |
| unclassified.ordinal_107.library_gemm | 1 | 20 | 0.000 | 0.168 | n/a | 0.000 |
| unclassified.ordinal_115.library_gemm | 1 | 20 | 0.000 | 0.168 | n/a | 0.000 |
| value.v1_conv | 1 | 20 | 0.000 | 0.159 | n/a | 0.000 |
| policy.p1_conv | 1 | 20 | 0.000 | 0.125 | n/a | 0.000 |
| policy.g1_conv | 1 | 20 | 0.000 | 0.118 | n/a | 0.000 |
| policy.gpool_to_bias_matmul | 1 | 20 | 0.000 | 0.108 | n/a | 0.000 |
| policy.gpool_to_pass_matmul | 1 | 20 | 0.000 | 0.106 | n/a | 0.000 |
| value.ownership_conv | 1 | 20 | 0.000 | 0.081 | n/a | 0.000 |
| policy.p2_conv | 1 | 20 | 0.000 | 0.078 | n/a | 0.000 |
| value.score_matmul | 1 | 20 | 0.000 | 0.070 | n/a | 0.000 |
| value.v3_matmul | 1 | 20 | 0.000 | 0.070 | n/a | 0.000 |
| frontend.initial_global_matmul | 1 | 20 | 0.000 | 0.053 | n/a | 0.000 |
| policy.gpool_to_pass_matmul2 | 1 | 20 | 0.000 | 0.046 | n/a | 0.000 |
| value.ownership_conv_splitk_reduce | 1 | 20 | 0.000 | 0.028 | n/a | 0.000 |
| frontend.initial_global_matmul_splitk_reduce | 1 | 20 | 0.000 | 0.025 | n/a | 0.000 |

## Top ordinal hotspots by summed excess

The worst peer is the highest median S2/S1 slowdown among peer families observed at least four times for that ordinal.

| rank | ordinal | logical position | family | calls | isolated us | S2 us | S2/S1 | excess ms | common peer | worst peer |
|---:|---:|---|---|---:|---:|---:|---:|---:|---|---|
| 1 | 157 | unclassified.ordinal_157.fused_ffn | fused_ffn | 20 | n/a | 41.152 | n/a | 0.000 | idle (20) | n/a |
| 2 | 81 | unclassified.ordinal_81.fused_ffn | fused_ffn | 20 | n/a | 40.977 | n/a | 0.000 | idle (20) | n/a |
| 3 | 249 | unclassified.ordinal_249.fused_ffn | fused_ffn | 20 | n/a | 40.800 | n/a | 0.000 | idle (20) | n/a |
| 4 | 241 | unclassified.ordinal_241.fused_ffn | fused_ffn | 20 | n/a | 40.769 | n/a | 0.000 | idle (20) | n/a |
| 5 | 313 | unclassified.ordinal_313.fused_ffn | fused_ffn | 20 | n/a | 40.833 | n/a | 0.000 | idle (20) | n/a |
| 6 | 185 | unclassified.ordinal_185.fused_ffn | fused_ffn | 20 | n/a | 40.560 | n/a | 0.000 | idle (20) | n/a |
| 7 | 117 | unclassified.ordinal_117.fused_ffn | fused_ffn | 20 | n/a | 40.608 | n/a | 0.000 | idle (20) | n/a |
| 8 | 257 | unclassified.ordinal_257.fused_ffn | fused_ffn | 20 | n/a | 40.273 | n/a | 0.000 | idle (20) | n/a |
| 9 | 109 | unclassified.ordinal_109.fused_ffn | fused_ffn | 20 | n/a | 40.193 | n/a | 0.000 | idle (20) | n/a |
| 10 | 165 | unclassified.ordinal_165.fused_ffn | fused_ffn | 20 | n/a | 40.816 | n/a | 0.000 | idle (20) | n/a |
| 11 | 73 | unclassified.ordinal_73.fused_ffn | fused_ffn | 20 | n/a | 40.257 | n/a | 0.000 | idle (20) | n/a |
| 12 | 45 | unclassified.ordinal_45.fused_ffn | fused_ffn | 20 | n/a | 39.776 | n/a | 0.000 | idle (20) | n/a |
| 13 | 193 | unclassified.ordinal_193.fused_ffn | fused_ffn | 20 | n/a | 39.328 | n/a | 0.000 | idle (20) | n/a |
| 14 | 17 | unclassified.ordinal_17.fused_ffn | fused_ffn | 20 | n/a | 39.312 | n/a | 0.000 | idle (20) | n/a |
| 15 | 53 | unclassified.ordinal_53.fused_ffn | fused_ffn | 20 | n/a | 38.577 | n/a | 0.000 | idle (20) | n/a |
| 16 | 285 | unclassified.ordinal_285.fused_ffn | fused_ffn | 20 | n/a | 38.288 | n/a | 0.000 | idle (20) | n/a |
| 17 | 173 | unclassified.ordinal_173.fused_ffn | fused_ffn | 20 | n/a | 38.849 | n/a | 0.000 | idle (20) | n/a |
| 18 | 269 | unclassified.ordinal_269.fused_ffn | fused_ffn | 20 | n/a | 38.752 | n/a | 0.000 | idle (20) | n/a |
| 19 | 277 | unclassified.ordinal_277.fused_ffn | fused_ffn | 20 | n/a | 38.449 | n/a | 0.000 | idle (20) | n/a |
| 20 | 137 | unclassified.ordinal_137.fused_ffn | fused_ffn | 20 | n/a | 38.416 | n/a | 0.000 | idle (20) | n/a |
| 21 | 61 | unclassified.ordinal_61.fused_ffn | fused_ffn | 20 | n/a | 38.529 | n/a | 0.000 | idle (20) | n/a |
| 22 | 129 | unclassified.ordinal_129.fused_ffn | fused_ffn | 20 | n/a | 38.624 | n/a | 0.000 | idle (20) | n/a |
| 23 | 25 | unclassified.ordinal_25.fused_ffn | fused_ffn | 20 | n/a | 38.192 | n/a | 0.000 | idle (20) | n/a |
| 24 | 33 | unclassified.ordinal_33.fused_ffn | fused_ffn | 20 | n/a | 38.304 | n/a | 0.000 | idle (20) | n/a |
| 25 | 101 | unclassified.ordinal_101.fused_ffn | fused_ffn | 20 | n/a | 38.032 | n/a | 0.000 | idle (20) | n/a |
| 26 | 89 | unclassified.ordinal_89.fused_ffn | fused_ffn | 20 | n/a | 37.920 | n/a | 0.000 | idle (20) | n/a |
| 27 | 305 | unclassified.ordinal_305.fused_ffn | fused_ffn | 20 | n/a | 37.361 | n/a | 0.000 | idle (20) | n/a |
| 28 | 297 | unclassified.ordinal_297.fused_ffn | fused_ffn | 20 | n/a | 36.065 | n/a | 0.000 | idle (20) | n/a |
| 29 | 229 | unclassified.ordinal_229.fused_ffn | fused_ffn | 20 | n/a | 35.840 | n/a | 0.000 | idle (20) | n/a |
| 30 | 213 | unclassified.ordinal_213.fused_ffn | fused_ffn | 20 | n/a | 35.776 | n/a | 0.000 | idle (20) | n/a |
| 31 | 221 | unclassified.ordinal_221.fused_ffn | fused_ffn | 20 | n/a | 35.904 | n/a | 0.000 | idle (20) | n/a |
| 32 | 145 | unclassified.ordinal_145.fused_ffn | fused_ffn | 20 | n/a | 35.441 | n/a | 0.000 | idle (20) | n/a |
| 33 | 201 | unclassified.ordinal_201.fused_ffn | fused_ffn | 20 | n/a | 33.697 | n/a | 0.000 | idle (20) | n/a |
| 34 | 230 | unclassified.ordinal_230.linear2_residual | linear2_residual | 20 | n/a | 21.104 | n/a | 0.000 | idle (20) | n/a |
| 35 | 214 | unclassified.ordinal_214.linear2_residual | linear2_residual | 20 | n/a | 21.041 | n/a | 0.000 | idle (20) | n/a |
| 36 | 202 | unclassified.ordinal_202.linear2_residual | linear2_residual | 20 | n/a | 21.088 | n/a | 0.000 | idle (20) | n/a |
| 37 | 222 | unclassified.ordinal_222.linear2_residual | linear2_residual | 20 | n/a | 21.056 | n/a | 0.000 | idle (20) | n/a |
| 38 | 242 | unclassified.ordinal_242.linear2_residual | linear2_residual | 20 | n/a | 21.088 | n/a | 0.000 | idle (20) | n/a |
| 39 | 18 | unclassified.ordinal_18.linear2_residual | linear2_residual | 20 | n/a | 20.960 | n/a | 0.000 | idle (20) | n/a |
| 40 | 146 | unclassified.ordinal_146.linear2_residual | linear2_residual | 20 | n/a | 20.960 | n/a | 0.000 | idle (20) | n/a |

## Full fixed-forward ordinal map

| ordinal | logical position | family | resource signature | calls | isolated us | S2 us | S2/S1 | overlap | excess ms | common peer | worst peer |
|---:|---|---|---|---:|---:|---:|---:|---:|---:|---|---|
| 0 | input.extract_mask | head_elementwise | head_elementwise; extractChannel0KernelNHWC; g10x1x1; b512x1x1; r16; s0 | 20 | n/a | 1.184 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 1 | input.mask_half_to_float | copy_reformat | copy_reformat; copyFromHalfKernel; g10x1x1; b512x1x1; r16; s0 | 20 | n/a | 0.960 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 2 | input.mask_sum | sumChannelsNCHWKernel | sumChannelsNCHWKernel; sumChannelsNCHWKernel; g1x1x13; b256x2x1; r22; s2048 | 20 | n/a | 1.696 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 3 | frontend.initial_conv_nhwc_padding_0 | cudnn | cudnn; nhwcAddPaddingKernel; g340x1x1; b768x1x1; r34; s0 | 20 | n/a | 1.280 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 4 | frontend.initial_conv_nhwc_padding_1 | cudnn | cudnn; nhwcAddPaddingKernel; g340x1x1; b768x1x1; r34; s0 | 20 | n/a | 1.536 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 5 | frontend.initial_conv | cudnn | cudnn; Kernel; g296x3x1; b128x1x1; r94; s81920 | 20 | n/a | 19.584 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 6 | frontend.initial_global_matmul | library_gemm | library_gemm; Kernel2; g8x1x3; b128x1x1; r128; s24576 | 20 | n/a | 2.624 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 7 | frontend.initial_global_matmul_splitk_reduce | library_gemm | library_gemm; splitKreduce_kernel; g24x1x1; b32x16x1; r49; s0 | 20 | n/a | 1.280 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 8 | frontend.initial_global_broadcast_add | head_elementwise | head_elementwise; addNCBiasInplaceNHWCHalfKernel; g3x361x13; b256x1x1; r16; s0 | 20 | n/a | 7.744 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 9 | unclassified.ordinal_9.affine_silu | affine_silu | affine_silu; affineSiluHalf2Kernel; g4693x1x1; b384x1x1; r17; s0 | 20 | n/a | 4.960 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 10 | unclassified.ordinal_10.library_gemm | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 20 | n/a | 11.664 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 11 | unclassified.ordinal_11.rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 20 | n/a | 2.368 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 12 | unclassified.ordinal_12.copy_reformat | copy_reformat | copy_reformat; kernel_cutlass_kernel___main__FixedAtomLayoutGemmmakelocals_Kernel_object_at__CopyAtom_ThrID10_TVLayoutSrc1819201_TVLayoutDst1819201_Valuetypef16_tensor00o4693384111100_CopyAtom_ThrID10_T_0; g1x1x170; b288x1x1; r107; s99328 | 20 | n/a | 13.920 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 13 | unclassified.ordinal_13.qk_rope | qk_rope | qk_rope; batchSharedPackedFusedQKRoPE19Half2Kernel; g361x1x1; b192x1x1; r28; s0 | 20 | n/a | 5.504 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 14 | unclassified.ordinal_14.fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 20 | n/a | 11.712 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 15 | unclassified.ordinal_15.library_gemm | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 20 | n/a | 8.464 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 16 | unclassified.ordinal_16.rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 20 | n/a | 2.432 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 17 | unclassified.ordinal_17.fused_ffn | fused_ffn | fused_ffn; fused_ffn_candidate_a_reuse_kernel; g18x37x1; b128x1x1; r136; s32768 | 20 | n/a | 39.312 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 18 | unclassified.ordinal_18.linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 20 | n/a | 20.960 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 19 | unclassified.ordinal_19.rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 20 | n/a | 2.464 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 20 | unclassified.ordinal_20.copy_reformat | copy_reformat | copy_reformat; kernel_cutlass_kernel___main__FixedAtomLayoutGemmmakelocals_Kernel_object_at__CopyAtom_ThrID10_TVLayoutSrc1819201_TVLayoutDst1819201_Valuetypef16_tensor00o4693384111100_CopyAtom_ThrID10_T_0; g1x1x170; b288x1x1; r107; s99328 | 20 | n/a | 14.240 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 21 | unclassified.ordinal_21.qk_rope | qk_rope | qk_rope; batchSharedPackedFusedQKRoPE19Half2Kernel; g361x1x1; b192x1x1; r28; s0 | 20 | n/a | 5.696 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 22 | unclassified.ordinal_22.fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 20 | n/a | 12.000 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 23 | unclassified.ordinal_23.library_gemm | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 20 | n/a | 8.528 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 24 | unclassified.ordinal_24.rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 20 | n/a | 2.464 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 25 | unclassified.ordinal_25.fused_ffn | fused_ffn | fused_ffn; fused_ffn_candidate_a_reuse_kernel; g18x37x1; b128x1x1; r136; s32768 | 20 | n/a | 38.192 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 26 | unclassified.ordinal_26.linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 20 | n/a | 20.881 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 27 | unclassified.ordinal_27.rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 20 | n/a | 2.400 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 28 | unclassified.ordinal_28.copy_reformat | copy_reformat | copy_reformat; kernel_cutlass_kernel___main__FixedAtomLayoutGemmmakelocals_Kernel_object_at__CopyAtom_ThrID10_TVLayoutSrc1819201_TVLayoutDst1819201_Valuetypef16_tensor00o4693384111100_CopyAtom_ThrID10_T_0; g1x1x170; b288x1x1; r107; s99328 | 20 | n/a | 14.112 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 29 | unclassified.ordinal_29.qk_rope | qk_rope | qk_rope; batchSharedPackedFusedQKRoPE19Half2Kernel; g361x1x1; b192x1x1; r28; s0 | 20 | n/a | 5.616 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 30 | unclassified.ordinal_30.fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 20 | n/a | 11.920 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 31 | unclassified.ordinal_31.library_gemm | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 20 | n/a | 8.512 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 32 | unclassified.ordinal_32.rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 20 | n/a | 2.465 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 33 | unclassified.ordinal_33.fused_ffn | fused_ffn | fused_ffn; fused_ffn_candidate_a_reuse_kernel; g18x37x1; b128x1x1; r136; s32768 | 20 | n/a | 38.304 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 34 | unclassified.ordinal_34.linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 20 | n/a | 20.753 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 35 | unclassified.ordinal_35.affine_silu | affine_silu | affine_silu; affineSiluHalf2Kernel; g4693x1x1; b192x1x1; r17; s0 | 20 | n/a | 3.344 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 36 | unclassified.ordinal_36.library_gemm | library_gemm | library_gemm; Kernel2; g148x1x1; b256x1x1; r154; s73728 | 20 | n/a | 14.368 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 37 | unclassified.ordinal_37.affine_silu | affine_silu | affine_silu; affineSiluHalf2Kernel; g4693x1x1; b384x1x1; r17; s0 | 20 | n/a | 5.088 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 38 | unclassified.ordinal_38.library_gemm | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 20 | n/a | 11.920 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 39 | unclassified.ordinal_39.rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 20 | n/a | 2.432 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 40 | unclassified.ordinal_40.copy_reformat | copy_reformat | copy_reformat; kernel_cutlass_kernel___main__FixedAtomLayoutGemmmakelocals_Kernel_object_at__CopyAtom_ThrID10_TVLayoutSrc1819201_TVLayoutDst1819201_Valuetypef16_tensor00o4693384111100_CopyAtom_ThrID10_T_0; g1x1x170; b288x1x1; r107; s99328 | 20 | n/a | 14.080 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 41 | unclassified.ordinal_41.qk_rope | qk_rope | qk_rope; batchSharedPackedFusedQKRoPE19Half2Kernel; g361x1x1; b192x1x1; r28; s0 | 20 | n/a | 5.568 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 42 | unclassified.ordinal_42.fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 20 | n/a | 11.840 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 43 | unclassified.ordinal_43.library_gemm | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 20 | n/a | 8.528 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 44 | unclassified.ordinal_44.rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 20 | n/a | 2.432 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 45 | unclassified.ordinal_45.fused_ffn | fused_ffn | fused_ffn; fused_ffn_candidate_a_reuse_kernel; g18x37x1; b128x1x1; r136; s32768 | 20 | n/a | 39.776 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 46 | unclassified.ordinal_46.linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 20 | n/a | 20.912 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 47 | unclassified.ordinal_47.rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 20 | n/a | 2.448 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 48 | unclassified.ordinal_48.copy_reformat | copy_reformat | copy_reformat; kernel_cutlass_kernel___main__FixedAtomLayoutGemmmakelocals_Kernel_object_at__CopyAtom_ThrID10_TVLayoutSrc1819201_TVLayoutDst1819201_Valuetypef16_tensor00o4693384111100_CopyAtom_ThrID10_T_0; g1x1x170; b288x1x1; r107; s99328 | 20 | n/a | 14.208 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 49 | unclassified.ordinal_49.qk_rope | qk_rope | qk_rope; batchSharedPackedFusedQKRoPE19Half2Kernel; g361x1x1; b192x1x1; r28; s0 | 20 | n/a | 5.601 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 50 | unclassified.ordinal_50.fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 20 | n/a | 11.889 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 51 | unclassified.ordinal_51.library_gemm | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 20 | n/a | 8.528 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 52 | unclassified.ordinal_52.rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 20 | n/a | 2.432 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 53 | unclassified.ordinal_53.fused_ffn | fused_ffn | fused_ffn; fused_ffn_candidate_a_reuse_kernel; g18x37x1; b128x1x1; r136; s32768 | 20 | n/a | 38.577 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 54 | unclassified.ordinal_54.linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 20 | n/a | 20.736 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 55 | unclassified.ordinal_55.rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 20 | n/a | 2.432 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 56 | unclassified.ordinal_56.copy_reformat | copy_reformat | copy_reformat; kernel_cutlass_kernel___main__FixedAtomLayoutGemmmakelocals_Kernel_object_at__CopyAtom_ThrID10_TVLayoutSrc1819201_TVLayoutDst1819201_Valuetypef16_tensor00o4693384111100_CopyAtom_ThrID10_T_0; g1x1x170; b288x1x1; r107; s99328 | 20 | n/a | 14.177 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 57 | unclassified.ordinal_57.qk_rope | qk_rope | qk_rope; batchSharedPackedFusedQKRoPE19Half2Kernel; g361x1x1; b192x1x1; r28; s0 | 20 | n/a | 5.616 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 58 | unclassified.ordinal_58.fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 20 | n/a | 11.872 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 59 | unclassified.ordinal_59.library_gemm | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 20 | n/a | 8.464 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 60 | unclassified.ordinal_60.rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 20 | n/a | 2.432 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 61 | unclassified.ordinal_61.fused_ffn | fused_ffn | fused_ffn; fused_ffn_candidate_a_reuse_kernel; g18x37x1; b128x1x1; r136; s32768 | 20 | n/a | 38.529 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 62 | unclassified.ordinal_62.linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 20 | n/a | 20.736 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 63 | unclassified.ordinal_63.affine_silu | affine_silu | affine_silu; affineSiluHalf2Kernel; g4693x1x1; b192x1x1; r17; s0 | 20 | n/a | 3.328 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 64 | unclassified.ordinal_64.library_gemm | library_gemm | library_gemm; Kernel2; g148x1x1; b256x1x1; r154; s73728 | 20 | n/a | 14.384 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 65 | unclassified.ordinal_65.affine_silu | affine_silu | affine_silu; affineSiluHalf2Kernel; g4693x1x1; b384x1x1; r17; s0 | 20 | n/a | 5.024 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 66 | unclassified.ordinal_66.library_gemm | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 20 | n/a | 11.921 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 67 | unclassified.ordinal_67.rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 20 | n/a | 2.416 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 68 | unclassified.ordinal_68.copy_reformat | copy_reformat | copy_reformat; kernel_cutlass_kernel___main__FixedAtomLayoutGemmmakelocals_Kernel_object_at__CopyAtom_ThrID10_TVLayoutSrc1819201_TVLayoutDst1819201_Valuetypef16_tensor00o4693384111100_CopyAtom_ThrID10_T_0; g1x1x170; b288x1x1; r107; s99328 | 20 | n/a | 14.064 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 69 | unclassified.ordinal_69.qk_rope | qk_rope | qk_rope; batchSharedPackedFusedQKRoPE19Half2Kernel; g361x1x1; b192x1x1; r28; s0 | 20 | n/a | 5.600 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 70 | unclassified.ordinal_70.fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 20 | n/a | 11.856 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 71 | unclassified.ordinal_71.library_gemm | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 20 | n/a | 8.480 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 72 | unclassified.ordinal_72.rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 20 | n/a | 2.432 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 73 | unclassified.ordinal_73.fused_ffn | fused_ffn | fused_ffn; fused_ffn_candidate_a_reuse_kernel; g18x37x1; b128x1x1; r136; s32768 | 20 | n/a | 40.257 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 74 | unclassified.ordinal_74.linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 20 | n/a | 20.880 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 75 | unclassified.ordinal_75.rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 20 | n/a | 2.464 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 76 | unclassified.ordinal_76.copy_reformat | copy_reformat | copy_reformat; kernel_cutlass_kernel___main__FixedAtomLayoutGemmmakelocals_Kernel_object_at__CopyAtom_ThrID10_TVLayoutSrc1819201_TVLayoutDst1819201_Valuetypef16_tensor00o4693384111100_CopyAtom_ThrID10_T_0; g1x1x170; b288x1x1; r107; s99328 | 20 | n/a | 14.113 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 77 | unclassified.ordinal_77.qk_rope | qk_rope | qk_rope; batchSharedPackedFusedQKRoPE19Half2Kernel; g361x1x1; b192x1x1; r28; s0 | 20 | n/a | 5.632 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 78 | unclassified.ordinal_78.fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 20 | n/a | 11.904 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 79 | unclassified.ordinal_79.library_gemm | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 20 | n/a | 8.496 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 80 | unclassified.ordinal_80.rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 20 | n/a | 2.464 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 81 | unclassified.ordinal_81.fused_ffn | fused_ffn | fused_ffn; fused_ffn_candidate_a_reuse_kernel; g18x37x1; b128x1x1; r136; s32768 | 20 | n/a | 40.977 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 82 | unclassified.ordinal_82.linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 20 | n/a | 20.832 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 83 | unclassified.ordinal_83.rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 20 | n/a | 2.449 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 84 | unclassified.ordinal_84.copy_reformat | copy_reformat | copy_reformat; kernel_cutlass_kernel___main__FixedAtomLayoutGemmmakelocals_Kernel_object_at__CopyAtom_ThrID10_TVLayoutSrc1819201_TVLayoutDst1819201_Valuetypef16_tensor00o4693384111100_CopyAtom_ThrID10_T_0; g1x1x170; b288x1x1; r107; s99328 | 20 | n/a | 14.113 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 85 | unclassified.ordinal_85.qk_rope | qk_rope | qk_rope; batchSharedPackedFusedQKRoPE19Half2Kernel; g361x1x1; b192x1x1; r28; s0 | 20 | n/a | 5.664 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 86 | unclassified.ordinal_86.fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 20 | n/a | 11.888 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 87 | unclassified.ordinal_87.library_gemm | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 20 | n/a | 8.544 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 88 | unclassified.ordinal_88.rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 20 | n/a | 2.448 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 89 | unclassified.ordinal_89.fused_ffn | fused_ffn | fused_ffn; fused_ffn_candidate_a_reuse_kernel; g18x37x1; b128x1x1; r136; s32768 | 20 | n/a | 37.920 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 90 | unclassified.ordinal_90.linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 20 | n/a | 20.817 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 91 | unclassified.ordinal_91.affine_silu | affine_silu | affine_silu; affineSiluHalf2Kernel; g4693x1x1; b192x1x1; r17; s0 | 20 | n/a | 3.392 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 92 | unclassified.ordinal_92.library_gemm | library_gemm | library_gemm; Kernel2; g148x1x1; b256x1x1; r154; s73728 | 20 | n/a | 14.480 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 93 | unclassified.ordinal_93.affine_silu | affine_silu | affine_silu; affineSiluHalf2Kernel; g4693x1x1; b384x1x1; r17; s0 | 20 | n/a | 5.056 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 94 | unclassified.ordinal_94.library_gemm | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 20 | n/a | 11.952 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 95 | unclassified.ordinal_95.rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 20 | n/a | 2.417 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 96 | unclassified.ordinal_96.copy_reformat | copy_reformat | copy_reformat; kernel_cutlass_kernel___main__FixedAtomLayoutGemmmakelocals_Kernel_object_at__CopyAtom_ThrID10_TVLayoutSrc1819201_TVLayoutDst1819201_Valuetypef16_tensor00o4693384111100_CopyAtom_ThrID10_T_0; g1x1x170; b288x1x1; r107; s99328 | 20 | n/a | 14.257 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 97 | unclassified.ordinal_97.qk_rope | qk_rope | qk_rope; batchSharedPackedFusedQKRoPE19Half2Kernel; g361x1x1; b192x1x1; r28; s0 | 20 | n/a | 5.616 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 98 | unclassified.ordinal_98.fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 20 | n/a | 11.920 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 99 | unclassified.ordinal_99.library_gemm | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 20 | n/a | 8.512 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 100 | unclassified.ordinal_100.rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 20 | n/a | 2.432 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 101 | unclassified.ordinal_101.fused_ffn | fused_ffn | fused_ffn; fused_ffn_candidate_a_reuse_kernel; g18x37x1; b128x1x1; r136; s32768 | 20 | n/a | 38.032 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 102 | unclassified.ordinal_102.linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 20 | n/a | 20.768 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 103 | unclassified.ordinal_103.rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 20 | n/a | 2.464 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 104 | unclassified.ordinal_104.copy_reformat | copy_reformat | copy_reformat; kernel_cutlass_kernel___main__FixedAtomLayoutGemmmakelocals_Kernel_object_at__CopyAtom_ThrID10_TVLayoutSrc1819201_TVLayoutDst1819201_Valuetypef16_tensor00o4693384111100_CopyAtom_ThrID10_T_0; g1x1x170; b288x1x1; r107; s99328 | 20 | n/a | 14.081 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 105 | unclassified.ordinal_105.qk_rope | qk_rope | qk_rope; batchSharedPackedFusedQKRoPE19Half2Kernel; g361x1x1; b192x1x1; r28; s0 | 20 | n/a | 5.600 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 106 | unclassified.ordinal_106.fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 20 | n/a | 11.824 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 107 | unclassified.ordinal_107.library_gemm | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 20 | n/a | 8.400 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 108 | unclassified.ordinal_108.rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 20 | n/a | 2.432 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 109 | unclassified.ordinal_109.fused_ffn | fused_ffn | fused_ffn; fused_ffn_candidate_a_reuse_kernel; g18x37x1; b128x1x1; r136; s32768 | 20 | n/a | 40.193 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 110 | unclassified.ordinal_110.linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 20 | n/a | 20.640 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 111 | unclassified.ordinal_111.rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 20 | n/a | 2.368 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 112 | unclassified.ordinal_112.copy_reformat | copy_reformat | copy_reformat; kernel_cutlass_kernel___main__FixedAtomLayoutGemmmakelocals_Kernel_object_at__CopyAtom_ThrID10_TVLayoutSrc1819201_TVLayoutDst1819201_Valuetypef16_tensor00o4693384111100_CopyAtom_ThrID10_T_0; g1x1x170; b288x1x1; r107; s99328 | 20 | n/a | 14.208 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 113 | unclassified.ordinal_113.qk_rope | qk_rope | qk_rope; batchSharedPackedFusedQKRoPE19Half2Kernel; g361x1x1; b192x1x1; r28; s0 | 20 | n/a | 5.568 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 114 | unclassified.ordinal_114.fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 20 | n/a | 11.776 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 115 | unclassified.ordinal_115.library_gemm | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 20 | n/a | 8.416 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 116 | unclassified.ordinal_116.rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 20 | n/a | 2.464 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 117 | unclassified.ordinal_117.fused_ffn | fused_ffn | fused_ffn; fused_ffn_candidate_a_reuse_kernel; g18x37x1; b128x1x1; r136; s32768 | 20 | n/a | 40.608 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 118 | unclassified.ordinal_118.linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 20 | n/a | 20.529 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 119 | unclassified.ordinal_119.affine_silu | affine_silu | affine_silu; affineSiluHalf2Kernel; g4693x1x1; b192x1x1; r17; s0 | 20 | n/a | 3.296 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 120 | unclassified.ordinal_120.library_gemm | library_gemm | library_gemm; Kernel2; g148x1x1; b256x1x1; r154; s73728 | 20 | n/a | 14.192 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 121 | unclassified.ordinal_121.affine_silu | affine_silu | affine_silu; affineSiluHalf2Kernel; g4693x1x1; b384x1x1; r17; s0 | 20 | n/a | 5.008 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 122 | unclassified.ordinal_122.library_gemm | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 20 | n/a | 11.745 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 123 | unclassified.ordinal_123.rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 20 | n/a | 2.432 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 124 | unclassified.ordinal_124.copy_reformat | copy_reformat | copy_reformat; kernel_cutlass_kernel___main__FixedAtomLayoutGemmmakelocals_Kernel_object_at__CopyAtom_ThrID10_TVLayoutSrc1819201_TVLayoutDst1819201_Valuetypef16_tensor00o4693384111100_CopyAtom_ThrID10_T_0; g1x1x170; b288x1x1; r107; s99328 | 20 | n/a | 14.033 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 125 | unclassified.ordinal_125.qk_rope | qk_rope | qk_rope; batchSharedPackedFusedQKRoPE19Half2Kernel; g361x1x1; b192x1x1; r28; s0 | 20 | n/a | 5.568 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 126 | unclassified.ordinal_126.fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 20 | n/a | 11.792 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 127 | unclassified.ordinal_127.library_gemm | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 20 | n/a | 8.480 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 128 | unclassified.ordinal_128.rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 20 | n/a | 2.448 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 129 | unclassified.ordinal_129.fused_ffn | fused_ffn | fused_ffn; fused_ffn_candidate_a_reuse_kernel; g18x37x1; b128x1x1; r136; s32768 | 20 | n/a | 38.624 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 130 | unclassified.ordinal_130.linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 20 | n/a | 20.832 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 131 | unclassified.ordinal_131.rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 20 | n/a | 2.416 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 132 | unclassified.ordinal_132.copy_reformat | copy_reformat | copy_reformat; kernel_cutlass_kernel___main__FixedAtomLayoutGemmmakelocals_Kernel_object_at__CopyAtom_ThrID10_TVLayoutSrc1819201_TVLayoutDst1819201_Valuetypef16_tensor00o4693384111100_CopyAtom_ThrID10_T_0; g1x1x170; b288x1x1; r107; s99328 | 20 | n/a | 14.144 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 133 | unclassified.ordinal_133.qk_rope | qk_rope | qk_rope; batchSharedPackedFusedQKRoPE19Half2Kernel; g361x1x1; b192x1x1; r28; s0 | 20 | n/a | 5.632 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 134 | unclassified.ordinal_134.fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 20 | n/a | 11.952 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 135 | unclassified.ordinal_135.library_gemm | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 20 | n/a | 8.512 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 136 | unclassified.ordinal_136.rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 20 | n/a | 2.433 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 137 | unclassified.ordinal_137.fused_ffn | fused_ffn | fused_ffn; fused_ffn_candidate_a_reuse_kernel; g18x37x1; b128x1x1; r136; s32768 | 20 | n/a | 38.416 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 138 | unclassified.ordinal_138.linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 20 | n/a | 20.832 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 139 | unclassified.ordinal_139.rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 20 | n/a | 2.432 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 140 | unclassified.ordinal_140.copy_reformat | copy_reformat | copy_reformat; kernel_cutlass_kernel___main__FixedAtomLayoutGemmmakelocals_Kernel_object_at__CopyAtom_ThrID10_TVLayoutSrc1819201_TVLayoutDst1819201_Valuetypef16_tensor00o4693384111100_CopyAtom_ThrID10_T_0; g1x1x170; b288x1x1; r107; s99328 | 20 | n/a | 14.144 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 141 | unclassified.ordinal_141.qk_rope | qk_rope | qk_rope; batchSharedPackedFusedQKRoPE19Half2Kernel; g361x1x1; b192x1x1; r28; s0 | 20 | n/a | 5.633 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 142 | unclassified.ordinal_142.fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 20 | n/a | 11.936 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 143 | unclassified.ordinal_143.library_gemm | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 20 | n/a | 8.512 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 144 | unclassified.ordinal_144.rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 20 | n/a | 2.432 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 145 | unclassified.ordinal_145.fused_ffn | fused_ffn | fused_ffn; fused_ffn_candidate_a_reuse_kernel; g18x37x1; b128x1x1; r136; s32768 | 20 | n/a | 35.441 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 146 | unclassified.ordinal_146.linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 20 | n/a | 20.960 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 147 | unclassified.ordinal_147.affine_silu | affine_silu | affine_silu; affineSiluHalf2Kernel; g4693x1x1; b192x1x1; r17; s0 | 20 | n/a | 3.361 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 148 | unclassified.ordinal_148.library_gemm | library_gemm | library_gemm; Kernel2; g148x1x1; b256x1x1; r154; s73728 | 20 | n/a | 14.528 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 149 | unclassified.ordinal_149.affine_silu | affine_silu | affine_silu; affineSiluHalf2Kernel; g4693x1x1; b384x1x1; r17; s0 | 20 | n/a | 5.120 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 150 | unclassified.ordinal_150.library_gemm | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 20 | n/a | 12.049 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 151 | unclassified.ordinal_151.rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 20 | n/a | 2.465 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 152 | unclassified.ordinal_152.copy_reformat | copy_reformat | copy_reformat; kernel_cutlass_kernel___main__FixedAtomLayoutGemmmakelocals_Kernel_object_at__CopyAtom_ThrID10_TVLayoutSrc1819201_TVLayoutDst1819201_Valuetypef16_tensor00o4693384111100_CopyAtom_ThrID10_T_0; g1x1x170; b288x1x1; r107; s99328 | 20 | n/a | 14.113 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 153 | unclassified.ordinal_153.qk_rope | qk_rope | qk_rope; batchSharedPackedFusedQKRoPE19Half2Kernel; g361x1x1; b192x1x1; r28; s0 | 20 | n/a | 5.680 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 154 | unclassified.ordinal_154.fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 20 | n/a | 11.952 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 155 | unclassified.ordinal_155.library_gemm | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 20 | n/a | 8.512 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 156 | unclassified.ordinal_156.rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 20 | n/a | 2.433 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 157 | unclassified.ordinal_157.fused_ffn | fused_ffn | fused_ffn; fused_ffn_candidate_a_reuse_kernel; g18x37x1; b128x1x1; r136; s32768 | 20 | n/a | 41.152 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 158 | unclassified.ordinal_158.linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 20 | n/a | 20.976 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 159 | unclassified.ordinal_159.rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 20 | n/a | 2.432 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 160 | unclassified.ordinal_160.copy_reformat | copy_reformat | copy_reformat; kernel_cutlass_kernel___main__FixedAtomLayoutGemmmakelocals_Kernel_object_at__CopyAtom_ThrID10_TVLayoutSrc1819201_TVLayoutDst1819201_Valuetypef16_tensor00o4693384111100_CopyAtom_ThrID10_T_0; g1x1x170; b288x1x1; r107; s99328 | 20 | n/a | 14.240 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 161 | unclassified.ordinal_161.qk_rope | qk_rope | qk_rope; batchSharedPackedFusedQKRoPE19Half2Kernel; g361x1x1; b192x1x1; r28; s0 | 20 | n/a | 5.600 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 162 | unclassified.ordinal_162.fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 20 | n/a | 11.921 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 163 | unclassified.ordinal_163.library_gemm | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 20 | n/a | 8.496 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 164 | unclassified.ordinal_164.rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 20 | n/a | 2.432 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 165 | unclassified.ordinal_165.fused_ffn | fused_ffn | fused_ffn; fused_ffn_candidate_a_reuse_kernel; g18x37x1; b128x1x1; r136; s32768 | 20 | n/a | 40.816 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 166 | unclassified.ordinal_166.linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 20 | n/a | 20.848 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 167 | unclassified.ordinal_167.rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 20 | n/a | 2.432 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 168 | unclassified.ordinal_168.copy_reformat | copy_reformat | copy_reformat; kernel_cutlass_kernel___main__FixedAtomLayoutGemmmakelocals_Kernel_object_at__CopyAtom_ThrID10_TVLayoutSrc1819201_TVLayoutDst1819201_Valuetypef16_tensor00o4693384111100_CopyAtom_ThrID10_T_0; g1x1x170; b288x1x1; r107; s99328 | 20 | n/a | 14.145 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 169 | unclassified.ordinal_169.qk_rope | qk_rope | qk_rope; batchSharedPackedFusedQKRoPE19Half2Kernel; g361x1x1; b192x1x1; r28; s0 | 20 | n/a | 5.568 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 170 | unclassified.ordinal_170.fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 20 | n/a | 11.872 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 171 | unclassified.ordinal_171.library_gemm | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 20 | n/a | 8.480 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 172 | unclassified.ordinal_172.rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 20 | n/a | 2.401 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 173 | unclassified.ordinal_173.fused_ffn | fused_ffn | fused_ffn; fused_ffn_candidate_a_reuse_kernel; g18x37x1; b128x1x1; r136; s32768 | 20 | n/a | 38.849 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 174 | unclassified.ordinal_174.linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 20 | n/a | 20.753 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 175 | unclassified.ordinal_175.affine_silu | affine_silu | affine_silu; affineSiluHalf2Kernel; g4693x1x1; b192x1x1; r17; s0 | 20 | n/a | 3.296 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 176 | unclassified.ordinal_176.library_gemm | library_gemm | library_gemm; Kernel2; g148x1x1; b256x1x1; r154; s73728 | 20 | n/a | 14.336 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 177 | unclassified.ordinal_177.affine_silu | affine_silu | affine_silu; affineSiluHalf2Kernel; g4693x1x1; b384x1x1; r17; s0 | 20 | n/a | 5.024 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 178 | unclassified.ordinal_178.library_gemm | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 20 | n/a | 11.872 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 179 | unclassified.ordinal_179.rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 20 | n/a | 2.432 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 180 | unclassified.ordinal_180.copy_reformat | copy_reformat | copy_reformat; kernel_cutlass_kernel___main__FixedAtomLayoutGemmmakelocals_Kernel_object_at__CopyAtom_ThrID10_TVLayoutSrc1819201_TVLayoutDst1819201_Valuetypef16_tensor00o4693384111100_CopyAtom_ThrID10_T_0; g1x1x170; b288x1x1; r107; s99328 | 20 | n/a | 14.048 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 181 | unclassified.ordinal_181.qk_rope | qk_rope | qk_rope; batchSharedPackedFusedQKRoPE19Half2Kernel; g361x1x1; b192x1x1; r28; s0 | 20 | n/a | 5.600 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 182 | unclassified.ordinal_182.fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 20 | n/a | 11.873 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 183 | unclassified.ordinal_183.library_gemm | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 20 | n/a | 8.480 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 184 | unclassified.ordinal_184.rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 20 | n/a | 2.496 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 185 | unclassified.ordinal_185.fused_ffn | fused_ffn | fused_ffn; fused_ffn_candidate_a_reuse_kernel; g18x37x1; b128x1x1; r136; s32768 | 20 | n/a | 40.560 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 186 | unclassified.ordinal_186.linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 20 | n/a | 20.896 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 187 | unclassified.ordinal_187.rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 20 | n/a | 2.496 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 188 | unclassified.ordinal_188.copy_reformat | copy_reformat | copy_reformat; kernel_cutlass_kernel___main__FixedAtomLayoutGemmmakelocals_Kernel_object_at__CopyAtom_ThrID10_TVLayoutSrc1819201_TVLayoutDst1819201_Valuetypef16_tensor00o4693384111100_CopyAtom_ThrID10_T_0; g1x1x170; b288x1x1; r107; s99328 | 20 | n/a | 14.112 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 189 | unclassified.ordinal_189.qk_rope | qk_rope | qk_rope; batchSharedPackedFusedQKRoPE19Half2Kernel; g361x1x1; b192x1x1; r28; s0 | 20 | n/a | 5.665 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 190 | unclassified.ordinal_190.fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 20 | n/a | 11.968 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 191 | unclassified.ordinal_191.library_gemm | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 20 | n/a | 8.496 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 192 | unclassified.ordinal_192.rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 20 | n/a | 2.432 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 193 | unclassified.ordinal_193.fused_ffn | fused_ffn | fused_ffn; fused_ffn_candidate_a_reuse_kernel; g18x37x1; b128x1x1; r136; s32768 | 20 | n/a | 39.328 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 194 | unclassified.ordinal_194.linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 20 | n/a | 20.992 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 195 | unclassified.ordinal_195.rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 20 | n/a | 2.432 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 196 | unclassified.ordinal_196.copy_reformat | copy_reformat | copy_reformat; kernel_cutlass_kernel___main__FixedAtomLayoutGemmmakelocals_Kernel_object_at__CopyAtom_ThrID10_TVLayoutSrc1819201_TVLayoutDst1819201_Valuetypef16_tensor00o4693384111100_CopyAtom_ThrID10_T_0; g1x1x170; b288x1x1; r107; s99328 | 20 | n/a | 14.272 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 197 | unclassified.ordinal_197.qk_rope | qk_rope | qk_rope; batchSharedPackedFusedQKRoPE19Half2Kernel; g361x1x1; b192x1x1; r28; s0 | 20 | n/a | 5.616 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 198 | unclassified.ordinal_198.fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 20 | n/a | 11.968 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 199 | unclassified.ordinal_199.library_gemm | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 20 | n/a | 8.480 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 200 | unclassified.ordinal_200.rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 20 | n/a | 2.433 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 201 | unclassified.ordinal_201.fused_ffn | fused_ffn | fused_ffn; fused_ffn_candidate_a_reuse_kernel; g18x37x1; b128x1x1; r136; s32768 | 20 | n/a | 33.697 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 202 | unclassified.ordinal_202.linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 20 | n/a | 21.088 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 203 | unclassified.ordinal_203.affine_silu | affine_silu | affine_silu; affineSiluHalf2Kernel; g4693x1x1; b192x1x1; r17; s0 | 20 | n/a | 3.376 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 204 | unclassified.ordinal_204.library_gemm | library_gemm | library_gemm; Kernel2; g148x1x1; b256x1x1; r154; s73728 | 20 | n/a | 14.624 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 205 | unclassified.ordinal_205.affine_silu | affine_silu | affine_silu; affineSiluHalf2Kernel; g4693x1x1; b384x1x1; r17; s0 | 20 | n/a | 5.121 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 206 | unclassified.ordinal_206.library_gemm | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 20 | n/a | 12.049 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 207 | unclassified.ordinal_207.rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 20 | n/a | 2.464 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 208 | unclassified.ordinal_208.copy_reformat | copy_reformat | copy_reformat; kernel_cutlass_kernel___main__FixedAtomLayoutGemmmakelocals_Kernel_object_at__CopyAtom_ThrID10_TVLayoutSrc1819201_TVLayoutDst1819201_Valuetypef16_tensor00o4693384111100_CopyAtom_ThrID10_T_0; g1x1x170; b288x1x1; r107; s99328 | 20 | n/a | 14.273 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 209 | unclassified.ordinal_209.qk_rope | qk_rope | qk_rope; batchSharedPackedFusedQKRoPE19Half2Kernel; g361x1x1; b192x1x1; r28; s0 | 20 | n/a | 5.648 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 210 | unclassified.ordinal_210.fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 20 | n/a | 12.000 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 211 | unclassified.ordinal_211.library_gemm | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 20 | n/a | 8.544 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 212 | unclassified.ordinal_212.rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 20 | n/a | 2.432 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 213 | unclassified.ordinal_213.fused_ffn | fused_ffn | fused_ffn; fused_ffn_candidate_a_reuse_kernel; g18x37x1; b128x1x1; r136; s32768 | 20 | n/a | 35.776 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 214 | unclassified.ordinal_214.linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 20 | n/a | 21.041 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 215 | unclassified.ordinal_215.rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 20 | n/a | 2.432 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 216 | unclassified.ordinal_216.copy_reformat | copy_reformat | copy_reformat; kernel_cutlass_kernel___main__FixedAtomLayoutGemmmakelocals_Kernel_object_at__CopyAtom_ThrID10_TVLayoutSrc1819201_TVLayoutDst1819201_Valuetypef16_tensor00o4693384111100_CopyAtom_ThrID10_T_0; g1x1x170; b288x1x1; r107; s99328 | 20 | n/a | 14.304 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 217 | unclassified.ordinal_217.qk_rope | qk_rope | qk_rope; batchSharedPackedFusedQKRoPE19Half2Kernel; g361x1x1; b192x1x1; r28; s0 | 20 | n/a | 5.696 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 218 | unclassified.ordinal_218.fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 20 | n/a | 12.048 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 219 | unclassified.ordinal_219.library_gemm | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 20 | n/a | 8.624 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 220 | unclassified.ordinal_220.rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 20 | n/a | 2.496 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 221 | unclassified.ordinal_221.fused_ffn | fused_ffn | fused_ffn; fused_ffn_candidate_a_reuse_kernel; g18x37x1; b128x1x1; r136; s32768 | 20 | n/a | 35.904 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 222 | unclassified.ordinal_222.linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 20 | n/a | 21.056 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 223 | unclassified.ordinal_223.rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 20 | n/a | 2.464 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 224 | unclassified.ordinal_224.copy_reformat | copy_reformat | copy_reformat; kernel_cutlass_kernel___main__FixedAtomLayoutGemmmakelocals_Kernel_object_at__CopyAtom_ThrID10_TVLayoutSrc1819201_TVLayoutDst1819201_Valuetypef16_tensor00o4693384111100_CopyAtom_ThrID10_T_0; g1x1x170; b288x1x1; r107; s99328 | 20 | n/a | 14.257 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 225 | unclassified.ordinal_225.qk_rope | qk_rope | qk_rope; batchSharedPackedFusedQKRoPE19Half2Kernel; g361x1x1; b192x1x1; r28; s0 | 20 | n/a | 5.744 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 226 | unclassified.ordinal_226.fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 20 | n/a | 12.064 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 227 | unclassified.ordinal_227.library_gemm | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 20 | n/a | 8.608 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 228 | unclassified.ordinal_228.rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 20 | n/a | 2.449 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 229 | unclassified.ordinal_229.fused_ffn | fused_ffn | fused_ffn; fused_ffn_candidate_a_reuse_kernel; g18x37x1; b128x1x1; r136; s32768 | 20 | n/a | 35.840 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 230 | unclassified.ordinal_230.linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 20 | n/a | 21.104 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 231 | unclassified.ordinal_231.affine_silu | affine_silu | affine_silu; affineSiluHalf2Kernel; g4693x1x1; b192x1x1; r17; s0 | 20 | n/a | 3.376 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 232 | unclassified.ordinal_232.library_gemm | library_gemm | library_gemm; Kernel2; g148x1x1; b256x1x1; r154; s73728 | 20 | n/a | 14.672 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 233 | unclassified.ordinal_233.affine_silu | affine_silu | affine_silu; affineSiluHalf2Kernel; g4693x1x1; b384x1x1; r17; s0 | 20 | n/a | 5.120 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 234 | unclassified.ordinal_234.library_gemm | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 20 | n/a | 12.096 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 235 | unclassified.ordinal_235.rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 20 | n/a | 2.464 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 236 | unclassified.ordinal_236.copy_reformat | copy_reformat | copy_reformat; kernel_cutlass_kernel___main__FixedAtomLayoutGemmmakelocals_Kernel_object_at__CopyAtom_ThrID10_TVLayoutSrc1819201_TVLayoutDst1819201_Valuetypef16_tensor00o4693384111100_CopyAtom_ThrID10_T_0; g1x1x170; b288x1x1; r107; s99328 | 20 | n/a | 14.240 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 237 | unclassified.ordinal_237.qk_rope | qk_rope | qk_rope; batchSharedPackedFusedQKRoPE19Half2Kernel; g361x1x1; b192x1x1; r28; s0 | 20 | n/a | 5.696 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 238 | unclassified.ordinal_238.fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 20 | n/a | 12.001 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 239 | unclassified.ordinal_239.library_gemm | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 20 | n/a | 8.624 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 240 | unclassified.ordinal_240.rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 20 | n/a | 2.464 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 241 | unclassified.ordinal_241.fused_ffn | fused_ffn | fused_ffn; fused_ffn_candidate_a_reuse_kernel; g18x37x1; b128x1x1; r136; s32768 | 20 | n/a | 40.769 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 242 | unclassified.ordinal_242.linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 20 | n/a | 21.088 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 243 | unclassified.ordinal_243.rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 20 | n/a | 2.464 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 244 | unclassified.ordinal_244.copy_reformat | copy_reformat | copy_reformat; kernel_cutlass_kernel___main__FixedAtomLayoutGemmmakelocals_Kernel_object_at__CopyAtom_ThrID10_TVLayoutSrc1819201_TVLayoutDst1819201_Valuetypef16_tensor00o4693384111100_CopyAtom_ThrID10_T_0; g1x1x170; b288x1x1; r107; s99328 | 20 | n/a | 14.240 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 245 | unclassified.ordinal_245.qk_rope | qk_rope | qk_rope; batchSharedPackedFusedQKRoPE19Half2Kernel; g361x1x1; b192x1x1; r28; s0 | 20 | n/a | 5.665 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 246 | unclassified.ordinal_246.fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 20 | n/a | 12.032 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 247 | unclassified.ordinal_247.library_gemm | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 20 | n/a | 8.528 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 248 | unclassified.ordinal_248.rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 20 | n/a | 2.448 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 249 | unclassified.ordinal_249.fused_ffn | fused_ffn | fused_ffn; fused_ffn_candidate_a_reuse_kernel; g18x37x1; b128x1x1; r136; s32768 | 20 | n/a | 40.800 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 250 | unclassified.ordinal_250.linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 20 | n/a | 20.960 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 251 | unclassified.ordinal_251.rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 20 | n/a | 2.481 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 252 | unclassified.ordinal_252.copy_reformat | copy_reformat | copy_reformat; kernel_cutlass_kernel___main__FixedAtomLayoutGemmmakelocals_Kernel_object_at__CopyAtom_ThrID10_TVLayoutSrc1819201_TVLayoutDst1819201_Valuetypef16_tensor00o4693384111100_CopyAtom_ThrID10_T_0; g1x1x170; b288x1x1; r107; s99328 | 20 | n/a | 14.177 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 253 | unclassified.ordinal_253.qk_rope | qk_rope | qk_rope; batchSharedPackedFusedQKRoPE19Half2Kernel; g361x1x1; b192x1x1; r28; s0 | 20 | n/a | 5.648 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 254 | unclassified.ordinal_254.fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 20 | n/a | 12.000 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 255 | unclassified.ordinal_255.library_gemm | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 20 | n/a | 8.480 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 256 | unclassified.ordinal_256.rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 20 | n/a | 2.464 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 257 | unclassified.ordinal_257.fused_ffn | fused_ffn | fused_ffn; fused_ffn_candidate_a_reuse_kernel; g18x37x1; b128x1x1; r136; s32768 | 20 | n/a | 40.273 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 258 | unclassified.ordinal_258.linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 20 | n/a | 20.928 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 259 | unclassified.ordinal_259.affine_silu | affine_silu | affine_silu; affineSiluHalf2Kernel; g4693x1x1; b192x1x1; r17; s0 | 20 | n/a | 3.360 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 260 | unclassified.ordinal_260.library_gemm | library_gemm | library_gemm; Kernel2; g148x1x1; b256x1x1; r154; s73728 | 20 | n/a | 14.480 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 261 | unclassified.ordinal_261.affine_silu | affine_silu | affine_silu; affineSiluHalf2Kernel; g4693x1x1; b384x1x1; r17; s0 | 20 | n/a | 5.056 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 262 | unclassified.ordinal_262.library_gemm | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 20 | n/a | 12.049 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 263 | unclassified.ordinal_263.rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 20 | n/a | 2.432 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 264 | unclassified.ordinal_264.copy_reformat | copy_reformat | copy_reformat; kernel_cutlass_kernel___main__FixedAtomLayoutGemmmakelocals_Kernel_object_at__CopyAtom_ThrID10_TVLayoutSrc1819201_TVLayoutDst1819201_Valuetypef16_tensor00o4693384111100_CopyAtom_ThrID10_T_0; g1x1x170; b288x1x1; r107; s99328 | 20 | n/a | 14.209 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 265 | unclassified.ordinal_265.qk_rope | qk_rope | qk_rope; batchSharedPackedFusedQKRoPE19Half2Kernel; g361x1x1; b192x1x1; r28; s0 | 20 | n/a | 5.632 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 266 | unclassified.ordinal_266.fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 20 | n/a | 11.936 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 267 | unclassified.ordinal_267.library_gemm | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 20 | n/a | 8.560 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 268 | unclassified.ordinal_268.rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 20 | n/a | 2.464 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 269 | unclassified.ordinal_269.fused_ffn | fused_ffn | fused_ffn; fused_ffn_candidate_a_reuse_kernel; g18x37x1; b128x1x1; r136; s32768 | 20 | n/a | 38.752 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 270 | unclassified.ordinal_270.linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 20 | n/a | 20.768 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 271 | unclassified.ordinal_271.rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 20 | n/a | 2.432 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 272 | unclassified.ordinal_272.copy_reformat | copy_reformat | copy_reformat; kernel_cutlass_kernel___main__FixedAtomLayoutGemmmakelocals_Kernel_object_at__CopyAtom_ThrID10_TVLayoutSrc1819201_TVLayoutDst1819201_Valuetypef16_tensor00o4693384111100_CopyAtom_ThrID10_T_0; g1x1x170; b288x1x1; r107; s99328 | 20 | n/a | 14.208 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 273 | unclassified.ordinal_273.qk_rope | qk_rope | qk_rope; batchSharedPackedFusedQKRoPE19Half2Kernel; g361x1x1; b192x1x1; r28; s0 | 20 | n/a | 5.600 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 274 | unclassified.ordinal_274.fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 20 | n/a | 11.873 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 275 | unclassified.ordinal_275.library_gemm | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 20 | n/a | 8.496 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 276 | unclassified.ordinal_276.rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 20 | n/a | 2.448 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 277 | unclassified.ordinal_277.fused_ffn | fused_ffn | fused_ffn; fused_ffn_candidate_a_reuse_kernel; g18x37x1; b128x1x1; r136; s32768 | 20 | n/a | 38.449 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 278 | unclassified.ordinal_278.linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 20 | n/a | 20.641 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 279 | unclassified.ordinal_279.rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 20 | n/a | 2.400 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 280 | unclassified.ordinal_280.copy_reformat | copy_reformat | copy_reformat; kernel_cutlass_kernel___main__FixedAtomLayoutGemmmakelocals_Kernel_object_at__CopyAtom_ThrID10_TVLayoutSrc1819201_TVLayoutDst1819201_Valuetypef16_tensor00o4693384111100_CopyAtom_ThrID10_T_0; g1x1x170; b288x1x1; r107; s99328 | 20 | n/a | 14.033 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 281 | unclassified.ordinal_281.qk_rope | qk_rope | qk_rope; batchSharedPackedFusedQKRoPE19Half2Kernel; g361x1x1; b192x1x1; r28; s0 | 20 | n/a | 5.568 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 282 | unclassified.ordinal_282.fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 20 | n/a | 11.808 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 283 | unclassified.ordinal_283.library_gemm | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 20 | n/a | 8.480 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 284 | unclassified.ordinal_284.rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 20 | n/a | 2.464 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 285 | unclassified.ordinal_285.fused_ffn | fused_ffn | fused_ffn; fused_ffn_candidate_a_reuse_kernel; g18x37x1; b128x1x1; r136; s32768 | 20 | n/a | 38.288 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 286 | unclassified.ordinal_286.linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 20 | n/a | 20.561 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 287 | unclassified.ordinal_287.affine_silu | affine_silu | affine_silu; affineSiluHalf2Kernel; g4693x1x1; b192x1x1; r17; s0 | 20 | n/a | 3.328 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 288 | unclassified.ordinal_288.library_gemm | library_gemm | library_gemm; Kernel2; g148x1x1; b256x1x1; r154; s73728 | 20 | n/a | 14.288 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 289 | unclassified.ordinal_289.affine_silu | affine_silu | affine_silu; affineSiluHalf2Kernel; g4693x1x1; b384x1x1; r17; s0 | 20 | n/a | 5.040 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 290 | unclassified.ordinal_290.library_gemm | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 20 | n/a | 11.809 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 291 | unclassified.ordinal_291.rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 20 | n/a | 2.464 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 292 | unclassified.ordinal_292.copy_reformat | copy_reformat | copy_reformat; kernel_cutlass_kernel___main__FixedAtomLayoutGemmmakelocals_Kernel_object_at__CopyAtom_ThrID10_TVLayoutSrc1819201_TVLayoutDst1819201_Valuetypef16_tensor00o4693384111100_CopyAtom_ThrID10_T_0; g1x1x170; b288x1x1; r107; s99328 | 20 | n/a | 14.016 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 293 | unclassified.ordinal_293.qk_rope | qk_rope | qk_rope; batchSharedPackedFusedQKRoPE19Half2Kernel; g361x1x1; b192x1x1; r28; s0 | 20 | n/a | 5.584 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 294 | unclassified.ordinal_294.fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 20 | n/a | 11.808 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 295 | unclassified.ordinal_295.library_gemm | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 20 | n/a | 8.464 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 296 | unclassified.ordinal_296.rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 20 | n/a | 2.432 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 297 | unclassified.ordinal_297.fused_ffn | fused_ffn | fused_ffn; fused_ffn_candidate_a_reuse_kernel; g18x37x1; b128x1x1; r136; s32768 | 20 | n/a | 36.065 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 298 | unclassified.ordinal_298.linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 20 | n/a | 20.864 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 299 | unclassified.ordinal_299.rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 20 | n/a | 2.496 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 300 | unclassified.ordinal_300.copy_reformat | copy_reformat | copy_reformat; kernel_cutlass_kernel___main__FixedAtomLayoutGemmmakelocals_Kernel_object_at__CopyAtom_ThrID10_TVLayoutSrc1819201_TVLayoutDst1819201_Valuetypef16_tensor00o4693384111100_CopyAtom_ThrID10_T_0; g1x1x170; b288x1x1; r107; s99328 | 20 | n/a | 14.273 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 301 | unclassified.ordinal_301.qk_rope | qk_rope | qk_rope; batchSharedPackedFusedQKRoPE19Half2Kernel; g361x1x1; b192x1x1; r28; s0 | 20 | n/a | 5.664 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 302 | unclassified.ordinal_302.fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 20 | n/a | 11.952 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 303 | unclassified.ordinal_303.library_gemm | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 20 | n/a | 8.512 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 304 | unclassified.ordinal_304.rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 20 | n/a | 2.432 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 305 | unclassified.ordinal_305.fused_ffn | fused_ffn | fused_ffn; fused_ffn_candidate_a_reuse_kernel; g18x37x1; b128x1x1; r136; s32768 | 20 | n/a | 37.361 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 306 | unclassified.ordinal_306.linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 20 | n/a | 20.881 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 307 | unclassified.ordinal_307.rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 20 | n/a | 2.432 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 308 | unclassified.ordinal_308.copy_reformat | copy_reformat | copy_reformat; kernel_cutlass_kernel___main__FixedAtomLayoutGemmmakelocals_Kernel_object_at__CopyAtom_ThrID10_TVLayoutSrc1819201_TVLayoutDst1819201_Valuetypef16_tensor00o4693384111100_CopyAtom_ThrID10_T_0; g1x1x170; b288x1x1; r107; s99328 | 20 | n/a | 14.225 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 309 | unclassified.ordinal_309.qk_rope | qk_rope | qk_rope; batchSharedPackedFusedQKRoPE19Half2Kernel; g361x1x1; b192x1x1; r28; s0 | 20 | n/a | 5.680 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 310 | unclassified.ordinal_310.fa4 | fa4 | fa4; kernel_cutlass_kernel_flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120_object_at__tensorptrf16gmemalign16oi64div81i64div8i64div8_tensorptrf16gmemalign16oi64div81i64div8i64div8_tens_0; g3x12x13; b128x1x1; r247; s24576 | 20 | n/a | 12.000 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 311 | unclassified.ordinal_311.library_gemm | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r164; s81920 | 20 | n/a | 8.464 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 312 | unclassified.ordinal_312.rmsnorm | rmsnorm | rmsnorm; rmsNorm384Vec8Kernel; g1174x1x1; b128x1x1; r41; s0 | 20 | n/a | 2.496 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 313 | unclassified.ordinal_313.fused_ffn | fused_ffn | fused_ffn; fused_ffn_candidate_a_reuse_kernel; g18x37x1; b128x1x1; r136; s32768 | 20 | n/a | 40.833 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 314 | unclassified.ordinal_314.linear2_residual | linear2_residual | linear2_residual; linear2_residual_kernel; g3x37x1; b128x1x1; r162; s65536 | 20 | n/a | 20.960 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 315 | unclassified.ordinal_315.affine_silu | affine_silu | affine_silu; affineSiluHalf2Kernel; g4693x1x1; b192x1x1; r17; s0 | 20 | n/a | 3.376 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 316 | unclassified.ordinal_316.library_gemm | library_gemm | library_gemm; Kernel2; g148x1x1; b256x1x1; r154; s73728 | 20 | n/a | 14.400 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 317 | trunk.tip_norm_silu | affine_silu | affine_silu; affineSiluHalf2Kernel; g4693x1x1; b384x1x1; r17; s0 | 20 | n/a | 5.056 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 318 | policy.p1_conv | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r80; s98304 | 20 | n/a | 6.240 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 319 | policy.g1_conv | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r80; s98304 | 20 | n/a | 5.888 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 320 | policy.g1_norm_silu | head_elementwise | head_elementwise; applyCScaleBiasNHWCSiluHalfKernel; g1x73x13; b96x5x1; r16; s0 | 20 | n/a | 2.128 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 321 | policy.g1_half_to_float | copy_reformat | copy_reformat; copyFromHalfKernel; g880x1x1; b512x1x1; r16; s0 | 20 | n/a | 1.568 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 322 | policy.g1_global_pool | head_elementwise | head_elementwise; gPoolChannelsNHWCKernel; g2x1x13; b64x8x1; r22; s4096 | 20 | n/a | 4.448 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 323 | policy.gpool_to_bias_matmul | library_gemm | library_gemm; gemmSN_NN_kernel; g2x2x1; b256x1x1; r64; s21504 | 20 | n/a | 5.377 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 324 | policy.p1_half_to_float | copy_reformat | copy_reformat; copyFromHalfKernel; g880x1x1; b512x1x1; r16; s0 | 20 | n/a | 1.504 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 325 | policy.gpool_bias_add | head_elementwise | head_elementwise; addNCBiasInplaceNHWCKernel; g1x73x13; b96x5x1; r16; s0 | 20 | n/a | 1.792 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 326 | policy.p1_norm_silu | head_elementwise | head_elementwise; applyCScaleBiasNHWCSiluKernel; g1x73x13; b96x5x1; r16; s0 | 20 | n/a | 2.144 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 327 | policy.p2_conv | library_gemm | library_gemm; Kernel2; g74x1x1; b128x1x1; r90; s98304 | 20 | n/a | 3.936 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 328 | policy.gpool_to_pass_matmul | library_gemm | library_gemm; gemmSN_NN_kernel; g2x2x1; b256x1x1; r64; s21504 | 20 | n/a | 5.264 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 329 | policy.pass_bias_silu | head_elementwise | head_elementwise; addCBiasInplaceNCKernelSilu; g1x3x1; b96x5x1; r16; s0 | 20 | n/a | 1.024 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 330 | policy.gpool_to_pass_matmul2 | library_gemm | library_gemm; gemmSN_NN_kernel; g1x2x1; b256x1x1; r64; s21504 | 20 | n/a | 2.304 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 331 | value.v1_conv | library_gemm | library_gemm; Kernel2; g148x1x1; b128x1x1; r118; s98304 | 20 | n/a | 7.952 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 332 | value.v1_norm_silu | head_elementwise | head_elementwise; applyCScaleBiasNHWCSiluHalfKernel; g1x181x13; b192x2x1; r16; s0 | 20 | n/a | 3.168 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 333 | value.v1_half_to_float | copy_reformat | copy_reformat; copyFromHalfKernel; g1760x1x1; b512x1x1; r16; s0 | 20 | n/a | 2.144 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 334 | value.v1_global_pool | head_elementwise | head_elementwise; valueHeadPoolChannelsNHWCKernel; g3x1x13; b64x8x1; r22; s2048 | 20 | n/a | 3.248 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 335 | value.v2_matmul | library_gemm | library_gemm; gemmSN_NN_kernel; g3x2x1; b256x1x1; r64; s21504 | 20 | n/a | 9.504 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 336 | value.v2_bias_silu | head_elementwise | head_elementwise; addCBiasInplaceNCKernelSilu; g1x7x1; b192x2x1; r16; s0 | 20 | n/a | 1.024 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 337 | value.v3_matmul | library_gemm | library_gemm; gemmSN_NN_kernel; g1x2x1; b256x1x1; r64; s21504 | 20 | n/a | 3.472 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 338 | value.v3_bias | head_elementwise | head_elementwise; addCBiasInplaceNCKernel; g1x1x1; b3x170x1; r16; s0 | 20 | n/a | 0.960 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 339 | value.score_matmul | library_gemm | library_gemm; gemmSN_NN_kernel; g1x2x1; b256x1x1; r64; s21504 | 20 | n/a | 3.504 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 340 | value.score_bias | head_elementwise | head_elementwise; addCBiasInplaceNCKernel; g1x1x1; b6x85x1; r16; s0 | 20 | n/a | 0.928 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 341 | value.ownership_conv | library_gemm | library_gemm; Kernel2; g8x19x3; b128x1x1; r118; s33792 | 20 | n/a | 4.032 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 342 | value.ownership_conv_splitk_reduce | library_gemm | library_gemm; splitKreduce_kernel; g147x1x1; b32x16x1; r49; s0 | 20 | n/a | 1.376 | n/a | 0.0% | 0.000 | idle (20) | n/a |
| 343 | value.ownership_half_to_float | copy_reformat | copy_reformat; copyFromHalfKernel; g10x1x1; b512x1x1; r16; s0 | 20 | n/a | 0.928 | n/a | 0.0% | 0.000 | idle (20) | n/a |
