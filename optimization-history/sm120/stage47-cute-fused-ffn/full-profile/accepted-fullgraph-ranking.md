# Current full-graph Nsys + NCU ranking

- NCU fixed-forward coverage: 344 ordinals.
- Nsys timed topology: 30 iterations x 2 streams.
- Nsys supplies S2 timing and interference weight. NCU is replayed S1 evidence for resources and stalls; its replay duration is not used as S2 performance.

## Largest S2 work

| logical group | work us/fwd | work share | excess us/fwd | S2/S1 | regs | smem KiB | waves/SM | occ % | eligible % | tensor % | wait/issue |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| transformer.ffn_linear1_gate_swiglu | 1364.7 | 24.55% | 370.8 | 1.373x | 96 | 50.2 | 1.00 | 34.1 | 35.1 | 54.3 | 3.09 |
| transformer.ffn_linear2_residual | 1025.9 | 18.45% | 335.3 | 1.486x | 162 | 65.5 | 0.65 | 8.3 | 9.2 | 39.7 | 4.98 |
| transformer.attention_qkv_projection | 750.6 | 13.50% | 280.6 | 1.597x | 107 | 99.3 | 1.00 | 18.5 | 15.9 | 57.1 | 4.01 |
| transformer.attention_out_projection_residual | 577.7 | 10.39% | 295.9 | 2.050x | 164 | 81.9 | 0.87 | 8.3 | 15.4 | 30.9 | 3.40 |
| transformer.attention_fa4 | 563.5 | 10.13% | 169.8 | 1.431x | 247 | 24.6 | 1.38 | 14.6 | 37.6 | 47.7 | 1.37 |
| transformer.attention_qk_rope | 270.4 | 4.86% | 83.1 | 1.444x | 28 | 0.0 | 0.27 | 25.5 | 9.3 | 0.0 | 1.85 |
| outer.post_projection_c384_to_c768_residual | 226.2 | 4.07% | 66.3 | 1.415x | 154 | 73.7 | 0.87 | 16.6 | 19.6 | 37.9 | 3.13 |
| outer.pre_projection_c768_to_c384 | 184.5 | 3.32% | 52.9 | 1.402x | 164 | 81.9 | 0.87 | 8.3 | 17.6 | 45.2 | 3.88 |
| transformer.attention_rmsnorm | 132.5 | 2.38% | 51.4 | 1.635x | 41 | 0.0 | 0.69 | 43.7 | 30.6 | 0.0 | 1.25 |
| transformer.ffn_rmsnorm | 118.5 | 2.13% | 37.1 | 1.456x | 41 | 0.0 | 0.69 | 43.6 | 31.2 | 0.0 | 1.25 |
| outer.pre_norm_silu | 92.3 | 1.66% | 36.3 | 1.648x | 17 | 0.0 | 6.90 | 72.4 | 49.8 | 0.0 | 2.22 |
| outer.post_norm_silu | 56.6 | 1.02% | 19.3 | 1.518x | 17 | 0.0 | 3.45 | 61.2 | 42.5 | 0.0 | 2.20 |
| frontend.initial_conv | 24.8 | 0.45% | 5.3 | 1.274x | 94 | 81.9 | 5.22 | 8.3 | 24.3 | 41.9 | 0.74 |
| policy.g1_conv | 15.1 | 0.27% | 9.1 | 2.522x | 80 | 98.3 | 0.87 | 8.3 | 15.4 | 29.0 | 2.51 |
| value.v1_conv | 13.9 | 0.25% | 5.9 | 1.735x | 118 | 98.3 | 0.87 | 8.3 | 15.4 | 43.4 | 3.51 |
| value.v2_matmul | 12.3 | 0.22% | 2.8 | 1.297x | 64 | 0.0 | 0.01 | 16.5 | 27.8 | 0.0 | 0.81 |
| frontend.initial_global_broadcast_add | 9.5 | 0.17% | 1.8 | 1.228x | 16 | 0.0 | 13.80 | 68.1 | 25.2 | 0.0 | 2.97 |
| policy.p1_conv | 8.8 | 0.16% | 2.6 | 1.413x | 80 | 98.3 | 0.87 | 8.4 | 14.8 | 28.9 | 2.51 |
| policy.p2_conv | 8.7 | 0.16% | 4.7 | 2.186x | 90 | 98.3 | 0.44 | 8.3 | 18.7 | 10.5 | 1.40 |
| trunk.tip_norm_silu | 7.9 | 0.14% | 2.8 | 1.551x | 17 | 0.0 | 6.90 | 75.2 | 51.8 | 0.0 | 2.23 |
| value.ownership_conv | 7.8 | 0.14% | 3.8 | 1.945x | 118 | 33.8 | 1.34 | 14.1 | 38.6 | 5.6 | 1.16 |
| policy.gpool_to_pass_matmul | 7.2 | 0.13% | 1.9 | 1.354x | 64 | 0.0 | 0.01 | 16.8 | 21.1 | 0.0 | 0.92 |
| policy.gpool_to_bias_matmul | 6.7 | 0.12% | 1.3 | 1.239x | 64 | 0.0 | 0.01 | 16.9 | 21.0 | 0.0 | 0.92 |
| policy.g1_global_pool | 6.5 | 0.12% | 1.9 | 1.429x | 22 | 4.1 | 0.05 | 31.5 | 19.0 | 0.0 | 2.97 |
| value.v1_norm_silu | 5.3 | 0.10% | 2.2 | 1.680x | 16 | 0.0 | 3.46 | 70.8 | 38.4 | 0.0 | 2.82 |
| value.v1_global_pool | 5.1 | 0.09% | 1.9 | 1.585x | 22 | 2.0 | 0.08 | 31.7 | 22.7 | 0.0 | 2.28 |
| frontend.initial_global_matmul | 4.7 | 0.08% | 2.1 | 1.784x | 128 | 24.6 | 0.04 | 8.3 | 29.3 | 1.3 | 1.32 |
| value.v3_matmul | 4.4 | 0.08% | 1.0 | 1.276x | 64 | 0.0 | 0.00 | 16.6 | 18.5 | 0.0 | 1.05 |
| value.score_matmul | 4.1 | 0.07% | 0.6 | 1.175x | 64 | 0.0 | 0.00 | 16.6 | 18.5 | 0.0 | 1.04 |
| policy.p1_norm_silu | 3.9 | 0.07% | 1.8 | 1.815x | 16 | 0.0 | 1.86 | 68.6 | 36.9 | 0.0 | 2.78 |

## Largest S2 interference excess

| logical group | excess us/fwd | work us/fwd | S2/S1 | waves/SM | eligible % | long-scoreboard/issue |
|---|---:|---:|---:|---:|---:|---:|
| transformer.ffn_linear1_gate_swiglu | 370.8 | 1364.7 | 1.373x | 1.00 | 35.1 | 0.64 |
| transformer.ffn_linear2_residual | 335.3 | 1025.9 | 1.486x | 0.65 | 9.2 | 0.69 |
| transformer.attention_out_projection_residual | 295.9 | 577.7 | 2.050x | 0.87 | 15.4 | 1.29 |
| transformer.attention_qkv_projection | 280.6 | 750.6 | 1.597x | 1.00 | 15.9 | 0.87 |
| transformer.attention_fa4 | 169.8 | 563.5 | 1.431x | 1.38 | 37.6 | 0.51 |
| transformer.attention_qk_rope | 83.1 | 270.4 | 1.444x | 0.27 | 9.3 | 29.72 |
| outer.post_projection_c384_to_c768_residual | 66.3 | 226.2 | 1.415x | 0.87 | 19.6 | 0.67 |
| outer.pre_projection_c768_to_c384 | 52.9 | 184.5 | 1.402x | 0.87 | 17.6 | 0.29 |
| transformer.attention_rmsnorm | 51.4 | 132.5 | 1.635x | 0.69 | 30.6 | 11.52 |
| transformer.ffn_rmsnorm | 37.1 | 118.5 | 1.456x | 0.69 | 31.2 | 11.48 |
| outer.pre_norm_silu | 36.3 | 92.3 | 1.648x | 6.90 | 49.8 | 7.81 |
| outer.post_norm_silu | 19.3 | 56.6 | 1.518x | 3.45 | 42.5 | 8.11 |
| policy.g1_conv | 9.1 | 15.1 | 2.522x | 0.87 | 15.4 | 1.29 |
| value.v1_conv | 5.9 | 13.9 | 1.735x | 0.87 | 15.4 | 1.05 |
| frontend.initial_conv | 5.3 | 24.8 | 1.274x | 5.22 | 24.3 | 0.70 |
| policy.p2_conv | 4.7 | 8.7 | 2.186x | 0.44 | 18.7 | 1.16 |
| value.ownership_conv | 3.8 | 7.8 | 1.945x | 1.34 | 38.6 | 0.46 |
| value.v2_matmul | 2.8 | 12.3 | 1.297x | 0.01 | 27.8 | 3.68 |
| trunk.tip_norm_silu | 2.8 | 7.9 | 1.551x | 6.90 | 51.8 | 7.11 |
| policy.p1_conv | 2.6 | 8.8 | 1.413x | 0.87 | 14.8 | 1.15 |
| value.v1_norm_silu | 2.2 | 5.3 | 1.680x | 3.46 | 38.4 | 9.77 |
| frontend.initial_global_matmul | 2.1 | 4.7 | 1.784x | 0.04 | 29.3 | 0.12 |
| policy.g1_global_pool | 1.9 | 6.5 | 1.429x | 0.05 | 19.0 | 5.85 |
| value.ownership_conv_splitk_reduce | 1.9 | 3.3 | 2.374x | 0.43 | 20.4 | 1.04 |
| value.v1_global_pool | 1.9 | 5.1 | 1.585x | 0.08 | 22.7 | 11.22 |
| policy.gpool_to_pass_matmul | 1.9 | 7.2 | 1.354x | 0.01 | 21.1 | 3.62 |
| policy.p1_norm_silu | 1.8 | 3.9 | 1.815x | 1.86 | 36.9 | 8.59 |
| frontend.initial_global_broadcast_add | 1.8 | 9.5 | 1.228x | 13.80 | 25.2 | 18.31 |
| frontend.initial_conv_nhwc_padding_1 | 1.7 | 3.3 | 2.139x | 1.00 | 40.4 | 3.89 |
| policy.g1_norm_silu | 1.5 | 3.6 | 1.698x | 1.86 | 38.0 | 8.37 |
