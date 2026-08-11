# Current full-graph Nsys + NCU ranking

- NCU fixed-forward coverage: 344 ordinals.
- Nsys timed topology: 30 iterations x 2 streams.
- Nsys supplies S2 timing and interference weight. NCU is replayed S1 evidence for resources and stalls; its replay duration is not used as S2 performance.

## Largest S2 work

| logical group | work us/fwd | work share | excess us/fwd | S2/S1 | regs | smem KiB | waves/SM | occ % | eligible % | tensor % | wait/issue |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| transformer.ffn_linear1_gate_swiglu | 1366.7 | 24.60% | 369.6 | 1.371x | 96 | 50.2 | 1.00 | 34.2 | 34.9 | 54.3 | 3.08 |
| transformer.ffn_linear2_residual | 1044.3 | 18.79% | 350.1 | 1.504x | 162 | 65.5 | 0.65 | 8.3 | 9.2 | 39.7 | 4.98 |
| transformer.attention_qkv_projection | 773.5 | 13.92% | 302.5 | 1.642x | 107 | 99.3 | 1.00 | 18.4 | 15.9 | 56.9 | 4.01 |
| transformer.attention_out_projection_residual | 578.6 | 10.41% | 296.2 | 2.048x | 164 | 81.9 | 0.87 | 8.3 | 15.4 | 31.0 | 3.40 |
| transformer.attention_fa4 | 563.3 | 10.14% | 167.3 | 1.422x | 247 | 24.6 | 1.38 | 14.6 | 37.6 | 47.8 | 1.37 |
| transformer.attention_qk_rope | 276.7 | 4.98% | 89.1 | 1.475x | 28 | 0.0 | 0.27 | 25.5 | 9.3 | 0.0 | 1.85 |
| outer.post_projection_c384_to_c768_residual | 215.6 | 3.88% | 55.2 | 1.344x | 154 | 73.7 | 0.87 | 16.6 | 19.6 | 38.0 | 3.13 |
| outer.pre_projection_c768_to_c384 | 175.1 | 3.15% | 42.9 | 1.325x | 164 | 81.9 | 0.87 | 8.3 | 17.6 | 45.2 | 3.88 |
| transformer.attention_rmsnorm | 140.6 | 2.53% | 59.3 | 1.730x | 41 | 0.0 | 0.69 | 43.5 | 31.3 | 0.0 | 1.25 |
| transformer.ffn_rmsnorm | 118.2 | 2.13% | 36.8 | 1.451x | 41 | 0.0 | 0.69 | 44.0 | 31.4 | 0.0 | 1.25 |
| outer.pre_norm_silu | 86.6 | 1.56% | 30.4 | 1.540x | 17 | 0.0 | 6.90 | 72.4 | 49.9 | 0.0 | 2.22 |
| outer.post_norm_silu | 48.1 | 0.87% | 10.7 | 1.287x | 17 | 0.0 | 3.45 | 60.6 | 42.4 | 0.0 | 2.20 |
| frontend.initial_conv | 22.6 | 0.41% | 3.2 | 1.163x | 94 | 81.9 | 5.22 | 8.3 | 24.6 | 42.7 | 0.74 |
| policy.g1_conv | 13.3 | 0.24% | 7.3 | 2.219x | 80 | 98.3 | 0.87 | 8.3 | 15.2 | 28.8 | 2.51 |
| value.v2_matmul | 11.6 | 0.21% | 2.1 | 1.224x | 64 | 0.0 | 0.01 | 16.6 | 27.2 | 0.0 | 0.81 |
| value.v1_conv | 9.6 | 0.17% | 1.6 | 1.201x | 118 | 98.3 | 0.87 | 8.3 | 15.4 | 42.9 | 3.50 |
| policy.p1_conv | 8.5 | 0.15% | 2.2 | 1.358x | 80 | 98.3 | 0.87 | 8.3 | 15.0 | 28.1 | 2.51 |
| frontend.initial_global_broadcast_add | 8.5 | 0.15% | 0.8 | 1.101x | 16 | 0.0 | 13.80 | 65.1 | 24.4 | 0.0 | 2.97 |
| trunk.tip_norm_silu | 7.5 | 0.14% | 2.4 | 1.461x | 17 | 0.0 | 6.90 | 72.7 | 50.7 | 0.0 | 2.22 |
| policy.p2_conv | 6.6 | 0.12% | 2.6 | 1.660x | 90 | 98.3 | 0.44 | 8.3 | 18.7 | 10.5 | 1.40 |
| policy.gpool_to_bias_matmul | 6.3 | 0.11% | 0.8 | 1.151x | 64 | 0.0 | 0.01 | 16.6 | 21.2 | 0.0 | 0.92 |
| policy.gpool_to_pass_matmul | 6.1 | 0.11% | 0.7 | 1.140x | 64 | 0.0 | 0.01 | 16.7 | 21.4 | 0.0 | 0.91 |
| policy.g1_global_pool | 5.8 | 0.11% | 1.3 | 1.295x | 22 | 4.1 | 0.05 | 31.6 | 19.0 | 0.0 | 2.97 |
| value.ownership_conv | 4.8 | 0.09% | 0.8 | 1.193x | 118 | 33.8 | 1.34 | 13.9 | 38.8 | 5.6 | 1.16 |
| value.v3_matmul | 4.2 | 0.08% | 0.8 | 1.217x | 64 | 0.0 | 0.00 | 16.6 | 18.5 | 0.0 | 1.05 |
| value.score_matmul | 4.1 | 0.07% | 0.6 | 1.181x | 64 | 0.0 | 0.00 | 16.7 | 18.5 | 0.0 | 1.04 |
| value.v1_global_pool | 4.0 | 0.07% | 0.8 | 1.239x | 22 | 2.0 | 0.08 | 30.1 | 22.2 | 0.0 | 2.27 |
| frontend.initial_global_matmul | 4.0 | 0.07% | 1.4 | 1.516x | 128 | 24.6 | 0.04 | 8.3 | 29.3 | 1.3 | 1.32 |
| value.v1_norm_silu | 3.6 | 0.06% | 0.4 | 1.124x | 16 | 0.0 | 3.46 | 69.9 | 38.5 | 0.0 | 2.82 |
| policy.gpool_to_pass_matmul2 | 3.3 | 0.06% | 1.0 | 1.409x | 64 | 0.0 | 0.00 | 16.6 | 19.2 | 0.0 | 1.21 |

## Largest S2 interference excess

| logical group | excess us/fwd | work us/fwd | S2/S1 | waves/SM | eligible % | long-scoreboard/issue |
|---|---:|---:|---:|---:|---:|---:|
| transformer.ffn_linear1_gate_swiglu | 369.6 | 1366.7 | 1.371x | 1.00 | 34.9 | 0.64 |
| transformer.ffn_linear2_residual | 350.1 | 1044.3 | 1.504x | 0.65 | 9.2 | 0.69 |
| transformer.attention_qkv_projection | 302.5 | 773.5 | 1.642x | 1.00 | 15.9 | 0.87 |
| transformer.attention_out_projection_residual | 296.2 | 578.6 | 2.048x | 0.87 | 15.4 | 1.29 |
| transformer.attention_fa4 | 167.3 | 563.3 | 1.422x | 1.38 | 37.6 | 0.51 |
| transformer.attention_qk_rope | 89.1 | 276.7 | 1.475x | 0.27 | 9.3 | 29.72 |
| transformer.attention_rmsnorm | 59.3 | 140.6 | 1.730x | 0.69 | 31.3 | 11.46 |
| outer.post_projection_c384_to_c768_residual | 55.2 | 215.6 | 1.344x | 0.87 | 19.6 | 0.66 |
| outer.pre_projection_c768_to_c384 | 42.9 | 175.1 | 1.325x | 0.87 | 17.6 | 0.29 |
| transformer.ffn_rmsnorm | 36.8 | 118.2 | 1.451x | 0.69 | 31.4 | 11.43 |
| outer.pre_norm_silu | 30.4 | 86.6 | 1.540x | 6.90 | 49.9 | 7.78 |
| outer.post_norm_silu | 10.7 | 48.1 | 1.287x | 3.45 | 42.4 | 8.24 |
| policy.g1_conv | 7.3 | 13.3 | 2.219x | 0.87 | 15.2 | 1.28 |
| frontend.initial_conv | 3.2 | 22.6 | 1.163x | 5.22 | 24.6 | 0.67 |
| policy.p2_conv | 2.6 | 6.6 | 1.660x | 0.44 | 18.7 | 1.16 |
| trunk.tip_norm_silu | 2.4 | 7.5 | 1.461x | 6.90 | 50.7 | 7.40 |
| policy.p1_conv | 2.2 | 8.5 | 1.358x | 0.87 | 15.0 | 1.32 |
| value.v2_matmul | 2.1 | 11.6 | 1.224x | 0.01 | 27.2 | 3.77 |
| frontend.initial_conv_nhwc_padding_0 | 2.0 | 3.2 | 2.531x | 1.00 | 40.0 | 3.44 |
| value.v1_conv | 1.6 | 9.6 | 1.201x | 0.87 | 15.4 | 1.05 |
| frontend.initial_global_matmul | 1.4 | 4.0 | 1.516x | 0.04 | 29.3 | 0.12 |
| policy.g1_global_pool | 1.3 | 5.8 | 1.295x | 0.05 | 19.0 | 5.85 |
| policy.gpool_to_pass_matmul2 | 1.0 | 3.3 | 1.409x | 0.00 | 19.2 | 1.84 |
| policy.g1_norm_silu | 0.9 | 3.1 | 1.440x | 1.86 | 38.6 | 7.87 |
| policy.gpool_to_bias_matmul | 0.8 | 6.3 | 1.151x | 0.01 | 21.2 | 3.59 |
| value.ownership_conv | 0.8 | 4.8 | 1.193x | 1.34 | 38.8 | 0.44 |
| frontend.initial_global_broadcast_add | 0.8 | 8.5 | 1.101x | 13.80 | 24.4 | 18.84 |
| value.v1_global_pool | 0.8 | 4.0 | 1.239x | 0.08 | 22.2 | 11.60 |
| value.v3_matmul | 0.8 | 4.2 | 1.217x | 0.00 | 18.5 | 2.71 |
| policy.gpool_to_pass_matmul | 0.7 | 6.1 | 1.140x | 0.01 | 21.4 | 3.58 |
