# Current full-graph Nsys + NCU ranking

- NCU fixed-forward coverage: 344 ordinals.
- Nsys timed topology: 30 iterations x 2 streams.
- Nsys supplies S2 timing and interference weight. NCU is replayed S1 evidence for resources and stalls; its replay duration is not used as S2 performance.

## Largest S2 work

| logical group | work us/fwd | work share | excess us/fwd | S2/S1 | regs | smem KiB | waves/SM | occ % | eligible % | tensor % | wait/issue |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| transformer.ffn_linear1_gate_swiglu | 1479.4 | 25.94% | 203.5 | 1.158x | 136 | 32.8 | 1.31 | 19.7 | 23.1 | 43.4 | 3.19 |
| transformer.ffn_linear2_residual | 1029.1 | 18.05% | 341.2 | 1.496x | 162 | 65.5 | 0.65 | 8.3 | 9.2 | 39.6 | 4.98 |
| transformer.attention_qkv_projection | 785.1 | 13.77% | 149.6 | 1.235x | 136 | 65.5 | 1.96 | 8.3 | 6.9 | 42.7 | 4.89 |
| transformer.attention_out_projection_residual | 616.9 | 10.82% | 337.1 | 2.205x | 164 | 81.9 | 0.87 | 8.3 | 15.4 | 31.0 | 3.40 |
| transformer.attention_fa4 | 592.0 | 10.38% | 200.2 | 1.511x | 247 | 24.6 | 1.38 | 14.6 | 37.6 | 47.7 | 1.37 |
| outer.post_projection_c384_to_c768_residual | 254.2 | 4.46% | 96.1 | 1.607x | 154 | 73.7 | 0.87 | 16.6 | 19.6 | 38.0 | 3.13 |
| outer.pre_projection_c768_to_c384 | 181.5 | 3.18% | 50.6 | 1.387x | 164 | 81.9 | 0.87 | 8.3 | 17.6 | 45.2 | 3.88 |
| transformer.attention_qk_rope | 177.9 | 3.12% | 51.0 | 1.402x | 16 | 0.0 | 3.45 | 66.8 | 25.5 | 0.0 | 2.02 |
| transformer.attention_rmsnorm | 127.4 | 2.23% | 47.3 | 1.590x | 41 | 0.0 | 0.69 | 43.5 | 30.5 | 0.0 | 1.25 |
| transformer.ffn_rmsnorm | 114.9 | 2.02% | 34.3 | 1.425x | 41 | 0.0 | 0.69 | 43.9 | 31.3 | 0.0 | 1.25 |
| outer.pre_norm_silu | 94.7 | 1.66% | 39.2 | 1.706x | 17 | 0.0 | 6.90 | 72.6 | 50.1 | 0.0 | 2.22 |
| outer.post_norm_silu | 59.7 | 1.05% | 22.9 | 1.621x | 17 | 0.0 | 3.45 | 60.7 | 42.8 | 0.0 | 2.20 |
| frontend.initial_conv | 24.2 | 0.42% | 4.7 | 1.240x | 94 | 81.9 | 5.22 | 8.3 | 24.4 | 42.1 | 0.74 |
| value.v1_conv | 15.8 | 0.28% | 7.9 | 1.994x | 118 | 98.3 | 0.87 | 8.3 | 15.4 | 43.1 | 3.51 |
| policy.g1_conv | 12.2 | 0.21% | 6.3 | 2.060x | 80 | 98.3 | 0.87 | 8.5 | 15.3 | 28.8 | 2.51 |
| value.v2_matmul | 11.9 | 0.21% | 2.4 | 1.250x | 64 | 0.0 | 0.01 | 16.7 | 27.2 | 0.0 | 0.81 |
| frontend.initial_global_broadcast_add | 10.1 | 0.18% | 2.4 | 1.312x | 16 | 0.0 | 13.80 | 65.8 | 23.9 | 0.0 | 2.97 |
| policy.p1_conv | 8.5 | 0.15% | 2.3 | 1.374x | 80 | 98.3 | 0.87 | 8.4 | 14.7 | 27.7 | 2.51 |
| trunk.tip_norm_silu | 8.1 | 0.14% | 3.0 | 1.595x | 17 | 0.0 | 6.90 | 71.8 | 50.6 | 0.0 | 2.22 |
| policy.g1_global_pool | 7.7 | 0.13% | 3.2 | 1.712x | 22 | 4.1 | 0.05 | 31.5 | 19.1 | 0.0 | 2.97 |

## Largest S2 interference excess

| logical group | excess us/fwd | work us/fwd | S2/S1 | waves/SM | eligible % | long-scoreboard/issue |
|---|---:|---:|---:|---:|---:|---:|
| transformer.ffn_linear2_residual | 341.2 | 1029.1 | 1.496x | 0.65 | 9.2 | 0.69 |
| transformer.attention_out_projection_residual | 337.1 | 616.9 | 2.205x | 0.87 | 15.4 | 1.29 |
| transformer.ffn_linear1_gate_swiglu | 203.5 | 1479.4 | 1.158x | 1.31 | 23.1 | 1.02 |
| transformer.attention_fa4 | 200.2 | 592.0 | 1.511x | 1.38 | 37.6 | 0.51 |
| transformer.attention_qkv_projection | 149.6 | 785.1 | 1.235x | 1.96 | 6.9 | 2.09 |
| outer.post_projection_c384_to_c768_residual | 96.1 | 254.2 | 1.607x | 0.87 | 19.6 | 0.67 |
| transformer.attention_qk_rope | 51.0 | 177.9 | 1.402x | 3.45 | 25.5 | 24.67 |
| outer.pre_projection_c768_to_c384 | 50.6 | 181.5 | 1.387x | 0.87 | 17.6 | 0.29 |
| transformer.attention_rmsnorm | 47.3 | 127.4 | 1.590x | 0.69 | 30.5 | 11.66 |
| outer.pre_norm_silu | 39.2 | 94.7 | 1.706x | 6.90 | 50.1 | 7.57 |
| transformer.ffn_rmsnorm | 34.3 | 114.9 | 1.425x | 0.69 | 31.3 | 11.39 |
| outer.post_norm_silu | 22.9 | 59.7 | 1.621x | 3.45 | 42.8 | 8.17 |
| value.v1_conv | 7.9 | 15.8 | 1.994x | 0.87 | 15.4 | 1.04 |
| policy.g1_conv | 6.3 | 12.2 | 2.060x | 0.87 | 15.3 | 1.29 |
| frontend.initial_conv | 4.7 | 24.2 | 1.240x | 5.22 | 24.4 | 0.71 |
| policy.g1_global_pool | 3.2 | 7.7 | 1.712x | 0.05 | 19.1 | 5.86 |
| trunk.tip_norm_silu | 3.0 | 8.1 | 1.595x | 6.90 | 50.6 | 7.54 |
| frontend.initial_global_broadcast_add | 2.4 | 10.1 | 1.312x | 13.80 | 23.9 | 19.02 |
| frontend.initial_global_matmul | 2.4 | 5.0 | 1.914x | 0.04 | 29.3 | 0.12 |
| value.v2_matmul | 2.4 | 11.9 | 1.250x | 0.01 | 27.2 | 3.73 |
