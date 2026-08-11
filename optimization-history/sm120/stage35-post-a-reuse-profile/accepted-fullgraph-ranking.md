# Current full-graph Nsys + NCU ranking

- NCU fixed-forward coverage: 344 ordinals.
- Nsys timed topology: 30 iterations x 2 streams.
- Nsys supplies S2 timing and interference weight. NCU is replayed S1 evidence for resources and stalls; its replay duration is not used as S2 performance.

## Largest S2 work

| logical group | work us/fwd | work share | excess us/fwd | S2/S1 | regs | smem KiB | waves/SM | occ % | eligible % | tensor % | wait/issue |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| transformer.ffn_linear1_gate_swiglu | 1502.7 | 26.26% | 224.1 | 1.175x | 136 | 32.8 | 1.31 | 19.7 | 23.1 | 43.6 | 3.18 |
| transformer.ffn_linear2_residual | 1058.1 | 18.49% | 370.3 | 1.538x | 162 | 65.5 | 0.65 | 8.3 | 9.2 | 39.6 | 4.98 |
| transformer.attention_qkv_projection | 832.7 | 14.55% | 197.1 | 1.310x | 136 | 65.5 | 1.96 | 8.3 | 6.9 | 42.6 | 4.89 |
| transformer.attention_out_projection_residual | 606.1 | 10.59% | 326.2 | 2.165x | 164 | 81.9 | 0.87 | 8.3 | 15.4 | 31.0 | 3.40 |
| transformer.attention_fa4 | 558.5 | 9.76% | 166.1 | 1.423x | 247 | 24.6 | 1.38 | 14.6 | 37.5 | 47.7 | 1.37 |
| transformer.attention_qk_rope | 218.9 | 3.83% | 83.1 | 1.612x | 16 | 0.0 | 3.45 | 68.5 | 27.2 | 0.0 | 1.91 |
| outer.post_projection_c384_to_c768_residual | 206.3 | 3.60% | 47.9 | 1.302x | 154 | 73.7 | 0.87 | 16.6 | 19.6 | 38.2 | 3.13 |
| outer.pre_projection_c768_to_c384 | 186.7 | 3.26% | 55.6 | 1.424x | 164 | 81.9 | 0.87 | 8.3 | 17.6 | 45.1 | 3.88 |
| transformer.attention_rmsnorm | 140.5 | 2.46% | 60.0 | 1.745x | 41 | 0.0 | 0.69 | 43.7 | 31.2 | 0.0 | 1.25 |
| transformer.ffn_rmsnorm | 119.5 | 2.09% | 38.7 | 1.479x | 41 | 0.0 | 0.69 | 44.1 | 31.3 | 0.0 | 1.25 |
| outer.pre_norm_silu | 83.0 | 1.45% | 27.5 | 1.495x | 17 | 0.0 | 6.90 | 72.3 | 50.0 | 0.0 | 2.22 |
| outer.post_norm_silu | 46.4 | 0.81% | 9.6 | 1.261x | 17 | 0.0 | 3.45 | 61.3 | 43.1 | 0.0 | 2.20 |
| frontend.initial_conv | 21.5 | 0.38% | 2.0 | 1.103x | 94 | 81.9 | 5.22 | 8.3 | 24.3 | 42.4 | 0.74 |
| policy.g1_conv | 11.2 | 0.20% | 5.3 | 1.890x | 80 | 98.3 | 0.87 | 8.6 | 15.3 | 28.9 | 2.51 |
| value.v2_matmul | 10.9 | 0.19% | 1.3 | 1.142x | 64 | 0.0 | 0.01 | 16.8 | 27.2 | 0.0 | 0.81 |
| value.v1_conv | 9.9 | 0.17% | 2.0 | 1.245x | 118 | 98.3 | 0.87 | 8.3 | 15.5 | 42.5 | 3.51 |
| policy.p1_conv | 9.4 | 0.16% | 3.2 | 1.510x | 80 | 98.3 | 0.87 | 8.3 | 14.7 | 27.8 | 2.51 |
| frontend.initial_global_broadcast_add | 8.4 | 0.15% | 0.6 | 1.084x | 16 | 0.0 | 13.80 | 66.1 | 23.8 | 0.0 | 2.97 |
| trunk.tip_norm_silu | 7.5 | 0.13% | 2.4 | 1.478x | 17 | 0.0 | 6.90 | 71.8 | 50.5 | 0.0 | 2.22 |
| policy.g1_global_pool | 6.5 | 0.11% | 2.0 | 1.448x | 22 | 4.1 | 0.05 | 31.6 | 19.0 | 0.0 | 2.96 |

## Largest S2 interference excess

| logical group | excess us/fwd | work us/fwd | S2/S1 | waves/SM | eligible % | long-scoreboard/issue |
|---|---:|---:|---:|---:|---:|---:|
| transformer.ffn_linear2_residual | 370.3 | 1058.1 | 1.538x | 0.65 | 9.2 | 0.69 |
| transformer.attention_out_projection_residual | 326.2 | 606.1 | 2.165x | 0.87 | 15.4 | 1.29 |
| transformer.ffn_linear1_gate_swiglu | 224.1 | 1502.7 | 1.175x | 1.31 | 23.1 | 1.04 |
| transformer.attention_qkv_projection | 197.1 | 832.7 | 1.310x | 1.96 | 6.9 | 2.10 |
| transformer.attention_fa4 | 166.1 | 558.5 | 1.423x | 1.38 | 37.5 | 0.51 |
| transformer.attention_qk_rope | 83.1 | 218.9 | 1.612x | 3.45 | 27.2 | 23.68 |
| transformer.attention_rmsnorm | 60.0 | 140.5 | 1.745x | 0.69 | 31.2 | 11.49 |
| outer.pre_projection_c768_to_c384 | 55.6 | 186.7 | 1.424x | 0.87 | 17.6 | 0.29 |
| outer.post_projection_c384_to_c768_residual | 47.9 | 206.3 | 1.302x | 0.87 | 19.6 | 0.65 |
| transformer.ffn_rmsnorm | 38.7 | 119.5 | 1.479x | 0.69 | 31.3 | 11.40 |
| outer.pre_norm_silu | 27.5 | 83.0 | 1.495x | 6.90 | 50.0 | 7.78 |
| outer.post_norm_silu | 9.6 | 46.4 | 1.261x | 3.45 | 43.1 | 8.13 |
| policy.g1_conv | 5.3 | 11.2 | 1.890x | 0.87 | 15.3 | 1.29 |
| policy.p1_conv | 3.2 | 9.4 | 1.510x | 0.87 | 14.7 | 1.38 |
| trunk.tip_norm_silu | 2.4 | 7.5 | 1.478x | 6.90 | 50.5 | 7.52 |
| policy.g1_global_pool | 2.0 | 6.5 | 1.448x | 0.05 | 19.0 | 5.85 |
| frontend.initial_conv | 2.0 | 21.5 | 1.103x | 5.22 | 24.3 | 0.70 |
| value.v1_conv | 2.0 | 9.9 | 1.245x | 0.87 | 15.5 | 1.05 |
| policy.p2_conv | 1.7 | 5.6 | 1.429x | 0.44 | 18.7 | 1.16 |
| value.v2_matmul | 1.3 | 10.9 | 1.142x | 0.01 | 27.2 | 3.73 |
