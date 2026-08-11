# Current full-graph Nsys + NCU ranking

- NCU fixed-forward coverage: 344 ordinals.
- Nsys timed topology: 30 iterations x 2 streams.
- Nsys supplies S2 timing and interference weight. NCU is replayed S1 evidence for resources and stalls; its replay duration is not used as S2 performance.

## Largest S2 work

| logical group | work us/fwd | work share | excess us/fwd | S2/S1 | regs | smem KiB | waves/SM | occ % | eligible % | tensor % | wait/issue |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| transformer.ffn_linear1_gate_swiglu | 1477.6 | 25.90% | 119.1 | 1.086x | 146 | 32.8 | 1.31 | 20.3 | 23.4 | 40.5 | 3.07 |
| transformer.ffn_linear2_residual | 1077.0 | 18.88% | 395.4 | 1.580x | 162 | 65.5 | 0.65 | 8.3 | 9.2 | 39.7 | 4.98 |
| transformer.attention_qkv_projection | 833.9 | 14.62% | 204.2 | 1.324x | 136 | 65.5 | 1.96 | 8.3 | 6.9 | 42.6 | 4.89 |
| transformer.attention_out_projection_residual | 581.5 | 10.20% | 303.8 | 2.094x | 164 | 81.9 | 0.87 | 8.3 | 15.4 | 30.9 | 3.40 |
| transformer.attention_fa4 | 492.6 | 8.64% | 103.7 | 1.267x | 247 | 24.6 | 1.38 | 14.6 | 37.5 | 47.7 | 1.37 |
| transformer.attention_qk_rope | 247.4 | 4.34% | 112.5 | 1.834x | 16 | 0.0 | 3.45 | 68.0 | 27.3 | 0.0 | 1.91 |
| outer.post_projection_c384_to_c768_residual | 237.8 | 4.17% | 80.8 | 1.515x | 154 | 73.7 | 0.87 | 16.6 | 19.6 | 37.9 | 3.13 |
| outer.pre_projection_c768_to_c384 | 189.2 | 3.32% | 59.4 | 1.457x | 164 | 81.9 | 0.87 | 8.3 | 17.6 | 45.2 | 3.88 |
| transformer.attention_rmsnorm | 137.5 | 2.41% | 57.6 | 1.721x | 41 | 0.0 | 0.69 | 43.8 | 31.4 | 0.0 | 1.25 |
| transformer.ffn_rmsnorm | 134.9 | 2.37% | 54.6 | 1.680x | 41 | 0.0 | 0.69 | 43.8 | 31.1 | 0.0 | 1.25 |
| outer.pre_norm_silu | 89.8 | 1.57% | 34.7 | 1.629x | 17 | 0.0 | 6.90 | 72.3 | 49.9 | 0.0 | 2.22 |
| outer.post_norm_silu | 45.0 | 0.79% | 8.4 | 1.229x | 17 | 0.0 | 3.45 | 61.2 | 42.7 | 0.0 | 2.20 |
| frontend.initial_conv | 21.7 | 0.38% | 2.2 | 1.114x | 94 | 81.9 | 5.22 | 8.3 | 24.4 | 42.0 | 0.74 |
| value.v2_matmul | 10.4 | 0.18% | 1.0 | 1.102x | 64 | 0.0 | 0.01 | 16.7 | 27.3 | 0.0 | 0.81 |
| policy.p1_conv | 10.3 | 0.18% | 4.1 | 1.668x | 80 | 98.3 | 0.87 | 8.3 | 14.8 | 27.7 | 2.51 |
| value.v1_conv | 9.5 | 0.17% | 1.6 | 1.197x | 118 | 98.3 | 0.87 | 8.8 | 16.2 | 43.2 | 3.51 |

## Largest S2 interference excess

| logical group | excess us/fwd | work us/fwd | S2/S1 | waves/SM | eligible % | long-scoreboard/issue |
|---|---:|---:|---:|---:|---:|---:|
| transformer.ffn_linear2_residual | 395.4 | 1077.0 | 1.580x | 0.65 | 9.2 | 0.69 |
| transformer.attention_out_projection_residual | 303.8 | 581.5 | 2.094x | 0.87 | 15.4 | 1.29 |
| transformer.attention_qkv_projection | 204.2 | 833.9 | 1.324x | 1.96 | 6.9 | 2.08 |
| transformer.ffn_linear1_gate_swiglu | 119.1 | 1477.6 | 1.086x | 1.31 | 23.4 | 1.77 |
| transformer.attention_qk_rope | 112.5 | 247.4 | 1.834x | 3.45 | 27.3 | 23.55 |
| transformer.attention_fa4 | 103.7 | 492.6 | 1.267x | 1.38 | 37.5 | 0.51 |
| outer.post_projection_c384_to_c768_residual | 80.8 | 237.8 | 1.515x | 0.87 | 19.6 | 0.67 |
| outer.pre_projection_c768_to_c384 | 59.4 | 189.2 | 1.457x | 0.87 | 17.6 | 0.29 |
| transformer.attention_rmsnorm | 57.6 | 137.5 | 1.721x | 0.69 | 31.4 | 11.41 |
| transformer.ffn_rmsnorm | 54.6 | 134.9 | 1.680x | 0.69 | 31.1 | 11.40 |
| outer.pre_norm_silu | 34.7 | 89.8 | 1.629x | 6.90 | 49.9 | 7.79 |
| outer.post_norm_silu | 8.4 | 45.0 | 1.229x | 3.45 | 42.7 | 8.15 |
| policy.p1_conv | 4.1 | 10.3 | 1.668x | 0.87 | 14.8 | 1.36 |
| trunk.tip_norm_silu | 2.9 | 8.0 | 1.587x | 6.90 | 50.3 | 7.54 |
| policy.g1_global_pool | 2.9 | 7.3 | 1.661x | 0.05 | 19.3 | 5.62 |
| frontend.initial_conv | 2.2 | 21.7 | 1.114x | 5.22 | 24.4 | 0.70 |
