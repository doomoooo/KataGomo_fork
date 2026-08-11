# Stage 48 macro critical-path audit

## Scope

RTX 4090 SM89, exact 19x19 B13 FP16, two server streams. The initial source
trace is Stage 40's accepted S2 Nsys capture; all percentages below are interval
unions, not sums of overlapping kernel durations. Before implementation, Stage
48 reprofiles the frozen current-best binary with the short full-graph protocol
in `../ITERATION-PROTOCOL.md`. The old trace is only a hypothesis-ranking input
until the new binary-bound checkpoint confirms it.

The captured benchmark reports 3195.843 nnEval/s. For the two timed streams
(100 and 114), the combined kernel span is 269.119 ms and the GPU-busy union is
264.027 ms: 98.108% busy, with 5.092 ms of uncovered gaps. Thus whole-forward
CUDA Graph can recover at most 1.892% in this trace before its scheduling costs;
the historical B19/S2 graph experiment was already -0.164%.

## Current critical path

| Family | Family union (ms) | Only-family exclusive (ms) | Union busy share | Exclusive busy share |
|---|---:|---:|---:|---:|
| FFN dual projection + SwiGLU | 91.109 | 47.362 | 34.51% | 17.94% |
| QKV + RoPE | 63.339 | 28.169 | 23.99% | 10.67% |
| FlashAttention | 69.876 | 27.910 | 26.47% | 10.57% |
| FFN linear2 + residual | 56.949 | 7.525 | 21.57% | 2.85% |
| attention out-projection | 43.582 | 5.271 | 16.51% | 2.00% |
| RMSNorm | 23.704 | 3.657 | 8.98% | 1.39% |
| outer postConv | 15.493 | 2.166 | 5.87% | 0.82% |
| outer preConv | 11.474 | 1.848 | 4.35% | 0.70% |
| heads/frontend/other | 8.755 | 6.149 | 3.32% | 2.33% |
| outer C384 BN + SiLU | 5.546 | 1.102 | 2.10% | 0.42% |
| outer/trunk C768 BN + SiLU | 6.606 | 0.464 | 2.50% | 0.18% |

## Macro decisions

1. A conventional FFN projection-to-linear2 fusion is not viable on SM89.
   Avoiding the 4693x1152 intermediate makes each independent output-N CTA
   recompute the input projection. Current tiling implies roughly 3-6x redundant
   projection work, and SM89 has no cluster DSM for cross-CTA sharing.
2. QKV-to-Flash fusion has the same producer/consumer decomposition mismatch:
   projection tiles span channels while attention tiles span heads and sequence.
   Keeping QKV on chip requires either recomputation or a persistent cooperative
   design substantially different from the current kernel.
3. Whole-model CUDA Graph is not the next lever: the trace is already 98.11%
   busy and the prior S2 implementation regressed.
4. The outer postConv dual-output epilogue is feasible, but its removable C768
   pointwise boundary has only 0.18% exclusive share and the prior SM120/B19
   fused-SiLU experiment regressed 2.06%. It remains below RMS folding.
5. RMSNorm algebraic folding is the next implementable macro candidate. Gamma
   can be folded into QKV and FFN weights; a reduction-only kernel emits one
   invRMS scalar per row; QKV and dual-FFN epilogues apply that scalar before
   RoPE/SwiGLU. This removes 66 full C384 normalized-tensor writes and reads per
   forward while preserving the existing GEMM/attention decomposition.
6. Dual-stream trunk phase control remains the highest-level scheduling route,
   but its common SM89/SM120 interface is owned by the separate design track.

TensorRT is not a performance target to copy blindly: the locked exact-B13/S2
baseline is 2432.198 nnEval/s versus 3145.511 for CUDA in the same cross-backend
ABBA (+29.33% CUDA). Its value here is fusion evidence, not throughput authority.
