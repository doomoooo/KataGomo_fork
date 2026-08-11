# Hypothesis H7: wide QKV projection on SM120

Created: 2026-08-05 (UTC), before measurement. Target is RTX 5090D, fixed
19x19, B13, two streams, FP16.

Each attention block currently submits three `M=384, N=4693, K=384` GEMMs over
the same RMS-normalized input. A strided batch-count-3 call preserves separate
contiguous Q/K/V buffers and needs no layout conversion. A true M=1152 GEMM
may be faster but interleaves Q/K/V per token, so it would require a fused or
layout-aware RoPE before FA4.

Early gates:

- strided batch-3 must beat three ordinary GEMMs under both one and two
  streams to justify direct integration;
- a true M=1152 GEMM must save at least 10% under two streams to justify the
  additional RoPE/layout work.

Any integrated candidate must then show a non-regressed QKV+RoPE subgraph in
Nsys, at least 0.5% whole-network ordered A/B gain, and pass the Stage-3
full-FP32 accuracy gates.
