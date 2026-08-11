# Hypothesis H6: single-wide FFN projection on SM120

Created: 2026-08-05 (UTC), before candidate microbenchmarking. Target regime
is RTX 5090D, fixed 19x19, B13, two independent server streams, FP16 I/O.

## Evidence and mechanism

Stage 5 rejected `cublasHgemmStridedBatched(batchCount=2)` by 2.147% at the
whole-network level. That API still asks cuBLAS to schedule two matrices. A
different formulation interleaves the two weight matrices by input channel
and computes one real `M=2304, N=4693, K=384` GEMM. Its output is
`[token][linear1(1152), gate(1152)]`; a layout-aware SwiGLU can read that
layout and write the compact 1152-channel activation consumed by linear2.

## Prediction and early gate

The original early gate required at least 10% GEMM savings under both one and
two streams. The microbenchmark produced 6.89% at S1 and 0.83% at S2. After
review, the candidate is reopened specifically for S1: 6.89% of the largest
FFN projection subgraph is enough to justify measuring the real layout-aware
SwiGLU rather than estimating it. S2 remains a required non-regression check,
not the reason to discard an S1-specific win.

Nsys must show one M=2304 projection plus one layout-aware
SwiGLU replacing the original two projections plus SwiGLU, with a non-regressed
combined subgraph. S1 acceptance requires at least 0.5% ordered A/B improvement
and the same Stage-3 full-FP32 accuracy gates. S2 is measured separately; a
regression there means the optimization remains opt-in for S1 rather than the
default S2 path.

## Risks

cuBLAS may already schedule the two independent M=1152 GEMMs efficiently, and
M=2304 may choose a worse tile. The interleaved output changes memory access
for SwiGLU and requires an out-of-place write, although total bytes read and
written are unchanged.
