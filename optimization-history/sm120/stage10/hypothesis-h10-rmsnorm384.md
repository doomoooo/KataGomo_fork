# Hypothesis H10: one-warp C384 RMSNorm on SM120

Created: 2026-08-05 (UTC), before implementation. Target is the accepted
fused-residual baseline on RTX 5090D, fixed 19x19, B13/S2 and B13/S1, FP16.

The existing FP16 C384 RMSNorm launches one 192-thread block per token. Six
warps reduce through shared memory and two block barriers. For the fixed C384
transformer norm, one warp can own a row: each lane keeps 12 FP32 values,
performs one warp-shuffle reduction, and four warps process four rows in one
128-thread block. Gamma/beta arithmetic remains FP32 before FP16 storage.

Nsys must reduce RMSNorm kernel time without adding launches. Short A-B-B-A
must improve at least one production topology before full validation. A
retained candidate needs at least 0.5% long whole-network gain and the Stage-3
8,192-row full-FP32 accuracy gates. Unsupported channel counts, masks,
precisions, and board shapes retain the official kernel.

Risk: the reduction tree changes from six-warp hierarchical order to a
single-warp order, so full accuracy is mandatory even if the isolated output
error is small.
