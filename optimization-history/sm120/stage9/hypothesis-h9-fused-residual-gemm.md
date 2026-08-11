# Hypothesis H9: GEMM beta residual fusion on SM120

Created: 2026-08-05 (UTC), before implementation. Primary target is RTX
5090D, fixed 19x19, B13/S2, FP16; S1 is measured independently.

## Evidence and mechanism

The accepted Stage-3 S2 trace shows the FP16 residual-add kernel consumes about
16.95 ms of raw time across 60 two-stream forwards. Each attention out-proj and
FFN linear2 currently writes a full 384-channel temporary, then launches a
kernel that reads the temporary and trunk and writes trunk.

For exact 19x19 inputs `maskBuf` is null. cuBLAS can instead write the
projection directly to trunk with GEMM beta=1, eliminating the temporary's
downstream read and the residual launch. Projection dimensions and weights are
unchanged; only the residual addition moves into the GEMM accumulation.

## Gates

Nsys must remove both transformer residual launches per block and show a
non-regressed projection-plus-residual subgraph. Short A-B-B-A must improve the
target S2 topology before long testing. Any retained topology requires at
least 0.5% long whole-network gain and must pass the Stage-3 full-FP32 8,192-row
accuracy gates. Masked, non-FP16, non-19x19, and unsupported projection shapes
fall back to the official temporary plus residual path.

The numerical risk is explicit: beta=1 may add trunk at a different point in
the GEMM accumulation than a separately rounded FP16 projection followed by a
half residual kernel.
