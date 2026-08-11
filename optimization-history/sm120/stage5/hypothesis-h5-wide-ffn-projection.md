# Hypothesis H5: wide FFN input projection on SM120

Created: 2026-08-05 (UTC), before implementation or candidate measurement.
Target regime is RTX 5090D, B13, two independent server streams,
`requireExactNNLen=true`, fixed 19x19, FP16 I/O, unlocked clocks.

## Evidence

The accepted Stage-3 Nsys trace shows that the two FFN input projections are
the largest remaining transformer subgraph. The 2x-wide projection shape
(`M=1152`, `N=4693`, `K=384` at B13) accounts for about 65.9 ms of raw kernel
time across 60 forwards, and the following SwiGLU adds about 18.2 ms. Each FFN
block currently submits `linear1` and `linearGate` as two separate, identical
cuBLAS GEMM shapes over the same normalized input.

The independent SM89 implementation demonstrated that cuBLAS can submit these
two GEMMs as one strided-batched call when the weights and outputs are stored
contiguously. SM120 still needs independent evidence: kernel choice and
concurrent-stream scheduling differ from SM89.

## Mechanism and prediction

Concatenate each block's FP16 `linear1` and `linearGate` weights once, allocate
one contiguous output for the two projections, and replace two
`cublasHgemm` calls with one `cublasHgemmStridedBatched` call with batch count
2 and input stride 0. RMSNorm, both matrix products, SwiGLU, linear2, residual,
and all head code remain unchanged.

Expected profiler signal:

- one batched projection launch replaces two single projection launches per
  FFN block;
- projection plus SwiGLU time does not regress under the real B13/S2 topology;
- whole-network throughput improves by at least 0.5% in an ordered A/B, or the
  candidate is rejected as operationally insignificant.

## Risks and predeclared gates

cuBLAS may choose a slower batched kernel on SM120, or the wider output
allocation may change scratch reuse. Arithmetic should remain in the same FP16
GEMM class, but full-network accuracy is still required because kernel
selection can change reduction order.

The candidate is rejected immediately if a local/profile measurement is
slower. If it is faster, it must pass the same direct-to-full-FP32 8,192-row
gates used in Stage 3:

| metric | gate |
|---|---:|
| policy top-1 agreement | >= 99.70% |
| optimistic-policy top-1 agreement | >= 99.60% |
| policy probability RMSE | <= 1.5e-4 |
| policy total variation | <= 2.5e-3 |
| policy Jensen-Shannon divergence | <= 8e-6 |
| max policy absolute error | <= 0.03 |
| weighted p0loss delta vs reference | <= 5e-4 |
| outcome RMSE | <= 1.5e-2 |
| score-mean RMSE | <= 1.0e-2 |
| ownership sigmoid RMSE | <= 4.0e-4 |

## Validation order

1. Build and run a fixed-19x19 smoke/short benchmark with the hook on and off.
2. Use Nsys under B13/S2 to verify launch replacement and compare the FFN
   projection subgraph.
3. Run the full 8,192-row all-head replay against the full-FP32 reference.
4. Run ordered whole-network A/B with equal thermal preparation and duration.
5. Accept only if profiler, accuracy, and whole-network evidence agree.
