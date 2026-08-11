# Stage 36 hypothesis: QKV B-tile copy map 16x2

## Frozen target

- RTX 4090 (SM89), exact 19x19, B13, FP16, S2 only.
- Accepted post-Stage-35 source is the control; S1 is out of scope.
- Candidate is default-off behind `cudaUseQKVCopyMap16x2Sm89`.

## Evidence

The current full S2 Nsys trace ranks fused QKV+RoPE among the three largest
kernels: 2,178 calls at 29.602 us average and 64.472 ms summed trace time.

Current 3-sample S2 NCU is 24.70/29.63/30.24 us (29.63 us median). The launch
uses 240 registers/thread, 49.15 KiB shared memory, 1.30 waves/SM, 16.67%
theoretical occupancy, about 15% achieved occupancy, no spills, and roughly
79.3% no-eligible cycles. Stage 16 NCU additionally reported poor global-load
sector utilization and about 2.1-way shared-store bank conflicts.

The SM120 optimization history contains an untried-on-SM89 analogue: changing
the wide-QKV copy atom from 2x2 to 4x2 reduced its kernel from 39.325 to 37.717
us and improved the full graph by 0.373%.

## Mechanism and single variable

Keep the fixed M128xN128xK32, warp M64xN64xK32, stage-3 GEMM, 128-bit access,
RoPE epilogue, grid, and arithmetic unchanged. Change only operand B's CUTLASS
warp-raked copy map:

- control: warp arrangement 8 contiguous x 4 strided, per-thread iterations 2x2;
- candidate: 16 contiguous x 2 strided, per-thread iterations 1x4.

Both maps issue four 128-bit copies per thread per stage. The candidate covers
the full N128 row across a warp before advancing K, which may improve sector
coalescing and change the shared-store bank mapping without increasing work or
changing numerical order.

## Falsifiable gates

1. Build and 26-row replay must be byte-identical to control.
2. Three-sample S2 NCU must retain the same grid/resource class, have no spills,
   and reduce median QKV duration by at least 2% (below 29.04 us).
3. Only if the NCU gate passes, run 20-iteration S2 Nsys in forward and reverse
   order. Both must reduce QKV and improve end-to-end throughput; report summed
   kernel time and GPU busy union.
4. Only if all short gates pass, run one locked 100-iteration S2 ABBA, followed
   by the full 8192-row FP32-reference replay.

Any failed gate rejects and reverts the candidate. No S1 result is used.

## Result

Rejected and fully reverted before candidate performance profiling. The
candidate built successfully, but the 26-row S2 replay failed catastrophically
rather than showing ordinary FP16 drift: policy top-1 agreement was 0/26,
policy probability RMSE was 0.0299084, total variation was 0.915982, and value
raw-logit RMSE was 7.02379. The candidate and control replay SHA-256 values were
`fee03dd84fd1ebbbb51c0242a4a6e1b5b9eb5499aa0c2422a4b730e6d8b9c1bd`
and `00f4110d9b2770fc942f8ac88e1ecea6e17519dc99994bc6ca2f73233396be97`.

Static inspection after the failure found the missing contract. CUTLASS
`DefaultMmaCore` limits the row-major B contiguous warp arrangement to
`min(ShapeN / elementsPerAccess, 8)`. Its
`RowMajorTensorOpMultiplicandCongruous` shared-memory iterator is coupled to
that lane map. Substituting 16x2 compiles because the alias has no matching
static assertion, but the shared-memory producer and tensor-op consumer no
longer agree on lane ownership.

Therefore the SM120 CuTe atom4x2 result cannot be transferred to this CUTLASS
2.x mainloop as a one-line thread-map alias. This does not close the underlying
strategy: it may be reopened only as a faithful CuTe/custom mainloop whose
global tiled copy, shared layout, and MMA consumer mapping are designed and
validated together. No candidate NCU, Nsys, ABBA, or 8192-row replay was run,
because the correctness gate failed first. The option and implementation were
removed, the accepted source rebuilt, and the post-revert binary SHA-256 was
`8d12ab9266ce20ada665a8f83ce3fb619f1808e3115325669b8b955ceb4337c1`.
