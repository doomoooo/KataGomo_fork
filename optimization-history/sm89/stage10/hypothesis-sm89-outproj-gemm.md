# Stage 10 hypothesis: fixed-B13 attention out-projection GEMM

Date: 2026-08-05 UTC

## Evidence and mechanism

- Frozen target: RTX 4090 SM89, exact 19x19, B13, FP16, S2.
- The accepted stage9 Nsys capture leaves 77 `grid(3,37)` cuBLAS kernels per
  forward. Grouping by `gridZ` and call sequence identifies 33 wide-QKV
  launches (`gridZ=3`), 33 attention out-projection launches, and 11
  bottleneck projections (`gridZ=1`).
- This experiment changes only the 33 attention out-projections. Their real
  boundary is row-major `D = A*B + C`, `M=4693,N=384,K=384`, FP16
  `alpha=1,beta=1`.
- Like linear2, cuBLAS launches 111 M128xN128 CTAs on 128 SMs. K is three
  times smaller, so the best pipeline depth and warp shape may differ from
  stage9; the S2 winner must be selected by a direct dual-stream benchmark.

## Falsifiable test

1. Sweep fixed CUTLASS M/N tiles, warp shapes, and stage counts on the exact
   `beta=1` call boundary. Require correctness against cuBLAS.
2. Select by 10,000-iteration dual-stream ABBA, not by single-stream latency.
3. Integrate behind a strict B13/S361/C384 shape guard only if the winner is
   at least 3% faster at the isolated S2 boundary.
4. Require positive forward and reverse whole-network ABBA, Nsys proof that
   exactly 33 calls per forward changed, exact NCU resource evidence, and a
   full 8192-row all-head comparison against FP32.

## Result

- Isolated winner: threadblock `M128xN128xK32`, warp `M64xN64xK32`, four
  stages, swizzle 1. Exact random inputs were bit-exact against cuBLAS.
- The 10,000-iteration dual-stream ABBA pair median improved from about
  15.24us to 14.23us (+7.1%). Smaller tiles lost; stage2/3 and the 8-warp
  variant were weaker.
- Exact NCU: cuBLAS 13.47us, 186 registers/thread, no dynamic shared memory;
  CUTLASS 14.05us, 162 registers/thread, 64KiB dynamic shared memory. Both
  launch 111 CTAs and have zero spill. The isolated benefit is specifically
  an S2 concurrency effect, not a single-stream NCU latency win.
- Unlocked 300-forward ABBA conflicted: forward +1.079%, reverse -1.054%,
  combined +0.053%.
- Three independent 2400MHz, thermally primed, 500-forward ABBA rounds also
  failed to stabilize. Their combined changes were +0.598%, +0.404%, and
  -0.548%. Across all 12 control and 12 candidate samples the median was
  +0.523%, but grouped forward order was -0.527% while reverse order was
  +0.593%.
- Decision: rejected for the fixed B13/S2 target because the required
  forward/reverse sign agreement failed. The runtime switch remains default
  false. Reopen only with a fused upstream/downstream boundary or a protocol
  that explains and removes the two persistent whole-network performance
  plateaus.
