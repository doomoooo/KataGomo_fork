# Stage 12 hypothesis: fixed-B13 nested postConv residual GEMM

Date: 2026-08-05 UTC

## Evidence and mechanism

- Frozen target: RTX 4090 SM89, exact 19x19, B13, FP16, S2.
- Each of the 11 nested blocks ends with an accumulating NHWC 1x1 postConv.
  The exact row-major boundary is `M=4693,N=768,K=384`, `alpha=1,beta=1`.
- The accepted stage-11 complete Nsys capture contains exactly 1166 calls
  (`106 * 11`) to cuBLAS `ampere_h1688gemm_256x64_ldg8_stages_32x1_nn`, with
  `grid=(3,74,1)`, average `25.212us`, and summed time `29.397ms`.
- The 222 logical output tiles are close to two waves on 128 SMs. A fixed
  CUTLASS tile can test whether a wider N tile, a different warp decomposition,
  or deeper staging improves the actual S2 boundary.

## Falsifiable test

1. Sweep exact beta-one CUTLASS threadblock/warp/stage combinations, including
   128- and 256-wide N tiles where shared-memory limits permit.
2. Compare all 3,604,224 output half elements against cuBLAS for every binary.
3. Select only from 10,000-iteration dual-stream forward/reverse ABBA, requiring
   at least 3% isolated improvement.
4. Integrate behind strict B13/S361/384-to-768/accumulate guards. Require
   positive whole-network forward/reverse ABBA, exact Nsys call replacement,
   no NCU spill regression, and full 8192-row FP32 accuracy acceptance.

## Outcome

- Swept 25 fixed-shape combinations spanning threadblocks 64/128/256 in M and
  64/128/256 in N, four/eight/sixteen warps, stages 2-6, and swizzles 1/2/4.
  The 256x256 cases were explicitly compiled and executed but CUTLASS rejected
  them at `can_implement`; they were not skipped due to toolchain version.
- All implementable candidates were bit-exact on 3,604,224 fixed-random half
  outputs. The isolated winner was M128xN128xK32, warp M64xN64xK32, stage3,
  swizzle1: 10,000-iteration S2 ABBA pair median `26.775851us -> 23.138238us`
  (`+15.721219%` throughput ratio).
- Whole-network performance rejected the change. Three independent,
  thermally primed, 2400MHz-locked 500-iteration forward/reverse ABBA rounds
  changed by `-0.126022%`, `-0.162781%`, and `-0.088559%`. Across 12 samples
  per path the median was `2905.853920 -> 2900.915541 nnEval/s` (`-0.169946%`);
  forward, reverse, and actual-wall groups were all negative.
- Decision: rejected and left disabled. NCU and full accuracy were not run
  because the pre-registered whole-network performance gate failed; isolated
  exact-boundary correctness was bit-exact.
