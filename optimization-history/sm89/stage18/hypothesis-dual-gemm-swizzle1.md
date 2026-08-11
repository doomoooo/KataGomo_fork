# Stage 18 hypothesis: dual-GEMM swizzle 1 versus accepted swizzle 2

Date: 2026-08-05 UTC

## Frozen target

- RTX 4090 (SM89), exact 19x19, B13, FP16, S2.
- Stage 16 final configuration and binary source state are the control. The
  only compile-time variable is `KATAGO_DUAL_GEMM_SWIZZLE` 2 versus 1.

## Evidence and mechanism

- In the final Stage 16 trace, fused FFN dual-GEMM is the largest individual
  kernel family: `85.831ms` summed over the last 30 complete forwards on both
  streams, averaging `43.349us` under contention.
- Exact production swizzle-2 NCU measures `39.81us`, 666 CTA, 168
  registers/thread, 49.15KiB dynamic shared memory, 15.33% achieved occupancy,
  2.60 waves, zero spill, and 79.27% no-eligible cycles.
- Swizzle 4 was already rejected because it pads 666 logical tiles to 740
  launched CTA. Swizzle 1 and 2 both launch exactly 666 CTA, so this test changes
  only the `(M,N)` tile traversal/raster order. Swizzle 1 may improve concurrent
  input reuse; swizzle 2 may preserve better weight locality.
- Stage 8 measured swizzle 1 only in noisy single-stream micro runs. It never
  received a locked, final-network S2 comparison against swizzle 2.

## Falsifiable test

1. Build swizzle-1 and swizzle-2 executables from the identical current source
   tree, record hashes, and restore the working build to accepted swizzle 2.
2. Prime both paths, lock SM at 2400MHz, then run forward and reverse process
   order with the frozen 500-forward B13/S2 benchmark.
3. Reject if order groups disagree or the paired signal is below drift. Since
   raster order does not change per-tile arithmetic, replay should be byte
   identical, but any accepted result still requires verification.

## Risks

- The expected effect is small and process-state drift can dominate.
- A traversal that helps isolated cache locality may reduce fairness or overlap
  between the two independent server streams.

## Result

- Rejected; swizzle 2 remains the compiled default and accepted configuration.
- Three locked 2400MHz forward/reverse rounds changed by `+1.02%`, `-0.65%`,
  and `-1.17%`. Only 3 of 12 adjacent pairs favored swizzle 1 and their median
  change was `-0.56%`.
- The pooled median changed from `3083.024 nnEval/s` for swizzle 2 to
  `3066.456 nnEval/s` for swizzle 1 (`-0.54%`). Forward and reverse aggregates
  were both negative (`-0.74%` and `-0.11%`), as was actual wall time
  (`-0.41%`).
- This closes the previously unmeasured final-network S2 case. Swizzle 1 and 2
  both launch 666 CTA; the raster-order change does not improve the target.
  Reopen only if the tile shape, stream topology, or cache policy changes.
