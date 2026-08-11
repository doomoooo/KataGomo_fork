# Stage 14 hypothesis: precompute learnable RoPE cos/sin on the final B13 stack

Date: 2026-08-05 UTC

## Evidence and mechanism

- Frozen target: RTX 4090 SM89, exact 19x19, B13, FP16, S2.
- Accepted stage-13 fused RoPE runs 33 times per forward and averages `8.924us`
  in the last-30-forward Nsys window.
- NCU measures one launch at `9.95us`, `74.26%` DRAM throughput (about
  `724.95GB/s`) and `34.82%` SM compute throughput. Intermediate Q/K traffic is
  the dominant cost, so this is intentionally a bounded probe rather than an
  assumption that removing `sincos` will remove the whole kernel cost.
- Learnable frequencies and exact 19x19 coordinates are invariant for the
  model lifetime. A one-time GPU kernel can compute a float2 `(cos,sin)` table
  using the same `__sincosf` expression as the accepted kernel. The steady-state
  kernel then replaces two frequency loads, coordinate arithmetic and `sincos`
  with one float2 table load while preserving Q/K traffic.

## Falsifiable test

1. Add a separate `cudaUsePrecomputedQKRoPESm89` switch, default false. Keep
   the accepted stage-13 path as the exact control in the same binary.
2. Require the precomputed-table replay to be byte-identical to stage 13 before
   any performance claim.
3. Compare locked/primed B13/S2 forward/reverse ABBA. Reject if aggregate
   throughput or either order direction is not positive.
4. If the bounded probe is rejected, move the optimization boundary into the
   QKV GEMM epilogue or FlashAttention load path, which can remove the Q/K
   intermediate read/write instead of only removing arithmetic.

## Result

- Rejected; the implementation remains behind
  `cudaUsePrecomputedQKRoPESm89=false` and is not enabled in the production
  B13/S2 configuration.
- Correctness passed strongly: the full 8192-row replay is byte-identical to
  stage 13 with SHA256
  `7dde3f6b36e240eb4e92ffc632ecc578d059052fb2c13816b043d0a7093ba484`.
- The isolated kernel improved only from `8.773us` to `8.459us` (`3.58%`),
  implying roughly `10.4us` savings over 33 launches, about `0.12%` of one
  forward. One-time table construction cost was `0.234ms` over 132 captured
  setup launches and is outside steady-state timing.
- Pooled 12+12 medians appeared positive (`2962.647 -> 3003.906`, `+1.393%`),
  but adjacent-pair analysis exposed order drift: only 6/12 throughput pairs
  were positive and the paired median was just `+0.381%`. Forward aggregate
  was `+0.027%` while reverse aggregate was `+1.480%`.
- Nsys contradicted a whole-network gain: the last 30 complete two-stream
  forward groups increased from `259.534586ms` to `265.927779ms` (`+2.463%`).
  The two server models add about `36.6MB` of float2 tables, which can disturb
  L2 residency and concurrent scheduling despite the individual kernel saving.
