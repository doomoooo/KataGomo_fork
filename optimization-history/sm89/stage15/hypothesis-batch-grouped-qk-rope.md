# Stage 15 hypothesis: share fused RoPE math across B13 rows

Date: 2026-08-05 UTC

## Evidence and mechanism

- Frozen target: RTX 4090 SM89, exact 19x19, B13, FP16, S2.
- Stage 13 launches one block for each `(xy,batch)` pair. All 13 blocks at a
  fixed `xy` load the same frequencies and evaluate the same `__sincosf` for
  every head/pair; only their Q/K addresses differ.
- Stage 14 proved that removing repeated trigonometry makes the individual
  kernel faster, but a 36.6MB float2 table harms the whole network.
- A grouped kernel can compute `(cos,sin)` once per thread and loop over several
  batch rows. It preserves the accepted arithmetic exactly and adds no model
  memory. Candidate groups `2,3,4,7,13` trade fewer repeated calculations and
  fewer CTAs against memory-latency hiding; group 1 is the exact stage-13 control.

## Falsifiable test

1. Add runtime `cudaRoPEBatchGroupSm89`, default 1, with AOT template variants
   for groups 2, 3, 4, 7, and 13.
2. Run same-binary B13/S2 screening for every group. Require byte-identical
   replay for the winner.
3. Only the fastest stable group proceeds to three independent 2400MHz-locked,
   thermally primed 500-iteration forward/reverse ABBA rounds.
4. Require positive aggregate, forward, reverse, and adjacent-pair evidence,
   plus Nsys kernel and complete-forward critical-path reduction.

## Result

- Rejected for production; `cudaRoPEBatchGroupSm89` remains at group 1.
- Structural sweep found group 2 was the only convincing kernel winner:
  `8.860us` (group 1), `8.457us` (2), `8.562us` (3), `8.683us` (4),
  `8.821us` (7), and `9.645us` (13). Group 2 reduced the RoPE kernel by
  `4.55%`, but the last-20-forward two-stream Nsys union changed only from
  `177.291436ms` to `177.256590ms` (`-0.0197%`).
- Group 2 remained byte-identical on all 8192 replay rows, SHA256
  `7dde3f6b36e240eb4e92ffc632ecc578d059052fb2c13816b043d0a7093ba484`.
- Three locked rounds did not stabilize: pooled medians changed
  `2963.814260 -> 2969.056319` (`+0.1769%`), but individual rounds were
  `+2.8498%`, `-1.1671%`, and `+0.3124%`. Adjacent-pair median was `+0.2640%`
  with only 8/12 throughput pairs positive; forward was `+0.1769%` and reverse
  `+1.3646%`. These effects exceed and conflict with the measured structural
  upper bound, so they are attributed to process-level state drift rather than
  accepted as a grouped-kernel gain.
- One attempted third round timed out during candidate model startup before any
  benchmark timing. A fresh `r3b` completed; the failed raw startup log is kept.
