# Stage 51 hypothesis: eliminate exact-19 mask preprocessing

## Frozen target and evidence

- Control: commit `acf588c`, RTX 5090 D, exact 19x19 B13, FP16/NHWC,
  natural S2.
- The accepted Stage 47 Nsys/NCU profile still contains one mask channel
  extraction, one half-to-float copy, and one channel-sum reduction per
  forward.  Together they are roughly 5 us/forward and three launches.
- `requireExactNNLen` already sets both downstream mask pointers to null.  Only
  `maskSum` survives for head pooling normalization.
- On a full 19x19 board, all 361 mask values are one and the required FP32 sum
  is exactly representable as `361.0f`.
- RTX 4090 Stage 50 removed the identical boundary, produced byte-identical
  8,192-row output, and improved locked natural S2 by 1.061%.  That throughput
  result is treated as phase-amplified mechanism evidence, not a portable
  prediction; the expected SM120 range is 0.05%--0.30%.

## Single mechanism

For only RTX 5090 D, max/runtime batch 13, exact 19x19, FP16/NHWC, allocate a
13-element persistent device vector containing `361.0f` once per compute
handle.  In the official model forward, use it as `maskSum` and skip all three
mask preprocessing launches.  Preserve the existing scratch allocation order
and every downstream pointer/operation.  All other shapes, batches, precision,
layouts, devices, and disabled-option runs execute the complete official path.

## Gates

1. Build and run B13 candidate plus B12 fallback smoke; the specialization log
   must appear only for B13.
2. Compare candidate and control replay output before performance testing;
   target is byte identity.
3. Use a short natural Nsys trace to verify exactly three launches per forward
   disappear, with no replacement kernel or synchronization.
4. Run natural-S2 ABBA plus reverse BAAB.  Promote only with non-negative order
   aggregates and positive pooled mean; confirm with 400 iterations.
5. Acceptance requires full 8,192-row all-head accuracy, a fresh S2 Nsys and
   complete 344-to-341 ordinal S1 NCU profile, history update, default-on exact
   dispatch, and one commit.
