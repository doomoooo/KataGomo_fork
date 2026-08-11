# Hypothesis H10b: one-warp RMSNorm with official reduction tree

Created: 2026-08-05 (UTC), after H10 failed one predeclared accuracy gate and
before H10b implementation or measurement.

H10's faster one-warp kernel passed every metric except optimistic-policy
top-1: 99.5972% versus the 99.60% gate, a one-row shortfall in 8,192 rows. H10
combined the six original warp groups inside each lane before its warp reduce,
changing the FP32 addition tree.

H10b keeps one warp per C384 row and four rows per block, but maintains six
separate pair sums. It independently warp-reduces each original 64-channel
group, then places those six group sums in lanes 0..5 and performs the same
second warp reduction as the official kernel. This should preserve the
official sum order while still eliminating shared memory and both block
barriers.

Expected result: replay returns to the accepted both16 numerical class (ideally
byte-identical), while RMSNorm and whole-network performance retain a material
part of H10's gain. The same >=0.5% long performance and Stage-3 accuracy gates
apply.
