# H43 result: prefetch-preserving FFN A reuse

## Decision

Rejected. The FFN early-linear-prefetch axis is closed for the current fixed
B13 kernel. No KataGo source or target configuration changed.

## Two exact implementations

| implementation | registers | spills | S1 result |
|---|---:|---:|---:|
| accepted immediate A reuse | 136 | 0 | 36.860 us |
| retain two A K-subtile fragments across early prefetch | 151 | 0 | stopped at resource gate |
| issue each half-tile copy after its reused A subtile | 135 | 0 | 38.640 us (`+4.830%` slower) |

The half-tile version preserves copy count and total dynamic barrier count,
is bit-identical over all 5,406,336 boundary FP16 outputs, and even reduces
registers by one. Its large S1 regression proves that moving the barriers/copy
groups earlier harms the realized pipeline. The retain-A version cannot
preserve the accepted resource footprint. No NCU or whole-graph S2 was needed
after these coherent mechanism failures.

Reopen only with a different producer mechanism that issues earlier without
either retaining a second A fragment or moving a block barrier into the K
subtile loop. Last result: `2026-08-06 11:52:44 UTC`.
