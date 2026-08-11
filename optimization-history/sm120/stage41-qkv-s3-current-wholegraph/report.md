# H41 result: lower-smem wide-QKV S3 schedule

## Decision

Rejected as no-signal on the current fixed RTX 5090 D, B13, 19x19, natural
two-stream whole graph. No source or target configuration changed.

## Evidence

| Arm | Mean throughput (nnEval/s) |
|---|---:|
| accepted K64/S2 QKV control | 3890.548 |
| M128N128K32/S3 QKV candidate | 3887.944 |

The mean change was `-0.067%`. Adjacent changes were `-0.342%`, `-0.630%`,
`+0.162%`, and `+0.549%`: two positive and two negative. Thus the historical
S1 benefit and lower shared-memory footprint do not yield a resolvable current
whole-graph S2 improvement. Per the frozen rule, no long run, replay, or fresh
profile was collected.

Last result timestamp: `2026-08-06 11:25:44 UTC`.
