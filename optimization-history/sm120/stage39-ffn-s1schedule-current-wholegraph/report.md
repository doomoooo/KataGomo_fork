# H39a result: existing three-stage FFN schedule

## Decision

Rejected for the current fixed RTX 5090 D, B13, 19x19, two-stream whole graph.
No source or target configuration changed.

## Real whole-graph screen

| Arm | Mean throughput (nnEval/s) |
|---|---:|
| accepted two-stage A-reuse control | 3900.067 |
| existing three-stage S1 schedule | 3754.695 |

The candidate regressed the mean by `3.727%`. All four adjacent comparisons
were negative: `-3.628%`, `-3.473%`, `-4.521%`, and `-3.282%`.

The old S1 gain therefore does not transfer to the real current S2 graph. The
extra pipeline stage raises dynamic shared memory from 32,768 to 49,152 bytes
per CTA and is consistent with reduced scheduling flexibility. Per the frozen
stop rule, no long run, accuracy replay, or candidate profiler collection was
performed.

Raw results are in `short/`; the last artifact was saved at
`2026-08-06 11:07:56 UTC`.
