# Stage 8: combined S1 projection optimizations

Status: accepted for S1, rejected for S2 (2026-08-05 UTC).

The candidate enables only the independently accepted single-wide FFN and
strided-batched QKV switches. The short A-B-B-A measured S1
`2662.479 -> 2728.873 nnEval/s` (+2.493%) and S2
`3165.329 -> 3059.653 nnEval/s` (-3.339%).

The combined 8,192-row replay is byte-identical to both individual candidate
replays (SHA256
`db4af085bd372d3ce63f21d93229de3f54a3fd51d8607c0a8a56f92752d8b4ae`)
and passes all Stage-3 full-FP32 gates.

After a 1,500-iteration thermal precondition, the 1,000-iteration symmetric
S1 A-B-B-A/B-A-A-B result was:

| mode | values (nnEval/s) | mean | median |
|---|---|---:|---:|
| control | 2641.263 / 2632.824 / 2632.295 / 2629.773 | 2634.039 | 2632.560 |
| combined | 2705.394 / 2704.763 / 2700.358 / 2695.825 | 2701.585 | 2702.561 |

Mean improvement is 2.564%; median improvement is 2.659%. Accept the combined
switches for B13/S1. Both remain disabled for S2 because the measured combined
regression is 3.34%.
