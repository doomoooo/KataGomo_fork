# H37: reopen fused Q/K RoPE half2 I/O

## Observation

The accepted-state Stage35 profile assigns fused Q/K RoPE `218.9 us` of work
and `83.1 us` of S2 excess per stream-forward (`1.612x` S2/S1). Stage15's
half2 implementation reduced direct RoPE kernel time by `7.60%` and improved
the then-current real whole graph by `+0.301%` in S1 and `+0.443%` in S2, but
was stopped solely because it missed the old `0.5%` promotion threshold.

That threshold is no longer valid. The implementation already exists behind
`cudaUseFusedQKRoPEHalf2Sm120`, default false, with the scalar kernel as exact
fallback.

## Prediction and decision

On the current post-A-reuse mainline, half2 should produce a small positive
real-graph S2 signal. Run only the symmetric whole-graph
`A-B-B-A-B-A-A-B` screen at B13/19x19/S2. Advance to 1000/30 long
confirmation when the mean is positive and at least three of four adjacent
comparisons are positive. Reject on a coherent real-graph regression or no
signal. Do not run homogeneous or synthetic mixed S2.

If accepted, run the complete 8,192-row replay and refresh S2 Nsys plus the
344-ordinal S1 NCU before selecting another target.
