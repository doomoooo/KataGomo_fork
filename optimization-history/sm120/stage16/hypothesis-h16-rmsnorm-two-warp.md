# Hypothesis H16: two-warp exact-tree C384 RMSNorm

Created: 2026-08-05 (UTC), after NCU on the accepted one-warp kernel and before
implementation or measurement.

NCU on B13 reports the accepted RMSNorm at 5.28 us, grid 1,174 x block 128,
38 registers/thread, 0.58 waves, 49.08% achieved occupancy, and 80.4% of issue
distance in long-scoreboard stalls. Increasing rows per block would make the
grid smaller. H16 instead assigns two warps to one row: each warp holds three
of the six 64-channel groups, halving per-thread value state and doubling the
number of warps for the same rows.

The accepted arithmetic tree is preserved exactly. Each group publishes the
XOR reduction result from its corresponding lane, warp 0 performs the same
six-group XOR reduction, and all 32 lane-specific scales are shared back to
both warps. This requires two block barriers and small shared storage.

Nsys must reduce direct RMSNorm time. Short A-B-B-A must improve a target
topology before full validation. A retained candidate requires at least 0.5%
long gain and all full-FP32 8,192-row accuracy gates. Unsupported shapes and
precisions retain the accepted one-warp kernel.
