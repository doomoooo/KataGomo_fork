# Hypothesis H27: aligned vec8 C384 RMSNorm for B13/S2

Created: 2026-08-06 UTC, before implementation. Target is fixed 19x19,
B13/S2, FP16 on RTX 5090D.

The accepted exact-tree C384 RMSNorm issues six separated `half2` loads and
stores per thread. NCU measured 5.28 us under instrumentation, 38 registers per
thread, 0.58 waves, 49.08% achieved occupancy, and long-scoreboard stalls as
the dominant issue-distance source. The 5080 history retained a four-warp,
one-row-per-warp vec8 schedule, but that vector access schedule has not been
reproduced on 5090D.

For each 384-half row, split storage into an aligned 256-half region and an
aligned 128-half region. Each warp lane loads/stores one `uint4` from the first
region and one `uint2` from the second. This reduces six memory instructions to
two vector instructions per input, gamma, beta, and output while retaining one
warp per row and four rows per CTA.

This changes the FP32 addition tree, so it is a numerical candidate rather
than a bit-exact scheduling change. It remains behind a default-off switch.
First require lower isolated/Nsys RMSNorm time and a positive B13/S2 ABBA
signal. Only then run the complete 8,192-row all-head FP32 comparison using the
existing gates; failure of any gate rejects the route regardless of speed.

Risks: wider per-thread live state may increase registers, the changed memory
mapping may worsen coalescing, and S2 overlap may erase an isolated gain.
