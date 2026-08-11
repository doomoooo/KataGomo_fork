# H26: A work-equivalent M256xN64 linear2 tile may improve S2 coexistence

## Evidence

- The fixed-B13 S2 interference report matches every kernel by forward ordinal.
  Linear2 runs at `1.62x` its isolated duration and contributes 17.1 ms of
  summed excess duration over the 20-iteration/two-stream trace.
- NCU on the accepted M128xN128xK32/S4/T128 tile reports 65.54 KiB dynamic
  shared memory, 162 registers/thread, one shared-memory-limited CTA/SM,
  8.23% achieved occupancy, 0.65 waves/SM, and 91.20% no-eligible cycles.
- The old M128xN64/S3 tile launches 222 CTAs and won an isolated dual-linear2
  microbenchmark, but regressed whole S2 throughput. Its larger grid occupies
  the card more completely and removes room for peer stages.

## Mechanism

M256xN64 computes the same 16,384 output elements per CTA as the accepted
M128xN128 tile. At fixed M=4693,N=384 it launches 6x19=114 CTAs, nearly the
same 111-CTA footprint, but 256 threads split the accumulator work across
twice as many warps. Its shared input/weight footprint is balanced differently
and can reduce dependency stalls or registers without returning to the harmful
222-CTA grid.

## Falsifiable prediction

- At least one M256xN64 pipeline variant improves isolated and dual micro time
  or materially lowers registers/no-eligible stalls while staying near the
  111-CTA launch footprint.
- Only the best credible micro/resource candidate is integrated.
- It must improve B13/S2 whole-network throughput by at least 0.5% in an
  ordered A/B and remain positive at both natural and aligned phase regimes.

## Search and gates

Search K32 stages 2/3/4 and K64 stages 2/3 with 256 threads. This is one tile
family chosen from profiler evidence, not a broad combinatorial sweep. Reject
compile failures and candidates slower than the accepted tile in both single
and dual microbenchmarks. An integrated arithmetic candidate must pass the
8,192-row all-head FP32-reference comparison.
