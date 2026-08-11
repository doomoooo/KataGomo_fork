# Stage 34 hypothesis: exact C384 affine + SiLU vec8

## Frozen target

- RTX 4090 (SM89), exact 19x19, B13, FP16, S2 only.
- The accepted post-Stage-33 source is the control.
- The candidate changes only a new default-off
  `cudaUseScaleBiasSiluVec8C384Sm89` switch. The accepted C768 vec8 path remains
  enabled and identical in both arms.

## Pre-implementation Nsys/NCU evidence

The accepted S2 Nsys trace contains 726 C384 `applyCScaleBiasNHWCSiluHalfKernel`
calls across 66 forwards, exactly 11 calls per forward, averaging 8.030 us and
5.83 ms of summed trace time.

Three locked-2400 NCU samples measure 6.85, 6.85, and 6.91 us. The scalar launch
uses `(1,361,13)` CTAs with 384 threads, or 4,693 CTAs and 1,802,112 launched
threads per invocation. It has 16 registers/thread, no spill, 100% theoretical
but 63.7-64.8% achieved occupancy, 9.17 waves/SM, about 35% compute and memory
throughput, 54.8-55.0% no-eligible cycles, and about 99% L2 hit rate.

This rejects a DRAM-bandwidth or register-pressure theory. The likely cost is
excess scalar thread/CTA scheduling and repeated indexing for a small elementwise
kernel, the same mechanism validated for C768 in Stage 22.

The FlashAttention audit performed before selecting this target is retained in
`ncu-flash-s2-control`: its 27.97-28.22 us kernel already reaches three CTA/SM,
25% theoretical occupancy, zero spill, and the previously swept winning tile.
It did not expose a new independent optimization variable.

## Candidate and mechanism

Add a separate exact C384 kernel. One thread owns eight adjacent half elements,
using one aligned 16-byte input/scale/bias load and one aligned 16-byte store.
The eight elements retain the accepted arithmetic independently:

1. half `__hfma(input, scale, bias)`;
2. convert the rounded half result to float;
3. scalar float `expf` SiLU;
4. round once to half.

For `(N,XY,C)=(13,361,384)`, the launch falls to 880 CTAs and 225,280 launched
threads. The expected benefit is less CTA/thread indexing overhead plus eight
independent SiLU chains per thread. Reduced parallelism and 31-ish registers are
the explicit risks.

## Falsifiable gates

1. A 26-row S2 smoke must be finite and byte-identical to control.
2. Three-sample S2 NCU must show 880 CTAs, no spills, and median duration at most
   5.0 us (at least 27% faster than 6.85 us).
3. Twenty-iteration S2 Nsys forward and reverse order runs must both reduce the
   C384 aggregate and improve end-to-end throughput. Any order reversal rejects
   the candidate.
4. Only if all short gates pass: one locked 100-iteration S2 ABBA and the full
   8192-row FP32-reference replay.

Failure of the single-kernel threshold or either full-network order gate causes
immediate revert and skips the long tests.
