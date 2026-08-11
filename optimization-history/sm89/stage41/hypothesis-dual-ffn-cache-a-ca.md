# Stage 41 hypothesis: dual-FFN cache A at all levels

## Scope

- RTX 4090 SM89, locked at 2400 MHz.
- Exact FP16 19x19, batch 13, S2 only.
- Target: accepted CUTLASS dual-GEMM linear1 + linearGate + SwiGLU.

## Pre-implementation evidence

The fresh stage-40 S2 Nsys capture ranks dual-FFN first by summed kernel time:
2178 launches, 43.233 us average, and 94.161 ms total. Accepted-kernel NCU
reports 74.18% L2 throughput, 99.79% L2 hit rate, and 0% L1/TEX hit rate.
Compute throughput is 68.61%, No Eligible is about 78%, and there is no spill.

CUTLASS chooses `CacheOperation::Global` (`cp.async.cg`) for both 128-bit
operands. With swizzle 2, adjacent N tiles share the same A tile. Caching A at
all levels (`cp.async.ca`) may merge or satisfy repeated A requests in L1 and
reduce pressure on a hot, almost entirely hit-resident L2 path. B remains
`CacheOperation::Global` so this experiment changes only A's cache policy.

## Falsification and gates

1. Record three fresh accepted-binary NCU samples from an S2 invocation.
2. Change only the DualMma A cache operation from `Global` to `Always`.
3. Require 26-row S2 replay to be byte-identical.
4. Record three S2 NCU candidate samples. The hypothesis requires a nonzero
   L1 hit signal or reduced L2 pressure and at least 3% lower median duration,
   with no spill or occupancy regression.
5. Only if NCU passes: 20-iteration S2 Nsys in forward and reverse binary order.
6. Only if both Nsys orders pass: locked-clock S2 ABBA with 100 timed iterations.
7. Only after the performance gate: 8192-row accuracy.

Failure restores the pristine pinned CUTLASS checkout. S1 is not an admissible
measurement or fallback.

## Result

The 26-row replay was byte-identical. The cache-policy mechanism produced a
small L1 signal (2.02-2.24%) and reduced reported L2 throughput from about 73%
to about 69.5%, but median duration regressed from 41.28 to 42.62 us (+3.25%)
and No Eligible rose to about 79.1%. The candidate failed the NCU gate and was
reverted without Nsys comparison, ABBA, or 8192-row accuracy.
