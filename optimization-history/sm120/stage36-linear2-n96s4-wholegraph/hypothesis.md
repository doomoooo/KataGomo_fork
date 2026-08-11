# H36: direct whole-graph test of linear2 M128N96/S4

## Observation

The fresh post-A-reuse full-graph profile ranks transformer linear2 residual as
the largest S2 interference source: `1058.1 us` of work and `370.3 us` of
excess per stream-forward (`1.538x` S2/S1). The current M128N128/S4 kernel uses
111 CTAs, 162 registers/thread, and 65.54 KiB dynamic shared memory.

Stage32 produced a bit-exact M128N96/S4 schedule with 148 CTAs, the same 162
registers/thread, 57.34 KiB dynamic shared memory, and a roughly 21% isolated
S1 improvement. It was rejected only through local homogeneous/mixed S2 proxy
gates that are no longer valid. Those historical measurements will not be
rerun and have no authority here.

## Change

Integrate the exact Stage32 generated source as a separate AOT kernel behind
`cudaUseLinear2ResidualAotN96S4Sm120`, default false. Gate it inside the
existing FP16, mask-free, exact B13, 19x19, C1152-to-C384 residual path. The
current M128N128/S4 kernel remains the complete fallback.

This changes one schedule only: M tile 128, N tile 128 -> 96. Arithmetic and
the FP16 residual epilogue are unchanged.

## Prediction and decision

The larger grid should reduce the current linear2 critical-path contribution
enough to improve the real B13/19x19/S2 graph. After build and smoke:

1. Confirm the checked-in candidate remains bit-exact and spill-free with its
   Stage32 resource signature.
2. Run short real whole-graph S2 in symmetric `A-B-B-A-B-A-A-B` order.
3. Advance to 1000/30 long ABBA/BAAB when the candidate mean is positive and
   at least three of four adjacent comparisons are positive.
4. Reject only on real whole-graph regression/no signal or correctness
   failure. Do not run homogeneous or synthetic mixed S2.
5. If accepted, run full 8,192-row replay and refresh the complete S2 Nsys plus
   344-ordinal S1 NCU before selecting the next target.
