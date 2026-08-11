# Stage 35 hypothesis: phase-moderated C384 affine + SiLU vec4

## Frozen target

- RTX 4090 (SM89), exact 19x19, B13, FP16, S2 only.
- Accepted post-Stage-34 source is the control; S1 is out of scope.
- Candidate is default-off behind `cudaUseScaleBiasSiluVec4C384Sm89`.

## Evidence and mechanism

Stage 34 proved that C384 vectorization is locally valid: vec8 changed NCU median
from 6.85 to 4.67 us, remained byte-identical, and reduced summed full-trace kernel
time by 0.99-1.46%. It nevertheless increased S2 GPU busy union by 1.39-2.91%
and reduced throughput in both run orders. The rejected vec8 launch reduced the
kernel from 4,693 CTAs / 9.17 waves to 880 CTAs / 1.15 waves.

This stage tests whether that S2 regression is sensitive to the magnitude and
resource shape of the local change. Each thread processes four contiguous half
elements instead of eight. The exact launch is 1,760 CTAs x 256 threads, about
2.29 waves/SM. It retains more parallel work and a longer phase contribution than
vec8 while still removing 75% of the scalar threads and indexing operations.

Arithmetic remains element-independent and identical to control: half `__hfma`,
float `expf` SiLU, one final half rounding. Input, parameter, and output accesses
are aligned 8-byte vectors.

## Falsifiable gates

1. The 26-row S2 replay must be byte-identical to control.
2. Three-sample S2 NCU must have no spills and median duration below 6.0 us.
3. Twenty-iteration S2 Nsys forward and reverse runs must both reduce the C384
   target and improve end-to-end throughput. Summed kernel time and GPU busy union
   are reported to distinguish useful work reduction from overlap regression.
4. Only if all short gates pass: one locked 100-iteration S2 ABBA and the full
   8192-row FP32-reference replay.

If either full-network order is non-positive, reject and revert. This is a narrow
S2 scheduling experiment, not a reason to restore S1 as an optimization target.

## Result

- Smoke was byte-identical across all 26 rows.
- Candidate NCU median was 4.32 us versus the 6.85 us accepted-source control,
  with 23 registers/thread, 2.29 waves/SM, and no spills.
- Both short S2 orders passed: throughput improved by 0.865% and 1.623%; GPU
  busy union improved by 0.335% and 1.425%.
- The locked 100-iteration ABBA failed: control median 3250.984 versus candidate
  3224.363 nnEval/s (-0.819%), with one positive and one negative adjacent pair.

Decision: reject and revert. The local mechanism is valid, but its S2 benefit is
not stable without explicit phase control. Reopen after the common dual-stream
phase-controller interface is implemented; do not reopen by switching to S1.
