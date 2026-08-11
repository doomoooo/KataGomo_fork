# Stage 40 hypothesis: CUTLASS dual-FFN stage 2

## Scope

- GPU: RTX 4090 (SM89), locked at 2400 MHz.
- Shape: exact FP16 19x19, batch 13.
- Execution topology: S2 only. No S1 measurements are admissible.
- Target: the fused CUTLASS DualGemm linear1 + linearGate + SwiGLU kernel.

## Pre-implementation evidence

The accepted CUTLASS stage-3 kernel was sampled three times by NCU from an S2
benchmark invocation in stage 33:

- duration: 41.02, 41.47, 41.34 us (median 41.34 us);
- dynamic shared memory: 49,152 bytes per CTA;
- resident CTAs: 2 per SM;
- theoretical occupancy: 16.67%;
- achieved occupancy: 15.30-15.33%;
- No Eligible: 78.23-78.51%;
- registers: 168 per thread, with no local/shared spill.

Recent accepted-path S2 Nsys captures place this kernel at roughly 43 us and
make it the largest repeated kernel family in the full graph. Stage 40 records a
fresh accepted-binary 20-iteration S2 capture before editing as a topology and
hotspot check.

## Mechanism

Change only CUTLASS `Stages` from 3 to 2 while retaining the exact tile,
warp shape, instruction shape, swizzle, layouts, epilogues, and arithmetic.
Expected dynamic shared memory is about 32 KiB, permitting 3 resident CTAs per
SM and raising theoretical occupancy from 16.67% to 25%.

This differs from stage 33: stage 33 replaced CUTLASS with a TileLang kernel and
therefore changed code generation and the dependency schedule. The exact
CUTLASS stage-2 instantiation has not been measured.

## Falsification and gates

1. Correctness smoke: 26 rows, accepted frozen binary versus candidate.
2. S2 NCU: three target launches. The mechanism requires about 32 KiB shared
   memory, 3 resident CTAs/SM, no spill, and at least 3% lower median duration.
3. Only if gate 2 passes: S2 Nsys, 20 timed iterations in forward and reverse
   binary order. Both orders must improve target-kernel time and full-graph
   throughput without a contradictory GPU-busy-union result.
4. Only if gate 3 passes: one shortened locked-clock S2 ABBA run with 100 timed
   iterations per process.
5. Only after the performance gate: 8192-row accuracy comparison.

Failure at a gate stops the experiment, restores CUTLASS stage 3, rebuilds the
accepted binary, and records the rejection. No S1 fallback is allowed.

## Result

The bundled CUTLASS wrapper initially rejected stage 2. A minimal experimental
patch reused the stage-3 `DefaultMma` descriptor types while passing the real
stage count to `DualMmaMultistage`; this compiled and produced byte-identical
smoke output. NCU confirmed the mechanism but falsified the performance claim:
shared memory fell to 32.77 KiB and occupancy rose to about 21.9%, while median
duration regressed from 41.34 to 42.27 us (+2.25%) and No Eligible remained near
78.8%. The candidate and third-party compatibility patch were reverted without
running Nsys comparison, ABBA, or 8192-row accuracy.
