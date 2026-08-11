# Stage 47: exact-B13 CuTe fused FFN accepted

Scope: RTX 5090 D, exact 19x19 B13, FP16/NHWC, natural S2.  Control is the
Stage44 accepted packed-QKV graph plus the default-off Stage45 code retention.

## Evidence and hypothesis

The accepted TileLang FFN already fused two C384->C1152 projections with
SwiGLU, but the implementation remained the largest S2 work family.  Historical
plain-GEMM probes showed that replacing an unchanged cuBLAS GEMM with CuTe was
not competitive.  The actionable difference was therefore the boundary: use
one CuTe M4693/N2304/K384 mainloop, pair each 64 linear channels with 64 gate
channels, and apply SwiGLU in the epilogue so the wide intermediate is never
materialized.

TileLang supplied the boundary-local strategy evidence: K32 was materially
better for this FFN shape.  CuTe tile and launch choices were then selected
from the candidate's own NCU evidence rather than copied from packed QKV.

## Falsified variants

| candidate | direct/mechanism result | whole graph result | decision |
|---|---|---|---|
| K64, AB2, epilogue stage 1 | lower epilogue precision | incorrect, RMSE 0.01279 | reject |
| K64, epilogue stages 2/4 | lower error but still above direct boundary tolerance | incorrect | reject |
| K64, epilogue stage 6, grid170 | 107 regs, 91.14 KiB smem, zero spills; direct boundary about 3.16% faster | S2 -8.454% | reject |
| K64, epilogue stage 6, nonpersistent grid666 | direct boundary slower | S1 -2.305%, S2 -9.821% | reject |
| K32 fused SwiGLU, grid170 | 96 regs, 50.18 KiB smem, theoretical 2 CTA/SM but only half a launch wave | S1 -0.871%, S2 -1.753% | reject |
| K32 fused SwiGLU, grid340 | fills two CTA/SM while retaining 96 regs/50.18 KiB | accepted | keep |

The grid170 result is the key NCU-directed correction.  The kernel had enough
register and shared-memory capacity for two resident CTAs, but launching only
170 CTAs left the 170-SM GPU at half a wave.  Doubling the persistent grid to
340 raised achieved occupancy to about 33% and eligible warps/scheduler to
about 0.47 without changing the tile or arithmetic.

## Direct boundary and graph performance

The accepted grid340 candidate directly compares the same complete boundary:

- accepted TileLang fused FFN: median 33.4351 us;
- CuTe paired-projection plus SwiGLU: median 31.1976 us;
- direct improvement: 6.692%.

Initial common-wall ABBA:

- S1: 3176.078 -> 3362.849 nnEval/s, +5.881%; both adjacent pairs positive;
- S2: 3955.620 -> 3991.806 nnEval/s, +0.915%; both adjacent pairs positive.

The longer 400-iteration S2 ABBA measured 3926.900 -> 3944.540 nnEval/s,
+0.449%, with adjacent gains +0.327% and +0.572%.  After productionizing the
generator and making the exact RTX5090D/B13/S2 path automatic, a fresh short
default-on ABBA measured 3920.789 -> 3958.469 nnEval/s, +0.961%; both adjacent
pairs remained positive.

## Correctness

Standalone versus the accepted TileLang boundary had 214 differing FP16 words
over the full output, max absolute error 3.8147e-6, RMSE 6.3825e-9, and no
NaN/Inf.  The full 8,192-row replay passed all predeclared all-head gates:

- versus Stage44: policy top-1 99.7925%, probability RMSE 0.00010243,
  value outcome RMSE 0.0010049, score-all6 RMSE 0.0038694, ownership sigmoid
  RMSE 0.00025147;
- versus full FP32: policy top-1 99.8169%, probability RMSE 0.00010339,
  value outcome RMSE 0.0022950, score-all6 RMSE 0.0074450, ownership sigmoid
  RMSE 0.00024499.

## Final accepted profile

The final production binary was rebuilt from the checked-in generator and a
clean CUTLASS `e05f953a5b3d38adc240df2ff928e0421c2abba3` source.  The generator
pins dense source SHA256
`613052799aff35d5564d49c8bbb4bbac2e22bc58cb3e27499c4c9c3ee95c6e03`.

- final Nsys S1: 3401.631 nnEval/s;
- final Nsys S2: 4050.536 nnEval/s, common-wall 4086.317;
- final NCU: exactly 344 ordinals, IDs 0..343;
- fused FFN: 33 launches/forward, 96 registers/thread, 50.176 KiB dynamic
  shared memory, 1.00 waves/SM, median NCU duration 30.400 us, no known spill
  regression.

Final full-graph ranking still places fused FFN first by S2 work at 24.60%,
but it is now a 96-register/50.2-KiB kernel.  The largest open GEMM family is
linear2+residual: 18.79% of S2 work, 350.1 us/forward interference excess,
162 registers, 65.5 KiB shared memory, 0.65 waves/SM, 8.3% occupancy, and only
9.2% eligible cycles.

Artifacts:

- `fused-swiglu-v2-grid340.json`: direct boundary ABBA and correctness;
- `short-abba/summary.json`, `medium-abba-grid340/summary.json`: graph gates;
- `accuracy-full/`: fixed 8,192-row all-head replay;
- `accepted-aot/`: reproducible AOT metadata and hashes;
- `final-profile/`: final S1/S2 Nsys, 344-ordinal NCU, and full-graph ranking.

## Decision

Accepted and automatic only for the exact RTX5090D, 19x19 B13, configured S2
topology when the generated AOT object is linked.  All other shapes/devices and
stub-only builds retain the prior tactic fallback.  This stage supersedes the
TileLang fused FFN as the B13/S2 winner; TileLang remains the strategy explorer
and measured competitor, not a mandatory runtime backend.

