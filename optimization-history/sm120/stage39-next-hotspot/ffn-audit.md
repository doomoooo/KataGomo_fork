# Stage 39 FFN read-only audit

## Scope and decision basis

- Target is frozen to RTX 5090 D, B13, 19x19, FP16/NHWC and the natural
  whole-graph S2 topology.
- Homogeneous S2 and synthetic mixed-pair S2 are permanently excluded. After
  an S1 + NCU screen, a surviving candidate goes directly to the real
  whole-graph S2 A/B.
- This audit is read-only. It does not authorize another tile/stage/thread
  sweep, a batch search, or a multi-change bundle.

Stage38 makes fused FFN the largest current work bucket: `1479.4 us` per
stream-forward (`25.94%`), with `203.5 us` interference excess and an S2/S1
ratio of `1.158x`. It is therefore still a valid high-value target even though
linear2/out-projection have larger *excess* individually.

The full-forward NCU replay contains all 33 FFN ordinals. Their median signature
is:

| metric | median / fixed value |
|---|---:|
| NCU replay duration (explanatory only) | `39.136 us` |
| registers/thread | `136` |
| allocated shared memory/CTA | `33.792 KiB` (`32 KiB` dynamic) |
| waves/SM | `1.31` |
| achieved occupancy | `19.68%` |
| eligible warps/cycle | `0.299` (`23.1%` in the joined summary) |
| tensor active, elapsed | `43.38%` |
| issue active | `23.11%` |
| wait / issue | `3.188` |
| math-pipe throttle / issue | `2.169` |
| long scoreboard / issue | `1.019` |
| barrier / issue | `0.511` |
| L2 sector hit rate | approximately `100%` |

This is not a DRAM-bound signature. The remaining plausible local wins are
instruction/dependency reductions which do not enlarge the 136-register,
32-KiB resource footprint. Stage33 already validated that pattern: reusing A
fragments cut S1 `37.680 -> 34.921 us` (`-7.321%`), reduced registers
`146 -> 136`, and improved the long real graph `3824.934 -> 3859.725 nn/s`
(`+0.910%`).

## Candidate 1 (execute first): remove the second main-loop block barrier

### Evidence -> mechanism

The current A-reuse source computes both linear and gate MMA from the same A
fragments, then synchronizes at line 101. It next writes only
`linear_shared[0,8192)` and commits that cp.async group, synchronizes again at
line 107, and only then writes the disjoint `input_shared[8192,24576)` and
`gate_shared[24576,32768)` regions.

PTX confirms two `bar.sync 0` operations in the repeated body: one immediately
before the next-linear cp.async group and another immediately after its
`cp.async.commit_group`. The SM120 SASS has the same pair at `0x15e0` and
`0x1700`, with the loop backedge at `0x17b0`. Thus the latter executes once for
each of the 10 repeated K iterations.

The first barrier is required: it proves all warps have finished reading the
current input/linear/gate stages before any stage is overwritten. The second is
not a visibility or alias barrier:

- all current input and gate reads were already completed before the first
  barrier;
- the intervening async writes touch only `linear_shared`;
- subsequent async writes touch only the disjoint input/gate regions; and
- `cp.async.commit_group` does not require a block barrier before another
  per-thread commit group.

The single variable is therefore deletion of only the line-107 barrier. Keep
both commit groups, `wait_group<2>`, all other barriers, grid, swizzle and math
unchanged.

### Falsifiable prediction

- Output must remain bit-identical for all `1,802,112` boundary FP16 values.
- PTX/SASS must lose exactly the second repeated-body barrier; cp.async group
  count/order must remain unchanged.
- Registers must remain `<=136`, shared memory `33.792 KiB`, with zero spills.
- S1 should fall by roughly `0.3-1.5%`; barrier/issue should fall from the
  current `0.511`, without worsening wait/issue or tensor-active cycles.
- A `1%` FFN reduction has an ideal work-share ceiling of about `0.26%` on the
  graph. A realistic whole-graph expectation is `+0.1-0.4%`.

### Risk

The main risk is a hidden cp.async ordering assumption in the generated
template. It is directly falsified by deterministic boundary comparison and
Compute Sanitizer; no numerical tolerance is needed.

## Candidate 2 (largest raw upside): fast FP32 quotient only in SwiGLU

### Evidence -> mechanism

The epilogue at source lines 186-221 produces 64 values/thread. For every value
it evaluates an approximate FP16 exponential, converts to FP32, and performs a
precise FP32 `x / (1 + exp(-x))` before multiplying by the gate and rounding to
FP16.

The SM120 PTX contains exactly `64` `div.rn.f32` operations. Their SASS expands
to `MUFU.RCP`, `FCHK`, a branch/slow-path scaffold, and several refinement
FFMAs per quotient. Static SASS for the kernel contains `64 FCHK`, `64 CALL`,
`396 FFMA`, and `134 MUFU`; inspection of a representative quotient shows five
refinement FFMAs after the reciprocal. NCU correspondingly reports substantial
non-tensor execution (`10.19%` FMA-heavy elapsed activity) and
math-pipe-throttle/issue `2.169`, while tensor elapsed activity is only
`43.38%`.

The single variable is to replace only those 64 precise quotients with the
CUDA fast FP32 division/reciprocal equivalent, retaining the existing `hexp`,
FP32 gate multiply, FP16 MMA order and FP16 final rounding. Do **not** enable
global `--use_fast_math`, switch to tanh-SiLU, or alter the exponential in the
same experiment.

### Falsifiable prediction

- PTX must replace `64 div.rn.f32` with fast reciprocal/divide operations;
  SASS must eliminate the per-quotient FCHK/slow-path and most refinement
  FFMAs.
- Dynamic instructions should fall by at least `5%`; math-pipe-throttle/issue
  and wait/issue should fall. Registers must not exceed `136`, shared memory
  and launch geometry must be unchanged, and there must be no spills.
- Expected S1 reduction is roughly `3-8%`. The ideal graph ceiling is
  `0.78-2.08%`; a realistic natural-S2 expectation is about `+0.5-1.2%`.

### Risk

This is the only numerical candidate in the shortlist. Fast quotient error can
cross a final FP16 rounding boundary, so bit identity is not an acceptance
requirement. It must first show no NaN/Inf and bounded boundary error, then pass
the established 8192-row all-head comparison directly against the frozen FP32
reference. Any accuracy-gate failure rejects it; do not compensate by changing
the exponential or thresholds.

## Candidate 3: fixed-grid specialization of the existing swizzle

### Evidence -> mechanism

Source line 29 calls the fully generic `rasterization2DRow<10>()`, although the
launch is permanently `grid(18,37,1)`. The SM120 PTX still contains five
runtime `div.u32` operations for this helper. SASS addresses approximately
`0x0000-0x0400` implement the generic grid/panel arithmetic through runtime
integer-to-float conversion, reciprocal, multiply-high, predicates and
branches before useful tile setup begins.

The single variable is a fixed `18 x 37`, panel-width-10 helper which computes
the **identical** logical `(x,y)` permutation using compile-time divisors
(`180`, `10`, and tail stride `7`). Do not replace the permutation with raw
`blockIdx`; that would also change L2 traversal and would be a different
hypothesis.

### Falsifiable prediction

- Exhaustively compare all 666 raw CTA coordinates: candidate logical
  coordinates must exactly equal `rasterization2DRow<10>()` and form a
  permutation of `[0,18) x [0,37)`.
- PTX runtime `div.u32` count in the swizzle should become zero; the prologue
  should shrink materially. Grid, registers (`<=136`), shared memory, spills
  and all tensor/memory instructions after address setup must be unchanged.
- Expected S1 reduction is `0.2-0.8%`, dynamic instructions roughly
  `0.5-1.2%`, and whole-graph gain roughly `+0.05-0.2%`.

### Risk

An off-by-one in the final seven-row panel silently permutes or drops tiles.
The exhaustive coordinate test plus bit-identical full boundary output is a
hard gate. The other risk is ptxas failing to reduce the fixed arithmetic; in
that case NCU/SASS falsifies the mechanism and the candidate should be removed
without an S2 run.

## Minimal protocol for each candidate

Run candidates sequentially, never as a bundle. If one is accepted, it becomes
the next candidate's control and a fresh full-graph profile reranks the work.

1. Build behind an exact-shape, default-false switch. Run launch/fallback smoke,
   deterministic boundary comparison, and Compute Sanitizer. Candidates 1 and
   3 require bit identity; candidate 2 additionally runs the frozen full FP32
   accuracy gate before acceptance.
2. Run interleaved S1 A/B. Collect candidate NCU against the exact accepted
   control and inspect PTX/SASS. Promote only if S1 is directionally stable and
   NCU shows the predicted instruction/stall reduction with no resource
   regression or spills.
3. Integrate the sole candidate and go **directly** to natural whole-graph S2
   short symmetric A/B. There is no homogeneous or synthetic mixed S2 gate.
4. If positive, run long symmetric S2 and full replay, commit that one accepted
   optimization, then immediately collect fresh whole-graph S2/S1 Nsys plus
   the complete 344-ordinal S1 NCU profile before choosing further work.

Risk-adjusted execution order is barrier removal -> fast quotient -> fixed
swizzle. The fast quotient has the largest raw upside, but the barrier deletion
is first because its dependency proof is strong, its correctness contract is
bit-exact, and it is a very small follow-up to the just-accepted A-reuse
schedule.

## Evidence files

- `results/rebuild/stage38-post-rope-half2-profile/accepted-fullgraph-ranking.md`
- `results/rebuild/stage38-post-rope-half2-profile/ncu/accepted-s1-full-forward.ncu-rep`
- `results/rebuild/stage33-fused-ffn-a-reuse/report-h33b-accepted.md`
- `katago/cpp/neuralnet/tilelang_aot/fused_ffn_b13_a_reuse.cu`

