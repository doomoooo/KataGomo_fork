# Stage 30: RTX 5090 D fixed-B13 initial-convolution plan search

## Decision

Nominate cuDNN frontend engine 47, tag
`eng47_k2=2_k6=1_k13=1_k14=0_k22=2`, for later fixed B13/S2 integration.
It clears the preregistered isolated margin by a wide amount, uses zero
workspace, and has a profiler-supported mechanism. No shared KataGo source or
configuration was changed in this stage.

The exact target is ordinal 5's FP16 NHWC convolution:
`[13,22,19,19] * [768,22,3,3] -> [13,768,19,19]`, pad 1, stride/dilation 1,
FP32 compute, beta 0. The probe uses the production logical dimensions,
strides, filter layout, legacy algorithm, non-blocking stream, and a full
frontend graph-call timing boundary.

## Enumeration

The final probe attempted 67 source records: one explicit engine-45 plan, 31
heuristic-A plans, 31 heuristic-B plans, and four fallback plans. All 67
validated, built, fit under the 64 MiB cap, executed, and passed correctness.
Heuristic A and B returned the same ordered 31 tags, leaving 36 unique plan
identities overall.

`plans-enumeration.json` retains every engine, every knob type/value encoded in
the complete tag, workspace, behavior notes, graph/build/execute status,
correctness, and short alternating-order timing. This includes slow outcomes:
20 of the 36 unique tags tied or lost to legacy, with the worst plans ranging
from about 61 us to 500 us. They were not discarded.

## Long isolated confirmation

Each result below uses four alternating 5,000-call measurements after 100
warmups per leg. The control is exact
`CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM` in the same process.

| plan | source | workspace | legacy median us | candidate median us | speedup | correctness |
|---|---|---:|---:|---:|---:|---|
| `eng47_k2=2_k6=1_k13=1_k14=0_k22=2` | heurA | 0 | 29.150 | **16.172** | **1.802x** | maxabs 6.10e-5, rel-L2 7.70e-6 |
| `eng47_k2=15_k6=1_k13=0_k14=0_k22=1` | heurA | 0 | 29.131 | 18.201 | 1.601x | maxabs 6.10e-5, rel-L2 7.70e-6 |
| `eng71` | fallback | 0 | 29.318 | 20.274 | 1.446x | bit exact |
| `eng45_k14=2_k2=0` | explicit45 | 557,056 B | 29.206 | 21.078 | 1.386x | bit exact |

Winner raw samples were legacy
`[28.980,30.527,29.319,28.917] us` and candidate
`[15.866,16.451,15.893,16.454] us`. The candidate-first legs are about 3.6%
slower than candidate-second legs, but even the weakest paired comparison is
far above the required 10% margin. The local result is therefore not a
one-order false positive.

The engine-47 winner's complete backend knob identity is:

| backend knob | name | value |
|---:|---|---:|
| 2 | `TILE_SIZE` | 2 |
| 6 | `LDGA_DEPRECATED` | 1 |
| 13 | `TILEK` | 1 |
| 14 | `STAGES` | 0 |
| 22 | `LDGC_DEPRECATED` | 2 |

Because the high-level frontend `KnobType_t` no longer exposes deprecated
knobs 6 and 22, later integration should query heuristic A, select this exact
full tag, and build only that index. It must fail closed to legacy if the tag
is absent. Do not select by ordinal index 30 alone, since heuristic ordering is
not a stable contract.

## Profiler explanation

Nsys ranges separate the timed legacy and winner calls:

| phase/kernel | grid | block | regs/thread | dynamic smem | calls | Nsys avg us |
|---|---:|---:|---:|---:|---:|---:|
| legacy padding kernel | 340 | 768 | 34 | 0 | 40 | 1.178 |
| legacy main convolution | `296x3` | 128 | 94 | 81,920 B | 20 | 19.288 |
| engine-47 single convolution | `24x37` | 128 | 128 | 4,096 B | 20 | 15.640 |

Each legacy convolution emits two padding kernels, matching ordinals 3 and 4
in the whole-network trace, followed by ordinal 5's main kernel. Engine 47
implements the same padded graph in one kernel. On kernel time alone the
boundary changes from roughly `2*1.178 + 19.288 = 21.644 us` to 15.640 us, a
27.7% reduction. The larger 44.5% event-boundary reduction also removes two
launches and their stream gaps.

NCU confirms that both main kernels launch 888 CTAs, but resource scheduling is
materially different:

| metric | legacy main | engine 47 |
|---|---:|---:|
| registers/thread | 94 | 128 |
| dynamic shared memory | 81.92 KiB | 4.10 KiB |
| active CTA limit from shared memory | 1 | 12 |
| effective theoretical active CTAs/SM | 1 | 4 (register limited) |
| theoretical occupancy | 8.33% | 33.33% |
| achieved occupancy | 8.32% | 26.07% |
| waves/SM | 5.22 | 1.31 |

NCU replay instrumentation inflates durations, so its timing values are not
used for the speed comparison; Nsys and unprofiled CUDA events provide timing,
while NCU provides the resource explanation.

## S2 handoff

The local margin is sufficient to justify a narrowly guarded S2 integration
experiment. The expected whole-network effect remains small: based on the
roughly 6 us kernel-boundary saving and one call per forward, the direct share
of a roughly 6.9--7.1 ms per-server forward is under 0.1%, although launch-gap
and phase changes can move the observed result. The earlier 5080/4090 outcomes
make approximately 0.1--0.3% a reasonable measurement target, not a guarantee.

Integration requirements:

1. Gate on SM120, B13, 19x19, C22->C768, 3x3, FP16 NHWC, beta 0.
2. Build per cuDNN handle/server by exact tag and record zero fallback.
3. Include both ordinal-3/4 padding removal and ordinal-5 replacement in Nsys
   attribution; comparing ordinal 5 alone understates the local change.
4. Run forward/reverse S2 ABBA before accepting. A local win does not establish
   stream coexistence or phase benefit.
5. Run the full 8192-row all-head accuracy comparison before changing a
   default. Engine 47 is close but not bit exact on deterministic random input.

Top candidates are retained in `long-eng47-tile2.json`,
`long-eng47-tile15.json`, `long-eng71.json`, and `long-explicit45.json`.
Profiler artifacts are `nsys-eng47-tile2.nsys-rep`,
`ncu-eng47-tile2.ncu-rep`, and `ncu-legacy-main.ncu-rep`.
