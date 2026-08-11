# SM89 / SM120 positive-history closure (2026-08-08)

This is the migration contract, not a backlog. A historically positive and
numerically valid tactic is present only when all four links below exist at
the same revision:

1. an executable backend implementation;
2. a distinct B4-B32 scan coordinate;
3. a runtime activation marker emitted only after that implementation runs;
4. a non-empty plan-apply mapping that reproduces the runtime configuration.

`python/cuda_tactic_history.py` is the machine-readable inventory and
`validate_positive_history_closure()` enforces the four links while the search
space is materialized. The contract currently closes 62 SM89 records and 64
SM120 records for every exact batch B4-B32.

## Scope rule

- The architecture family lists are allowed to differ.
- Their ordered union is the complete workflow vocabulary.
- A later S2 or whole-graph regression does not erase an earlier real local/S1
  win. Such a coordinate remains searchable and normally default-off.
- Strictly negative, accuracy-failed, or never performance-measured experiments
  are excluded.
- Byte-identical output is valid and particularly strong precision evidence.
- A config parser entry, candidate name, or archived report alone is never
  counted as an implementation.

## Audited evidence

- `/workspace/results/sm89/HISTORY.md`
- `/workspace/results/sm89/INTRINSIC-S2-AUDIT.md`
- `/workspace/results/sm89/stage*/`
- `/workspace/results/docs/cuda-optimization-history.md`
- `/workspace/results/sm120/HISTORY.md`
- `/workspace/results/sm120/stage*/`
- `/workspace/results/docs/SM89_SM120_AUTOTUNE_HANDOVER_20260807.md`
- frozen SM89 implementation `/workspace/katago-4090`
- frozen SM120 implementation `/workspace/katago-packed-qkv-b1b32`
- retained SM120 line
  `ee4d1d8 -> 67f034d -> acf588c -> 91f0e80 -> 3526b13 ->
  1c963d6 -> a2178ec -> fa468e0`

Three independent history reviews were reconciled against the current source.
The final inclusion rule is the positive-history rule above; an older audit's
`pending` label is not an exclusion if the experiment had real positive and
numerically valid evidence.

## Restored SM89 coordinates

In addition to the frozen retained routes, the closure contains the historical
coordinates that were previously collapsed into one winner:

- Stage 7 FA native-D32 FP32: M128N112, M128N96, M64N96 packed-GQA, and
  M64N96 unpacked, plus retained both16. The FlashAttention source patch now
  exposes tile M/N, warp count, packing, and accumulation as real template
  coordinates.
- Stage 8 dual-FFN CUTLASS swizzle2 and the locally positive swizzle4, plus the
  retained half2-tanh and exact-batch generated neighborhood.
- Stage 9 linear2 CUTLASS warp64x32 S3/S4 and warp64x64 S3-S6.
- Stage 10 out-projection CUTLASS warp64x32 S2-S4 and warp64x64 S3/S4.
- Stage 11 pre-convolution CUTLASS warp64x32 S3/S4 and warp64x64 S3-S6.
- Stage 12 post-convolution CUTLASS retained 128x128 geometries and the
  positive 128x256 / 256x128 coordinates, including their distinct swizzles.
- Stage 25 policy-P1 launch geometries 96x1 and 96x5.
- Stage 6 downstream exact-mask/attention-bias elision and Stage 50 mask
  preprocessing elision as independent, composable coordinates.
- Stage 30 QKV epilogue RoPE with a model-lifetime precomputed float2 table.
- Stage 34 C384 vec8 pointwise and Stage 39 eight-row RMSNorm.
- The Stage 2 wide-QKV + wide-FFN result is represented as one bundle because
  its +4.8% evidence did not isolate the two members independently.

The old QKV split `v2` name was removed: the backend only had two distinct
implementations, FP32 epilogue (`v0`) and native-half (`v1`).

## Restored SM120 coordinates

The scan branch had diverged from the retained implementation branch. The
following positive routes now have backend, scan, activation, and plan links:

- strided and AOT wide-QKV, including TileLang K64/S2, lower-smem K32/S3,
  and CuTe packed atom2x2/atom4x2;
- the combined S1 strided-QKV + single-wide-FFN projection bundle;
- exact-mask preprocessing elision;
- scalar, half2, batch-shared, unrolled, and packed-QKV+precomputed-RoPE;
- FA N64 and N128 for FP32/qk16/pv16/both16, and N96 only for the evidenced
  both16 variant;
- original dynamic-batch CUTLASS shared-A dual-FFN, Stage20 TileLang exp,
  historical TileLang tanh, Stage33b A-fragment reuse, the historical
  three-stage schedule, and Stage47 CuTe paired-projection grid340;
- CUTLASS and generated residual linear2/out-projection routes, including all
  numerically valid Stage22 locally-positive tile/stage/thread geometries and
  the later locally-positive M128N96/S4 linear2;
- independent outer down/up projections, the Stage93 down+up warp64x64
  bundle, and Stage45 postConv+BN+SiLU;
- ordered, exact one-warp, and vec8 RMSNorm; SwiGLU and all positive affine
  SiLU variants;
- persisting-L2 trunk/inner coordinates, model-weight sharing, initial-conv
  engine45 and engine47, initial-global, policy-P1, head BN, full C384 wide
  head, and partial C288 g1+v1 wide head.

The SM89 value-terminal path is also executable and searchable on SM120 as a
portable implementation, but it is intentionally absent from the SM120
positive-history table because no SM120 positive measurement was found.

## Runtime enforcement

Generated AOT candidates are linked through an all-family fat registry. The
artifact bundle verifies source/object/metadata hashes and exported launch
symbols against the exact binary. Scanner and final-joint runs require every
selected activation marker.

For SM120, a plan-explicit FA, QKV, QKV+RoPE, FFN, linear2, or out-projection
tactic is also a backend hard contract: missing registration, unmet packed
preconditions, or launch failure throws instead of silently using an official
fallback. Initial-conv frontend plans likewise throw if the selected exact
batch plan is unavailable. The official path is entered only by explicitly
disabling the architecture backend; a selected optimization may not silently
fall back.

## Explicit exclusions

Examples excluded by the contract include SM89 Stage55 head pooling, Stage58
projection-to-RMS, FA N128 both16 and min-blocks=4; and SM120 two-way RoPE,
rounding-changing RMS folds, fused head pooling order drift, register-Q FA,
prefetch-preserving A-reuse, accuracy-failing fast-tree RMS, and other measured
negative/no-signal routes. These remain in the archived evidence, not in the
production search space.
