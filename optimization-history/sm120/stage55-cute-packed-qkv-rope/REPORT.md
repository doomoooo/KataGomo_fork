# Stage 55: exact-B13 CuTe packed QKV + both16 RoPE epilogue

## Decision

Retain the implementation **default-off**.  The final kernel is a real local
improvement and passes full-network accuracy, but it is not the accepted
RTX5090D/B13/S2 default because the natural two-stream graph regresses by
`1.136%`.

Replacing QKV plus an independent RoPE launch by a faster fused boundary
changes which kernels overlap and makes the following FA4 kernels materially
slower under contention.  It is not evidence that the fused QKV+RoPE kernel
itself is slower. Explicit cross-stream phase control was subsequently removed
from the optimization roadmap because its GTP integration complexity outweighs
the expected benefit; eager scheduling is left to CUDA and the GPU.

## Fixed regime

- GPU: RTX 5090 D, SM120
- board: exactly 19x19
- batch: exactly B13
- deployment topology: natural S2
- QK and PV accumulation: FP16 (`both16`)
- accepted control: Stage 47 CuTe fused FFN graph with Stage 44 packed QKV
- option: `cudaUseQKVGemmRopeAot`, default `false`

## Implementations screened

1. Shared-memory post-pass: correct but raised registers to 167 and slowed
   S1 by `26.36%`; rejected.
2. Generic register-fragment coordinates: correct mapping, 126 registers,
   but slowed S1 by `2.228%`; rejected as an implementation failure.
3. Static coordinates with duplicated or FP32 tables: reduced registers to
   96, but either increased DRAM traffic or removed useful address-generation
   ILP; S1 remained about `3.2-3.5%` slower.
4. Static coordinates, cached FP16 cos/sin table, FP16 rotary arithmetic:
   retained implementation.  The register-fragment mapping was derived from
   the printed CuTe R2S layout, not guessed from a warp shape.

The retained tactic is
`qkv-packed-cute-precomputed-rope-static-register-both16-epilogue`.
It writes Q and K after RoPE directly from the GEMM epilogue and writes V
unchanged.  A per-model/layer `[361,192,2]` FP16 cos/sin table is built once
from the learnable FP32 frequencies and cached for the model lifetime.

## Local mechanism evidence

The 400-cycle boundary ABBA result was:

- control QKV + RoPE: `19.4078 / 19.4271 us`
- fused candidate: `18.6126 / 18.6306 us`
- pooled boundary latency: `-4.099%`

Short Nsys measured `18.6531 us` for the fused kernel versus
`14.1046 + 5.3197 = 19.4243 us` for control QKV plus RoPE.

Targeted NCU for the retained candidate reported:

- 96 registers/thread, no stack/local spills
- 99.328 KiB dynamic shared memory
- one wave/SM launch geometry
- 23.39 us under replay profiling
- 11.62% DRAM and 53.40% L2 throughput
- 0.19 eligible warps/scheduler

The control QKV uses 107 registers and the separate RoPE uses 20, but the
candidate is not strictly better in every counter: candidate executed SASS
instructions were 4,657,419 versus 4,246,432 summed across the two control
launches.  Retention is therefore based on the measured local latency and
deleted global-memory/launch boundary, not a claim of universal resource
dominance.

Artifacts:

- `ncu-static-half-candidate.ncu-rep`
- `ncu-half-control-pair.ncu-rep`
- `nsys-half-candidate.nsys-rep`
- `nsys-half-control.nsys-rep`

## Natural S2 gate

One shortened A-B-B-A run used 100 timed iterations and 20 warmups per arm:

| arm | common-wall nnEval/s |
|---|---:|
| control 1 | 3993.038 |
| candidate 1 | 3942.094 |
| candidate 2 | 3942.733 |
| control 2 | 3982.416 |

Pooled control was `3987.720 nnEval/s`; pooled candidate was
`3942.413 nnEval/s`, a `-1.136%` regression.

The shortened natural-S2 Nsys run explains the reversal:

- fused QKV+RoPE: `26.395 us` average under concurrent graph load
- control QKV + RoPE: `22.541 + 8.267 = 30.808 us`
- following FA4: `14.595 -> 18.980 us`

Per-stream data is even clearer.  Control has a favorable asymmetric phase:
one stream's QKV+RoPE averages about `34.41 us`, the other `27.51 us`.
Fusion makes both QKV kernels about `26.46 us`, after which both streams'
FA4 kernels inflate to about `20.6-21.0 us`.  The local saving is real, but
the new phase increases destructive FA4 overlap and loses end-to-end S2.

Artifacts are under `natural-s2-abba/` and `natural-s2-nsys/`.

## Accuracy and fallback validation

The full 8,192-row replay passed the established both16 accuracy regime:

- policy top-1 versus FP32: `99.8413%`
- policy probability RMSE versus FP32: `1.02437e-4`
- outcome RMSE versus FP32: `0.00232237`
- score all-6 RMSE versus FP32: `0.00744188`
- policy top-1 versus Stage 47: `99.7803%`

The candidate is numerically different by design because table storage and
rotary arithmetic are FP16.  Its FP32-reference accuracy is in the same range
as Stage 47 and policy top-1 is slightly higher (`99.8413%` versus
`99.8169%`).

Additional invariants:

- a repository-generated AOT object is byte-identical in 26-row replay output
  to the development object;
- option-off B13 output is byte-identical to the prior control;
- option-on B12 falls back and is byte-identical to the prior B12 control.

## Reproducibility

The repository generator is
`python/sm120_generate_cute_qkv_rope_aot.py`.  It validates the pinned CUTLASS
commit and dense-GEMM source hash, emits the AOT header/object, bridge, patched
CuTe source, and provenance JSON without allocating a GPU.  CMake accepts the
generated bridge/object through `SM120_QKV_ROPE_CUTE_SOURCE` and
`SM120_QKV_ROPE_CUTE_OBJECT`; the default stub leaves the tactic unavailable.

No fresh accepted full-graph profile was produced because the default graph
did not change.  Stage 47 remains the accepted profile and throughput
baseline.
