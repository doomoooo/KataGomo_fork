# SM120 exact-batch TileLang fat scan

The fat-scan path trades a larger one-time compile/link for cheap whole-graph
measurements. It generates every explicitly selected TileLang `(batch, tactic
ID)` translation unit first, links the complete family once, and dispatches by
the exact runtime batch and requested tactic ID. It does not infer an anchor or
restrict generation to the left edge of a throughput plateau.

The default build remains unchanged in behavior: each family has an empty fat
registry and the existing single-candidate search-slot ABI remains available.
Official/library fallback candidates also remain available. Fat entries are
explicit-only and are never selected by `auto`.

## One-command full B1-B32 family scan

Generate a schema-2 space containing all batches first. For example:

```sh
python3 python/sm120_tactic_search.py space \
  --gpu-class rtx5090d --device 2 --batches 1-32 --streams 2 \
  --output results/space-5090d-b1-b32-s2.json
```

`space` queries the selected CUDA ordinal and records its compute capability,
SM count, memory/resource limits, PCI identity, and CUDA driver/runtime
versions. A non-SM120 device is rejected; hardware capabilities are never
inferred from the `rtx5080`/`rtx5090d` label.

Then add `--fat-scan` to the normal runner. The remaining benchmark arguments
are the same as the single-slot workflow:

```sh
python3 python/sm120_run_tactic_search.py \
  --fat-scan \
  --space results/space-5090d-b1-b32-s2.json \
  --family ffn \
  --repo . \
  --build-dir build/fat-ffn \
  --active-source-dir build/fat-ffn/active \
  --config docs/baseline-configs/bench-cuda-gpu2-5090d-s2.cfg \
  --model /workspace/models/model.bin.gz \
  --device 2 --batches 1-32 --streams 2 \
  --iterations 80 --warmup 15 --repeats 1 \
  --output results/ffn-fat-b1-b32.json
```

`--candidate-ids id-a,id-b` is an optional global filter. Without it, every
TileLang implementation in the selected family's materialized B1-B32 space is
included. Fallback and non-TileLang implementations are not put in the fat
bundle; the runner still measures fallback and records unsupported generators
as before. `--reuse-fat-generated` reuses a prior TU only when its source,
metadata, generator hash, and exact candidate projection match. Pure
space-level provenance additions such as CUDA device records do not force AOT
regeneration; the rewritten manifest records the source-space hash.

When a completed fat bundle already exists, use its manifest as a read-only
input and avoid repeating S1 generation:

```sh
python3 python/sm120_run_tactic_search.py \
  --fat-scan \
  --fat-manifest /abs/path/ffn-fat/manifest.json \
  --candidate-selection /abs/path/ffn-s1-selection.json \
  --space /abs/path/space.json --family ffn \
  --repo . --build-dir build/ffn-s2 --active-source-dir build/ffn-s2/active \
  --config /abs/path/bench.cfg --model /abs/path/model.bin.gz --device 2 \
  --batches 4-32 --streams 2 --iterations 80 --warmup 15 --repeats 3 \
  --runner 'gpu-lock with --gpu 2 --' \
  --output results/ffn-s2.json
```

The runner verifies the manifest's search-space hash, exact candidate
parameters, source/metadata hashes, registry hash, and requested exact keys;
it refuses an incomplete or mismatched bundle. `--candidate-selection` is a
per-family, per-batch S1 retention file. It is a narrowing filter only: final
selection still comes from natural whole-graph S2 throughput.

FA4's global exact-batch candidate set includes `tile_n=64,96,128`. The N96
point was first observed on the RTX 5090D at B13, but it is intentionally
materialized for every requested batch and GPU class; failed or slower builds
are rejected by the normal correctness/S2 result instead of being removed from
the search space in advance.

Generation still invokes TileLang once per exact shape because each kernel is
specialized for `batch * 361` tokens. The speedup is that CMake configuration,
the full KataGo link, and binary hashing happen once for the whole family,
instead of once per candidate.

## Prepare and build separately

The bundle can also be prepared without starting whole-graph measurements:

```sh
python3 python/sm120_prepare_tilelang_fat_scan.py \
  --space results/space-5090d-b1-b32-s2.json \
  --family ffn --batches 1-32 --device 2 \
  --output-dir results/generated/ffn-fat
```

The emitted `manifest.json` records all exact keys, hashes, source paths, and
the generated registry. Configure a build manually with:

```sh
cmake -S cpp -B build/fat-ffn -DUSE_BACKEND=CUDA \
  -DSM120_SEARCH_FFN_FAT_REGISTRY_SOURCE=/abs/path/sm120_search_ffn_fat_registry.cu \
  '-DSM120_SEARCH_FFN_FAT_SOURCES=/abs/path/a.cu;/abs/path/b.cu'
cmake --build build/fat-ffn -j4
```

Only one family should be fat-linked for a low-cost scan. The runner resets all
other family fat registries and legacy active slots to stubs, avoiding stale
CMake-cache state. This source-slot reset is not an all-off network reset: the
accepted SM120 runtime defaults and supplied config remain active, and
`--isolate-family` is diagnostic-only. Acceptance remains natural whole-graph
S2 total throughput; the fat mechanism does not add homogeneous or mixed
local-S2 gates.

## Link safety

Each generated TU receives a deterministic symbol token derived from family,
exact batch, and the SHA-256 of the tactic ID. Both the CUDA kernel and launcher
use that token. TileLang's header-defined debug helpers (`PrintTraits`,
`debug_print_*`, and `device_assert*`) are macro-renamed with the same token,
which prevents the duplicate linker definitions seen when ordinary generated
TUs are linked together.

Run the CPU/static regression tests with:

```sh
python3 -m unittest python/tests/test_sm120_fat_scan.py
```

## Export and distribute a tactic plan

After discovery has measured every candidate in every requested family and
exact batch, export the independent winning candidate table. This is a seed
for accumulated coordinate search, not a deployable result. A plan is refused
when any candidate is missing or failed; `--allow-partial` is only for
inspecting incomplete discovery.

```sh
python3 python/sm120_tactic_plan.py build \
  results/rebuild/cross-batch-search/full-5090d/ffn.json \
  results/rebuild/cross-batch-search/full-5090d/qkv.json \
  results/rebuild/cross-batch-search/full-5090d/linear2.json \
  results/rebuild/cross-batch-search/full-5090d/fa4.json \
  results/rebuild/cross-batch-search/full-5090d/l2.json \
  --space results/rebuild/cross-batch-search/space-5090d-b4-32-s2-v7.json \
  --families ffn,qkv,linear2,fa4,l2 --batches 4-32 \
  --output results/rebuild/cross-batch-search/tactic-plan-5090d-b4-32.json
```

Run the small family spaces again as an accumulated per-batch coordinate
search. Every candidate is measured on the natural two-stream full graph with
the currently accepted choices for the other four families pinned. The winner
is written into the baseline before advancing to the next family; extra
passes may revisit interactions without searching the Cartesian product.
The currently selected tactic is itself a required candidate in every
coordinate, so it is remeasured as the explicit no-op. Exact ties and gains
below the default 0.1% discovery threshold keep that incumbent. The exported
decision chain records both throughputs, the threshold, and the full
before/after state; the joint and final gates reject missing, regressing,
sub-threshold, or non-accumulated coordinate evidence.
The current v7 space includes FA4 `tile_n=64,96,128` for every requested
batch; v4/v5 artifacts predate the global N96 addition and must not be reused
as complete coverage.

```sh
gpu-lock with --gpu 2 -- python3 python/sm120_coordinate_search.py \
  --seed-plan results/rebuild/cross-batch-search/tactic-plan-5090d-b4-32.json \
  --space results/rebuild/cross-batch-search/space-5090d-b4-32-s2-v7.json \
  --repo . --build-dir build-cuda-coordinate-5090d-sm120 \
  --active-dir results/rebuild/cross-batch-search/coordinate-active-5090d \
  --output results/rebuild/cross-batch-search/coordinate-5090d-b4-32.json \
  --plan-output results/rebuild/cross-batch-search/tactic-plan-5090d-b4-32-coordinate.json \
  --config docs/baseline-configs/bench-cuda-gpu2-5090d-s2.cfg \
  --model /workspace/models/model.bin.gz --device 2 \
  --batches 4-32 --streams 2 --iterations 100 --warmup 30 --repeats 1
```

The coordinate plan contains exact candidate parameters, per-batch overrides,
model/config/search-space hashes, CUDA-reported device capabilities, measured
evidence, and reproducibility snapshots. Producer-only absolute paths are
evidence and are not strict receiver gates. It remains
`ready_for_scan_bypass=false` until its exact selections pass the long joint
gate.
Receiver validation queries CUDA again. Compute capability, SM count, and
tactic-relevant resource limits (shared memory, registers, block/thread
limits, L2/persisting-L2, and memory-bus width when reported) must match;
driver, CUDA, cuDNN, compiler, and library versions remain reproduction
evidence rather than byte-for-byte compatibility gates.
An older independent seed may be rebound to a newer superset space only when
its source space still matches its recorded hash and every selected candidate
is exactly equal in both spaces. This is seed migration only: the coordinate
runner still measures every candidate in the new space.

After that gate, attach the result to the exact coordinate plan. Finalization
checks the plan-file hash as well as space, model, config, stream topology,
selected tactic IDs, iteration count, sample count, and sample spread:

```sh
python3 python/sm120_tactic_plan.py finalize \
  --plan results/rebuild/cross-batch-search/tactic-plan-5090d-b4-32-coordinate.json \
  --space results/rebuild/cross-batch-search/space-5090d-b4-32-s2-v7.json \
  --model /workspace/models/model.bin.gz \
  --config docs/baseline-configs/bench-cuda-gpu2-5090d-s2.cfg \
  --batches 4-32 --streams 2 \
  --joint-result results/rebuild/cross-batch-search/joint-plan-5090d-s2-long.json \
  --output results/rebuild/cross-batch-search/tactic-plan-5090d-b4-32-final.json
```

Every requested batch must have a `long_stable` joint row with the same five
tactic IDs, at least 1000 timed iterations, at least two samples, and no more
than 10% relative sample spread. Only then does the plan become
`ready_for_scan_bypass=true`. Validate that final plan before use:

```sh
python3 python/sm120_tactic_plan.py validate \
  --plan results/rebuild/cross-batch-search/tactic-plan-5090d-b4-32-final.json \
  --space results/rebuild/cross-batch-search/space-5090d-b4-32-s2-v7.json \
  --model /workspace/models/model.bin.gz \
  --config docs/baseline-configs/bench-cuda-gpu2-5090d-s2.cfg \
  --family ffn --batches 4-32 --streams 2 --device 2
```

To bypass exhaustive candidate scanning in the normal runner, pass the final
long-gated plan. The runner validates all five common-graph families and, for the
requested family invocation, generates, builds, and measures only its selected
exact-batch tactic; it does not invoke the scan candidate loop. Invoke it once
per family when materializing a complete build, or consume the plan's
`apply.per_batch_tactic_overrides` map in the deployment-side dispatcher.

```sh
python3 python/sm120_run_tactic_search.py \
  --tactic-plan results/rebuild/cross-batch-search/tactic-plan-5090d-b4-32-final.json \
  --space results/rebuild/cross-batch-search/space-5090d-b4-32-s2-v7.json \
  --family ffn --repo . --build-dir build/plan-ffn \
  --active-source-dir build/plan-ffn/active \
  --config docs/baseline-configs/bench-cuda-gpu2-5090d-s2.cfg \
  --model /workspace/models/model.bin.gz --device 2 \
  --batches 4-32 --streams 2 --output results/plan-ffn.json
```

The plan has passed the producer's long joint throughput gate, but is not an
unconditional numerical-correctness certificate. The receiver should still
run correctness checks and may repeat ABBA/BAAB validation. Environment capture records Python packages, `pip freeze`, CUDA
toolchain/NVIDIA driver output, cuDNN as reported by PyTorch, compiler/CMake
versions, relevant environment variables, repository/third-party revisions,
config text and hashes, exact CMake commands, and the runner invocation.

The plan's selection metric is still the natural whole-graph `benchmarknn`
throughput. A faster isolated QKV/Linear2 kernel is not automatically a faster
network: the receiver must compare the plan candidate against a control with
the other planned families held fixed. The validation helper emits both ABBA
and BAAB orderings and stores the complete commands and environment snapshot:

```sh
python3 python/sm120_validate_tactic_plan.py \
  --plan results/rebuild/cross-batch-search/tactic-plan-5090d-b4-32-final.json \
  --space results/rebuild/cross-batch-search/space-5090d-b4-32-s2-v7.json \
  --family qkv --batches 32 \
  --binary build/plan-b32/katago \
  --config docs/baseline-configs/bench-cuda-gpu2-5090d-s2.cfg \
  --model /workspace/models/model.bin.gz --device 2 \
  --iterations 300 --warmup 50 --repeats 3 --order both \
  --runner 'gpu-lock with --gpu 2 --' \
  --output results/validation-b32-qkv-long.json
```

For packed CuTe QKV, the first generated candidate configures a stable bridge,
header, and object path; later exact batches replace those files and relink
without forcing a complete C++ target rebuild. The per-candidate metadata also
records the CUTLASS revision, generator parameters, artifact hashes, and (when
the correctness replay is run) the CUBLAS comparison. This is useful for
reproducing a plan on a similar, but not byte-identical, CUDA installation.

## Joint-plan full-graph curve

The per-family scan is not itself a deployable curve: all five family choices
must be materialized for the same exact batch before measuring the graph. The
joint runner materializes the selected generated/historical FFN, planar or
CuTe QKV, TileLang Linear2, FA4 AOT, and persisting-L2 choices together:

```sh
gpu-lock with --gpu 2 -- python3 python/sm120_measure_joint_plan.py \
  --plan results/rebuild/cross-batch-search/tactic-plan-5090d-b4-32-coordinate.json \
  --space results/rebuild/cross-batch-search/space-5090d-b4-32-s2-v7.json \
  --repo . --build-dir build-cuda-joint-plan-5090d-sm120 \
  --active-dir results/rebuild/cross-batch-search/joint-plan-active-5090d \
  --output results/rebuild/cross-batch-search/joint-plan-5090d-s2-full.json \
  --config /workspace/bench-cuda-gpu2-5090d-s2.cfg \
  --model /workspace/models/b11c768h12nbt3tflrs-fson-silu.bin.gz \
  --device 2 --batches 4-32 --streams 2 \
  --iterations 1000 --warmup 30 --repeats 3
```

The joint runner's default is a long measurement. Only rows marked
`measurement_kind=long_stable` are eligible for the final peak report:

```sh
python3 python/sm120_report_joint_plan.py \
  results/rebuild/cross-batch-search/joint-plan-5090d-s2-long.json \
  --output results/rebuild/cross-batch-search/joint-plan-5090d-s2-report.json
```

This reports the highest `stable_long_nn_evals_per_sec` and its batch/tactic
selection. Short S2 scan medians remain useful for pruning and ranking, but
are not final performance claims.

The first run is retained in
`results/rebuild/cross-batch-search/joint-plan-5090d-s2-full.json`, but it must
not be used as CuTe-QKV evidence: its generated CuTe bridge was copied beside
the object while CMake still compiled the ordinary QKV stub. The fixed runner
selects the bridge itself as `SM120_SEARCH_QKV_SOURCE`, while linking the
generated device object through `SM120_SEARCH_QKV_OBJECT`.

An earlier corrected RTX 5090D diagnostic run is recorded in
`results/rebuild/cross-batch-search/joint-plan-5090d-s2-fixed-qkv.json`. Key
short whole-graph observations were B13 4,265.6, B14 4,332.3, B15 4,306.6,
B16 4,251.5,
B18 4,263.7, B19 4,368.8, B20 4,065.8, B25 4,079.5, B27 4,047.5, and B32
4,154.0 nnEval/s. The curve is substantially smoother than the pre-fix
numbers, but B20 and the B25/B27 region remain real discontinuities in the
current search space. These are not final long-stable claims. The JSON stores
exact per-batch sources, binary hashes,
commands, model/config hashes, and the full environment snapshot.

## Corrected Nsight evidence and resource gaps

Full-graph Nsight Systems reports are under
`results/rebuild/cross-batch-search/nsight-joint-5090d-s2/`, including
`nsys-fixed-b13.nsys-rep`, `nsys-fixed-b14.nsys-rep`, `nsys-b16.nsys-rep`,
`nsys-fixed-b19.nsys-rep`, `nsys-fixed-b20.nsys-rep`, `nsys-fixed-b25.nsys-rep`,
and `nsys-fixed-b27.nsys-rep`. The corresponding `stats-fixed-*` CSV files
are exported from the same reports. B19's full-graph top kernels are FFN
47.84 us, CuTe QKV 30.10 us, and FA4 26.32 us; B20 is FFN 44.12 us, Linear2
43.00 us, CuTe QKV 30.79 us, and FA4 26.40 us. B25 exposes a different
problem: the fallback residual GEMM contributes 114.90 ms over the report and
the first `128x256` cuBLAS kernel is limited to one CTA/SM. B27 instead has
FFN 60.34 us and AOT Linear2 57.83 us at grids `18x77` and `3x77`.

Nsight Compute basic-set reports target one matching kernel from the same
two-server full graph. They are named `ncu-fixed-b*-*.ncu-rep` in that
directory. The resource signatures show the missing search dimensions:

| kernel | representative resource signature |
| --- | --- |
| historical TileLang FFN, B14/B19/B20 | 167 regs/thread, 32.768 KiB dynamic smem, 3 CTA/SM; only grid Y and wave count grow with batch |
| CuTe packed QKV, B13/B14/B19/B20 on the 5090D | 288 threads, 107 regs/thread, 99.328 KiB dynamic smem, 1 CTA/SM, observed cluster grid Z=170 |
| TileLang Linear2, B13/B27 | 162 regs/thread, 65.536 KiB dynamic smem, 3 CTA/SM |
| TileLang Linear2, B20 | 210 regs/thread, 49.152 KiB dynamic smem, 2 CTA/SM |
| FA4, B14/B19/B20 | 168 regs/thread, 16.384 KiB dynamic smem, 3 CTA/SM; grid Z follows batch |

Thus the hand-written backends do account for some resource knobs (tile,
stages, threads, `min_blocks`, dynamic smem, FA4 shape, and exact-batch grid),
but they do not yet perform closed-loop SM resource tuning. CuTe's atom layout
and CTA shape remain fixed. The default `max_active_clusters` is now queried
from `cudaDevAttrMultiProcessorCount` for the target CUDA ordinal while the
AOT object is materialized; explicit values are retained only as named
wave-search candidates. There is no search over register caps, cluster
dimensions, active-cluster count, or a cost model for wave boundaries.
The accumulated coordinate runner now measures each family candidate with the
other current winners pinned, then carries the accepted choice into the next
family. This closes the independent-maxima workflow error without exploding
into the full Cartesian product. It remains a local search: register caps,
cluster dimensions, and new candidate families still require explicit space
extensions. The B20 Linear2 signature and the B25/B27 L2/fallback switches are
examples of why repeated coordinate passes may still be useful.

The small L2 follow-up also matters: B25 measured about 4,164 nnEval/s with
its selected 0.75 ratio, 4,104 at ratio 1.0, and 4,073 with L2 off; B27
measured about 4,072 at ratio 1.0 versus 4,035 with L2 off. L2 is therefore a
real discrete plan dimension. The accumulated search measures it after the
other accepted coordinates, while the final long curve remains the smoothing
and stability check.
