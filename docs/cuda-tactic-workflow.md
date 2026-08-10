# Unified SM89/SM120 cross-batch tactic workflow

`python/cuda_tactic_workflow.py` is the single SM89/SM120 optimization workflow. Its
scope is SM89 and SM120, exact 19x19 batches and the natural two-stream
whole-graph topology. SM120 support and repository-wide integration are owned
by final-migration.

The workflow turns the retained SM89 and SM120 optimization histories into a
finite, reproducible search. B13 is neither an anchor nor a special case.
Every selected batch starts from a complete explicit official-equivalent
tactic baseline and completes its own ordered optimization flow. Every stage
measures its explicit off control, an explicit keep-incumbent no-op, and local
variants. A replacement must exceed the freshly measured incumbent by at
least 0.1%; otherwise the incumbent is carried forward.

The baseline covers every runtime tactic key, and generated plans carry it
verbatim before applying measured winners. No `keep` result can inherit a
parser default, and the runtime has no `auto`/B13-special winner selection.

## Implementation catalogs and decision groups

The 19 family names are backend implementation catalogs, not 19 independent
operators in a trunk block. The actual ordered coordinates are 10 groups on
both SM89 and SM120. Exact 19x19 is a backend invariant, not a tactic:

| SM89 group | Catalogs |
| --- | --- |
| projection/attention | `wide_projection`, `qkv_rope`, `dual_ffn` |
| residual/pointwise | `fused_residual`, `linear2`, `outproj`, `postconv_bn`, `pointwise` |
| normalization | `rmsnorm` |
| attention kernel | `fa4` |
| pre-convolution | `preconv` |
| persisting L2 | `l2` |
| weight sharing | `weight_sharing` |
| initial convolution | `initial_conv` |
| heads | `wide_head`, `initial_global`, `policy_p1`, `head_bn` |
| value terminal | `value_terminal` |

| SM120 group | Catalogs |
| --- | --- |
| attention/projection | `fa4`, `wide_projection`, `qkv_rope`, `dual_ffn` |
| residual projections | `fused_residual`, `linear2`, `outproj` |
| outer projection/pointwise | `postconv_bn`, `preconv`, `pointwise` |
| heads | `wide_head`, `policy_p1`, `head_bn` |
| normalization | `rmsnorm` |
| persisting L2 | `l2` |
| weight sharing | `weight_sharing` |
| initial convolution | `initial_conv` |
| initial global | `initial_global` |
| value terminal | `value_terminal` |

A static space gate rejects a runtime key or declared dependency that crosses
a decision-group boundary. Historical bundles are measured before the finer
catalogs in their group, and those catalogs may refine only explicitly owned
keys. In particular, QKV/RoPE bundles cannot change the FA tile or accumulator
selected by the FA catalog.

Only candidates with distinct runtime implementations are materialized. In
particular, this branch has generated tactic registries for dual-FFN and
linear2. Pre/postConv, QKV/RoPE, FA4, pointwise, policy and initial-conv use
their linked historical implementations and real config switches; the search
space does not advertise unparsed AOT IDs or imaginary launch shapes.

## 1. Select batches and materialize the search space

The release entry point first runs an explicit artifact-free stable optimized
graph for every exact batch B4-B32. It ranks physical `nnEval/s` and selects
the fastest three shapes. Only those three receive generated artifacts,
discovery, and the long gate. Each selected batch completes its entire ordered
decision flow before the next batch begins. Use `--full-batch-scan` to retain
the exhaustive 29-batch behavior.

```bash
./run-autotune.sh --device 0
./run-autotune.sh --device 0 --full-batch-scan
```

The lower-level commands below use exhaustive SM89 paths as examples for
debugging the workflow itself:

```bash
python3 python/cuda_tactic_workflow.py space \
  --architecture sm89 --gpu-class rtx4090 --device 0 \
  --batches 4-32 --streams 2 \
  --output results/portable-sm89/history-space-b4-b32.json

python3 python/cuda_tactic_workflow.py generation-plan \
  --space results/portable-sm89/history-space-b4-b32.json \
  --phase full \
  --output results/portable-sm89/history-generation-b4-b32.json
```

`generation-plan --phase seed` is only a pipeline smoke test. It is marked
incomplete and cannot justify scanning or plan distribution.

## 2. Generate and link exact-batch tactics

The two 4090s can generate the independent bundles concurrently:

```bash
/workspace/venv/bin/python3 python/portable_prepare_tilelang_fat_scan.py \
  --space results/portable-sm89/history-space-b4-b32.json \
  --family dual_ffn --batches 4-32 --device 0 \
  --output-dir results/portable-sm89/fat/dual-ffn

/workspace/venv/bin/python3 python/portable_prepare_tilelang_fat_scan.py \
  --space results/portable-sm89/history-space-b4-b32.json \
  --family linear2 --batches 4-32 --device 1 \
  --output-dir results/portable-sm89/fat/linear2
```

Each entry records the generator command, Torch/TileLang/CUDA environment,
Torch correctness error, S1 diagnostic timing, generated source hash, object
hash and exact nvcc command. Configure the search binary with both generated
registries/source lists and `-DSM89_CUDA_ARCHITECTURES=89`.

After linking, build an auditable bundle. This checks all recorded file hashes
and proves every generated extern-C launcher is present in `nm -a` output:

```bash
python3 python/cuda_tactic_workflow.py artifact-bundle \
  --space results/portable-sm89/history-space-b4-b32.json \
  --binary build-sm89-search/katago \
  --manifests results/portable-sm89/fat/dual-ffn/manifest.json \
              results/portable-sm89/fat/linear2/manifest.json \
  --output results/portable-sm89/artifact-bundle.json
```

If a regenerated search space changes only non-AOT controls or runtime routing
flags, an older generation manifest may be rebound without recompilation only
when the complete generated-candidate parameter projection is identical. The
bundle records that compatibility proof and still rechecks source/object/
metadata hashes, correctness evidence, compile commands, and linked symbols.

## 3. Complete discovery

Discovery visits every candidate for one batch before moving to the next
batch. It starts from the explicit official-equivalent baseline, scans the
batch's decision groups in order, retains each whole-graph winner, and uses the
accumulated winner config as the next stage's base. B13 follows exactly the
same path as every other batch.

Each winner records the incumbent throughput, observed improvement, minimum
acceptance margin, and whether the stage changed. Resume identity includes the
workflow/CUDA-query/fat-scan source hashes as well as space, binary, config,
model and measurement regime, so a code change cannot silently mix old and
new coordinate semantics.

Every scan subprocess sets `cudaDisableWarmup=true`. `benchmarkPureForward`
still performs a full target-batch `getOutput` and the requested benchmark
warmups before timing, so the exact lazy graph is compiled outside the timed
region. This avoids the normal evaluator's duplicate generic graph warmup and
changes only process setup time, not the measured forward path. The recorded
override also keeps `cudaWarmupOnlyMaxBatchSize=true` as a defensive fallback.
Discovery enforces at least 100 timed iterations and 50 warmups. Repeated B8
controls showed that 20/5 still sampled the GPU clock ramp, whereas 100/50
removed the resulting multi-percent bimodal noise at little additional wall
time relative to model startup.

```bash
python3 python/cuda_tactic_workflow.py scan \
  --space results/portable-sm89/history-space-b4-b32.json \
  --artifact-bundle results/portable-sm89/artifact-bundle.json \
  --binary build-sm89-search/katago \
  --config docs/baseline-configs/bench-cuda-gpu0-4090-s2.cfg \
  --model /workspace/models/b11c768h12nbt3tflrs-fson-silu.bin \
  --model-identity /workspace/models/b11c768h12nbt3tflrs-fson-silu.bin.gz \
  --batches 4-32 --phase discovery --iterations 100 --warmup 50 --repeats 1 \
  --output results/portable-sm89/history-discovery-b4-b32.json
```

Discovery values are pruning evidence only. They are never final `nnEval/s`.

The uncompressed `.bin` is an execution copy used to avoid inflating the same
model in every short-lived search subprocess. `--model-identity` keeps the
original `.bin.gz` hash as the portable plan identity. Results record both
paths and hashes, so this startup optimization is reproducible and receivers
continue validating against the distributed source model.

CUDA ordinals are local evidence and are not required to match on the
receiving machine. Each benchmark monitors the selected GPU's per-PID SM
activity; an external process that consumes SM time invalidates the
measurement, while a memory-only context is allowed.

## 4. Long final-joint gate

Only the fully accumulated joint winner is reported. The gate requires at
least 1000 timed iterations, two repeats and at most 10% relative spread:

```bash
python3 python/cuda_tactic_workflow.py gate \
  --space results/portable-sm89/history-space-b4-b32.json \
  --discovery results/portable-sm89/history-discovery-b4-b32.json \
  --artifact-bundle results/portable-sm89/artifact-bundle.json \
  --binary build-sm89-search/katago \
  --config docs/baseline-configs/bench-cuda-gpu0-4090-s2.cfg \
  --model /workspace/models/b11c768h12nbt3tflrs-fson-silu.bin \
  --model-identity /workspace/models/b11c768h12nbt3tflrs-fson-silu.bin.gz \
  --batches 4-32 --iterations 1000 --warmup 50 --repeats 2 \
  --output results/portable-sm89/history-long-gate-b4-b32.json
```

## 5. Accuracy certification and plan distribution

Rank the long-gate results by stable physical `nnEval/s`. Replay only the
fastest config on the fixed 8192-row corpus, compare it to the established
FP32 `.krnn` reference with `python/katago/train/compare_replay_krnn.py`, then
attach that one report:

The release SDK automates the complete identity-bound form and should be used
for qualification:

```bash
./run-autotune.sh --device 0 --phase reference  # once, official CUDA FP32
./run-autotune.sh --device 0 --phase accuracy   # fastest long-gate winner only
```

It binds each report to the exact batch, selected overrides, candidate binary,
model, corpus, candidate dump, and one shared FP32 reference SHA-256. The
physical tail is padded to the selected exact batch, while only the 8,192 real
rows are serialized. Candidate `.krnn` files are removed after comparison.
The lower-level `certify` example below expects reports carrying that identity
metadata; raw comparison JSON alone is intentionally insufficient.

```bash
python3 python/cuda_tactic_workflow.py certify \
  --gate results/portable-sm89/history-long-gate-b4-b32.json \
  --comparison 4=results/portable-sm89/replay-b4-vs-fp32.json \
  --comparison 5=results/portable-sm89/replay-b5-vs-fp32.json \
  --output results/portable-sm89/history-long-gate-certified.json
```

Certification requires 8192 rows, aggregate metrics, and the same
worst-per-request max-absolute/per-head RMSE policy/value/score/ownership FP32
envelope used by the GTP-shaped CPU verifier. A distributable plan combines the
selected batch's discovery, long gate, and certificate. The lower-level
example remains exhaustive only to document that supported mode:

```bash
python3 python/cuda_tactic_workflow.py plan \
  --space results/portable-sm89/history-space-b4-b32.json \
  --results results/portable-sm89/history-discovery-b4-b32.json \
            results/portable-sm89/history-long-gate-certified.json \
  --batches 4-32 \
  --output results/portable-sm89/rtx4090-s2-b4-b32-plan.json

python3 python/cuda_tactic_workflow.py apply \
  --plan results/portable-sm89/rtx4090-s2-b4-b32-plan.json \
  --batches 4-32 --output results/portable-sm89/receiver-overrides.json
```

`ready_for_scan_bypass=true` means every history point was discovered and
every final joint config passed the long gate. `production_ready=true`
additionally requires the 8192-row certification. Receivers validate the
architecture, GPU class, board, streams, model/config identity and candidate
parameters, then apply per-batch overrides without scanning.

## Reproducibility record

Scan and generation provenance captures:

- git revision, dirty state and diff stat;
- model, config, binary, space, source and object hashes;
- complete generation, nvcc, CMake and benchmark commands;
- the search binary's own CMake cache and `compile_commands.json` (configure
  with `-DCMAKE_EXPORT_COMPILE_COMMANDS=ON`);
- compiler, CMake, NVIDIA driver, nvcc, CUDA and cuDNN versions;
- linked-library (`ldd`) output, git submodules and the pinned flash-attention
  revision/status;
- GPU names and PCI bus IDs from driver tooling, plus CUDA Runtime device
  properties recorded by `benchmarknn`: compute capability, SM/warp/thread and
  register limits, shared-memory/L2 sizes, memory bus/clock, async engines and
  concurrent-kernel support;
- the CUDA Runtime query captured before scanning and the independent
  `benchmarknn` CUDA property record, including the actual SM count;
- Python, Torch CUDA/cuDNN, TileLang and `pip freeze`;
- relevant CUDA, CUTLASS, TileLang, Triton and compiler environment variables.

Versions are reproducibility evidence, not strict equality gates. The receiver
may use compatible newer libraries while retaining enough information to
reconstruct the producer environment exactly when needed.

Runtime kernel policy does not inspect product names. FA4 receives the CUDA
Runtime-reported SM count, persisting-L2 uses CUDA attributes/limits, and AOT
launchers are dispatched by SM89 architecture, exact batch, stream topology
and tactic ID. The distributed plan records the numeric producer capabilities
as compatibility evidence; a receiver need not have the same marketing model.
