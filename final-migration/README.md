# KataGo plan-driven CUDA fork

[中文](README.zh-CN.md) | English

This repository is a clean fork of official
[`lightvector/KataGo`](https://github.com/lightvector/KataGo) at commit
`6a1fc5de9fc253723ac475a0683bf0b9d9b7bd19` (`v1.17.2`, fetched on
2026-08-07). It retains KataGo's GTP, analysis, search, model, and game logic
and adds a shape-specialized, plan-driven CUDA inference path for NVIDIA SM89
and SM120.

This project only optimizes the CUDA backend. TensorRT is not required or used
by the production path. These changes are not part of upstream KataGo.

## Quick start

Extract the source-complete tar in a persistent writable directory, then use
the certified plan already carried for the selected GPU:

- NVIDIA GeForce RTX 4090 D: B12, two streams
- NVIDIA GeForce RTX 5080: B16, two streams

The CUDA product name is the plan registry key. Other GPU models must first
run autotune to produce their own plan.

```bash
./setup.sh
./build-for-plan.sh --device 0
./run.sh --device 0
```

`setup.sh` installs only below the extracted directory. `build-for-plan.sh`
looks up CUDA Runtime device 0 by product name, validates that plan, builds
only its selected artifacts, and does not run autotune. `run.sh` verifies the
resulting manifest and starts KataGo GTP with the certified batch, streams,
scheduler, and CUDA pipeline.

The same entry-point names are checked in at the root of a direct source
clone. In that layout `./setup.sh` builds the current dependency sources and
installs the current system CUDA stack, keeps managed files under
`.final-migration-env`, then prepares the model and 8192-row corpus used by
autotune. Local archives and `KATAGO_MODEL` are checked
before the pinned model's official GitHub release; the training corpus is
resolved from the current KataGo archive. The canonical script name is
`run-autotune.sh` (with a hyphen).

To search and certify a new plan instead:

```bash
./setup.sh
./run-autotune.sh --device 0
./run.sh --device 0
```

## What this fork adds

| Area | Method |
| --- | --- |
| Kernel selection | An offline whole-graph scanner emits an explicit, versioned tactic plan |
| Shape contract | Exact 19x19 FP16/NHWC execution with a certified physical batch |
| Batch dispatch | Full batches launch immediately; an idle GPU may launch an underfilled logical group padded to the plan batch |
| Host submission | One persistent worker per inference lane, without serial host-side completion waits between lanes |
| Transfers | Pinned staging, dedicated upload/download streams, and CUDA event dependencies |
| Buffer reuse | One device slot guarded by input-consumed and output-consumed events |
| CUDA streams | Every optimized launch uses the owning NN-server stream explicitly |
| Multi-GPU | Device-local streams, events, handles, buffers, and idle state |
| Correctness | Immutable 8192-row full-FP32 reference, request identity checks, and a GTP-shaped stress harness |
| Distribution | Source-complete autotune tar and a separate non-invasive prebuilt runtime tar |

## Performance reference

This is the only performance summary in the README. All rows use the same
70M-parameter model, exact 19x19, FP16, two inference streams, and physical
`launched_batches * batch / wall_time`. Both official backends were scanned
over B4-B32 on each listed GPU; the table reports only each scan winner's
confirmed value. Plan rows are current hardware certificates. Results should
be compared within one GPU model, not across hosts or models.

| Backend | GPU | Batch | Physical nnEval/s | Evidence |
| --- | --- | ---: | ---: | --- |
| Official CUDA baseline | RTX 4090 D | 13 | 1889.8 | [B4-B32 scan](records/rtx4090d-official-backend-baselines-20260811.md) |
| Official TensorRT baseline | RTX 4090 D | 12 | 2339.7 | [B4-B32 scan](records/rtx4090d-official-backend-baselines-20260811.md) |
| Checked-in CUDA plan | RTX 4090 D | 12 | 3110.7 | [plan certificate](plans/sm89/rtx4090d-b12-s2/README.md) |
| Official CUDA baseline | RTX 5080 | 9 | 1631.1 | [B4-B32 scan](records/rtx5080-official-backend-baselines-20260811.md) |
| Official TensorRT baseline | RTX 5080 | 17 | 2026.7 | [B4-B32 scan](records/rtx5080-official-backend-baselines-20260811.md) |
| Checked-in CUDA plan | RTX 5080 | 16 | 2836.2 | [plan certificate](plans/sm120/rtx5080-b16-s2/README.md) |

TensorRT is shown only as a comparison. It is not included in the environment,
build, runtime, or release artifacts.

## Plan-driven backend

The autotuner emits a schema-1 `cuda-tactic-plan` JSON file. A production plan
binds all of the following:

- architecture and receiver capability constraints;
- exact 19x19, FP16/NHWC, model SHA-256, batch, and streams per device;
- a self-contained override map for every selected implementation catalog;
- source, generated artifact, configuration, and measured binary hashes;
- discovery and stable long whole-graph evidence;
- closure of every retained positive-history implementation; and
- one immutable full-FP32 correctness certificate for the winning plan.

`cpp/neuralnet/cudatacticplan.cpp` loads the plan before evaluator creation.
It rejects an incompatible schema, model, board, precision, architecture,
batch, stream topology, device capability, or tactic. A planned implementation
cannot silently fall back to an official kernel.

Backend activation is recorded only after the selected implementation actually
launches. Every searchable implementation therefore closes four links:

1. backend implementation;
2. materialized scan candidate;
3. post-launch activation marker;
4. exact plan-apply mapping.

`python/cuda_tactic_history.py` is the retained positive-history contract. Plan
generation fails if any record lacks one of those links on any supported exact
batch.

### Catalogs and decision groups

The catalog names inventory implementations; they are not a claim that one
transformer block contains that many operators or that all tuning axes are
orthogonal. Both architectures expose ten ordered decision groups. A static
closure gate requires every shared runtime key and declarative dependency to
belong to exactly one group.

Bundles are measured inside their owning group. Later groups cannot rewrite an
earlier group's state. On SM120, packed QKV is an input-layout choice and does
not force a FlashAttention tile or accumulator mode.

The maintained union includes initial paths, pointwise activation paths, wide
projections, QKV and RoPE routes, FlashAttention, fused FFN and SwiGLU,
residual/projection GEMMs, normalization and head paths, persisting L2, and
model-weight sharing with a real cache-hit activation proof.

The optimized backend supports only exact 19x19 FP16/NHWC inference. There is
no mask tactic, dynamic-board compatibility path, B13 privilege, or old-option
compatibility layer.

## Autotune workflow

The outer orchestration, plan schema, history contract, measurement,
correctness, and packaging code are shared by SM89 and SM120. Hardware-specific
candidate generation remains separate only where the implementation requires
it.

The default flow is:

1. Detect the selected CUDA device. Compute capability 8.9 selects SM89 and
   compute capability 12.0 selects SM120.
2. Measure B4-B32 with a self-contained, artifact-free stable optimized graph.
3. Select the three highest-throughput batches.
4. For each selected batch independently, build its exact-shape artifacts and
   materialize the complete catalog space.
5. Run one activation-gated first pass in decision-group order while carrying
   a self-contained accumulated graph.
6. On the improved graph, rescan each catalog's first-pass top ten. Confirm
   provisional changes with longer ABBA measurements and repeat bounded
   refinement until unchanged.
7. Run a stable whole-graph long gate and rank batches by physical nnEval/s.
8. Replay only the fastest plan once against the 8192-row FP32 reference.
9. Emit `best-tactic-plan.json` only after history, activation, stability, and
   accuracy gates all pass.

This is a per-batch flow: one batch completes the whole decision process before
another batch is optimized. `--full-batch-scan` applies the full flow to every
B4-B32 shape and is deliberately disabled by default.

Discovery timings are pruning evidence, not release performance claims.

### Contention policy

Every benchmark subprocess is monitored with `nvidia-smi pmon`. A process that
only owns device memory and consumes zero SM time is allowed. If a foreign PID
uses nonzero SM time during measurement, the sample is invalidated. The
autotuner waits 30 seconds, confirms that the device is idle, and reruns it
instead of exiting on temporary contention. Failure to establish the monitoring
state is a measurement failure.

The distributed workflow does not contain GPU locks and does not change power
limits or clocks.

## Runtime pipeline

The frontend exposes two independent switches:

- `nnBatchAwareDispatch` controls request gathering and exact physical batch
  padding.
- `cudaAsyncInferPipeline` moves staging, H2D, D2H, and completion off the
  compute critical path with pinned memory, dedicated DMA streams, persistent
  workers, and CUDA events.

The plan loader requires batch-aware dispatch because exact-shape tactics are
unsafe without it. The asynchronous pipeline remains separately configurable
because it changes scheduling and memory lifetime rather than tactic choice.

```text
CPU fills pinned input
        |
        v
upload stream --inputReady--> compute stream --applyComplete--> download stream
       ^                            |                                  |
       |                            v                                  v
 inputConsumed permits       planned kernels                 outputReady wakes CPU
 host slot refill                                                 |
       ^                                                           v
       +---------------- outputConsumed permits in-place device reuse
```

The externally supplied output-consumed event prevents the backend from
dirtying its single device output slot before the previous result is consumed.
This removes the need for a ping-pong output allocation. H2D and D2H use copy
engines where supported and do not intentionally occupy SMs.

Each inference lane has its own persistent host worker. The scheduler can keep
one batch waiting on each lane while search produces another batch, instead of
waiting for one stream's complete host submission before feeding the next.

### Multi-GPU mapping

Plan stream count is per device. A two-stream plan uses two NN-server threads
on one GPU and four threads on two GPUs. Each pair must map to its receiver
device. Stream, event, copy, CUDA Graph, cuBLAS, buffer, and idle-state
operations first select that receiver device.

An idle GPU may authorize its own partial logical group only. Another device's
idle state cannot authorize an underfilled launch on a busy GPU.

## Running GTP

To use the certified plan carried by the source-complete tar without running
autotune, use:

```bash
./setup.sh
./build-for-plan.sh --device 0
./run.sh
```

`build-for-plan.sh` uses the CUDA product name as the unique registry key, then
validates the selected plan's receiver contract. It restricts generation to
the plan's one batch, selected tactics, and recursive artifact dependencies
before compiling KataGo.
It does not run batch prescan, candidate benchmarks, refinement, the long gate,
or accuracy replay. The resulting binary, artifact bundle, and restricted
space are content-hashed in `plan-build.json`; `run.sh` verifies that manifest
before accepting the locally compiled binary.

To search a new plan for the receiver instead, replace the build-only step
with `./run-autotune.sh --device 0`, then run `./run.sh` normally.

The launcher detects the receiver, selects a compatible certified plan, checks
the model, plan file, and measured binary hashes, and supplies exact batch,
two-lane device mapping, batch-aware dispatch, the asynchronous event pipeline,
and a search-thread budget. When a valid `plan-build.json` exists, it prefers
the locally compiled binary bound by that manifest; otherwise it uses the
measured binary hash recorded by the plan. Another rebuilt binary is accepted
automatically only when a retained result proves the same target, batch, and
complete apply mapping. Backend activation remains fail-closed. Select another
CUDA Runtime ordinal or override an input when needed:

```bash
./run.sh --device 1
./run.sh --model /data/model.bin.gz --config /data/gtp.cfg
```

Use `./run.sh --help` for explicit plan/binary overrides and GTP argument
forwarding. The loader still validates exact batch, full-board shape,
FP16/NHWC, maximum-batch-only warmup, and every tactic override.

A starting search-thread budget is:

```text
numSearchThreads = batch * (total inference lanes + 1) + C
```

`C` is a modest host/search long-tail allowance and must be measured for the
target CPU and playing-strength objective. `visits/s` must remain strictly
greater than real logical `nnEval/s`; fixed-shape backend comparisons use the
physical padded metric defined above.

See [RUNTIME.md](RUNTIME.md) for the compact receiver contract.

### CUDA Graph boundary

`cudaEventPipelineUseGraph=true` requires the async pipeline and fixed-batch
dispatcher. Input-ready and output-consumed events remain outside capture and
gate replay on their owning streams. CUDA Graph is an optional searched/runtime
mode, not an assumption of the plan format.

## Correctness gates

The immutable reference is generated through the official full-FP32 path with
both optimized architecture backends disabled. Metadata binds the binary,
model, corpus, row count, exact-batch behavior, and hashes.

Certification requires:

- exactly 8192 logical rows;
- byte-identical targets and all input sections between reference and
  candidate;
- matching model and corpus SHA-256;
- exact maximum-batch and fixed-tail-padding evidence;
- aggregate policy, value, score, ownership, and weighted-loss thresholds;
- per-request worst-case maximum-absolute and per-head RMSE thresholds.

`katago runnngtpstresstest` sends repeated search-shaped requests through the
ordinary evaluator scheduler, compares every output head on CPU, and stops on
the first error. It exercises the subject under test rather than replacing its
scheduler or backend.

## Reproducible distribution

Two non-invasive artifacts are produced:

1. The source-complete autotune SDK contains the KataGo source, pinned CPython,
   CUDA build toolkit, cuDNN, optimizer/library source trees, locked wheels,
   model, corpus, plans, patches, licenses, and SHA-256 manifests. The target
   does not clone GitHub or search for dependencies.
2. The prebuilt runtime tar contains the compiled CUDA backend, required
   user-space runtime libraries, plans, installer, licenses, and hashes. The
   receiver only needs a compatible NVIDIA driver.

Critical optimizer dependencies are built from the carried source. Small PyPI
dependencies are exact-version and hash locked. The setup detects supported
Ubuntu releases rather than hard-coding Ubuntu 24.04. Build parallelism is
memory-aware and conservative.

Development environment:

```bash
./final-migration/environment/setup.sh all
```

Source-complete autotune tar:

```bash
AUTOTUNE_CORPUS=/path/to/8192-full19.npz \
AUTOTUNE_CORPUS_MANIFEST=/path/to/8192-full19.manifest.json \
./final-migration/autotune/package-autotune.sh
```

After extraction into a persistent writable directory:

```bash
./setup.sh
./build-for-plan.sh --device 0
./run.sh
```

To generate a new plan instead of using the bundled one:

```bash
./setup.sh
./run-autotune.sh --device 0
./run.sh
```

Prebuilt runtime tar:

```bash
./final-migration/environment/setup.sh package
```

Every outer tar has an adjacent `.sha256`. The runtime installer also verifies
its internal manifest before installing into an isolated prefix.

## Repository map

- `cpp/neuralnet/cudatacticplan.*`: production plan loader and receiver checks.
- `cpp/neuralnet/cudabackend_sm89*`: maintained SM89 backend.
- `cpp/neuralnet/cudabackend_sm120*` and `sm120_aot/`: maintained SM120 backend.
- `cpp/neuralnet/nneval.*`: batch-aware dispatcher and asynchronous scheduler.
- `python/cuda_tactic_workflow.py`: unified architecture-aware scanner.
- `python/cuda_tactic_history.py`: retained positive-history contract.
- `final-migration/autotune/`: offline SDK entry points and specification.
- `final-migration/environment/`: environment and runtime packaging.
- `final-migration/plans/`: one current production plan per qualified GPU model.
- `final-migration/records/`: detailed build, experiment, and qualification logs.
- `docs/cuda-tactic-workflow.md`: detailed search contract.

Only the CUDA backend and exact 19x19 FP16/NHWC path are in scope. General
KataGo and GTP behavior remains documented by the unchanged upstream files.
