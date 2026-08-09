# KataGo plan-driven CUDA fork

[中文](README.zh-CN.md) | English

This repository is a clean fork of official
[`lightvector/KataGo`](https://github.com/lightvector/KataGo), built from
official `master` commit `6a1fc5de9fc253723ac475a0683bf0b9d9b7bd19`
(`v1.17.2`, fetched on 2026-08-07). It adds a shape-specialized, plan-driven
CUDA inference system for NVIDIA SM89 and SM120 while retaining KataGo's GTP,
analysis, search, model, and game logic.

This is a CUDA-backend project. TensorRT is neither required nor used by the
optimized path. The checked-in changes are not claimed to be part of or
supported by upstream KataGo.

## What this fork adds

| Area | Official fork point | This fork |
| --- | --- | --- |
| Kernel choice | backend defaults and ordinary library heuristics | explicit, versioned tactic plan selected by an offline whole-graph scanner |
| Batch shape | current request count may reach inference directly | exact physical batch; full batch launches immediately, partial batch only when its GPU is idle, with tail padding |
| Host submission | evaluator threads can serialize submission and completion | one persistent host worker per inference lane, with concurrent nonblocking submission |
| Transfers | preprocessing, H2D, compute, D2H, and postprocessing share more of the critical path | pinned staging plus dedicated upload/download streams and CUDA events |
| Buffer reuse | completion is normally synchronized before reuse | single device slot protected by input-consumed and output-consumed events; no ping-pong allocation is required |
| CUDA streams | some custom paths historically used implicit/per-thread streams | every optimized launch receives and uses the owning NN-server stream |
| Multi-GPU | evaluator threads are assigned to devices | stream, event, cuBLAS handle, buffer, and idleness ownership are receiver-device-local and fail closed |
| CUDA Graph | no plan-level exact-shape contract | optional SM89 exact-shape replay, with external readiness/consumption events outside capture |
| Correctness | normal KataGo backend tests | immutable 8192-row full-FP32 reference gate, input identity checks, exact-batch tail checks, and GTP-shaped stress harness |
| Distribution | ordinary source/build workflow | reproducible source-complete autotune tar and a separate non-invasive prebuilt-runtime tar |

The goal is not a collection of special cases for one batch. The scanner
searches the same complete tactic families at every exact batch in B4-B32. The
union of all historically positive, numerically valid SM89 and SM120 work is
the maintained search space.

## Current qualification status

| Architecture | Implemented search space | Production plan | Hardware qualification |
| --- | --- | --- | --- |
| SM89 | 20 families, 62 positive-history records, 3738 candidates across exact B4-B32 | checked-in RTX 4090 D B12/S2 plan | complete: GTP load, one/two GPU, 8192-row all-head FP32 replay |
| SM120 | 23 families, 64 positive-history records, 4234 candidates across exact B4-B32 | pending a fresh unified scan | backend/static closure complete; production performance and GTP qualification pending |

The previous RTX 5080 B18 result (`2586.579` physical nnEval/s) came from an
incomplete pre-unification space. Its old plan is intentionally rejected and
is not shipped as a production plan.

The checked-in SM89 plan is:

```text
final-migration/plans/sm89/rtx4090d-b12-s2/best-tactic-plan.json
```

Its file SHA-256 is
`57aba0d9f5ff009f0103fe792766bd3fe065d156c13396cb99bc40b5488f9edb`.
The long whole-graph gate measured `3026.196859` physical nnEval/s. Subsequent
ordinary evaluator scheduling verified 8192/8192 requests at 3035.87 physical
nnEval/s on one RTX 4090 D and 6072.97 on two RTX 4090 D devices. These values
are evidence for the tested host, clocks, model, batch, and topology, not a
universal performance guarantee.

## Plan-driven backend

The autotuner emits schema-1 `cuda-tactic-plan` JSON. A production plan binds:

- architecture and receiver capability constraints;
- exact 19x19 board, FP16/NHWC precision, model SHA-256, batch, and streams per
  device;
- a self-contained override map for every selected family;
- source, generated-artifact, configuration, and binary hashes;
- discovery and long whole-graph measurements;
- positive-history closure; and
- the selected plan's full-FP32 correctness certificate.

`cpp/neuralnet/cudatacticplan.cpp` loads the plan before evaluator
construction. It rejects a wrong schema, incomplete status, missing history
link, wrong model, board, precision, architecture, batch, stream topology, or
receiver capability. An unsupported tactic is a startup error. It never
silently falls back to an official kernel while claiming that the planned
tactic ran.

The backend reports activation only after the selected implementation actually
launches. Scanner candidates therefore need four links:

1. a backend implementation;
2. a materialized scan candidate;
3. a post-launch activation marker; and
4. an exact plan-apply mapping.

`python/cuda_tactic_history.py` records the historically positive contract.
Plan generation is blocked unless every record closes all four links for all
supported exact batches.

### Maintained tactic families

The exact families differ where the architecture really differs, but the
outer workflow is shared. They cover, among other components:

- exact-mask preprocessing and downstream mask elision;
- initial convolution/global paths and pointwise BN/activation paths;
- wide QKV/FFN/head projections and projection bundles;
- fused QKV + RoPE and packed QKV + RoPE routes;
- FlashAttention accumulation modes and tile/warp variants;
- fused dual FFN + SwiGLU implementations from CUTLASS, CuTe, and TileLang;
- residual GEMM, linear2, output/pre/post-convolution projections;
- RMSNorm, head BN, post-convolution BN + SiLU, and value terminal paths;
- persisting-L2 placement and model-weight sharing when a real cache hit is
  observed.

There is no B13 runtime privilege and no compatibility alias layer for old
experimental option names. Obsolete B13-only generated objects and redundant
SM120 search scripts were removed.

## Unified autotuner

The offline SDK automatically detects the selected device through the CUDA
Runtime:

- compute capability 8.9 selects SM89;
- compute capability 12.0 selects SM120.

The outer orchestration, plan schema, history contract, build, measurement,
accuracy, and packaging code are shared. Only architecture-specific candidate
generation and backend source remain separate when the hardware requires it.

The default domain is exact B4-B32 with two inference streams per device. Each
batch completes its own full decision flow; the scanner does not make one
decision across all batches and then move to the next decision. This prevents
a tactic that is profitable only at a particular shape from becoming a hidden
fixed-batch special case.

For each batch, the workflow materializes the complete family space, starts
from an explicit official-equivalent baseline, performs activation-gated
discovery, accumulates a self-contained configuration, runs the final joint
whole-graph state, and records its result. After all 29 batches complete, the
long gate ranks stable whole-graph throughput. Only the highest-throughput
stable plan is replayed once against the 8192-row FP32 reference and emitted as
`best-tactic-plan.json`.

Discovery timings are not release performance claims. A plan becomes
production-ready only after the long whole-graph gate and accuracy gate pass.

### GPU contention policy

The distributed project contains no `gpu-lock` wrapper. That tool was useful
only for coordinating local development sessions.

Instead, every benchmark subprocess is monitored with `nvidia-smi pmon`.
A process that only owns device memory with zero SM activity is allowed. If an
external PID begins consuming nonzero SM time during a measurement, the
benchmark process group is stopped and the sample is invalidated. If the
monitor cannot establish the state, measurement fails closed.

## Batch-aware frontend and asynchronous pipeline

The frontend has two independent opt-in features:

- `nnBatchAwareDispatch`: waits for a complete batch, except that an entirely
  idle target GPU may accept an underfilled request group; the physical launch
  is still padded to the plan batch.
- `cudaAsyncInferPipeline`: moves preprocessing/staging, H2D, D2H, and output
  completion off the compute critical path with pinned host memory, dedicated
  DMA streams, persistent submission workers, and CUDA events.

The plan loader forces the first feature because an exact-shape tactic cannot
be used safely without it. The async pipeline remains a separate switch, as it
changes host scheduling and memory lifetime rather than tactic selection.

The single-slot event protocol is:

```text
CPU fills pinned input
        |
        v
upload stream --inputReady--> compute stream --applyComplete--> download stream
       ^                            |                                  |
       |                            v                                  v
 inputConsumed allows        planned kernels                 outputReady wakes CPU
 host slot refill                                                 |
       ^                                                           v
       +---------------- outputConsumed permits in-place device reuse
```

The output-consumed event is supplied externally to the backend. The compute
stream waits for it before dirtying the single device output slot, eliminating
the need for ping-pong buffers without creating an output race. H2D and D2H use
copy engines when supported and do not intentionally occupy SMs.

Each inference lane has its own persistent host worker. The central scheduler
may have one batch waiting on each lane and one additional batch being produced
by search, avoiding the old pattern where the scheduler waited for a stream's
entire host submission before feeding the next stream.

## Multi-GPU isolation

Plans specify streams per device, not a global stream total. With a two-stream
plan, one GPU uses two NN-server threads and two GPUs use four. Each pair is
assigned to its receiver device in the GTP config.

Every stream/event/copy/graph/cuBLAS operation first selects the handle's owning
device. GPU idleness is tracked per physical device, and a partial request on
one GPU cannot be authorized because another GPU is idle. The 8192-row dual-GPU
certificate distributed requests across all four lanes.

## Running GTP with a checked-in plan

The plan is bound to model SHA-256
`1881600caab9e9d85a3dd6a019e9b8e7d2c237b5f984e13ed49a8645be3077c6`.
Use an absolute plan path in the normal GTP config.

Single GPU, two streams:

```cfg
cudaTacticPlanFile = /path/to/final-migration/plans/sm89/rtx4090d-b12-s2/best-tactic-plan.json
cudaTacticPlanBatch = 12

numNNServerThreadsPerModel = 2
cudaDeviceToUseThread0 = 0
cudaDeviceToUseThread1 = 0

cudaAsyncInferPipeline = true
cudaEventPipelineUseGraph = false
```

Two GPUs, two streams per GPU:

```cfg
numNNServerThreadsPerModel = 4
cudaDeviceToUseThread0 = 0
cudaDeviceToUseThread1 = 0
cudaDeviceToUseThread2 = 1
cudaDeviceToUseThread3 = 1
```

The loader supplies and verifies exact B12, exact 19x19, FP16/NHWC,
`nnBatchAwareDispatch=true`, maximum-batch-only warmup, and every planned CUDA
override. A conflicting user value is rejected.

A useful initial search-thread budget is:

```text
numSearchThreads = batch * (total inference lanes + 1) + C
```

`C` is only a modest CPU/search long-tail allowance, commonly 12-32, and must
be tuned for the host and search-strength objective. `visits/s` and neural
evaluations are different: visits must remain strictly greater than real
nnEval. Performance comparisons for fixed shapes use physical
`launched_batches * exact_batch / wall_time`, including padding.

See [RUNTIME.md](RUNTIME.md) for the compact runtime contract.

## CUDA Graph boundary

`cudaEventPipelineUseGraph=true` requires the async pipeline and fixed-batch
dispatcher. Input-ready and output-consumed events remain outside capture and
gate replay on the owning streams. This avoids attempting to capture a changing
external event dependency into the graph.

Graph replay is functionally correct on SM89. On the certified RTX 4090 D host,
eager submission was about 0.8% faster, so eager is currently recommended.
SM120 graph replay is not yet a certified production path.

## Correctness guardrails

Release accuracy uses an immutable offline full-FP32 output generated with both
custom SM89 and SM120 backends disabled. Reference metadata binds the binary,
model, input corpus, row count, exact batch behavior, and hashes.

The comparator requires:

- exactly 8192 logical rows;
- byte-identical targets and all input sections between reference and
  candidate;
- the expected model and corpus SHA-256;
- the candidate's exact maximum batch and fixed-tail-padding metadata;
- policy top-1 and probability, value, score, ownership, and weighted-loss
  thresholds.

The physical tail is filled by repeating real requests, but only the original
8192 logical rows are serialized. This keeps exact-batch AOT kernels active on
the tail without hiding request reordering or count errors.

`katago runnngtpstresstest` sends prepared search-shaped requests through the
ordinary evaluator scheduler, checks every output head on CPU against the
offline FP32 results, and stops on the first error. Repeating the corpus provides
a stability and multi-GPU guard without replacing the subject under test.

## Reproducible environment and offline artifacts

There are two release artifacts:

1. The autotune SDK carries the complete source tree, pinned CPython 3.12.13,
   CUDA 13.2 build toolkit, cuDNN 9.25, source trees for CUTLASS/CuTe,
   FlashAttention, TileLang, Triton, Quack, TVM-FFI and zlib, pinned wheels,
   model, 8192-row corpus, plans, patches, and SHA-256 manifests. A target does
   not clone GitHub, use a package index, run APT, or search for dependencies.
2. The prebuilt runtime tar carries the compiled KataGo CUDA backend, its
   user-space CUDA/cuDNN/C++/glibc runtime, plans, installer, and hashes. It is
   installed into one isolated prefix and only requires a compatible NVIDIA
   driver.

Release construction resolves current upstream source revisions; those exact
revisions are then frozen into the tar. Critical optimizer dependencies are
built from the carried source in the private Python environment. Small PyPI
dependencies are exact-version and hash locked. The release is not hard-coded
to Ubuntu 24.04; the source setup detects supported Ubuntu versions, while the
offline SDK targets Linux x86-64 with glibc 2.28 or newer.

Build parallelism is memory-aware. It never blindly uses `-j$(nproc)` and does
not encode fixed `-j4`/`-j8` defaults; an explicit job count remains available.

### Build a development environment

From the repository root:

```bash
./final-migration/environment/setup.sh all
```

### Build the source-complete autotune tar

```bash
AUTOTUNE_CORPUS=/path/to/8192-full19.npz \
AUTOTUNE_CORPUS_MANIFEST=/path/to/8192-full19.manifest.json \
./final-migration/autotune/package-autotune.sh
```

After extraction into a persistent writable directory:

```bash
./setup.sh
./run-autotune.sh --device 0
```

### Build the precompiled inference tar

```bash
./final-migration/environment/setup.sh package
```

Every outer tar receives an adjacent `.sha256`; the prebuilt runtime also
receives a hash-checked non-invasive installer.

## Repository map

- `cpp/neuralnet/cudatacticplan.*`: production plan loader and receiver checks.
- `cpp/neuralnet/cudabackend_sm89*`: maintained SM89 backend.
- `cpp/neuralnet/cudabackend_sm120*` and `sm120_aot/`: maintained SM120 backend.
- `cpp/neuralnet/nneval.*`: batch-aware dispatcher and async event scheduler.
- `cpp/tests/testnnbatchingdispatcher.cpp`: scheduler state-machine tests.
- `cpp/tests/testnngtpharness.cpp`: full-output GTP-shaped correctness harness.
- `python/cuda_tactic_workflow.py`: unified architecture-aware scanner.
- `python/cuda_tactic_history.py`: positive-history four-link contract.
- `final-migration/autotune/`: offline SDK build and execution entry points.
- `final-migration/environment/`: development and prebuilt-runtime packaging.
- `final-migration/plans/`: checked-in production plans.
- `final-migration/records/`: compact qualification records.

The complete optimization-history audit is in
[OPTIMIZATION_HISTORY_AUDIT_20260808.md](OPTIMIZATION_HISTORY_AUDIT_20260808.md).
The SM89 runtime certificate is in
[records/plan-runtime-sm89-20260809.md](records/plan-runtime-sm89-20260809.md).

## Known boundaries

- Only the CUDA backend is optimized by this work; TensorRT is not required.
- Production plans currently require exact 19x19 and the bound model hash.
- The checked-in production plan is SM89/RTX 4090 D B12/S2. Receiver checks
  deliberately reject incompatible devices or models.
- SM120 implementation and scanning are present, but a fresh unified hardware
  scan and best-plan certificate are still required.
- CUDA Graph is optional and currently not the fastest certified SM89 mode.
- High search-thread counts may reduce playing strength in short/fixed-visit
  searches even when they increase GPU utilization.

For upstream KataGo behavior and general GTP documentation, continue to use the
unchanged top-level `README.md` and official documentation.
