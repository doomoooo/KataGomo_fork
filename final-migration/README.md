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

The goal is not a collection of special cases for one batch. Every exact batch
in B4-B32 can materialize the same complete implementation catalogs. The
default workflow first ranks B4-B32 with a stable artifact-free optimized graph, then runs
the complete tactic flow for the three fastest shapes. `--full-batch-scan`
retains the exhaustive 29-batch mode. The union of all historically positive,
numerically valid SM89 and SM120 work is the maintained search space.

## Current qualification status

| Architecture | Implemented search space | Production plan | Hardware qualification |
| --- | --- | --- | --- |
| SM89 | 19 implementation catalogs in 10 decision groups; 60 positive-history records; 3564 full-domain candidates | certified RTX 4090 D B12/S2 plan | long gate and 8192-row full-FP32 gate passed |
| SM120 | 19 implementation catalogs in 10 decision groups; 63 positive-history records; 3944 full-domain candidates | RTX 5080 B19/S2 plan being refreshed | previous search/gate/accuracy evidence retained; full-board plan refresh pending |

The previous RTX 5080 B18 result (`2586.579` physical nnEval/s) came from an
incomplete pre-unification space. A later coupling-audited B19 graph reached
`2838.9148995` physical nnEval/s, but its plan predates the fixed-full-board
contract and is intentionally not shipped.

The old official-fallback prescan ranked B7/B8/B9 on this RTX 5080 host and
missed the optimized B19 peak. It has been replaced by an explicit,
artifact-free optimized graph that is much closer to the final operating
state. Use `--full-batch-scan` when global B4-B32 coverage matters more than
the roughly 88% reduction in candidate evaluations provided by top-three mode.

The current checked-in production plan is:

```text
final-migration/plans/sm89/rtx4090d-b12-s2/best-tactic-plan.json
```

Its SHA-256 is
`e089cd8ef2ca65eadacb4e5014f39622de00c50bd99df22424612e31ce795a94`.
The RTX 4090 D B12/S2 long gate measured `3110.690824` physical nnEval/s from
samples `3110.484420` and `3110.897228`; its single 8192-row all-head replay
passed every aggregate and per-request full-FP32 threshold. The old SM120 file
remains excluded until the RTX 5080 is available for the same current-source
certification.

The stable optimized B4-B32 prescan on that RTX 4090 D selected B12, B13, and
B14 for full search. Its B12 sample measured `3079.829482` physical nnEval/s;
every prescan sample recorded an empty foreign-SM PID set. A normal GTP startup
then loaded the checked-in plan on two NN-server lanes, observed every planned
post-launch activation marker on both lanes, started fixed-B12 dispatch and the
event-gated single-slot scheduler, and completed basic 19x19 GTP commands.
A 64-thread search benchmark over ten 800-visit positions measured `3075.81`
visits/s, `2973.42` logical nnEvals/s, and `255.46` launched batches/s at an
average logical batch of `11.64`. Its fixed-shape physical rate was therefore
`3065.52` nnEval/s, 1.45% below the compute-only long gate; `nvidia-smi pmon`
recorded no foreign PID with nonzero SM activity during the run.

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

### Implementation catalogs and decision groups

The 19 catalog names are not 19 operators in one trunk block and are not
claimed to be 19 independent performance dimensions. They inventory backend
implementations. The scanner exposes 10 ordered groups on both architectures.
A static closure gate requires every shared runtime key and
every declarative dependency to remain inside exactly one group. A bundle is
measured first inside its group; later catalog stages may explicitly refine
only their own keys. No later group can rewrite an earlier group's state.
On SM120, packed QKV is an input-layout choice and is explicitly independent
of FA's QK/PV accumulation precision; packed routes consume the selected FA
tactic without forcing a tile or accumulator mode.

The catalogs differ only where the architecture really differs. They cover,
among other components:

- initial convolution/global paths and pointwise BN/activation paths;
- wide QKV/FFN/head projections and projection bundles;
- fused QKV + RoPE and packed QKV + RoPE routes;
- FlashAttention accumulation modes and tile/warp variants;
- fused dual FFN + SwiGLU implementations from CUTLASS, CuTe, and TileLang;
- residual GEMM, linear2, output/pre/post-convolution projections;
- RMSNorm, head BN, post-convolution BN + SiLU, and value terminal paths;
- persisting-L2 placement and model-weight sharing when a real cache hit is
  observed.

The optimized CUDA backends accept only exact 19x19 FP16 NHWC inference.
Full-board execution is a fail-closed backend contract, not a configurable or
searchable component; no mask tactic, runtime key, candidate, or plan mapping
exists.

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

The default selection domain is exact B4-B32 with two inference streams per
device. An artifact-free stable optimized graph measures all 29 shapes and selects the
three highest-throughput batches. Each selected batch then completes its own
full decision flow; the scanner does not make one decision across all batches
and then move to the next decision. This prevents a tactic that is profitable
only at a particular shape from becoming a hidden fixed-batch special case.

For each selected batch, the workflow materializes the complete catalog space,
starts from an explicit self-contained baseline, performs
activation-gated discovery in decision-group order, accumulates a
self-contained configuration, runs the final joint whole-graph state, and
records its result. The long gate ranks those three stable whole-graph results.
Only the fastest stable plan is replayed once against the 8192-row FP32
reference and emitted as `best-tactic-plan.json`. Pass
`--full-batch-scan` to apply the same full flow to all 29 batches; this is
deliberately default-off because it costs roughly eight to nine times as much.

Discovery timings are not release performance claims. A plan becomes
production-ready only after the long whole-graph gate and accuracy gate pass.

### GPU contention policy

Every benchmark subprocess is monitored with `nvidia-smi pmon`.
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

The RTX 5080 plan is temporarily absent while current-source B19 certification
is pending. Do not reuse the pre-full-board SM120 plan from Git history.

Two GPUs, two streams per GPU:

```cfg
numNNServerThreadsPerModel = 4
cudaDeviceToUseThread0 = 0
cudaDeviceToUseThread1 = 0
cudaDeviceToUseThread2 = 1
cudaDeviceToUseThread3 = 1
```

The loader supplies and verifies the plan's exact batch (B12 in the checked-in
SM89 example), exact 19x19, FP16/NHWC,
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
- the same worst-per-request max-absolute and per-head RMSE limits used by the
  GTP-shaped CPU verifier, in addition to aggregate 8192-row metrics;
- policy top-1 and probability, value, score, ownership, and weighted-loss
  thresholds.

The per-request value-probability policy allows max-absolute `0.06` and
max-RMSE `0.05`; all other per-request and aggregate limits remain unchanged.
This admits the historically fastest SM120 TN64/both16 FA coordinate, whose
observed worst request was max-absolute `0.0559005` and max-RMSE `0.0456366`.
The coordinate remains subject to the complete 8192-row certification rather
than receiving a tactic-specific exception.

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
- The only currently checked-in production plan is SM89/RTX 4090 D B12/S2.
  The registry permits one current plan per GPU model. Receiver checks
  deliberately reject incompatible devices or models.
- The RTX 5080 current-source B19 plan, long gate, FP32 certificate, and GTP
  qualification remain pending until that host is available again.
- CUDA Graph is optional and currently not the fastest certified SM89 mode.
- High search-thread counts may reduce playing strength in short/fixed-visit
  searches even when they increase GPU utilization.

For upstream KataGo behavior and general GTP documentation, continue to use the
unchanged top-level `README.md` and official documentation.
