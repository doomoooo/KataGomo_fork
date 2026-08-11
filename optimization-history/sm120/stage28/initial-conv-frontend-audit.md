# RTX 5090 D B13/19x19 initial-conv cuDNN frontend audit

Date: 2026-08-06

## Scope and conclusion

This is a read-only implementation audit. No CUDA/C++ source was changed and no
GPU command was run.

The fixed-shape initial convolution is a valid remaining experiment, but the
5080 result does **not** justify hard-coding engine 45 as the 5090 D winner. The
recommended implementation is:

1. Add an exact B13/19x19-only cuDNN frontend path to the official `ConvLayer`.
2. Enumerate and time plans in an isolated diagnostic run, recording the complete
   engine/knob identity. Include 5080's engine 45, tile 0, stages 2 as one explicit
   candidate.
3. Compile/configure the chosen plan as a fixed runtime plan and compare a small
   finalist set under the real S2 network schedule.
4. Fall back to the existing legacy `IMPLICIT_PRECOMP_GEMM` path for every other
   shape, unsupported plan, or construction failure.

The current 5090 D profile bounds the likely benefit: the initial convolution is
ordinal 5, with isolated median 19.569 us and S2 median 21.697 us. Across 60 S2
forwards it contributes 1.305 ms total and only 0.131 ms measured concurrency
excess. It runs once per forward, so the old 5080 whole-network gain of 0.461%
must be treated as evidence to test, not as the expected gain. A realistic prior
for 5090 D is roughly 0.1-0.3%, subject to S2 phase effects.

At the observed roughly 3.67-3.78k nn/s and B13, a two-stream per-server forward
is about 6.9-7.1 ms. The current 21.697 us convolution is only about 0.31% of
that latency; even deleting it entirely would not provide a large linear gain.
This is a scale estimate, not a hard bound, because changing the kernel can also
change cross-stream phase and overlap.

## Evidence reconstruction

| Platform / source | Fixed workload | Plan or path | Local kernel evidence | Whole-network evidence | Interpretation |
| --- | --- | --- | ---: | ---: | --- |
| RTX 5080 history | B19/S2, 19x19, CUDA 13.2, cuDNN 9.24 | frontend engine 45, tile 0, stages 2 | 36.417 us | 2835.976 -> 2849.036 nn/s, +0.461%, bit-exact | Accepted historical result, but different GPU and batch |
| RTX 4090 stage24 | B13/S2, 19x19, cuDNN 9.25 | engine 45, tile 0, stages 2, tag `eng45_k14=2_k2=0`, workspace 557056 B | NCU 31.200 -> 22.048 us; Nsys forward 33.041 -> 24.328 us, reverse 32.533 -> 23.867 us | pooled 3251.925 -> 3257.140 nn/s, +0.160%; all four adjacent pairs positive; 8192 replay byte-identical | Independent confirmation that the exact-shape frontend route can survive S2 validation |
| RTX 5090 D stage24/27 current | B13/S2, 19x19, CUDA 13.2, cuDNN 9.25 | legacy cuDNN forced `IMPLICIT_PRECOMP_GEMM` | isolated 19.569 us; S2 21.697 us | 1.305 ms total/60 calls, 0.131 ms excess | Baseline is already much faster; remaining end-to-end headroom is small |

The current Nsys symbol is unresolved as the generic `Kernel`. Its exact observed
ordinal signature is
`cudnn|Kernel|g296x3x1|b128x1x1|r94|s81920`: grid 296x3x1, block 128x1x1,
94 registers/thread, and 81,920 bytes shared memory. The code-level tactic is
unambiguous despite the generic symbol: the legacy `ConvLayer` selects
`CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM` for this FP16/dilation-1
case. Use both ordinal/NVTX identity and the launch signature when checking a
replacement; the short kernel name alone is not sufficient.

The 5080 raw artifacts referenced by `/workspace/cuda-optimization-history.md`
live under `/data/wangyize/...` and are not mounted here. Therefore the 5080 row
can be reconstructed only from the retained history summary. The engine number
alone is insufficient provenance and is not portable across GPU architecture,
cuDNN release, graph shape, or plan-builder behavior.

Useful local corroborating artifacts:

- `/workspace/results/4090/stage24/hypothesis-initial-conv-frontend.md`
- `/workspace/results/4090/stage24/final-decision-summary.json`
- `/workspace/results/4090/stage24/cudnn-frontend-plan.log`
- `/workspace/katago-4090/cpp/neuralnet/cudabackend_sm89_forward.cpp`
- `/workspace/results/rebuild/stage27/current-s2-ordinal-attribution.json`

## Exact graph contract

Only this graph is in scope. There is no mask input or dynamic spatial shape.

| Property | Required value |
| --- | --- |
| Batch / board | B13, 19x19 exactly |
| Input X | FP16 NHWC logical `[13,22,19,19]`, strides `[7942,1,418,22]` |
| Filter W | FP16 NHWC logical `[768,22,3,3]`, strides `[198,1,66,22]` |
| Output Y | FP16 NHWC logical `[13,768,19,19]`, strides `[277248,1,14592,768]` |
| Convolution | cross-correlation, pad `[1,1]`, stride `[1,1]`, dilation `[1,1]` |
| Arithmetic | FP32 compute/intermediate, FP16 I/O |
| Output semantics | `accumulate == false`; no fused bias in this experiment |
| Device gate | SM120 / RTX 5090 D |

The frontend path must reject any mismatch in batch, spatial dimensions,
channels, precision, layout, filter layout, convolution parameters, or
accumulation mode. It must not generalize to other convolutions merely because
their spatial shape happens to be 19x19.

## Current implementation gap

The official implementation in
`/workspace/katago/cpp/neuralnet/cudabackend.cpp` constructs legacy cuDNN tensor,
filter, and convolution descriptors in `ConvLayer`. For non-1x1 FP16 convolutions
with dilation 1 it forcibly selects
`CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM`, then executes
`cudnnConvolutionForward`. The initial convolution is constructed as an ordinary
`ConvLayer` and applied once with `accumulate=false`.

`cudaUseInitialConvFrontend` is parsed in
`/workspace/katago/cpp/neuralnet/cudabackend_sm120.cpp` with default `true`, and
is declared in the SM120 option structure, but it has no runtime consumer. The
current `CudaHandles` hook set contains no convolution hook. There is also an
ordering issue: the official `Model` and its `ConvLayer` objects are constructed
before `Sm120Model` installs its hooks. A frontend plan cannot simply be appended
to the existing hook installation without changing that lifetime.

The current build is suitable for the experiment: Release, CUDA 13.2.86,
architectures `89;120`, cuDNN headers/runtime 9.25.0, and vendored
`cudnn_frontend` 1.24.0. The frontend graph API exposes heuristic A/B/fallback
plan creation, explicit engine/knob creation, per-plan name/workspace queries,
support checking, plan building, autotuning, and indexed execution.

## Recommended implementation

### 1. Ownership and construction

Put optional frontend state directly in `ConvLayer`, which already owns the
transformed filter and participates in convolution workspace sizing. Add an
explicit constructor property such as `isInitialConv`, defaulting to false, and
set it only at the initial-convolution construction site. Plumb the enable/mode
option into the ordinary model before `ConvLayer` construction; the later SM120
hook installation point is too late.

Plan and graph objects must be owned per NN server/cuDNN handle. Do not share a
mutable execution-plan object across the two S2 handles or streams unless cuDNN
explicitly guarantees that use and it has been tested.

For an accepted runtime build, change the option default to false until a 5090 D
plan passes the gates below. The current parsed-but-unused default true is
misleading.

### 2. Low-compute plan search

Use the one exact graph above. Search in S1 or a dedicated initial-convolution
probe, never while both production streams are concurrently autotuning.

Candidate sources:

- Explicit historical candidate: engine 45 with `TILE_SIZE=0`, `STAGES=2`.
- cuDNN heuristic modes A and B.
- cuDNN fallback heuristic, retained mainly as a robustness reference.

Reject unsupported plans and plans requiring more than 64 MiB workspace. Build
all remaining plans before timing. Warm up every plan sufficiently to exclude
module loading, JIT, graph validation, and plan construction. Time round-robin or
randomized repetitions on the actual CUDA stream and device buffers. Retain the
best three materially distinct plans rather than only the isolated winner.

The diagnostic search may use `BuildPlanPolicy_t::ALL` and `autotune`, but the
production S2 path must execute a fixed recorded plan. Final selection is based
on controlled S2 whole-network ABBA, because isolated convolution latency cannot
predict resource overlap or phase changes.

### 3. Runtime and workspace

At construction:

1. Check the complete exact-shape gate.
2. Build only the selected fixed engine/knob plan.
3. Validate support, build the plan, query its workspace, and enforce the cap.
4. On any failure, record the exact failing phase and retain the legacy path.

`requiredWorkspaceBytes()` must return at least the maximum of the legacy and
frontend requirements. A diagnostic build that can execute several plans must
reserve the maximum workspace of its retained candidates. The 4090 prototype
used a separate 64 MiB convolution workspace; blindly copying that allocation
would add unnecessary permanent memory if the official shared workspace can
safely satisfy both paths.

At execution, require B13 and `accumulate=false` again. A build/support failure
or any non-target invocation falls back to legacy. An execute failure during an
explicit warmup/qualification phase may disable the candidate once and log the
fallback. An execute failure after timed measurement begins should be fatal,
because silently mixing frontend and legacy samples invalidates the benchmark.

Do not fuse bias in this change. The 5080 history records a correct frontend-bias
route but no accepted whole-network performance result.

## Required plan record

Write one JSON record per attempted plan and retain it with the benchmark
artifact. At minimum record:

| Group | Fields |
| --- | --- |
| Environment | UTC timestamp, git commit and dirty state, binary SHA256, hostname, GPU name/UUID/SM, driver, CUDA toolkit/runtime, cuDNN header/runtime, cudnn_frontend version |
| Workload | B, H, W, C-in/out, filter, padding, stride, dilation, tensor data types, compute type, dimensions, strides, tensor UIDs, layout, accumulate flag |
| Discovery | source (`explicit45`, `heurA`, `heurB`, `fallback`), enumeration index, plan name/tag, numerical and behavior notes |
| Engine | engine global index, every knob type/value pair, kernel-cache state where available |
| Build | validate/build-op/create/check-support/build-plan status and full error text |
| Resources | workspace bytes and cap, emitted kernel name/signature, grid/block, registers, static/dynamic shared memory where profiler data is available |
| Timing | warmup count, timed count, order/random seed, per-run samples, median, p10/p90, graph-level time and kernel time |
| Runtime | selected plan ID, fixed-plan build result for each NN server, execution count, fallback count and reason |

The startup log should print one concise line per S2 server containing the
selected plan tag, engine, knobs, workspace, and whether fixed-plan construction
succeeded. Avoid enabling the cuDNN library-wide debug logger: the 4090 work saw
a crash during concurrent construction. A narrow application-owned logger around
this graph is sufficient.

## Fallback matrix

| Condition | Required behavior |
| --- | --- |
| Option disabled | Legacy `IMPLICIT_PRECOMP_GEMM` |
| Any shape/layout/type/conv mismatch | Legacy, without attempting frontend construction |
| Runtime batch other than 13 | Legacy |
| `accumulate=true` | Legacy |
| Explicit engine/knob rejected by current cuDNN | Log construction phase/status; legacy |
| Workspace exceeds cap or supplied buffer is too small | Log required/available bytes; legacy before timing |
| Execute failure during qualification | Log once, disable candidate for that server, legacy |
| Execute failure after measurement starts | Abort the run; do not silently mix paths |

## Verification protocol

The following commands are a protocol for the implementation phase. They were
not executed by this audit. All GPU work must retain the project lock wrapper.
Assume:

```bash
ROOT=/workspace
BIN=/workspace/katago/build-cuda/katago
CFG=/workspace/bench-cuda-gpu2-5090d-s2.cfg
MODEL=/workspace/models/b11c768h12nbt3tflrs-fson-silu.bin.gz
OUT=/workspace/results/rebuild/stage28/initial-conv
```

First record the build and environment, then run the isolated plan probe. The
exact probe command depends on the implemented diagnostic entry point; it should
produce the JSON schema above and must use B13/19x19:

```bash
source /workspace/container-setup/nvidia-env.sh
gpu-lock with --gpu 2 -- "$BIN" testinitialconv \
  -model "$MODEL" -batch-size 13 -boardsize 19 \
  -plan-search explicit45,heurA,heurB,fallback \
  -workspace-cap-mib 64 -output "$OUT/plans.json"
```

If extending `testEvaluateConv` instead of adding `testinitialconv`, preserve the
same exact graph, logging, warmup, and plan-selection semantics. Do not identify
the initial convolution by the generic Nsys/NCU name `Kernel`; later CUTLASS
kernels can share that name. Use an NVTX range, exact graph ordinal, or an
isolated convolution probe.

Profile each fixed finalist in S1 to verify plan identity and exclude build/JIT
from the measured region:

```bash
source /workspace/container-setup/nvidia-env.sh
gpu-lock with --profile --gpu 2 -- env DEBUGINFOD_URLS= DEBUGINFOD_TIMEOUT=1 \
  nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none \
  --resolve-symbols=false --force-overwrite=true \
  --output="$OUT/finalist-PLAN-s1" \
  "$BIN" benchmarknn -config "$CFG" \
  -override-config "numNNServerThreadsPerModel=1,cudaUseInitialConvFrontend=true,cudaInitialConvFrontendPlan=PLAN" \
  -model "$MODEL" -iterations 20 -warmup 10 -batch-size 13 -boardsize 19 -json
```

For S2, thermally prime first (600 warmup / 1200 measured is the established
pattern), then run both ABBA and reverse-order B/A/A/B with 1000 iterations and
30 warmup per leg. Here `A` is frontend disabled and `B` is one fixed finalist:

```bash
source /workspace/container-setup/nvidia-env.sh
gpu-lock with --gpu 2 -- "$BIN" benchmarknn -config "$CFG" \
  -override-config "cudaUseInitialConvFrontend=false" \
  -model "$MODEL" -iterations 1200 -warmup 600 -batch-size 13 -boardsize 19 -json

source /workspace/container-setup/nvidia-env.sh
gpu-lock with --gpu 2 -- "$BIN" benchmarknn -config "$CFG" \
  -override-config "cudaUseInitialConvFrontend=ENABLE,cudaInitialConvFrontendPlan=PLAN" \
  -model "$MODEL" -iterations 1000 -warmup 30 -batch-size 13 -boardsize 19 -json
```

Capture a short S2 Nsys trace for baseline and winner and use ordinal 5/NVTX to
verify that both server instances built and executed the requested plan and that
the fallback count is zero:

```bash
source /workspace/container-setup/nvidia-env.sh
gpu-lock with --profile --gpu 2 -- env DEBUGINFOD_URLS= DEBUGINFOD_TIMEOUT=1 \
  nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none \
  --resolve-symbols=false --force-overwrite=true \
  --output="$OUT/winner-s2" \
  "$BIN" benchmarknn -config "$CFG" \
  -override-config "cudaUseInitialConvFrontend=true,cudaInitialConvFrontendPlan=PLAN" \
  -model "$MODEL" -iterations 20 -warmup 10 -batch-size 13 -boardsize 19 -json
```

Finally run the full 8192-position replay against the stage1 FP32 reference:

```bash
source /workspace/container-setup/third_party_env.sh
source /workspace/container-setup/nvidia-env.sh
gpu-lock with --gpu 2 -- "$BIN" replaynn -config "$CFG" \
  -override-config "cudaUseInitialConvFrontend=true,cudaInitialConvFrontendPlan=PLAN" \
  -model "$MODEL" \
  -corpus /workspace/trainingdata/accuracy/2026-08-01-19x19-8192-seed20260803-full19.npz \
  -output "$OUT/replay-winner.krnn" -batch-size 13

python /workspace/katago/python/katago/train/compare_replay_krnn.py \
  --reference /workspace/results/rebuild/stage1/accuracy/replay-stage1-fp32-full19.krnn \
  --candidate "$OUT/replay-winner.krnn" \
  --output "$OUT/replay-winner-vs-fp32.json"
```

Before using these commands verbatim, confirm `replaynn` accepts `-model` in the
current tree's CLI form; retain the project's existing stage accuracy runner if
its model/config syntax differs.

## Acceptance and stop criteria

A finalist is acceptable only if all of the following hold:

- Both S2 server instances report the same requested fixed plan and zero fallback.
- Exact B13/19x19 dispatch is visible in the trace; no frontend plan construction,
  JIT, or autotune occurs in the timed interval.
- ABBA and reverse-order comparisons are directionally consistent. Because the
  expected effect is small, require positive adjacent-pair results rather than a
  single pooled mean.
- The 8192 replay meets the established accuracy thresholds; bit-exactness is a
  bonus, not an assumption.
- Workspace and long-run stability are acceptable, with no handle/stream errors.

Stop and reject the optimization if no fixed plan beats legacy reproducibly in
both orderings, if S2 gains disappear despite a local kernel win, or if the best
result requires concurrent runtime autotuning or silent fallback. The small
critical-path share makes a complicated or fragile implementation unjustified
without a repeatable whole-network improvement.

## Main risks

- Engine indices and knob values are cuDNN implementation details and can change
  with GPU, driver, cuDNN, batch, or graph details.
- S2 concurrent autotune measures transient peer interference and may select a
  plan that is not intrinsically or steadily best.
- Incorrect tensor strides, weight transformation, tensor UIDs, or convolution
  mode can produce a fast but semantically different graph.
- Frontend graph execution overhead can erase a kernel-only improvement.
- FP32 accumulation does not guarantee bit-identical output when reduction order
  changes; full replay remains mandatory.
- Per-handle graph/plan lifetime and stream affinity must match the two-server
  execution model.
- Workspace growth can silently invalidate the global scratch allocation or
  reduce memory headroom.
- Silent fallback creates false-positive benchmark results by mixing two code
  paths under one candidate label.
- The current convolution's already-low 5090 D latency and once-per-forward
  frequency cap the likely payoff; S2 phase shifts can erase even a large local
  percentage improvement.
