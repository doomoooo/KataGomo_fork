# RTX 5090 D B13/19x19 initial global matmul + broadcast-add audit

Date: 2026-08-06

## Scope and decision

This is a read-only audit for the one fixed B13/19x19 network shape. No shared
C++/CUDA source was modified and no GPU command was run.

The current boundary is a good bounded fusion candidate: it executes exactly
once per forward as three consecutive kernels, two of which are launch-bound
small GEMM work and one of which traverses the complete C768 spatial tensor. The
implementation should reproduce the validated FP32-dot design from the 5080 and
4090 work, but the 4090 S2 regression is material counterevidence. The 5090 D
candidate must therefore be selected by S2 whole-network measurements, not by
the approximately 45% S1 boundary win observed on 4090.

Conditional on a positive S2 result, the plausible whole-network gain is about
0.05-0.15%. The current complete boundary is only about 0.19-0.21% of a
per-server forward; a much larger claimed steady-state gain would indicate a
phase change and requires stronger repeated evidence. A negative S2 outcome is
entirely plausible.

## Real call boundary

The official path is in `/workspace/katago/cpp/neuralnet/cudabackend.cpp`:

1. `Trunk` constructs `initialMatMul` from `desc->initialMatMul`.
2. After `initialConv` writes `trunkScratch.buf`, `Trunk::apply` calls
   `initialMatMul->apply(..., inputGlobalBuf, trunkBuf, ...)`.
3. `MatMulLayer::apply` calls `cublasHgemm` because the current
   `cudaUseProjectionGemmLt` default is false and the target config does not
   override it.
4. `customCudaAddNCBiasInplaceNHWC<half>` broadcasts the `[B,C]` result from
   `trunkBuf` into the initial-convolution output in `trunkScratch.buf`.
5. Any SGF metadata projection and broadcast-add remains later in program order
   and is outside this fusion.

`cudaUseInitialGlobalMatMulAdd` is present in the SM120 options and is parsed with
default `true`, but has no consumer. It is absent from
`/workspace/bench-cuda-gpu2-5090d-s2.cfg`, so changing it currently has no effect.
An implementation should default it to false until the exact 5090 D candidate is
accepted.

The existing generic `sm120MatMul` hook is not a sufficient integration point.
It can replace the GEMM but receives only the small GEMM output pointer, not the
spatial `trunkScratch` pointer, so the broadcast kernel would remain. The fusion
needs a dedicated hook around the complete two-operation source boundary in
`Trunk::apply`.

## Exact tensor and arithmetic contract

| Tensor | Storage and layout | Exact shape / strides | Role |
| --- | --- | --- | --- |
| `inputGlobalBuf` | FP16, contiguous row-major | `[13,19]`, strides `[19,1]` | Global features; host FP32 input has already been rounded to half |
| `initialMatMul.matBuf` | FP16, contiguous row-major `inC x outC` | `[19,768]`, strides `[768,1]` | Same transformed/scaled weights used by `cublasHgemm` |
| baseline `trunkBuf` | FP16, contiguous row-major | `[13,768]`, strides `[768,1]` | Temporary global projection; removed by fusion |
| `trunkScratch.buf` | FP16 NHWC | physical `[13,361,768]`, strides `[277248,768,1]` | Initial-convolution output, updated in place |

The equivalent scalar result is:

```text
dot[n,c] = round_half(sum_fp32(k=0..18, inputGlobal[n,k] * weight[k,c]))
spatial[n,xy,c] = half_add(spatial[n,xy,c], dot[n,c])
```

The fixed dispatch gate must require all of:

- SM120, B13, 19x19 (`xy=361`), K19, C768;
- FP16 input, weight, and spatial storage;
- NHWC/channel-contiguous spatial output;
- the initial-global call site, not another K19/C768 matrix multiplication.

The current call site uses `cublasHgemm` with half alpha/beta and half output; it
does not select an explicit compute type. The observed library tactic emits a
main Tensor Core kernel plus a split-K reduction. The custom candidate should
make its numerical policy explicit: convert half operands to float, accumulate
the 19 terms with ordered FP32 RN FMAs, round once to half, then perform the same
half addition as the baseline broadcast kernel.

## Current 5090 D ordinal attribution

The stage27 trace has 30 timed forwards on each of two streams. Every ordinal
below has 60 calls, proving one complete boundary per forward.

| Ordinal / logical position | Kernel signature | Isolated median | S2 median | S2 / isolated | S2 total (60 calls) | Excess | Dominant worst peer |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| 6 `frontend.initial_global_matmul` | `library_gemm|Kernel2|g8x1x3|b128x1x1|r128|s24576` | 2.624 us | 3.760 us | 1.433x | 0.229603 ms | 0.072163 ms | cuDNN, 1.622x over 30 calls |
| 7 `frontend.initial_global_matmul_splitk_reduce` | `library_gemm|splitKreduce_kernel|g24x1x1|b32x16x1|r49|s0` | 1.280 us | 1.344 us | 1.050x | 0.082464 ms | 0.006880 ms | cuDNN, 1.100x over 20 calls |
| 8 `frontend.initial_global_broadcast_add` | `head_elementwise|addNCBiasInplaceNHWCHalfKernel|g3x361x13|b256x1x1|r16|s0` | 7.729 us | 8.128 us | 1.052x | 0.545732 ms | 0.082022 ms | library GEMM, 1.416x over 30 calls |

Boundary totals:

- Sum of isolated medians: 11.633 us.
- Sum of S2 medians: 13.232 us, 13.8% above the isolated sum.
- Actual S2 total: 0.857799 ms / 60 forwards = 14.297 us per forward.
- Summed concurrency excess: 0.161065 ms / 60 = 2.684 us per forward.
- Broadcast-add accounts for 63.6% of S2 boundary time and 50.9% of its excess;
  the main GEMM contributes another 44.8% of excess.

This attribution explains why replacing only the GEMM is weak. Most absolute
time is the broadcast traversal, while most interference is split between that
traversal and the small, high-resource GEMM (`128` registers/thread and 24 KiB
shared memory).

At roughly 3.67-3.78k nn/s, B13/S2 implies about 6.9-7.1 ms per forward on each
server. The observed 13.2-14.3 us boundary is about 0.19-0.21% of that latency.
Saving 4-8 us maps linearly to only about 0.06-0.12% throughput.

## Historical cross-check

### RTX 5080, B19/S2

The retained optimization history records an accepted fusion of global-feature
dot and spatial broadcast-add. The first version measured
`2857.798 -> 2862.815 nn/s` (+0.176%), but its numerical behavior was not the
accepted policy. The retained FP32-dot version measured
`2857.765 -> 2861.736 nn/s` (about +0.139%) and passed p0loss and all-head gates.
The final 5080 config enabled `cudaUseInitialGlobalMatMulAdd`.

The raw artifact is under `/data/wangyize/...` and is not mounted locally. Its
filename records `fp32dot-cbbc-4`, but that label is not enough to reconstruct
the exact launch mapping safely. Only the algorithmic invariant and S2 result
should be carried forward, not an inferred meaning for `cbbc-4`.

### RTX 4090, B13/19x19

The locally available Stage 27 implementation is fully inspectable. One thread
owns one contiguous output channel, performs 19 scalar FP32 FMAs, rounds once to
half, then reuses the result across `xyPerBlock` spatial rows. With 256 threads,
its launch is:

```text
grid = [768/256, ceil(361/xyPerBlock), 13]
block = [256,1,1]
```

The geometry sweep rejected a non-coalesced four-channel layout at
37.50-37.82 us. Channel-contiguous variants measured:

| Rows reused per CTA | Kernel time | Grid CTAs | Result |
| ---: | ---: | ---: | --- |
| 4 | 12.00 us | 3549 | slower than the best |
| 8 | 9.44-9.60 us | 1794 | selected; 40 registers, 2.34 waves/SM, 82.9% occupancy, no spill |
| 16 | 9.70-9.76 us | 897 | slightly slower |

S1 Nsys reduced the complete boundary from 15.872/15.875 us to
8.733/8.747 us in forward/reverse order, about 45%, and removed two launches.
S1 ABBA was consistently positive: `2460.772 -> 2463.670 nn/s` (+0.118%), all
four adjacent pairs positive.

However, the one-order topology probe measured S2
`3271.145 -> 3232.313 nn/s` (-1.187%). The fusion was therefore accepted only
for the 4090 S1 config and explicitly left off for S2. Its 8192-row replay passed
the established FP32 envelope, but it was not byte-identical to the prior stage;
policy top-1 agreement with Stage 25 was 99.744%. This is direct evidence that
both reduction order and stream topology matter.

Primary local evidence:

- `/workspace/results/4090/stage27/hypothesis-initial-global-matmul-add.md`
- `/workspace/results/4090/stage27/final-decision-summary.json`
- `/workspace/katago-4090/cpp/neuralnet/cudabackend_sm89_kernels.cu`
- `/workspace/results/rebuild/stage27/current-s2-ordinal-attribution.json`
- `/workspace/cuda-optimization-history.md`

## Exact-shape fusion design

### Integration

Add an optional `Sm120InitialGlobalMatMulAddFn` hook to `CudaHandles`. At the
current `Trunk::apply` boundary, call it with the stream, `initialMatMul.matBuf`,
`inputGlobalBuf`, `trunkScratch.buf`, B, XY, K, C, precision, and layout. If and
only if it returns true, skip both `initialMatMul->apply` and the following
broadcast-add. Otherwise execute the existing path unchanged.

The hook should be installed with the other SM120 runtime hooks. It is stateless
apart from using the passed stream and device pointers, so unlike a frontend
graph it does not need to be constructed before the official `Model`. Preserve
the later SGF metadata projection/add ordering exactly. The option and any
geometry selector should be logged once per server.

### Kernel

Use the 4090 channel-contiguous mapping as the reference implementation:

1. One thread owns channel `c`, keeping all writes coalesced.
2. Load the same server-owned half weights at `weight[k*768+c]` and half input at
   `inputGlobal[n*19+k]`.
3. Accumulate exactly 19 ordered `__fmaf_rn` operations in float.
4. Convert once with `__float2half_rn`.
5. Apply `__hadd` to `spatial[(n*361+xy)*768+c]` for a fixed number of rows.

No temporary `[13,768]` output is materialized. The existing `trunkBuf` allocation
is still needed by later trunk operations, so this is a traffic/launch reduction,
not a scratch-allocation reduction.

### Bounded geometry search

Search only these three channel-contiguous candidates on 5090 D:

- 256 threads, 4 spatial rows reused per CTA (`3549` CTAs).
- 256 threads, 8 rows (`1794` CTAs), the 4090 winner.
- 256 threads, 16 rows (`897` CTAs).

This is a low-compute search over the concurrency/reuse balance. It also tests
whether the 5090 D's larger SM count and S2 phase prefer more or fewer CTAs.
Do not repeat the 4090 non-coalesced channel-group layout unless NCU reveals a
materially different bottleneck; it was over 3x slower. Do not search other
batches or spatial shapes.

Choose the locally best two geometries, not only the fastest isolated one, for
the S2 screen. The 4090 result shows that a slightly slower isolated kernel may
be the safer concurrency shape.

## Numerical and functional risks

- FP32 scalar FMA order differs from the library GEMM and split-K reduction.
  Passing an FP32 reference envelope is required; bit identity is not expected.
- The global projection must be rounded to half exactly once before the spatial
  half add. Keeping a float bias through the add changes the network.
- Replacing `__hadd` with a float add followed by conversion also changes the
  rounding boundary.
- Weights are stored as `[K,C]`, not `[C,K]`; a transposition mistake can remain
  in-bounds and produce plausible-looking output.
- The host global input is FP32 but the normal FP16 path has already converted it
  to half. Reading the host-staging FP32 buffer would implement different math.
- The in-place destination is NHWC `[B,XY,C]`. NCHW indexing must reject rather
  than silently dispatch.
- Preserve ordering before any optional metadata encoder contribution.
- Each S2 server owns its own model weights and stream; the hook must use the
  pointer passed by that server and must not cache a pointer globally.
- A launch error must not silently produce a partially updated tensor. Exact
  shape mismatches may fall back before launch; post-launch errors should fail
  the run.
- Fusing three kernels changes the overlap phase. The 4090 S2 regression proves
  that a local win can make the complete two-stream schedule worse.

## Measurement plan

No command below was executed during this audit. All GPU commands must use the
project lock wrapper. Suggested variables:

```bash
BIN=/workspace/katago/build-cuda/katago
CFG=/workspace/bench-cuda-gpu2-5090d-s2.cfg
MODEL=/workspace/models/b11c768h12nbt3tflrs-fson-silu.bin.gz
OUT=/workspace/results/rebuild/stage28/initial-global
```

### 1. Local boundary gate

For each rows-per-CTA value 4, 8, and 16, capture S1 Nsys in both control-first
and candidate-first order. Use a unique candidate kernel name and preferably an
NVTX range around the whole baseline/candidate boundary. The candidate must be
one kernel per forward and ordinals 6-8 must collapse to that one range.

```bash
source /workspace/container-setup/nvidia-env.sh
gpu-lock with --profile --gpu 2 -- env DEBUGINFOD_URLS= DEBUGINFOD_TIMEOUT=1 \
  nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none \
  --resolve-symbols=false --force-overwrite=true \
  --output="$OUT/rows-ROWS-s1" \
  "$BIN" benchmarknn -config "$CFG" \
  -override-config "numNNServerThreadsPerModel=1,cudaUseInitialGlobalMatMulAdd=true,cudaInitialGlobalRowsPerCTA=ROWS" \
  -model "$MODEL" -iterations 20 -warmup 10 -batch-size 13 -boardsize 19 -json
```

Only after Nsys shows a shorter complete boundary should NCU collect launch,
occupancy, SOL, and memory metrics for at most the best two variants. Require no
local/shared spill and record registers, CTA count, waves/SM, achieved occupancy,
L1/L2 throughput, and duration. Avoid a generic `Kernel2` filter without a launch
skip/range: it can match unrelated library kernels.

### 2. S2 topology screen

Thermally prime the current control, then run control/candidate/candidate/control
and the reverse candidate/control/control/candidate sequence for each of the two
finalists. Keep B13, 19x19, S2, all other options, clocks, power policy, and
binary constant.

```bash
source /workspace/container-setup/nvidia-env.sh
gpu-lock with --gpu 2 -- "$BIN" benchmarknn -config "$CFG" \
  -override-config "cudaUseInitialGlobalMatMulAdd=ENABLE,cudaInitialGlobalRowsPerCTA=ROWS" \
  -model "$MODEL" -iterations 1000 -warmup 30 -batch-size 13 -boardsize 19 -json
```

Use the established longer thermal prime before the sequence (600 warmup / 1200
measured is the current convention). A short S2 Nsys trace for control and each
finalist must report per-stream kernel counts, target-boundary/ordinal latency,
two-stream union, overlap peer, and iteration phase offset. The decision metric
is whole-network nn/s; local S2 latency is explanatory evidence.

### 3. Accuracy

Run the full 8192-position replay for the surviving S2 candidate and compare it
both with the currently accepted CUDA output and the Stage 1 FP32 reference:

```bash
source /workspace/container-setup/third_party_env.sh
source /workspace/container-setup/nvidia-env.sh
gpu-lock with --gpu 2 -- "$BIN" replaynn -config "$CFG" \
  -override-config "cudaUseInitialGlobalMatMulAdd=true,cudaInitialGlobalRowsPerCTA=ROWS" \
  -model "$MODEL" \
  -corpus /workspace/trainingdata/accuracy/2026-08-01-19x19-8192-seed20260803-full19.npz \
  -output "$OUT/replay-candidate.krnn" -batch-size 13

python /workspace/katago/python/katago/train/compare_replay_krnn.py \
  --reference /workspace/results/rebuild/stage1/accuracy/replay-stage1-fp32-full19.krnn \
  --candidate "$OUT/replay-candidate.krnn" \
  --output "$OUT/replay-candidate-vs-fp32.json"
```

Record all policy, value, score, and ownership metrics. The fused result must
stay within the established all-head FP32 envelope and contain no nonfinite
values. Also retain a direct current-CUDA comparison to quantify the reduction
order change.

## Acceptance and stop conditions

Accept only if all of these hold:

- The exact dispatch is B13/19x19/K19/C768/FP16/NHWC and every other invocation
  takes the untouched fallback.
- One candidate launch replaces all three baseline launches on both S2 streams,
  with no runtime fallback or launch error.
- The complete local boundary improves in both profiling orders and the selected
  kernel has no spill.
- S2 ABBA and reverse-order results are positive and adjacent-pair consistent.
  A single pooled improvement or an S1-only gain is insufficient.
- The full 8192 replay passes the established FP32 envelope for every head.

Stop and reject this route if:

- neither finalist is locally faster than the 11.6 us isolated boundary;
- both finalist geometries are flat or negative in S2, even if S1 is faster;
- the apparent S2 gain is smaller than run-order noise or reverses by order;
- accuracy requires half-dot arithmetic, float spatial addition, or another
  change to the explicit FP32-dot/half-round/half-add contract;
- a stable gain requires broadening to other batches or shapes.

The only justified reopen after a clean S2 rejection is a materially different
boundary, such as incorporating the global contribution into a proven initial
convolution epilogue. That would be a separate hypothesis with a larger
correctness and implementation surface, not another geometry tweak to this
kernel.
