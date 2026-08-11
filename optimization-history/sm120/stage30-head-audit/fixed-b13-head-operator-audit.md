# RTX 5090 D fixed B13/19x19 trunk-tip and head operator audit

Date: 2026-08-06

## Scope and executive result

This is a read-only audit of the exact B13/19x19 FP16 NHWC path. No shared
C++/CUDA source was modified and no GPU command was run.

The stage27 attribution identifies 27 launches after the last transformer block,
ordinals 317-343. Their measured S2 total is 7.039432 ms across 60 forwards, or
117.324 us/forward. Summed concurrency excess is 1.541243 ms, or 25.687
us/forward. At the observed 3.67-3.78k nn/s, this complete audited tail occupies
roughly 1.65-1.70% of one server's forward latency, but only a subset is
realistically removable.

The three requested SM120 options are parsed with default `true` but have no
consumer:

- `cudaUseFusedPolicyP1`
- `cudaUseHeadBNHalfToFloat`
- `cudaUseWideHeadProjection`

They are absent from `/workspace/bench-cuda-gpu2-5090d-s2.cfg`, and toggling them
currently changes no execution. The recommended implementation order is:

1. Implement exact-shape fused policy P1 as the low-cost prerequisite.
2. Implement one no-split C768-to-C384 wide head projection; this has the largest
   expected payoff.
3. Only if the wide projection wins under S2, test BN-to-float on top of it as a
   separate combined candidate.

Do not repeat standalone BN-to-float first. It improved both local boundaries on
4090 but regressed all four S2 ABBA pairs by a pooled 1.261%. Do not reopen fused
head pooling as an isolated change either; the 5080 reverse-order result was
flat and the apparent forward gain was attributed to run-order drift.

## Current code path

The live implementation is in
`/workspace/katago/cpp/neuralnet/cudabackend.cpp`.

After the last outer projection, trunk-tip affine+SiLU produces the common half
C768 tensor. `Model::apply` then runs `PolicyHead::apply` followed by
`ValueHead::apply`.

Policy:

1. Independent 1x1 FP16 projections C768->C96 for p1 and g1.
2. g1 half BN+SiLU, half-to-float copy, three-way global pooling, and FP32
   C288->C96 projection for the per-batch p1 bias.
3. p1 half-to-float, FP32 global-bias add, FP32 BN+SiLU, then the policy output
   projection.
4. A small FP32 policy-pass MLP.

Value:

1. Independent 1x1 FP16 projection C768->C192.
2. v1 half BN+SiLU and half-to-float copy. The float tensor feeds three-way
   global pooling; the half tensor remains live for ownership.
3. A small FP32 value MLP with separate value and score projections.
4. A half ownership 1x1 projection, split-K reduction, then half-to-float copy.

The three C768 head projections use the ordinary 1x1 NHWC FP16 GEMM path. The
current generic `sm120MatMul` hook is disabled because
`cudaUseProjectionGemmLt=false`. The live trunk-tip kernel is the SM120 half2
affine+SiLU route; there is no flat vec8 variant.

## Fixed tensor contracts

| Boundary | Exact physical shape / storage | Notes |
| --- | --- | --- |
| Common trunk / trunk tip | `[13,361,768]` FP16 NHWC | 4,693 rows, channel-contiguous |
| p1 projection | `[4693,768] x [768,96] -> [4693,96]` FP16 | policy spatial branch |
| g1 projection | `[4693,768] x [768,96] -> [4693,96]` FP16 | policy pooling branch |
| v1 projection | `[4693,768] x [768,192] -> [4693,192]` FP16 | value pooling and ownership branch |
| Wide projection candidate | `[4693,768] x [768,384] -> [4693,384]` FP16 | slices p1 `[0,96)`, g1 `[96,192)`, v1 `[192,384)` |
| g1 BN/copy | FP16 `[13,361,96]` -> FP32 same shape | half result has no later consumer |
| v1 BN/copy | FP16 `[13,361,192]` -> FP16 and FP32 same shape | half result is required by ownership |
| g1 pool | FP32 `[13,361,96]` -> `[13,288]` | three statistics per channel |
| v1 pool | FP32 `[13,361,192]` -> `[13,576]` | three statistics per channel |
| p1 post-bias boundary | half `[13,361,96]` plus float `[13,96]` -> float `[13,361,96]` | half-to-float, bias, affine, SiLU |

Any optimized dispatch must check B13, XY361, the exact channel sizes, FP16
trunk/head storage, NHWC row layout, and the specific logical call site. All
other invocations use the current path unchanged.

## Current S2 attribution

The source is
`/workspace/results/rebuild/stage27/current-s2-ordinal-attribution.json`. It has
30 timed forwards on each of two streams and 60 calls for every row below.
`S2 total` and `excess` are totals across those 60 calls; medians are per call.

| Ord | Logical position | Kernel signature | Isolated | S2 median | S2 total | Excess |
| ---: | --- | --- | ---: | ---: | ---: | ---: |
| 317 | trunk.tip_norm_silu | `affineSiluHalf2Kernel; g4693; b384; r17; s0` | 5.120 us | 8.272 us | 0.490468 ms | 0.183268 ms |
| 318 | policy.p1_conv | `Kernel2; g148; b128; r80; s98304` | 6.240 us | 11.344 us | 0.682344 ms | 0.307944 ms |
| 319 | policy.g1_conv | `Kernel2; g148; b128; r80; s98304` | 5.952 us | 8.096 us | 0.478408 ms | 0.121288 ms |
| 320 | policy.g1_norm_silu | `applyCScaleBiasNHWCSiluHalfKernel; g1x73x13; b96x5; r16` | 2.112 us | 2.672 us | 0.160385 ms | 0.033665 ms |
| 321 | policy.g1_half_to_float | `copyFromHalfKernel; g880; b512; r16` | 1.568 us | 1.856 us | 0.111905 ms | 0.017825 ms |
| 322 | policy.g1_global_pool | `gPoolChannelsNHWCKernel; g2x1x13; b64x8; r22; s4096` | 4.512 us | 6.848 us | 0.443814 ms | 0.173094 ms |
| 323 | policy.gpool_to_bias_matmul | `gemmSN_NN_kernel; g2x2; b256; r64; s21504` | 5.409 us | 6.800 us | 0.409987 ms | 0.085477 ms |
| 324 | policy.p1_half_to_float | `copyFromHalfKernel; g880; b512; r16` | 1.536 us | 1.760 us | 0.107104 ms | 0.014944 ms |
| 325 | policy.gpool_bias_add | `addNCBiasInplaceNHWCKernel; g1x73x13; b96x5; r16` | 1.824 us | 2.064 us | 0.126017 ms | 0.016577 ms |
| 326 | policy.p1_norm_silu | `applyCScaleBiasNHWCSiluKernel; g1x73x13; b96x5; r16` | 2.208 us | 2.464 us | 0.148737 ms | 0.016257 ms |
| 327 | policy.p2_conv | `Kernel2; g74; b128; r90; s98304` | 3.968 us | 4.624 us | 0.291074 ms | 0.052994 ms |
| 328 | policy.gpool_to_pass_matmul | `gemmSN_NN_kernel; g2x2; b256; r64; s21504` | 5.376 us | 6.000 us | 0.368423 ms | 0.045863 ms |
| 329 | policy.pass_bias_silu | `addCBiasInplaceNCKernelSilu; g1x3; b96x5; r16` | 1.008 us | 1.152 us | 0.114369 ms | 0.053889 ms |
| 330 | policy.gpool_to_pass_matmul2 | `gemmSN_NN_kernel; g1x2; b256; r64; s21504` | 2.336 us | 2.704 us | 0.173762 ms | 0.033634 ms |
| 331 | value.v1_conv | `Kernel2; g148; b128; r118; s98304` | 8.064 us | 9.216 us | 0.557443 ms | 0.073603 ms |
| 332 | value.v1_norm_silu | `applyCScaleBiasNHWCSiluHalfKernel; g1x181x13; b192x2; r16` | 3.136 us | 3.392 us | 0.268769 ms | 0.080609 ms |
| 333 | value.v1_half_to_float | `copyFromHalfKernel; g1760; b512; r16` | 2.176 us | 2.496 us | 0.175520 ms | 0.044960 ms |
| 334 | value.v1_global_pool | `valueHeadPoolChannelsNHWCKernel; g3x1x13; b64x8; r22; s2048` | 3.264 us | 3.680 us | 0.227459 ms | 0.031619 ms |
| 335 | value.v2_matmul | `gemmSN_NN_kernel; g3x2; b256; r64; s21504` | 9.568 us | 10.112 us | 0.611236 ms | 0.037156 ms |
| 336 | value.v2_bias_silu | `addCBiasInplaceNCKernelSilu; g1x7; b192x2; r16` | 1.024 us | 1.088 us | 0.066273 ms | 0.004833 ms |
| 337 | value.v3_matmul | `gemmSN_NN_kernel; g1x2; b256; r64; s21504` | 3.505 us | 3.808 us | 0.231234 ms | 0.020964 ms |
| 338 | value.v3_bias | `addCBiasInplaceNCKernel; g1; b3x170; r16` | 0.960 us | 1.056 us | 0.064162 ms | 0.006562 ms |
| 339 | value.score_matmul | `gemmSN_NN_kernel; g1x2; b256; r64; s21504` | 3.520 us | 3.808 us | 0.231204 ms | 0.020004 ms |
| 340 | value.score_bias | `addCBiasInplaceNCKernel; g1; b6x85; r16` | 0.928 us | 1.024 us | 0.062402 ms | 0.006722 ms |
| 341 | value.ownership_conv | `Kernel2; g8x19x3; b128; r118; s33792` | 4.032 us | 4.561 us | 0.273411 ms | 0.031779 ms |
| 342 | value.ownership_conv_splitk_reduce | `splitKreduce_kernel; g147; b32x16; r49` | 1.377 us | 1.440 us | 0.088832 ms | 0.006732 ms |
| 343 | value.ownership_half_to_float | `copyFromHalfKernel; g10; b512; r16` | 0.929 us | 1.120 us | 0.074690 ms | 0.018981 ms |

### Aggregated boundaries

| Boundary | Ordinals | Isolated median sum | S2 median sum | S2 total / 60 | Excess / 60 |
| --- | --- | ---: | ---: | ---: | ---: |
| Trunk tip | 317 | 5.120 us | 8.272 us | 8.175 us/forward | 3.054 us/forward |
| Entire policy head | 318-330 | 44.049 us | 58.384 us | 60.272 us/forward | 16.224 us/forward |
| Entire value head | 331-343 | 42.482 us | 46.801 us | 48.877 us/forward | 6.409 us/forward |
| Three narrow head projections | 318,319,331 | 20.256 us | 28.656 us | 28.637 us/forward | 8.381 us/forward |
| Fused-policy-P1 target | 324-326 | 5.568 us | 6.288 us | 6.364 us/forward | 0.796 us/forward |
| Head-BN-to-float target | 320-321,332-333 | 8.992 us | 10.416 us | 11.943 us/forward | 2.951 us/forward |
| BN/copy plus both pools | 320-322,332-334 | 16.768 us | 20.944 us | 23.131 us/forward | 6.363 us/forward |
| Policy pass MLP | 328-330 | 8.720 us | 9.856 us | 10.943 us/forward | 2.223 us/forward |
| Value dense tail | 335-340 | 19.505 us | 20.896 us | 21.109 us/forward | 1.604 us/forward |
| Ownership tail | 341-343 | 6.337 us | 7.121 us | 7.282 us/forward | 0.958 us/forward |

The three narrow projections are the strongest target: they account for 44.3%
of policy/value/trunk-tip excess. The standalone P1 boundary is much smaller;
its value comes from eliminating two launches and a 1.72 MiB FP32 intermediate
round trip rather than from a large isolated kernel.

The peer attribution also explains why launch-count wins are not sufficient.
The p1 projection overlaps `library_gemm` on all 60 calls and reaches 1.818x
slowdown; g1's worst peer is likewise `library_gemm` (1.487x). The value
projection instead most often overlaps `head_elementwise` and reaches 1.147x.
For the BN/copy routes, `library_gemm` is the worst peer throughout; value v1 BN
reaches 1.842x in its 30 such overlaps. The P1 pointwise sequence is less
interference-sensitive at 1.116x-1.146x overall. Therefore wide projection and
BN changes must be judged by the complete S2 union and phase shift, whereas P1
is the lower-risk first integration step.

## 5080 evidence

The 5080 history is B19/S2 rather than the target B13, so it establishes useful
mechanisms and S2 direction, not portable launch constants.

| Route | Historical result | Decision |
| --- | --- | --- |
| Fused policy P1 | fused kernel 3.777 us; forward +0.191%, reverse +0.169% | accepted; arithmetic unchanged |
| Head BN half-to-float | fused kernel 5.408 us; forward +0.233%, reverse +0.162% | accepted; final accuracy revalidated |
| First wide-head projection | +0.027% | rejected as noise-sized |
| No-split wide head | +0.255% | accepted in final config and full FP32 replay |
| Fused head pooling | forward +0.211%, reverse +0.003% | rejected as run-order drift |
| Flat vec8 C768 affine+SiLU | NCU 15.840 -> 14.208 us; whole network +0.216% | accepted across the repeated C768 affine family |

The raw 5080 artifacts are referenced under `/data/wangyize/...` and are not
mounted here. In particular, no undocumented wide-head tile should be inferred
from the summary alone.

## 4090 evidence

### Fused policy P1, Stage 25 S2

The exact B13 implementation fused half-to-float, global bias, FP32 affine, and
FP32 SiLU. A 96x5 thread layout measured 3.20-3.23 us with no spill and replaced
three launches with one. Nsys reduced the target boundary by about 63% in both
orders and reduced the S2 union in both orders. The short ABBA result was only
`+0.074%`, with 3/4 positive pairs and reverse aggregate `-0.042%`, so confidence
was explicitly low. Replay was byte-identical to the prior stage.

### Head BN half-to-float, Stage 26 S2

Policy g1 and value v1 local boundaries improved by 46-49% and 37-39%, removing
two copies. Yet S2 union direction changed with order, and full ABBA was negative
in every pair: `3246.156 -> 3205.214 nn/s` (-1.261%). It was rejected and no
accuracy replay was run. The recorded reopen condition was a wider fusion that
also changes pooling or ownership scheduling.

### No-split wide head, Stage 28 S1

The implementation concatenated the p1/g1/v1 weights and emitted one C384
tensor without materializing three splits. First consumers used row stride 384
and offsets 0/96/192. A fixed M4693/N384/K768 AOT kernel measured 15.94 us versus
35.164 us for three narrow projections (-54.7%), with no spill. S1 ABBA was
`+0.599%`, all four pairs positive, and replay was byte-identical. It was not
tested or enabled in S2.

### BN-to-float after wide head, Stage 29 S1

Direct FP32 output from strided g1/v1 slices reduced the local boundary by about
41% and removed two copies. S1 ABBA was `+0.078%`, all four pairs positive, and
replay remained byte-identical. S2 was not retested; the Stage 26 S2 rejection
was explicitly left unchanged.

Primary artifacts:

- `/workspace/results/4090/stage25/final-decision-summary.json`
- `/workspace/results/4090/stage26/final-decision-summary.json`
- `/workspace/results/4090/stage28/final-decision-summary.json`
- `/workspace/results/4090/stage29/final-decision-summary.json`
- `/workspace/katago-4090/cpp/neuralnet/cudabackend_sm89_forward.cpp`
- `/workspace/katago-4090/cpp/neuralnet/cudabackend_sm89_kernels.cu`
- `/workspace/cuda-optimization-history.md`

## Ranked implementation experiments

The ranking below is by expected 5090 D S2 value, considering both gain and
implementation/risk cost. The expected ranges are priors, not acceptance
thresholds.

| Rank | Experiment | Expected whole-network gain | Cost / risk | Recommendation |
| ---: | --- | ---: | --- | --- |
| 1 | One no-split C384 wide head, with stride-aware first consumers | +0.10% to +0.30% | high implementation, medium-high S2 phase risk | highest-payoff target; implement after P1 plumbing |
| 2 | Exact fused policy P1 | +0.03% to +0.15% | low-medium cost, low numerical risk, medium noise risk | implement first; required by the proven wide-head design |
| 3 | BN-to-float on top of an accepted wide head | 0% to +0.10% | medium cost, high S2 scheduling risk | combined-only experiment; one S2 screen, reject quickly if order-sensitive |
| 4 | BN producer fused directly into policy/value pooling | +0.05% to +0.15% | high cost and numerical risk | only reopen after simpler BN route fails; must preserve pooling reduction order |
| 5 | Trunk-tip-only flat vec8 affine+SiLU | 0% to +0.04% | low cost, low numerical risk | cheap isolated check; do not extrapolate the 5080 family-wide +0.216% to one ordinal |
| 6 | Small FP32 tail epilogues/wide outputs | +0.02% to +0.10% combined | medium-high cost, reduction-order risk | defer until the three established routes are settled |
| 7 | Ownership conversion epilogue | 0% to +0.03% | medium cost for very small boundary | lowest priority |

Standalone `cudaUseHeadBNHalfToFloat` is not ranked as a positive experiment. Its
best available exact-shape S2 evidence is a decisive rejection. The combined
wide-head version is a distinct hypothesis because it changes both producer
geometry and consumer stride.

Implementation sequence differs slightly from gain rank:

```text
fused P1 -> wide no-split -> S2 confirmation -> wide+BN-to-float
```

This sequence provides the P1 stride-aware consumer needed by the 4090-proven
wide design and preserves one-variable measurements.

## Concrete designs

### A. Fused policy P1

Add a dedicated policy-P1 hook at the current ordinals 324-326 boundary. Inputs
are a half p1 row (stride 96 normally, stride 384/offset 0 under wide head), the
FP32 `[13,96]` global bias, and p1's FP32 scale/bias. Emit contiguous FP32
`[13,361,96]` for p2.

Preserve this operation order exactly:

```text
float value = half_to_float(p1)
value = value + global_bias
value = value * scale + bias
out = value / (1 + exp(-value))
```

Use the validated 96x5 geometry as the first 5090 D candidate. It maps one CTA
to five spatial rows and all 96 channels. On any exact-gate failure, execute all
three original operations, not a partial mixture.

### B. No-split wide head

At model construction, concatenate the existing 1x1 weights into a row-major
`[768,384]` buffer with p1/g1/v1 offsets 0/96/192. At `Model::apply`, run one
fixed M4693/N384/K768 projection before policy, keep its half output alive through
value-head completion, and pass row stride/offset metadata to the first consumer
of each slice.

Do not split or copy the C384 tensor. If any first consumer cannot read its
slice, fall back to all three original projections. Materializing three slices
would restore much of the traffic the route is intended to remove.

The current SM120 outer-pre-projection search has the same M4693/N384/K768
mathematical shape. Its accepted finalist, if any, is the right initial AOT
candidate, but head integration still needs a separate local and S2 test because
weights, phase, and following consumers differ.

This boundary crosses policy and value ownership. A generic per-`ConvLayer`
matmul hook cannot implement it safely; the hook/state must live at `Model`
scope or an equivalent head coordinator. Each S2 server must own or safely share
the correct combined weights and have independent scratch lifetime.

### C. Wide-head BN-to-float

Policy g1 reads stride 384/offset 96 and emits only contiguous FP32 for pooling.
Value v1 reads stride 384/offset 192 and emits both contiguous half for ownership
and the exact float conversion for pooling.

Preserve the official half arithmetic boundary:

```text
half affine = half_fma(input, half_scale, half_bias)
float activated_input = half_to_float(affine)
half activated = round_half(silu_float(activated_input))
float pooled_input = half_to_float(activated)
```

Policy may omit the unused half store. Value may not. This route gets one
combined S2 attempt after wide-head acceptance; the 4090 standalone result is
strong evidence against a broad geometry search.

### D. Wider producer-to-pool fusion

If C is rejected, the only justified BN reopen is to eliminate the full FP32
maps as well as the copies:

- policy: g1 half BN+SiLU directly produces the three C96 pooling statistics;
- value: v1 still writes the exact half ownership input while directly producing
  the three C192 pooling statistics.

The reduction tree, fixed-361 normalization, half rounding, and final FP32
statistics must match the official full-board path closely enough to pass full
replay.
Because the 5080 isolated pooling fusion failed reverse-order confirmation, this
is high-risk and should not precede the proven P1/wide-head routes.

### E. Remaining small boundaries

- Trunk tip: add a fixed B13 C768 flat vec8 candidate against ordinal 317 only.
  It is already half2, so require a clear local improvement before S2.
- Policy pass: consider a single exact B13 kernel for ordinals 328-330 only if
  NCU confirms the two small GEMMs are launch-bound; the SiLU prevents simple
  algebraic combination.
- Value dense tail: a combined value+score projection can replace ordinals
  337-340 with one wider FP32 projection plus two bias slices. It changes GEMM
  reduction order and needs replay.
- Ownership: fuse the final half-to-float conversion only with a proven fixed
  ownership projection epilogue; the current 7.28 us complete tail is too small
  to justify a standalone complex kernel.

## Accuracy and lifecycle risks

- The wide projection changes GEMM tiling/reduction order. Full 8192-position policy,
  value, score, and ownership comparison is mandatory even if 4090 was
  byte-identical.
- Combined weight packing must preserve the official `[inC,outC]` layout and
  the exact p1/g1/v1 offsets. An in-bounds offset error can corrupt only one head
  and evade superficial checks.
- The wide buffer must survive the sequential policy call until value completes.
  Reusing it as ordinary scratch too early creates data-dependent failures.
- P1 uses FP32 affine and SiLU; g1/v1 BN uses half FMA and a half rounding point.
  These paths are numerically different and must not share one generic formula.
- Value v1's half output is required by ownership. Dropping it or regenerating it
  from float changes values.
- Pooling is a reduction, so changing its tree or normalization is a higher
  correctness risk than fusing pointwise copies.
- Each S2 server has a distinct stream and model lifetime. Combined weight and
  scratch ownership must be explicit; no process-global mutable pointer.
- A failed exact dispatch must fall back before writing any output. Silent
  partial fallback invalidates both correctness and performance attribution.
- Local launch reduction can worsen S2 phase, as demonstrated by the 4090 BN
  result. S1 evidence never overrides the S2 decision metric.

## Measurement and stop protocol

No command in this section was executed by the audit.

For each experiment:

1. Hold B13/19x19/S2, binary, clocks/power policy, and every unrelated option
   fixed. Build a control and one candidate geometry.
2. Use S1 Nsys first to prove the exact ordinal boundary and launch removal.
3. Run NCU only after a local win; record registers, shared memory, spills,
   waves/SM, occupancy, and memory/compute SOL.
4. Capture short S2 Nsys in both control-first and candidate-first order. Record
   per-stream kernel count, target boundary, union time, peer family, and phase
   offset.
5. Thermally prime, then run S2 A/B/B/A and reverse B/A/A/B. Gains in this audit
   are small enough that adjacent-pair consistency matters more than one pooled
   mean.
6. Run the full 8192-position replay only after the performance gate passes,
   comparing both with the accepted CUDA output and the FP32 reference.

Suggested benchmark form after implementation:

```bash
source /workspace/container-setup/nvidia-env.sh
gpu-lock with --gpu 2 -- \
  /workspace/katago/build-cuda/katago benchmarknn \
  -config /workspace/bench-cuda-gpu2-5090d-s2.cfg \
  -override-config "cudaUseFusedPolicyP1=P1,cudaUseWideHeadProjection=WIDE,cudaUseHeadBNHalfToFloat=BN" \
  -model /workspace/models/b11c768h12nbt3tflrs-fson-silu.bin.gz \
  -iterations 1000 -warmup 30 -batch-size 13 -boardsize 19 -json
```

Stop a route when any of these holds:

- its complete local boundary is not faster in both profile orders;
- the candidate spills or materializes split-equivalent traffic;
- S2 union direction reverses by run order;
- ABBA and reverse ABBA are negative or no larger than run-order noise;
- exact accuracy requires changing the stated precision/rounding contract;
- a positive result depends on any batch or spatial shape outside B13/19x19.

Specific route stops:

- P1: reject if the one-kernel boundary is not below about 3.5 us or S2 is not
  directionally positive; do not search many geometries beyond 96x5.
- Wide head: reject if one C384 projection plus first consumers is not materially
  below the three original projections plus consumers, or if splits are needed.
- BN-to-float: one combined wide-head S2 test only; any repeat of the Stage 26
  order sensitivity ends the route.
- Pooling: do not proceed past a micro/local prototype unless its exact reduction
  semantics are demonstrated before full-network timing.

## Final priority call

The head is not exhausted. The highest-confidence missing implementation is
fused policy P1, and the largest missing opportunity is no-split wide head. They
should be treated as one staged program: land/validate P1 first, then measure the
wide projection as the actual payoff candidate. BN-to-float is not an independent
priority despite its dead option; current S2 evidence says it is harmful unless a
wider producer boundary changes the schedule.
