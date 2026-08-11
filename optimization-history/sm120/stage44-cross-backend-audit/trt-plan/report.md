# RTX 5090 D fixed-B13/19x19 TensorRT plan cross-backend audit

Date: 2026-08-06 UTC

## Executive result

This audit rebuilt and inspected a fresh TensorRT 10.16.1 plan for the exact
RTX 5090 D, B13, 19x19, FP16, natural two-server topology. It did not modify
KataGo source or the target CUDA configuration.

The TensorRT plan is **not a performance oracle** for the current CUDA backend:
the natural-S2 TensorRT benchmark measured `3232.470 nn/s`, while the latest
accepted CUDA long result is `3855.728 nn/s` under a different long-run protocol.
Those absolute results must not be turned into a paired percentage. Layer-profile
durations below are similarly useful for attribution inside each backend, not for
cross-protocol speedup arithmetic.

The strongest genuinely new transferable observation is:

1. **A partial no-split C288 head is viable without fused P1.** TensorRT combines
   `policy_head.conv1g` (C96) and `value_head.conv1` (C192), including their
   per-slice affine scale/bias, into one C288 correlation layer, while leaving
   `policy_head.conv1p` separate. This removes the
   previously assumed dependency on a stride-aware/fused P1 before *any* useful
   wide-head experiment can start. The exact CUDA experiment would be one FP16
   C768-to-C288 projection plus stride-aware g1/v1 consumers, leaving P1 unchanged.
   The current g1+v1 projections account for about `28.0 us/fwd`; their impossible
   zero-cost ceiling is about `0.42%` of a `6.743 ms` forward, so a realistic
   search budget is only about `0.1%--0.25%` whole-graph S2.
2. **The outer projections merit only a materially different SM120 exact-grid
   mainloop experiment.** TensorRT launches exactly `37 x 3 = 111` CTAs for both
   directions, whereas the current CUTLASS route launches 148 CTAs because of
   swizzle/padding. This is a possible S2 resource/fairness mechanism, but not
   evidence that TensorRT is faster. Stage 28 already rejected conventional
   CUTLASS2 outer tactics; only an SM120-native CuTe/TMA/JIT family that preserves
   the exact grid would be new work.
3. TensorRT's frontend and trunk-tip Myelin fusion boundaries are structurally
   interesting but have small current CUDA ceilings. They rank below the current
   FFN/QKV hotspot work and below the two candidates above.

TensorRT's split Q/K RoPE, separate SwiGLU, MHA, FP32 head tactics, and residual
relocation should **not** be copied. The current CUDA backend already has better
fusion boundaries and directional profiler evidence for those regions.

## Scope, device identity, and measurement topology

The machine exposes two distinct ordinal systems. They must not be conflated:

| Namespace | Ordinal | PCI | UUID | Device |
| --- | ---: | --- | --- | --- |
| CUDA runtime / KataGo config | 2 | `0000:21:00.0` | `GPU-a05bf432-3f6b-83c1-98b1-aec8f8a6fd69` | RTX 5090 D, CC 12.0, 170 SM |
| NVML / `nvidia-smi` | 1 | `00000000:21:00.0` | `GPU-a05bf432-3f6b-83c1-98b1-aec8f8a6fd69` | RTX 5090 D |

`gpu-lock --gpu 2` accepts the CUDA ordinal, resolves it to this PCI device, and
reports acquiring NVML GPU 1. KataGo's `trtDeviceToUseThread0/1 = 2`, `trtexec
--device=2`, and the profiler report's `Device=2, CC=12.0, #SMs=170` all agree.

The plan was built from a fresh `homeDataDir`, so its timing cache was created by
the target RTX 5090 D instead of reusing an existing cache. The benchmark created
two real TensorRT backend server instances on CUDA device 2. It used B13, fixed
19x19, 20 warmup iterations, and 1000 timed iterations:

| Result | Value |
| --- | ---: |
| Per-server median GPU time | `8.042400 ms`, `8.044368 ms` |
| Combined natural-S2 throughput | `3232.470332 nn/s` |
| Actual wall time | `8.201324 s` |

The separate `trtexec --infStreams=2` run is only a two-context layer-attribution
diagnostic. It measured `256.346 qps`, equivalent to `3332.5 nn/s` at B13, and a
mean GPU latency of `7.743 ms`. It is not the KataGo natural-S2 acceptance metric.

## Exported graph and realized plan

### ONNX inventory

The emitted ONNX graph uses IR 9 and opset 20. It contains 1,660 nodes and 886
initializers.

| ONNX op | Count |
| --- | ---: |
| Mul | 420 |
| MatMul | 319 |
| Add | 241 |
| Transpose | 167 |
| Reshape | 132 |
| ReduceMean | 68 |
| Sqrt / Div | 66 / 66 |
| Gather | 66 |
| Sigmoid | 61 |
| Softmax | 33 |
| Conv | 13 |
| Other | 11 |

Bindings are `InputSpatial [B,22,19,19]`, `InputGlobal [B,19,1,1]`, and
`InputMask [B,1,19,19]`, plus five float outputs. `InputMask` has no ONNX-node
consumer and no realized engine-layer consumer. It survives only as a binding.
Because the optimization target is fixed 19x19 and mask work is explicitly out
of scope, this audit does not rank mask pruning as a candidate.

### Engine inventory

The detailed inspector JSON contains 571 realized layers.

| Realized layer kind | Count |
| --- | ---: |
| Myelin/generated kernel (`kgen`) | 269 |
| GEMM | 154 |
| Reshape | 66 |
| Cast | 66 |
| Correlation/convolution | 12 |
| Average pool | 2 |
| Max pool | 1 |
| Runtime shape call | 1 |

There are 1,031 Half-format, 198 Int64-format, and 122 Float-format tensor
occurrences in layer records. The 133 tactic-free layers are exactly the 66
reshapes, 66 casts, and one runtime shape call. They are visible because the
engine profile is dynamic in batch (`min=1`, `max=13`) even though the audit runs
B13; the current CUDA fixed-B13 AOT path has no corresponding runtime shape-node
opportunity.

The exporter/plan keeps the repeated transformer trunk in Half/NHWC around GEMMs,
but uses Float around RMS reductions and much of the head. The ONNX assignments
pin 313 nodes to Float. This is an accuracy/layout choice in this plan, not a
reason to move the current CUDA FP16 head projection to TF32.

### Repeated transformer realization

All 33 transformer blocks realize the same broad chain:

`RMSNorm/Myelin -> packed QKV GEMM -> K-RoPE kgen + Q-RoPE kgen -> fused MHA
scores/scale/softmax/PV -> out-projection GEMM -> residual/RMS Myelin -> wide
linear1+gate GEMM -> SwiGLU kgen -> linear2 GEMM -> residual/next RMS Myelin`

Counts are 33 each for packed QKV, MHA, out-projection, wide FFN linear1+gate,
SwiGLU, and linear2; Q-RoPE and K-RoPE contribute 66 separate generated kernels.
There are also 11 outer pre-projections and 11 outer post-projections.

The two dominant TensorRT GEMM tactic names occur 110 and 44 times:

- NN `128x128x32`, stage 4, `2x2` warps;
- TN `128x256x32`, stage 3, `2x4` warps.

The names contain `sm80_xmma`, but the plan was profiled on CC 12.0. They identify
TensorRT's internal implementation family, not an instruction to compile the
current CUDA kernels for SM80 or to copy a surface tile constant.

## TensorRT internal layer attribution

The following is the sum of TensorRT's separate profile-run layer times. Percent
shares are internal to the `7.140120 ms` profiled-layer sum.

| Logical group | Time (ms) | Share |
| --- | ---: | ---: |
| FFN linear1 + gate, 33 | 1.227542 | 17.19% |
| FFN linear2, 33 | 0.886127 | 12.41% |
| Attention MHA, 33 | 0.866026 | 12.13% |
| Attention packed QKV, 33 | 0.783035 | 10.97% |
| Transformer residual/RMS family, 99 | 0.769828 | 10.78% |
| Q/K RoPE, 66 | 0.722550 | 10.12% |
| Attention out-projection, 33 | 0.513906 | 7.20% |
| FFN SwiGLU, 33 | 0.376840 | 5.28% |
| Outer affine/SiLU/residual family | 0.308438 | 4.32% |
| Outer pre-projection, 11 | 0.220428 | 3.09% |
| Outer post-projection, 11 | 0.211026 | 2.96% |
| Heads | 0.143068 | 2.00% |
| Shape runtime layers, 133 | 0.064998 | 0.91% |
| Frontend initial convolution family | 0.029129 | 0.41% |
| Frontend global family | 0.017180 | 0.24% |

Residual/RMS sub-boundaries are `0.269396 ms` after attention,
`0.346790 ms` after FFN, and `0.153642 ms` at outer entries. This matters when
comparing apparently fast TensorRT GEMMs with CUDA: TensorRT moves residual and
normalization work into adjacent generated kernels.

## Directional comparison with the current CUDA graph

Stage 38's current CUDA Nsys attribution is a natural-S2 full graph, whereas the
TensorRT numbers above come from `trtexec`'s separate profile run. Therefore the
table is a boundary sanity check only; no cross-backend percentage is calculated.

| Boundary | TensorRT diagnostic | Current CUDA Stage 38 | Interpretation |
| --- | ---: | ---: | --- |
| FFN linear1/gate + SwiGLU | `1.6044 ms` | fused `1.4794 ms` | Do not split current fused FFN |
| Q/K RoPE | `0.7226 ms` | combined `0.1779 ms` | TensorRT's two-kernel layout path is much worse directionally |
| Attention | `0.8660 ms` | FA4 `0.5920 ms` | No evidence to replace FA4 with TRT-style MHA |
| Packed QKV | `0.7830 ms` | `0.7851 ms` | Roughly equal across incompatible protocols; no new plan win |
| Out-proj + post-attn residual/RMS | `0.7833 ms` | out-proj/residual + FFN RMS `0.7318 ms` | TRT's smaller bare GEMM is offset by shifted boundary work |
| Outer pre + post projections | `0.4315 ms` | `0.4357 ms` | Similar magnitude; exact-grid geometry is a hypothesis, not a measured win |

This eliminates the tempting but incorrect interpretation that TensorRT's bare
linear2 or out-projection tactic should replace the CUDA fused/residual boundary.
The complete calling boundary is the unit that must win.

## Transferable candidates, ranked

### 1. Partial C288 no-split g1+v1 head

TensorRT layer 545 combines `model.value_head.conv1` and
`model.policy_head.conv1g`, and folds each slice's affine scale/bias into the same
correlation tactic; `model.policy_head.conv1p` remains a separate layer. The
structural grouping is useful even though TensorRT executes this head region in
Float/TF32 and is not faster than the current CUDA FP16 narrow kernels.

Proposed falsifiable CUDA experiment:

- concatenate only the g1 C96 and v1 C192 weights into one C288 weight;
- produce one `[4693,288]` FP16 row-major output;
- make the first g1 and v1 consumers accept explicit row stride and slice offset;
- leave P1 and its current consumer chain unchanged;
- do not materialize two compact outputs or add a split kernel.

The first experiment should keep the current CUDA affine/SiLU arithmetic as
stride-aware consumers. Folding their scale/bias into the GEMM epilogue is a
second, separate accuracy/schedule hypothesis; it should not be bundled merely
because TensorRT does so in Float.

Mechanism: one wider GEMM reuses the C768 trunk-tip input and removes one GEMM
launch/input reread without requiring P1 plumbing. The current g1/v1 projection
work is `12.2 + 15.8 = 28.0 us/fwd`; eliminating all of it would be an impossible
`28/(6743-28) = 0.42%` ceiling. A practical stop budget is `0.1%--0.25%` S2.

Acceptance requirements: exact B13/19x19 dispatch with ordinary fallback,
no intermediate split, no spill, complete g1/v1 consumer-boundary timing, full
FP32-reference replay because GEMM accumulation grouping changes, and natural-S2
whole-graph ABBA/BAAB. If C288 does not beat the two complete narrow boundaries,
the full C384 path still should not be inferred to win.

### 2. SM120-native exact-grid outer projection family

Representative launch data:

| Boundary | Implementation | Block | Grid | Reg/thread | Dynamic smem | Waves/SM |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| C768->C384 pre | current CUDA | 128 | 148 | 164 | 81.92 KiB | 0.87 |
| C768->C384 pre | TensorRT | 128 | `37x3=111` | 186 | 65.54 KiB | 0.65 |
| C384->C768 post | current CUDA | 256 | 148 | 154 | 73.73 KiB | 0.87 |
| C384->C768 post | TensorRT | 256 | `37x3=111` | 172 | 73.73 KiB | 0.65 |

The exact-grid geometry removes 37 CTAs per layer, or 814 CTA launches across the
22 outer projections in a forward. It also raises register use and launches fewer
blocks than the 170 SMs; NCU warns about the 0.65-wave underfill. Thus its possible
value is reduced S2 contention/padding work, not higher nominal occupancy.

Stage 28 already rejected conventional CUTLASS2 expand/contract searches: the
contract route became consistently slower in longer S2 runs and expand was at
noise level. The only valid reopening condition is a materially different
SM120-native CuTe/TMA/JIT mainloop that achieves the 111-block exact grid without
copying the old rejected tactic family. Preserve residual epilogues and compare
the complete boundary. Plausible whole-graph budget: `0.1%--0.4%`; stop quickly if
NCU does not show reduced issued work/sectors or natural-S2 interference.

### 3. Frontend broadcast/reformat/affine fusion

TensorRT separately computes the padded input-global projection and initial
spatial convolution, then Myelin fuses broadcast add, NCHW-to-NHWC movement, and
the first C768 affine/bias/SiLU. That generated layer is `0.018206 ms` in the TRT
diagnostic.

The transferable boundary is to fold the initial-global broadcast add into the
first affine/SiLU/reformat consumer, ideally on top of the already identified
cuDNN frontend engine-47 initial convolution. The current initial-global matmul
and broadcast add are only `5.0 + 10.1 us/fwd`, and the initial convolution is
`24.2 us/fwd`, so this is a low-ceiling follow-up (`~0.05%--0.15%`), not a reason
to displace FFN/QKV or C288 work.

### 4. Trunk-tip terminal fusion

TensorRT's final generated layer fuses the last outer residual add,
NHWC-to-NCHW movement, affine/bias/SiLU, and Half-to-Float. It takes
`0.011569 ms` in the diagnostic. Current CUDA trunk-tip norm/SiLU is only
`8.1 us/fwd` plus small downstream conversion/reformat work. This is a valid
boundary hint but has a sub-`0.15%` ceiling and should be attempted only if it
eliminates a materialized layout/cast while preserving the head input contract.

### 5. Tiny-head output padding, lowest priority

TensorRT often emits aligned 4- or 8-channel outputs and slices down to logical
1/2/3/6-channel head outputs. This may improve its convolution tactic shape, but
the current CUDA/library path may already pad internally. It is only worth a
microbenchmark if a later full-graph profile promotes a tiny head to a hotspot.

## Non-transferable or already superseded plan choices

- **Separate Q- and K-RoPE kernels:** 66 launches and `0.7226 ms` internally;
  current combined half2 RoPE is already much smaller.
- **Separate FFN SwiGLU:** current CUDA fuses projection/gate/SwiGLU and has the
  better complete boundary directionally.
- **TensorRT MHA tactic:** current FA4 is already the stronger boundary.
- **Bare linear2/out-projection tactic timing:** residual and RMS work has merely
  moved to adjacent Myelin layers; compare the total boundary.
- **FP32/TF32 head projection:** only the C288 grouping is transferable. Copying
  its precision/layout would add conversions and change the established accuracy
  contract.
- **Runtime reshape/cast pruning:** those 133 plan layers arise from TensorRT's
  dynamic batch profile. The current CUDA B13 AOT graph does not pay the same
  runtime shape graph.
- **Input mask pruning:** mask is unused, but fixed 19x19 mask work is outside the
  requested optimization scope and there is no current CUDA kernel cost to remove.
- **Surface tactic names/layouts:** internal `sm80_xmma` labels and TensorRT/Myelin
  storage are not portable implementation recipes.

## Reproduction commands

Device mapping:

```bash
/usr/local/bin/cuda-device-map
nvidia-smi --query-gpu=index,pci.bus_id,uuid,name --format=csv,noheader
```

Fresh-plan build and natural-S2 KataGo benchmark:

```bash
gpu-lock with --gpu 2 -- \
  /workspace/katago/build-trt/katago benchmarknn \
  -config /workspace/bench-trt-gpu2-5090d-s2.cfg \
  -model /workspace/models/b11c768h12nbt3tflrs-fson-silu.bin.gz \
  -iterations 1000 -warmup 20 -batch-size 13 -boardsize 19 -json \
  -override-config \
  "homeDataDir=/workspace/results/rebuild/stage44-cross-backend-audit/trt-plan/home,trtDumpDebugPlanToDir=/workspace/results/rebuild/stage44-cross-backend-audit/trt-plan/debug"
```

Two-context TensorRT layer diagnostic:

```bash
gpu-lock with --gpu 2 -- trtexec \
  --device=2 \
  --loadEngine=/workspace/results/rebuild/stage44-cross-backend-audit/trt-plan/debug/plan_19x19_fp16_exact.plan \
  --shapes=InputMask:13x1x19x19,InputSpatial:13x22x19x19,InputGlobal:13x19x1x1 \
  --duration=0 --iterations=50 --warmUp=200 --infStreams=2 \
  --noDataTransfers --separateProfileRun \
  --exportProfile=/workspace/results/rebuild/stage44-cross-backend-audit/trt-plan/trtexec-profile-infstreams2.json
```

The two NCU launches used the same engine/shapes under `gpu-lock with --profile
--gpu 2`, `trtexec --device=2`, one inference stream, `--noDataTransfers`, and a
single `LaunchStats` capture selected by the exact tactic-name regex. The pre
capture selected NN `128x128x32 stage4`; the post capture selected TN
`128x256x32 stage3` after skipping the first three matching launches. The report
files preserve the exact process, kernel, device, launch geometry, and metrics.

## Artifacts and hashes

Source checkout `/workspace/katago` was clean at
`5587388321cdb45f56e78c917eb2665e129d0572`. The TensorRT executable reports
revision `090caa6115c2ae86a75839d1b4fddeacd23d7444-dirty-trt` and TensorRT 10.16.1.

| Artifact | SHA-256 |
| --- | --- |
| `/workspace/katago/build-trt/katago` | `95d8d17612819011eba0b5e12926081ecc2b9430fb35b696865e91d24f11bc77` |
| Model | `1881600caab9e9d85a3dd6a019e9b8e7d2c237b5f984e13ed49a8645be3077c6` |
| `debug/plan_19x19_fp16_exact.onnx` | `cbbb8efa532e1ba0be827a4677792680fbdee93d889de9964056a7e5b950e626` |
| `debug/plan_19x19_fp16_exact.plan` | `ac1c4b2d47e0faf2af6632b5f62bda5c9f81ce572a252e8aecd973498bface41` |
| `debug/plan_19x19_fp16_exact.engine.json` | `bddd5a89f2650d6808dab71bd401eb375e3e9cac350c8d47e90bbd66b51f8cfb` |
| Fresh RTX 5090 D timing cache | `da4ea3b9ade67636c6e3fa7eebf80bdfe2d96a4406eff3fe84b6a410717c3712` |
| `trtexec-profile-infstreams2.json` | `399271daca0672a0a13ea975566eac7d8f32215bbd93382c0fd2b94ce7d4a72d` |
| `trt-outer-pre-launch.ncu-rep` | `1059922871183c86b984d67908bfb6990cc71b4a346b2d81e6922f2b817d3e68` |
| `trt-outer-post-launch.ncu-rep` | `d3d9304fa7ed53900d26b17bb5f5a7d40fc09cbec17f6f8ecd514b45dd107c40` |

The initial device audit mistakenly overrode KataGo to CUDA ordinal 1, which is
PCI `c1:00.0`, RTX 4090. Those artifacts are retained under
`sm89-mistake-debug/` and `sm89-mistake-home/` as **contaminated/invalid for this
audit**. They were not used for any conclusion above and were not deleted, in
accordance with the optimization-history requirement to preserve contaminated
results.

## Overall priority implication

This plan audit does not overturn the Stage 38/39 full-graph ranking. FFN fused
linear1/gate/SwiGLU, linear2, QKV, out-projection, and FA4 still dominate current
CUDA S2 work/interference. TensorRT mainly contributes two bounded hypotheses:

1. C288 g1+v1 no-split head, now independent of fused P1;
2. exact-grid SM120-native outer GEMMs, only as a new mainloop family.

Both should go through the normal sequence: single hypothesis, local complete
boundary, NCU mechanism check, natural-S2 whole graph, and full-output accuracy.
They should not trigger an unbounded tactic sweep or delay a higher-ceiling FFN
or QKV experiment that already has current-CUDA profiler support.
