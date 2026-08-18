# SM103 B29 optimization stages

This is the append-only engineering log for B300/B29 optimization. An entry is
not an accepted tactic unless it records correctness, whole-graph S2, profiler
evidence, and a dedicated retained commit.

## Stage 00 — fixed batch and controls

Status: complete; contract/instrumentation only.

- Target: NVIDIA B300 SXM6 AC, CC10.3, `sm_103a`, B29/S2, R10469.
- Model SHA-256:
  `1881600caab9e9d85a3dd6a019e9b8e7d2c237b5f984e13ed49a8645be3077c6`.
- TensorRT 10.16.1.11 fixed baseline binary SHA-256:
  `883024dc8bbc02e7f6b05b0431034652931acc760b76e7fd455dc996af278612`.
- Long confirmation: `6733.719141` combined nnEval/s, 5 samples × 1000
  iterations, relative spread `0.0044150056421250064`.
- Fixed anchor:
  `final-migration/plans/sm103/b300-b29-s2/baseline-anchor.json`.

Accuracy control against the same CUDA FP32 replay:

- TensorRT 10.16 policy probability worst request: abs `0.01109946`, RMSE
  `0.00062217`.
- TensorRT value probability worst request: abs `0.03413337`, RMSE
  `0.02786641`.
- TensorRT score raw worst request: abs `0.46197557`, RMSE `0.19242840`.
- TensorRT ownership probability worst request: abs `0.00732553`, RMSE
  `0.00153762`.

Acceptance policy: a retained B300 tactic must remain in the same numerical
error regime as this TensorRT 10.16 control, in addition to aggregate gates.
The executable interpretation is fail-closed and per metric: every head's
worst-request maximum-absolute error and maximum RMSE must be at most `2.0x`
its TensorRT control value. This admits the official CUDA FP16 control (largest
observed ratio `<1.65x`) while rejecting decimal-order drift. The immutable
control and replay identities are recorded in
`final-migration/plans/sm103/b300-b29-s2/request-gate-control.json`.

## Stage 01 — cuDNN Frontend OSS GEMM+SwiGLU

Status: profiling; not accepted; no optimization commit.

Candidate:
`cudnn-fe-1_27-oss-dense-gemm-swiglu-fp16-b29`.

Isolated kernel evidence:

- S1 direct API: `0.028600 ms` vs torch two-GEMM+SwiGLU `0.044241 ms`
  (`1.547x`).
- S2 concurrent round: `0.053456 ms` vs `0.088634 ms` (`1.658x`).
- Isolated max-absolute error: `2.92376e-4`.
- Artifact target: `sm_103a`; PIC object SHA-256:
  `6375aeabdfd7825e5a8435cd62e2124297bf031e5884756b79e49b6f0bb632a0`.

Whole-graph discovery:

- Official CUDA S2 short control: `4953.691655` nnEval/s.
- Candidate S2 short result: `5667.090819` nnEval/s (`+14.40%`).
- Activation marker observed on both server lanes.

8192-row replay against the same CUDA FP32 reference:

- Aggregate gates are small (`top1=0.995117`, policy RMSE `0.0003144`, value
  RMSE `0.003456`, ownership sigmoid RMSE `0.0006558`).
- Worst-request policy abs/RMSE: `0.16795908 / 0.01229177`.
- Worst-request value abs/RMSE: `0.06489986 / 0.05299003`.
- Worst-request score abs/RMSE: `0.52620357 / 0.24151874`.
- Worst-request ownership abs/RMSE: `0.11509225 / 0.01542100`.

Conclusion before profiling: performance is promising, but policy and
ownership worst-request errors are about an order of magnitude above the
TensorRT 10.16 control. The tactic is not retainable in its current numerical
form. Do not commit it as an effective optimization.

Next evidence:

1. Add the legacy FP16 projection-rounding point without changing the tile,
   accumulator, activation approximation, or AB12 behavior.
2. Run the isolated official-semantics check and 8192-row TRT16-relative gate.
3. Only after correctness passes, repeat the same NSYS/NCU protocol and test
   removal of unused AB12 stores as a separate optimization stage.

Profiler evidence (NSYS `2025.3.2`, NCU `2025.3.1`):

- Whole-graph NSYS used identical B29/S2, 50-iteration runs. The profiler-side
  throughput changed from `4707.997048` to `5298.712167` nnEval/s (`+12.5471%`)
  and summed GPU-kernel time from `1001695068` to `894897414 ns` (`-10.6617%`).
  These profiler numbers establish attribution; they are not substitutes for
  an unprofiled long throughput gate.
- Across the trace, `8184` projection GEMMs (`98.567142 ms`) plus `4092`
  standalone SwiGLU launches (`138.796411 ms`) became `4092` fused launches
  (`101.216966 ms`, average/median `24.7353/24.704 us`). The exact grouped work
  is `2.3451x` faster. The linear2 GEMM remains unchanged.
- Targeted NCU profiled exactly one isolated native-C-ABI fused launch, not the
  whole application. It launches 148 blocks x 192 threads, one wave over all
  148 SMs, uses 104 registers/thread and `214016 B` dynamic shared memory,
  has no spills, and reaches `9.38%/8.89%` theoretical/achieved occupancy.
- NCU reports compute/tensor utilization `40.21%/35.21%`, memory `62.62%`,
  L1/TEX `72.01%`, L2 hit `75.82%`, and DRAM only `8.73%`. Schedulers have
  `1.48` active but `0.21` eligible warps; `80.21%` of active cycles have none.
  Long-scoreboard stalls account for `3.87/7.52` cycles (`51.47%`).
- Therefore no evidence supports a blind tile sweep, DRAM optimization, branch
  work, or spill work. Once numerical parity is restored, the measured next
  performance target is the unused `48241152 B` AB12 store/shared pipeline.

Raw evidence (local, intentionally ignored; compact metrics are tracked in
`final-migration/plans/sm103/b300-b29-s2/evidence/`):

- `baseline.nsys-rep` SHA-256:
  `1866ffe3480ae391faf69c021a596d0e98f9cf2574f50471bec5378e6c297d9d`.
- `candidate.nsys-rep` SHA-256:
  `c6d5db167726e39243c9ea520683c778618ae20cee613d9788c992fe33955de4`.
- main `candidate-ffn.ncu-rep` SHA-256:
  `4626979847f78f2c38ce1694cf95711af355891b97265b641bbe416aaf1f9e95`.
- stall `candidate-ffn-stalls.ncu-rep` SHA-256:
  `20b1505ae68c56b671cec5b3d0a117daec5e6bedfd5f012ff67995c7abc0b664`.

The complete commands are preserved in the local profiler summary. NCU used
an exact kernel-name regex, `--launch-count 1`, and only `LaunchStats`,
`Occupancy`, `SpeedOfLight`, compute/memory/scheduler/work-distribution plus a
second `SourceCounters`/`WarpStateStats` pass. Both runs followed an idle-GPU
process check.

## Retrospective discovery closures

These candidates were evaluated before the Stage 01 profiler protocol was
fixed. They are recorded here so that the negative results are not repeated.
They are not effective optimizations and have no retained optimization commit.

### FlashInfer CuTe-DSL attention control

Status: dropped as a standalone operator; source retained only as a fusion
reference.

- Exact shape B29/S361/H12/D32, FP16, noncausal, scale `1/sqrt(32)`.
- Correctness passed; observed maximum absolute error was approximately
  `5.38e-5`.
- Identical coordinator-event timing, S1/S2: FlashInfer
  `0.043824/0.072512 ms`, cuDNN SDPA `0.039424/0.063600 ms`, and forced Torch
  Flash `0.048672/0.081728 ms`.
- FlashInfer was `11.16%/14.01%` slower than the cuDNN control, so it did not
  enter whole-graph profiling. This closes the standalone D32 attention path;
  its B300-native exp2 implementation remains useful source material.

### Triton persistent-TMA plain GEMM controls

Status: dropped; AOT/C ABI retained only as a fusion substrate.

- Both exact B29 kernels compiled for `sm_103a` with managed ptxas 13.3 and
  contained tcgen05 MMA plus bidirectional TMA. Each stream used an independent
  `56832 B`, 128-byte-aligned workspace.
- Wide QKV was `2.099x/2.712x` slower than its S1/S2 control; dual FFN was
  `2.200x/2.258x` slower. Expanding the plain tile search is disabled.
- Cubin identities: wide QKV `cee45860...de8df`; dual FFN
  `7542d390...4d2b` (full hashes remain in the local AOT manifests).
- These results establish that tcgen05 use alone is not an optimization for
  the B29 shapes. Subsequent Triton work must remove a launch or fuse an
  epilogue and must beat the isolated control before whole-graph NSYS.

### Triton linear2 plus residual experiment

Status: isolated drop; no whole-graph entry.

- Exact FP16 `M=10469,K=1152,N=384`, FP32 dot/residual arithmetic followed by
  one FP16 store, in-place residual output, full-board only (`maskBuf==NULL`).
- Resource validation rejected s4 (`237668 B` shared memory exceeds B300's
  `232448 B` limit); s3 compiled at grid 148/block 384, `188516 B` shared
  memory, TMEM 512, and `56832 B` workspace/stream.
- Fair isolated S1/S2 timing with reset copies outside the timed region:
  candidate `0.030592/0.055584 ms`, `torch.mm+add`
  `0.020256/0.033648 ms`, and `torch.addmm` `0.020928/0.033680 ms`.
- The candidate lost by roughly `51%/65%`; it was therefore stopped before a
  whole-graph or NCU run. The artifact SHA begins `3880ca21` and the formal
  validator remains available if this path is revisited.

## Stage 01 numerical-semantics audit

Status: complete; Variant A selected; implementation/testing in progress.

The official CUDA FP16 path performs two independent half-output GEMMs, then
the SwiGLU kernel loads both half projections into FP32 and finally rounds the
product back to half. With `H` denoting FP16 round-to-nearest, its observable
boundary is:

`H(SiLU(float(H(linear1))) * float(H(linearGate)))`.

The original cuDNN-OSS fused epilogue instead applies fast exp2 and an
approximate reciprocal directly to the two FP32 TMEM accumulators, rounds only
the final C, and rounds AB12 afterward. AB12 is never read by C. Its effective
boundary is therefore:

`H(SiLU_fast(linear1_fp32) * linearGate_fp32)`.

This missing projection round-trip is the largest known, deterministic
semantic difference. Packing is correct (gate then linear1), alpha is exactly
one, and the C ABI only forwards pointers/stream. The old isolated reference
also omitted the half projection boundaries, so its `2.92376e-4` maximum error
did not establish parity with the official CUDA path.

Relative to the immutable TensorRT 10.16 request control, the original fused
candidate is:

- policy max-abs/RMSE: `15.132x / 19.756x` (fail/fail);
- value: `1.901x / 1.902x` (pass/pass);
- score: `1.139x / 1.255x` (pass/pass);
- ownership: `15.711x / 10.029x` (fail/fail).

Variant A changes one factor only: cast both FP32 projections to FP16 in
registers at the existing AB12 cast point, widen them back to FP32, and run the
unchanged fast-exp2/approx-reciprocal fused epilogue. It keeps the FP32
accumulator, 128x128 tile, 1x1 cluster, persistent grid, AB12 store, and C ABI.
If Variant A still fails, the next separately measured variant may replace the
activation approximation with the official `expf + divide` order. Changing
the accumulator or tile is not justified by current evidence.

## Stage 02 — portable proven-tactic baseline

Status: in progress; user-prioritized before further SM103-native tuning.

Rationale: new tcgen05 candidates must be compared against a CUDA graph that
already contains the architecture-neutral optimizations proven on SM89/SM120.
Otherwise B29/S2 contention and headroom are mischaracterized. Variant A is
preserved as an uncommitted experiment and uses no GPU during this stage.

The Stage 01 baseline NSYS determines the first port order:

1. beta=1 residual GEMM to remove the `8184` standalone residual-add launches
   (`13.96%` of baseline kernel time);
2. C384 RMSNorm and fused Q/K RoPE (`10.83%` and `10.55%` of baseline time);
3. wide/strided QKV and wide FFN launch reduction, plus portable affine SiLU;
4. persisting-L2 windows, model-weight sharing, initial-global and policy/head
   pointwise paths after their exact guards are verified.

Only implementations built from plain CUDA, cuBLAS/cuBLASLt, cuDNN, or CUDA
runtime calls are eligible for this reuse stage. SM89/SM120 AOT objects,
CUTLASS architecture tags, FA cubins, tile winners, and device-resource
assumptions are explicitly excluded. The adapter must be opt-in on exact
CC10.3 and reject any unsafe SM120 option before model construction.

Discovery may enable a coherent portable bundle to close the baseline quickly,
but retention will be decomposed by profiler-guided ablation. Each effective
method receives its own long gate and commit; raw S2 throughput is not used as
attribution evidence without the corresponding whole-graph NSYS delta.

Discovery result:

- Same-binary backend-off: `5018.311535` nnEval/s; portable-empty:
  `5010.303978` and `5014.403463` nnEval/s. The adapter itself has no positive
  performance signal.
- The first high-confidence bundle produced `7014.624278` and `7061.740970`
  nnEval/s, already above TensorRT. Full-board activation markers appeared on
  both lanes for every selected hook.
- Accumulated 100-iteration ablation exposed two negative transfers: L2 trunk
  plus inner was `-1.886%`, and adding fused-policy/head-BN afterward was
  another `-1.642%`. Both are excluded from the B300 portable baseline.

Profiler attribution, same binary and identical B29/S2 command:

- backend-off NSYS: `4703.198880` nnEval/s, `986013223 ns` summed kernel time,
  `70036` launches;
- retained portable set: `6930.278801` nnEval/s, `753574298 ns`, `57592`
  launches;
- deltas: `+47.352%` profiler-side throughput, `-23.574%` summed kernel time,
  and `-17.768%` launches;
- standalone residual-add launches fall from `8052` to zero;
- RMSNorm changes from `106.790629 ms` to `50.570860 ms`;
- Q/K RoPE changes from two launches totaling `103.731177 ms` to one fused
  launch family totaling `44.269752 ms`;
- affine-SiLU changes from `63.926040 ms` to `35.834133 ms`;
- SwiGLU changes from `137.043318 ms` to `116.029728 ms`.

Long accumulated ladder (each sample is 1000 timed iterations; all values are
combined B29/S2 nnEval/s):

1. portable-empty: `5018.968251 / 4981.821598`, median `5000.394925`, spread
   `0.743%`;
2. beta=1 residual: `5617.302900 / 5583.296655`, median `5600.299778`,
   `+11.997%`, spread `0.607%`;
3. C384 warp4-vec8 RMSNorm: `5978.789799 / 5969.539899`, median
   `5974.164849`, `+6.676%`, spread `0.155%`;
4. planar fused Q/K RoPE: `6497.882942 / 6492.744926`, median `6495.313934`,
   `+8.723%`, spread `0.079%`;
5. half2 affine-SiLU: `6712.092932 / 6763.835599`, median `6737.964266`,
   `+3.736%`, spread `0.768%`;
6. half8 C1152 SwiGLU: the first warmup-80 sample was discarded after a
   `1.72%` pair spread; warmup-stable samples are `7150.536082 / 7149.647579`,
   median `7150.091831`, `+6.116%`, spread `0.012%`.

The retained portable set is `+6.183%` over the fixed TensorRT 10.16 median.
Its 8192-row replay passes all aggregate gates and every TensorRT-relative
request gate. Request error ratios range from `0.720x` to `1.097x` of the
TensorRT control; policy top1 vs FP32 is `0.997925`.

Evidence identities:

- binary SHA-256:
  `b514d85c5e4d8b33dc0ab2eb26a26030d199d6a5d53a1c48b6b66c21bb3f8738`;
- backend-off NSYS SHA-256:
  `3d99d889d980f60e0a65c873cd24f0a3330558ec1792744d4bfb6fb58ac90984`;
- retained-set NSYS SHA-256:
  `f38732ce3c9900f0746f834f93cc8cc1533bb95adeb90c4b4e0ba9552bb8dcfc`;
- replay SHA-256:
  `8a62462357c81cc6c50974a95b31d47f8dfd62ee3eadf9ee923392dba6942d22`;
- comparison SHA-256:
  `5745c12600a64995b5ac47bbe12ffebd74b2b7eb46f288c4c1086d22ad4a0f0c`.

Status after gates: five portable methods are individually effective in the
accumulated graph. They remain uncommitted until the paused FFN experiment is
removed from the core diff and each retained config delta can be committed
separately.

Commit/cleanup closure:

- The paused cuDNN-OSS hook was removed from the core/CMake diff before the
  portable adapter commit; its Python/bridge sources and local AOT artifact
  remain available for the later SM103-native loop.
- Clean-split binary SHA-256:
  `a00157f08a37b91bbf12956d36f6c92fae06f7d9a0856d614a6a38499e7b0625`;
  the retained accumulated graph rechecked at `7139.417690` nnEval/s.
- Portable adapter infrastructure commit: `be6b4107`.
- B29/TRT accuracy-control commit: `10afb5ca`.
- beta=1 residual commit: `6723ea0c`.
- C384 RMSNorm commit: `abdeb8e7`.
- planar fused Q/K RoPE commit: `f88e423a`.
- half2 affine-SiLU commit: `23f42fb9`.
- half8 C1152 SwiGLU commit: `1c2aad4c`.

Portable reuse is now exhausted for the high-value SM89/SM120 methods. The
next stage returns to whole-graph B29/S2 profiling on the accumulated portable
baseline. Architecture-bound SM89/SM120 artifacts are not reuse candidates;
their graph boundaries are hypotheses to reimplement with the expanded SM103
toolchain (cuDNN Frontend, CUTLASS/CuTe DSL, Triton, TileLang, FlashInfer,
Liger, MSLK, or hand-written CUDA).

## Stage 03a — SM103 tcgen05 fused FFN

Status: retained; fast FP16-roundtrip variant.

The accumulated portable graph exposed the two FFN projections plus SwiGLU as
`219.652845 ms` of `753.574298 ms` summed kernel time (`29.15%`). Targeted NCU
on one current projection found a 420-block 2-CTA launch with 2.84 waves,
`231.42 KiB` dynamic shared memory, `12.5%` theoretical occupancy, `81.96%`
no-eligible scheduler cycles, and a 124-block partial-wave tail. This supported
a one-wave persistent fusion rather than another tile sweep.

The retained cuDNN-Frontend 1.27/CuTe DSL 4.7 derivative performs both tcgen05
projections, forces each projection through the official FP16 rounding point,
and applies fast SwiGLU in one 148-block launch. Full-graph NSYS verifies that
8052 projection/SwiGLU launches disappear; the fused boundary is `1.987x`
faster and graph throughput rises `13.506%` over the portable baseline.

Five 1000-iteration B29/S2 samples are
`8071.385110, 8072.391535, 8089.072732, 8077.636942, 8084.336061` nnEval/s;
median is `8077.636942`, spread `0.219%`. This is `19.958%` above the fixed TRT
10.16 baseline.

Bad-case study: fast math's worst row is row4124/move358, a real full-B29 row,
not tail padding. Inputs/targets are byte-identical; the move is legal; raw
policy-logit correlation remains `0.9999865`; there is no permutation, fixed
index, or spatial-layout signature. Strict exp/div fixes this row but reduces
full throughput to `5554` nnEval/s. One Newton reciprocal refinement preserves
throughput but merely moves the worst case to row7111/move111. The error is an
accumulated numerical compromise, not an implementation error. Per user
direction, the implementation-error request guard is calibrated to `2.25x`
TRT; fast's maximum request ratio is `2.148x`, while the broken no-roundtrip
implementation remains at `10-20x` and fails closed.

## Stage 03b — remove unused fused-FFN AB12 output

Status: retained; cumulative on Stage 03a fast math.

KataGo consumes only fused C, but the provider kernel also wrote the two FP16
projections as a `48,241,152 B` AB12 tensor per stream. The bounded derivative
retains the external ABI/shape argument yet removes the AB12 device pointer,
descriptor, four-stage shared ring, register/shared copies, and all TMA stores.
The tcgen05 mainloop, FP16 projection round-trip, fast SwiGLU, and C store are
unchanged.

Targeted NCU verifies the exact hypothesis:

- dynamic shared memory `214016 -> 181248 B`;
- registers/thread `74 -> 66`, spills remain zero;
- L2 sectors `2262595 -> 754848` (`-66.64%`);
- replay duration `44.58 -> 40.10 us`;
- the 148-block/192-thread one-wave launch is unchanged.

Full-graph NSYS improves `7866.259705 -> 8315.914450` profiled nnEval/s;
fused-kernel time falls `110.523899 -> 95.244823 ms`. Five unprofiled
1000-iteration samples are `8552.534123, 8623.889233, 8592.663766,
8578.288206, 8587.808904`; median `8587.808904`, spread `0.831%`. This is
`+6.316%` over Stage 03a and `+27.534%` over TRT 10.16.

The full replay is not byte-identical to the parent because changed resource
scheduling perturbs final reduction ulps, but accuracy improves: policy RMSE
is `9.4938e-5`, worst request returns to the common row5183 family, maximum TRT
request ratio is `1.608x`, and every aggregate/request gate passes.
