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

## Stage 04 — fused planar QKV + learnable RoPE resource rounding

Status: dropped after profiler-guided full-graph and resource-threshold gates.

The no-AB12 graph spends about `117.1 ms` on three QKV projections and
`45.5 ms` on planar Q/K RoPE per profiled run.  A CuTe DSL 4.7 derivative
combined all four launches into one SM103a tcgen05/TMEM/TMA kernel while
retaining planar Q/K/V output for the existing cuDNN SDPA.  Isolated
correctness had at most one FP16 ULP on rotated Q/K and bitwise V.  Isolated S1
fell to `22.756 us`, but the first full-graph B29/S2 result was only
`8048.267422` nnEval/s versus the retained `8587.808904`.

NSYS established why the single-stream result did not transfer:

- summed kernel time still improved `637.220 -> 622.476 ms`;
- kernel union regressed `426.832 -> 448.245 ms`;
- overlap fell `210.388 -> 174.232 ms`;
- the fused kernel was exclusive for `101.477 ms`, versus about `54.9 ms`
  exclusive for the old QKV+RoPE boundary;
- FFN overlap-covered time fell `45.73 -> 23.35 ms`.

NCU measured `114.82 KiB/CTA`, 58 registers/thread, two CTAs/SM, 90.03% no
eligible cycles, and 75.9% long-scoreboard stalls.  The one allowed resource
hypothesis forced the exact three-CTA shared-memory boundary.  The compiled
AB2/C1 version reached `73.86 KiB` and three blocks/SM, but serializing the TMA
store ring made S1/S2 `138.03/269.38 us`.  Removing the TMA C ring reached
about `65.66 KiB`, but the stock direct-store fragment used 172 registers,
remained limited to two CTAs, and produced `120.27/233.12 us`.  Both failed
before full-graph execution.  Evidence is recorded in
`stage-04-qkv-rope-profile.json`; no QKV runtime tactic is retained.

### Stage 04b — static-persistent G148 QKV+RoPE retry

Status: rejected after isolated correctness/NCU, same-binary A-B-B-A, and
whole-graph NSYS. No long or replay gate was run.

The Stage04 arithmetic was reopened because it had reduced summed QKV work.
CUTLASS 4.7's pinned `PersistentDenseGemmKernel` mapped the same 738 logical
M128xN128 tiles onto exactly 148 physical CTAs: 146 CTAs execute five tiles
and two execute four. The compiled schedule was AB3/ACC2/C2, block192,
`114.82 KiB` dynamic shared memory and 62 registers/thread. NCU proved two
resident CTAs/SM, 18.75% theoretical occupancy, zero spills, and exactly one
executed CTA/SM in S1. This was the intended one-CTA-per-lane resource shape.

Correctness matched Stage04: Q/K differed from the FP16 projection-round plus
FP32-RoPE oracle by at most one FP16 ULP (268/266 elements), and V was
bitwise. Isolated S1/S2 were `24.649/41.119 us`; relative to the old
nonpersistent one-launch candidate S1 regressed 8.32%, while coordinated S2
regressed only 1.36%. The isolated same-kernel pairing therefore behaved as
predicted.

The full graph did not. Same-binary B29/S2 A-B-B-A was
`8938.254 / 8512.087 / 8534.050 / 8947.018`, a `-4.692%` center regression.
NSYS still showed real local work reduction: QKV boundary sum fell
`189.732 -> 177.060 ms` (-6.68%) and its union fell 13.34%. But overlap-covered
QKV time halved `104.978 -> 50.685 ms`; exposed QKV time rose
`69.509 -> 100.533 ms` (+44.63%). Whole-graph kernel sum rose 0.56%, union rose
2.82%, and overlap fell 3.86%. FFN covered time fell 62.73%, its exclusive
time rose 41.86%, and one-SM GEMM exclusive time rose 222.33%.

This establishes the missing distinction: two repeated G148 grids can occupy
the two slots, but the real peer is often FA4 (`232 KiB`), no-AB12 FFN
(`181 KiB`), or another resource-heavy GEMM, none of which can co-reside with
the `114.82 KiB` QKV CTA. A persistent CTA on every SM lengthens that
heterogeneous blocking interval. No G148 parameter sweep followed. Complete
evidence is in `stage-04b-qkv-rope-persistent-g148-result.json`.

### Stage 04c — ordered split Q+RoPE / K+RoPE / V

Status: rejected after correctness, same-binary A-B-B-A, and whole-graph
NSYS. The QKV route is closed; packed layout was not attempted.

The one remaining scheduling hypothesis kept the pair-major table, arithmetic,
M128xN128 tile, AB3/C2 resource budget, planar outputs, and sequential DAG,
but emitted three `(82,3,1)=246`-CTA kernels. It still removed the separate
RoPE launch and did not introduce aux streams. Q/K retained the one-ULP
envelope and V remained bitwise. Isolated S1 was `31.378 us`, 1.277x faster
than the production four-launch S1 boundary; coordinated repeated S2 was
`56.688 us`.

Same-binary B29/S2 A-B-B-A was
`8947.306 / 7981.823 / 8239.891 / 8941.532`; centers were
`8944.419 / 8110.857`, or `-9.319%`. NSYS explains why shorter grids did not
transfer. Global overlap did increase `245.334 -> 297.987 ms` (+21.46%), and
QKV union fell 6.60%, so the scheduler did exploit the added boundaries.
However, Q/K/V summed service time grew `192.195 -> 259.449 ms` (+34.99%):
the boundary expanded from `40.445` to `54.598 us` under real contention,
despite its `31.378 us` isolated result. FA4 slowed 6.63%, one-SM GEMMs slowed
22.59%, and QKV-to-FFN overlap collapsed `17.045 -> 0.046 ms`. Whole-graph
kernel sum rose 12.14%, union rose 7.40%, and trace span rose 5.98%.

Thus the split recovered 52.65ms of overlap but created 88.39ms of extra
kernel service time. Both the long persistent and short-grid forms have now
failed for measured heterogeneous resource contention, not lookup layout,
numerical error, or lack of scheduler visibility. Complete evidence is in
`stage-04c-qkv-rope-split3-result.json`.

## Stage 05 — fused FFN two-resident-CTA boundary

Status: dropped after NCU proof and full-graph A/B/B/A.

The accepted no-AB12 fused FFN uses exactly one 148-CTA wave and `181248 B`
dynamic shared memory.  Since its five A/B stages consume `32768 B` each, the
bounded AB3 derivative removed exactly two stages while preserving every math,
tile, scheduler, and output choice.  The prediction was deliberately placed on
B300's resource cliff: `115712 B` dynamic plus `1024 B` driver shared memory is
`116736 B`, exactly half of `233472 B/SM`.

NCU validated the hypothesis exactly: `115.71 KiB` dynamic shared memory,
66 registers/thread, block-limit shared-memory=2, and theoretical occupancy
`18.75%`.  Tight C correctness passed and AB12 remained untouched.  However,
isolated S1 regressed `23.288 -> 24.960 us`, while S2 round remained essentially
flat at `43.517 -> 43.649 us`.

Accumulated full-graph A/B/B/A samples were
`8535.266384 / 8369.002605 / 8371.041515 / 8577.526889` nnEval/s.  NSYS
normalized per forward showed kernel union `3.499 -> 3.601 ms`, overlap
`1.724 -> 1.690 ms`, and FFN exclusive time `0.392 -> 0.507 ms`.  Thus the
hardware residency cliff was real, but K384 pipeline starvation cost more than
the scheduler could recover.  The five-stage no-AB12 parent remains retained;
the complete negative result is in `stage-05-ffn-ab3-profile.json`.

## Stage 05a — cuDNN Frontend SDPA engine surface

Status: current native engine retained; engine/knob surface exhausted.

SDPA is the largest single exclusive interval in the no-AB12 timeline:
`78.934 ms` exclusive and only `30.045 ms` overlap-covered.  Its current
engine uses grid 696, block 512, 128 registers/thread, `232.45 KiB` dynamic
shared memory, and 4.70 waves.  A dedicated cuDNN Frontend 1.27 harness
enumerated 22 engines and 18,532 knob tuples.  Only 61 finalized, 52 ran, and
34 passed correctness.

The existing engine 10 configuration remains fastest after contention-guarded
7x1000 measurement: S1 `26.2951 us`, S2 round `45.6243 us`.  Its fastest
equivalent alias measured `45.6201 us`, a `0.009%` difference with identical
resources.  Reducing TN from two to one measured `47.2577 us`, and NCU proved
that grid, block, registers, shared memory, and wave count did not change.
FROST reduced shared memory to `197.01 KiB` but required CGA2 and regressed S2
to `67.462-68.733 us`.  Engine 8 fell back to SM80 WMMA, used 158 registers and
7.05 waves, and measured about `110.9 us`; 18 manually forced engine-8 tuples
were also numerically wrong.  The search result SHA is
`d61a2dd888bac2b86af0a772dc8932829187edf9c1edf3ad574bdd2c58ad7d32`.
Further SDPA work therefore requires a new SM103 algorithm, not another cuDNN
knob sweep.

## Stage 05b — final-linear2 + residual + C384 affine-SiLU

Status: fusion boundary closed after three isolated implementations.

Call-graph and NSYS adjacency checks showed that all 1342 C384 affine launches
in the profiled run are immediately preceded on the same stream by the final
linear2 residual GEMM.  This made the complete boundary a valid fusion target.
The unfused control measured S1/S2 `14.609/27.819 us`.

The initial exact CuTe implementation was `91.409/157.885 us`.  A coalesced
exact epilogue reduced it to `77.896/68.592 us`, but NCU still showed 128
registers/thread, 6.25% occupancy, 86.69% no-eligible cycles, and too little
epilogue parallelism.  The single allowed 64x64 fast packed-SFU variant kept
residual bit-exact and affine output within `1.526e-5` max abs, but its shallow
TMA ring serialized the boundary to `546.991/1053.751 us`.  None entered the
full graph; the core/CMake path remains unchanged.

## Stage 06 — native SM103a FA4 D32 attention

Status: retained; cumulative on portable tactics plus the no-AB12 fused FFN.

Stage 05a exhausted cuDNN's engine and knob surface at S1/S2
`26.2951/45.6243 us`, with a 696-block, 4.70-wave kernel contributing
`78.934 ms` exclusive critical time. Local FlashAttention main at commit
`0251105a` contains a true SM103 path: native exp2, FP32 QK/PV accumulation,
no D32 `tcgen05.ld.red`, and a static-persistent M128N128/q2 schedule whose
derived KV ring has 24 stages. The unmerged `origin/subtiling` seed `526c18d`
was audited but not used: its M64 SMEM-P constructor explicitly rejects SM103
and was tested only on SM100.

The exact B29/S361/H12/D32 planar-QKV control passed the isolated FP32 check
at `4.8804e-4` max absolute error and `2.4891e-5` RMSE. Direct-launch medians
were S1 `21.244 us` and S2 round `37.701 us`, respectively `1.238x` and
`1.210x` faster than cuDNN. Native NCU then measured grid 148, block 512,
one wave, 128 registers/thread, `232.45 KiB` dynamic shared memory,
25.0/23.15% theoretical/achieved occupancy, 53.97% no-eligible cycles, and no
local-memory spills. The one-wave persistent work distribution, rather than a
smaller resource tile, is the material difference from cuDNN.

The first C export exposed a real ABI defect before any performance result was
accepted. CuTe DSL 4.7 could not generate C for the upstream PEP-604
`max_seqlen_q: Int32 | int | None` annotation. Removing that argument from
the header shifted the stream and return slots; cuda-gdb showed garbage
`grid.x`/stream fields and a SIGSEGV inside `cuLaunchKernelEx`. The retained
generator instead rectifies only the C-export annotation to `cutlass.Int32`,
keeps the runtime slot, and the native bridge supplies exact value 361. CMake
authenticates source identity and header/object/runtime hashes, links only a
build-private object copy, and the default build remains a fail-closed stub.

Exact-warmup B29/S2 NSYS contains 4752 FA4 calls and zero cuDNN SDPA calls;
the same-binary control contains 4752 cuDNN calls. Kernel sum falls
`753.832 -> 729.672 ms` (-3.205%), kernel union falls
`505.850 -> 482.695 ms` (-4.577%), while overlap remains essentially flat
at `247.982 -> 246.976 ms`. Attention total/exclusive/overlap changes from
`128.900/94.229/34.671 ms` to `109.840/76.872/32.968 ms`. The savings
therefore land almost entirely on the exposed critical path, unlike the
dropped QKV+RoPE fusion.

Five 1000-iteration B29/S2 samples are
`8933.984650, 8948.551416, 8949.114828, 8963.775297, 8960.522266` nnEval/s;
median is `8949.114828`, spread `0.333%`. This is `+4.207%` over no-AB12 and
`+32.900%` over fixed TRT 10.16. The 8192-row replay passes every 2.25x TRT
request guard: policy RMSE is `9.5291e-5`, policy top-1 agreement is
`99.8291%`, and the maximum request ratio is `1.946x`. Complete profiler,
artifact, ABI-debug, throughput, and replay hashes are recorded in
`stage-06-fa4-native-profile.json`.

## Stage 06b — native FA4 QK/PV accumulator modes

Status: closed after the fixed four-mode isolated S2 gate; FP32 remains
retained.

The SM89/SM120 implementations ultimately use reduced QK/PV accumulation, so
the same control, QK16, PV16, and both16 matrix was tested on the retained
SM103a FA4 kernel. Every non-accumulator axis remained fixed at
B29/S361/H12/D32, M128N128, q2/kv24, and static persistence. Upstream
FlashAttention commit `0251105a` exposes no accumulator switch and hard-codes
both types to FP32. Attribute-only FP16 substitutions therefore failed at
the FP32-specific TMEM copy layouts. The compile-failure hashes are
`c50cbc57` (QK16), `1b68eed2` (PV16), and `65a30350` (both16).

An isolated, hash-locked typed derivative covered the complete data path
rather than casting around the compiler error. TCGen05's leading-2 FP16
layout dimension is a packed M/datapath mode, not a halved N-column footprint;
all modes retain physical TMEM offsets S `0/128`, P `64/192`, O `256/288`, and
320 total 32-bit columns. QK16 uses a packed TMEM load followed by immediate
FP32 softmax arithmetic while preserving P as packed FP16 values in raw FP32
cells. PV16 requires threshold zero to avoid scaled-P overflow and requires
typed pack/unpack plus FP32 correction arithmetic and FP16 RNE writeback. A
zero-Q mean(V) case isolated the final correction error to a D16 typed-iterator
stride, after which every mode passed the random-input correctness guard.

The final S1/S2 medians in microseconds were FP32 `21.261/37.682`, QK16
`23.610/41.486`, PV16 `22.489/39.792`, and both16 `24.305/42.779`. Relative
to FP32, the reduced modes lose S2 by `10.10%`, `5.60%`, and `13.53%`.
Because none reduces the 320-column physical TMEM allocation and every one
already loses under isolated two-stream contention, they were stopped before
NCU, whole-graph NSYS, long throughput, and replay. The committed Stage 06
FP32 core is unchanged. Typed root-cause history, artifact hashes, correctness
metrics, and the full timing matrix are recorded in
`stage-06b-fa4-accumulator-modes.json`.

## Stage 07 — fused FFN M128xN64/AB4 resource reshaping

Status: dropped at the fixed isolated S2 `+10%` cutoff.

Stage 05 proved that `115712 B` dynamic shared memory admits two FFN CTAs per
B300 SM, but reducing the M128xN128 parent to AB3 starved its six-K64
mainloop.  The only follow-up shape therefore halved packed N from 128 to 64
and retained AB4/C2.  Static accounting again lands exactly on the boundary:
four `24576 B` A/B stages, two `8192 B` C stages, and `1024 B` aligned control
storage total `115712 B`.  Each N64 tile contains exactly one complete
gate-N32/linear1-N32 pair; no packed pair crosses a tile boundary.

The first native launch exposed a real N64-specific implementation race, not
a precision compromise.  Although launch status was zero, AB12 remained
untouched, and no output was NaN, output max/RMSE error was
`0.103554/0.001533`.  The inherited C-store logic counted raw packed subtiles:
with `subtile_cnt=2`, `tile_count*subtile_cnt mod C2` is always zero, so every
persistent tile reused C buffer zero while buffer one stayed idle.  Five
identical launches after reducing only the C TMA flight depth produced five
different output hashes and `213-222` corrupt M/N tiles.  The final derivative
both binds the TMA flight depth to C2 and rotates buffers by completed
gate/linear1 pairs.  Tight correctness then passed at `6.104e-5` max absolute
error and `2.292e-7` RMSE, with AB12 still untouched.

The corrected final artifact has derivative/object/DSO hashes
`eb566ede31bd... / b14be94b4524... / c006b848b953...`.  It nevertheless
measured S1 `28.795 us` and S2 round `51.642 us`, versus the accepted parent
`23.288/43.517 us`: regressions of `23.65%` and `18.67%`.  This exceeds the
user-fixed S2 shutdown threshold, so the candidate stopped before NCU,
full-graph NSYS, throughput, or request replay.  No core/CMake path changed
and no commit is created.  CPU contracts, race diagnosis, all samples, and
complete hashes are in `stage-07-ffn-n64-ab4-profile.json`.

## Stage 07a — explicit cuBLASLt QKV resource schedule

Status: retained; cumulative on native FA4 and the no-AB12 fused FFN.

The committed FA4 graph still issues three beta-zero Q/K/V GEMMs for each of
33 transformer pairs. A first B29/S2 A/B/B/A showed about `+1.18%` from the
generic shape-autotuned cuBLASLt hook, but its algorithm identity was not
stable. Four 1000-iteration processes chose the same id70 tuple and measured
`9042-9073` nnEval/s; a fifth chose two different id71 tuples across its two
handles and fell to `8780.674`. Replay also created two handle generations:
the warmup generation chose id70 while the output-producing generation chose
id71. The generic runtime autotuner is therefore retained only as a discovery
tool and is not production eligible.

The accepted candidate constructs the complete QKV tuple directly with
`cublasLtMatmulAlgoInit`, sets tile 23, split-K 1, reduction 0, swizzle 0,
custom option 0, stages 35, inner shape 0, and cluster shape 5 for algorithm
70, checks it with `cublasLtMatmulAlgoCheck`, and reads every attribute back.
Any construction, readback, or launch mismatch fails closed. Every handle in
smoke, short, replay, and long testing logged exactly that tuple and zero
workspace bytes. The explicit route is QKV-only: initial-global remains on
legacy `cublasHgemm`. An id71/tile19/stages35/cluster6 tuple was pinned as a
diagnostic control; it measured `-0.043%` and is closed.

Exact final-binary NSYS explains the gain as a scheduling trade, not lower raw
work. Across the last 50 timed forwards on each stream, both graphs launch
40,596 kernels. Exactly 9,900 launches—99 per forward—move from the 2-SM
cluster family to the 1-SM family. Kernel sum rises
`505.696 -> 526.821 ms` (`+4.177%`), but overlap rises
`171.021 -> 194.363 ms` (`+13.649%`) and interval union falls
`334.675 -> 332.458 ms` (`-0.663%`). Profiled throughput rises
`8665.936 -> 8704.812` nnEval/s. This is the intended dual-stream resource
rounding behavior.

Application-replay NCU confirms the physical change. The control is a
252-block, 2-SM-cluster kernel with 71 registers, `231.42 KiB` dynamic shared
memory, 1.70 waves, and `14.43 us` isolated duration. The id70 kernel is a
126-block, one-SM kernel with 96 registers, `215.04 KiB`, 0.85 waves, and
`19.42 us`; neither spills. The candidate is deliberately slower in isolation
and has 87.64% no-eligible cycles, but its smaller scheduling unit restores
cross-stream overlap. NCU application replay is required because kernel replay
invalidates the process-local CuTe driver-module handles after the captured
library kernel.

Removing initial-global from the Lt route also isolated the earlier ownership
request failure. Explicit id70 and diagnostic id71 produce identical replay
outputs, and id70 is byte-identical in every raw output section to the accepted
Stage 06 FA4 graph. The 8192-row comparison therefore retains policy RMSE
`9.5291e-5`, policy top-1 `99.8291%`, and maximum TRT request ratio `1.946x`;
all request and aggregate guards pass.

The final five 1000-iteration B29/S2 samples are
`9028.856178, 9058.808862, 9033.266454, 9066.341924, 9063.157243` nnEval/s.
Median is `9058.808862`, spread `0.414%`, `+1.226%` over Stage 06 and
`+34.529%` over fixed TRT 10.16. Complete configuration identities, auto-drift
evidence, profiler reports, replay hashes, and long samples are recorded in
`stage-07-cublaslt-qkv.json`.

## Stage 08 — id70 Q/K/V auxiliary-stream dependency DAG

Status: retained; cumulative on Stage 07 explicit id70, native FA4, and the
no-AB12 fused FFN.

The queue-depth audit had already disproved a host-window explanation: each
outer stream stays about 1024 descriptors and 17.45 ms ahead, but same-stream
FIFO normally leaves only the two stream heads dependency-ready. Stage 08
therefore changes the DAG without changing a single GEMM algorithm. After
pre-RMS the primary stream records one ready event; Q retains the Stage 07
primary id70 state while K and V use two persistent nonblocking aux streams,
two independent cuBLASLt handles/plans, and private workspaces. K/V record
done events and the primary waits both before the existing RoPE and FA4.
Every one of the three plans logs the complete id70/tile23/stages35/cluster5
tuple and zero algorithm workspace bytes.

The first full replay found a real lifetime bug that short benchmark processes
could not expose. `replaynn` constructs and destroys a two-handle warmup
evaluator, then creates a second output evaluator in the same process. C++
reverse member destruction originally unloaded `Sm103Model`'s native FA4
module before `Sm120Model` synchronized and destroyed the aux streams. Both a
normal replay and `CUDA_LAUNCH_BLOCKING=1` consequently failed in generation
two with FA4 `LAUNCH_FAILED (-4)`. `ComputeHandle::~ComputeHandle` now makes
the device current and calls an explicit `noexcept` synchronization of both
aux streams before reverse destruction can unload FA4. Both synchronize
statuses are checked directly and failure uses `Global::fatalError`, so no
exception can escape the destructor. No FA4 bridge change and no CUDA error
clearing were used. The same 8192-row replay then completed; the
source-contract test pins this ordering, the `noexcept` fatal path, and every
direct Lt/event API check.

The same-binary 50-warmup/300-iteration A-B-B-A measured controls
`9043.339415/9032.061925` and aux
`9178.690707/9258.774635` nnEval/s, a center gain of `2.003%`. NSYS proves
that this is scheduling, not less work. Across the last 50 forwards on each
outer stream, both graphs launch exactly 40,596 kernels and exactly 9,900 QKV
kernels. K/V move as 1650 kernels onto each of four aux streams. Candidate
minus control has exactly 9900 event records and 13,200 waits: three records
and four waits for each of 3300 attention boundaries. Kernel sum changes only
`526.993 -> 525.009 ms`, while union falls `331.831 -> 322.454 ms`
(`-2.826%`) and overlap rises `195.162 -> 202.555 ms` (`+3.787%`). QKV sum
actually grows `100.438 -> 109.451 ms`, but its union falls
`90.124 -> 66.377 ms` (`-26.349%`). Full-trace concurrency-depth-at-least-3
time grows `9.470 -> 30.456 ms`. This closes the ready-frontier hypothesis.

The altered resource schedule is not byte-identical to Stage 07, consistent
with the known schedule-sensitive accumulated FP16 rounding in this fused
graph, but the difference is tiny: policy top-1 versus Stage 07 is 100% and
probability RMSE is `2.866e-6`. Against FP32, policy RMSE remains
`9.5291e-5`, top-1 remains `99.8291%`, and the maximum TRT16 request ratio is
the existing ownership value `1.946x`; every 2.25x request and aggregate gate
passes.

Five locked 200-warmup/1000-iteration B29/S2 samples are
`9219.883683, 9201.745296, 9255.677664, 9201.663678, 9178.214200` nnEval/s.
Median is `9201.745296`, spread `0.842%`, `+1.578%` over Stage 07 and
`+36.652%` over fixed TRT 10.16. Complete DAG accounting, lifecycle
case-study, profiler, replay, source, and binary hashes are recorded in
`stage-08-qkv-aux-streams.json`.

## Stage 08b — one-aux Q/K/V topology diagnostic

Status: dropped at the same-binary short gate; Stage 08 aux2 remains retained.

The only bounded simplification of Stage 08 gave each outer handle one
persistent nonblocking aux stream and one independent id70 Lt state. Q stayed
on the primary stream; K and V ran sequentially on the single aux stream. The
boundary therefore used one ready record/wait and one done record/wait, four
event operations instead of aux2's seven. Every math, buffer, algorithm, and
downstream kernel remained fixed.

The 50-warmup/300-iteration symmetric sequence
control-aux1-aux2-aux2-aux1-control measured control
`9046.300071/9017.547660`, aux1 `9059.046621/9067.317222`, and retained aux2
`9238.057770/9258.301562` nnEval/s. Centers are `9031.923866`, `9063.181922`,
and `9248.179666`: aux1 is only `+0.346%` over control and is `-2.000%` behind
aux2.

The failure is structural. One aux stream adds a FIFO K-to-V edge, so only Q
and K are dependency-ready at the fork; V cannot join the ready frontier until
K completes. Saving four event APIs per attention does not compensate for
losing the third sibling grid. The fixed gate requires aux1 to beat aux2, so
the experiment stopped before NSYS, replay, request, or long testing. All
aux1 runtime and test wiring was removed; retained aux2 source remains
byte-identical to commit `b5e17d76`. The exact samples and temporary build
identity are in `stage-08b-qkv-aux1-diagnostic.json`.

## Stage 09 — attention outproj + next-FFN pre-RMS closure audit

Status: dropped after current-best same-binary A-B-B-A and NSYS; Stage 08
aux2 remains retained.

The retained aux2/id70/FA4 trace contains exactly 4752 same-stream
FA4→outproj→next-FFN-RMS triples, 33 per forward. The boundary contributes
`0.545 ms` summed and `0.173 ms` exposed union per forward, so the earlier
isolated negative still required one accumulated-graph closure test. The
authenticated M64xN192 `(1,2)`-cluster CuTe kernel keeps every official FP16
projection/residual boundary and reproduces the incumbent C384 Vec8 XOR RMS
tree exactly. Standalone validation was elementwise identical. NCU had already
proved 52 registers/thread, `114.82 KiB` dynamic plus `1.02 KiB` driver shared
memory, exactly two CTAs/SM, 1.11 waves, no spills, 86.8% no-eligible cycles,
and 60.4% long-scoreboard stalls. Its isolated S2 round was nevertheless
`28.509 us` versus `26.277 us`.

A temporary config-selected `dlopen` hook allowed control and candidate to use
one binary while carrying the fused RMS scratch across the adjacent
attention/FFN objects. Locked 50-warmup/300-iteration A-B-B-A measured control
`9225.855157/9249.631477` and candidate
`8827.727061/8933.192196` nnEval/s, a center regression of `3.868%`.

Same-binary NSYS explains why current-best scheduling cannot reverse it. The
candidate removes exactly 4752 launches, and 89.5% of its interval overlaps
other work; boundary-exclusive time even falls `24.897 -> 18.451 ms`.
However, its contended fused service averages `36.863 us`, versus only
`16.519 us` for the exact control outproj+RMS pair. Kernel sum therefore rises
`756.045 -> 873.367 ms` (`+15.518%`) and whole-graph union rises
`466.821 -> 481.016 ms` (`+3.041%`). Neighboring FFN and RoPE summed time also
rises `10.638%/17.781%`. More concurrency is real—overlap grows 35.656%—but it
cannot hide the added work.

The current-best graph is retained. No long or replay gate ran. All temporary
runtime wiring was removed, core/CMake are byte-clean against HEAD, and the
current-best binary was rebuilt without the hook. Full identities, isolated
measurements, A-B-B-A samples, NSYS hashes, union attribution, and cleanup
proof are in `outproj-rmsnorm-fusion-profile.json`.

## Stage 10 — mixed C384-half2 / C768-flat-vec8 affine SiLU

Status: retained after targeted NCU, same-binary A-B-B-A/NSYS, five-sample
long gate, and the 8192-row aggregate/request accuracy gate.

The final Stage 08 trace exposed C768 affine SiLU for `10.628 ms`, while the
C384 affine work was almost entirely covered. Targeted NCU showed that the
existing C768 flat-vec8 kernel reduced replay duration `26.18 -> 18.75 us`,
executed instructions by `16.41%`, waves/SM `14.15 -> 3.32`, no-eligible
cycles `42.68% -> 28.17%`, and long-scoreboard cycles/issue `8.82 -> 3.43`,
with zero spills and improved achieved occupancy. However, its historical
selector also returned C384 to the official kernel. Whole-graph NSYS proved
that coupling erased the local gain: C384 exclusive time rose by `926%`.

The one bounded fix added no device kernel. The exact selector
`half2-c384-flat-vec8-c768-b29` keeps the retained half2 path for C384 and uses
the existing flat-vec8 path only for C768. It is default-off and accepts only
the portable SM103 adapter with exact max/runtime B29 FP16/NHWC; native SM120
defaults are unchanged.

Same-binary 50-warmup/300-iteration A-B-B-A measured control
`9228.251092/9153.427883` and candidate `9377.453701/9319.844271` nnEval/s,
or `+1.7170%` by the two-point centers. Steady last-50 NSYS preserved exactly
40,596 launches and proved 1,100 C384 half2 calls, 1,200 C768 flat-vec8 calls,
and zero official C384 fallbacks. C768 union fell `18.022 -> 13.339 ms` and
exclusive time fell `10.665 -> 2.751 ms`; C384 remained slightly faster. The
whole-graph union fell `322.543 -> 317.958 ms` (`-1.422%`) while overlap rose
`4.035%`.

Five 1000-iteration samples were `9260.359321`, `9304.307285`,
`9307.161993`, `9317.568460`, and `9409.425562` nnEval/s. The median is
`9307.161993`, spread `1.602%`, `+1.146%` over Stage 08 and `+38.217%` over
the fixed TensorRT 10.16 median. Every sample exceeds the maximum Stage 08
sample.

The 8192-row replay kept all input/target sections byte-identical. Aggregate
policy top1 is `0.998291`, policy probability RMSE `9.5291e-5`, value outcome
RMSE `0.0021228`, score mean RMSE `0.00183684`, and ownership sigmoid RMSE
`0.00023349`. Every aggregate gate passes. The largest request ratio is
ownership max-absolute at `1.94636x`, below the fixed `2.25x` TensorRT-control
limit. Exact source, binary, NCU, NSYS, replay and comparison identities are in
`stage-10-mixed-affine-silu.json`.
