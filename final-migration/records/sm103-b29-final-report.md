# B300 B29/S2 final optimization report

Date: 2026-08-18

The final retained CUDA graph reaches **9307.161993 combined nnEval/s** at
B29/S2, with **1.602% relative spread**. Against the fixed TensorRT
10.16.1.11 median of **6733.719141 nnEval/s**, this is **+38.217%**. The
result is an optimization qualification for this fixed workload; this report
does not make a deployment-certification claim.

## Fixed contract

| Item | Fixed value |
| --- | --- |
| Device and target | NVIDIA B300 SXM6 AC, 148 SMs, compute capability 10.3, accelerated target `sm_103a` |
| Workload | Exact batch 29, two independent inference streams (S2), 19x19 board |
| Model | `b11c768h12nbt3tflrs-fson-silu`; SHA-256 `1881600caab9e9d85a3dd6a019e9b8e7d2c237b5f984e13ed49a8645be3077c6`; 10469 flattened rows |
| CUDA graph format | FP16/NHWC |
| Objective | Maximize combined B29/S2 nnEval/s while passing the fixed 8192-row aggregate and per-request correctness gates |
| Comparison anchor | TensorRT 10.16.1.11; binary SHA-256 `883024dc8bbc02e7f6b05b0431034652931acc760b76e7fd455dc996af278612`; five 1000-iteration samples; median `6733.719141`; spread `0.4415%` |

The immutable identities and benchmark contract are in the
[baseline anchor](../plans/sm103/b300-b29-s2/baseline-anchor.json) and the
[request-gate control](../plans/sm103/b300-b29-s2/request-gate-control.json).
All rates below are the combined rate of both inference streams.

## Retained lineage and measured ladder

The final accumulated lineage is:

`portable tactics -> fused FFN -> no-AB12 -> native FA4 -> pinned QKV cuBLASLt id70 -> QKV aux2 -> mixed affine`

| Step | Accumulated change | Median nnEval/s | Spread | Increment vs parent | Vs TRT 10.16 | Retained commit |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| Fixed comparison | TensorRT 10.16.1.11 baseline | 6733.719141 | 0.442% | -- | -- | control only |
| Measurement anchor | CUDA portable-empty, no tactic retained | 5000.394925 | 0.743% | -- | -25.741% | -- |
| 02a | beta=1 residual GEMM | 5600.299778 | 0.607% | +11.997% | -16.832% | `6723ea0c` |
| 02b | C384 warp4-vec8 RMSNorm | 5974.164849 | 0.155% | +6.676% | -11.280% | `abdeb8e7` |
| 02c | planar fused Q/K RoPE | 6495.313934 | 0.079% | +8.723% | -3.540% | `f88e423a` |
| 02d | half2 affine-SiLU | 6737.964266 | 0.768% | +3.736% | +0.063% | `23f42fb9` |
| 02e | half8 C1152 SwiGLU | 7150.091831 | 0.012% | +6.116% | +6.183% | `1c2aad4c` |
| 03a | fused FFN with the required FP16 projection round-trip | 8077.636942 | 0.219% | +12.972% | +19.958% | `5d7adaa2` |
| 03b | remove the unused fused-FFN AB12 output and store path | 8587.808904 | 0.831% | +6.316% | +27.534% | `38fe3334` |
| 06 | native FA4 D32 attention, FP32 QK/PV accumulation | 8949.114828 | 0.333% | +4.207% | +32.900% | `230dd195` |
| 07a | exact QKV cuBLASLt id70 tuple | 9058.808862 | 0.414% | +1.226% | +34.529% | `d40282a2` |
| 08 | Q on primary plus K/V on two auxiliary streams | 9201.745296 | 0.842% | +1.578% | +36.652% | `b5e17d76` |
| 10 | C384 half2 plus C768 flat-vec8 affine-SiLU | **9307.161993** | **1.602%** | **+1.146%** | **+38.217%** | this commit |

The portable substage ladder used two 1000-iteration samples per step and was
closed by the accumulated 8192-row replay; Stage 03a onward used five
1000-iteration long samples after 200 warmups. Incremental percentages use
each stage's declared parent, not the TensorRT row. The canonical retained
selection is recorded in the
[B29/S2 tactic plan](../plans/sm103/b300-b29-s2/portable-baseline.json).

### How the retained line emerged

1. The baseline and numerical controls were frozen before tuning. The first
   fused FFN prototype was fast but omitted the official FP16 projection
   boundary; its policy and ownership request errors were roughly 10--20x the
   TensorRT control, so it was rejected. Restoring the projection round-trip
   produced the retainable Stage 03a implementation.
2. Before native work continued, five already-proven portable tactics were
   accumulated and ablated. Residual fusion, C384 RMSNorm, fused planar Q/K
   RoPE, affine-SiLU, and C1152 SwiGLU retained; persisting-L2 trunk/inner and
   the policy/head bundle regressed by `1.886%` and another `1.642%`,
   respectively, and were excluded.
3. The fused FFN removed 8052 projection/SwiGLU launches and made that boundary
   `1.987x` faster in the whole-graph profile. Removing its dead AB12 output
   then delivered a separate `+6.316%` long gain.
4. Several QKV/RoPE and FFN resource-shaping variants failed whole-graph or
   isolated gates. The retained attention replacement then reduced exposed
   critical time and added `+4.207%`. Reduced-precision accumulator controls
   did not transfer and the FP32 mode stayed fixed.
5. Generic QKV cuBLASLt autotuning sometimes selected id71 and lost the gain,
   including a `8780.674` nnEval/s process. The exact id70/tile23/stages35/
   cluster5/zero-workspace tuple was therefore constructed and read back
   explicitly. The last step kept that exact algorithm and changed only the
   Q/K/V dependency topology.
6. Targeted NCU showed that the existing C768 flat-vec8 kernel was substantially
   faster, but its old selector also disabled the retained C384 half2 path. A
   single mixed selector removed that coupling without adding a device kernel:
   C384 stays half2 and only C768 uses flat-vec8. Whole-graph union fell 1.422%
   and the long gate added another 1.146%.

## New architecture-neutral techniques worth carrying forward

Only the following two items are new, broadly reusable optimization findings
from this work. The portable tactic bundle is established prior art in this
repository; FA4, tcgen05, and SM103-specific instruction choices are facts of
the final implementation, not the general lessons claimed here.

### 1. Change the ready DAG to widen the ready frontier

**Mechanism.** A deep host submission queue is not the same as a wide set of
dependency-ready GPU grids. The control trace already had about 1016 pending
descriptors per outer stream (1024 at p99/max), with the host approximately
17.45 ms ahead. Same-stream FIFO dependencies nevertheless exposed normally
only the two stream heads. Increasing `CUDA_SCALE_LAUNCH_QUEUES` to 2x or 4x,
or Hyper-Q connections from the default eight to sixteen, did not help.

A post-hoc inspection of the fixed TensorRT engine found `num_aux_streams=0`
and all 439 kernels of its exact-B29 structural probe on one kernel stream. Its
Q/K/V projection is one grouped grid rather than an auxiliary ready DAG. This
confirms that the retained frontier change was not copied from that plan.

Stage 08 records one event after pre-RMS, leaves Q on the primary stream, and
launches K and V on two persistent nonblocking auxiliary streams with
independent cuBLASLt handles/plans and private workspaces. The primary stream
joins both done events before the existing RoPE and attention work. Q, K, and
V are therefore sibling nodes: the ready frontier grows from at most two Q
grids across S2 to as many as six Q/K/V grids, without changing a GEMM
algorithm, tensor, launch count, or downstream order.

**Evidence.** In the steady 100-forward comparison, both variants launch
exactly 40,596 kernels, including 9,900 QKV kernels. QKV summed service time
actually increases `100.438 -> 109.451 ms` (`+8.974%`), while QKV interval
union falls `90.124 -> 66.377 ms` (`-26.349%`). Whole-graph union falls
`331.831 -> 322.454 ms` (`-2.826%`), overlap rises `+3.787%`, and trace time at
concurrency depth at least three grows `9.470 -> 30.456 ms`. The five-sample
long gate gains `+1.578%`. A one-aux diagnostic, which reintroduced a FIFO
K-to-V edge, gained only `+0.346%` over control and was `-2.000%` behind aux2;
this isolates frontier width rather than event-count reduction as the cause.

**Migration conditions.** Apply this only when sibling operations share
read-only inputs, write disjoint outputs, and have a clear downstream join.
Each branch needs correctly owned stream-local library state and workspace;
events, allocator lifetimes, teardown, and graph capture must preserve the
join. The kernels' resource shapes must also leave a plausible whole-graph
scheduling opportunity. Validate the actual ready DAG and end-to-end union;
adding streams to a serial or resource-saturated dependency chain is not
sufficient.

Evidence: [submission-window analysis](../plans/sm103/b300-b29-s2/evidence/scheduler-submission-window-qkv-aux-plan.json),
[queue environment probe](../plans/sm103/b300-b29-s2/evidence/scheduler-queue-environment-probe.json),
[Stage 08 result](../plans/sm103/b300-b29-s2/evidence/stage-08-qkv-aux-streams.json),
[aux1 diagnostic](../plans/sm103/b300-b29-s2/evidence/stage-08b-qkv-aux1-diagnostic.json),
and [TensorRT engine inventory](../plans/sm103/b300-b29-s2/evidence/tensorrt-engine-plan-audit.json).

### 2. Delete downstream-unconsumed outputs and their storage traffic

**Mechanism.** The fused FFN provider produced final C and also materialized
the two FP16 projections as AB12, but KataGo consumed only C. AB12 was a
`48,241,152 B` output per stream. Stage 03b kept the externally visible ABI
argument and every required numerical boundary, but removed the AB12 device
pointer/descriptor, register and shared fragments, four-stage shared ring,
copies, and all TMA stores from the device path. The tcgen05 mainloop, FP16
projection round-trip, fast SwiGLU arithmetic, and C store remained unchanged.
This is not merely an arithmetic reduction: the arithmetic was deliberately
held fixed while an unused value and its complete storage pipeline were
deleted.

**Evidence.** With the same 148-block/192-thread launch, dynamic shared memory
falls `214016 -> 181248 B`, registers fall `74 -> 66` per thread, and L2
sectors fall `2262595 -> 754848` (`-66.64%`). NCU replay duration falls
`44.58 -> 40.10 us`; the whole-graph fused-FFN total falls
`110.524 -> 95.245 ms`; and the long accumulated result gains `+6.316%`.
The AB12 sentinel remained untouched, the tight output check passed, and the
full replay passed all gates.

**Migration conditions.** Prove from the consumer graph that the output is
dead and has no alias, callback, synchronization, or externally observable
side effect. Preserve semantic rounding boundaries that the old output path
may have imposed even when the materialized tensor disappears. Remove the
whole descriptor/copy/store/ring path rather than merely suppressing the last
store, then remeasure resources, whole-graph scheduling, and numerical output;
resource changes can perturb accumulated floating-point rounding even with
unchanged source-level arithmetic.

Evidence: [Stage 03b no-AB12 profile](../plans/sm103/b300-b29-s2/evidence/stage-03-no-ab12-profile.json).

## Correctness and stability result

The final gate uses the fixed 8192-row, 19x19 corpus and the same CUDA FP32
replay. For each of policy probability, value probability, raw score, and
ownership probability, both worst-request maximum-absolute error and RMSE must
be no more than `2.25x` the corresponding TensorRT 10.16 control; every metric
must pass, in addition to the aggregate gates.

| Final Stage 10 metric | Result |
| --- | ---: |
| Policy probability RMSE | `9.52908e-5` |
| Policy top-1 agreement vs FP32 | `99.8291%` |
| Value outcome RMSE | `0.00212281` |
| Score mean RMSE | `0.00183684` |
| Ownership sigmoid RMSE | `0.000233494` |
| Maximum request ratio vs TensorRT control | `1.94636x` (pass, limit `2.25x`) |

Inputs and targets are byte-identical to the reference. The Stage 08 schedule was not
byte-identical to Stage 07 because the changed schedule perturbs accumulated
FP16 rounding, but policy top-1 agreement between them is 100% and probability
RMSE is only `2.866e-6`; the fixed FP32/TRT gates pass.

The Stage 03a bad-case investigation also closed a potential implementation
error. Its worst fast-math case, row 4124/move 358, is a legal real B29 row,
not padding; inputs and targets are byte-identical, raw-logit correlation is
`0.9999865`, and no permutation, fixed-index, or spatial-layout signature was
found. Strict exp/div fixes that row but drops whole throughput to about
`5554` nnEval/s. One Newton refinement preserves throughput but moves the
worst case to row 7111/move 111. The retained fast variant's maximum request
ratio is `2.148x`, while the broken no-roundtrip variant remains around
10--20x and fails. This was classified as nonlinear amplification of a small
arithmetic-path difference, not an implementation bug. See the
[bad-case study](../plans/sm103/b300-b29-s2/evidence/stage-03a-bad-case-study.json).

## Appendix A: negative routes and lessons

These are experiment closures, not recommended optimization techniques.

| Route | Measured result and disposition |
| --- | --- |
| Original fused FFN without projection rounding | Whole-graph speed was promising, but policy max-abs/RMSE reached `15.132x/19.756x` and ownership `15.711x/10.029x` of the TensorRT controls. Rejected; the missing FP16 projection boundary was restored. |
| Fused planar QKV+RoPE | Isolated S1 improved to `22.756 us`, yet whole-graph throughput fell `8587.809 -> 8048.267` nnEval/s. Kernel sum fell, but union rose `5.02%` and overlap fell `17.19%`: a heterogeneous resource-lifetime conflict. The G148 retry regressed `4.692%` in A-B-B-A and raised whole-graph union `2.82%`; the ordered split retry regressed `9.319%`. Route closed. |
| Attention outproj + next-FFN RMSNorm | Exact output and more overlap did not repay service inflation: current-best A-B-B-A center regressed `3.868%`, and whole-graph union rose `3.041%` (`466.821 -> 481.016 ms`). Dropped. |
| FA4 FP16 accumulator controls | Against FP32 S2 `37.682 us`, QK16/PV16/both16 measured `41.486/39.792/42.779 us`: `10.10%/5.60%/13.53%` slower. All kept the same physical TMEM footprint, so FP32 remained retained. |
| DeepGEMM BF16 | Cast-inclusive coordinated S2 was `38.2016 us` versus id70 FP16 `18.4898 us`, or `2.066x` (`+106.61%`) slower; raw DeepGEMM BF16 was already `17.41%` slower. Closed before graph integration. |
| FlashInfer standalone D32 attention | Correct, but `11.16%/14.01%` slower than cuDNN in S1/S2. Kept only as source reference; no whole-graph trial. |
| Triton controls | Persistent-TMA plain wide QKV was `2.099x/2.712x` slower in S1/S2 and dual FFN `2.200x/2.258x` slower. Triton linear2+residual lost about `51%/65%`. Plain-kernel routes were stopped before whole-graph work. |
| FFN resource reshaping | AB3 reached the predicted two-CTA residency boundary but lost in the accumulated graph; corrected M128xN64/AB4 was `18.67%` slower in isolated S2. Resource residency alone was not a win. |
| Other audited swaps | The cuDNN SDPA engine/knob surface kept its existing winner; MSLK 1.3 offered no drop-in SM103a FP16 path; other surveyed backends required large ports or precision changes without a measured reason to displace the retained graph. They were not promoted to optimization claims. |
| TensorRT engine inventory | The fixed plan exposed grouped QKV, two composite RoPE launches, `_gemm_mha_v2`, and a three-launch FFN, but no untried drop-in operator. Normal GTP consumes ownership; the benchmark-only null output pointer is not a dead production output and was explicitly closed. |

Detailed negative evidence is indexed by the
[append-only stage log](sm103-b29-optimization-stages.md), including the
[QKV+RoPE profile](../plans/sm103/b300-b29-s2/evidence/stage-04-qkv-rope-profile.json),
[persistent G148 result](../plans/sm103/b300-b29-s2/evidence/stage-04b-qkv-rope-persistent-g148-result.json),
[split-QKV result](../plans/sm103/b300-b29-s2/evidence/stage-04c-qkv-rope-split3-result.json),
[outproj/RMSNorm result](../plans/sm103/b300-b29-s2/evidence/outproj-rmsnorm-fusion-profile.json),
[accumulator controls](../plans/sm103/b300-b29-s2/evidence/stage-06b-fa4-accumulator-modes.json),
[DeepGEMM kill test](../plans/sm103/b300-b29-s2/evidence/deepgemm-bf16-kill-test.json),
and [MSLK audit](../plans/sm103/b300-b29-s2/evidence/mslk-1.3-sm103-low-hanging-audit.json).

## Reproduction and evidence entry points

1. Install the exact managed environment with `./setup.sh install`, enter it
   with `source final-migration/environment/activate-sm103.sh`, then run
   `./setup.sh verify` and `./setup.sh build`. The Python 3.14.7 archive,
   80 exact package pins, CUDA/cuDNN identities, source commits and managed
   `ptxas` paths are checked by the committed environment scripts.
2. Start from the [baseline anchor](../plans/sm103/b300-b29-s2/baseline-anchor.json),
   [request control](../plans/sm103/b300-b29-s2/request-gate-control.json),
   and [retained tactic selection](../plans/sm103/b300-b29-s2/portable-baseline.json).
3. Follow the [stage log](sm103-b29-optimization-stages.md) and its tracked
   evidence JSON. The retained native-stage records are
   [fused FFN](../plans/sm103/b300-b29-s2/evidence/stage-03a-fused-ffn.json),
   [no-AB12](../plans/sm103/b300-b29-s2/evidence/stage-03-no-ab12-profile.json),
   [native attention](../plans/sm103/b300-b29-s2/evidence/stage-06-fa4-native-profile.json),
   [pinned id70](../plans/sm103/b300-b29-s2/evidence/stage-07-cublaslt-qkv.json),
   [aux2](../plans/sm103/b300-b29-s2/evidence/stage-08-qkv-aux-streams.json),
   and [mixed affine](../plans/sm103/b300-b29-s2/evidence/stage-10-mixed-affine-silu.json).
4. Rebuild and run the tracked source-contract tests, including
   [FFN](../../python/tests/test_sm103_cudnn_oss_ffn_hook.py),
   [FA4](../../python/tests/test_sm103_fa4_native_hook.py),
   [id70](../../python/tests/test_sm103_projection_gemm_lt_contract.py), and
   [aux2/lifetime](../../python/tests/test_sm103_qkv_aux_contract.py). Serialize
   GPU work through [with-gpu-lock.sh](../environment/with-gpu-lock.sh).
5. Repeat the exact B29/S2 long protocol (200 warmups, five independent
   1000-iteration samples for the final stages), then the 8192-row replay and
   fixed request comparison. Evidence JSON files carry the binary, report,
   replay, comparison, and source hashes; ignored local raw artifacts are not
   substitutes for those tracked identities.

The retained commit chain is:

```text
be6b4107  portable adapter infrastructure
10afb5ca  fixed B29/TRT control
6723ea0c  beta=1 residual
abdeb8e7  C384 RMSNorm
f88e423a  planar Q/K RoPE
23f42fb9  affine-SiLU
1c2aad4c  C1152 SwiGLU
5d7adaa2  fused FFN with projection round-trip
38fe3334  no-AB12
230dd195  native FA4
d40282a2  pinned QKV id70
b5e17d76  QKV aux2 ready-DAG
this commit  mixed C384-half2 / C768-flat-vec8 affine-SiLU
```

Same-binary A/B/B/A runs, NSYS interval-union/overlap accounting, targeted NCU
resource measurements, fail-closed selection checks, bad-case analysis, and
GPU locking were the experimental and validation methods used to decide this
lineage. They are methodology, not additional optimization techniques.
