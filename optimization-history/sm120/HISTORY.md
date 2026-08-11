# 5090D/4090 SM120 rebuild optimization history

Regime IDs (measurement conditions; never chain percentages across regimes):

| ID | backend | GPU | requireExactNNLen | batch | servers | FP16 | clock | note |
|---|---|---|---:|---:|---|---|---|
| S2-FA4OFF-EXACT | CUDA | 5090D | true | 13 | 2 | yes | unlocked | stage-1 control (official attention) |
| S2-FA4ON-EXACT | CUDA | 5090D | true | 13 | 2 | yes | unlocked | stage-1 FA4 AOT attention |
| S2-OFFICIAL-MASKED | CUDA | 5090D | false | 13 | 2 | yes | unlocked | old B13 baseline (mask/bias included) |
| S2-FA4-ACCUM | CUDA | 5090D | true | 13 | 2 | yes | unlocked | same-toolchain FP32 vs qk16/pv16/both16 AOT objects |
| FINAL-S1-B13 | CUDA SM120 | 5090D | true | 13 | 1 | yes | unlocked | S1-only wide FFN and strided QKV enabled |
| FINAL-S2-B13 | CUDA SM120 | 5090D | true | 13 | 2 | yes | unlocked | final maximum-throughput topology |

After every accepted optimization, the next hypothesis must be selected from a
fresh whole-forward profile of the resulting mainline: S2 Nsys supplies the
critical-path and interference weight, and a matching 344-ordinal S1 NCU pass
supplies resource and stall evidence. Partial NCU captures and microbenchmarks
may explain or reject a candidate, but may not set the following priority by
themselves.

Candidate retention and S2 adjudication policy (effective 2026-08-06 after
the Stage33/H33b workflow correction):

1. Do not run homogeneous-S2 or synthetic mixed-peer microbenchmarks. They do
   not represent the heterogeneous forward scheduler and may invert the real
   whole-graph result. Existing artifacts remain historical evidence only and
   have no acceptance, rejection, or promotion authority.
2. A candidate is `resource-positive` when it is correct, is positive in both
   isolated-S1 order aggregates at the predeclared resolution, has no spills,
   and strictly reduces relevant work or an NCU/ptxas resource while leaving
   the other hard residency resources no worse.  There is no arbitrary 0.5%
   retention threshold: multiple strict local improvements may compose into a
   target-topology gain.  Grid expansion and pipeline changes are recorded
   separately and can prevent a claim of strict dominance.
3. A resource-positive candidate is retained behind an exact-shape,
   default-false switch and goes directly to a short current-mainline
   B13/19x19/S2 whole-forward A/B. The current whole-graph Nsys/NCU explains
   why the candidate is worth testing, but only the controlled whole graph
   decides target-topology performance.
4. Promote a short whole-graph signal to symmetric long ABBA/BAAB. Reject for
   performance only after a reproducible whole-graph regression or no signal
   at the predeclared long-run resolution. Correctness failure remains an
   immediate hard rejection.
5. Every accepted change still requires full all-head accuracy and a fresh S2
   Nsys plus matching 344-ordinal S1 NCU before selecting the next target.
   Resource-positive candidates that were rejected only by the removed proxy
   gates must be audited and marked `reopened`, `confirmed rejected`, or
   `superseded`; historical raw results are never rewritten.
6. Portable 30/10 batch/device scans are candidate discovery only. Rebuild the
   candidate on the fixed accepted revision and use runtime A/B where possible.
   Inspect the first/last control drift explicitly: longer unlocked-GPU arms can
   add thermal or power drift rather than precision, so extend total evidence
   with short interleaved arms instead of assuming one long arm is authoritative.

For fixed-B13 arithmetic validation, use a stable one-server replay so every
non-tail batch is exactly 13.  A multi-server replay queue may emit timing-
dependent partial batches and select different fallback tactics across runs;
that topology is valid for throughput testing but is not a byte-identity
oracle for a batch-specialized kernel.

Dual-stream phase control is out of scope for the GTP deployment path. Do not
add cross-stream checkpoints, event rendezvous, or phase offsets. Keep the eager
launch queue deep enough and leave ready-kernel ordering to CUDA and the GPU
scheduler; `CUDA_DEVICE_MAX_CONNECTIONS` remains a queue-parallelism setting,
not a phase-control mechanism.

## Main timeline

| result save time (UTC) | stage | conclusion | regime | kernel / subgraph | throughput nnEval/s | accuracy evidence |
|---|---|---|---|---:|---:|---|
| 2026-08-05 14:08 | Nsys (FA4 on/off) | mechanism confirmed | trace | attention 49 -> 19 us/block | 2808 -> 3063 (traced) | n/a |
| 2026-08-05 14:09 | official control (FA4 off, exact) | baseline for stage 1 | S2-FA4OFF-EXACT | SDPA 1.75 ms/iter-pair | 2818.53 / 2816.85 (3x300) | new FP32 ref + official FP16 replay |
| 2026-08-05 14:09 | FA4 AOT attention | accepted (+11.3%) | S2-FA4ON-EXACT | FA4 0.89 ms/iter-pair (19 us/block) | 3143.68 / 3129.88 (3x300) | 8192-row all-head JSON passes |
| 2026-08-05 14:10 | NCU FA4 | headroom quantified | kernel | 8.3% occ, 37% MMA stall | n/a | n/a |
| 2026-08-05 15:00:40 | FA4 accumulator ablation microbench | both16 selected | S2-FA4-ACCUM, isolated | FP32 `16.3599`; QK16 `13.4522`; PV16 `12.7965`; both16 `11.8000` | not measured | all four standalone 19x19 smoke tests pass |
| 2026-08-05 15:06:02 | 8,192-row accumulator accuracy matrix | all modes pass; both16 retained | ACC | not measured | not measured | both16 policy top-1 `99.7437%`, optimistic top-1 `99.7314%`, all predeclared gates pass |
| 2026-08-05 15:07:26 | FP32 vs both16 Nsys | mechanism confirmed under S2 | S2-FA4-ACCUM, trace | B13 FA4 `22.387 -> 17.946 us`; registers `255 -> 247` | traced result only | arithmetic already covered by ACC |
| 2026-08-05 15:10:51 | both16 ordered A/B | accepted | S2-FA4-ACCUM, 1000-iter ABBA/BAAB | see direct micro/Nsys rows | mean `3118.006 -> 3131.905` (`+0.446%`); median `3115.581 -> 3132.553` (`+0.545%`) | both16 all-head JSON passes |
| 2026-08-05 15:11:22 | both16 NCU | causal explanation complete | S2-FA4-ACCUM, isolated NCU | `22.34 -> 17.02 us`; no-eligible `80.11% -> 69.63%`; issued warp/scheduler `0.20 -> 0.30` | not measured | n/a |
| 2026-08-05 15:19:06 | fixed-S361 tail mask | rejected (slower) | S2-FA4-ACCUM, isolated micro ABBA/BAAB | generic `11.8072 ->` fixed `11.8351 us` (`+0.236%` slower) | not measured by predeclared early-stop rule | standalone attention output byte-identical; full replay not run after performance rejection |
| 2026-08-05 15:40 | strided-batched FFN input projection | rejected (slower) | S2-FA4-ACCUM, short A-B-B-A | two Hgemm calls replaced by one batch-count-2 call | mean `3111.511 -> 3044.701` (`-2.147%`) | full replay not run after predeclared performance early stop |
| 2026-08-05 15:54 | single-wide FFN + layout SwiGLU | accepted S1, rejected S2 | B13/S1 and B13/S2 | S1 FFN subgraph `16.216 -> 14.801 ms/30 forwards` | S1 long mean `2639.469 -> 2666.794` (`+1.035%`); S2 short `-2.907%`; full 8192-row gates pass |
| 2026-08-05 16:04 | strided-batched QKV | accepted S1, rejected S2 | B13/S1 and B13/S2 | QKV launches `3 -> 1` per block; about `104 us/forward` saved in S1 trace | S1 long mean `2641.578 -> 2678.688` (`+1.405%`); S2 short `-0.548%`; full 8192-row gates pass |
| 2026-08-05 16:08 | combined S1 projections | accepted S1, rejected S2 | B13/S1 and B13/S2 | single-wide FFN plus strided QKV | S1 long mean `2634.039 -> 2701.585` (`+2.564%`); S2 short `-3.339%`; replay byte-identical to individual candidates |
| 2026-08-05 16:18 | GEMM beta residual fusion | accepted S1/S2 | B13/S2 primary | residual launches `1440 -> 0`; related GEMMs `64.746 -> 69.283 ms`, net subgraph faster | S2 long mean `3128.629 -> 3221.153` (`+2.957%`); full 8192-row gates pass |
| 2026-08-05 16:21:52 | one-warp C384 RMSNorm fast-tree | rejected on accuracy | B13/S1 and B13/S2, ACC | not profiled after accuracy failure | short S1 `+2.68%`, S2 `+3.07%` | optimistic-policy top-1 `99.5972%`, one row below the predeclared `99.60%` gate |
| 2026-08-05 16:29:01 | one-warp C384 RMSNorm exact-tree | accepted S1/S2 | B13/S2 primary | RMSNorm `47.876 -> 34.027 ms` over 8,844 launches (`-28.93%`) | S2 long mean `3227.127 -> 3297.223` (`+2.172%`); median `+2.296%` | full replay byte-identical to accepted Stage9, all 8,192-row gates pass |
| 2026-08-05 16:39:02 | fused learnable Q/K RoPE | accepted S1/S2 | B13/S2 primary | launches `8844 -> 4422`; RoPE `40.764 -> 23.426 ms` (`-42.5%`) | S2 long mean `3277.804 -> 3374.489` (`+2.950%`); median `+2.982%` | full replay byte-identical to Stage10, all 8,192-row gates pass |
| 2026-08-05 16:50:35 | contiguous half8 C1152 SwiGLU | accepted S1/S2 | B13/S2 primary | `45.461 -> 39.445 ms` over 4,422 launches (`-13.23%`) | S2 long mean `3388.722 -> 3424.859` (`+1.066%`); median `+1.142%` | full replay byte-identical to Stage11, all 8,192-row gates pass |
| 2026-08-05 17:03:18 | half2 C384/C768 affine-SiLU | accepted S1/S2 | B13/S2 primary | combined direct kernel total `23.643 -> 16.349 ms` (`-30.85%`) | S2 long mean `3424.013 -> 3455.877` (`+0.931%`); median `+1.161%` | full replay byte-identical to Stage12, all 8,192-row gates pass |
| 2026-08-05 17:10 | B13-shared fused Q/K RoPE | rejected | B13/S1 and B13/S2 | B13 grid `361x13 -> 361x1`, direct kernel `17.328 -> 23.614 ms` (`+36.3%` slower) | short S1 `2935.799 -> 2910.726` (`-0.854%`); S2 `3489.311 -> 3489.137` (`-0.005%`) | full replay not run after predeclared performance early stop |
| 2026-08-05 17:15 | fused Q/K RoPE half2 I/O | rejected below threshold | B13/S1 and B13/S2 | `23.671 -> 21.873 ms` over 4,422 launches (`-7.60%`) | short S1 `+0.301%`; S2 `+0.443%` | full replay not run after performance early stop |
| 2026-08-05 17:25:06 | two-warp exact-tree C384 RMSNorm | rejected (slower) | B13/S2, trace | `33.171 -> 37.446 ms` over 8,844 launches (`+12.89%` slower) | traced `3466.752 -> 3428.277` (`-1.11%`) | full replay not run after direct-kernel performance rejection |
| 2026-08-05 17:32:35 | final topology long confirmation | S2/B13 selected | FINAL-S2-B13 | S2/S3/S4 independent server streams | S2 `3461.079`; S3 `3400.520`; S4 `3362.087` | final replay follows |
| 2026-08-05 17:34:07 | final S1/S2 full replay | all gates pass | ACC, 8,192 rows | not measured | S1 long `2996.035`; S2 long `3461.079` | policy top-1 `99.7437%`, optimistic `99.7314%`, all-head gates pass |
| 2026-08-05 17:34:35 | final S2 Nsys | independent overlap confirmed | FINAL-S2-B13, trace | streams 65/82, `256.777 ms` overlap (`70.85%` of union busy) | traced `3474.409` | arithmetic covered by final replay |
| 2026-08-06 01:11 | same-source TensorRT competition baseline | CUDA ahead, but rebuild incomplete | B13/S2, 1000/50, TRT-CUDA-CUDA-TRT | not measured | TRT `3260.834`; CUDA `3509.727` (`+7.633%`) | both backends previously pass full-FP32 gates |
| 2026-08-06 01:15 | 5080-history implementation cross-check | Stage-17 stop condition withdrawn | source audit | fused FFN and fixed projection AOT families absent | not measured | no arithmetic candidate evaluated in this audit |
| 2026-08-06 02:27 | fixed-B13 TileLang fused FFN | accepted | B13/S2 | two GEMMs + SwiGLU -> shared-input fused kernel; 46.83 us mixed-load | `3564.580 -> 3656.759` (`+2.586%`) | 8,192-row all-head gates pass |
| 2026-08-06 03:07 | fixed-B13 planar wide QKV AOT | accepted | B13/S2 | three QKV GEMMs -> one 24.21 us kernel | `3674.181 -> 3814.026` (`+3.806%`) | metrics identical to Stage 20; all gates pass |
| 2026-08-06 05:56:18 | aligned vec8 C384 RMSNorm | accepted for B13/S2 | B13/S2, 1000/30 symmetric ABBA/BAAB | NCU `5.28 -> 4.58 us`; S2 trace `19.305 -> 16.461 ms` over 3,960 launches | long mean `3690.205 -> 3719.712` (`+0.800%`); median `+1.142%` | 8,192-row all-head gates pass; policy top-1 `99.7803%`, optimistic `99.7070%` |
| 2026-08-06 06:57:29 | persisting-L2 C768 trunk window | accepted for B13/S2 | B13/S2, 1000/30 symmetric ABBA/BAAB | B13 C768 consumer `6.592 -> 5.888 us`; L2 hit `52.53% -> 94.60%`; DRAM `7.214 -> 5.073 MB` | long mean `3709.814 -> 3739.137 nn/s` (`+0.790%`); all four adjacent pairs positive | 8,192-row replay byte-identical to accepted Stage27 output |
| 2026-08-06 07:04 | flat vec8 C768 affine-SiLU | rejected | B13/S2 on accepted trunk-L2 baseline, 1000/30 symmetric ABBA/BAAB | grid `4693x384 -> 1760x256`; arithmetic order retained | short `+0.092%`; long mean `3800.148 -> 3793.430 nn/s` (`-0.177%`), only 1/4 adjacent pairs positive | performance early stop; experimental runtime code removed |
| 2026-08-06 07:40:09 | persisting-L2 C384 inner window | accepted for B13/S2 | B13/S2 on accepted trunk-L2 baseline, two clean 1000/30 symmetric ABBA/BAAB rounds | C384 consumer `4.784 -> 4.448 us`; L2 hit `53.29% -> 96.97%`; DRAM `5.106 -> 2.396 MB`; Nsys total kernel `668.618 -> 663.144 ms` | pooled long mean `3806.692 -> 3861.783 nn/s` (`+1.447%`); all adjacent pairs positive in both rounds | 8,192-row replay byte-identical to accepted Stage28 output |
| 2026-08-06 08:32:22 | post-inner-L2 whole-graph Nsys + NCU rerank | linear2 selected as the next sole primary target | fixed B13/19x19; S2 Nsys 30 forwards/stream plus matching S1 NCU over all 344 ordinals | linear2 `1077.0 us/stream-forward`, `395.4 us` S2 excess, `1.580x` S2/S1; NCU 162 regs, 65.5 KiB smem, 0.65 waves/SM, 8.3% occupancy, 9.2% eligible cycles | no candidate throughput measurement; current accepted pooled long remains `3861.783` | no arithmetic change |
| 2026-08-06 09:13:03 | linear2 M128N96 coexistence rerank | rejected; mainline unchanged | fixed B13/19x19; exact private-weight S1, homogeneous S2, and mixed-peer ABBA | current/N96-S4 homogeneous S2 `33.246 -> 33.092 us` (`-0.462%`); with fused FFN `50.073 -> 48.585 us`; with wide QKV `33.324 -> 33.673 us` (`+1.049%` regression) | not measured by the predeclared mixed-peer rejection gate; current accepted pooled long remains `3861.783` | direct AOT/cublasHgemm boundary output bit-exact; full replay not run after performance rejection |
| 2026-08-06 09:29:01 | fused-FFN A-fragment reuse | rejected at homogeneous-S2 gate; mainline unchanged | fixed B13/19x19; private weights; 8-cycle symmetric alternating ABBA | S1 `37.680 -> 34.921 us` (`-7.321%`); homogeneous S2 pair `62.682 -> 62.854 us` (`+0.275%` regression); registers `146 -> 136`, zero spills | not measured by the predeclared S2 rejection gate; current accepted pooled long remains `3861.783` | deterministic FP16 output bit-exact (`0` mismatches, max abs `0`); full replay not run after performance rejection |
| 2026-08-06 10:14:24 | fused-FFN A-fragment reuse H33b | accepted; old proxy rejection overturned | fixed B13/19x19; real S2 whole graph, 1000/30 symmetric ABBA/BAAB | same kernel geometry and smem; registers `146 -> 136`; long adjacent deltas all positive | `3824.934 -> 3859.725 nn/s` (`+0.910%`); short independently `+0.911%` | 8,192-row replay byte-identical, SHA256 `ed0ed808...e1a02`; fresh S2 Nsys and 344-ordinal S1 NCU complete |
| 2026-08-06 10:20 | linear2 M128N96/S4 direct whole-graph recheck | confirmed rejected | fixed B13/19x19; current accepted S2 graph, 400/25 symmetric ABBA/BAAB | 148 CTAs, 162 regs, 57,344 B smem versus accepted 111 CTAs, 162 regs, 65.54 KiB | `3865.088 -> 3827.442 nn/s` (`-0.974%`); all 4 adjacent pairs negative | Stage32 boundary bit-exact; full replay not run after coherent graph regression |
| 2026-08-06 10:27 | fused Q/K RoPE half2 current-mainline recheck | accepted; old 0.5% threshold overturned | fixed B13/19x19; real S2 whole graph, 1000/30 symmetric ABBA/BAAB | accepted profile RoPE `218.9 -> 177.9 us/fwd`, excess `83.1 -> 51.0 us/fwd` | long `3845.421 -> 3855.728 nn/s` (`+0.268%`), 3/4 adjacent positive; short `+0.788%` | 8,192-row replay byte-identical, SHA256 `ed0ed808...e1a02`; fresh S2 Nsys and 344-ordinal S1 NCU complete |
| 2026-08-06 11:07:56 | existing three-stage FFN schedule on current accepted graph | rejected; mainline unchanged | fixed B13/19x19; real S2 whole graph, 400/25 symmetric ABBA/BAAB | historical S1 whole graph `3047.568 -> 3089.095`; dynamic smem `32,768 -> 49,152 B/CTA` | `3900.067 -> 3754.695 nn/s` (`-3.727%`); all four adjacent pairs negative | no arithmetic change; long/replay/profile stopped after coherent graph regression |
| 2026-08-06 11:19:20 | fused-FFN remove post-commit main-loop barrier | rejected at S1 mechanism gate; mainline unchanged | fixed B13/19x19; bit-exact boundary plus 8-cycle S1 alternating ABBA | SASS static barrier sites `5 -> 4`; `36.945 -> 37.681 us` (`1.993%` slower); 136 regs, 32,768 B dynamic smem, zero spills unchanged | not measured after coherent S1 regression | boundary bit-identical (`0` mismatches, max abs `0`); NCU/whole graph/full replay stopped by rule |
| 2026-08-06 11:25:44 | lower-smem wide-QKV M128N128K32/S3 recheck | rejected as current whole-graph no-signal; mainline unchanged | fixed B13/19x19; real S2 whole graph, 400/25 symmetric ABBA/BAAB | historical direct QKV `18.840 -> 17.128 us`; dynamic smem `65,536 -> 49,152 B`, registers `136 -> 142`, no spills | `3890.548 -> 3887.944 nn/s` (`-0.067%`); adjacent pairs 2 positive / 2 negative | no arithmetic change; long/replay/profile stopped by directionality rule |
| 2026-08-06 11:35:01 | fused-FFN fast FP32 SwiGLU quotient | rejected at compile/resource gate; mainline unchanged | fixed B13/19x19; isolated AOT compile/SASS audit | control 136 regs/0 spill; `__fdividef` 168 regs/64 B spill; inline approximate divide 145 regs/0 spill; forced 136 regs caused 36 B spill | not measured | numerical route never reached performance or accuracy gates |
| 2026-08-06 11:39:24 | fused-FFN fixed 18x37 panel-10 swizzle | rejected at S1 mechanism gate; mainline unchanged | fixed B13/19x19; exhaustive coordinate proof, bit-exact boundary, 8-cycle S1 alternating ABBA | SASS instructions `2792 -> 2728`; `36.708 -> 36.848 us` (`0.383%` slower); 136 regs, 32,768 B smem, zero spills unchanged | not measured after S1 regression | 666/666 mapping exact; boundary bit-identical (`0` mismatches, max abs `0`) |
| 2026-08-06 11:52:44 | prefetch-preserving fused-FFN A reuse | rejected; FFN early-prefetch axis closed | fixed B13/19x19; compile/resource audit plus bit-exact boundary and 8-cycle S1 alternating ABBA | retain-two-A version 151 regs; half-tile prefetch version 135 regs/0 spill but `36.860 -> 38.640 us` (`4.830%` slower) | not measured after resource/S1 mechanism failures | half-tile version bit-identical over 5,406,336 FP16 outputs |
| 2026-08-06 20:46:02 | Stage44 exact-B13 packed-QKV CuTe completion | accepted; automatic RTX5090D/B13/S2 winner | fixed B13/19x19; common-wall natural S2, 100/20 ABBA + reversed BAAB | QKV `19.228 -> 14.164 us/block`; composed QKV-to-FA4 boundary `35.451 -> 32.214 us/block` (`-9.131%`); QKV regs `136 -> 107`, zero spills | common-wall mean `3896.899 -> 3920.182 nn/s` (`+0.597%`); all four adjacent pairs positive | full 8,192-row all-head gates pass; policy top-1 `99.7803%`, optimistic `99.7070%`; fresh S2 Nsys and 344-ordinal S1 NCU complete |
| 2026-08-06 21:17:48 | Stage45 postConv-to-following-C768-affine-SiLU CUTLASS2 fusion | retained default-off for resource-positive S1; rejected as S2 default | fixed B13/19x19; targeted NCU, S1 ABBA+BAAB 20/5, S2 short ABBA 20/5 | launches `2 -> 1` at 11 boundaries; final 256-thread tactic 108 regs, 50.176 KiB total smem, 0 spills versus postConv 154 regs and 74.752 KiB total smem | S1 `3267.010 -> 3272.912` (`+0.181%`); S2 `4047.917 -> 4030.194` (`-0.438%`) | full 8,192-row output byte-identical to Stage44, SHA256 `1503a84b...b18d96`; accepted Stage44 profiles reused |
| 2026-08-06 22:57:12 | Stage47 exact-B13 CuTe paired-projection FFN + SwiGLU | accepted; automatic RTX5090D/B13/S2 winner | fixed B13/19x19; direct boundary, short S1/S2 ABBA, 400/50 S2 ABBA, rebuilt-default ABBA | TileLang fused FFN `33.435 -> 31.198 us` (`-6.692%`); final 96 regs, 50.176 KiB dynamic smem, 1.00 waves/SM; K32 grid340 fills two CTAs/SM | medium common-wall `3926.900 -> 3944.540 nn/s` (`+0.449%`), both adjacent pairs positive; rebuilt default short `+0.961%`; final traced S2 `4050.536` | 8,192-row all-head gates pass versus Stage44 and FP32; policy top-1 vs FP32 `99.8169%`; fresh S1/S2 Nsys and complete 344-ordinal NCU |
| 2026-08-07 03:39 | Stage51 exact-B13 mask preprocessing elision | retained default-off for strict S1/work reduction; not promoted to S2 default | fixed B13/19x19; mechanism Nsys, S2 100/20 and 400/40 ABBA+BAAB, S1 400/40 ABBA+BAAB | channel extract + half-to-float + sum removed; target counts `28 -> 24` in trace, exactly four B13 forwards, no replacement work | S1 pooled `3335.031 -> 3338.695` (`+0.110%`), both orders positive; S2 short `+0.353%`, 400 pooled `+0.099%` but reverse order `-0.250%` | stable fixed-B13 8,192-row replay byte-identical, SHA256 `3a732971...e7ff`; accepted Stage47 profile remains current |
| 2026-08-07 03:52 | Stage52 exact-B13 initial-conv cuDNN engine 47 | retained default-off for strict S1/resource improvement; not promoted to S2 default | fixed B13/19x19; mechanism Nsys, S2 100/20 ABBA+BAAB, S1 400/40 ABBA+BAAB | B13 boundary `2 padding + legacy main -> 1 engine-47 kernel`; shared memory `81.92 -> 4.10 KiB`, achieved occupancy `8.32% -> 26.07%`, zero workspace | S1 pooled `3328.325 -> 3334.744` (`+0.193%`), both orders positive; S2 pooled `+0.032%` with conflicting signs | stable fixed-B13 8,192-row replay byte-identical, SHA256 `3a732971...e7ff`; accepted Stage47 profile remains current |
| 2026-08-07 05:41 | Stage55 exact-B13 CuTe packed-QKV + both16 RoPE epilogue | retained default-off for local boundary improvement; not promoted to S2 default | fixed B13/19x19; 400-cycle boundary ABBA, targeted NCU, natural S2 100/20 ABBA and shortened Nsys | QKV+RoPE `19.424 -> 18.653 us` in short Nsys; boundary ABBA `-4.099%`; 96 registers and zero spills, but executed instructions `4.657M` vs `4.246M` summed control | natural S2 pooled `3987.720 -> 3942.413 nn/s` (`-1.136%`); following FA4 expands `14.595 -> 18.980 us` after the streams re-phase | 8,192-row all-head comparison passes; policy top-1 vs FP32 `99.8413%`; option-off B13 and option-on B12 fallback remain byte-identical; accepted Stage47 profile remains current |
| 2026-08-07 06:12 | Stage56 exact-B13 FA4 M128N64 both16 tactic | retained default-off for strict S1/residency improvement; not promoted to S2 default | fixed B13/19x19; standalone reference/event timing, full-set NCU, natural S1/S2 A-B-B-A | registers `247 -> 168`, smem `24.58 -> 16.38 KiB`, theoretical occupancy `16.67 -> 25.00%`, eligible warps/scheduler `0.38 -> 0.53`; executed instructions `+8.10%` | S1 common-wall `3371.991 -> 3386.731 nn/s` (`+0.437%`); S2 100/20 `-0.097%`, 400/50 `-0.189%` with `-1.784%` bracketing-control thermal drift | full 8,192-row gates pass; policy top-1 vs FP32 `99.7314%`; B12 fallback byte-identical; accepted Stage47 profile remains current |
| 2026-08-07 06:30 | Stage57 exact-B13 FA4 M128N96 both16 tactic | accepted; automatic RTX5090D/B13/S2 winner and final optimization stage | fixed B13/19x19; matching NCU, symmetric short S1/S2, decisive same-binary runtime ABBA+BAAB | registers `247 -> 233`, smem `24.58 -> 20.48 KiB`, executed instructions `7.329M -> 7.280M`, zero spills; occupancy class unchanged | same-binary common-wall pooled `3978.527 -> 4003.411 nn/s` (`+0.625%`), all 4 adjacent comparisons positive; final traced S2 `4092.818` combined / `4120.446` common-wall | full 8,192-row gates pass; policy top-1 vs FP32 `99.7681%`; B12 fallback byte-identical; fresh S1/S2 Nsys and complete 344-ordinal NCU |

## Rejected / paused

- H1 cuBLASLt GEMM (2026-08-05, 5090D): REJECTED. fp32 compute via cuBLASLt
  selected worse SM80 kernels (throughput 2594 vs 3136, -19%); fp16 compute
  was faster in benchmarknn (3414, +7%) but crashed in replaynn
  (invalid resource handle) and changes GEMM accumulation numerics.
- CuTe/DSL GEMM probes (2026-08-05, 5090D): CUTLASS blackwell_geforce example
  and quack GemmSm120 initially failed to compile with NVVM errors. Root
  cause found: TMA/accelerated features require `CUTE_DSL_ARCH=sm_120a`
  (plain sm_120 rejects TMA ops). With sm_120a + cutlass-dsl 4.6.1/4.7.0 and
  correct rank-3 tensor usage, the CUTLASS example is correct but still
  slower than cublasHgemm (21.9 us vs 16.5 us for M=1152xN4693xK384).
  TileLang best plain GEMM 24.6 us. Plain GEMM replacement rejected across
  all tools; fused FFN (dual-GEMM+SwiGLU) remains the GEMM-side candidate.
  Environment note: cuBLAS Hgemm already runs ~250 TFLOPS on these shapes,
  so isolated GEMMs are near practical limits.
- QK16-only and PV16-only accumulator modes (2026-08-05): REJECTED as final
  choices. Both pass the full accuracy gates and are faster than FP32
  accumulation, but both16 is faster than either and passes the same gates.
- Fixed-S361 tail mask (2026-08-05): REJECTED. Hard-coding the final tile's
  105 valid columns was byte-identical but slowed the kernel from 11.8072 to
  11.8351 us in micro ABBA/BAAB. Reopen only if static S also removes dynamic
  loop/block scheduling, not just the tail-limit expression.
- Strided-batched FFN input projection (2026-08-05): REJECTED. Combining the
  two same-shape FFN input projections as `cublasHgemmStridedBatched` with
  batch count 2 reduced B13/S2 throughput by 2.147%. A true single M=2304
  projection remains a distinct candidate because it uses a different cuBLAS
  shape and output layout.
- Single-wide FFN GEMM (2026-08-05): ACCEPTED FOR S1, REJECTED FOR S2. The
  integrated wide GEMM plus layout-aware SwiGLU improves the S1 FFN subgraph by
  8.73% and long whole-network mean by 1.035%, with full accuracy passing. It
  regresses S2 by 2.907%, so `cudaUseWideFFNSingleGemm` defaults false and is
  only enabled for the S1 regime.
- Strided-batched QKV (2026-08-05): ACCEPTED FOR S1, REJECTED FOR S2. It
  replaces three Q/K/V Hgemm launches with one batch-count-3 call, improves S1
  long whole-network mean by 1.405%, and passes full accuracy. It regresses S2
  by 0.548%, so `cudaUseQKVStridedSm120` defaults false.
- One-warp RMSNorm fast-tree (2026-08-05): REJECTED ON ACCURACY. It improved
  short S1/S2 throughput materially, but optimistic-policy top-1 was 99.5972%,
  one row below the predeclared 99.60% gate. The accepted exact-tree variant
  retains six independent original warp-group reductions, is byte-identical to
  Stage9, and improves long S2 throughput by 2.172%.
- B13-shared fused Q/K RoPE (2026-08-05): REJECTED. It reduced the B13 grid by
  13x but increased direct kernel time by 36.3%, regressed short S1 by 0.854%,
  and was neutral on S2 (-0.005%). Reopen only with a materially faster inner
  loop that does not further reduce grid parallelism.
- Fused Q/K RoPE half2 I/O (2026-08-05): REJECTED BELOW THRESHOLD. Direct RoPE
  improved 7.60%, but short whole-network gains were only 0.301% S1 and 0.443%
  S2. Reopen only as an independently measured component of a larger RoPE
  change.
- Two-warp exact-tree C384 RMSNorm (2026-08-05): REJECTED. Despite reducing
  each thread's live input values, block barriers and four times as many CTAs
  increased direct RMSNorm time by 12.89% and reduced traced S2 throughput by
  1.11%. NCU shows the accepted one-warp kernel already has 100% theoretical
  occupancy; reopen only if the cross-warp synchronization can be removed.
- Linear2 M128N96 coexistence tile (2026-08-06 09:13:03): REJECTED. The S4
  candidate improved isolated S1 substantially and the homogeneous S2 pair by
  only 0.462%, below the predeclared 0.5% gate, while regressing the mixed
  wide-QKV pair by 1.049%. S2/S3 reduced shared memory but were 6.5--8.3%
  slower in the homogeneous S2 screen; adjacent M96N96 was at least 13.0%
  slower in S1. Reopen linear2 only with a mechanism that both improves the
  111-CTA coverage and permits co-residency with the 65.5 KiB wide-QKV CTA.
- Fused-FFN A-fragment reuse (2026-08-06 09:29:01): ORIGINAL REJECTION
  SUPERSEDED BY H33b ACCEPTANCE AT 10:14:24. Removing half
  of the non-transposed shared-memory matrix loads reduced registers from 146
  to 136 and improved S1 by 7.321%, but moving gate MMA ahead of the next
  linear-weight prefetch regressed the homogeneous S2 pair by 0.275%. The
  candidate was bit-exact and spill-free. The real whole graph subsequently
  measured `+0.910%`, so this entry is retained only as evidence of the proxy
  gate's false negative.
- Existing three-stage FFN S1 schedule on the accepted S2 graph (2026-08-06
  11:07:56): REJECTED. Although it had improved the historical S1 graph, it
  regressed the current real S2 graph from `3900.067` to `3754.695 nnEval/s`
  (`-3.727%`) with all four adjacent pairs negative. Reopen only if the extra
  stage no longer raises per-CTA shared memory or a separate mechanism removes
  the resulting scheduling constraint.
- Fused-FFN post-commit barrier removal (2026-08-06 11:19:20): REJECTED. It
  removed one static SASS barrier site while preserving 136 registers, shared
  memory, and zero spills, but slowed S1 by `1.993%`. The barrier is therefore
  useful to the realized copy pipeline even though it is not required for
  correctness. Reopen only with a jointly redesigned wait/commit schedule.
- Lower-smem wide-QKV M128N128K32/S3 schedule (2026-08-06 11:25:44):
  REJECTED AS CURRENT WHOLE-GRAPH NO-SIGNAL. It retains a historical S1 gain
  and lowers dynamic shared memory to 49,152 B, but the current natural S2
  graph measured `-0.067%` with two positive and two negative adjacent pairs.
  Reopen only after a different QKV mechanism changes the copy pipeline.
- Fused-FFN fast FP32 quotient (2026-08-06 11:35:01): REJECTED AT RESOURCE
  GATE. Fast quotient code raised registers from 136 to 145 without spills;
  `__fdividef` was worse at 168 registers with 64 B spills, and forcing 136
  registers still spilled 36 B. Reopen only with an independently ablated
  epilogue live-range change that restores the accepted zero-spill footprint.
- Fused-FFN fixed-grid swizzle (2026-08-06 11:39:24): REJECTED. It removed 64
  static SASS instructions while preserving exact coordinates, bit identity,
  136 registers and zero spills, but slowed S1 by `0.383%`. Reopen only if
  source counters identify the prologue as an executed critical stall.
- Prefetch-preserving fused-FFN A reuse (2026-08-06 11:52:44): REJECTED. A
  version retaining both K-subtile A fragments required 151 registers; a
  135-register, zero-spill version moved each half-tile copy earlier but slowed
  S1 by `4.830%` despite bit identity. Reopen only with a producer mechanism
  that neither extends the A live range nor moves a block barrier into K.
- PostConv-to-following-C768-affine-SiLU CUTLASS2 fusion (2026-08-06
  21:17:48): RETAINED DEFAULT-OFF. NCU-driven conversion from 128 to 256
  threads reduced registers from 168 to 108, raised eligible warps/scheduler
  from about 0.197 to 0.341, used 50.176 KiB total shared memory, and had zero
  spills. Pooled short S1 was reproducibly positive by `0.181%`, but natural
  S2 regressed `0.438%` with both adjacent comparisons negative. Full replay
  is byte-identical. Reopen the deployment candidate using an SM120-native CuTe
  GEMM plus the same verified dual-output epilogue, not an unsupported CUTLASS2
  warp-orientation sweep.
- SM120-native CuTe postConv dual-output fusion (2026-08-06): REJECTED AT
  DIRECT-BOUNDARY/RESOURCE GATE. Transferring the accepted packed-QKV
  M128/N128/K64 atom-4x2 schedule produced a correct 26-row replay and deleted
  all 11 target boundaries, but natural kernel time was 27.607 us versus about
  24.575 us for the original postConv+affine boundary (+12.3%). NCU reported
  153 registers/thread, 99.33 KiB dynamic shared memory, only 0.17 eligible
  warps/scheduler, and no spills. Reusing residual fragment storage for the
  activated output made compiler allocation strictly worse at 167 registers
  and 0.14 eligible warps/scheduler. No S2 or full-profile run was justified;
  source integration was removed. CuTe schedules must be derived from the
  target boundary's own NCU evidence, not mechanically copied from a neighbor
  with a materially simpler epilogue.
- Partial C288 no-split policy-g1 + value-v1 head (2026-08-07): REJECTED.
  TensorRT's grouping was reproduced as one C768->C288 cuBLAS GEMM with direct
  stride-aware consumers and byte-identical B13 output. Nsys/NCU verified one
  fewer launch and a local reduction from 16.93 to 14.66 us, but 400/40 natural
  S2 was -0.383% forward, -0.013% reverse, and -0.198% pooled. Conditional S1
  was -0.002% forward and +0.116% reverse, so it failed the both-order retention
  gate. Source integration was removed; no new full-graph profile was needed.
- Attention out-projection / following FFN RMSNorm boundary (2026-08-07):
  REJECTED BEFORE GRAPH INTEGRATION. cuDNN 9.25 rejected the explicit
  `matmul -> view reshape -> add -> RMSNorm` operation graph with
  `CUDNN_STATUS_NOT_SUPPORTED_GRAPH_PATTERN`. The bounded CuTe alternative
  folded gamma into the Stage-47 paired FFN weights, replaced full RMSNorm
  with a 16-register scale-only kernel, and applied FP32 row scale in the
  CuTe epilogue. NCU verified the scale kernel was strictly lighter
  (2.432 vs 3.296 us; 16 vs 36 registers), but the complete CuTe FFN kept
  96 registers/50.176 KiB and became slower under targeted NCU. A 400-iter
  boundary ABBA measured S1 +0.290% slower and S2 wall +0.022% slower.
  No source integration, commit, or fresh full-graph profile was justified.
- Exact-B13 CuTe packed-QKV + both16 RoPE epilogue (Stage55, 2026-08-07):
  RETAINED DEFAULT-OFF. The final register-fragment implementation removes the
  independent RoPE launch and global round trip, improves the S1 boundary by
  `4.099%`, uses 96 registers with no spills, and passes full 8,192-row
  accuracy. Natural S2 nevertheless regresses `1.136%`: short Nsys shows the
  fused QKV boundary remains faster, but it synchronizes the streams into a
  phase where following FA4 averages `18.980 us` instead of `14.595 us`.
  Do not reopen through explicit stream-phase control; that complexity is out
  of scope for GTP. Reconsider only after another independently useful graph
  change alters this adjacency. No fresh accepted full-graph profile was
  required.
- Exact-B13 FA4 M128N64 both16 tactic (Stage56, 2026-08-07): RETAINED
  DEFAULT-OFF. A portable 30/10 scan suggested `+1.02%`, but fixed-revision
  validation found isolated event timing `1.042%` slower and natural S2
  statistically neutral/slightly negative. NCU nevertheless verifies a major
  residency improvement (247 to 168 registers, 24.58 to 16.38 KiB shared
  memory, 16.67% to 25% theoretical occupancy), and natural whole-forward S1
  improves `0.437%`. Full accuracy passes. Keep it as a composable B13 tactic;
  do not promote without a future current-graph S2 win.

## Resolved blockers

- both16 initially failed to build because flash-attn's typed tail mask and PV
  rescale path assumed FP32 accumulator storage. flash-attn 4.0.0b25 already
  supplied accumulator selection and a typed QK mask; the local PV-rescale
  conversion completed the path. both16 is accepted in Stage 3.
- CuTe DSL 4.7 AOT handles became invalid when replaynn destroyed its warmup
  models and unloaded a process-global generated library. The AOT module is
  now loaded once for process lifetime; full replay passes after the fix.
