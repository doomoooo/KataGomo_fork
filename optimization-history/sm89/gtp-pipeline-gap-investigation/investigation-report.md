# RTX 4090 GTP pipeline-gap investigation

Date: 2026-08-06 UTC

## Decision

For exact 19x19, FP16 NHWC, B13 and two inference streams on RTX 4090, keep
the event-gated pipeline but disable CUDA Graph replay:

```ini
nnBatchAwareDispatch = true
cudaAsyncInferPipeline = true
cudaEventPipelineUseGraph = false
```

The deployment loss was primarily CUDA Graph fixing the two saturated compute
streams in a worse resource-contention phase. Eager submission is the accepted
fix. It improves physical B13 launch rate by 1.735%, real `nnEval/s` by 1.690%,
and visits/s by 2.111% in the real GTP ABBA. The corrected eager GTP rate is
within 0.196% of the direct common-wall forward rate.

Do not set `CUDA_DEVICE_MAX_CONNECTIONS=1`, and do not give one inference
stream a higher priority. Values 2 through 32 produced no repeatable gain over
the default connection regime; asymmetric priority caused 4-9% regressions.

The earlier 3400-class numbers have not disappeared. They are default-Boost
peak/thermal-regime results. The controlled attribution runs below use observed
2205 MHz SM and 10251 MHz memory clocks and therefore sit around 3200-3250
physical rows/s.

## Scope and build identity

- Worktree: `/workspace/katago-gtp-pipeline-gap`
- Investigative harness source: commit `7e6ec01`
- Base/integration behavior: `d5c5b36`, with tested production implementation
  inherited from `c4b92e3`
- Model SHA-256:
  `1881600caab9e9d85a3dd6a019e9b8e7d2c237b5f984e13ed49a8645be3077c6`
- Config SHA-256:
  `b001224e9f00aee4e8b14f711642a6a95374b4719ff50e96d16b323ec909c367`
- Main benchmark/profile binary SHA-256:
  `4478b64190fc8014880ac9dc8aa85fefc495f126dc2871bb3b185ea743a4931c`
- Exact topology: RTX 4090 GPU 0, 19x19, FP16 NHWC, B13, S2
- All GPU work used `gpu-lock with --gpu 0`; profilers also used the profile
  lock.

The scripts requested a 10501 MHz memory lock, but telemetry during every
decision run observed 10251 MHz. The SM clock was exactly 2205 MHz. There were
no active power-cap or thermal-slowdown samples. Results remain valid A/B
comparisons because both sides observed the same clocks; the observed clock,
not the request, is the measurement regime.

## Corrected metric

Every backend launch is padded to physical B13, so the aligned metric is:

```text
physical launch row rate = completed launches * 13 / common timed wall seconds
```

The old `benchmarknn` aggregate summed `B13 / per-lane median GPU duration`.
That is a latency estimator, not completed work over one shared interval. Under
the corrected controlled direct runs:

| direct run | old median-sum | common-wall physical rows/s |
|---|---:|---:|
| A | 3198.8345 | 3197.7298 |
| B | 3288.9410 | 3287.0109 |
| mean | 3243.8878 | 3242.3703 |

The old estimator was optimistic by 1.5175 rows/s (0.047%) in these runs. More
importantly, even with fixed clocks, independent direct S2 processes differed
by 2.79% because their two streams settled into different phases. Therefore the
old two-median sum cannot be used as a stable aggregate-throughput baseline.

Artifacts: `common-wall-core-memory-locked/`.

## Direct, event, copy, and graph controls

Short common-wall controls use the same B13/S2 physical work and queue depth 1.
The graph/eager short runs retain visible phase variance, so the real GTP ABBA
below is the acceptance result.

| control | runs (physical rows/s) | mean |
|---|---:|---:|
| direct eager, device-resident | 3197.7298, 3287.0109 | 3242.3703 |
| graph + copies, primary pair | 3224.5636, 3202.3898 | 3213.4767 |
| graph + copies, copy-control pair | 3216.0193, 3195.8162 | 3205.9178 |
| graph, copies omitted | 3189.2542, 3220.4193 | 3204.8367 |
| eager + copies | 3256.5203, 3208.5717 | 3232.5460 |
| paired graph control | 3197.3150, 3191.6337 | 3194.4744 |

Omitting H2D/D2H changed the copy-control mean by -1.0811 rows/s (-0.034%):
there is no repeatable copy-path wall-rate penalty.

### NCU copy-contention control

NCU profiled the same 12 timed `DualGemm` launches in eager copy versus
no-copy modes.

| metric | copies | no copies |
|---|---:|---:|
| duration mean | 43.3653 us | 43.3387 us |
| DRAM read | 5.4008 MB | 5.4027 MB |
| DRAM write | 2.4720 MB | 1.7467 MB |
| DRAM throughput | 18.4858% | 16.7824% |
| L2 throughput | 70.8845% | 70.9340% |
| SM throughput | 67.4950% | 67.5343% |

Copies raise observed DRAM activity, but `DualGemm` duration changes only
0.0615%; L2 and SM throughput are unchanged. This agrees with the complete
wall-boundary control. Artifacts: `ncu-copy-control/`.

## Real GTP acceptance ABBA

Five positions, 2000 visits, t96, exact B13/S2:

| mode | visits/s | real nnEval/s | nnBatches/s | physical B13 rows/s |
|---|---:|---:|---:|---:|
| graph A | 3479.74 | 3127.61 | 244.40 | 3177.20 |
| graph B | 3493.89 | 3134.12 | 244.96 | 3184.48 |
| graph mean | 3486.815 | 3130.865 | 244.680 | 3180.840 |
| eager A | 3556.28 | 3182.72 | 248.93 | 3236.09 |
| eager B | 3564.54 | 3184.86 | 248.92 | 3235.96 |
| eager mean | 3560.410 | 3183.790 | 248.925 | 3236.025 |

Eager minus graph is +55.185 physical rows/s (+1.735%), +52.925 real
`nnEval/s` (+1.690%), and +73.595 visits/s (+2.111%).

A 26-row exact-19 replay is byte-identical between graph and eager for all
saved outputs. Comparison metrics are all zero error and top-1 match is 1.0.
Artifacts: `gtp-graph-eager-core-memory-locked/` and `correctness-smoke/`.

## Nsys mechanism

The last 30 complete forwards per lane contain 295 kernels per forward.

| mode | timed wall | kernel union | union busy | two-stream overlap / union | trace rows/s |
|---|---:|---:|---:|---:|---:|
| graph | 255.046 ms | 246.058 ms | 96.476% | 63.269% | 3058.27 |
| eager | 242.218 ms | 241.470 ms | 99.691% | 60.073% | 3220.24 |

Graph replay both introduces visible node-trace submission gaps and increases
simultaneous two-stream overlap. The latter worsens resource contention in the
hot kernels:

| kernel family | graph mean | eager mean | graph delta |
|---|---:|---:|---:|
| DualGemm | 46.435 us | 44.507 us | +4.33% |
| regular CUTLASS GEMM | 31.853 us | 31.494 us | +1.14% |
| FlashAttention | 28.348 us | 26.519 us | +6.90% |
| main Ampere GEMM | 23.273 us | 21.221 us | +9.67% |
| RMSNorm | 7.876 us | 6.735 us | +16.94% |

The graph is therefore not merely paying a host launch overhead. It fixes a
worse two-stream scheduling phase with more concurrent resource competition.
Artifacts: `nsys-graph-eager/`.

## Steady regions versus GTP transitions

The original graph-node GTP trace has 425 search launches after two precapture
launches. Its four continuously supplied regions contain 409 complete launches
in 1.599064 s, approximately 3325.5 physical rows/s. Aggregate physical rate
was 3290.17 rows/s. First/last ramp, position transitions, and drain therefore
cost about 1.07%; 16 launches sit outside the continuous regions.

The post-fix eager trace has two main continuously supplied regions with 249
complete launches in 1.011525 s, approximately 3200 rows/s. Aggregate physical
rate was 3174.47 rows/s, a ramp/transition/drain cost of about 0.80%.

These are profile-specific decompositions, not cross-run throughput A/Bs.
They show that search supply is a small aggregate cost, not the 2-4% backend
gap originally suspected.

The t96/t112/t128 supply control found physical graph rates of 3142.30,
3137.55, and 3145.74 rows/s respectively, with average real batch 12.78-12.81.
More search threads do not improve backend supply.

## `CUDA_DEVICE_MAX_CONNECTIONS`

This variable controls CUDA device connections/Hyper-Q work queues, not the
number of host-enqueued kernels per stream. An ascending/descending short scan
gave:

| connections | mean physical rows/s |
|---:|---:|
| 1 | 2471.605 |
| 2 | 3243.715 |
| 4 | 3198.205 |
| 8 | 3222.710 |
| 16 | 3254.273 |
| 32 | 3251.733 |

One connection clearly serializes or strongly constrains S2. Values 2-32 are
inside known phase variability. A natural-execution Nsys comparison falsified
the apparent 16-over-8 screen gain: 8 connections produced 3218.1 trace rows/s,
99.727% union busy, and 59.396% two-stream overlap; 16 produced 3211.2 rows/s,
99.134% busy, and 60.361% overlap. Leave the environment variable unset rather
than pinning a nondefault value.

NCU replay is not a valid acceptance boundary for this variable because it
alters the cross-stream scheduling being tested. Nsys natural execution is the
mechanistic evidence. Artifacts: `max-connections-screen/` and
`nsys-max-connections/`.

## Asymmetric stream priority

The RTX 4090 reports CUDA priority range `least=0`, `greatest=-5`. A slight
`-1` priority was tested on each lane independently.

| mode | both 0 | lane 0 = -1 | lane 1 = -1 |
|---|---:|---:|---:|
| eager physical rows/s | 3200.361 | 3024.269 (-5.50%) | 3002.149 (-6.19%) |
| graph physical rows/s | 3178.427 | 3051.394 (-4.00%) | 2988.498 (-5.98%) |

Eager Nsys verifies that the requested priority was active. Baseline was
3159.36 trace rows/s with 99.476% kernel-union busy. Lane 0 at -1 fell to
2874.65 rows/s and 93.034% busy. The high-priority lane's kernel sum remained
191.39 ms versus 190.92 ms baseline, while the low-priority lane expanded from
191.56 to 212.88 ms and large union gaps appeared. Priority created starvation
and a worse phase rather than an averaged chaotic schedule. The code experiment
was reverted; no production change remains.

NCU is intentionally omitted here because priority does not change kernel
instructions/resources and replay destroys the scheduling effect. Artifacts:
`stream-priority-screen/` and `nsys-stream-priority/`.

## CUDA Graph regression context

NVIDIA's current CUDA Graph guidance explicitly says graphs primarily help
CPU-bound workloads and can regress an already GPU-bound workload; it recommends
eager-versus-graph Nsys comparison and warns that node tracing itself adds
significant overhead:
https://docs.nvidia.com/dl-cuda-graph/latest/troubleshooting/performance-issues.html

NVIDIA also documents that graph node creation order influences scheduling
heuristics and can create bubbles in work availability:
https://developer.nvidia.com/blog/constant-time-launch-for-straight-line-cuda-graphs-and-other-performance-enhancements

The same troubleshooting guide describes device-connection serialization from
CUDA Graph internal stream expansion. That is a real general failure mode, but
it is not the dominant mechanism here: this workload has two compute streams,
`MAX_CONNECTIONS=1` is catastrophically slower, and graph Nsys shows *more*
two-stream overlap rather than serialization. Our exact mechanism is inferred
from the measured timeline: graph replay makes the two independent saturated
forwards eligible in a more repeatable and more contended phase, while eager
host submission introduces enough natural phase variation to reduce contention.

## Quantified root cause and uncertainty

Under the corrected same-run regime:

```text
direct common-wall                         3242.370 rows/s
graph GTP                                 3180.840 rows/s
current direct-to-graph gap                  61.530 rows/s (1.934% of graph)
eager GTP                                 3236.025 rows/s
graph penalty removed by eager              55.185 rows/s (1.735% of graph)
remaining direct-to-eager GTP gap             6.345 rows/s (0.196% of eager)
```

The 6.345-row residual includes position ramp/drain, queue/admission,
pre/postprocess/publication, and remaining phase variation. It is below the
requested 0.5 percentage-point attribution threshold.

The historical 112.22-row equal-core gap is not algebraically mixed with this
decomposition. That older protocol did not record/lock memory and compared
different process phase realizations; controlled direct S2 alone spans 2.79%.
It is retained as historical evidence, not used to estimate a current term.

## Final recommendation

1. Deploy eager event submission (`cudaEventPipelineUseGraph=false`).
2. Keep both compute streams at equal default priority.
3. Leave `CUDA_DEVICE_MAX_CONNECTIONS` unset; explicitly reject value 1.
4. Use common-wall completed physical launches for backend throughput and keep
   per-lane medians only as latency diagnostics.
5. Report default-Boost peak throughput separately from the 2205/10251
   attribution regime. Run a final thermally stabilized unlocked test only
   after the implementation is fixed.
6. Do not spend more optimization time on CUDA Graph for this exact B13/S2
   4090 path unless a future driver/toolkit changes the eager-versus-graph Nsys
   mechanism.

## Artifact timeline

| UTC result time | stage | decision | physical forward / GTP result | precision/profile evidence |
|---|---|---|---:|---|
| 2026-08-06 18:41:12 | common-wall metric | keep harness | direct mean 3242.370 | exact physical counters |
| 2026-08-06 18:42:50 | eager vs graph GTP | accept eager | 3180.840 -> 3236.025 | ABBA |
| 2026-08-06 18:44:14 | graph/eager correctness | pass | not a perf test | 26 rows byte-identical |
| 2026-08-06 18:46:02 | graph/eager Nsys | mechanism confirmed | 3058.27 -> 3220.24 trace rows/s | union/overlap/hot kernels |
| 2026-08-06 18:48:28 | copy contention | reject as cause | no wall signal | NCU DualGemm +0.0615% |
| 2026-08-06 18:59:46 | max connections | reject tuning | 2-32 no repeatable gain | Nsys; 1 is catastrophic |
| 2026-08-06 19:06:04 | asymmetric priority | reject/reverted | -4% to -9% | Nsys starvation evidence |
