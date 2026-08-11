# Handover: investigate the remaining RTX 4090 pure-forward versus GTP launch-rate gap

Date: 2026-08-06 UTC

This request is for the active RTX 4090 optimization session. Finish, commit,
or cleanly revert the session's current dirty experiment before starting this
work. Do not reset, overwrite, or otherwise disturb the current uncommitted
files in `/workspace/katago-4090`. Do not edit the integration worktree in
place; use a new worktree if instrumentation is required. Every GPU command
must run through `gpu-lock with --gpu 0`. Do not use the RTX 5090 D.

## Question to answer

The accepted pure-forward path is still faster than the physical padded-B13
launch rate observed through normal GTP, even after matching duration and
eliminating thermal/boost variation. Determine where the remaining 2-4% comes
from and whether it is:

1. a measurement-definition artifact in `benchmarknn`;
2. position ramp/drain or search-side batch-supply loss;
3. CUDA Graph/external-event submission overhead;
4. H2D/D2H copy-engine or memory-system contention with compute;
5. host preprocess/postprocess/publication overhead that is not fully hidden;
6. a real difference in kernel duration or stream scheduling under the event
   pipeline; or
7. another mechanism supported by evidence.

Do not assume that 100% compute-kernel union busy proves equal forward
throughput. Copy traffic can slow kernels without creating an idle gap, and
aggregate GTP `nnBatches/s` includes position boundaries that a selected steady
Nsys interval does not.

## Mandatory metric definition

Every backend launch is padded to physical B13. Therefore the aligned backend
comparison metric is:

```text
physical launch row rate = nnBatches/s * 13
```

The printed real `nnEval/s` excludes dummy rows and is the correct
search-facing metric, but it is not aligned with `benchmarknn` pure-forward.
Continue to report visits/s, real nnEval/s, average real batch, and physical
launch row rate separately. Do not conflate them.

## Current controlled evidence

Integration worktree/revision:

- `/workspace/katago-infer-sm89-event-pipeline`
- `c4b92e3` for the tested binary; report-only commits follow it
- optimizer stream-interface base: `bd6b8a6`
- exact 19x19, FP16 NHWC, B13, S2, t96, graph event pipeline

Original unmatched figures:

- pure-forward: 3475.31 nnEval/s
- GTP: `258.37 nnBatches/s * 13 = 3358.81` physical launch rows/s
- aligned gap: 116.50 rows/s, 3.47% relative to GTP

Matched-duration default-boost ABBA (`pure, GTP, GTP, pure`):

| workload | run A | run B | mean physical rows/s |
| --- | ---: | ---: | ---: |
| pure-forward, 800 timed iterations | 3432.09 | 3413.87 | 3422.98 |
| GTP, ten positions | 3348.15 | 3345.16 | 3346.66 |

The matched gap is 76.32 rows/s, 2.28%. The hottest pure run was only 18.23
rows/s below the cooler run. Temperature reached 73 C; software and hardware
thermal-slowdown samples were always inactive. Under default boost, GTP's mean
clock was actually higher than pure-forward's because pure-forward was more
power-limited.

Strict equal-clock ABBA:

- requested 2200 MHz; every active sample of all four runs was exactly the
  hardware 2205 MHz bin;
- no power-cap or thermal-slowdown sample;
- pure-forward: 3261.27, 3258.55; mean 3259.91 rows/s;
- GTP: `242.35 * 13 = 3150.55`, `241.91 * 13 = 3144.83`; mean 3147.69 rows/s;
- aligned gap: 112.22 rows/s, 3.56%; and
- the two pure runs differed by only 2.72 rows/s despite different
  temperatures.

Thermal drift is therefore not the root cause. Matching duration removes about
40 rows/s from the original gap, but a repeatable gap remains.

## Existing Nsys evidence

Graph-node trace:

- 427 exact-B13 graph launches, 295 kernel nodes per launch;
- all graph kernels on two caller-owned non-blocking compute streams 15/51;
- no steady kernel on legacy/default/PTDS streams;
- H2D on streams 48/84 and D2H on 49/85, with no kernels on copy streams;
- 96.507% of H2D and 96.758% of D2H duration overlaps compute;
- both compute streams overlap for roughly 63% of collective busy time;
- four continuously supplied regions are 99.9953-99.9981% compute-kernel
  union busy; and
- no nonzero CUDA Runtime return code.

There is one 5.445 ms node-trace submission outlier correlated with 6.91/9.63
ms `cudaGraphLaunch` calls. It is not present as a throughput regression in the
unprofiled ten-position run and must not be generalized without a new trace.

## High-priority measurement concern

Audit `NNEvaluator::benchmarkPureForward` in
`cpp/neuralnet/nneval.cpp` around the calculation of
`combinedNNEvalsPerSec`. It sums `batchSize / perServerMedianSeconds` for two
servers. This is not the same estimator as physical batches divided by one
common wall interval. `actualWallSeconds` includes warmups and lifecycle
overhead, so it is not yet a clean replacement either.

Build a common wall-clock counter that measures completed physical B13 launches
over the same steady interval for both the direct pure-forward path and the
event-pipeline path. Report medians as latency diagnostics, not as the sole
aggregate-throughput definition. Establish how much of the gap disappears
under this corrected metric before changing kernels or scheduling.

## Requested investigation sequence

1. **Metric audit.** Measure completed B13 launches over a common timed wall
   interval for S2 direct forward and S2 event pipeline. Exclude initialization
   identically. Quantify the bias from summing per-server medians.
2. **Steady versus aggregate GTP.** From Nsys/NVTX or explicit counters, report
   physical batch rate for continuously supplied regions and separately the
   cost of first-batch ramp, each position transition, and final drain.
3. **Thread/supply control.** Repeat t96 and at least t112/t128 at equal clock.
   Confirm whether higher search concurrency changes physical batch rate or
   only visits/cache behavior.
4. **Copy-contention control.** At equal clock, compare key kernel durations and
   complete graph duration with and without overlapping pinned H2D/D2H. A
   synthetic/preloaded event-pipeline mode is acceptable as an investigative
   harness. Check DRAM/L2 effects, not merely SM busy union.
5. **Graph/event control.** Compare direct/eager event submission and CUDA
   Graph replay under the same physical input/output work and the same
   wall-clock counter. Preserve the external input-consumed/output-consumed
   contract.
6. **Host-path attribution.** Attribute preprocess, queue/admission, graph
   launch, D2H, postprocess/publication, and search-supply time. State which
   terms are hidden and which reach the critical path.
7. **Root-cause decision.** Produce a quantified decomposition that explains
   the gap to within 0.5 percentage point. Only then propose or implement a
   fix. Validate any fix with correctness replay and node-level Nsys.

Do not compare different commits, clocks, batch-capacity rounding, model
shapes, or real-row counts. Use ABBA order for performance decisions and record
all experiments in the existing 4090 history/protocol once the current dirty
stage is resolved.

## Required reading and artifacts

- Thermal/equal-clock analysis:
  `/workspace/results/4090/infer-event-pipeline-stage68/thermal-gap/analysis.md`
- Machine-readable summary:
  `/workspace/results/4090/infer-event-pipeline-stage68/thermal-gap/summary.txt`
- Telemetry and markers:
  `/workspace/results/4090/infer-event-pipeline-stage68/thermal-gap/`
- Nsys analysis:
  `/workspace/results/4090/infer-event-pipeline-stage68/nsys-gtp-graph-node/analysis.md`
- Nsys report/SQLite:
  `/workspace/results/4090/infer-event-pipeline-stage68/nsys-gtp-graph-node/gtp-c4b92e3-v2.nsys-rep`
  and `.sqlite`
- Current unprofiled GTP log:
  `/workspace/results/4090/infer-event-pipeline-stage68/aot/gtp-graph-t96-n10.log`
- Pure-forward log:
  `/workspace/results/4090/infer-event-pipeline-stage68/aot/pure-forward-s2-i100.log`
- Integration report:
  `/workspace/katago-infer-sm89-event-pipeline/docs/rtx4090-gtp-integration-report.md`

## Required handback

Write a concise investigation report under
`/workspace/results/4090/gtp-pipeline-gap-investigation/` containing:

- the corrected common-wall metric and raw counters;
- ABBA tables and telemetry;
- steady versus transition/drain decomposition;
- direct/eager/graph/copy-control results;
- Nsys/NCU evidence for any copy or kernel slowdown;
- root cause with uncertainty;
- recommended fix or a justified no-change decision; and
- exact source commit/worktree used.

Send the conclusion back to the integration session before merging anything.
