# RTX 4090 B13/S2 optimization iteration protocol

Target deployment scope is fixed to RTX 4090 SM89, exact 19x19, batch 13,
FP16, and two server streams. S2 remains the only end-to-end optimization
target. S1 may be used only as a phase-free attribution test for an operator
whose NCU evidence already shows less work or lower resource use.

After every retained optimization (`intrinsic-accepted` or a reusable
`mechanism-accepted` implementation), rerun the current-best full-graph
checkpoint. After a rejected or statically disproved experiment that is fully
reverted, restore and verify the current-best source/binary, then reuse the most
recent clean checkpoint; do not spend another Nsys/NCU pass measuring identical
code and configuration.

The full-graph checkpoint must use the currently deployed submission path. For
exact B13/S2 on RTX 4090 this is the eager event-gated pipeline, not CUDA Graph.
Stage 71 measured graph replay 1.735% slower in real GTP because it fixed a more
resource-contended two-stream phase. Do not enable graph merely to reduce launch
calls when eager Nsys is already continuously supplied.

1. Restore or freeze the current best binary and record its SHA256.
   Record observed SM and memory clocks from telemetry; a successful lock API
   call or requested value is not a substitute for the observed clock.
2. Run a short full-graph S2 Nsys checkpoint: 10 warmups and 20 timed forwards
   per stream at locked 2400 MHz. Rank families by two-stream interval union and
   only-family exclusive time, not summed kernel duration alone.
3. Run a broad S2 NCU census with one representative of every distinct launch
   geometry and a small metric set. Combine its resource/throughput evidence
   with the Nsys ranking.
4. Take 2-3 detailed NCU samples only for the newly ranked top actionable
   family. A "single-kernel" sample must be isolated from an S2 invocation.
5. Write a falsifiable hypothesis before implementation. Prefer the largest
   graph boundary or scheduling mechanism that the evidence supports; do not
   keep descending into one kernel after the full-graph ranking changes.
6. Build and smoke-test, then profile 2-3 candidate samples and run short
   forward/reverse 20-iteration S2 Nsys. A stable positive candidate enters one
   locked 100-iteration S2 ABBA.
7. If S2 is neutral, order-sensitive, or negative while NCU proves strictly
   fewer launches/bytes/instructions or lower occupancy pressure, run a short
   S1 Nsys and one 100-iteration S1 ABBA. This is an attribution test, not a
   topology change or a new throughput target.
8. Classify implementation retention separately from S2 deployment:
   - `intrinsic-accepted / S2-enabled`: intrinsic evidence passes and S2 gains;
     keep the implementation and enable it in the current S2 config.
   - `intrinsic-accepted / S2-neutral`: S1 and NCU pass, S2 has no significant
     regression; keep the implementation default-off and queue it for phase-
     controller retest.
   - `intrinsic-accepted / S2-regressed`: S1 and NCU pass but S2 regresses;
     keep the implementation default-off with the measured regression.
   - `mechanism-accepted / throughput-neutral`: NCU proves less work and local
     boundaries improve, but neither S1 nor S2 has stable throughput gain; keep
     a simple reusable implementation default-off without calling it accepted.
   - `mechanism-rejected`: correctness fails, NCU disproves the mechanism, or
     both isolated and deployed measurements regress; reverting source is then
     appropriate.
9. Run the 8,192-position FP32-reference regression for every retained
   implementation, including throughput-neutral reusable mechanisms, after its
   performance/attribution gate. Record deployment status separately from
   implementation status.
10. After every retained outcome, rerun the current S2-enabled full-graph Nsys
    and broad NCU checkpoint to choose the next macro boundary. For a fully
    reverted rejection, link the latest clean checkpoint instead. Re-profile a
    rejection only if the restored binary/config differs or the earlier
    checkpoint was invalid. A retained default-off implementation does not
    silently become the new current-best profile.

For aggregate throughput, count completed physical B13 launches over one common
wall interval. Never sum per-lane median latencies and label the result as S2
throughput. Per-lane medians remain latency diagnostics. In real GTP, report
visits/s, real `nnEval/s`, average real batch, and `nnBatches/s * 13` physical
rows separately.

Keep the two inference streams at equal default priority and leave
`CUDA_DEVICE_MAX_CONNECTIONS` unset. Stage 71 found that one connection destroys
S2 concurrency, 2-32 have no repeatable winner, and a one-level priority
asymmetry causes low-priority-lane starvation. Any future retest of these global
scheduling controls needs a natural-execution Nsys boundary; NCU replay cannot
preserve the cross-stream phase under test.

## Strict-local accumulation lane

An optimization is added to a cumulative `strict-local bundle` when correctness
passes, NCU plus the complete local boundary prove strictly less launch, byte,
instruction, or resource work, and S1 does not show a stable regression. Stable
positive S1 evidence earns the stronger `intrinsic-accepted` label; S1-neutral
mechanisms remain explicitly labeled `mechanism-accepted` but may still
accumulate. An individual S2 result may be neutral, negative, or order-sensitive.
The bundle is a candidate implementation set, not the deployed topology.

After every newly qualifying strict-local optimization:

1. Retest the entire cumulative bundle against the current S2-enabled control
   with short forward/reverse full-graph Nsys. Do not require each component to
   pass an independent S2 gate before entering the bundle.
2. If the bundle is stable-positive in both orders, run one short locked S2
   ABBA and the joint 8,192-position accuracy gate, then enable the bundle as a
   unit if both pass.
3. If the bundle remains phase-sensitive or regresses, keep each qualifying
   component default-off with its precise intrinsic/mechanism label and carry
   the complete bundle forward. Add the next strict local improvement and
   retest; do not delete accumulated work merely because the uncontrolled
   two-stream phase currently masks it.
4. Avoid an exhaustive subset search. Prefix/subset ablations are diagnostic
   only when needed to explain a stable regression; the primary experiment is
   always deployed-current-best versus the complete intrinsic bundle.

This lane tests whether many individually small reductions in launch count,
bytes, instructions, or occupancy pressure eventually cross a meaningful S2
threshold. It does not redefine S1 as the deployment target.

The broad NCU census is deliberately not a full detailed replay of every launch:
that destroys iteration speed and serializes away the S2 schedule. Its job is to
screen all kernel families; detailed replay is reserved for the top candidate.

Historical routes rejected only by S2 phase behavior must be audited under this
split classification. Existing code should not be deleted merely because an
uncontrolled two-stream phase shift hid an otherwise demonstrated reduction in
work.

## Composed-boundary measurement rule

Nsight Compute replay durations are not additive. Never reject a fusion, split,
reorder, or intermediate-materialization change by summing independently
profiled control kernels and comparing that sum with a separately replayed
candidate kernel (or the reverse). NCU remains authoritative for instruction,
register, shared-memory, spill, occupancy, cache, and stall-mechanism evidence,
but it is not a cross-kernel latency model.

For every one-to-many, many-to-one, or reordered boundary:

1. Measure both complete variants directly in one natural, non-NCU-replayed
   execution using Nsys/NVTX intervals or a CUDA-event subgraph benchmark.
   Include all launches, launch gaps, copies, reformats, and materialized
   intermediates inside the boundary.
2. Use the same stream topology and cache/warmup protocol for control and
   candidate. Report per-stream elapsed boundary plus S2 union/exclusive time;
   do not substitute summed overlapping durations.
3. An unfavorable sum of isolated NCU durations may lower priority, but may not
   stop the short complete-boundary measurement. Record any disagreement
   between NCU replay sums and the natural boundary as profiler perturbation,
   not as unexplained benchmark noise.
4. Only the naturally measured boundary can pass or fail the local performance
   gate. Whole-graph S2 still separately controls deployment.

Stage 64 established the motivating counterexample: isolated NCU suggested
plain-QKV plus standalone RoPE was slightly slower than fused QKV+RoPE, while a
natural S1 Nsys trace measured the complete launch-to-completion boundary
10.34% faster (the less-complete sum of raw kernel durations was 12.71% lower).
Historical Stage 55 and Stage 58 rejections used the now-invalid additive-NCU
gate and must be classified as pending revalidation rather than conclusively
mechanism-rejected.
