# CUDA Dual-Stream Trunk Phase Control

Status: design handoff

Date: 2026-08-06 UTC

Owners: CUDA common execution layer, with SM89 and SM120 integration owners

## 1. Summary

For exact 19x19/B13 inference, two independent NN server streams deliver much
higher throughput than one stream, but their uncontrolled relative phase makes
small kernel optimizations unstable. A locally faster kernel can move one
stream into a worse overlap pattern and reduce total S2 throughput.

This design introduces an architecture-neutral CUDA phase controller. The
controller deliberately staggers two same-GPU streams at the trunk boundary:
the follower may enter its trunk only after the leader has enqueued a selected
logical trunk checkpoint. SM89 and SM120 report the same logical checkpoints;
they do not implement pairing, policy, or peer-stream operations themselves.

Version 1 is an experimental benchmark/saturated-topology facility. It is not
enabled for production query queues until unequal-work and cancellation
semantics are implemented and tested.

## 2. Motivation and evidence

- Current RTX 4090 exact-19x19/B13 results are about 2476 nnEval/s for S1 and
  3271 nnEval/s for the recent S2 probe. S2 remains about 32% faster.
- The trunk contains nearly all useful work. Historical structured S2 traces
  attribute only about 0.4% exclusive busy to initial convolution and about
  0.6% to heads/other, while FFN input projection, QKV, attention, outer
  projection, linear2, normalization, and residual work all live in trunk.
- Multiple 4090 experiments reduced a local boundary but hurt S2. The observed
  mechanism is changed overlap and resource competition, not incorrect local
  profiling.
- Both implementations already use one private per-thread default CUDA stream
  per NN server thread. `serverThreadIdx` reaches `createComputeHandle`, and
  `ComputeContext` outlives every handle, so a shared controller can be added
  without changing search semantics.

The initial objective is not to make each forward lower latency. It is to find
a repeatable relative phase that minimizes the union of GPU-busy intervals for
two saturated streams.

## 3. Goals

1. Express phase policy once in the common CUDA layer.
2. Support the current SM89 independent forward and the current SM120 official
   forward with the same logical checkpoint numbering.
3. Keep all CUDA calls for a lane on that lane's owning host thread.
4. Use stream-ordered event dependencies, not device synchronization.
5. Permit controlled scans of leader, checkpoint, maximum lead, and policy.
6. Default to disabled and preserve current behavior bit-for-bit when disabled.
7. Expose enough NVTX and counters to explain every throughput result.
8. Fail closed to the normal uncontrolled path when the topology is unsupported.

## 4. Non-goals

- General scheduling across more than two streams in version 1.
- Pairing streams on different GPUs.
- A production-safe scheduler for sparse or unequal request queues in version 1.
- Kernel-internal synchronization between architectures.
- Replacing CUDA's block scheduler or assigning fixed SM partitions.
- Enabling CUDA Graph capture around dynamic host rendezvous in version 1.
- Hiding a backend-specific timing delay behind a supposedly common option.

## 5. Existing integration points

### Shared NN lifecycle

- `NNEvaluator` creates one shared `ComputeContext`.
- Each NN server thread creates one `ComputeHandle` and already passes
  `gpuIdxForThisThread` and `serverThreadIdx`.
- The context is destroyed only after all server threads and handles stop.
- CUDA uses `cudaStreamPerThread`; cuBLAS and cuDNN are explicitly bound to it.

### SM120 / RTX 5090 D

The current SM120 wrapper delegates the full forward to the official CUDA
`Model::apply`. The common official path owns `Trunk::apply`, recursive
`BlockStack` execution, and the trunk/head boundary. Version 1 checkpoints for
SM120 must be emitted in this official path. The wrapper must not emit the same
checkpoint a second time.

### SM89 / RTX 4090

SM89 owns an independent `Sm89Forward`. It must emit equivalent checkpoints at
its own trunk boundary, outer-block loop, and nested inner-block loop. Its
`Sm89Ctx::stream` is still the owning server thread's PTDS and can be passed to
the common lane API.

## 6. Public CUDA-only interface

Add a small common CUDA component, for example:

```text
cpp/neuralnet/cudastreamphase.h
cpp/neuralnet/cudastreamphase.cpp
```

It is compiled only for CUDA builds. It must not include SM89 or SM120 types.

```cpp
enum class CudaPhasePoint : uint8_t {
  ForwardBegin,
  TrunkBegin,
  TrunkCheckpoint,
  TrunkEnd,
  ForwardEnd,
};

struct CudaPhaseToken {
  uint64_t generation;
  uint32_t lane;
  bool active;
};

class CudaPhaseLane {
 public:
  CudaPhaseToken beginForward(cudaStream_t stream, bool isWarmup);

  void arrive(
    CudaPhaseToken& token,
    CudaPhasePoint point,
    uint32_t ordinal,
    cudaStream_t stream
  );

  void endForward(CudaPhaseToken& token, cudaStream_t stream);
};
```

`CudaPhaseLane` is a per-handle view of a shared `CudaPhaseController`. Backend
code can only report its own progress. It cannot name or manipulate a peer
stream.

`beginForward` and `endForward` are the only operations that create and retire
a forward generation. The caller invokes each exactly once per device forward.
They also represent the logical `ForwardBegin` and `ForwardEnd` points;
backend code must not send duplicate `arrive(ForwardBegin/ForwardEnd)` calls.
`arrive` is used for the trunk points between them.

### Required ownership

- `ComputeContext` owns one controller registry.
- The registry is keyed by physical CUDA device.
- `createComputeHandle(context, ..., gpuIdx, serverThreadIdx)` lazily registers
  a lane using `(gpuIdx, serverThreadIdx)`.
- `ComputeHandle` or `CudaHandles` owns the returned lane handle.
- Lane unregister wakes all host waiters and marks outstanding generations
  cancelled.
- The controller and CUDA events are destroyed after all lanes are gone.

The `gpuIdxs` passed to `createComputeContext` are deduplicated and therefore do
not describe lane count. Lane count must be learned from handle registration.

## 7. Logical checkpoint contract

The first implementation must report these points for every device forward:

```text
ForwardBegin
TrunkBegin
TrunkCheckpoint(0..N-1)
TrunkEnd
ForwardEnd
```

`TrunkCheckpoint` uses a flattened deterministic ordinal. The ordinal is a
property of the model execution order, not the source-code nesting depth.

Recommended version-1 sequence:

1. checkpoint after initial trunk projection and global-feature addition;
2. checkpoint after each top-level trunk block;
3. for nested bottleneck blocks, checkpoint after each inner transformer block;
4. checkpoint after nested post-projection/residual completion;
5. checkpoint after final trunk normalization.

Both backends must produce the same sequence for the same model. Add a debug
mode that records `(point, ordinal, block kind)` and rejects activation if the
two lanes report different sequences during a dry run.

The selected marker should normally be inside trunk. Waiting for `TrunkEnd`
would serialize almost all useful work and is allowed only as a negative
control.

## 8. Phase policy

Version 1 supports exactly two lanes on the same device:

- leader: configurable lane, default lower `serverThreadIdx`;
- follower: the other lane;
- marker: one flattened trunk checkpoint ordinal;
- follower gate: follower `TrunkBegin(g)` waits for leader marker `(g, N)`;
- maximum lead: optional generation backpressure preventing the leader from
  running arbitrarily far ahead.

The minimal policy is:

```text
leader generation g:
  enqueue trunk work through checkpoint N
  record event E[g]
  publish E[g]

follower generation g:
  complete frontend work
  wait on host until E[g] is published or cancelled
  cudaStreamWaitEvent(followerStream, E[g])
  enqueue trunk
```

Frontend/head work remains free to overlap. Only trunk admission is gated.

### Preventing phase drift

A one-way follower wait does not by itself stop the leader from getting several
forwards ahead. The controller therefore needs a configurable lead window:

```text
cudaDualStreamPhaseMaxLead = 1..ringSize-1
```

Before publishing generation `g`, the leader waits on the host until the
follower has enqueued its wait for generation `g-maxLead`. This is enqueue
backpressure, not GPU synchronization. A stricter cyclic policy that also gates
leader generation `g+1` on a follower checkpoint may be added as a separate
mode and measured independently.

## 9. Event and generation protocol

Use `cudaEventCreateWithFlags(..., cudaEventDisableTiming)`.

Each device pair owns an event ring. Every slot stores at least:

```cpp
struct Slot {
  cudaEvent_t event;
  uint64_t generation;
  enum { Empty, Published, WaitEnqueued, Cancelled } state;
};
```

Protocol:

1. Both lanes increment a local forward generation only for active forwards.
2. Leader waits until the target slot is reusable.
3. Leader records the event on its own stream at the configured checkpoint.
4. Under the controller mutex, leader sets exact generation and `Published`,
   then notifies the host condition variable.
5. Follower waits on the host for the exact generation to become `Published`.
6. Follower calls `cudaStreamWaitEvent` on its own stream.
7. Follower marks `WaitEnqueued`; only then may the slot eventually be reused.

Exact generation matching is mandatory. A ring index alone has an ABA hazard
when host enqueue runs ahead of GPU execution.

A slot for generation `g` is reusable only when its previous exact generation
is `g - ringSize` and its state is `WaitEnqueued` (or the pair has been
cancelled). Comparing only the ring index or an aggregate acknowledgement
counter is incorrect.

No coordinator thread may record into or wait on a saved
`cudaStreamPerThread`. Every CUDA API call occurs in the corresponding NN
server thread.

## 10. Activation and deadlock safety

### Warmup

Warmup is asynchronous across NN server threads and happens before all servers
are ready. `beginForward(..., isWarmup=true)` must always return an inactive
token. Registering the second lane must not implicitly activate phase control.

### Explicit activation

Version 1 activation occurs only after all benchmark server threads have:

1. created handles;
2. completed backend/model warmup;
3. completed the untimed input-populating `getOutput`;
4. reached the benchmark start barrier.

Add a CUDA backend interface such as:

```cpp
bool setPhaseControlActive(ComputeHandle* handle, bool active);
```

This is a pair-wide rendezvous, not a per-handle switch. Each benchmark thread
calls it after the shared start flag and before its first timed/warm benchmark
forward. An `active=true` call only submits that lane's ready state. The
controller changes `Activating -> Active` atomically only after both registered
lanes are ready, assigns a shared activation epoch and generation zero, and
then releases both calls with the same success result. Neither lane may obtain
an active token while the other lane is still inactive.

Stopping uses a symmetric pair-wide protocol:

1. both benchmark lanes stop starting new forwards;
2. both finish their last forward and leave `benchmarkOutput`;
3. both reach a benchmark end barrier;
4. both call `setPhaseControlActive(..., false)`;
5. only after both requests arrive does the controller atomically transition
   `Draining -> Inactive` and release both callers.

The current benchmark has no suitable end barrier, so version 1 must add one.
One lane must never deactivate the controller while its peer can still enqueue
an active forward. Handle teardown remains a cancellation fallback, not the
normal deactivation protocol.

The controller state machine is therefore at least `Inactive`, `Activating`,
`Active`, `Draining`, and `Cancelled`. Activation/deactivation rendezvous waits
are bounded and report one pair-wide result; a timeout cancels the pair and
wakes both callers.

Do not infer activation from `isWarmup == false`: the untimed normal
`getOutput` before the benchmark barrier would otherwise deadlock.

### Unsupported topology

Activation returns false and logs a single reason if any condition fails:

- not exactly two registered lanes;
- lanes use different physical GPUs;
- duplicate or unstable `serverThreadIdx`;
- CUDA Graph mode is active;
- checkpoint sequence mismatch;
- controller is cancelled;
- mode is not explicitly permitted by the caller.

The caller then continues with ordinary uncontrolled execution.

### Cancellation

Handle destruction, exception propagation, or benchmark abort marks the pair
cancelled and wakes every host waiter. No thread may block indefinitely waiting
for a generation that cannot be published.

Version 1 must use a bounded host wait with a diagnostic failure in benchmark
mode. It must not silently continue after one lane has already inserted only
part of a phase protocol.

## 11. Production scope

Real query queues can leave one server idle or give the two servers different
batch counts. Pairing local forward number `g` blindly can deadlock or increase
latency. Therefore:

- version 1 is enabled only by `benchmarknn` or an explicit saturated-run mode;
- ordinary search/analysis serving ignores the phase options;
- production enablement requires scheduler-issued pair IDs or an opportunistic
  protocol with timeout/cancel semantics designed at the queue layer;
- production latency and throughput must be reported separately.

This restriction is part of correctness, not merely a rollout preference.

## 12. Configuration

All options are common CUDA keys parsed by `ComputeContext`, not SM89/SM120
options:

```text
cudaDualStreamPhaseControl = false
cudaDualStreamPhaseMode = trunk-stagger
cudaDualStreamPhaseLeader = 0
cudaDualStreamPhaseCheckpoint = 0
cudaDualStreamPhaseEventRing = 8
cudaDualStreamPhaseMaxLead = 1
cudaDualStreamPhaseScope = benchmark
cudaDualStreamPhaseDebug = false
```

Validation:

- event ring: 2..64;
- max lead: 1..eventRing-1;
- checkpoint: within the dry-run sequence;
- scope: initially only `benchmark`;
- unknown mode or invalid combination: configuration error, not silent clamp.

When disabled, no events, host waits, NVTX ranges, or extra per-block atomic
operations should be present in the forward path.

## 13. Backend integration

### Common CUDA handle

`CudaHandles` gains:

```cpp
std::shared_ptr<CudaPhaseLane> phaseLane;
CudaPhaseToken phaseToken;
```

`ComputeHandle::apply` calls `beginForward` once before backend dispatch and
`endForward` once after dispatch. A small inline/no-op wrapper prevents
checkpoint code from depending on controller internals.

### Official / SM120 path

- `Model::apply`: `TrunkBegin` and `TrunkEnd`; `ComputeHandle::apply` owns the
  corresponding `beginForward`/`endForward` calls.
- `Trunk::apply` / `BlockStack`: flattened `TrunkCheckpoint` calls.
- The SM120 wrapper does not duplicate these while it delegates official apply.
- If SM120 later owns a standalone full forward, move checkpoint ownership to
  that forward and disable official reporting for that dispatch.

### SM89 path

- `Sm89Forward::apply`: `TrunkBegin` and `TrunkEnd`; `ComputeHandle::apply`
  owns the corresponding `beginForward`/`endForward` calls.
- `Sm89Trunk` outer/nested loops: same flattened checkpoint cursor.
- Pass the common lane/token into SM89; do not create an SM89 controller.

## 14. Instrumentation

Emit NVTX ranges/marks only when phase control or debug is enabled:

```text
phase/generation/<g>/leader-marker/<N>
phase/generation/<g>/follower-host-wait
phase/generation/<g>/follower-stream-wait
phase/generation/<g>/cancel
```

Record counters in a structured summary:

- paired and unpaired forward counts;
- per-lane generation range;
- host wait p50/p90/p99/max;
- event-wait enqueue count;
- cancellation/timeout count;
- slot reuse waits;
- checkpoint sequence/hash for each backend;
- selected policy and config;
- kernel-union, overlap, and idle-gap statistics from Nsys.

## 15. Experimental method

Follow the GPU optimization evidence chain for each policy point.

### A. Baseline characterization

1. Lock the target GPU clocks using the architecture's established regime.
2. Capture S2 exact-19x19/B13 Nsys with phase control disabled.
3. Segment frontend, each flattened trunk checkpoint interval, trunk end, and
   heads for both streams.
4. Measure union busy, simultaneous busy, single-stream exclusive busy, gaps,
   and relative checkpoint offsets.
5. Use NCU on representative overlapping pairs to identify compute, memory,
   register, shared-memory, wave, or cache contention.

### B. Marker scan

Use short scans first:

- both choices of leader;
- checkpoint 0..N-1, coarse outer-block scan before inner-block refinement;
- max lead 1 and 2;
- disabled control interleaved with candidates;
- 20 timed forwards per Nsys order;
- 100 timed forwards for the one best candidate only.

Reject immediately if:

- a wait serializes most of trunk;
- union busy or combined throughput is negative in both orders;
- event or host gaps become a new material critical path;
- generations mismatch or cancellation occurs.

### C. Confirmation

For a surviving marker:

1. forward and reverse locked ABBA;
2. Nsys proof of the intended stable phase and improved union;
3. NCU proof that the improvement matches the predicted contention change;
4. S1 unchanged check;
5. complete 8192-row all-head FP32 replay, even though ordering should not
   change arithmetic within a stream;
6. repeated startup/teardown and injected failure tests.

Do not use long tests for broad scanning. Only one profiler-supported policy
reaches the short ABBA confirmation in an iteration.

## 16. Acceptance criteria

An implementation is accepted per architecture only if all are true:

- disabled path has no measurable regression and no extra CUDA work;
- exact two-lane/same-GPU activation is deterministic;
- warmup, cancellation, and unsupported topology cannot deadlock;
- Nsys shows the configured phase rather than incidental host timing;
- S2 combined throughput improves in forward and reverse order;
- improvement survives the architecture's locked short ABBA;
- no new material host gap, device-wide sync, or H2D/D2H appears;
- complete FP32-reference accuracy remains within the frozen envelope;
- SM89 and SM120 use the same public interface and logical checkpoint contract.

Acceptance on one architecture does not imply acceptance on the other. The
interface is shared; the optimal leader/checkpoint policy is architecture data.

## 17. Test plan

### Unit/state-machine tests

- lane registration and exact-two activation;
- duplicate lane and cross-GPU rejection;
- generation/ring wrap without ABA;
- leader/follower cancellation at every state;
- timeout wakes both lanes;
- warmup always returns inactive;
- checkpoint sequence mismatch disables the pair;
- repeated activate/deactivate cycles.

### CUDA integration tests

- two PTDS streams with synthetic kernels verify event ordering;
- event reuse after `WaitEnqueued` does not change prior waits;
- no `cudaDeviceSynchronize` or legacy-stream dependency;
- handle teardown after queued waits completes safely;
- CUDA Graph combination is rejected;
- disabled execution has identical launch sequence.

### Model tests

- SM89 and SM120 produce the same logical checkpoint trace for this model;
- S1 ignores phase configuration;
- S2 B13 exact path activates;
- B12/B14, non-19 boards, S3, and multi-GPU fall back cleanly in version 1;
- accuracy replay is unchanged.

## 18. Rollout

1. Land common state machine and synthetic tests with all runtime options off.
2. Integrate official/SM120 checkpoint reporting and validate the 5090 D trace.
3. Integrate SM89 checkpoint reporting and validate the 4090 trace.
4. Compare checkpoint hashes and reconcile any semantic mismatch.
5. Run short marker scans independently on each architecture.
6. Enable only in benchmark configs for an architecture with positive evidence.
7. Design queue-issued pair IDs before any production-search experiment.

## 19. Open questions

- Is the best marker stable across model revisions with the same block layout?
- Is a checkpoint ordinal sufficient, or should policy target a checkpoint kind
  plus occurrence to remain readable across architecture implementations?
- Does leader generation backpressure need a follower GPU checkpoint rather
  than host `WaitEnqueued` acknowledgement?
- Would stream priorities provide a simpler secondary control after a stable
  phase is established?
- Can CUDA Graph child graphs represent the trunk segments without losing the
  dynamic inter-stream event dependency?
- For production, should the queue form explicit two-batch cohorts, or should a
  lane opportunistically run uncontrolled after a bounded admission deadline?

These questions are experimental follow-ups. They do not weaken the version-1
benchmark safety restrictions above.
