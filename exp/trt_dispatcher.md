# TensorRT Dispatcher Module

## Status

This document is the main design note for the dispatcher module that packages the conclusions from:

- `exp/plan/trt_io_design.md`
- `exp/test/README.md`
- `exp/test_scheduler/README.md`

## Module Goal

Expose a single-request API to callers while internally batching requests, scheduling them onto multi-GPU TensorRT slots, and overlapping:

1. host-side request filling
2. H2D
3. infer
4. D2H
5. host-side output consumption

The module is intentionally single-threaded on the dispatcher side. Callers may live on arbitrary threads, but all scheduling and slot-state mutation happens on one dispatcher thread to avoid locks in the hot scheduling path.

## External API

```cpp
class SingleRequestHandle {
public:
  void* inputs_mem_addr;
  void* outputs_mem_addr;
  Event input_mem_ready;
  Event output_mem_ready;
  Event output_mem_consumed;
};

SingleRequestHandle getOutput();
```

Expected caller usage:

```cpp
auto h = getOutput();
preprocess(data, h.inputs_mem_addr);
h.input_mem_ready.set();
h.output_mem_ready.wait();
postprocess(h.outputs_mem_addr, data);
h.output_mem_consumed.set();
```

The module owns all memory behind `inputs_mem_addr` and `outputs_mem_addr`. The caller only borrows it for the lifetime represented by the handle events.

## Design Summary

### 1. Host-side request storage

- The module maintains one global pinned host-memory ring.
- Each ring element is a `host slot`.
- One host slot contains all input and output memory for one dispatched batch.
- Input and output slabs are laid out contiguously within the same slot for CPU cache locality.
- The in-slot layout must preserve the "full batch can be copied as one contiguous slab" invariant.
- Default ring size:
  - `3 * sum(device_batch_size * device_slot_count)` across all GPUs.

### 2. Device-side execution storage

- Each logical execution slot is `(device, slot_index)`.
- Each logical execution slot owns two device banks: `ping` and `pong`.
- A device bank contains:
  - fixed input slab
  - fixed output slab
  - fixed TensorRT execution context
  - fixed infer stream
  - completion events for infer-end and D2H-end
- Infer-end-triggered launch of the next batch is allowed only because device-side storage is double-buffered.

### 3. Execution model

- The dispatcher is a single thread.
- Internally it uses coroutines / event-loop style state machines.
- No CUDA host callback is allowed.
- All CUDA completion handling is funneled back into the dispatcher thread.
- Wait policy is configurable:
  - `spin`
  - `park` / event-driven
- Both wait policies share the same state machine and scheduling logic.

## Core Invariants

### 1. Handle lifetime

For one request handle:

1. `getOutput()` returns writable input memory and reserved output memory.
2. Caller writes inputs.
3. Caller sets `input_mem_ready`.
4. Dispatcher eventually dispatches a batch containing this request.
5. Dispatcher sets `output_mem_ready` only after the request's output bytes are host-visible.
6. Caller reads outputs.
7. Caller sets `output_mem_consumed`.
8. Only after `output_mem_consumed` is observed may the host slot generation be reused.

### 2. Ring reuse must be generation-safe

- Every host slot carries a generation counter.
- Request-local events are generation-tagged.
- Reuse of a physical host slot must never observe stale `ready` / `consumed` state from the previous generation.

### 3. Device-bank reuse rules

- A device bank may start the next infer as soon as its previous infer is complete.
- A logical execution slot may keep overall throughput high because its alternate bank can still finish D2H for the previous batch.
- Reuse decisions are based on bank state, not only on logical slot state.

### 4. Scheduling state is dispatcher-owned

Only the dispatcher thread mutates:

- batch fill state
- ring head / tail ownership
- slot EWMA timing state
- bank state transitions
- launch reservations

Cross-thread communication from caller threads is restricted to event publication with release/acquire semantics.

## Memory Layout

## Host slot

Each host slot is one contiguous pinned allocation:

```text
+---------------------------------------------------------------+
| input slab | output slab | metadata / event generation state |
+---------------------------------------------------------------+
```

Rules:

- input slab and output slab are separately aligned
- tensor starts must preserve the packed-slab layout required by the full-batch fast path
- the layout must be identical across all host slots

## Device bank

Each device bank owns one fixed-address device allocation pair:

```text
+-------------------------+    +--------------------------+
| device input slab       |    | device output slab       |
+-------------------------+    +--------------------------+
```

TensorRT bindings are set once at initialization and never rebound on the hot path.

## Initialization

### Baseline initialization

1. Pre-size the global pinned host ring.
2. Initialize all host-slot metadata and generation counters.
3. Initialize per-device logical slots.
4. For every logical slot, allocate two device banks.
5. Initialize TensorRT engine resources.
6. Warm up every CUDA device context with:

```cpp
cudaSetDevice(device);
cudaFree(0);
```

### Architecture-phase placeholder

During architecture bring-up, actual infer may be replaced with `sleep(1ms)` or an equivalent synthetic stage, but the memory and scheduling state machines should remain unchanged.

## Hot-path Behavior

## 1. `getOutput()`

The dispatcher maintains an active host slot currently being filled.

When `getOutput()` is called:

1. Reserve one request row within the active host slot.
2. Return a `SingleRequestHandle` bound to that row.
3. If this reservation fills the host slot's batch capacity, arm an async dispatch coroutine for this batch.
4. Advance the ring head to the next host slot.
5. Before exposing the new host slot to callers, wait for the previous generation on that slot to report `output_mem_consumed`.

`getOutput()` is intentionally simple:

- one ring-slot reservation
- optional "batch just became full" transition
- optional wait for next-slot reuse safety

It does not itself perform heavy GPU work.

## 2. Full-batch dispatch

When a host slot becomes full:

1. Wait until all request rows in that host slot publish `input_mem_ready`.
2. Estimate the best `(device, logical_slot, bank)` using earliest-finish-time.
3. Launch:
   - H2D
   - infer
   - D2H
4. Mark the corresponding request handles `output_mem_ready` when their output bytes are host-visible.

Scheduling score remains:

```text
pred_finish =
  max(now + submit_budget, bank_ready_at)
  + pred_h2d
  + pred_infer
  + pred_d2h
  + uncertainty_penalty
```

The batch-ready timestamp is derived from the latest required input-ready event, not the earliest one.

## 3. Infer-end-triggered partial flush

After infer finishes on one logical slot:

1. If that logical slot has no other active infer in flight, check the current host-slot head.
2. If the head contains at least one request but is not full, dispatch it immediately as a partial batch.
3. This partial-batch path reuses the already chosen logical slot family, so it does not need a fresh cross-device estimate.

This is intentionally a throughput-oriented policy:

- steady state prefers full batches
- idle gaps are filled by partial batches rather than waiting indefinitely

## Wait Policy

The dispatcher supports two interchangeable wait policies.

### `spin`

- Busy-poll CUDA events and caller-visible events.
- Used for lowest-latency experiments.
- Higher CPU overhead is expected.

### `park`

- Use eventfd / condition-variable style wakeups, or another OS-backed blocking primitive.
- Same state machine, lower CPU overhead.
- Used to compare event-driven overhead against spin waiting.

The implementation must not fork into two independent schedulers. Wait policy is only a mechanism for suspending and resuming the single dispatcher thread.

## State Machines

## Host slot state

```text
empty
  -> filling
  -> waiting_input_ready
  -> dispatched
  -> output_ready
  -> output_consumed
  -> empty(next generation)
```

## Device bank state

```text
idle
  -> h2d_infer_d2h_submitted
  -> infer_done_d2h_pending
  -> host_visible
  -> idle
```

Logical-slot scheduling can treat:

- infer-complete bank and D2H-pending sibling bank as still healthy overlap
- bank-local readiness as the real resource boundary

## Current Recommendation

Use this as the v1 production shape:

- single dispatcher thread
- global pinned host ring
- per-logical-slot ping/pong device banks
- infer-end-triggered next launch
- full-batch preferred, partial-batch idle flush
- configurable `spin` vs `park` wait policy
- no CUDA host callback
- no hot-path TensorRT rebinding
