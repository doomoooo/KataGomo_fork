# Hypothesis: common-wall physical-launch harness

Date: 2026-08-06 UTC

Base: `/workspace/katago-gtp-pipeline-gap`, `d5c5b36` (tested source is `c4b92e3`;
the two later commits are documentation only).

The current `benchmarknn` aggregate is the sum of `B13 / per-lane median GPU
duration`. Its host wall interval includes warmup plus creation/destruction of
two timing events for every iteration on every lane, so it cannot serve as a
physical-launch wall-rate check. The median sum may also differ from launches
completed over one common concurrent interval.

Add an investigative benchmark phase with these invariants:

- all lanes finish initialization, input preparation and warmup before a
  common start barrier;
- each lane submits exactly the requested number of physical B13 launches;
- no per-iteration timing event is inserted in this phase;
- the common interval ends only after both lanes' final work completes;
- report `completed launches * 13 / common wall seconds`;
- retain the old per-lane event medians only as a separate latency diagnostic.

Use the same phase for four backend submission modes:

1. direct eager forward, device-resident inputs/outputs;
2. event pipeline eager with pinned H2D/D2H and external events;
3. event pipeline CUDA Graph with the same copies/events;
4. event pipeline CUDA Graph with event topology preserved but copies omitted.

Predictions:

- the corrected direct common-wall rate will be below the sum-of-medians
  estimator if the old estimator is optimistic;
- graph+copy versus graph-no-copy isolates copy-engine/memory contention;
- graph+copy versus direct isolates the full device-side event/copy/graph cost;
- if graph+copy steady rate is close to direct while aggregate GTP remains
  lower, position ramp/drain or search supply is the dominant residual.

All production defaults remain unchanged. Copy omission is benchmark-only and
must never be used by the normal scheduler.
