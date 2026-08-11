# H25: Make fixed-B13 S2 phase sensitivity observable

## Purpose

This is diagnostic infrastructure, not a production scheduling change. The
fixed B13/S2 forward consists of two identical, repeatable kernel sequences,
but their overlap depends on the initial stream phase and resource contention.
Aggregate kernel duration cannot distinguish a robust kernel improvement from
a favorable phase accident.

## Evidence

- The Stage-24 timed trace contains exactly 344 kernels per B13 forward on
  each stream.
- Its natural iteration-start offset has median `-3.78 us` but ranges from
  `-8.38` to `+34.18 us`.
- Sequence-aligned S2/S1 duration ratios identify large contention in
  library GEMMs (`1.70x`), linear2 (`1.63x`), and wide QKV (`1.28x`), while
  fused FFN is only `1.10x` despite being the largest absolute family.

## Mechanism

Add an opt-in `benchmarknn --phase-offset-us` diagnostic. After each server
has completed backend warmup and synchronized the GPU, all benchmark threads
meet at a host barrier. Thread 0 begins the timed enqueue loop immediately;
thread N waits `N * phase-offset-us` before recording its first timed event.
The delay is outside every per-forward CUDA-event interval.

The option defaults to disabled (`-1`), preserving the established benchmark
protocol. Offset `0` enables the barrier with aligned release. Positive values
seed a controlled initial phase; the streams then run freely, so the result
still exposes the GPU scheduler's natural steady state rather than imposing a
per-iteration barrier.

## Validation

1. Build CUDA and TensorRT declarations consistently.
2. Default (`-1`) smoke must retain current behavior and JSON compatibility.
3. S2 offsets must appear in JSON and Nsys must show the requested direction
   and a materially changed initial phase.
4. Sweep offsets across at least one forward period with short runs. Record
   throughput and the actual Nsys phase for representative minimum/maximum
   points.
5. Use the curve diagnostically: an optimization is robust only if its gain is
   positive over representative phases, not merely at the original phase.

Host wakeup precision is not assumed to equal the requested microseconds. Nsys
actual GPU start offsets, rather than the requested value, are the evidence.
