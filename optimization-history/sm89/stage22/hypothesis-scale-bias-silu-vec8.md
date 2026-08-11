# Stage 22 hypothesis: C768 affine + SiLU vec8 flat launch

Date: 2026-08-06 UTC

## Frozen target

- RTX 4090 (SM89), exact 19x19, B13, FP16, S2.
- Accepted Stage21 configuration is the control.
- This stage changes only `cudaUseScaleBiasSiluVec8Sm89=false/true`.
- Dispatch is restricted to unmasked NHWC SiLU with `(N,XY,C)=(13,361,768)`.

## Pre-implementation evidence

- The official scalar kernel launches `(3,361,13)x256`: 14,079 CTAs,
  3,604,224 threads, and 18.33 waves/SM for one C768 invocation.
- NCU records 11.648us and 12.000us for the sampled C768 invocations.
  L2 hit rate is 99.57% and 99.54% under the accepted persisting-L2 policy,
  while DRAM throughput is only 0.06%. This rejects a DRAM-bandwidth theory.
- Compute and memory-pipe throughput are both about 40%, achieved occupancy is
  61.6-61.9% despite 100% theoretical occupancy, and there is no spilling.
  The kernel is latency/issue limited with excessive scalar thread and address
  work rather than limited by a saturated execution or memory resource.

## Mechanism

Flatten the exact tensor into groups of eight contiguous half elements. One
thread performs one 16-byte vector load from input, scale, and bias, preserves
the official half FMA plus float `expf` SiLU for each element, and performs one
16-byte vector store. The launch falls to 1,760 CTAs and about 2.29 waves/SM.
This removes seven-eighths of the thread indexing and CTA scheduling work and
provides eight independent SiLU operations per thread for instruction-level
parallelism.

## Falsifiable tests and risks

1. The candidate must reduce isolated C768 duration and CTA/thread count. NCU
   must show vectorized bytes with no local spilling; register growth and the
   reduced 2.29-wave launch are explicit risks.
2. Nsys must retain 12 C768 invocations per forward with no added copies,
   allocations, or synchronization, and reduce their summed duration and the
   full S2 kernel union.
3. Only after both profiler checks pass, run locked-2400 forward/reverse ABBA.
4. Run the complete 8192-row all-head replay. The intended result is byte
   identity with Stage21 because arithmetic and per-element operation order are
   unchanged.

The route is rejected if lower parallelism or register pressure makes the
isolated kernel slower, even if the implementation matches the SM120 history.

## Result

Accepted and enabled for exact 19x19, B13, FP16, S2 on RTX 4090.

- Valid NCU changed the C768 median from 11.680us to 6.896us (-40.96%),
  grid size from 14,079 to 1,760, and threads from 3,604,224 to 450,560.
  Registers increased from 16 to 31 without spilling; achieved occupancy rose
  from 61.62% to 77.05%.
- Nsys retained 311 kernels/forward/stream. For the last 30 complete forwards
  on both streams, the target changed from 14.777us to 8.632us (-41.58%),
  kernel union from 251.930ms to 245.016ms (-2.74%), and summed kernel time
  from 387.119ms to 380.670ms (-1.67%). Copies, memsets, and synchronization
  counts were unchanged.
- Three valid locked-2400 forward/reverse ABBA rounds (r1/r3/r4) changed the
  pooled median from 3125.481 to 3155.317 nnEval/s (+0.9546%). Every round was
  non-negative, both order directions were positive, and 10/12 adjacent pairs
  were positive. r2 was discarded in full after one candidate process crashed
  during model startup before graph warmup or any timed forward.
- The complete 8192-row replay is byte-identical to Stage21, with the same
  SHA256 `7dde3f6b36e240eb4e92ffc632ecc578d059052fb2c13816b043d0a7093ba484`.

The first candidate NCU used an inherited launch-skip of 500 and captured no
target kernel because the new filter sees only C768 launches; it is retained as
an invalid no-sample attempt. The first Nsys dispatch audit included OS runtime
tracing and timed out during report generation after the benchmark completed;
it is also retained as contaminated and excluded from the decision.
