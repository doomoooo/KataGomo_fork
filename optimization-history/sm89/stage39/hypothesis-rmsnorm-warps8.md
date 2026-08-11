# Stage 39 hypothesis: eight rows/warps per RMSNorm CTA

## Frozen target

- RTX 4090 (SM89), exact 19x19, B13, FP16, S2 only.
- Accepted post-Stage-38 source and binary are the control.
- The only candidate variable is the C384 RMSNorm CTA geometry: four rows and
  128 threads become eight rows and 256 threads.
- One warp still owns exactly one row, each lane still processes the same 12
  scalar half elements, and the FP32 sum/shuffle/rsqrt/output order is unchanged.

## Pre-implementation profiler evidence

Current S2 Nsys contains 4,356 RMSNorm launches per short capture, or 66 per
forward, at roughly 5.8-6.8 us average and about 26 ms summed capture time.

The established three-sample S2 NCU control is 4.70/4.54/4.61 us. It launches
1,174 CTA x 128 threads, uses 40 registers/thread and no dynamic shared memory,
has 0.76 waves/SM and 100% theoretical occupancy, but achieves only about
54.7% occupancy. About 66.5% of scheduler cycles have no eligible warp and
long-scoreboard stalls account for 8.3-8.5 cycles per issued instruction.

Stage32 changed per-lane memory mapping and failed because wider load-use chains
increased scoreboard stalls. Stage39 leaves all per-warp work and memory mapping
unchanged. It tests only whether halving block scheduling/tail bookkeeping from
1,174 to 587 CTA improves execution while retaining the same total warp count
and nominal 0.76-wave launch.

## Falsifiable gates

1. Build and 26-row S2 replay must be byte-identical to the frozen control
   binary.
2. Three-sample candidate NCU must show 587 CTA x 256 threads, 40 registers,
   no spill, and a median duration at least 3% below the 4.61 us control.
   Achieved occupancy and no-eligible cycles must not regress.
3. Only if NCU passes, run locked-2400 20-iteration S2 Nsys with frozen control
   and candidate binaries in forward and reverse order. Both must reduce
   RMSNorm duration and improve end-to-end throughput; report summed kernel
   time and GPU busy union.
4. Only if both short orders pass, run one locked-2400 100-iteration S2 ABBA.
5. Only after the ABBA performance gate passes, run 8192-row FP32 accuracy.

Any failed gate restores the four-warp source and rebuilds. No S1 measurement
is used.

## Result

Rejected and reverted at the NCU gate.

- The 26-row S2 replay was byte-identical to the frozen control.
- Candidate NCU confirmed the intended geometry: 587 CTA x 256 threads versus
  1,174 x 128, with the same 0.76 waves/SM, 40 registers/thread, 100%
  theoretical occupancy, and zero spill.
- Achieved occupancy improved from the control's roughly 54.7% to
  58.8-60.1%, and no-eligible cycles improved slightly from about 66.5% to
  66.0-66.1%.
- Duration was 4.58/4.61/4.58 us, a 4.58 us median versus the 4.61 us control
  (-0.65%). This failed the predeclared 3% local threshold.

The CTA grouping mechanism is real but its absolute gain is only about 0.03 us
per launch, below the threshold needed to justify a phase-sensitive full-graph
test. No Nsys, ABBA, or 8192-row replay was run. The four-warps-per-CTA source
was restored and rebuilt.
