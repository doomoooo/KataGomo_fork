# Stage 38 hypothesis: legal column-major B path for fused QKV+RoPE

## Frozen target

- RTX 4090 (SM89), exact 19x19, B13, FP16, S2 only.
- Accepted post-Stage-37 source is the control.
- Candidate is default-off behind `cudaUseQKVColumnMajorBSm89`.
- The QKV tile, warp tile, stage count, epilogue arithmetic, grid, and output
  layout remain fixed. Only the weight storage and CUTLASS B layout change.

## Pre-implementation profiler evidence

Current S2 Nsys records fused QKV+RoPE as 2,178 launches at about 29.5-30.2 us
average and roughly 64-65 ms summed per short capture, making it a top-three
kernel family.

Current three-sample NCU measures 24.70/29.63/30.24 us (29.63 us median), 240
registers/thread, 49.15 KiB dynamic shared memory, 1.30 waves/SM, about 15%
achieved occupancy, about 79.3% no-eligible cycles, and zero spill. Stage16's
full NCU sections additionally flag poor global-load sector utilization and
about 2.1-way shared-store bank conflicts.

Stage36 proved that directly changing row-major B's warp-raked thread map is
invalid because the global-copy lane map is coupled to its congruous shared
layout. Static inspection identifies a legal single-variable alternative:
CUTLASS's row-major-A/column-major-B specialization uses its own 4x8 B warp
map and `ColumnMajorTensorOpMultiplicandCrosswise` shared layout, with the
global producer and MMA consumer generated as one matched type.

## Mechanism and implementation

At model initialization, transpose each fixed 384x384 Q/K/V weight matrix from
row-major KxN into a column-major KxN device copy. At runtime, select a second
instantiation of the same M128xN128xK32, warp M64xN64xK32, stage-3 batched GEMM
and the same RoPE output iterator. No runtime transpose, extra launch, split,
or arithmetic change is allowed.

This tests whether the alternative legal B iterator reduces the observed
global-sector waste and shared-store conflicts enough to lower latency. It is
not the invalid Stage36 16x2 alias substitution and does not depend on a newer
toolkit.

## Falsifiable gates

1. Build and 26-row S2 replay must be byte-identical to control.
2. Three-sample candidate NCU must show the expected matched column-major B
   type, no spills, and a median duration at least 2% below 29.63 us. Report
   registers, shared memory, occupancy, no-eligible cycles, global-load sector
   utilization, and shared-store bank conflicts even if duration fails.
3. Only if NCU passes, run locked-2400 20-iteration S2 Nsys in forward and
   reverse order. Both must improve QKV and end-to-end throughput; also report
   summed kernel time and GPU busy union.
4. Only if both Nsys orders pass, run one locked-2400 100-iteration S2 ABBA.
5. Only if ABBA passes, run the full 8192-row FP32-reference replay and enable
   the option in the S2 config.

Any failed gate rejects and removes the candidate implementation. No S1
measurement is used.

## Result

Rejected and fully reverted after the two-order S2 Nsys gate.

- The candidate built and its 26-row S2 replay was byte-identical to control.
- Paired full-section NCU measured control at 24.64/30.05/29.89 us (29.89 us
  median) and candidate at 24.93/24.99/24.96 us (24.96 us median, -16.49%).
  Both retained 240 registers/thread, 49.15 KiB dynamic shared memory, 1.30
  waves/SM, about 14.6-14.8% achieved occupancy, and zero spill.
- The proposed access mechanism was not supported. Both layouts reported
  1,351,968 excessive global sectors (55%). Shared excessive wavefronts became
  worse, from about 127.4k (9%) for row-major B to about 170.0k (11%) for
  column-major B.
- In the real forward-order S2 trace, QKV slowed 29.993 -> 30.614 us (+2.07%),
  GPU busy union increased 267.878 -> 275.105 ms (+2.70%), and throughput fell
  3203.028 -> 3167.817 nnEval/s (-1.10%).
- In reverse order, QKV slowed 29.520 -> 30.597 us (+3.65%) and throughput fell
  3115.454 -> 3103.961 nnEval/s (-0.37%). Summed kernel time also increased
  0.41%; busy union was effectively flat (-0.11%).

The isolated NCU timing did not predict the sustained two-stream kernel timing,
and its counters directly contradicted the conflict-reduction theory. No ABBA
or 8192-row replay was run. The option, transposed weight copy, alternate kernel
instantiation, and plumbing were removed; the accepted source was rebuilt.
