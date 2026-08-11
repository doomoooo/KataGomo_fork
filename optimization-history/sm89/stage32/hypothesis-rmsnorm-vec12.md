# Stage 32 hypothesis: aligned vec12 C384 RMSNorm for B13 S2

## Frozen target

- GPU: RTX 4090 (GPU 0), SM clock requested at 2400 MHz.
- Model/shape: `b11c768h12nbt3tflrs-fson-silu`, FP16, batch 13, 19x19.
- Production topology and every full-graph gate: two NN servers / two CUDA streams (S2).
- Control binary SHA256: `d39452745eb4ba9aa897dc01badab9830852740980d23e670c032a82d8975317`.
- Control path: accepted four-warps-per-CTA RMSNorm, one row per warp, twelve scalar half
  elements per lane.
- Candidate scope: exact B13 / 19x19 / C384 with `mask == NULL`; all other shapes and
  mask semantics retain the accepted control path.

## Pre-implementation evidence

The clean Stage 31 S2 Nsys controls contain 4,356 RMSNorm calls over 66 measured
forwards, or 66 calls per forward. Mean duration is 6.068 us in forward order and
5.987 us in reverse order, for 400.5 us and 395.1 us of summed kernel duration per
forward respectively.

Three NCU samples were collected by running the S2 benchmark and filtering only
`sm89RMSNormNHWCHalfKernel`. They took 4.70, 4.54, and 4.61 us. Each launch uses
1,174 CTAs, 128 threads, 40 registers per thread, no dynamic shared memory, no local
spills, and 0.76 waves/SM. L2 hit rate is 98.38-98.56%, but L1/TEX throughput is
63.62-64.08% and L2 throughput is 47.00-48.67%. Only about 33.3% of scheduler
cycles have an eligible warp. Long-scoreboard dependencies consume 8.3-8.5 of
roughly 20 cycles per issued instruction (41.3-42.6%). NCU warned that its replay
did not recognize fixed clocks; absolute NCU duration is therefore not mixed with
Nsys duration, and only same-protocol control/candidate NCU comparisons are valid.

The cross-GPU optimization history records an accepted C384 four-warp RMSNorm with
aligned `uint4+uint2` loads, but the current 4090 path has never tested that memory
mapping. The older full-graph number is not used as a predicted gain because its
baseline also differed in other ways.

## Falsifiable mechanism

Keep the accepted 128-thread CTA, four rows per CTA, one warp per row, FP32 sum,
warp-shuffle reduction, `rsqrtf`, and FP16 output conversion. Change only lane-to-
channel ownership and memory instruction width:

- lane `l` loads channels `[8*l, 8*l+8)` with one aligned `uint4` transaction;
- the same lane loads channels `[256+4*l, 256+4*l+4)` with one aligned `uint2`;
- gamma, beta, and output use the same aligned mapping.

Each lane still processes exactly 12 values, while source-level input/gamma/beta
loads and output stores fall from 48 scalar half operations to eight vector
operations. The row stride is 768 bytes, so every row base, `uint4` group, and
`uint2` group is naturally aligned. The total bytes and CTA geometry do not change.

## Expected profiler changes

- Candidate NCU duration must improve by at least 8% against the 4.62 us control
  median under the same S2-filtered protocol.
- Long-scoreboard cycles per issued instruction and/or no-eligible scheduler cycles
  must decrease. A duration change without the predicted instruction-dependency
  signal is treated as unexplained noise.
- Registers may rise, but local spilling must remain zero and theoretical occupancy
  must remain 100%.
- S2 Nsys mean RMSNorm duration and whole-forward throughput must improve in both
  forward and reverse run orders. A single-order gain is rejected as phase noise.

## Risks and gates

- The lane mapping changes the FP32 summation tree. Run a 26-row finite/error smoke
  immediately after build; full 8,192-row comparison against the frozen FP32
  reference is mandatory only after performance gates pass.
- Wider instructions could increase transaction replay or register dependency even
  with fewer instructions. Reject if the NCU duration threshold or mechanism fails.
- A faster isolated kernel can worsen two-stream overlap. Therefore no S1 throughput
  run is performed, and S2 full-graph behavior is the acceptance authority.
- Use S2 Nsys with 20 timed forwards in both A->B and B->A order. Only if both are
  positive, run one locked 100-iteration S2 ABBA confirmation, then full accuracy.

## Result

Rejected and reverted. The candidate kept 40 registers/thread, 100% theoretical
occupancy, and zero local spills, but the predicted dependency reduction moved in
the wrong direction. Same-protocol NCU median duration changed from 4.61 to 5.38 us
(+16.7%). Long-scoreboard stall rose from 8.3-8.5 to 14.3-14.6 cycles per issued
instruction, no-eligible scheduler cycles rose from about 66.6% to 80.3-80.7%, and
achieved occupancy fell from about 54.7% to 52.2-52.8%.

The short S2 Nsys full-graph check agreed in both orders. In A->B order, RMSNorm
mean duration changed 5.489 -> 6.419 us (+16.94%) and throughput changed
3207.440 -> 3136.197 nnEval/s (-2.221%). In B->A order, RMSNorm changed
6.084 -> 6.353 us (+4.42%) and throughput changed 3234.771 -> 3154.948 nnEval/s
(-2.468%). The wider operations reduced source instruction count but lengthened
coarse load-use dependency chains enough to reduce latency hiding.

The 26-row S2 smoke completed with finite outputs and 100% policy/optimistic-policy
top-1 agreement against the control. Because both profiler and full-graph gates
failed, no ABBA or 8,192-row FP32 comparison was run. The runtime switch and kernel
were removed, the control source was rebuilt, and a two-iteration S2 smoke passed.
