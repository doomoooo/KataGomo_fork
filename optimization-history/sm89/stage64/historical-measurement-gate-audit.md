# Historical audit after the Stage 64 NCU/Nsys disagreement

## New rule

Independent Nsight Compute replay durations are not additive. For a fusion,
split, reorder, or materialization change, NCU explains the mechanism and
resources; only a naturally executed Nsys/NVTX interval or CUDA-event subgraph
measurement can accept or reject the complete local boundary.

Stage 64 is the counterexample. Isolated NCU predicted fused QKV+RoPE at
30.005 us versus plain QKV plus standalone RoPE at 30.538 us, apparently a
1.78% regression. Natural S1 Nsys instead measured launch-to-completion at
28.941 -> 25.948 us (-10.34%). Raw kernel-duration sums were
28.941 -> 25.261 us (-12.71%). The profiler/resource evidence was useful, but
the replay-duration sum was not a latency model.

## Historical candidates requiring action

### Reopen: Stage 55 head BN/SiLU plus pooling fusion

The rejection compared independently replayed control kernels with separately
replayed fused kernels:

- policy: 3.168 + 5.600 = 8.768 us control versus 10.368 us fused;
- value: 4.960 + 4.320 = 9.280 us control versus 10.400 us fused.

No S1 or S2 Nsys boundary was collected. The old result still warns that the
pooling grid has only 0.07/0.10 waves per SM, but it no longer proves a local
regression. Restore behind a default-off switch and collect a short natural S1
boundary before deciding. This is the closest historical analogue to Stage 64.

### Reclassify as pending, lower priority: Stage 58 projection plus RMSNorm

The rejection compared 23.072 us linear2 plus 4.640 us RMSNorm with a separately
replayed 47.810 us fused kernel. No comparative Nsys boundary was collected.
The candidate's 33% padded output MMA, 184 registers/thread, 73.73 KiB shared
memory, and 0.58 waves/SM make a win unlikely, so its payoff/cost rank remains
low. Nevertheless, the additive-NCU result alone is no longer a conclusive
mechanism rejection.

### Reopen under strict-local accumulation: Stage 39 RMSNorm 8-warps/CTA

The candidate was byte-identical, kept 40 registers/thread and zero spill,
halved CTA count, improved achieved occupancy from about 54.7% to 58.8-60.1%,
and changed the NCU median 4.61 -> 4.58 us (-0.65%). It was reverted solely for
missing an arbitrary 3% local threshold and never received Nsys. The absolute
payoff is small, so it ranks below macro boundaries, but the new workflow should
retain it if a natural boundary/S1 test confirms no regression.

## Historical decisions that remain supported

- Stage 48 attention RMS folding had a strict isolated NCU win, but natural S2
  Nsys measured the actual attention boundary +2.20%/+3.79%; Stage 60 later
  retested the changed graph and found the natural S1 boundary slower. This was
  not an additive-NCU-only rejection.
- Stage 49 FFN RMS folding was slower in both NCU and natural Nsys.
- Stages 40-44 and 63 compared one kernel with the same kernel boundary; their
  mechanisms were flat or slower rather than rejected by cross-kernel sums.
- Stage 35's premature S2-gated revert was already repaired in Stage 54: the
  vec4 implementation was revalidated by NCU and S1 and retained default-off in
  the strict-local bundle.
- Stage 26/37 locally faster head-boundary implementations were not lost; their
  descendants are retained default-off and have already been re-bundled.

This audit is about whether the evidence justified deletion, not an assertion
that every reopened candidate will improve deployed S2 throughput.
