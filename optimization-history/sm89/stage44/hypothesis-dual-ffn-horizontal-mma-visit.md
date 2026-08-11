# Stage 44 hypothesis: dual-FFN horizontal HMMA visit

## Scope and evidence

- RTX 4090 SM89, exact FP16 19x19 B13, S2 only, 2400 MHz.
- Fresh accepted NCU median: 41.28 us; 168 registers; 49.15 KiB shared;
  no spill; No Eligible 78.28-78.57%.
- Warp State reports 8.5 cycles between instructions, including 2.8 cycles
  (33.2%) waiting for an oversubscribed execution pipe.
- CUTLASS explicitly selects vertical accumulator visitation for SM89.

The warp operator computes 16 independent accumulator fragments per K group.
Horizontal versus vertical visitation preserves the K accumulation order of
each output element but changes register access and HMMA issue order. Override
the SM89 vertical visit only in the dual-FFN translation unit; all other AOT
kernels remain unchanged.

## Gates

1. Candidate SASS/hash must differ while HMMA instruction count is unchanged.
2. 26-row S2 replay must be byte-identical.
3. Three S2 NCU samples must reduce median latency by at least 3% and improve
   scheduler/pipe behavior without spill or occupancy regression.
4. Only if NCU passes: S2 Nsys 20 in forward and reverse binary order.
5. Only if both Nsys orders pass: locked S2 ABBA, 100 timed iterations.
6. Only after performance passes: 8192-row accuracy.

Failure restores the local macro and pinned CUTLASS header. S1 is forbidden.

## Result

The target SASS hash changed while HMMA count remained 64, and the 26-row
replay was byte-identical. NCU median duration regressed from 41.28 to 41.54 us
(+0.63%); Warp Cycles remained 8.45-8.60 and No Eligible remained about
78.3-78.6%. The upstream SM89 vertical visit is preferable. The candidate was
reverted without Nsys comparison, ABBA, or 8192-row accuracy.
