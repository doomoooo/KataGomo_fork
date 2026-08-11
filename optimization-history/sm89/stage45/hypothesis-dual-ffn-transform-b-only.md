# Stage 45 hypothesis: remove the second transformed-A write

## Scope and evidence

- RTX 4090 SM89, exact FP16 19x19 B13, S2 only.
- Dual-FFN remains the first-ranked full-graph hotspot.
- Accepted NCU median is 41.28 us, with 78.28-78.57% No Eligible and about
  33.2% of warp cycles attributed to execution-pipe stalls.

`DualMmaMultistage` transforms the shared A fragment with `warp_mma0`, then
calls the full transform again through `warp_mma1` solely to obtain transformed
B1. Both operators have the same A type and policy. Add a B-only transform and
reuse transformed A. This is mathematically identical.

## Gates

1. Compile and compare the exact target-function SASS. If SASS is identical,
   nvcc already eliminated the redundant A conversion and the experiment ends
   without GPU profiling.
2. If SASS differs while HMMA count is unchanged, require byte-identical 26-row
   S2 replay.
3. Then require three S2 NCU samples with at least 3% lower median duration and
   improved scheduler behavior, without spill or occupancy regression.
4. Only if NCU passes: S2 Nsys 20 in both binary orders, then locked S2 ABBA 100.
5. Only after performance passes: 8192-row accuracy.

Failure restores the pinned CUTLASS checkout. S1 is forbidden.

## Result

The candidate compiled, but the exact target-function SASS SHA256 was identical
to control. Both versions contain 64 HMMA and 97 move-like instructions. nvcc
already eliminates the repeated transformed-A assignment, so the source-level
change has no machine-code mechanism. Per the static gate, the candidate was
reverted without smoke, NCU, Nsys, ABBA, or 8192-row accuracy.
