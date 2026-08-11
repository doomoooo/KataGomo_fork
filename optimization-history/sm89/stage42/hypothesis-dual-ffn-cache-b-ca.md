# Stage 42 hypothesis: dual-FFN cache B at all levels

## Scope and evidence

- RTX 4090 SM89, exact FP16 19x19 B13, S2 only, clocks locked at 2400 MHz.
- Target: accepted CUTLASS dual-GEMM linear1 + linearGate + SwiGLU.
- Fresh stage-41 control NCU: 41.38/41.25/41.28 us, 0% L1 hit,
  72.72-73.77% L2 throughput, 99.78-99.79% L2 hit, zero spill.
- Fresh stage-40 full S2 Nsys: this is the first-ranked summed-time hotspot.

Each B0/B1 weight tile is reused across the approximately 37 M tiles, whereas
stage 41 found only about 2% L1 hits for A. Change only the B cache operation
from `cp.async.cg` to `cp.async.ca`; A remains `.cg`. The theory is that B's
higher cross-CTA reuse yields enough L1 hits to lower L2 pressure without the
net latency regression seen for A.

## Gates

1. 26-row S2 replay must be byte-identical to the frozen accepted binary.
2. Three S2 NCU samples must show meaningful L1 reuse and at least 3% lower
   median duration, with no spill or occupancy regression.
3. Only after NCU passes: S2 Nsys 20 in forward/reverse binary order.
4. Only after both Nsys orders pass: locked S2 ABBA, 100 timed iterations.
5. Only after performance passes: 8192-row accuracy.

Failure restores the pristine pinned CUTLASS checkout. S1 is forbidden.

## Result

The 26-row replay was byte-identical, but B `.ca` produced 0% L1 hits in all
three samples. Median duration regressed from 41.28 to 41.50 us (+0.53%), with
no material scheduler or L2 improvement. The candidate failed the NCU gate and
was reverted without Nsys comparison, ABBA, or 8192-row accuracy.
