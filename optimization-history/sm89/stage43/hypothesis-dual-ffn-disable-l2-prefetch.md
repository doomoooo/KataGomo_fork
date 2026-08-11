# Stage 43 hypothesis: remove dual-FFN L2::128B prefetch hint

## Scope and evidence

- RTX 4090 SM89, exact FP16 19x19 B13, S2 only, 2400 MHz.
- Fresh accepted NCU: median 41.28 us, L2 throughput 72.72-73.77%, L2 hit
  99.78-99.79%, compute throughput 67.15-68.26%, zero spill.
- Fresh accepted Nsys ranks dual-FFN first: 2178 launches and 94.161 ms total.
- Accepted SASS confirms every 16-byte mainloop copy uses
  `cp.async.cg.shared.global.L2::128B`.

Because the working data is already almost entirely L2-resident, the 128-byte
prefetch hint may amplify L2 request/transfer pressure without hiding DRAM
latency. Disable CUTLASS L2 prefetch only in the dual-FFN translation unit. The
copy remains `cp.async.cg`; tile, stage, swizzle, math, and epilogue are fixed.

## Gates

1. SASS must show the target changed from `.L2::128B` to plain `cp.async.cg`.
2. 26-row S2 replay must be byte-identical.
3. Three S2 NCU samples must reduce L2 pressure and improve median duration by
   at least 3%, with no spill or occupancy regression.
4. Only if NCU passes: 20-iteration S2 Nsys in both binary orders.
5. Only if both Nsys orders pass: locked S2 ABBA, 100 timed iterations.
6. Only after performance passes: 8192-row accuracy.

Failure restores both the local define and the pinned CUTLASS header. S1 is not
an admissible measurement.

## Result

SASS confirmed that all 24 dual-mainloop copies lost the `.L2::128B` hint, and
the 26-row replay was byte-identical. NCU median duration nevertheless changed
from 41.28 to 41.34 us (+0.15%), while L2 throughput remained about 73-74%.
The request-amplification theory was not supported. The candidate failed the
NCU gate and was reverted without Nsys comparison, ABBA, or 8192-row accuracy.
