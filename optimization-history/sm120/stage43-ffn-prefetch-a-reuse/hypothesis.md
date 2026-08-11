# H43: Preserve early linear prefetch while reusing FFN A fragments

## Frozen protocol

- RTX 5090 D, fixed B13/19x19 fused FFN.
- Preserve the original control's early next-linear `cp.async` position while
  removing duplicate A `ldmatrix` loads.
- The main loop retains both K-subtile A fragments across the prefetch window;
  the two tail phases use the already accepted immediate A-reuse schedule.
- Grid, CTA size, shared memory, MMA order, epilogue and output layout remain
  unchanged.
- Compile/resource feasibility is first: no spills and no more than the
  accepted 136 registers. Only then run bit-exact boundary, S1+NCU, and direct
  natural whole-graph S2. No local S2 proxies.

## Evidence and prediction

Accepted A-reuse removed half the non-transposed A loads and improved the real
graph by 0.910%, but delayed the next-linear copy until after gate MMA. Restoring
the old producer lead could recover roughly 0.5-1.5% whole-graph throughput if
the A fragments can remain live without losing the 136-register footprint.

## Stop rule

Any register increase above 136, spill, boundary mismatch, or S1 regression
closes this exact implementation before whole-graph testing.
