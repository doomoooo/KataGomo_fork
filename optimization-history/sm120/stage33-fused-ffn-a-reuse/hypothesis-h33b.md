# H33b: retain resource-positive fused-FFN A reuse for graph adjudication

> Workflow note: after this hypothesis was written and its mixed diagnostics
> were collected, homogeneous and synthetic mixed S2 microbenchmarks were
> removed from the project workflow entirely. Those results remain historical
> artifacts but have no decision weight. The short and long real whole-graph
> S2 measurements are the only performance gates for H33b.

## Why H33 is reopened

H33 correctly recorded a failure under its preregistered homogeneous-S2 gate,
but that gate was too strong. Two identical fused-FFN kernels do not represent
the current heterogeneous S2 graph. The candidate is bit-exact, improves S1
`37.680 -> 34.921 us` (`7.321%`), reduces registers/thread `146 -> 136`, has
zero spills, and preserves grid, threads, and shared memory. It is therefore a
resource-positive candidate under the revised workflow even though the
homogeneous pair changed `62.682 -> 62.854 us` (`0.275%` slower).

The original H33 files and rejection decision remain immutable evidence. H33b
is a separate follow-up requested after reviewing the gate semantics.

## Frozen target and single variable

- RTX 5090 D, fixed B13, 19x19, FP16/NHWC, S2.
- Current accepted Stage29 trunk+inner-L2 mainline and target configuration.
- Only the fused-FFN generated kernel changes to the already-built A-reuse
  candidate. No tile, grid, shared-memory layout, arithmetic, or other operator
  changes are allowed.

## Diagnostics and graph gate

1. Measure the candidate against the dominant real fused-FFN peer families
   identified by the current full graph: current linear2 and the current
   ordinary out-projection/library GEMM. Wide QKV may be included as the third
   diagnostic if the harness already has an exact current implementation.
2. Mixed-peer results are explanatory and do not independently accept or reject
   the candidate unless they show a reproducible crash or a large regression.
3. Integrate behind an exact B13/19x19/FP16/NHWC default-false switch, preserving
   the current kernel as fallback.
4. Run a short current-mainline S2 ABBA screen in both order directions. Advance
   if the candidate has a positive mean and at least three of four adjacent
   comparisons are positive, or if the result is within +/-0.25% and requires a
   longer run to resolve.
5. A positive/neutral short result advances to symmetric 1000-iteration
   ABBA/BAAB. Only that whole-graph result decides performance retention.
6. If retained, run the full 8,192-row all-head replay, then immediately collect
   fresh whole-graph S2 Nsys and all-344-ordinal S1 NCU before choosing another
   optimization target.

No new fused-FFN tile or schedule search is authorized in H33b.
