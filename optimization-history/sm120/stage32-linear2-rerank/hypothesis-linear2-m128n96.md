# H32: current full-graph rerank of B13 linear2

## Frozen target

- RTX 5090 D, exact 19x19, B13, FP16 NHWC, two independent server streams.
- Current accepted mainline includes C768 trunk and C384 inner persisting-L2.
- The candidate changes only the 33 repeated `C1152 -> C384 + residual`
  linear2 boundaries. Mask, head, frontend, phase offset, and other batch sizes
  are out of scope.

## Whole-graph evidence

The post-inner-L2 full-forward rerank combines a 30-forward-per-stream S2 Nsys
trace with a matching 344-ordinal S1 NCU capture.

- linear2 contributes `1077.0 us` of summed S2 kernel duration per stream
  forward, `18.88%` of all summed work;
- its `395.4 us` concurrency excess is the largest logical-operator excess in
  the graph, and its aggregate S2/S1 ratio is `1.580x`;
- NCU reports 162 registers/thread, 65.5 KiB dynamic shared memory, one
  shared-memory-limited CTA/SM, only `0.65` waves/SM, `8.3%` achieved
  occupancy, `9.2%` eligible cycles, and `4.98` wait-stall units per issue.

The accepted M128xN128 tile launches 111 CTAs. The rejected Stage26 M256xN64
family retained a similar 114-CTA footprint and reduced whole S2 throughput by
`2.305%`, so that geometry must not be replayed.

## Single hypothesis

An M128xN96 tile raises the fixed-shape grid from 111 to 148 CTAs while reducing
the accumulator and shared-weight footprint per CTA. This may fill more of the
170-SM device and expose more eligible warps without returning to the 222-CTA
M128xN64 grid that previously displaced peer work.

Only this tile family is in scope. Screen K32 stages 2/3/4 and K64 stage 2 with
128 threads. Do not expand into a general tile search.

## Falsifiable gates

1. Exact C++ boundary correctness must match the accepted FP16 accumulation
   tolerance and use the actual private-per-stream weight ownership.
2. A candidate must not regress isolated S1 linear2 by more than 5%, and must
   improve either homogeneous S2 linear2 or the dominant mixed-peer cases
   (linear2 with fused FFN and wide QKV).
3. NCU must show the predicted 148-CTA grid and a material improvement in at
   least two of registers, shared memory, eligible cycles, or achieved
   occupancy. A timing change without the mechanism is no signal.
4. Only the best local candidate may be integrated behind an exact-shape
   default-false switch. It must improve short whole-network B13/S2 ABBA in
   both forward and reverse order by at least 0.5% before long measurement.
5. An accepted arithmetic candidate requires the 8,192-row all-head accuracy
   suite. After acceptance, rerun the full S2 Nsys and all-344-ordinal S1 NCU
   profiles before choosing any following target.

## Stop and reopen conditions

Reject this family if all candidates miss the local/mixed-peer gate or if the
whole-network result fails either run order. Reopen linear2 only for a different
mainloop or fusion boundary supported by a fresh whole-graph rerank, not for
additional nearby TileLang tiles.
