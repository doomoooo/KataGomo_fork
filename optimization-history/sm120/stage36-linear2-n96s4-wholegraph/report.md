# H36 result: linear2 M128N96/S4

## Decision

Rejected for the fixed RTX 5090 D, B13, 19x19, two-stream target. The
candidate remains default-false as an auditable resource-positive experiment;
the accepted M128N128/S4 path and target configuration are unchanged.

No homogeneous or synthetic mixed S2 measurement was run. The decision comes
only from the real current-mainline whole graph.

## Evidence

- Correctness: Stage32 direct AOT versus `cublasHgemm(beta=1)` was bit-exact
  over all 1,802,112 FP16 outputs.
- Isolated S1: about 21% faster than the accepted tile.
- SM120 resources: 148 CTAs, 162 registers/thread, 57,344 B dynamic shared
  memory, no local or stack allocation. The accepted tile uses 111 CTAs, 162
  registers/thread, and 65.54 KiB dynamic shared memory.
- Build binary SHA256:
  `b7eb263bb2f012d6f67ed4b0a084c92f06449b799efb83f9aa8e75add48f6bd8`.

## Real whole-graph S2

Order `A-B-B-A-B-A-A-B`, 400 timed iterations and 25 warmups per arm:

| Variant | samples (nnEval/s) | mean |
|---|---|---:|
| accepted M128N128/S4 | 3880.583, 3877.347, 3854.273, 3848.149 | 3865.088 |
| M128N96/S4 | 3799.854, 3820.442, 3843.461, 3846.013 | 3827.442 |

Mean delta is `-0.974%`. All four adjacent deltas are negative:
`-2.080%`, `-1.468%`, `-0.281%`, and `-0.056%`. This is a coherent
whole-graph regression, so the preregistered short gate stops the candidate
before long replay, accuracy, or a candidate profile.

The accepted mainline did not change; the post-A-reuse Stage35 full-graph
Nsys/NCU profile remains the valid priority source.
