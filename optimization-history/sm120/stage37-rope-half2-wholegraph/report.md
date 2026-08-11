# H37 accepted: fused Q/K RoPE half2 I/O

## Decision

Accepted for the fixed RTX 5090 D, B13, 19x19, two-stream target. The old
`0.5%` promotion threshold was a false-negative policy: this candidate was
already positive in Stage15 and remains positive on the current post-A-reuse
mainline.

No homogeneous or synthetic mixed S2 measurement was run.

## Whole-graph performance

Both tests used symmetric `A-B-B-A-B-A-A-B` order on the real graph.

| Test | control | half2 | delta | positive adjacent pairs |
|---|---:|---:|---:|---:|
| short, 400/25 | 3877.805 | 3908.378 | `+0.788%` | 4/4 |
| long, 1000/30 | 3845.421 | 3855.728 | `+0.268%` | 3/4 |

The long adjacent deltas were `+0.249%`, `-0.376%`, `+1.168%`, and
`+0.034%`. The positive mean and 3/4 agreement pass the preregistered gate.

## Correctness and accepted state

The 8,192-row full replay is byte-identical to the post-A-reuse accepted
output, SHA256
`ed0ed80848d752bc6d64995e91f9bada55c059b5e55ac5bcccb13bf28a3e1a02`.
The target config now enables `cudaUseFusedQKRoPEHalf2Sm120 = true`; config
SHA256 is `ce579bca54cd59743bdf29d55b40f34398acdf6d233f71c8077963323d32f1a1`.

## Fresh accepted-state profile

Stage38 collected 30-forward S2 and matching S1 Nsys plus one complete
344-ordinal S1 NCU forward. The joined profile measures RoPE at `177.9 us`
work and `51.0 us` excess per stream-forward, down from the prior accepted
profile's `218.9 us` and `83.1 us` respectively.

The next highest interference buckets are linear2 (`341.2 us` excess),
attention out-projection (`337.1 us`), fused FFN (`203.5 us`), FA4
(`200.2 us`), and QKV (`149.6 us`). No next optimization is started before the
requested workspace cleanup and commits.

## Artifacts

- `short/summary.json`, SHA256
  `8baf17f7d50f89068ac06cb0ccd7f1a7255d09458b8b3d1bae326dc0d815901a`
- `long/summary.json`, SHA256
  `c2123984b575d0231d20df441f496be874838dc9c6746d74aa74bf7b07f15652`
- `accuracy/`: full replay and byte comparison
- `stage38-post-rope-half2-profile/accepted-fullgraph-ranking.{md,json}`
- `stage38-post-rope-half2-profile/nsys/accepted-s2.nsys-rep`
- `stage38-post-rope-half2-profile/ncu/accepted-s1-full-forward.ncu-rep`
