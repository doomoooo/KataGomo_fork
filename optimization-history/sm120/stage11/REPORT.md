# Stage 11: fused learnable Q/K RoPE

Status: accepted for S1 and S2 (2026-08-05 UTC).

The fixed 19x19 FP16 MHA path now rotates Q and K in one kernel. Q and K have
the same 12 heads and D32 layout, so each thread computes the FP32 angle and
`sincos` once and applies the unchanged rotation expressions to both buffers.
This removes one launch per transformer block and half of the transcendental
work. Unsupported shapes, precisions, non-learnable RoPE, non-MHA layouts, and
dimensions retain the official path.

Short A-B-B-A reported:

| topology | control mean | fused mean | change |
|---|---:|---:|---:|
| B13/S1 | 2808.127 | 2879.842 | +2.554% |
| B13/S2 | 3307.458 | 3403.254 | +2.897% |

The full 8,192-row replay is byte-identical to the accepted Stage10 replay.
Both files have SHA256
`9ead43c9e5567242defbba4b7b45110ce2b802c39146ee8f9c23a1a4863c3d62`,
and all direct full-FP32 gates pass.

S2 Nsys reduced RoPE launches from 8,844 to 4,422 and direct RoPE kernel time
from 40.764 ms to 23.426 ms (-42.5%). The traced whole-network result was
3322.507 to 3410.079 nnEval/s (+2.64%).

The 1,000-iteration symmetric S2 A-B-B-A/B-A-A-B result was:

| mode | values (nnEval/s) | mean | median |
|---|---|---:|---:|
| control | 3309.599 / 3277.112 / 3269.440 / 3255.065 | 3277.804 | 3273.276 |
| fused | 3399.251 / 3375.456 / 3366.299 / 3356.952 | 3374.489 | 3370.877 |

Mean improvement is 2.950%; median improvement is 2.982%. The first symmetric
block improves 2.854% and the reversed second block improves 3.046%. Accept
and enable by default on the gated SM120 fixed-19 path.
