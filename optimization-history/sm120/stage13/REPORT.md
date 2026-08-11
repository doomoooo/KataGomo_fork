# Stage 13: fixed-C384/C768 affine-SiLU half2

Status: accepted for S1 and S2 (2026-08-05 UTC).

The fixed 19x19 mask-free FP16 BatchNorm+SiLU path now handles C384 and C768
with half2 input, scale, bias, and output access. `__hfma2` preserves the
official per-lane half FMA; each lane then uses the unchanged FP32 SiLU and
half conversion. C768 uses one 384-thread CTA per token row instead of three
256-thread CTAs, while C384 uses one 192-thread CTA. Other shapes, channels,
activations, masks, and precisions retain the official path.

The accepted Stage12 trace grouped the scalar kernel's main B13 shapes as
C768 11.255 ms and C384 6.696 ms. In the direct Stage13 S2 trace, the official
kernel total was 23.643 ms. The candidate used 9.313 ms for C768, 6.263 ms for
C384, and 0.774 ms in remaining official fallbacks: 16.349 ms total (-30.85%)
with the same 3,350 launches. Traced whole-network throughput was 3428.350 to
3466.864 nnEval/s (+1.12%).

Short A-B-B-A reported S1 2900.123 to 2931.044 (+1.066%) and S2 3436.299 to
3474.806 nnEval/s (+1.121%). The full 8,192-row replay is byte-identical to
Stage12, with SHA256
`9ead43c9e5567242defbba4b7b45110ce2b802c39146ee8f9c23a1a4863c3d62`.

The 1,000-iteration symmetric S2 A-B-B-A/B-A-A-B result was:

| mode | values (nnEval/s) | mean | median |
|---|---|---:|---:|
| control | 3448.872 / 3415.878 / 3413.718 / 3417.585 | 3424.013 | 3416.732 |
| half2 | 3464.841 / 3461.601 / 3451.198 / 3445.870 | 3455.877 | 3456.399 |

Mean improvement is 0.931%; median improvement is 1.161%. The first symmetric
block improves 0.899% and the reversed second block improves 0.963%. Accept
and enable by default on the gated SM120 fixed-19 path.
