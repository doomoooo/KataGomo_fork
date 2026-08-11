# Stage 45 hypothesis: fuse outer post-projection with the next C768 affine+SiLU

Date: 2026-08-06 UTC

## Frozen target and accepted baseline

- Target: RTX 5090 D, exact 19x19, fixed B13, FP16 NHWC, two natural CUDA
  server streams. Portable B1--B32 behavior is out of scope and must retain the
  official fallback.
- Source baseline: `ee4d1d85ed6493f2710d938f924578b1ec6d46ca`.
- Accepted profile: Stage 44 packed-QKV S2 graph, exactly 344 kernels per
  forward on both streams.
- The 11 outer `C384 -> C768` post-projection residual GEMMs are immediately
  followed by the next outer pre-BN+SiLU, except that the final one is followed
  by trunk-tip BN+SiLU. There are therefore exactly 11 eligible boundaries per
  forward.

## Pre-implementation evidence

The fresh accepted S1 NCU profile reports:

- post-projection residual GEMM: median 15.296 us, 154 registers/thread,
  73.728 KiB dynamic shared memory, 0.87 waves/SM;
- C768 affine+SiLU: median 6.336 us, 17 registers/thread, no dynamic shared
  memory, 6.90 waves/SM;
- isolated complete kernel time: 21.632 us per boundary before launch gaps.

The fresh accepted natural S2 Nsys trace directly measures the complete
same-stream span from post-projection start through following activation end:

| stream | GEMM mean (us) | activation mean (us) | gap mean (us) | boundary mean (us) | sum/forward (us) |
|---:|---:|---:|---:|---:|---:|
| 65 | 17.1615 | 6.9892 | 0.4244 | 24.5751 | 270.3265 |
| 82 | 17.4331 | 7.1564 | 0.4121 | 25.0016 | 275.0175 |

Thus this boundary occupies about 272.67 us of same-stream span per forward.
The standalone GEMM is already a strong cuBLAS tactic; the opportunity is not
another ordinary GEMM tile but eliminating 11 activation launches and 11 full
C768 residual reads while preserving the residual output.

Independent Kimi K3 review ranked this as the highest-value remaining 5090D
candidate and estimated +0.2--0.7% natural-S2 throughput. RTX 4090 Stage 56 is
transfer evidence for the same graph mechanism: its complete boundary fell
26.496 -> 21.120 us (-20.29%) and locked natural S2 improved +0.365%.

## Mechanism

For the exact B13/S361/C384->C768 post-projection only, launch a residual GEMM
with a dual-output epilogue:

1. add the GEMM result to the existing FP16 residual and store the rounded
   residual back to trunk;
2. apply the following layer's FP16 affine FMA and FP32 `expf` SiLU to that
   rounded fragment;
3. store the activated fragment to the existing pre-projection scratch buffer;
4. mark the following outer pre-BN (or final trunk-tip BN) as already ready.

All other shapes, batches, GPU architectures, precisions, layouts, masks, and
topologies retain the existing path. The candidate is behind a default-false
runtime switch until accepted.

## Falsifiable gates

1. Compile and exact-B13 smoke; fallback paths must still execute unchanged.
2. Small replay must pass the established all-head envelope before profiling.
3. Targeted NCU from a real invocation must show no spills and candidate kernel
   time below the complete 21.632 us S1 control boundary. Resource reductions
   and launch deletion are mechanism evidence; NCU times are not added to make
   the final deployment decision.
4. Short natural S1 Nsys must show exactly 11 fewer activation launches per
   forward and a smaller complete boundary. Then short natural S2 forward and
   reverse order must be stable enough to justify ABBA.
5. Full 8,192-row FP32-reference comparison and natural-S2 ABBA+BAAB control
   deployment. A strict local/S1 win may be retained default-off if S2 is
   neutral or phase-sensitive; an S2 win is enabled and committed as one stage.
6. Only an accepted optimization gets fresh full-graph S2 Nsys plus matching
   full-forward S1 NCU. A rejected candidate reuses Stage 44's accepted graph.

## Reopening and risks

- Stage 27 already rejected ordinary standalone TileLang out-projection tiles:
  they serialized across streams even with lower static resource use. This
  stage is reopened only because the graph boundary is larger and deletes real
  work; it must not be interpreted as reopening the standalone tile sweep.
- Main risks are a custom GEMM losing cuBLAS's S2 interleaving, epilogue
  register growth, and changed FP16 reduction order. These are explicit NCU,
  natural-S2, and accuracy gates.
