# Stage 46 result: CuTe postConv dual-output fusion rejected

Target: RTX 5090 D, exact 19x19 B13, FP16/NHWC.  The accepted graph remains
Stage44.  Stage45's CUTLASS2 implementation remains correct and default-off.

## Candidate and correctness smoke

The candidate transferred exactly one evidence-backed schedule from the
accepted packed-QKV neighbor: CuTe DSL SM120a, M128/N128/K64, atom layout
4x2x1 (eight MMA warps plus one DMA warp), FP16 accumulation, and a persistent
170-cluster grid.  Its epilogue stores the rounded FP16 residual and also emits
the following C768 affine-SiLU output, deleting one launch and one residual
reread at each eligible boundary.

The 26-row replay completed without a crash or NaN.  Versus the current-binary
Stage44 control, policy top-1 and optimistic top-1 were both 100%, policy
probability RMSE was 9.94e-5, value-outcome RMSE 0.001488, all-six score RMSE
0.007652, and ownership-sigmoid RMSE 0.000368.  Non-byte-identical output is
expected from the changed FP16 MMA accumulation order.

## Natural boundary and NCU evidence

The first implementation replaced all 11 target boundaries without adding a
kernel, copy, or synchronization boundary.  Natural S1 Nsys measured 27.607 us
per fused kernel.  The corresponding original postConv plus affine boundary
was about 24.575 us, so the CuTe candidate was 12.3% slower before any
whole-graph test.  It was also slower than Stage45's 21--22 us CUTLASS2 kernel.

Targeted NCU reported:

| metric | CuTe v1 | Stage45 CUTLASS2 reference |
|---|---:|---:|
| registers/thread | 153 | 108 |
| dynamic shared memory/block | 99.33 KiB | 49.152 KiB |
| total shared memory/block | 100.35 KiB | 50.176 KiB |
| eligible warps/scheduler | 0.17 | about 0.341 |
| no eligible cycles | 85.48% | lower |
| achieved occupancy | 17.23% | higher |
| spills | 0 | 0 |

NCU additionally reported only 8 of 32 bytes utilized per global-load sector
on average and substantial L1TEX scoreboard stalls.  The limiting mechanism is
therefore not an unsupported guess about warp shape: this epilogue makes the
transferred persistent schedule both register/shared-memory limited and
latency limited.

One evidence-directed v2 change removed a full simultaneously-live activated
fragment by reusing residual-fragment storage.  No tile, atom, pipeline, or
grid parameter changed.  The compiler nevertheless increased allocation from
153 to 167 registers/thread; eligible warps fell from 0.17 to 0.14, no-eligible
cycles rose to 88.23%, and NCU replay duration rose from 39.55 to 52.06 us.
This ablation was strictly worse and falsifies the proposed live-range fix.

## Decision

Rejected before S1/S2 whole-graph timing.  The direct natural boundary is
already slower and NCU resources are strictly worse, so neither the corrected
default-off retention rule nor the S2 acceptance rule applies.  No full Nsys
or 344-ordinal NCU refresh is needed because the accepted graph did not change.
The source integration was removed and the working tree restored.

The lesson is narrower than "CuTe is worse": an accepted CuTe schedule is not
portable across materially different epilogues.  Future CuTe candidates must
start from the boundary's own NCU bottleneck and use TileLang only as strategy
evidence, not copy either backend's geometry without proof.
