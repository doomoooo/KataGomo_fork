# Stage 58 hypothesis: fuse residual projections with the following RMSNorm

## Frozen target and control

- RTX 4090 SM89, exact 19x19, B13, FP16 NHWC, S2 deployment target.
- Control is commit `91f6aae`, clean binary SHA256
  `3ecb8d6766b6a3e69fcd837cf4584c2b12e9d628213f2000f5b03f10fe1e7c05`.
- The deployed config is SHA256
  `5fc1f21d8490dbac6378e22684ca76fb5b6734ba8930d82c47b1e050652714d7`;
  Stage 56 is enabled and all strict-local bundle switches are disabled.

## Full-graph evidence before implementation

The post-Stage-57 S2 trace has 4,224 C384 RMSNorm launches in 64 complete
forwards, exactly 66 per forward. Every launch is immediately preceded by a
projection that has just produced the residual it normalizes:

| Producer -> RMSNorm | Captured pairs | Pairs/forward | Average launch gap |
|---|---:|---:|---:|
| attention out-projection + residual | 2,112 | 33 | 5.941 us |
| FFN linear2 + residual | 1,408 | 22 | 3.002 us |
| outer preConv | 704 | 11 | 2.330 us |

The remaining 11 final-inner linear2 boundaries feed an outer BN and are the
separate default-off Stage 57 route. They are not part of this experiment.

Current Nsys assigns RMSNorm 23.561 ms union and 3.443 ms exclusive, or 1.33%
of busy union. The complete capture is 98.526% busy, so the CPU launch gaps
between producer and RMSNorm are also material even though they are not a
kernel family. Broad S2 NCU measures RMSNorm at 6.944 us. The three producers
all write the same `4693x384` half output shape; linear2 is 23.072 us with 162
registers, 64 KiB shared and 0.87 waves/SM, while the fixed preConv is 16.544 us
with 162 registers, 80 KiB shared and 0.87 waves/SM. These kernels already have
one-CTA-per-SM residency, so adding a second resident CTA is not a valid goal.

## Candidate mechanism

Use a full-channel `N=384` output tile so one CTA owns complete output rows.
After the GEMM epilogue forms and rounds each residual half value, reuse the
mainloop shared-memory storage to stage complete rows, reproduce the existing
warp-per-row FP32 sum-of-squares reduction, preserve the residual output, and
write the following RMS-normalized half output. This removes one RMSNorm launch
and its producer-to-consumer launch gap without introducing a global barrier or
cross-CTA atomic protocol.

The first feasibility kernel is linear2 because it has the largest K and is the
strictest register/shared-memory case. Candidate threadblock shapes are limited
to `M32xN384xK32` and `M64xN384xK32`; arbitrary instruction-level tuning is out
of scope. If neither shape improves the complete linear2+RMS boundary, the
common full-channel design is rejected before implementing outProj or preConv.

## Falsifiable gates

1. Before code changes, collect three detailed S2-origin NCU samples of the
   RMSNorm and out-projection boundary, and retain the broad-NCU linear2/preConv
   evidence. Confirm zero spill and that the measured launch-gap plus RMS cost
   is large enough to cover a plausible full-width-tile penalty.
2. Compile one linear2 prototype and reject statically if ptxas exceeds 255
   registers, spills, exceeds the SM89 dynamic-shared limit, or cannot launch
   on the exact shape.
3. On 26 rows, require finite outputs and the established all-head smoke
   envelope. Then profile three candidate launches from S2.
4. The complete `linear2 residual -> following RMSNorm` boundary must improve
   by at least 5%. A faster fused kernel that loses this complete boundary is a
   rejection; do not extend it to the other producers.
5. Only after the prototype passes, apply the same conceptual epilogue to
   outProj and preConv, then run short forward/reverse S1 and S2 full-graph
   tests. Stable positive S2 enters one 100-iteration S2 ABBA. Strict NCU plus
   positive S1 but regressed/phase-sensitive S2 is retained default-off under
   the split intrinsic/deployment policy.
6. Every retained implementation receives the 8,192-row all-head FP32 gate and
   its own commit. After a retained outcome, rerun the deployed current-best full
   S2 Nsys/broad-NCU checkpoint. A fully reverted local rejection reuses the
   latest clean checkpoint.

## Main risk

The full-channel tile reduces the number of M tiles and can increase weight
reloads or register pressure. The saved RMS launch is only about 7 us, so the
prototype is rejected immediately if the projection slows by more than that;
the theoretical byte/launch saving is not sufficient evidence by itself.

## Outcome

The hypothesis was rejected at the local NCU gate on 2026-08-06. The two
literal `N=384` prototypes (`M64/warp64x64`, six warps, and
`M32/warp32x96`, four warps) violated CUTLASS's supported warp-grid mapping and
failed the residual correctness diagnostic. A conventional correctness-valid
feasibility shape therefore padded the output tile to `M64xN512xK32` with
eight `M64xN64` warps.

The padded candidate passed the 26-row all-head smoke envelope, but its three
S2-origin NCU samples were 48.32, 47.81, and 47.78 us (median 47.81 us). The
deployed separated boundary is 23.072 us linear2 plus 4.640 us RMSNorm, or
27.712 us. The candidate is 72.52% slower than that complete boundary and far
above the 26.326 us acceptance threshold.

NCU explains the failure: the candidate performs 33% padded output MMA, launches
only 74 CTAs (0.58 waves/SM), consumes 184 registers/thread and 73.73 KiB dynamic
shared memory, and is limited to 16.67% theoretical occupancy. This is a
structural cost of making one CTA own all 384 channels, not a launch-overhead
measurement artifact. The common projection-to-RMS design is therefore not
extended to outProj/preConv. Per the frozen gates, no S1/S2 Nsys comparison,
ABBA, or 8,192-row replay is run; all candidate source is reverted.
