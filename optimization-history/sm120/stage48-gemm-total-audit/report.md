# Stage 48: fixed-B13 GEMM total audit under the CuTe-first standard

Scope: RTX 5090 D, exact 19x19, batch 13, FP16/NHWC, natural S2.  The
accepted control is Stage 47 at commit `acf588c`.  The accepted profile is
`stage47-cute-fused-ffn/final-profile`: S2 Nsys plus a complete 344-ordinal
S1 NCU replay.

## Decision standard

1. Use cuBLAS as the exact plain-GEMM baseline and keep it when no boundary is
   removed.
2. Prefer CuTe DSL for a new fixed-SM120 fusion implementation.  CuTe does not
   win by identity: it must beat the complete natural boundary and the graph.
3. Use TileLang as a schedule explorer and a valid runtime competitor.  Its
   successful tile/pipeline is local evidence, not a CuTe configuration to
   copy mechanically.
4. Infer a tactic only from the target kernel's NCU/SASS evidence.  Static
   occupancy or a neighboring kernel's geometry is not sufficient.
5. A correct candidate with a clear S1 win and strictly improved resource
   signature may be retained default-off even without an S2 win.  Natural S2
   remains the deployment-default gate.
6. Rejected candidates do not trigger a new full graph profile.  Every accepted
   stage requires full accuracy, fresh S2 Nsys, complete 344-ordinal S1 NCU,
   and one commit.

## Complete historical disposition

| operation family | attempts reviewed | current conclusion | status under this audit |
|---|---|---|---|
| Generic/plain `C384 -> C1152` projection | legacy `cublasHgemm`; CuTe dense probe; TileLang dense probe | cuBLAS `16.5 us`, CuTe `21.9 us`, TileLang `24.6 us`; no boundary removed | **confirmed closed** for a plain rewrite |
| FFN linear/gate projection | two cuBLAS calls; strided-batched; single-wide cuBLAS; TileLang fused FFN; A-reuse/pipeline/swizzle variants; CuTe paired projection | Stage 47 CuTe is the winner: `33.435 -> 31.198 us` versus accepted TileLang and S2 `+0.449%`; 96 regs, 50.176 KiB | **accepted CuTe**; tile-only work is now lower priority |
| Q/K/V projections | three cuBLAS calls; strided-batched cuBLAS; TileLang wide planar; lower-smem schedule; CuTe packed QKV | Stage 44 CuTe is the winner: `19.228 -> 14.164 us`, QKV-to-FA4 `-9.131%`, S2 `+0.597%`; 107 regs | **accepted CuTe**; old strided/wide variants superseded |
| FFN linear2 + residual, `C1152 -> C384` | cuBLAS beta=1; TileLang M128N128/S4; M256N64 and M128N96/M96N96 reranks | accepted TileLang remains the best tested standalone path; Stage 47 profile: 18.79% work, 350.1 us/fwd excess, 162 regs, 65.5 KiB, 0.65 waves | **open only through a wider epilogue/boundary**, not another unsupported tile sweep |
| Attention out projection + residual, `C384 -> C384` | cuBLAS beta=1; TileLang/CUTLASS standalone searches | exact private-weight cuBLAS is `8.137 us` S1 and `11.431 us` S2/stream; best corrected TileLang finalist is `9.347 us` and `18.663 us` | **confirmed closed** standalone; open only through fusion |
| Outer pre/contract, `C768 -> C384` | cuBLAS; CUTLASS2 fixed-tile search | lower-resource candidate did co-reside but performed inferior GEMM work; long confirmation was consistently negative | **confirmed closed** standalone; wider fusion only |
| Outer post/expand + residual, `C384 -> C768` | cuBLAS beta=1; CUTLASS2 fixed tile; dual-output affine-SiLU; transferred CuTe dual-output | plain CUTLASS2 tied/lost; Stage 45 dual-output is S1/resource positive but S2 `-0.438%`; Stage 46 CuTe transfer lost boundary 12.3% at 153 regs/99.3 KiB | **retain Stage 45 default-off**; Stage 46 implementation rejected, fusion direction remains valid |
| Generic cuBLASLt routing | FP32 and FP16 routes; later cached fixed-B13 S1 retest | FP32 selected poor kernels; fixed-B13 FP16 retest measured control about `3142.76` versus candidate `3139.53 nn/s` | **confirmed closed** generically; reopen only for a named epilogue/tactic |
| Ordinary-weight sharing | exact private/shared pointer micro and cross-platform history | matrices already fit comfortably in 96 MiB L2; no compelling exact-5090D integration signal, and 4090 whole graph regressed | **closed as a GEMM backend route**; accepted persisting-L2 windows are the stronger cache mechanism |
| Small frontend/head matmuls | existing cuBLAS/cudnn paths; audit-only fusion notes | each contributes at most 0.21% of Stage 47 S2 work; combined value is below the major trunk boundaries | **defer** until larger trunk GEMM boundaries are exhausted |

Convolutions implemented by cuDNN GEMM-like kernels are not treated as plain
GEMM backend candidates here.  They remain separate convolution/frontend
experiments because their plan space and dataflow contract differ.

## Re-audit of historical rejections

### Old workflow false negative already corrected

Stage 33 FFN A-fragment reuse was bit-exact, improved S1 by 7.321%, reduced
registers `146 -> 136`, and had no spills, but the old homogeneous-S2 proxy
rejected it on a 0.275% pair regression.  The revised workflow sent it to the
real graph, where it gained 0.910% and was accepted.  Stage 47 later superseded
that runtime kernel.  This is the concrete lesson that motivated the new
retention and real-graph policy.

### Not false negatives

- Stage 32 M128N96 linear2 was not strict resource dominance: registers stayed
  at 162, the grid grew `111 -> 148`, and lower shared memory did not change the
  one-CTA residency or permit co-residency with the then-wide-QKV kernel.  Its
  later current-graph test was `3865.088 -> 3827.442 nn/s` (`-0.974%`) with all
  four adjacent comparisons negative.
- The old single-wide FFN and strided-batched QKV paths had valid S1 wins, but
  true fused TileLang and then CuTe implementations removed more boundaries and
  superseded them.  Retaining the older paths cannot improve the current S2
  graph.
- Conventional standalone out-projection and outer-projection searches did
  reduce static resource use in places, but exact cuBLAS controls remained
  faster in both useful kernel work and the target S2 boundary.  These results
  reject the implementation class, not the possibility of a wider fusion.
- Stage 46 does not prove that CuTe is unsuitable for dual-output epilogues.  It
  proves only that copying packed-QKV's K64 persistent schedule into a larger
  dual-output epilogue is invalid.  The 153-register/99.3-KiB signature and
  12.3% boundary loss are implementation-specific evidence.

## Priority from the accepted Stage 47 graph

| priority | opportunity | evidence | estimated graph-scale opportunity |
|---:|---|---|---:|
| 1 | final-inner linear2 residual + following outer C384 affine-SiLU | linear2 is 18.79% of work and 350.1 us/fwd excess; the exact boundary occurs 11 times; following affine is 48.1 us/fwd | roughly 0.5--0.8% if the mainloop is preserved |
| 2 | attention out projection + following reduction/norm boundary | 10.41% work, 296.2 us/fwd excess, 2.048x S2/S1; standalone rewrite conclusively loses to cuBLAS | potentially larger, but requires a real cross-tile reduction design |
| 3 | outer post residual + following affine-SiLU | fusion is already proven S1/resource positive; only 3.88% work plus 0.87% affine work | likely below 0.5%; retain while seeking a boundary-local mainloop |
| 4 | outer pre projection with preceding/following boundary | 3.15% work; standalone routes lose | low until a concrete fusion removes traffic |

The first candidate is deliberately not a plain CuTe rewrite.  Only the last
inner FFN of each of 11 nested outer blocks feeds the outer C384 affine-SiLU.
A fused linear2 residual epilogue can store the normal rounded residual and
also emit the activated C384 buffer, deleting 11 launches and 11 complete
residual rereads without a cross-CTA reduction.  This mechanism transferred
successfully at the boundary level on RTX 4090 Stage 57.  It is cheaper and
better supported than trying to fuse attention out projection with RMSNorm,
which needs an all-384-channel reduction across the three current N tiles.

## Stage 48 conclusion

No accepted GEMM optimization is assigned to the wrong backend merely because
CuTe is newer.  The current mix is evidence-optimal: CuTe for packed QKV and
fused FFN, TileLang for linear2 residual, and cuBLAS for unchanged projections.
The only historical GEMM false negative has already been recovered.  The next
stage should implement and compare the final-inner linear2-to-affine-SiLU
boundary, starting from the proven M128N128K32 linear2 schedule and using NCU
to decide whether CuTe can realize it without the Stage 46 resource explosion.

## Post-audit disposition

Stage 49 tested that boundary with CuTe epi4/epi8 and TileLang M128N128/N64
implementations.  Every correct candidate lost the complete S1 boundary by at
least 16.86%; the N64 resource-reduced variant still lost by 21.62%.  The
direction is therefore closed under the revised standard.  Priority advances
to attention out-projection plus the following reduction/norm boundary; its
standalone GEMM rewrite remains closed.
