# Post-Stage47 GEMM-family reaudit

Scope: RTX 5090 D, exact 19x19 B13, natural S2.  Standard: prefer CuTe for new
fixed-shape fusions, use TileLang as strategy explorer and valid competitor,
and retain cuBLAS when the operation is still a plain GEMM.  Backend identity
does not override direct-boundary, resource, whole-graph, or accuracy evidence.

| family | current winner | evidence under the common standard | disposition |
|---|---|---|---|
| unchanged/plain projections | cuBLAS | historical C384->C1152: cuBLAS 16.5 us, CuTe 21.9 us, TileLang 24.6 us | closed unless a boundary is eliminated |
| packed QKV | CuTe Stage44 | direct 19.228->14.164 us, QKV-to-FA4 boundary -9.131%, S2 +0.597%, 107 regs | keep CuTe |
| dual FFN projection + SwiGLU | CuTe Stage47 | TileLang 33.435->CuTe 31.198 us; medium S2 +0.449%; 96 regs/50.176 KiB | keep CuTe; further tile-only tuning is lower priority than remaining boundaries |
| linear2 C1152->C384 + residual | TileLang | 18.79% S2 work, 350.1 us excess, 162 regs/65.5 KiB, 0.65 waves/SM, 8.3% occupancy, 9.2% eligible | highest-value open GEMM target, but reopen only with a CuTe boundary/epilogue mechanism, not another unsupported tile sweep |
| attention out projection + residual | cuBLAS beta=1 | 10.41% S2 work, 296.2 us excess; prior TileLang standalone kernel slower than exact cuBLAS control | second open target only through fusion with the following normalization/reduction boundary |
| outer C768->C384 pre projection | cuBLAS | 3.15% S2 work; prior CUTLASS2 candidate slower | low priority; only wider fusion |
| outer C384->C768 post projection + residual | cuBLAS; Stage45 code default-off | fused following affine-SiLU proved S1/resource positive but S2 -0.438%; transferred CuTe schedule used 153 regs/99.3 KiB and lost direct boundary by 12.3% | retain evidence, do not promote; reopen only from its own NCU-derived mainloop |
| single-wide FFN cuBLAS | superseded | S1 +1.035%, S2 -2.907%; true fused CuTe is structurally and empirically better | closed |
| strided-batched QKV cuBLAS | superseded | S1 +1.405%, S2 -0.548%; packed CuTe wins | closed |
| generic cuBLASLt projection | rejected | poor FP32 tactics; FP16 changed numerics and exposed replay lifetime problems | closed except for a named, fully specified epilogue tactic |
| standalone TileLang residual/outproj variants | rejected or current linear2 winner | occupancy-only predictions did not survive exact whole graph except accepted linear2 | keep as strategy evidence, not a blind search queue |

The fresh accepted graph changes the priority from Stage46.  PostConv is no
longer first: it was directly falsified.  The next justified GEMM investment is
linear2 plus a wider adjacent boundary.  The key design constraint is that its
three N tiles jointly produce one C384 row, while a following RMSNorm reduction
needs all 384 channels.  A viable candidate must either make that cross-tile
reduction explicit and cheap or change the output tile so the fusion is real;
simply rewriting the existing residual GEMM in CuTe has no evidence-backed
advantage over the accepted TileLang tactic.

