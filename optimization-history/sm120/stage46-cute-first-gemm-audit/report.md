# Stage 46 CuTe-first audit of fixed-B13 GEMM work

Scope: RTX 5090 D, exact 19x19, B13, FP16/NHWC, natural S2.  The
accepted control is Stage44 at source revision
`ee4d1d85ed6493f2710d938f924578b1ec6d46ca`; Stage45 at `67f034d` is a
correct default-off CUTLASS2 postConv fusion and does not change the control.

## Policy

- Keep `cublasHgemm`/cuBLASLt as the plain-GEMM baseline and fallback.
- Start new SM120 fixed-shape fusion implementations in CuTe DSL.
- Treat TileLang as both a rapid schedule explorer and a valid competing
  implementation.  Its tile/pipeline result is hypothesis evidence, not a
  mechanically portable CuTe configuration.
- Do not rewrite an accepted TileLang kernel merely to unify the stack.  A
  CuTe replacement must win its directly measured boundary or expose a new
  fusion boundary, then survive the natural whole graph.
- CUTLASS2 is retained only for reference/prototyping.  No new warp/tile sweep
  is justified without NCU/SASS evidence.

## Complete GEMM-family disposition

| family | current winner/status | evidence | CuTe-first disposition |
|---|---|---|---|
| ordinary projections | cuBLAS | historical plain C384->C1152: cuBLAS 16.5 us, CuTe 21.9 us, TileLang 24.6 us | closed for unchanged plain replacement; cuBLAS remains baseline |
| packed QKV C384->C1152 | CuTe AOT, accepted Stage44 | 19.228->14.164 us/block; composed boundary -9.131%; S2 +0.597%; 107 regs, zero spills | already conforms; keep |
| fused FFN dual projection + SwiGLU | TileLang AOT, accepted Stage20/33 | Stage20 S2 +2.586%; A reuse later +0.910%; Stage44 family 63.240 ms S2 work/trace, 136 regs, 32 KiB smem | credible CuTe production candidate, but it optimizes an existing fused winner rather than deleting another boundary |
| linear2 C1152->C384 + residual | TileLang AOT, accepted | Stage44 family 39.445 ms S2 work/trace; 162 regs, 65.5 KiB smem; strong interference | do not repeat standalone tile sweeps; reopen as CuTe only with a new epilogue/fusion mechanism |
| attention out-projection C384->C384 + residual | cuBLAS beta=1 | exact direct S1 8.137 us; best corrected TileLang finalist 9.347 us and S2 63.3% slower | closed for standalone replacement; reopen only with a wider fusion |
| outer pre C768->C384 | cuBLAS | conventional CUTLASS2 candidate was consistently slower despite lower resources | closed unchanged; only wider fusion can reopen |
| outer post C384->C768 + residual | cuBLAS; Stage45 dual-output CUTLASS2 default-off | Stage45 deletes following affine launch/read, lowers postConv resources to 108 regs/50.176 KiB, S1 +0.181%, but S2 -0.438% | highest-confidence CuTe rewrite because the fusion boundary is proven and legacy mainloop is the remaining weakness |
| single-wide FFN cuBLAS | S1-only, superseded in S2 | S1 +1.035%, S2 -2.907%; accepted fused TileLang is structurally stronger | no reopen |
| strided-batched QKV cuBLAS | S1-only, superseded by packed CuTe | S1 +1.405%, S2 -0.548% | no reopen |
| cuBLASLt generic projection | rejected | FP32 chose poor tactics; FP16 changed numerics and had replay lifetime failure | no generic reopen; only a named epilogue/tactic with a corrected lifetime contract |
| standalone TileLang residual/outproj variants | rejected or superseded | exact cuBLAS controls and current whole graph falsified the occupancy-only mechanism | use schedules as historical evidence, not another search queue |

## Priority after audit

1. CuTe postConv + residual + following affine-SiLU dual-output kernel:
   attempted in Stage46 and rejected at the direct-boundary/resource gate.
2. CuTe fused-FFN reproduction using the accepted TileLang dataflow as the
   strategy source, only if Stage46 or a fresh accepted profile still ranks it
   highest.
3. CuTe linear2 + outer C384 affine-SiLU, not a plain linear2 rewrite.
4. Keep attention out-projection on cuBLAS until a concrete adjacent boundary
   can be eliminated.

This ordering uses the Stage44 full S2 Nsys and matching 344-ordinal S1 NCU.
Rejected Stage45 did not change the accepted graph, so no new full profile is
required before selecting Stage46.

## Stage46 disposition

The exact packed-QKV CuTe schedule did not transfer to the postConv
dual-output epilogue.  Natural boundary time regressed 12.3%; v1 used 153
registers and 99.33 KiB dynamic shared memory with only 0.17 eligible
warps/scheduler.  An NCU-directed fragment live-range ablation made allocation
worse at 167 registers and 0.14 eligible warps/scheduler.  The candidate was
removed without a whole-graph run.  The next open GEMM-family item is the
already-fused FFN, where the accepted TileLang schedule provides boundary-local
strategy evidence rather than a borrowed neighboring geometry.
