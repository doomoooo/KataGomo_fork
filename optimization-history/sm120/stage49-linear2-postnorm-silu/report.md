# Stage 49 result: final-inner linear2 residual + outer affine-SiLU

Target: RTX 5090 D, exact 19x19 B13, FP16/NHWC.  Control is the accepted
Stage 47 commit `acf588c`.  This stage tested the highest-value open GEMM
boundary identified by the Stage 48 total audit.  No candidate was integrated
into the graph, so the accepted Stage 47 full Nsys/NCU profile remains valid.

## Complete-boundary results

All timings include both the accepted linear2-residual control and the
standalone affine-SiLU it would replace.  Correctness compares both the rounded
residual and activated output.

| implementation | correctness | control (us) | fused (us) | delta | decision |
|---|---:|---:|---:|---:|---|
| CuTe M128N128K32, AB2, epi4 | failed | 23.617 | 27.355 | +15.83% | reject |
| CuTe M128N128K32, AB2, epi8 | bit-exact | 23.627 | 27.610 | +16.86% | reject |
| accepted TileLang M128N128 mainloop + epilogue | bit-exact | 23.606 | 38.505 | +63.11% | reject |
| NCU-directed TileLang M128N64K32/S3 + epilogue | bit-exact | 23.537 | 28.626 | +21.62% | reject |

The CuTe epi4 failure was caused by insufficient output staging for the
dual-output epilogue.  Epi8 restored exact output but remained slower and did
not strictly improve resources relative to the accepted TileLang control.

## NCU evidence

| candidate | replay duration (us) | regs/thread | dynamic smem | grid / block | waves/SM | theoretical / achieved occupancy | eligible warps/scheduler | no eligible | spills |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| accepted TileLang control | -- | 162 | 65.5 KiB | 111 / 128 | 0.65 | -- / 8.3% | -- | -- | 0 |
| CuTe epi8 | 32.96 | 151 | 66.56 KiB | 111 / 288 | 0.65 | 18.75% / 18.23% | 0.21 | 82.64% | 0 |
| TileLang M128N128 epilogue | 35.68 | 168 | 65.54 KiB | 111 / 128 | 0.65 | 8.33% / 8.37% | 0.11 | 88.82% | 0 |
| TileLang M128N64 epilogue | 28.61 | 128 | 36.86 KiB | 222 / 128 | 0.65 | 16.67% / 11.17% | 0.14 | 86.69% | 0 |

The M128N128 epilogue serializes 64 half2 activation pairs per thread and
worsens scheduler eligibility.  That NCU result, combined with historical
evidence that the N64 plain mainloop was nearly tied with N128, justified one
bounded N64 test rather than an unsupported tile sweep.  N64 materially lowers
register and shared-memory use, but doubles the grid and still loses the
complete boundary by 21.62%.  It therefore fails the retention rule: resource
dominance alone is insufficient when the natural single-stream boundary is a
large regression.

## Decision

Reject Stage 49.  The attempted boundary is not a missed low-cost fusion on
SM120: both a CuTe-first implementation and the accepted TileLang mainloop
family lose decisively.  No natural-S2 graph test or fresh full-graph profile
is warranted for a candidate that already fails the complete S1 boundary.
The repository was restored exactly to `acf588c`; there is no performance
commit for this rejected stage.

The GEMM audit priority now advances to attention out-projection plus its
following reduction/norm boundary.  That direction has a larger theoretical
opportunity but requires an explicit cross-CTA reduction design; a standalone
out-projection rewrite remains closed by prior cuBLAS-vs-custom evidence.
