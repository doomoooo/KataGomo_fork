# Stage 28: fixed-B13/S2 outer projection AOT search

## Decision

Rejected both conventional CUTLASS2 standalone AOT routes. Neither fixed
outer projection passed the exact private-weight S2 micro gate, so no shared
C++ source, default, whole-network benchmark, or full accuracy run was changed.

The tested production boundaries are:

- contract/pre: row-major `M=4693,N=384,K=768`, beta 0, 11 calls/forward;
- expand/post: row-major `M=4693,N=768,K=384`, beta 1 into the C768 residual,
  11 calls/forward.

This corrects the earlier `torch.addmm`-style ambiguity: each control is a
direct `cublasHgemm` in the same C++ process, and the two streams use
value-identical but pointer-distinct weights, matching the current two-model
ownership boundary.

## Profiler motivation

Stage 27 ordinal attribution over 60 forwards per stream measured:

| boundary | isolated total | S2 total | S2/S1 | interference excess |
|---|---:|---:|---:|---:|
| contract/pre | 7.936 ms | 11.614 ms | 1.464x | 3.678 ms |
| expand/post | 9.604 ms | 15.768 ms | 1.642x | 6.164 ms |

The 5080 history made this route credible: fixed expand/contract CUTLASS AOT
improved whole-network S2 by 1.198%, followed by 0.078% from an expand warp
64x32 refinement. The experiment asked whether a lower-resource fixed tile
could recreate that gain on 5090 D.

## Exact search results

The expand/post sweep searched valid 128x128 variants over warp 64x32/64x64
and 2--4 stages, plus wider/taller alternatives. The best result was
M128N128, warp 64x32, four stages:

| order | cuBLAS S2 us/stream | candidate S2 us/stream | change |
|---|---:|---:|---:|
| control then candidate | 21.014 | 20.932 | +0.389% |
| candidate then control | 20.819 | 20.812 | +0.031% |

This is noise-level and far below the preregistered 3% gate. Other standard
expand candidates tied or lost by up to about 5%. M128N192 and M192N128
nonstandard layouts were retained as correctness failures (relative L2
0.491 and 0.325), not timed nominees.

The 1000-iteration contract/pre screen initially nominated M64N128, warp
32x64, three stages at +2.70--2.77%. Four 10,000-iteration dual-stream
confirmations falsified that signal:

| order | cuBLAS S2 us/stream | candidate S2 us/stream | candidate change |
|---|---:|---:|---:|
| control then candidate | 21.748 | 21.885 | -0.63% |
| candidate then control | 21.706 | 21.914 | -0.95% |
| candidate then control | 21.846 | 21.923 | -0.35% |
| control then candidate | 21.889 | 22.203 | -1.43% |

The corresponding four S1 controls were 15.247--15.252 us and candidates
15.260--15.261 us, a small but consistent 0.058--0.088% regression. All
standard finalist comparisons were bit-exact against the cuBLAS control for
the generated deterministic input.

## Resource and overlap explanation

NCU and Nsys explain why lower static resource use did not create throughput:

| metric | cuBLAS contract | M64N128 candidate |
|---|---:|---:|
| grid / block | 148 / 128 | 222 / 128 |
| registers/thread | 164 | 118 |
| dynamic shared memory | 81.92 KiB | 36.86 KiB |
| shared-memory CTA limit/SM | 1 | 2 |
| theoretical occupancy | 8.33% | 16.67% |
| achieved occupancy | 8.32% | 10.93% |
| waves/SM | 0.87 | 0.65 |
| NCU isolated duration | 14.91 us | 16.74 us |
| Nsys dual-trace average kernel | 16.774 us | 22.276 us |

Both families exhibit substantial cross-stream interval overlap in the direct
trace: 775.912 us total for cuBLAS and 1,080.908 us for the candidate over the
profiled phases. The candidate therefore did achieve the intended
coexistence; it lost because its individual GEMM work was slower. More CTAs,
half the shared memory, and doubled theoretical occupancy were insufficient
to compensate for the less efficient CUTLASS2 kernel.

## Consequence

The 5080 conventional AOT result does not transfer to 5090 D's current cuBLAS
tactics. Do not spend another sweep on CUTLASS2 CTA/warp/stage permutations at
these same standalone boundaries. Reopen only for a materially different
kernel family (SM120-native CuTe/cuBLAS-like mainloop) or a fusion boundary
that removes additional traffic or launches. Do not retry the fused-SiLU EVT
route unchanged; it regressed the 5080 whole network by 2.057%.

Artifacts: `hypothesis-h28-outer-projection-aot.md`,
`outer_projection_cutlass_micro.cu`, `pre-search/`, `post-search/`,
`pre-finalist-confirm/`, `nsys-pre-finalist-direct.nsys-rep`,
`ncu-pre-candidate.ncu-rep`, and `ncu-pre-cublas.ncu-rep`.
