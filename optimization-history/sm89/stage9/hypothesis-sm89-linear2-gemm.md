# Stage 9 hypothesis: fixed-B13 FFN linear2 GEMM

Date: 2026-08-05 UTC

## Evidence and mechanism

- Frozen target: RTX 4090 SM89, exact 19x19, B13, FP16, S2.
- The accepted stage-8 trace has 77 single-batch `grid(3,37,1)` GEMMs per
  forward. Across the last 30 forwards per stream they sum to 142.856ms.
- Their duration distribution separates into attention out-projection
  (roughly 16-20us), bottleneck projection (28-32us), and FFN linear2
  (40-56us). This experiment changes only linear2: column-major
  `M=384,N=4693,K=1152`.
- cuBLAS uses an M128xN128 tile, producing 3x37=111 CTAs, fewer than the 128
  SMs on RTX 4090. A CUTLASS M64 or N64 tile would launch 222 CTAs and may
  recover the underfilled first wave at the cost of less reuse and more
  epilogue work.
- The real call is `linear2.applyAccumulate`, so the complete boundary is
  `D = A*B + C` with `beta=1`. Preliminary `beta=0` measurements are retained
  only as search evidence and are not acceptance evidence.

## Falsifiable test

1. Profile the exact cuBLAS call with NCU to verify wave/occupancy rather than
   infer it from the kernel name.
2. Sweep fixed AOT CUTLASS tensor-core tiles at the same shape and FP16
   accumulation/output. Compare output against cuBLAS and time the complete
   call boundary on one stream.
3. Integrate only if a candidate wins by at least 3% in alternating micro
   runs, then require positive B13/S2 forward and reverse ABBA plus all-head
   FP32 regression.

## Result

- Accepted tile: threadblock `M128xN128xK32`, warp `M64xN64xK32`, four
  stages, swizzle 1. The fixed boundary is FP16 `M=4693,N=384,K=1152`,
  `alpha=1,beta=1`.
- Exact-boundary random-input micro comparison was bit-exact against cuBLAS.
  The 10,000-iteration dual-stream ABBA pair median was about 32.73us versus
  36.70us for cuBLAS in the initial sweep. A repeated extended sweep measured
  32.58us for the accepted stage4/4-warp candidate.
- Smaller M/N tiles were slower. Stage4/8-warp was 33.08us, stage5 was
  33.76us, and stage6 was 34.44us in the same dual-stream extended sweep.
- Integrated 300-forward forward/reverse ABBA improved the combined median
  from 2948.330 to 3018.561 nnEval/s (+2.382%). All four paired comparisons
  were positive.
- Nsys replaced exactly 3498 cuBLAS launches with 3498 CUTLASS launches over
  the complete capture, or 33 calls per forward per stream. Last-30-forward
  dual-stream kernel union fell from 258.932ms to 252.386ms.
- Exact `beta=1` NCU: 162 registers/thread, 64KiB dynamic shared memory, zero
  local/shared spill, 111 CTAs, 8.29% achieved occupancy. The low occupancy is
  deliberate for the S2 winner; lower-stage higher-residency candidates lost
  the direct dual-stream comparison.
- The full 8192-row candidate replay is byte-identical to the accepted stage8
  replay. Therefore every output head retains stage8's FP32-reference result:
  policy top-1 99.6948%, candidate p0loss 1.59154546 versus FP32 1.59152794.
