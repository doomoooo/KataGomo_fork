# Stage 49 hypothesis: final-inner linear2 residual + outer postBN SiLU

## Frozen target and control

- GPU: RTX 5090 D, CUDA ordinal 2, held with `gpu-lock`.
- Exact shape/topology: 19x19, B13, FP16 NHWC, natural two-stream graph.
- Control: Stage 47 commit `acf588c`, including the accepted TileLang
  M128N128K32/S4 linear2-residual kernel.
- Accepted profile source: Stage 47 S2 Nsys and complete 344-ordinal S1 NCU.
- `CUDA_DEVICE_MAX_CONNECTIONS` is unset.

## Evidence

- Linear2-residual is 18.79% of Stage 47 S2 work and contributes 350.1 us per
  forward of interference excess.  Its accepted resource signature is 162
  registers/thread, 65.5 KiB dynamic shared memory, 0.65 waves/SM, 8.3%
  achieved occupancy, and 9.2% eligible cycles.
- Eleven of the 33 linear2 calls are the final inner FFN of a nested outer
  block.  Their output is immediately read by the outer C384 affine-SiLU
  (`outer.post_norm_silu`), which costs 48.1 us per forward and requires 11
  launches.
- This following operation is elementwise over the same C384 fragment.  Unlike
  RMSNorm, it needs no cross-CTA reduction.
- RTX 4090 Stage 57 validated the same dataflow: complete boundary
  `30.112 -> 24.640 us` (-18.17%) and S1 `+0.799%`.  Its natural S2 result was
  phase-sensitive and negative, so it is evidence for the mechanism, not an
  acceptance result for 5090 D.

## Single mechanism

Only for the final inner FFN in each nested outer block, extend the exact
linear2-residual epilogue to emit two outputs:

1. store the normal rounded FP16 residual back to the C384 inner trunk;
2. apply the following merged FP16 scale/bias FMA and FP32-exp SiLU to that
   rounded residual, then store the activated C384 scratch consumed by outer
   postConv.

When the specialized hook succeeds, skip the standalone outer postBN launch.
All other 22 linear2 calls and all unsupported shapes retain Stage 47.

## Backend candidates and bounded schedule

1. CuTe first: exact M4693/N384/K1152 with the target's proven M128/N128/K32
   tile, atom 4x2, AB2 and epilogue4.  This is not a warp-shape guess: K32 and
   M128/N128 come from the accepted linear2; AB2/epi4 is the minimum bounded
   realization that holds two output TMA buffers without reproducing Stage
   46's 99.3-KiB allocation.
2. If CuTe loses the complete boundary or its NCU resource signature is worse,
   keep the accepted TileLang mainloop byte-for-byte and modify only its
   epilogue.  This is a competing implementation, not a forced fallback.
3. No unrelated M/N/K/warp sweep is allowed without candidate NCU or SASS
   evidence.

## Falsifiable gates

1. Standalone output must match the control boundary within the established
   FP16 tolerance and contain no NaN/Inf.  A TileLang epilogue-only candidate
   is expected to be byte-identical.
2. Natural S1 Nsys must show exactly 11 postBN launches removed and compare the
   complete 11-boundary event span; NCU must report registers, dynamic shared
   memory, spills, eligible cycles, occupancy, and waves.
3. A candidate whose complete boundary is slower and whose resource signature
   is not strictly better is rejected before whole graph.
4. A correct, S1-positive, strictly resource-better implementation may remain
   default-off even if natural S2 is unresolved or negative.
5. Deployment acceptance requires positive natural-S2 common-wall ABBA with
   both adjacent comparisons non-negative, followed by a medium confirmation.
6. An accepted candidate requires full 8,192-row all-head accuracy, fresh S2
   Nsys, complete 344-ordinal S1 NCU, one history entry, and one commit.

## Scale of opportunity

The removed standalone affine family is only 48.1 us per forward, but the
fused epilogue also avoids 11 full C384 residual reads and launch gaps.  Based
on the 4090 boundary result and the Stage 47 forward scale, the expected whole
graph gain is roughly 0.5--0.8%.  A claim materially above 1% requires repeated
phase-order evidence rather than linear extrapolation.
