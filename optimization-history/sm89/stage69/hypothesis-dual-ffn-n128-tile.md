# Stage 69 hypothesis: dual-FFN N128 tile

## Frozen target

- RTX 4090, exact 19x19, B13, FP16 NHWC, two independent CUDA streams.
- Control revision: `bd6b8a6` with both16 FlashAttention and half2-tanh dual FFN.
- Control tile: threadblock M128xN64xK32, warp M64xN32xK32, 4 warps, stage 3.

## Evidence

The post-Stage68 natural S2 Nsys profile attributes 36.97% of busy union and 18.39% of
exclusive busy time to dual projection plus SwiGLU, making it the largest current family.
Broad NCU reports 41.632 us, 168 registers/thread, 49.152 KiB dynamic shared memory,
2.60 waves/SM, 15.3% active warps, 73.0% L2 throughput, and only 19.8% issue activity.

## Falsifiable mechanism

Change only the dual-GEMM output tile to threadblock M128xN128xK32 while retaining
warp M64xN32xK32, stage 3, FP16 accumulation, and the accepted half2-tanh epilogue.
The CTA grows from four to eight warps. Expected consequences:

- dynamic shared memory rises from about 49 KiB to about 74 KiB;
- residency changes from two 4-warp CTAs to one 8-warp CTA, preserving the theoretical
  eight resident warps/SM;
- the N dimension has roughly half as many CTAs, reducing CTA overhead and repeated A-tile
  traffic; total B work and mathematical output remain unchanged;
- waves/SM should fall from 2.60 to roughly 1.3-1.5 without reducing resident warps.

The hypothesis fails if registers exceed the launch limit, the larger shared footprint lowers
resident warps, NCU duration is not strictly lower, or a natural S1 boundary/whole-forward run
regresses. NCU durations are not added across kernels; the unchanged single-kernel boundary may
be directly compared, while whole-forward acceptance uses natural execution.

## Validation order

1. Build and 26-row exact-output smoke.
2. NCU: duration, registers, shared memory, waves, active warps, issue and L2.
3. Short natural S1 Nsys and whole-forward throughput.
4. Only if locally beneficial, short natural S2 Nsys and throughput; then the normal retained/
   deployed decision and full accuracy as required.

## First implementation result and refinement

N128 with the control's swizzle=2 is bit-identical, but NCU reports 370 CTAs and 2.89 waves/SM:
the odd count of nine logical N tiles is padded to ten by the swizzle. Since 73.73 KiB shared
memory permits only one resident CTA, the padding makes waves worse than the control's 2.60.
Median NCU duration regresses from 40.83 to 44.32 us (+8.55%) despite registers falling from
168 to 156 and achieved occupancy rising from about 15.3% to 16.65%.

The strict follow-up changes only the N128 kernel's swizzle from 2 to 1. It should launch exactly
37x9 = 333 CTAs and restore about 2.60 waves/SM. If this direct removal of the identified padding
does not make N128 strictly faster than N64, reject the entire N128 route without Nsys/long tests.
