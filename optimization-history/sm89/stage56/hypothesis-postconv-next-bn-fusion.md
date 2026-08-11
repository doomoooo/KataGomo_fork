# Stage 56 hypothesis: cross-block postConv plus next BN+SiLU

Date: 2026-08-06 UTC

## Frozen target and prior evidence

- Target is RTX 4090, exact 19x19, B13, FP16 NHWC. Deployment remains full
  graph S2; S1 is attribution evidence only.
- The post-Stage55 full-graph S2 Nsys capture has 352 strictly adjacent
  `postConv -> C768 BN+SiLU` pairs on each server stream: 32 forwards times 11
  nested blocks. PostConv averages 20.430/23.568us, C768 BN averages
  8.060/9.181us, and the complete adjacent span averages 38.442/48.175us.
  The same-stream launch gap alone averages 9.960/15.164us.
- Broad NCU reports the current cuBLAS postConv at 20.256-20.672us,
  186 registers/thread, 0.87 waves/SM, and the accepted C768 vec8 BN at
  7.424us, 31 registers/thread, 2.29 waves/SM.

## Mechanism

Each postConv computes a half residual output that is immediately read by the
next nested block's affine+SiLU, except the last postConv whose immediate
consumer is trunk-tip affine+SiLU. A custom CUTLASS output iterator can:

1. preserve the original beta-one postConv half output in the residual buffer;
2. apply the following layer's half FMA and scalar-float `expf` to that rounded
   half fragment;
3. store the activated fragment into the existing next-preConv scratch buffer.

The next block then skips its preBN, and the last block skips trunk-tip BN.
This removes 11 launches and 11 full C768 residual reads per forward without
removing the residual output needed by the next outer skip connection.

## Falsifiable gates

1. The 26-row replay must pass the established all-head accuracy comparison;
   byte identity is preferred because the half rounding and activation order
   are preserved.
2. NCU from a real S2 invocation must show the fused postConv duration below
   the complete control boundary (fresh postConv plus C768 BN medians), with no
   spill. A slower complete boundary rejects the mechanism before Nsys.
3. If NCU passes, short S1 forward/reverse Nsys must confirm exactly 11 fewer
   C768 BN launches per forward and a smaller postConv-to-next-preConv boundary.
4. A stable S1 gain plus strict local resource reduction retains the route in
   the cumulative intrinsic bundle. Full-graph S2 alone controls deployment.
5. Long ABBA and 8,192-row replay run only after the short gates pass.

The switch is `cudaUsePostConvBNSiluSm89`; it defaults to false.
