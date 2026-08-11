# Stage 57 hypothesis: final inner linear2 plus outer C384 BN+SiLU

Date: 2026-08-06 UTC

## Frozen target and full-graph evidence

- Target is RTX 4090, exact 19x19, B13, FP16 NHWC, deployed Stage56 S2.
- The post-Stage56 full-graph Nsys capture contains exactly 352 strictly
  adjacent `linear2+residual -> C384 BN+SiLU` pairs on each server stream:
  32 forwards times 11 nested blocks. Thus every nested block's last inner
  block is an FFN and feeds the outer postBN directly.
- The two streams report complete adjacent spans of 44.934us and 36.379us,
  including same-stream launch gaps of 5.169us and 3.391us.
- Broad S2 NCU reports the accepted fixed linear2 at 23.104us and the C384 BN
  at 7.008us, giving a 30.112us isolated control boundary.

## Mechanism

Generalize the Stage56 CUTLASS residual-plus-activation output iterator from a
fixed C768 auxiliary output to a compile-time channel count. The final inner
FFN's linear2 epilogue will preserve its original rounded half residual and
also apply the outer postBN's half FMA plus scalar-float `expf` to that fragment,
writing the activated C384 tensor into the existing postConv input buffer.
The nested block skips postBN only when this exact fused launch succeeds.

This removes all 11 outer C384 BN launches and all 11 full residual reads per
forward while leaving the other 22 inner linear2 launches unchanged.

## Falsifiable gates

1. Exact 26-row smoke must keep policy top-1 identical and pass the established
   all-head numerical envelope.
2. NCU from a real S2 invocation must put the fused kernel below the complete
   30.112us control boundary with no spill. Otherwise revert before Nsys.
3. Short S1 and S2 forward/reverse Nsys must confirm exactly 11 fewer C384 BN
   launches per forward and a smaller complete boundary.
4. Stable positive S2 proceeds to one locked 100-iteration ABBA and 8,192-row
   FP32 regression. A strict NCU plus stable S1-only win may be retained
   default-off under the split intrinsic/deployment policy.
5. If retained, this optimization gets one dedicated git commit.

The switch is `cudaUseLinear2PostBNSiluSm89`; it defaults to false.
