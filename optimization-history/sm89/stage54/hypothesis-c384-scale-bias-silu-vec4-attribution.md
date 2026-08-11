# Stage 54 hypothesis: recover the C384 affine+SiLU vec4 mechanism

Target remains RTX 4090 exact 19x19, B13, FP16, with S2 as the deployment
topology. S1 is used only for phase-free attribution.

The post-Stage53 current-best full graph launches the scalar C384 affine+SiLU
kernel 704 times across 64 timed forwards. It accounts for 5.866 ms of busy
union (2.24%) and 0.878 ms exclusive busy (0.33%). Broad S2 NCU reports 7.424
us, 4,693 CTAs, 9.17 waves/SM, and 16 registers/thread. Historical Stage35
proved that an exact-shape vec4 implementation is byte-identical and reduces
the NCU median from 6.85 to 4.32 us (-36.93%) with 1,760 CTAs, 2.29 waves/SM,
23 registers/thread, and no spills. It was deleted only because the old
workflow treated an order-conflicting S2 ABBA as an implementation rejection.

## Falsifiable test

1. Restore the same vec4 arithmetic behind a default-off exact-shape switch.
2. Require a 26-row byte-identical smoke test.
3. From real S2 invocations, collect three control and three candidate NCU
   samples. Require no spills and a candidate median below 5.0 us with lower
   CTA/wave work than control.
4. Run locked 20-forward S1 Nsys in forward/reverse order. If both are positive,
   run one 100-forward S1 ABBA. Stable positive evidence classifies the kernel
   intrinsic-accepted; neutral evidence may still retain it as strict-local.
5. Add a qualifying implementation to the complete strict-local bundle and run
   forward/reverse full-graph S2 Nsys. Only stable S2 gain may change deployment.
