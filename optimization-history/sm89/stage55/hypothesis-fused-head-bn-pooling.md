# Stage 55 hypothesis: fuse exact head BN+SiLU with pooling

Target is RTX 4090 exact 19x19, B13, FP16. The candidate builds on the complete
five-item strict-local bundle; S2 remains the deployment target and S1 is only
phase-free attribution.

In the current bundle, policy head BN writes a full B13x361x96 FP32 tensor that
is consumed only by gPool. Value head BN writes both B13x361x192 half (needed by
ownership) and FP32 (consumed only by value-pool). The two FP32 tensors account
for about 10.3 MiB of intermediate writes plus reads per forward.

Detailed NCU from a real S2 invocation reports:

- policy BN 3.168 us + gPool 5.600 us; gPool is 0.07 waves/SM and 9.0% SM SOL;
- value BN 4.960 us + value-pool 4.320 us; pool is 0.10 waves/SM and 5.8% SM SOL;
- all four kernels have zero local/shared spills.

## Falsifiable test

1. Use the original pool geometry (64 channels x 8 xy lanes) and original
   reduction order. Each input element executes the established half FMA,
   float expf SiLU, half rounding, then float conversion before accumulation.
2. Policy writes only the B13x288 pooled output. Value writes the established
   half ownership input plus B13x576 pooled output. Remove both FP32 spatial
   intermediates and both separate pool launches.
3. Require 26-row byte identity, then three S2-source NCU samples for both fused
   kernels. Each complete fused boundary must beat its corresponding two-kernel
   boundary without spills.
4. Require positive forward/reverse S1 Nsys before one 100-forward S1 ABBA.
   Retain a qualifying local implementation regardless of S2 phase behavior.
5. Add it to the full strict-local bundle and run forward/reverse full-graph S2
   Nsys. Only stable-positive S2 proceeds to S2 ABBA and deployment.
