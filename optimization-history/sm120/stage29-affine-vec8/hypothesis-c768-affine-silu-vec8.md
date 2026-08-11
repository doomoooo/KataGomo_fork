# Hypothesis: flat vec8 C768 affine-SiLU on B13/S2

Created: 2026-08-06 UTC, before implementation. Target is fixed 19x19,
B13/S2, FP16 on RTX 5090D.

The accepted half2 path launches one 384-thread CTA per spatial row for C768,
or 4,693 CTAs per invocation. Both the 5080 history and the independent 4090
rebuild retained a C768-only flat vec8 schedule. The 4090 result reduced direct
kernel time by about 41% and whole-network S2 throughput improved by 0.95%.

Flatten the B13x361x768 tensor into aligned groups of eight half values. One
thread loads input, scale, and bias as `uint4`, performs the same four half2
FMAs and scalar FP32 SiLU calculations, and stores one `uint4`. A 256-thread
flat launch reduces the grid from 4,693 to 1,760 CTAs while preserving the
per-element arithmetic order.

This candidate applies only to unmasked C768 SiLU on exact 19x19 and B13. C384
is excluded because its lower-work schedule regressed whole-network S2 on the
4090 despite a faster isolated kernel. Require direct Nsys/NCU improvement and
a positive S2 ABBA screen before full accuracy; retained arithmetic should be
byte-identical, but the full 8,192-row comparison remains mandatory.
