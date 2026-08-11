# Hypothesis H11: fused learnable Q/K RoPE

Created: 2026-08-05 (UTC), before implementation or measurement. Primary
target is RTX 5090D, fixed 19x19, B13/S2 and B13/S1, FP16, on top of the
accepted exact-tree RMSNorm baseline.

The current attention block launches the same learnable-RoPE kernel once for Q
and once for K. This model is MHA: Q and K both have 12 heads of dimension 32,
so corresponding Q/K elements use identical per-head frequencies and angles.
A single kernel can compute each FP32 `sincos` once and apply it to both buffers,
removing one launch per block and half of the transcendental work. The per-Q
and per-K rotation expressions and conversion order remain unchanged.

Nsys must show two RoPE launches per block becoming one and a lower direct
RoPE total. Short A-B-B-A must improve at least one target topology before full
validation. A retained topology requires at least 0.5% long whole-network gain
and all Stage-3 full-FP32 8,192-row accuracy gates. Non-FP16, non-learnable,
non-MHA, masked/non-19x19, or unsupported dimensions retain the official two
kernel path.
