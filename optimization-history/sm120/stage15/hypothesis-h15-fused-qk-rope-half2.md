# Hypothesis H15: fused Q/K RoPE half2 I/O

Created: 2026-08-05 (UTC), after H14 rejection and before implementation or
measurement. Baseline is the accepted per-batch fused-QK RoPE from Stage11,
with Stage12/13 optimizations enabled.

Each thread already owns one adjacent rotary pair, but the accepted kernel
uses two scalar half loads and stores for Q and again for K. H15 preserves the
`(361,batch)` grid, FP32 `sincos`, scalar FP32 rotation expressions, and half
conversion, while loading/storing each Q/K pair as half2. This changes only the
memory instruction shape and does not repeat H14's batch-sharing schedule.

Nsys must reduce direct RoPE time. Short A-B-B-A must improve a target topology
before full validation. A retained candidate requires at least 0.5% long gain
and all full-FP32 8,192-row accuracy gates. Unsupported shapes and precisions
retain the accepted scalar-I/O fused kernel.
