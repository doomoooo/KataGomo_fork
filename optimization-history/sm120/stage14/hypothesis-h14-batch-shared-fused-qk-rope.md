# Hypothesis H14: B13-shared fused Q/K RoPE

Created: 2026-08-05 (UTC), before implementation or measurement, on top of
the accepted Stage13 baseline.

Stage11 fused Q and K into one launch, but its grid remains `(361, batch)` and
each batch CTA recomputes the same learnable angle and FP32 `sincos`. For fixed
B13, H14 launches one CTA per xy, computes `sincos` once per head-pair, and
unrolls the 13 batch elements inside each thread. Q/K rotation expressions and
half conversion remain unchanged.

This reduces CTAs from 4,693 to 361 per transformer block and transcendental
work by 13x, but lengthens each CTA and lowers grid parallelism. S1 and S2 must
therefore be judged independently. Nsys must confirm the grid reduction and
explain any topology effect. Short A-B-B-A must improve a target topology
before full validation. A retained topology requires at least 0.5% long gain
and all full-FP32 8,192-row accuracy gates. Non-B13 and unsupported shapes use
the accepted per-batch fused kernel.
