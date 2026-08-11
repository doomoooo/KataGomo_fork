# Hypothesis H13: fixed-C384/C768 affine-SiLU half2

Created: 2026-08-05 (UTC), after the accepted Stage12 trace and before
implementation or measurement.

The scalar FP16 `BatchNorm + SiLU` kernel contributes 24.515 ms across 3,350
launches in the accepted S2 trace. Grouping by launch shape shows the fixed
C768 B13 path at 11.255 ms and fixed C384 B13 at 6.696 ms. C768 currently uses
three 256-thread CTAs per token row; C384 uses one 384-thread CTA.

For fixed 19x19, mask-free FP16, H13 uses half2 input/scale/bias/output access
and `__hfma2`, then applies the unchanged FP32 SiLU separately to each half
lane. C768 uses one 384-thread CTA per row instead of three CTAs; C384 uses one
192-thread CTA. Other channels, activations, masks, shapes, and precisions
retain the official kernel.

Nsys must reduce the direct affine-SiLU total. Short A-B-B-A must improve a
target topology before full validation. A retained candidate requires at least
0.5% long whole-network gain and all full-FP32 8,192-row accuracy gates.
