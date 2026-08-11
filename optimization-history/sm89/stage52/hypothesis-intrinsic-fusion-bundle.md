# Stage 52 hypothesis: bundle retained intrinsic frontend/head fusions

Scope is RTX 4090 exact 19x19, B13, FP16, S2. This is a declared bundle of
three existing default-off implementations, not a new un-attributed kernel:

- Stage 27 initial global dot plus spatial add: S1 +0.118%, local NCU -43%.
- Stage 28/37 no-split wide head: S1 +0.599%, local NCU -54.7%.
- Stage 29 wide-head BN direct FP32: S1 +0.078%, local NCU -42.4%.

All three passed the 8192-row all-head accuracy gate, and the wide-head plus
direct-FP32 dependency was tested together in Stage 29. They remain disabled
in the current S2 config only because their individual S2 results were phase
sensitive or negative.

The bundle removes about six launches per forward while changing frontend and
head phase together. The hypothesis is that their combined phase movement may
be less harmful than enabling each boundary independently. Existing individual
experiments provide the ablation evidence. Use one binary for control and
candidate; Stage 51 stays disabled in both.

Acceptance requires paired 20-iteration full-graph S2 Nsys in both orders and
then one 100-iteration S2 ABBA. If S2 remains negative or order-conflicting,
the bundle stays disabled but the three intrinsic implementations remain
retained under their revised classifications.
