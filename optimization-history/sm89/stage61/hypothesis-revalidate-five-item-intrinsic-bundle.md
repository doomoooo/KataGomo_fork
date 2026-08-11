# Stage 61 hypothesis: revalidate the five-item intrinsic fusion bundle after both16

## Frozen target and control

- GPU/backend: RTX 4090 SM89, exact 19x19 B13 FP16 NHWC, two independent NN
  server streams.
- Control: clean commit `7d299d0`, deployed Stage59 config with FlashAttention
  both16 enabled.
- Candidate changes only five already-retained runtime switches together:
  `cudaUseInitialGlobalMatMulAdd`, `cudaUseWideHeadProjection`,
  `cudaUseHeadBNHalfToFloat`, `cudaUseFusedValueTerminalSm89`, and
  `cudaUseScaleBiasSiluVec4C384Sm89`.
- No source, tile, precision, stream, phase-control, or batch change is in scope.

## Existing mechanism evidence

Each component already passed its local mechanism gate and remains in source
default-off:

- initial global matmul+broadcast-add boundary: about -43%;
- no-split wide head projection: -54.7%;
- wide-head BN direct FP32 output boundary: -42.39%;
- combined value/score terminal boundary: about -49%;
- C384 affine+SiLU vec4: 6.816 -> 4.224 us (-38.03%), with 4693 -> 1760 CTA,
  16 -> 23 registers/thread, higher achieved occupancy, and zero spill.

Together they remove eight launches per model forward. The bundle passed its
S1 attribution and accuracy gates, but the pre-Stage59 S2 graph regressed: the
five-item Stage54 Nsys result was -1.56%/-1.28%. It remained disabled.

## Reopen condition and prediction

Stage59 materially changed the production phase: FlashAttention raw time fell
19.74%, exclusive time fell 44.92%, and accepted S2 throughput rose 4.38%.
That satisfies the recorded reopen condition for locally beneficial,
phase-sensitive fusions. The candidate is supported only if a locked short
ABBA plus reverse BAAB is consistently positive on the new graph. A clear
negative result leaves all five switches disabled without a new full-graph
profile. A stable positive result advances to full 8,192-row accuracy and a
new whole-graph Nsys/broad-NCU checkpoint before deployment.

## Validation order

1. Reuse the component NCU/accuracy evidence; no implementation is changed.
2. Run 100-timed/10-warmup locked-2400 ABBA and reverse BAAB on real S2.
3. If positive, run full accuracy, enable the five flags, then profile the
   accepted full graph and update history. If negative, record and stop.
