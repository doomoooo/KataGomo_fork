# Stage 6 hypothesis: exact full-board mask elision

Date: 2026-08-05 UTC

## Frozen protocol

- GPU: CUDA device 0, NVIDIA RTX 4090 (SM89), held through `gpu-lock`.
- Model: `b11c768h12nbt3tflrs-fson-silu.bin.gz`, SHA-256
  `1881600caab9e9d85a3dd6a019e9b8e7d2c237b5f984e13ed49a8645be3077c6`.
- Shape and topology: fixed 19x19, FP16, two NN servers with independent
  per-thread streams, batch 13 per server.
- Timing: `benchmarknn`, 10 warmups and 300 timed device-resident forwards.
- Control: accepted stage-4-equivalent SM89 path, `requireMaxBoardSize=false`.
- Candidate: the same binary and SM89 path, with the existing production
  setting `requireMaxBoardSize=true`.

This is a new fixed-board regime, `L1-EXACT19`. It must not be chained to L0
percentages without displaying both absolute results. It is valid only for a
service that rejects boards smaller than the configured 19x19 NN dimensions.

## Evidence -> mechanism -> prediction

The stage-4-equivalent Nsys trace
`stage4-current.nsys-rep` shows the two timed streams still launching
`sm89MaskZeroNHWCHalfKernel` 66 times and
`maskToAttnBiasFullHalfKernel` 33 times per forward. Across the complete trace,
these families account for 4.3% and 2.9% of summed kernel duration,
respectively. The fixed replay corpus has an all-ones 19x19 mask, so these
operations are identities.

Setting `requireMaxBoardSize=true` makes the existing backend pass a null mask
through the trunk. This should remove both kernel families and select the
no-bias cuDNN SDPA graph. Because two streams overlap, summed durations are not
an Amdahl estimate; the acceptance metric is an ABBA end-to-end result plus a
new Nsys union/exclusive analysis. A repeatable throughput improvement of at
least 1% is expected.

## Risks and falsification

- Semantic scope: the candidate must fail rather than silently accept a board
  smaller than 19x19. The ordinary non-exact path remains the fallback.
- Numerical risk: removing multiply-by-one and zero attention bias may alter
  rounding or cuDNN plan selection. The candidate must pass all 8192 rows,
  every output head, and p0loss directly against the fixed full-FP32 reference.
- Performance falsification: reject or pause if ordered and reverse ordered
  comparisons disagree, if mask/bias kernels remain, or if the gain is below
  run-to-run dispersion.

## Validation metrics

1. `benchmarknn` ABBA and reverse-order medians in absolute `nnEval/s`.
2. Nsys kernel instances, two-stream union/exclusive busy time, and absence of
   the two mask families in the timed window.
3. `replaynn` plus `compare_replay_krnn.py` for all heads and p0loss.
4. Smaller-board smoke demonstrating the exact-size guard.
