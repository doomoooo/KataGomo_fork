# Stage 60 hypothesis: reopen attention RMSNorm-to-QKV folding

## Frozen target and control

- GPU/backend: RTX 4090 SM89, exact 19x19, B13, FP16 NHWC, two independent
  NN-server streams.
- Control source: commit `7d299d0`, with Stage59 FlashAttention both16 enabled.
- Single variable: `cudaUseFoldedRMSAttentionSm89`, default false. No FFN RMS
  folding, phase control, tile change, or other bundle is in scope.
- Non-exact shapes, non-FP16 paths, and unsupported QKV dispatches retain the
  current full RMSNorm and projection fallback.

## Current evidence

The accepted Stage59 full-graph checkpoint ranks QKV+RoPE second by both busy
union and exclusive critical time:

- QKV+RoPE: 63.258 ms raw, 28.534 ms exclusive, 25.33% busy union and 11.47%
  exclusive.
- RMSNorm: 22.942 ms raw, 4.296 ms exclusive.
- Broad S2-source NCU: QKV+RoPE 30.400 us and RMSNorm 6.304 us.

Stage48A previously implemented exactly this attention-only transform. Its
S2-source NCU boundary improved from 4.576+29.888=34.464 us to
2.848+26.880=29.728 us (-13.742%), with finite 26-row output and unchanged
policy/optimistic top-1. It was rejected only because the old workflow treated
two short whole-graph results (-0.271% and +0.160%) as a mandatory S2 gate.
Under the current workflow this is a resource-positive candidate that should
have been retained, and Stage59 has materially changed the dual-stream phase by
reducing Flash raw time 19.74% and exclusive time 44.92%.

## Mechanism and prediction

For attention blocks only:

1. Fold the FP32 RMS gamma into each Q/K/V weight once during model setup, then
   convert the folded weights to FP16.
2. Replace the full C384 normalized-tensor write with a warp-per-row reduction
   that writes one FP32 invRMS scalar per token.
3. Feed the original residual tensor to QKV GEMM and multiply the projected
   half accumulator result by invRMS in the existing RoPE epilogue before Q/K
   rotation (and before storing V).

This removes the 3.6 MiB normalized tensor write/read boundary per attention
block and leaves the QKV tile/grid unchanged. The candidate is supported if
S2-source NCU again shows a smaller `invRMS + QKV` boundary, no spill, and no
resource regression that outweighs the removed traffic.

The predeclared expectation is a local boundary reduction of at least 8%. Full
S2 throughput may be small because the old graph compressed a 13.7% local win
to noise; on the new graph any strict local NCU/resource win is retained
default-off even if short S2 is inconclusive. A reproducible S2 regression
prevents deployment.

## Numerical risk and validation order

Folding gamma into FP16 weights and moving invRMS scaling after Tensor Core
accumulation changes rounding. The user accepts the established approximate
precision envelope, but NaN/Inf or all-head regression outside that envelope
rejects the candidate.

1. Restore the old single-variable implementation and build.
2. Run 26-row control/candidate all-head smoke and exact-shape/fallback checks.
3. Capture 2-3 real S2 invRMS and folded-QKV launches with NCU; compare the
   complete boundary to current full RMSNorm+QKV.
4. If locally supported, run short locked-2400 ABBA and BAAB on the real S2
   graph.
5. Deploy only for a stable positive S2 result. Retain default-off for a strict
   local/resource win with inconclusive S2. Reject/revert only for correctness,
   mechanism, or clear graph regression.
6. Only a deployed result advances to 8,192-row all-head accuracy, accepted
   full-graph Nsys/broad NCU, one optimization commit, and history update.
