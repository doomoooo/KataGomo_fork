# Stage 28 hypothesis: one no-split wide head projection

Date: 2026-08-06 UTC

## Frozen target

- RTX 4090 (SM89), exact 19x19, B13, FP16, NHWC, S1.
- The accepted Stage27 S1 configuration is the control.
- Only the three 1x1 head projections from the common C768 trunk may change:
  policy p1 C96, policy g1 C96 and value v1 C192.

## Pre-implementation evidence and falsifiable test

- S1 Nsys observes two C768->C96 GEMMs at about 8.85us each and one
  C768->C192 GEMM at about 15.17us, or 32.86us and three launches per forward.
- Their weights can be concatenated into C384 and evaluated by one fixed
  M4693/N384/K768 AOT projection already used by nested preConv. The output is
  laid out as `[p1:96, g1:96, v1:192]` within each NHWC row.
- The candidate must not materialize three split tensors. The first consumer of
  each slice must accept `rowStride=384`; later tensors may remain contiguous.
- Before implementation, NCU must show the two narrow GEMM shapes have low
  launch waves or occupancy rather than a saturated throughput ceiling. At most
  three representative launches are sampled.
- After implementation, NCU must show the wide AOT kernel is below the summed
  baseline boundary with no spill. Nsys must replace three GEMMs with one and
  improve the complete policy+value projection/first-consumer boundary. Then a
  locked-2400 S1 100-iteration forward/reverse ABBA decides full-graph value.
- Because the projection accumulation/tile order changes, full 8192-row
  all-head FP32 replay is mandatory if performance passes.

## Rejection and reopen conditions

- Reject if stride-aware consumers add split-equivalent traffic, the wide AOT
  kernel is not locally faster, or the short S1 ABBA is negative in both orders.
- Reopen only with a different wide GEMM tile or a wider fused projection+BN
  epilogue.

## Result

- Accepted for exact 19x19/B13/S1; `cudaUseWideHeadProjection=true` is enabled
  only in `/workspace/bench-cuda-gpu0-4090-s1.cfg`.
- Pre-implementation NCU showed the C96/C96/C192 GEMMs at 9.60/9.50/16.064us,
  only 0.58/0.58/0.29 waves per SM and about 8.3% achieved occupancy.
- Candidate NCU measured the C384 projection at 15.94us versus the 35.164us
  summed control projection (-54.67%), with 162 registers/thread, 0.87 waves
  per SM, 8.32% achieved occupancy and zero local/shared spill. The two
  strided BN consumers were 3.20us and 4.45us with zero spill.
- Forward/reverse Nsys removed two launches per forward. The projection plus
  first-BN boundary changed 40.103->21.778us (-45.69%) and
  40.267->21.781us (-45.91%). The profiled whole-graph short runs improved
  2431.623->2439.303 (+0.316%) and 2427.105->2436.187 (+0.374%).
- Locked-2400 S1 100-iteration forward/reverse ABBA measured pooled medians
  2459.496->2474.237 nnEval/s (+0.599%). All four adjacent pairs were positive
  (+0.705%, +0.156%, +0.682%, +0.517%). No longer throughput run was used.
- The 8192-row candidate replay is byte-identical to Stage27 for every raw
  output. Its established all-head FP32 error envelope is therefore unchanged.
