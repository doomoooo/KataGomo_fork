# Stage 53 hypothesis: partial C288 no-split g1+v1 head

## Frozen target and evidence

- Control: commit `3526b13`, all retained candidates disabled, RTX 5090 D,
  exact 19x19 B13, FP16/NHWC, natural S2.
- Accepted Stage 47 Nsys/NCU places the separate policy-g1 C768->C96 and
  value-v1 C768->C192 projections at about 28 us/forward total.  Both reread
  the same 7.21 MiB trunk-tip tensor and launch separate library GEMMs.
- The target TensorRT plan independently realizes exactly these two slices as
  one C768->C288 correlation while leaving policy-p1 separate.  This removes
  the previously assumed dependency on fused P1 and establishes that the
  no-split grouping is structurally valid.
- The impossible zero-cost ceiling is about 0.42% of a Stage 47 forward;
  realistic natural-S2 budget is 0.10%--0.25%.

## Single mechanism

Pack the existing g1 and v1 FP16 1x1 weights into one column-major C288 matrix
and issue one C768->C288 projection over the 4,693 B13 spatial rows.  Keep the
wide tensor materialized until both heads finish.  New stride-aware affine-SiLU
consumers read g1 at offset 0 and v1 at offset 96 directly and produce the same
compact post-BN tensors expected downstream.  Do not split/copy the projection
output, do not modify p1, and do not fuse BN into the GEMM in this stage.

The first tactic is cuBLAS Hgemm because the Stage 48 total GEMM audit found
the current library superior for unfused plain GEMMs; the structural grouping,
not a guessed warp shape, is the variable.  If the grouped boundary is viable
but the tactic is weak, its measured NCU result may justify a CuTe schedule.

Dispatch only on RTX 5090 D, exact max/runtime B13, 19x19, FP16/NHWC, C768
input, g1 C96, v1 C192, and 1x1 projections.  Every other case runs the
complete official head paths.  The option defaults false.

## Gates

1. Fixed-B13 candidate/control and B12 fallback replay before performance.
2. Short Nsys/NCU of the complete projection+consumer boundary.  Require one
   C288 GEMM, no split/copy, and a strict reduction in launches/input reads;
   record realized tactic resources instead of guessing geometry.
3. Natural-S2 100/20 ABBA plus reverse BAAB, then 400/40 only if both order
   aggregates are non-negative and pooled mean is positive.
4. If S2 is inconclusive, retain default-off only with both-order positive S1,
   full correctness, and strict profiler work/resource evidence.
5. Default-on acceptance requires stable fixed-B13 8,192-row all-head accuracy,
   fresh full S2 Nsys and complete S1 NCU, history update, and one commit.
