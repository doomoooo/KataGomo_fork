# Stage 19 hypothesis: S2-specific dual-GEMM tile sweep

Date: 2026-08-05 UTC

## Frozen target

- RTX 4090 (SM89), exact 19x19, B13, FP16, two independent streams.
- Stage 16 final network and accepted dual-GEMM swizzle 2 remain the control.
- Only the dual-GEMM compile-time threadblock/warp/stage/K tile changes.

## Evidence and mechanism

- Fused FFN dual-GEMM is the largest final-trace kernel family at `85.831ms`
  summed over the last 30 complete forwards on both streams.
- Exact production NCU measures 168 registers/thread, 49.15KiB shared memory,
  15.33% occupancy, 2.60 waves, and 79.27% no-eligible cycles.
- The Stage 8 tile search timed one stream. Its result cannot establish the
  optimum when two independent model/weight sets contend for SM, L2, and tensor
  pipelines. Stage 16 demonstrated that S1 and S2 rankings can diverge sharply.

## Falsifiable test

1. Extend the existing microbenchmark with two non-blocking streams, two cuBLAS
   handles, and independent weights/input/intermediate/output allocations.
2. Compile the previously plausible tile families with swizzle 2. Require every
   candidate to pass the existing full-output local correctness check.
3. At locked 2400MHz, measure complete S1 and synchronized S2-pair boundaries.
   Bracket the sweep with the accepted tile. Integrate only a repeatable S2
   winner with enough margin to survive whole-network drift.
4. Any integrated winner must pass locked whole-network forward/reverse ABBA,
   Nsys union evidence, and the complete FP32 replay gates.

## Risks

- Smaller tiles can increase waves but duplicate input/weight traffic.
- Larger tiles can improve reuse while worsening registers, tail waves, and
  dual-stream fairness.
- A micro winner may still perturb overlap with FlashAttention, QKV, or linear2
  in the complete network.

## Result

- The accepted `TB128x64x32 / warp64x32x32 / stage3 / swizzle2` remains the
  S2 winner. Bracketing observations were `67.363us` and `68.236us` per
  synchronized pair.
- The nearest alternative, `warp32x64x32`, measured `68.561us`; its direction
  was slower and the margin is too small to justify whole-network integration.
- Other S2 pair times were: `TB128x128 80.927us`, `TB256x32 100.295us`,
  `TB64x128 104.464us`, `TB64x64 94.073us`, stage4 `96.454us`, K64
  `102.340us`, and warp64x64 `94.425us`.
- Every candidate passed the same local full-output check
  (`maxAbs=0.000244141`, `relativeL2=4.50e-6`, exact fraction 0.999972 versus
  cuBLAS+standalone SwiGLU). No alternative passed the performance gate, so no
  production code or full replay changed.
