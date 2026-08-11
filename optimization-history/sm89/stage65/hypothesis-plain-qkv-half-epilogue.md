# Stage 65 hypothesis: native-half no-scale epilogue for plain QKV

## Scope and evidence

- RTX 4090 SM89, exact 19x19, B13, FP16 NHWC.
- Stage 64's split path is a retained default-off S1 intrinsic route. The
  deployed S2 control remains Stage 62.
- The Stage 64 plain QKV kernel uses FP16 MMA accumulators and writes FP16 with
  `beta=0` and `ScaleType::Nothing`, but its CUTLASS `LinearCombination` compute
  type is FP32. Targeted SASS still contains 64 F2F instructions per launch.
- Plain QKV is 168 registers/thread, 49.152 KiB shared memory, 2.60 waves/SM,
  and zero spill. Natural S1 Nsys measures the full plain-QKV-to-RoPE boundary
  at 25.948 us versus 28.941 us for fused QKV+RoPE.

## Falsifiable mechanism

Instantiate only the plain-QKV epilogue with FP16 compute while retaining the
same FP16 accumulator, tile, stage count, swizzle, output type, and standalone
RoPE. Since the output op performs no scaling or source accumulation, direct
FP16 fragment storage should preserve every output bit while removing redundant
half-to-float-to-half conversion work.

Expected evidence:

1. 26-row output is byte-identical to the Stage 64 split path.
2. SASS reduces F2F or otherwise shortens the output path without changing the
   64 HMMA mainloop instructions.
3. Registers/shared memory/spills do not regress.
4. Same-protocol targeted NCU improves plain QKV, and natural S1 Nsys improves
   the complete plain-QKV-to-RoPE boundary. Independent NCU durations are not
   summed for the decision.
5. A strict local win is committed on the default-off split path. Only a large
   enough improvement to reopen S2 proceeds to S2 deployment testing.
