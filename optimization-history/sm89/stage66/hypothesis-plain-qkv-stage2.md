# Stage 66 hypothesis: stage-2 native-half plain QKV

## Evidence and scope

- RTX 4090 SM89, exact 19x19, B13, FP16 NHWC.
- This experiment only changes the retained default-off split-QKV path.
- Stage 65 variant 1 is the control: M128xN128xK32, warp 64x64x32,
  three-stage CUTLASS mainloop, native-half no-scale epilogue.
- It uses 168 registers/thread and 49.15 KiB dynamic shared memory, limiting
  residency to two 128-thread CTA/SM. The launch has 333 CTA, or 1.30 waves at
  the measured occupancy tier.

## Falsifiable mechanism

Variant 2 keeps tile, warp shape, instruction shape, swizzle, FP16 accumulator,
and native-half epilogue fixed, changing only the mainloop from three stages to
two. Expected shared memory is about 32.8 KiB, allowing three CTA/SM. The extra
resident work may hide load/MMA dependencies and reduce dual-stream contention,
but one fewer prefetch stage may instead starve Tensor Cores.

Validation:

1. Same-binary variant 1/2 smoke must be byte-identical.
2. NCU must confirm 49.15 -> about 32.8 KiB shared, no spills, and the intended
   occupancy tier. SASS/mainloop evidence must remain explainable.
3. Natural S1 Nsys directly measures the complete QKV-launch-to-RoPE-completion
   boundary; no independent NCU durations are added.
4. If local evidence passes, short natural S2 Nsys tests whether the third CTA
   improves the actual contention that rejected Stage 64.
5. Retain/commit only if the local mechanism is strictly useful; deploy only if
   real S2 improves.
