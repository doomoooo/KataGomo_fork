# H39b: Remove the redundant post-commit FFN barrier

## Frozen protocol

- RTX 5090 D, fixed B13/19x19 FFN boundary, FP16, accepted A-reuse kernel.
- Change one instruction-level dependency only: remove the second
  `__syncthreads()` in each main-loop iteration after the next-linear
  `cp.async` commit.
- Preserve the top-of-iteration `cp.async_wait<2>()` plus barrier and the
  barrier before overwriting `linear_shared`.
- Pre-screen with deterministic boundary bit comparison, interleaved S1, and
  NCU. Do not collect homogeneous or mixed local S2.
- A positive, explained S1 result proceeds directly to natural whole-graph S2.

## Evidence and mechanism

The removed barrier does not guard a producer/consumer boundary. The preceding
copy writes `linear_shared`; the following copies write disjoint `input_shared`
and `gate_shared` regions. Completion and visibility of the next pipeline tile
remain guarded by the following iteration's `cp.async_wait<2>()` and block
barrier. Stage38 reports `0.511` barrier-stall cycles per issue for this kernel.

## Prediction and stop rule

The output must be bit-identical. PTX/SASS must contain one fewer loop barrier
without changing registers, shared memory, grid, or spills. Continue only if
interleaved S1 is positive beyond run noise and NCU agrees with the mechanism.
