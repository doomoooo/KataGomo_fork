# Stage 62 hypothesis: FP16x2 tanh SwiGLU inside the accepted dual-FFN kernel

## Frozen target and control

- RTX 4090 SM89, exact 19x19 B13 FP16 NHWC, deployed S2 Stage59 graph.
- Control source is clean commit `7d299d0`; accepted dual-FFN geometry remains
  `M128xN64xK32`, warp `M64xN32xK32`, stage 3, swizzle 2.
- Single variable: a new default-off
  `cudaUseDualGemmSwiGLUHalf2TanhSm89` switch changes only the final SwiGLU
  epilogue arithmetic. GEMM mainloop, tensor-core accumulator type, tile,
  launch grid, weights, inputs, outputs, and all other operators stay fixed.

## Nsys/NCU/SASS evidence

The accepted Stage59 full graph ranks dual-FFN first at 92.494 ms raw and
45.504 ms exclusive: 37.08% of busy union and 18.30% of exclusive busy time.
Broad NCU measures 42.208 us, 168 registers/thread, 49.152 KiB shared memory,
2.60 waves/SM, and no spill.

The current epilogue declares `ElementCompute=float` even though both tensor
core accumulators and the stored SwiGLU output are FP16. A representative
control SASS launch contains:

- 64 `HMMA.16816.F16` instructions;
- 64 scalar `MUFU.EX2` plus the corresponding reciprocal path for sigmoid;
- 96 FP32-to-FP16 pack conversions;
- no half2 multiply/FMA in the activation path.

The full-section NCU capture reports math-pipe throttle as the largest
scheduler stall ratio (2.89 cycles per issued instruction), followed by wait
(1.85) and long scoreboard (1.58). This makes the FP32 scalar activation a
measured cost, not a source-only guess.

The 5090D optimization history supplies an independent prior: its accepted
TileLang dual-FFN uses half2 tanh SwiGLU, passed the 8,192-row all-head gate,
and beat its CUTLASS dual kernel by 1.99% whole-model throughput. Stage33 on
4090 tested that entire TileLang schedule and lost 16.1%; it did not isolate
the activation arithmetic inside the already-superior Ada CUTLASS mainloop.

## Mechanism and prediction

Keep the accepted mainloop and replace only its output functor with packed
FP16 arithmetic:

`silu(x) = x * (0.5 * tanh(0.5*x) + 0.5)`

CUTLASS's SM75+ half-array fast math emits `tanh.approx.f16x2`, allowing two
elements per transcendental and packed half2 multiply/FMA. This should remove
the FP32 conversion/reciprocal sequence and halve transcendental instruction
count. Predeclared local support requires:

- at least 5% lower complete dual-FFN kernel median across matched block
  ordinals or a clearly lower complete all-block total;
- SASS actually contains packed half2 tanh/arithmetic and fewer scalar
  MUFU/conversion instructions;
- no spill and no register/shared-memory/occupancy regression that outweighs
  the instruction reduction.

At 18.30% exclusive share, a 5-10% kernel reduction has an Amdahl ceiling of
roughly 0.9-1.8% S2 throughput before overlap effects. A strict local win
advances to short locked S2 ABBA+BAAB. Stable S2 gain advances to 8,192-row
accuracy, deployment, a fresh whole-graph Nsys/broad-NCU checkpoint, history,
and its own commit.

## Numerical risk

The FP16 sigmoid/tanh path changes SwiGLU rounding and is not expected to be
byte-identical. The user accepts approximate both16-style precision, but NaN,
Inf, top-1 collapse, or a material all-head error-envelope regression rejects
deployment. A 26-row control/candidate smoke precedes performance profiling;
the full FP32 comparison remains mandatory before acceptance.
