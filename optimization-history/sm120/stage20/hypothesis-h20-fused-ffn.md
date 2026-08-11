# H20: fixed-B13 shared-input fused FFN projection

## Evidence

- The Stage-17 S2 Nsys trace attributes 56.9% of summed GPU time to the
  dominant cuBLAS GEMM family and another 5.4% to standalone SwiGLU.
- `TransformerFFNBlock::apply` still launches linear1 GEMM, gate GEMM, and a
  separate C1152 SwiGLU kernel.
- The 5080 history measured the same C384->2xC1152 subgraph at
  `103.8 -> 63.75 us` using a shared-A dual GEMM, with a `+15.77%` whole-network
  improvement in that regime. Its later TileLang kernel used
  M128-N64-K32, two stages, 128 threads, and minimum three blocks per SM.

## Mechanism

For fixed 19x19 B13, launch a single TileLang kernel over M=4693, N=1152,
K=384. Each CTA loads one input tile once, loads independent linear/gate
weight tiles, computes both tensor-core accumulators, and writes only
`SiLU(linear) * gate`. This removes one global input read, two C1152 global
intermediate writes/reads, and two launches from the subgraph boundary.

## Expected change

- Microbenchmark: fused boundary materially below the two-GEMM plus SwiGLU
  boundary; target <=60 us as an initial gate.
- Nsys: 33 FFN blocks replace 99 launches with 33 fused launches per forward.
- Whole network S2/B13: expected >=3% after two-stream scheduling effects.

## Risks

- Register pressure from two accumulator fragments can reduce occupancy and
  stream overlap.
- FP32 accumulation or activation lowering can change the established FP16
  numerical envelope.
- Fixed M tail handling and row-major interpretation of KataGo's cuBLAS weight
  storage must be validated explicitly.

## Gates

1. Isolated output comparison has no NaN/Inf and satisfies `rtol=2e-2,
   atol=2e-2` against a PyTorch FP32-accumulation reference.
2. Isolated fused latency <=60 us and faster than its measured unfused
   boundary under the same stream/protocol.
3. Unsupported batch/shape/precision falls back to the current path.
4. Whole-network smoke, full fixed-corpus accuracy, and long paired throughput
   must pass before acceptance.
