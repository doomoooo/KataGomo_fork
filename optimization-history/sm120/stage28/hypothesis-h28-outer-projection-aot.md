# H28: fixed-B13/S2 outer projection AOT

## Frozen target and evidence

- Target: RTX 5090 D, fixed 19x19, B13, FP16, two independent CUDA
  streams. Each server owns a separate `Model` and therefore a separate
  `ConvLayer::matmulWeightBuf`; the current weight-sharing options are dead
  scaffolding and must not be assumed active.
- Each of 11 nested outer blocks has two fixed 1x1-convolution GEMMs:
  - pre/contract: row-major `M=4693,N=384,K=768`, `beta=0`;
  - post/expand: row-major `M=4693,N=768,K=384`, `beta=1` into the C768 trunk
    residual.
- Ordinal attribution reports, over 60 forwards per stream:
  - contract isolated 7.936 ms, S2 11.614 ms, 1.464x, excess 3.678 ms;
  - expand isolated 9.604 ms, S2 15.768 ms, 1.642x, excess 6.164 ms.
- Current contract launch signature is grid 148, block 128, 164
  registers/thread, 80 KiB dynamic shared memory. Expand is grid 148, block
  256, 154 registers/thread, 72 KiB dynamic shared memory.
- The 5080 fixed CUTLASS contract/expand bundle improved whole-network S2 by
  1.198%; a later expand warp-64x32 adjustment added 0.078%. The 5090 D
  implementation currently has only an unconsumed option, not an AOT path.

## Mechanism

Use fixed-shape CUTLASS FP16-accumulator GEMMs with direct beta-0/beta-1
epilogues. Search CTA shapes around 111--222 blocks and warp layouts that trade
single-kernel latency against cross-stream wave interleaving. Every candidate
is compared in the same C++ process against the exact backend
`cublasHgemm`, with private but value-identical weights and inputs per stream.

Expand is prioritized because it has the larger measured S2 excess. Contract
is kept as a separate variable and must pass its own gate.

## Falsifiable gate

- Standalone output must pass max absolute error <= 0.05 and relative L2 <=
  0.02 against direct cuBLAS.
- A nominee must beat exact cuBLAS S2 by at least 3% in both
  control-candidate and candidate-control order, with no material S1
  regression.
- Nsys must show genuine cross-stream overlap and NCU must explain the result
  through grid shape, shared memory, registers, occupancy, or waves.
- If no candidate passes the exact S2 gate, do not integrate or run full
  network accuracy. Retain raw failures and close conventional AOT tile search.

## Integration risks and later gates

- The beta-1 expand epilogue can change residual-add rounding.
- Isolated self/self overlap does not reproduce its real affine-SiLU,
  linear2, QKV, and library-GEMM peers.
- A passing micro candidate still requires one boundary at a time behind a
  strict 19x19/B13/C768-C384 guard, forward/reverse S2 ABBA, ordinal-aware
  Nsys interference, and the full 8192-row all-head FP32-reference comparison.
- Do not fuse affine-SiLU in this experiment; the 5080 fused-SiLU EVT route
  regressed whole-network throughput by 2.057%.

