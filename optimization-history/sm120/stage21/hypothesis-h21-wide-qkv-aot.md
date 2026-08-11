# H21: fixed-B13 wide QKV AOT with planar output

## Evidence

- The current B13/S2 attention block launches three independent C384->C384
  `cublasHgemm` projections over the same RMS-normalized input.
- Strided-batched QKV reduced launches but regressed S2 by 0.548%; it does not
  share the input tile inside a CTA.
- The 5080 history retained a true C384->QKV1152 CuTe AOT projection and
  measured a 1.5-2.1% whole-network gain in its two-stream regime.

## Mechanism

For fixed M=13*361, K=384, and N=3*384, pack Q/K/V weights once as a wide
KxN matrix. A single TileLang GEMM loads each input tile once and writes its
epilogue directly into three planar [M,384] Q/K/V regions. The existing fused
RoPE and FA4 kernels therefore keep their current contiguous layouts.

## Expected change

- The isolated wide projection must beat three FP16-accumulator GEMMs under
  two concurrent streams by at least 10%.
- Whole B13/S2 throughput must improve by at least 0.5% in ordered A/B.

## Risks

- A wide tile can consume enough registers/shared memory to reduce overlap
  with the second server stream.
- Weight packing or planar epilogue indexing can silently transpose Q/K/V.
- FP16 accumulation must remain inside the accepted numerical envelope.

## Gates

1. Isolated output passes `rtol=2e-2, atol=2e-2` against three PyTorch
   FP16-output projections with no NaN/Inf.
2. Unsupported shape, precision, or board size falls back unchanged.
3. Nsys confirms balanced independent streams and the expected launch
   replacement.
4. Long paired throughput and the 8,192-row all-head accuracy matrix pass
   before acceptance.
