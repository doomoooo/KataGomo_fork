# Stage 63 hypothesis: packed half2 RoPE in the fused QKV epilogue

## Frozen target and evidence

- RTX 4090 SM89, exact 19x19, B13, FP16 NHWC, S2 deployment.
- Control source is clean commit `6fd19dc`; Stage62 half2-tanh dual-FFN remains
  enabled and is not part of this variable.
- The accepted Stage62 S2 Nsys checkpoint ranks fused QKV+RoPE second at
  62.746 ms union (25.34% of GPU busy union) and 28.476 ms exclusive (11.50%).
- Broad NCU measures 30.784 us, 240 registers/thread, 49.152 KiB requested
  shared memory, 1.30 waves/SM and no reported spill.
- The current fused epilogue converts every FP16 RoPE pair to two FP32 scalars,
  performs four scalar multiplies plus two add/subtract operations, and converts
  two results back to FP16. Four pairs are handled per 128-bit epilogue access.
- The neighboring SM120 deployment accepted packed half2 RoPE at +0.630% whole
  model throughput with its 8,192-row accuracy gate passing. Its QKV and RoPE
  boundary is different, so this is prior evidence rather than a transferable
  speed claim.

## Single-variable mechanism

Add a default-off exact-shape switch
`cudaUseQKVRoPEGemmHalf2RoPESm89`. Keep the accepted CUTLASS QKV tensor-core
mainloop, M128xN128 tile, stage count, fused boundary, output layout, FP32
`__sincosf`, launch geometry and FlashAttention path unchanged.

Instantiate a separate compile-time QKV epilogue whose rotation converts the
FP32 `(cos,sin)` coefficients to FP16 and evaluates each pair as packed half2:

`(v0,v1) * (cos,cos) + (v1,v0) * (-sin,+sin)`.

This should replace scalar FP32 rotation arithmetic and pair conversions with
packed `HMUL2/HFMA2` operations. QKV tensor-core accumulation is already FP16;
the precision change is confined to the fused RoPE rotation.

## Falsifiable predictions and gates

1. Pre-change targeted NCU/SASS must confirm the current scalar rotation path
   and establish duration, resources, stalls and instruction counts.
2. Candidate smoke must be finite, preserve policy and optimistic top-1 on the
   26-row corpus, and the disabled path must remain numerically identical to
   Stage62.
3. Candidate SASS must retain the same HMMA count while adding packed half2
   arithmetic and reducing scalar conversion/arithmetic instructions.
4. The complete fused QKV+RoPE kernel must improve in matched NCU and short
   complete-boundary Nsys without spills or a worse occupancy/resource tier.
   A strict repeatable local win is enough to retain the implementation
   default-off; deployment additionally requires positive real S2 evidence.
5. A locally valid candidate proceeds to locked 100-timed S2 ABBA+BAAB. Both
   order directions should be non-negative and the pooled mean must be positive.
6. Deployment requires the 8,192-row all-head comparison against the fixed FP32
   reference to remain inside the accepted both16 envelope. Only a deployed
   candidate receives a new full-graph Nsys/broad-NCU checkpoint and commit.

## Risks

- FP16 coefficient rounding is applied 33 times per forward and is combined
  with the accepted both16 attention path; full accuracy, not the 26-row smoke,
  decides numerical acceptance.
- A runtime branch inside one kernel could keep both scalar and packed paths,
  increasing registers. Separate template instantiations are required.
- `__sincosf` may dominate the epilogue, limiting the local speedup even if the
  rotation arithmetic is strictly cheaper.
