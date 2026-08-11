# Stage 64 hypothesis: split the AOT QKV projection from fused Q/K RoPE

## Frozen target and evidence

- RTX 4090 SM89, exact 19x19, B13, FP16 NHWC, S2 deployment at clean commit
  `6fd19dc`.
- Stage62 full-graph Nsys ranks fused QKV+RoPE second at 25.34% union and
  11.50% exclusive GPU busy time. Together QKV+Flash occupy 118.014 ms, or
  47.66% of the 247.599 ms busy union.
- Fresh Stage63 control NCU measures fused QKV+RoPE at 30.005 us mean,
  240 registers/thread, 49.152 KiB shared, 1.30 waves/SM and zero spill.
- SASS contains 64 HMMA, 64 `MUFU.SIN`, 64 `MUFU.COS`, 128 F2F, 192 FFMA
  and 256 FMUL per launch. The RoPE epilogue is therefore a material part of a
  low-residency tensor-core kernel rather than a free fused operation.
- Stage13's existing one-launch Q+K RoPE kernel was accepted and Stage16 proved
  that the current fused-QKV result is byte-identical to that separated
  arithmetic. The neighboring SM120 deployment also rejected its fused
  QKV+RoPE AOT route and retained a separated boundary. These are prior
  correctness/mechanism signals, not transferable performance results.

## Single-variable mechanism

Add a default-off exact-shape switch `cudaUseSplitQKVRoPEGemmSm89`.

- Keep the accepted M128xN128xK32, warp64x64, stage-3 CUTLASS batched-QKV
  mainloop, weights, planar Q/K/V layout and FP16 tensor-core accumulation.
- Select the already-instantiated ordinary linear-combination epilogue instead
  of `RoPEOutputTileIterator`.
- Immediately run the existing single-launch `sm89ApplyRoPEQKHalfKernel` over
  Q and K, then enter the unchanged both16 FlashAttention kernel.
- The control and candidate stay in one binary through separate compile-time
  kernels; no runtime branch is added inside either GPU kernel.

This changes one connected boundary: one fused 240-register QKV+RoPE launch
becomes a lighter AOT QKV launch plus one fused-QK RoPE launch.

## Falsifiable break-even and gates

1. Before implementation, fresh NCU of the existing standalone fused-QK RoPE
   kernel establishes its current cost and resources. If that cost is `R`, the
   plain AOT QKV kernel must be strictly faster than `30.005-R` us merely to
   break even; the implementation gate is at least 2% faster for the complete
   `plain QKV + RoPE` boundary.
2. Candidate smoke should be byte-identical to Stage62, because Stage16 already
   established equality between fused-epilogue and separated RoPE arithmetic.
3. NCU/SASS must show the plain QKV kernel removes SIN/COS and materially lowers
   instruction count/register pressure without spills. The complete two-kernel
   boundary, not the plain kernel alone, decides local acceptance.
4. A strict complete-boundary win proceeds to short locked S2 ABBA+BAAB. Both
   order directions must be non-negative and the pooled mean positive.
5. A deployed candidate then receives the 8,192-row all-head check, full-graph
   Nsys/broad NCU checkpoint, history update and its own commit.

## Risks

- The separate kernel rereads and rewrites about 13.75 MiB of Q/K data per
  attention block and adds 33 launches per forward.
- Lower QKV registers may not raise resident CTA count because 49.152 KiB shared
  memory already limits the kernel to two CTA/SM.
- Even a faster summed boundary may perturb S2 overlap, so single-stream or
  homogeneous local timing cannot authorize deployment.
