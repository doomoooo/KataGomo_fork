# H46: SM120-native CuTe postConv residual + following affine-SiLU

## Frozen control

- Target: RTX 5090 D, exact 19x19, B13, FP16/NHWC, natural two-server S2.
- Accepted graph: Stage44, source `ee4d1d85ed6493f2710d938f924578b1ec6d46ca`.
- Working source starts at `67f034d`; its Stage45 CUTLASS2 candidate is
  default-off, so the automatic control is unchanged.
- `CUDA_DEVICE_MAX_CONNECTIONS` is unset and stream priorities remain equal.

## Evidence and mechanism

The accepted Stage44 profile reports 11 C384->C768 postConv residual GEMMs.
Each is immediately followed by a C768 affine-SiLU, except that the final
boundary feeds the trunk tip.  The accepted postConv uses 154 registers and
73.728 KiB dynamic shared memory; the separate affine uses 17 registers.

Stage45 proved the wider boundary and its rounding semantics:

- launches 2->1 at every one of the 11 boundaries;
- eliminates the activated output's full residual reread;
- 256-thread CUTLASS2 kernel: 108 registers, 50.176 KiB total shared memory,
  zero spills, eligible warps/scheduler about 0.341;
- S1 `3267.010 -> 3272.912` (+0.181%), but natural S2
  `4047.917 -> 4030.194` (-0.438%);
- full 8,192-row replay is byte-identical.

The remaining falsifiable hypothesis is that Stage45 lost because it used an
SM80 CUTLASS2 mainloop on SM120.  An SM120-native CuTe implementation retaining
the same dual-output epilogue can preserve the structural saving while reducing
mainloop cost enough to reverse the S2 result.

## Initial implementation choice

Use exactly the accepted Stage44 CuTe packed-QKV neighbor as the evidence-based
starting point: M128/N128/K64, atom layout 4x2x1 (eight MMA warps plus one DMA
warp), FP16 accumulation, and persistent scheduling capped at 170 clusters.
The shape has the same M=4693 and K=384; only N changes 1152->768 and the
epilogue adds residual/dual-output affine-SiLU work.  This is one transferred
configuration, not a tile/warp sweep.

No alternative atom/tile/pipeline shape will be tried unless NCU/SASS identifies
a concrete limiter such as spills, occupancy, tail waste, memory sectors, or a
dependency stall.

## Semantics and risks

- GEMM accumulation and residual result use FP16, matching the accepted CuTe
  QKV arithmetic mode and the current half output contract.
- Store the rounded FP16 residual output, then derive activation from that same
  rounded value.
- Affine uses FP16 scale/bias and FP16 FMA; SiLU evaluates exp/division in FP32
  and rounds back to FP16.
- Both outputs must be tail-safe for M=4693.
- CuTe's extra residual load and nonlinear epilogue may raise registers or make
  its accepted packed-QKV persistent schedule unsuitable.  NCU decides whether
  the hypothesis survives; geometry is not guessed around a failure.

## Gates

1. Generate and compile AOT without touching a GPU; preserve exact-shape
   dispatch and official fallback.
2. Boundary smoke against the existing Stage45 implementation and a small
   replay/NaN check.
3. Targeted NCU: record duration, registers, shared memory, spills, waves,
   occupancy, eligible warps and memory traffic against both cuBLAS+affine and
   Stage45 CUTLASS2.
4. Short forward/reverse S1, then natural whole-graph S2.  A correct strict
   resource/S1 improvement may remain default-off; only a reproducible S2 win
   becomes automatic.
5. Only an accepted S2 change receives the full 8,192-row all-head regression,
   fresh S2 Nsys, matching 344-ordinal S1 NCU, history update and one commit.
