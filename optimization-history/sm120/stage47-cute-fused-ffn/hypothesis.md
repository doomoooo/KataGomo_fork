# H47: CuTe single-accumulator wide fused FFN

## Frozen target and control

- RTX 5090 D, exact 19x19 B13, FP16/NHWC, natural S2.
- Accepted graph remains Stage44; rejected Stage46 changed no accepted code or
  profile.
- Direct control is the accepted TileLang A-reuse fused FFN: M=4693, output
  N=1152, K=384, 666 CTAs, 128 threads, 32 KiB dynamic shared memory, 136
  registers/thread, zero spills, and 34.921 us in its accepted isolated test.
- The matching Stage44 NCU forward reports median 38.880 us, 1.31 waves/SM,
  0.299 eligible warps/scheduler, and 42.66% SM throughput for this family.

## Evidence-backed mechanism

The accepted TileLang dataflow shares A across the linear and gate products,
keeps both FP16 accumulator fragments live, and emits only
`SiLU(linear) * gate`.  The accepted CuTe packed-QKV kernel proves an SM120
M128/N128/K64 mainloop with atom layout 4x2, eight MMA warps plus one DMA warp,
FP16 accumulation, and a 170-cluster persistent grid at the same M and K.  Its
full-profile footprint is 107 registers/thread with zero spills.

H47 packs each 64-column linear/gate pair into one 128-column B tile.  One CuTe
M128/N128 accumulator then contains both operands for 64 output channels.  The
epilogue pairs the two halves, evaluates FP32 SiLU, multiplies by the gate, and
stores only 64 FP16 columns.  This retains the accepted 18x37 logical work
tiles while using one accumulator rather than two separately scheduled GEMMs.

The eight-MMA-warp choice is not a warp-shape guess: the target NCU shows
register pressure at 136 registers, and the exact neighboring CuTe mainloop
demonstrates that distributing the same M128xN128 accumulator over eight MMA
warps lowers the footprint to 107 registers.  No alternative atom/tile shape
will be tried unless target NCU or generated SASS identifies a concrete
limiter.

## Staged falsification

Before writing the compressed custom epilogue, compile the exact CuTe
M128/N128/K64 N=2304 mainloop and pair it with the existing wide-SwiGLU kernel.
Measure it against the accepted TileLang fused kernel on identical tensors and
streams.

- If the wide CuTe GEMM alone is not faster than the complete accepted fused
  boundary, the mainloop has no epilogue budget and H47 stops.
- If CuTe GEMM plus standalone SwiGLU is within 15% of the accepted boundary,
  the eliminated intermediate write/read and launch provide a credible fused
  path; implement the custom epilogue.
- Targeted NCU then decides from registers, shared memory, spills, eligible
  warps, waves, memory sectors, and dependency stalls.  Geometry is not swept.

## Validation gates

1. Standalone wide-mainloop probe and identical-data comparison.
2. Custom fused output smoke with no NaN/Inf and full FP16-word/error report
   against the accepted TileLang output.
3. Targeted NCU and natural direct-boundary Nsys/events.
4. Short S1 and natural real S2 ABBA.  A strict NCU/S1 win may remain
   default-off; only reproducible real-S2 improvement becomes automatic.
5. Only an accepted automatic change receives the 8,192-row all-head replay,
   fresh S2 Nsys, matching 344-ordinal S1 NCU, history update, and one commit.
