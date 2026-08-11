# Stage 54 hypothesis: attention out-projection residual + following RMSNorm

## Frozen target and evidence

- Control: commit `3526b13`, retained candidates disabled, RTX 5090 D,
  exact 19x19 B13, FP16/NHWC, natural S2.
- Stage 47 accepted full profile attributes 578.6 us/forward to 33 attention
  out-projection residual GEMMs and 118.2 us/forward to the immediately
  following FFN RMSNorms. The out-projection has 2.048x S2/S1 slowdown,
  164 registers/thread, 81.9 KiB shared memory, 0.87 waves/SM and 8.3%
  achieved occupancy.
- Stage 49 already falsified final-inner linear2 + outer affine-SiLU for both
  CuTe and TileLang, so it is not repeated.

## Mechanism and bounded implementation order

Compute `C384xC384 out-projection + FP16 residual add + C384 RMSNorm` as one
described boundary while preserving both outputs: the rounded residual remains
the transformer trunk, and the normalized tensor feeds FFN linear1/gate.

1. Probe cuDNN 9.25 Graph `matmul -> add -> RMSNorm` first. It can express the
   existing physical `[1,C384,rows4693]` layout without a transpose. Enumerate
   actual engines and measure launch count/workspace/complete-boundary time.
2. Only if the graph cannot form a competitive plan, consider a CuTe kernel
   whose output tile spans all C384 channels for each owned row group. Tile and
   atom layout must be derived from the cuDNN/CuTe probe and target NCU, not
   guessed. The accepted TileLang/cuBLAS kernels remain strategy controls.

After cuDNN rejected the graph pattern, historical and implementation evidence
refined step 2 before any full-C384 tile was guessed. Transformer FFN RMSNorm
has beta=0, so gamma can be folded once into the paired FFN weights and a
scale-only kernel can emit one FP32 inv_rms per row. The accepted Stage-47 CuTe
mainloop can then multiply both accumulators by that row scale in its epilogue.
This preserves the winning cuBLAS out-projection, needs no cross-CTA reduction,
and provides a bounded microbenchmark of the same removed normalization
boundary. The old 5080 TileLang algebra-fold result (-0.016%) is treated as a
strong warning, not ignored.

## Gates

1. Independent plan probe before backend integration. Reject a multi-kernel or
   slower plan unless it still strictly lowers resources/traffic enough to
   justify an integrated S1 measurement.
2. Preserve the FP16 residual rounding boundary and compare both residual and
   normalized outputs. No NaN/Inf; byte identity is preferred, otherwise use
   the fixed all-head accuracy envelope.
3. Nsys must show a smaller complete boundary and NCU must report realized
   registers/shared memory/spills/occupancy/eligible cycles.
4. Natural S2 both-order gate decides default deployment. A correct, strictly
   work/resource-positive S1 candidate may remain default-off under the
   retained-candidate rule.
5. Only a default-accepted result receives a fresh full-graph profile; every
   beneficial retained/accepted result receives its own commit.
