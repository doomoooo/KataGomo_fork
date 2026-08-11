# Stage 31 hypothesis: Stream-K scheduling for B13 linear2

## Frozen target

- GPU: RTX 4090, SM clock locked to 2400 MHz
- Model/shape: `b11c768h12nbt3tflrs-fson-silu`, batch 13, 19x19, FP16
- Production target and full-graph gate: two NN servers / two CUDA streams (S2)
- NCU isolation only: one NN server, used solely to explain a representative kernel
- Control binary: SHA256 `6ac7e39efcb492a78d9959c08d5a28d61b994b56a22d9324d299dfa24b134fa1`
- Control path: accepted Stage 29 configuration with CUTLASS data-parallel linear2

## Evidence

Nsys attributes about 691.3 us/forward (13.54% of summed kernel time) to the 31
linear2 projections. Representative NCU launches are 22.18-22.34 us with a 111
CTA grid, 0.87 waves, one CTA/SM, 8.31% achieved occupancy, 64.18% compute and
70.63% L2 throughput. The 65.54 KB dynamic shared-memory footprint fixes
residency at one CTA/SM, while the 111 output tiles leave 17 of 128 SMs unused.
Prior M/N tile sweeps lost performance, so reducing tile size is not reopened.

## Falsifiable mechanism

Keep the accepted 128x128x32 tile, warp shape, four stages, FP16 accumulation,
and beta=1 residual epilogue unchanged. Replace only the identity data-parallel
swizzle with CUTLASS Ampere `ThreadblockSwizzleStreamK`. Its work-centric K
partition should spread the 111 tiles' 36 K iterations across roughly one full
SM-width wave and reduce wave-quantization loss.

## Expected profiler changes

- The main Stream-K launch should use approximately 128 CTAs rather than 111.
- Representative linear2 time must improve by at least 5% from 22.24 us.
- There must be no per-forward allocation, host synchronization, or steady-state
  workspace memset/reduction launch that erases the kernel gain.
- Registers/spills and L2 traffic may rise due to fixup, but zero local-memory
  spills is preferred and the total linear2 boundary must still improve.

## Risks and rejection gates

- K-split fixup adds global partial traffic and synchronization; reject if the
  isolated boundary improves by less than 5% or becomes less stable.
- Reduction order may change. A 26-row replay must be finite and within the
  existing FP32 thresholds; bit identity is not required. Run the full 8192-row
  accuracy suite only after performance gates pass.
- Proceed directly to S2 20-iteration forward/reverse Nsys after the isolated
  NCU mechanism check. Both linear2-family critical-path behavior and S2 whole-
  forward throughput must improve in both orders. Only then run one locked
  100-iteration S2 ABBA confirmation.
- S1 full-graph throughput is not an optimization target and is not an
  acceptance gate. The one-stream NCU run is isolation evidence only; a local
  kernel win that reduces S2 overlap or throughput is rejected.

## Result

Rejected and reverted. S2 Nsys confirms that the Stream-K kernel executed on
both streams, but CUTLASS selected 111 CTAs rather than the predicted full
128-SM wave. Registers increased from 162 to 168. In forward order, mean
linear2 duration changed 28.927 -> 28.616 us and throughput changed
3170.676 -> 3194.142 nnEval/s. In reverse order, both signs flipped:
28.322 -> 28.943 us and 3158.418 -> 3137.575 nnEval/s. The predicted scheduling
mechanism did not occur and the full-graph result is order-sensitive noise, so
no ABBA or 8192-row run was performed. The 26-row smoke replay was byte exact.

The post-implementation NCU script filtered for `Kernel`, while CUTLASS names
this specialization `Kernel2`; that report therefore does not contain the
Stream-K launch and is retained as a failed collection artifact, not evidence.
The pre-implementation NCU control report remains the basis for the original
wave-quantization hypothesis, while post-implementation grid/register/time and
S2 behavior come directly from the four Nsys traces.
