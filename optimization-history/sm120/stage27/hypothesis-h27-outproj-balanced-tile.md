# H27: B13/S2 attention out-projection coexistence tiles

## Frozen target and evidence

- Target is RTX 5090 D, fixed 19x19, `B13`, two independent CUDA streams.
- The repeated attention output boundary is
  `M=13*361=4693, N=384, K=384`, FP16 GEMM with a residual epilogue.
- It occurs once after FA4 in each of 33 transformer blocks. In the isolated
  S1 trace the current cuBLAS family is about 8.6 us per common invocation;
  in S2 the aggregate `library_gemm` family has 95% peer overlap and a 1.70x
  median S2/S1 slowdown.
- Stage 22 searched only five TileLang schedules. Its selected
  `M128N128K32-S4-T128` kernel launches 111 CTAs and reserves 64 KiB dynamic
  shared memory. Although its standalone TileLang measurement was 9.31 us,
  enabling it regressed the whole S2 network and was rejected.
- Stage 22's `torch.addmm` control was about 14.5 us, materially slower than
  the actual cuBLAS invocation observed by Nsys. It is not a valid acceptance
  baseline for this boundary.

## Mechanism

Search fixed tiles with either:

1. about 147--180 CTAs and smaller per-thread accumulator/shared-memory
   footprints (`M96N128`, `M128N96`, `M160N64`), matching the current library
   kernel's broad card coverage while leaving more resources for the peer; or
2. about 114 CTAs with 256 threads (`M256N64`), trading isolated occupancy for
   explicit two-stream coexistence.

The primary screen is absolute dual-stream candidate time and repeatability,
not speedup versus `torch.addmm`. A candidate should also avoid the phase drift
seen with the rejected balanced linear2 experiment.

## Expected change and falsification

- A useful candidate should be no slower than 9.0 us in isolated S1 and no
  slower than 17.5 us per stream in the direct S2 microbenchmark, with less
  than 3% range across repeated measurements.
- Prefer schedules with roughly 147--180 CTAs, at most 128 threads, and a
  generated shared-memory footprint compatible with at least two resident
  CTAs when register pressure permits.
- Falsify this route if all resource-balanced tiles are slower than the fixed
  thresholds or fail compilation/correctness. Passing the micro screen only
  nominates a candidate; it does not establish a whole-network gain.

## Risks and required next gates

- TileLang FP16 accumulation changes reduction order; standalone output must
  pass `rtol=2e-2, atol=2e-2`, and an integrated candidate requires the full
  8192-row all-head FP32-reference comparison.
- Tile padding or unsupported MMA layouts may erase resource benefits.
- Microbenchmarks do not reproduce FA4/QKV/FFN peer mixtures. Any nominee must
  be integrated behind the existing default-false B13 guard, then pass S2
  ABBA/reverse-order, Nsys phase/interference analysis, and NCU resource
  validation before acceptance.

## Protocol correction (2026-08-06 06:42 UTC)

The frozen hypothesis assumed shared out-projection weights. Source ownership
audit after the search found that the current two-server path constructs two
independent `Model` objects; `shareModelWeights` is not active runtime behavior.
The exact cuBLAS control and finalist were therefore rerun with value-identical
but pointer-distinct weights. This correction does not rescue the candidate:
cuBLAS is 11.431 us/stream and M96N128 is 18.663 us/stream in corrected S2.
