# Stage 21 hypothesis: persist the C384 nested-block residual in L2

Date: 2026-08-06 UTC

## Frozen target

- RTX 4090 (SM89), exact 19x19, B13, FP16, S2.
- Accepted Stage20 trunk persisting-L2 configuration is the control.
- This stage changes only `cudaUsePersistingL2Inner=false/true`; arithmetic,
  kernels, launch geometry, and the accepted outer C768 policy remain fixed.
- Attribution runs use 2400MHz SM clock, independent control/candidate primes,
  500 timed forwards, and both forward ABBA and reverse BAAB order.

## Pre-implementation Nsys/NCU evidence

- In each of 11 nested blocks, `mid` is a B13x361x384 FP16 residual that lives
  across six alternating attention/FFN inner blocks. It is 3,604,224 bytes per
  stream. The C384 `postBN` is the direct end-of-lifetime consumer.
- Reusing the accepted Stage20 candidate trace as the Stage21 baseline, Nsys
  records exactly 1166 C384 SiLU batchnorm consumers (`106 forwards * 11`) at
  7.919961us average; the last 30 forwards on both streams contain 660 calls at
  7.879509us average. The complete S2 kernel union is 251.585792ms.
- In the paired Stage20 NCU report, the same unprotected C384 shape has median
  duration 8.032us, L2 hit rate 52.464652%, 3,612,160 DRAM bytes, and zero
  evict-last sectors. This confirms that the long-lived C384 residual is being
  reloaded from device memory after the six inner blocks.
- The accepted C768 windows require 13.75MiB across S2. Adding one 3.44MiB C384
  residual per stream raises the simultaneous protected working set to 20.63MiB,
  below the RTX 4090's 49.50MiB persisting-L2 limit.

## Mechanism and implementation boundary

Install the C384 access-policy window immediately before preConv writes `mid`,
retain it across all six inner blocks and postBN, then restore the accepted C768
window before postConv accumulates into the outer residual. Increase the device
set-aside only from the exact S2 C768 working set to the exact combined C768+C384
working set. CUDA exposes one access-policy window per stream, so nested and
outer windows are switched, not installed simultaneously.

## Falsifiable post-implementation tests

1. NCU on the exact C384 consumer must show non-zero evict-last sectors plus a
   material L2-hit increase and DRAM-byte reduction. The C768 consumer must be
   rechecked to detect damage from window switching.
2. Nsys must show no new memcpy, allocation, or synchronization. It may add 22
   stream-attribute calls per forward per stream (set/restore for 11 nested
   blocks), but the S2 complete-forward kernel union must not increase.
3. A positive screen is required before three locked forward/reverse ABBA
   rounds. Acceptance requires both order directions positive, a positive
   adjacent-pair majority, and all three round medians non-negative.
4. If accepted, run the complete 8192-row all-head replay against the frozen
   FP32 reference and require byte identity with Stage20 because arithmetic is
   unchanged.

## Risks

- Switching the single stream access-policy window can allow the outer C768
  lines to age while inner blocks execute, even though the combined set-aside
  has capacity for both residuals.
- Forty-four additional host API calls per S2 forward pair can perturb launch
  pacing or CPU submission even if GPU kernel time improves.
- Protecting C384 residuals can evict higher-value weights; NCU cache metrics
  alone cannot accept the change without concurrent Nsys and locked full-graph
  evidence.

## Result

Accepted for exact 19x19, B13, FP16, S2 on RTX 4090.

- Clean split-metric NCU runs showed the C384 consumer at 6.62us -> 6.59us,
  L2 hit 52.51% -> 97.99%, and evict-last sectors 0 -> 112632. The accepted
  C768 consumer remained stable at 11.36us -> 11.30us and L2 hit 82.50% ->
  82.97%.
- The first combined-metric NCU candidate completed all samples and wrote a
  report, but exited with code 6 during process teardown (`double free or
  corruption`) after 12 kernels x 4 replay passes. It is retained under
  `ncu-paired/` as contaminated and is not used for acceptance. A count-2
  one-pass reproducer and all six split-metric control/candidate profiles exited
  normally; valid evidence is under `ncu-single-pass/`.
- Two independent Nsys pairs disagreed in their short kernel-union direction:
  +0.676% and -0.893%. Pooled equally, union changed from 248.017675ms to
  247.749103ms (-0.108%), summed kernel time from 391.660743ms to 384.863460ms
  (-1.736%), and the C384 consumer from 7.975656us to 7.895816us (-1.001%).
  Memcpy, memset, and synchronization counts were unchanged. Attribute calls
  increased from 2 to 24 per forward per stream, exactly matching the design.
- Three locked-2400 forward/reverse ABBA rounds changed the pooled median from
  3117.614216 to 3157.841614 nnEval/s (+1.2903%). Forward and reverse pooled
  directions were both positive (+0.7699% and +0.8554%), every round median was
  positive, and 8/12 adjacent pairs were positive.
- The complete 8192-row replay is byte-identical to Stage20 (SHA256
  `7dde3f6b36e240eb4e92ffc632ecc578d059052fb2c13816b043d0a7093ba484`),
  preserving the accepted all-head FP32 error envelope and p0loss.

Artifacts: `ncu-single-pass/`, `nsys-paired/`, `nsys-paired-r2/`,
`abba-pooled-3r-summary.json`, and `replay-persisting-l2-inner-vs-fp32.json`.
