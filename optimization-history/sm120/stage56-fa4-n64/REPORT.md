# Stage 56: exact-B13 FA4 N64 tactic audit

## Decision

Retain the `M128xN64`, one-stage, 128-thread both16 FA4 tactic behind
`cudaUseFlashAttentionN64Sm120=false`.  It is a strict natural-S1 and
residency-resource improvement, but it is not promoted to the RTX5090D/B13/S2
default because controlled S2 runs did not reproduce the portable scanner's
preliminary `+1.02%` result.

The accepted graph remains Stage 47.  No phase controller, cross-stream event,
or artificial phase offset is used or planned; eager launches are left to the
CUDA/GPU scheduler.

## Origin and fixed regime

The portable batch/device scanner reported, at exact B13, a 30/10-iteration
median of `4008.871 nn/s` for N64 versus `3968.559 nn/s` for N128.  That was a
candidate-discovery result from a different dirty build, not acceptance
evidence.  Stage 56 regenerated both tactics and compared them on the Stage-47
fixed 19x19, B13, both16 baseline on RTX 5090 D.

The only schedule change is `tile_n=128 -> 64`; tile M, stages, threads,
Q-in-registers mode, arithmetic types, tensor layout, and launch grid remain
fixed.

## Correctness and direct timing

Both standalone kernels pass the FlashAttention reference check:

- N128: max/mean absolute error `0.000244 / 0.000018`
- N64: max/mean absolute error `0.000305 / 0.000018`

The isolated steady-state event timer is negative for N64:

- N128 pooled median: `11.8034 us`
- N64 pooled median: `11.9264 us`
- delta: `+1.042%` latency

This is consistent with N64 executing more instructions to traverse twice as
many N tiles.  It does not invalidate the resource hypothesis, which concerns
residency and coexistence within a full forward.

## NCU mechanism evidence

One matching full-set NCU capture per tactic reported:

| metric | N128 | N64 |
|---|---:|---:|
| registers/thread | 247 | 168 |
| dynamic shared memory/CTA | 24.58 KiB | 16.38 KiB |
| theoretical occupancy | 16.67% | 25.00% |
| achieved occupancy | 14.29% | 20.19% |
| eligible warps/scheduler | 0.38 | 0.53 |
| one-or-more-eligible cycles | 30.94% | 36.34% |
| NCU replay duration | 19.26 us | 15.68 us |
| executed instructions | 7,328,880 | 7,922,304 |

Both kernels have zero local/shared spilling.  N64 reaches three register-bound
CTAs/SM rather than two, while using one third less shared memory.  The measured
resource improvement was derived from NCU; no warp shape was guessed.

## Whole-forward performance

Natural S1 100/20 A-B-B-A, common-wall pooled:

- N128: `3371.991 nn/s`
- N64: `3386.731 nn/s`
- delta: `+0.437%`

Both candidate arms were positive relative to both bracketing controls.  This,
combined with the NCU residency reduction and zero spills, satisfies the
resource-positive retention policy.

Natural S2 did not establish a benefit:

- 100/20 pooled: `4019.315 -> 4015.401 nn/s` (`-0.097%`)
- 400/50 pooled: `3986.624 -> 3979.085 nn/s` (`-0.189%`)

The 400/50 control itself drifted from `4022.501` to `3950.747 nn/s`
(`-1.784%`), much larger than the candidate delta.  This confirms that the
portable 30/10 scan is useful for discovery but cannot promote a tactic.  It
also shows why making individual arms longer is not automatically higher
quality on an unlocked consumer GPU: thermal/power drift can dominate a small
effect.  N64 therefore remains default-off rather than being rejected or
promoted.

## Full-network accuracy and fallback

The stable B13 8,192-row replay passes the established both16 regime:

- policy top-1 versus FP32: `99.7314%`
- policy probability RMSE versus FP32: `1.03166e-4`
- outcome RMSE versus FP32: `0.0022529`
- score all-6 RMSE versus FP32: `0.0072682`
- policy top-1 versus Stage 47: `99.7192%`

Option-on B12 and option-off B12 outputs are byte-identical.  Option-off B13 is
numerically identical to the prior accepted path (all tensor errors zero; one
header metadata byte differs).

## Implementation and artifacts

`build_aot.py` now accepts explicit tile/stage and symbol-prefix arguments.
The checked-in N64 artifact uses a distinct `fa4_n64` symbol namespace, so one
binary contains both tactics and selects N64 only when the option is true and
the runtime batch is exactly 13.  Other batch sizes continue through N128.

Artifacts are in `aot-n64-prefixed/`, `smoke/`, `microbench/`, `ncu/`,
`natural-s1-abba/`, `natural-s2-abba/`, `natural-s2-medium-abba/`,
`accuracy-full/`, and `fallback-smoke/`.

No fresh accepted full-forward Nsys/344-ordinal NCU profile was produced because
the default graph did not change; Stage 47 remains the accepted profile.
