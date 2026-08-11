# Stage 57: exact-B13 FA4 N96 accepted tactic

## Decision

Accept `M128xN96`, one-stage, 128-thread both16 FA4 as the automatic RTX5090D
19x19/B13 tactic.  `cudaUseFlashAttentionN96Sm120` now defaults true and is
gated on runtime batch exactly 13; N128 remains the fallback for every other
batch.  The retained N64 tactic stays available but default-off, and enabling
both alternatives is rejected as a configuration error.

Stage 57 is the final optimization stage for this effort.  No further target
selection is performed.

## Evidence-driven selection

The candidate was not chosen from an assumed warp shape.  Matching NCU data
for accepted N128 and experimental N64 exposed a resource/work tradeoff:

- N128: 247 registers, 24.58 KiB shared memory, two register-bound CTAs/SM.
- N64: 168 registers, 16.38 KiB shared memory and three CTAs/SM, but 8.1% more
  executed instructions and slower isolated steady-state timing.

N96 kept M128, 128 threads, one stage, `Q_in_regs=false`, both16 arithmetic,
launch grid, and tensor ABI fixed.  NCU then established its actual result:

| metric | N128 | N96 |
|---|---:|---:|
| registers/thread | 247 | 233 |
| dynamic shared memory/CTA | 24.58 KiB | 20.48 KiB |
| executed instructions | 7,328,880 | 7,280,208 |
| theoretical occupancy | 16.67% | 16.67% |
| eligible warps/scheduler | 0.38 | 0.38 |
| targeted NCU duration | 19.26 us | 16.74 us |

N96 remains in the same two-CTA occupancy class, but strictly reduces both
hard residency resources and executed work, with zero spilling.  The accepted
344-ordinal full-forward NCU independently confirms the N96 FA4 launch at 233
registers and 20.48 KiB dynamic shared memory.

Standalone reference validation passes with max/mean absolute error
`0.000244 / 0.000018`.  Isolated event timing is nearly neutral/slightly
negative (`11.8034 -> 11.8296 us`, `+0.222%` latency), demonstrating why the
whole-forward coexistence measurement is required.

## Natural whole-forward throughput

Separate-object natural S1, two symmetric 100/20 orders, pooled common wall:

- N128: `3362.272 nn/s`
- N96: `3375.501 nn/s`
- delta: `+0.393%`

Four short separate-object S2 sequences (two ABBA and two BAAB) pooled to
`3995.319 -> 4007.367 nn/s` (`+0.302%`), with six of eight adjacent comparisons
positive.  Short interleaved arms were used instead of a long single arm
because longer unlocked-GPU runs showed thermal/power drift larger than the
candidate effect.

The decisive same-binary exact-B13 runtime test held N128 fixed for B1-B12
warmup and changed only the B13 tactic:

| order | N128 common-wall | N96 common-wall | delta |
|---|---:|---:|---:|
| ABBA | 3992.606 | 4010.784 | +0.455% |
| BAAB | 3964.448 | 3996.038 | +0.797% |
| pooled | 3978.527 | 4003.411 | +0.625% |

All four adjacent comparisons in the same-binary gate are positive.  There is
no stream phase controller or phase offset; the benchmark uses the ordinary
eager launch path and leaves ordering to CUDA/GPU scheduling.

The final accepted Nsys capture reports:

- S2 combined: `4092.818 nn/s`
- S2 common wall: `4120.446 nn/s`
- S1 combined: `3411.300 nn/s`
- S1 common wall: `3406.455 nn/s`

For context, the prior Stage-47 accepted trace reported `4050.536` combined and
`4086.317` common-wall S2.  The cross-capture delta is supporting evidence only;
the same-binary symmetric test above is the acceptance authority.

## Accuracy and fallback

The stable B13 8,192-row replay passes the established both16 regime:

- policy top-1 versus FP32: `99.7681%`
- optimistic policy top-1 versus FP32: `99.7314%`
- policy probability RMSE versus FP32: `1.02121e-4`
- outcome RMSE versus FP32: `0.0023091`
- score all-6 RMSE versus FP32: `0.0073389`
- policy top-1 versus Stage 47: `99.7803%`

B12 default-on and N96-option-off replay outputs are byte-identical, confirming
that the automatic tactic is exact-B13 only.

## Implementation and final artifacts

The generated N96 artifact uses the distinct `fa4_n96` symbol prefix and is
reproducible with `fa4_aot/build_aot.py --tile-m 128 --tile-n 96
--num-stages 1 --symbol-prefix fa4_n96`.  Repository and reproduced header/object
SHA256 values match exactly.

Final evidence is under:

- `smoke/`, `microbench/`, `ncu/`
- `natural-s1-*`, `natural-s2-*`, `runtime-s2-*`
- `accuracy-full/`, `fallback-smoke/`
- `final-profile/nsys/`
- `final-profile/ncu/accepted-s1-full-forward.ncu-rep` (344 launches)

Stage 57 supersedes Stage 47 as the accepted graph/profile baseline.
