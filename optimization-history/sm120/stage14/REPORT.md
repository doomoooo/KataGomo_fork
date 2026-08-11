# Stage 14: B13-shared fused Q/K RoPE

Status: rejected (2026-08-05 UTC).

The candidate changed the fixed B13 fused-QK RoPE grid from `(361,13)` to
`(361,1)`, computing each FP32 `sincos` once and unrolling batch inside the
thread. Nsys confirmed the intended CTA reduction. However, the B13 direct
kernel total increased from 17.328 ms to 23.614 ms (+36.3%). The S2 profile
showed only 3486.112 to 3500.349 nnEval/s (+0.41%).

Short A-B-B-A rejected the candidate:

| topology | control mean | shared mean | change |
|---|---:|---:|---:|
| B13/S1 | 2935.799 | 2910.726 | -0.854% |
| B13/S2 | 3489.311 | 3489.137 | -0.005% |

The reduced grid did not compensate for longer CTAs, and the apparent profiled
S2 gain did not reproduce outside the profiler. Per the predeclared early-stop
rule, full replay and long testing were not run. The switch remains disabled.
Reopen only if batch sharing is combined with a materially faster inner loop
without reducing grid parallelism further.
