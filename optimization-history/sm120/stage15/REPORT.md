# Stage 15: fused Q/K RoPE half2 I/O

Status: rejected below whole-network threshold (2026-08-05 UTC).

The candidate kept the accepted `(361,batch)` grid and changed only each Q/K
rotary pair's two scalar half loads/stores to one half2 load/store. Nsys kept
4,422 launches and reduced direct RoPE time from 23.671 ms to 21.873 ms
(-7.60%). The traced whole-network result was 3485.251 to 3499.448 nnEval/s
(+0.41%).

Short A-B-B-A reported:

| topology | control mean | half2 mean | change |
|---|---:|---:|---:|
| B13/S1 | 2936.372 | 2945.223 | +0.301% |
| B13/S2 | 3490.137 | 3505.596 | +0.443% |

Both topology results were consistent but below the declared 0.5% threshold
for entering long validation. Full replay and long testing were not run. The
switch remains disabled. Reopen only as an independently ablated component of
a larger RoPE kernel change that clears the whole-network threshold.
