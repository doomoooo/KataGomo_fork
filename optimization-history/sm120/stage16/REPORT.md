# Stage 16: two-warp exact-tree C384 RMSNorm

Status: rejected by direct-kernel profiling (2026-08-05 UTC).

NCU on the accepted one-warp kernel reported 5.28 us at B13, 38
registers/thread, 100% theoretical occupancy, 49.08% achieved occupancy, 0.58
waves/SM, and 80.60% no-eligible cycles. H16 assigned two warps to each row
while preserving the accepted six-group XOR tree and all 32 lane-specific
scale values. The expected benefit was more active warps with half the live
input values per thread; the cost was two block barriers and four times as
many CTAs.

Nsys measured the opposite result over the same 8,844 RMSNorm launches:

| topology | one-warp total | two-warp total | change |
|---|---:|---:|---:|
| B13/S2 | 33.171 ms | 37.446 ms | 12.89% slower |

The traced whole-network result was 3466.752 to 3428.277 nnEval/s (-1.11%).
The candidate therefore stopped before short ABBA, full accuracy, and long
validation. The implementation remains behind
`cudaUseRMSNorm384TwoWarpSm120=false`; the accepted one-warp exact-tree kernel
remains the default.

The NCU launch limits also rule out a register-only follow-up: the accepted
kernel already has 100% theoretical occupancy, while its 0.58 waves are a
consequence of the fixed B13 row count. Reloading input to reduce registers
would add DRAM traffic without increasing the theoretical resident-warp
limit.
