# Stage 10: one-warp C384 RMSNorm

Status: H10 fast-tree rejected; H10b exact-tree accepted for S1 and S2
(2026-08-05 UTC).

The fixed 19x19 FP16 C384 path now assigns one warp to each token row and four
rows to each 128-thread block. H10b preserves the official arithmetic tree:
six independent 64-channel group reductions are followed by the same
six-group reduction. This removes shared memory and two block barriers without
changing the output. Unsupported shapes, channel counts, masks, and precisions
retain the official path.

## H10 fast-tree rejection

The first one-warp implementation combined all six groups in each lane before
the warp reduction. Short A-B-B-A improved S1 by about 2.68% and S2 by about
3.07%, but the full 8,192-row accuracy run missed the predeclared optimistic
policy top-1 gate by one row: 99.5972% versus the required 99.60%. It is
rejected rather than rounded up. The binary and accuracy artifacts remain in
`bin/katago-rms-fasttree` and `accuracy/`.

## H10b exact-tree evidence

Short A-B-B-A on the accepted fused-residual baseline reported:

| topology | control mean | H10b mean | change |
|---|---:|---:|---:|
| B13/S1 | 2750.968 | 2813.639 | +2.278% |
| B13/S2 | 3248.933 | 3316.692 | +2.086% |

The full replay is byte-identical to the accepted Stage9 replay. Both files
have SHA256
`9ead43c9e5567242defbba4b7b45110ce2b802c39146ee8f9c23a1a4863c3d62`.
The direct full-FP32 comparison therefore returns to the accepted numerical
class, including policy top-1 99.7437% and optimistic-policy top-1 99.7314%.

S2 Nsys reported the same 8,844 RMSNorm launches in both traces. Their direct
kernel time changed from 47.876 ms to 34.027 ms, a 28.93% reduction. The traced
whole-network result was 3250.059 to 3329.683 nnEval/s (+2.45%).

The 1,000-iteration symmetric S2 A-B-B-A/B-A-A-B result was:

| mode | values (nnEval/s) | mean | median |
|---|---|---:|---:|
| control | 3249.831 / 3231.885 / 3216.564 / 3210.228 | 3227.127 | 3224.225 |
| H10b | 3312.217 / 3302.651 / 3293.868 / 3280.158 | 3297.223 | 3298.259 |

Mean improvement is 2.172%; median improvement is 2.296%. The first symmetric
block improves 2.054% and the reversed second block improves 2.291%. Accept
and enable H10b by default on the gated SM120 fixed-19 path.
