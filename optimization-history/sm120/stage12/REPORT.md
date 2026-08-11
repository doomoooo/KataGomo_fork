# Stage 12: fixed-C1152 contiguous half8 SwiGLU

Status: accepted for S1 and S2 (2026-08-05 UTC).

The fixed 19x19 FP16 C1152 path now keeps the same eight scalar FP32 SiLU
operations per thread but loads each input and stores the output as contiguous
128-bit half8 vectors. The fixed element count removes the four per-step bounds
predicates in the official EPT4 half2 kernel. Unsupported shapes, channel
counts, and precisions retain the official path.

Baseline NCU on B13 reported 16.32 us, 75.37% DRAM throughput, 44.17% SM
throughput, 84.71% achieved occupancy, 19 registers/thread, and long-scoreboard
stalls accounting for about 63.8% of issue distance. This supported changing
the access instruction shape without increasing per-thread arithmetic.

S2 Nsys kept the same 4,422 launches and reduced direct SwiGLU kernel time from
45.461 ms to 39.445 ms (-13.23%). The traced whole-network result was 3390.004
to 3427.909 nnEval/s (+1.12%).

Short A-B-B-A reported:

| topology | control mean | half8 mean | change |
|---|---:|---:|---:|
| B13/S1 | 2873.650 | 2895.901 | +0.774% |
| B13/S2 | 3403.576 | 3437.997 | +1.011% |

The full 8,192-row replay is byte-identical to Stage11, with SHA256
`9ead43c9e5567242defbba4b7b45110ce2b802c39146ee8f9c23a1a4863c3d62`.
All direct full-FP32 gates pass.

The 1,000-iteration symmetric S2 A-B-B-A/B-A-A-B result was:

| mode | values (nnEval/s) | mean | median |
|---|---|---:|---:|
| control | 3420.610 / 3384.356 / 3376.988 / 3372.933 | 3388.722 | 3380.672 |
| half8 | 3449.528 / 3422.793 / 3415.767 / 3411.350 | 3424.859 | 3419.280 |

Mean improvement is 1.066%; median improvement is 1.142%. The first symmetric
block improves 0.990% and the reversed second block improves 1.144%. Accept
and enable by default on the gated SM120 fixed-19 path.
