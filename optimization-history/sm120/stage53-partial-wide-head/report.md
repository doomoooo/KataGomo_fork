# Stage 53: partial C288 no-split g1+v1 head

Status: **REJECTED and source integration removed**.

Control was commit `3526b13` with all retained candidates disabled. The exact
target was RTX 5090 D, 19x19, B13, FP16/NHWC. The candidate packed policy g1
(C96) and value v1 (C192) weights into one C768->C288 cuBLAS Hgemm and let both
heads consume their slices with stride-aware affine-SiLU kernels. There was no
split or copy of the wide result. All other shapes and batches used the
original path.

## Correctness

- B13 26-row replay was byte-identical across every output head.
- B12 fallback replay was byte-identical.
- Candidate activation was observed only on exact B13.

## Mechanism evidence

Nsys showed the intended boundary exactly once per forward:

- Control g1/v1 GEMMs: about 6.0 us + 7.9 us, two launches.
- Candidate C288 GEMM: about 11.8 us, one launch.
- Both stride-aware affine-SiLU consumers had essentially the same duration as
  the original compact-input consumers; no split/copy kernel appeared.

Targeted NCU confirmed the same direction under replay profiling:

| Boundary | Duration | Registers/thread | Dynamic shared | Achieved occupancy |
|---|---:|---:|---:|---:|
| Control g1 C96 | 7.33 us | 80 | 98.30 KiB | 8.33% |
| Control v1 C192 | 9.60 us | 118 | 98.30 KiB | 8.33% |
| Candidate C288 | 14.66 us | 164 | 81.92 KiB | 8.32% |

Thus the local candidate removed one launch, one reread of the common trunk
input, and 2.27 us of isolated GEMM time. It did not improve occupancy, and
the realized end-to-end signal was too small to retain.

## Throughput gates

Natural S2 100/20 short screen:

- Forward ABBA: +0.1821%
- Reverse BAAB: +0.5939%
- Pooled: +0.3873%

Natural S2 400/40 confirmation rejected the candidate:

- Forward ABBA: -0.3826%
- Reverse BAAB: -0.0129%
- Pooled: -0.1982%

The conditional S1 400/40 retention gate also failed its both-order rule:

- Forward ABBA: -0.0023%
- Reverse BAAB: +0.1159%
- Pooled: +0.0567%

No accepted graph changed, so no full-graph Nsys/NCU was recaptured. Stage 47
remains the accepted full-profile basis for choosing the next target.

## Reopen condition

Reopen only if a fused epilogue or a different measured GEMM tactic saves
materially more than the current 2.27 us local delta without increasing the
wide tensor lifetime. Merely repeating the same cuBLAS grouping is below the
natural-S2 noise/co-scheduling threshold.
