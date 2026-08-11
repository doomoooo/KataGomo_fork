# Stage 21: fixed-B13 wide QKV AOT

Status: accepted for RTX 5090D B13/S2 (2026-08-06 UTC).

The accepted TileLang kernel is M128-N128-K64, two pipeline stages, 128
threads, and minimum three blocks per SM with FP16 MMA accumulation. Q/K/V
weights are packed once into a K384xN1152 matrix. The epilogue writes three
planar [4693,384] regions, so fused RoPE and FA4 need no layout conversion.

## Evidence

The corrected isolated two-stream boundary measured 30.12 us for the wide
kernel versus 47.19 us for three independent output GEMMs. The 1,000-iteration
A-B-B-A result was:

| mode | values (nnEval/s) | mean |
|---|---:|---:|
| control | 3678.206 / 3670.156 | 3674.181 |
| wide QKV | 3817.368 / 3810.685 | 3814.026 |

Mean improvement is 3.806%.

Nsys measured 24.21 us under the full mixed load, 136 registers/thread and
64 KiB dynamic shared memory. It replaces 8,514 Q/K/V GEMM launches with
2,838 wide launches in the trace. Main stream spans were 345.34 and 349.26 ms.

The full 8,192-row output metrics are identical to Stage 20 and pass every
accuracy gate. Raw artifacts are in this directory and `accuracy/`.
