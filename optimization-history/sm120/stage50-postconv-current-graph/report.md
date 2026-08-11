# Stage 50 result: retained postConv fusion still rejected on Stage 47

Target: RTX 5090 D, exact 19x19 B13, FP16/NHWC, natural S2.  The binary was
rebuilt cleanly from commit `acf588c` before testing so that no Stage 49
experimental object remained linked.

The unchanged Stage 45 implementation was enabled with only
`cudaUsePostConvBNSiluSm120=true`.  Existing correctness and NCU evidence was
reused: byte-identical full replay, 108 versus 154 registers/thread, 50.176
versus 73.728 KiB shared memory, zero spills, and 11 affine-SiLU launches plus
11 C768 rereads removed per forward.

Natural-S2 short ABBA plus reverse BAAB, 100 timed and 20 warmup iterations per
arm:

| order | control (nnEval/s) | candidate (nnEval/s) | delta |
|---|---:|---:|---:|
| forward ABBA | 3963.518 | 3937.907 | -0.646% |
| reverse BAAB | 3947.298 | 3949.461 | +0.055% |
| pooled | 3955.408 | 3943.684 | -0.296% |

Decision: retain the correct S1/resource-positive code default-off, but reject
deployment on the current graph.  The order-sign conflict and negative pooled
mean fail the preregistered gate.  No long confirmation, accuracy rerun, or
fresh full-graph profile is warranted; Stage 47 remains the accepted profile.
