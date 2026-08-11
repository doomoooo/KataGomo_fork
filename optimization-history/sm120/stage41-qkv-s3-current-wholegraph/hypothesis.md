# H41: Re-open the lower-smem wide-QKV S3 schedule on the current whole graph

## Frozen protocol

- RTX 5090 D, fixed B13/19x19, FP16/NHWC, two real NN server streams.
- Control is clean `5587388` plus the current accepted target configuration.
- Candidate changes only `cudaUseWideQKVSingleStreamSchedule=true`, selecting
  the existing M128N128K32/S3 fixed-B13 QKV kernel.
- No homogeneous or synthetic mixed S2 measurement is collected or used.
- Short whole-graph order: A-B-B-A / B-A-A-B, 400 timed iterations, 25 warmups.

## Evidence and mechanism

Historical real S1 whole-graph ABBA improved `3115.039 -> 3155.965 nnEval/s`
(`+1.314%`), while direct QKV improved `18.840 -> 17.128 us`. Compared with
the accepted K64/S2 kernel, dynamic shared memory falls from 65,536 to 49,152
bytes and the shared-memory residency limit rises from one to two CTA/SM;
registers rise from 136 to 142 without local stack or spills. The route was
previously stopped only by a homogeneous local-S2 proxy, which no longer has
decision weight.

## Prediction and stop rule

The lower shared-memory footprint may improve current natural-S2 scheduling
without sacrificing serial efficiency. Continue only if the real short graph
is directionally positive with at least three of four adjacent comparisons
positive. A coherent negative result stops immediately; a positive result
proceeds to long S2, full replay, and fresh full-graph Nsys/NCU.
