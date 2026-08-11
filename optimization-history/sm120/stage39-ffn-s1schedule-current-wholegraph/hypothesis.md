# H39a: Re-evaluate the existing three-stage FFN schedule in the current whole graph

## Frozen protocol

- RTX 5090 D, fixed B13, 19x19, FP16/NHWC, two real NN server streams.
- Control is the clean `5587388` build and accepted target configuration.
- Candidate changes only `cudaUseFusedFFNSingleStreamSchedule=true`.
- No homogeneous-S2 or synthetic mixed-peer result is collected or used.
- Short real-graph order is A-B-B-A / B-A-A-B, 400 timed iterations and 25 warmups.

## Evidence and mechanism

The existing three-stage FFN kernel improved the historical fixed-B13 S1 whole graph
from `3047.568` to `3089.095 nnEval/s` in the first A/B leg (`+1.363%`). It uses an
extra pipeline stage (`49,152` versus `32,768` dynamic shared-memory bytes), so the
current S2 outcome cannot be inferred from S1. Since the accepted graph has since
changed through A-fragment reuse, persisting-L2, and half2 RoPE, directly re-test the
actual graph rather than preserving the old topology assumption.

## Falsifiable prediction and stop rule

If the serial efficiency survives current S2 scheduling, the candidate should improve
the short whole graph with at least three of four adjacent pairs positive. Any clear
negative mean or three negative pairs rejects it immediately. Only a positive result
may proceed to NCU/long/accuracy.
