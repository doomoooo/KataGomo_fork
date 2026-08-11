# Stage 51 result: retain exact-B13 mask elision default-off

Target: RTX 5090 D, exact 19x19 B13, FP16/NHWC.  Control is commit
`acf588c`, the accepted Stage 47 graph.

The candidate allocates one persistent 13-element FP32 vector containing
`361.0f` per compute handle.  Only an exact runtime B13 forward uses it as
`maskSum`; that path skips channel-0 extraction, half-to-float conversion, and
the channel-sum reduction.  B1--B12 and every non-target shape, precision,
layout, or GPU retain the original path.  The scratch allocation sequence is
unchanged.

## Mechanism and correctness

- 26-row B13 candidate/control replay: byte-identical.
- 26-row B12 fallback candidate/control replay: byte-identical; no
  specialization log.
- Short S1 Nsys: each of the three target kernel counts fell from 28 to 24.
  The difference is exactly the four B13 forwards in the trace; the 24
  retained calls are the deliberate B1--B12 warmup fallback.  No replacement
  kernel or synchronization was added.
- Fixed-B13 full replay, one server, 8,192 rows: candidate and control are
  byte-identical, SHA256
  `3a7329716ccef8cababb2ca71ebe0734ddd6b9f6f1a90411d3035b4dac7fe7ff`.

An ordinary two-server replay was not byte-identical across repeated control
runs: only 6--11 output rows changed while corpus/target sections stayed
identical.  The dynamic replay queue occasionally executes non-B13 fallback
batches, whose algorithm selection depends on host timing.  This is not a
valid bit-identity oracle for the fixed-B13 target.  The stable one-server B13
replay above is therefore the arithmetic gate; natural S2 remains the
throughput gate.

## Whole-forward throughput

Natural S2, short 100/20 ABBA plus reverse BAAB:

| order | control (nnEval/s) | candidate (nnEval/s) | delta |
|---|---:|---:|---:|
| forward | 3955.258 | 3965.985 | +0.271% |
| reverse | 3955.764 | 3972.932 | +0.434% |
| pooled | 3955.511 | 3969.458 | +0.353% |

Natural S2, 400/40 confirmation:

| order | control (nnEval/s) | candidate (nnEval/s) | delta |
|---|---:|---:|---:|
| forward | 3947.404 | 3965.064 | +0.447% |
| reverse | 3948.336 | 3938.482 | -0.250% |
| pooled | 3947.870 | 3951.773 | +0.099% |

S1, 400/40 confirmation:

| order | control (nnEval/s) | candidate (nnEval/s) | delta |
|---|---:|---:|---:|
| forward | 3339.755 | 3343.973 | +0.126% |
| reverse | 3330.307 | 3333.417 | +0.093% |
| pooled | 3335.031 | 3338.695 | +0.110% |

## Decision

Retain behind `cudaUseExactMaskElisionSm120=false`.  It is byte-exact for the
fixed target, strictly deletes three launches and their memory traffic, and is
positive in both S1 orders.  It is not promoted to the default S2 graph because
the longer S2 order aggregates conflict.  The accepted/default graph therefore
remains Stage 47, and its existing full S2 Nsys plus 344-ordinal S1 NCU profile
continues to select the next target; generating a nominally new full profile
with this switch disabled would measure the identical graph.
