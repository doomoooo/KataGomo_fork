# Versioned CUDA tactic plans

This directory contains compact production plans that are useful as runtime
and regression assets. The registry keeps exactly one current production plan
per normalized GPU model (`target.gpu_class`). Replacing a model's plan updates
that one entry; superseded plans remain available through Git history rather
than as parallel runtime choices. A plan is admitted only when it is
`production_ready=true`, has complete positive-history closure, passed its
long whole-graph gate, and passed an immutable full-FP32 correctness gate.

Plans are hardware-, model-, precision-, board-, batch-, and stream-specific.
They are not universal defaults. The runtime validates the receiver and fails
at startup on a mismatch. The CUDA device ordinal recorded at scan time is
provenance only and is not applied on the receiver.

Current assets:

- `sm89/rtx4090d-b12-s2/best-tactic-plan.json`: certified RTX 4090 D, exact
  B12, two streams per device, `3110.690824` long-gate physical nnEval/s.

The previous SM120 plan referenced a search component removed by the fixed
full-board backend contract, so it remains absent rather than being presented
as compatible. RTX 5080 B19/S2 will return only after it passes the current
long whole-graph and immutable full-FP32 gates.
