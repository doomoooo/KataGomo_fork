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
  B12, two streams per device.
- `sm120/rtx5080-b19-s2/best-tactic-plan.json`: certified RTX 5080, exact B19,
  two streams per device, `2763.4413825` physical nnEval/s.

The pre-unification RTX 5080 plan is not production-ready and remains excluded.
