# Versioned CUDA tactic plans

This directory contains compact production plans that are useful as immutable
runtime and regression assets. A plan is admitted only when it is
`production_ready=true`, has complete positive-history closure, passed its
long whole-graph gate, and passed an immutable full-FP32 correctness gate.

Plans are hardware-, model-, precision-, board-, batch-, and stream-specific.
They are not universal defaults. The runtime validates the receiver and fails
at startup on a mismatch. The CUDA device ordinal recorded at scan time is
provenance only and is not applied on the receiver.

Current assets:

- `sm89/rtx4090d-b12-s2/best-tactic-plan.json`: certified RTX 4090 D, exact
  B12, two streams per device.

No SM120 plan is checked in yet. The pre-unification RTX 5080 plan is not
production-ready and is intentionally excluded.
