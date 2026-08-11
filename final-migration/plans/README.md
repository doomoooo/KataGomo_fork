# Versioned CUDA tactic plans

This directory contains compact production plans that are useful as runtime
and regression assets. The registry keeps exactly one current production plan
per exact producer GPU fingerprint. `target.gpu_class` selects an implementation
space; the recorded CUDA product name and numeric capabilities identify the
qualified SKU. Replacing a model's plan updates that one entry; superseded
plans remain available through Git history rather than as parallel runtime
choices. A plan is admitted only when it is
`production_ready=true`, has complete positive-history closure, passed its
long whole-graph gate, and passed an immutable full-FP32 correctness gate.

Plans are hardware-, model-, precision-, board-, batch-, and stream-specific.
They are not universal defaults. The runtime validates the receiver and fails
at startup on a product-name, global-memory, SM/L2/memory-bus, execution-limit,
or other capability mismatch. The CUDA device ordinal recorded at scan time is
provenance only and is not applied on the receiver.

Production plans contain no scan-host absolute paths. Full commands and
environment snapshots stay in content-addressed scan records; plans retain
only portable identifiers, hashes, measurement/correctness summaries, and the
runtime apply mapping.

Current assets:

- `sm89/rtx4090d-b12-s2/best-tactic-plan.json`: certified RTX 4090 D, exact
  B12, two streams per device, `3110.690824` long-gate physical nnEval/s.
- `sm120/rtx5080-b16-s2/best-tactic-plan.json`: certified RTX 5080, exact B16,
  two streams per device, `2836.211933` long-gate physical nnEval/s.
