# SM120 RTX 5080 B19/S2 plan

- file: `best-tactic-plan.json`
- file SHA-256: `5f90e7fb5c02ac147e4cf535e664dca736f4fcbd9c0afd188ef5a5fd1e7b788b`
- semantic plan SHA-256: `f3776f13058f2c14de741aee9f984e9a078483435960b235bd7c74500d813b20`
- model SHA-256: `1881600caab9e9d85a3dd6a019e9b8e7d2c237b5f984e13ed49a8645be3077c6`
- target: SM120, RTX 5080, exact 19x19, FP16/NHWC, B19, two streams
  per device
- long gate: `2763.4413825` physical nnEval/s, 1000 iterations, 50 warmup,
  two samples, relative spread `0.0115909442`
- accuracy: 8192-row all-head replay against the immutable full-FP32 output,
  passed all seven recorded checks
- positive-history closure: 64/64 records, all four links present across B4-B32

The correctness metrics include policy top-1 agreement `0.9981689453`, policy
probability RMSE `0.0001017439`, value outcome RMSE `0.0021735888`, score mean
RMSE `0.0017378022`, and ownership sigmoid RMSE `0.0002451993`.

Use an absolute path in the GTP config:

```cfg
cudaTacticPlanFile = /absolute/path/to/best-tactic-plan.json
cudaTacticPlanBatch = 19
```

The scan-time CUDA ordinal and absolute evidence paths retained inside the JSON
are provenance only. Runtime device assignment remains local to the GTP config.
This plan has passed the unified whole-graph and FP32 accuracy gates; a separate
ordinary GTP-path qualification result has not yet been recorded for RTX 5080.
