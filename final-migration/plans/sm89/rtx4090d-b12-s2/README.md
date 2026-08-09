# SM89 RTX 4090 D B12/S2 plan

- file: `best-tactic-plan.json`
- file SHA-256: `57aba0d9f5ff009f0103fe792766bd3fe065d156c13396cb99bc40b5488f9edb`
- semantic plan SHA-256: `1a068fd146ad0776fb8be1ea69bea4eafa501d45ac481c78e3853d60d26bc0f5`
- model SHA-256: `1881600caab9e9d85a3dd6a019e9b8e7d2c237b5f984e13ed49a8645be3077c6`
- target: SM89, RTX 4090 class, exact 19x19, FP16/NHWC, B12, two streams
  per device
- long gate: `3026.196859` physical nnEval/s, 1000 iterations, 50 warmup,
  two samples, relative spread `0.0014528414`
- accuracy: 8192-row all-head replay against the immutable full-FP32 output,
  passed all recorded thresholds
- positive-history closure: 62/62 records, all four links present

The same plan subsequently passed the ordinary GTP-shaped scheduler harness on
one RTX 4090 D at 3035.87 physical nnEval/s and two RTX 4090 D devices at
6072.97 physical nnEval/s. See
`final-migration/records/plan-runtime-sm89-20260809.md` for the certificate and
contention caveats.

Use an absolute path in the GTP config:

```cfg
cudaTacticPlanFile = /absolute/path/to/best-tactic-plan.json
cudaTacticPlanBatch = 12
```

The scan-time CUDA ordinal and absolute evidence paths retained inside the JSON
are provenance only. Runtime device assignment remains local to the GTP config.
