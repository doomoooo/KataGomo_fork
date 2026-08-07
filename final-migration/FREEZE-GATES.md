# Freeze gates

## 4090 scanner gate — CLOSED

Phase 2 must not begin until the scanner-owning develop session provides all of:

- frozen Git commit;
- frozen plan schema version or representative output files;
- exact supported batch/shape domain;
- artifact list with SHA-256 hashes;
- canonical correctness command and FP32 reference identity;
- canonical performance command, warmup, duration, clocks/power policy, and
  expected range;
- known failures and intentionally unsupported components;
- an explicit statement that the 4090 scanner files will no longer change
  underneath migration.

Observing a recent commit, report, or generated plan does not open this gate.

## SM120 scanner gate — OBSERVATION ONLY

SM120 work is close to completion but is still externally owned. Its history and
generated outputs may be mapped read-only. Integration begins only when its own
freeze tuple is supplied.

## Backend gate — CLOSED

Backend cleanup begins after at least one frozen scanner plan can be validated
end-to-end and the plan schema boundary is agreed. This prevents refactoring
against transient experiment scaffolding.

## Frontend gate — CLOSED

Frontend scheduling changes are ported only after the plan-driven backend owns
its stream/event/buffer contract and has a multi-GPU correctness harness.
