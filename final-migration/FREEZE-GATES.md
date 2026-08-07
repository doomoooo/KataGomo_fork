# Freeze gates

## SM89 scanner gate — OPEN (frozen snapshot)

The user explicitly froze the SM89 implementation for migration on 2026-08-07,
while allowing migration to fix discovered bugs. The immutable integration
snapshot is:

- Git ref `refs/final-migration/frozen-sm89-working`;
- commit `fd4d452c` (the owning worktree's committed and uncommitted state);
- integration merge `89c45b6`;
- supported domain SM89, exact B4-B32, two streams;
- discovery 100/50/1 and long gate at least 1000/50/2;
- handover `/workspace/SM89_SM120_AUTOTUNE_HANDOVER_20260807.md`.

The immutable 8192-row FP32 golden itself is no longer present. This does not
block scanning or long-gate plans, but it does block `production_ready=true`
until a separately identified golden is attached. Candidate output must never
be promoted to its own reference.

## SM120 scanner gate — OPEN (frozen snapshot)

The user explicitly froze the SM120 implementation under the same bug-fix
allowance. Its immutable snapshot is
`refs/final-migration/frozen-sm120-working` at `335206cb`, merged by
`b45a24c`. The supported domain is SM120 exact B4-B32 with two streams and the
coordinate fat-binary workflow described in the same handover.

## Backend gate — OUTSIDE THIS DELIVERY

The current delivery packages the frozen optimizer and generates plans. General
backend cleanup begins only after at least one resulting plan is validated
end-to-end and its schema boundary is accepted.

## Frontend gate — CLOSED

Frontend scheduling changes are ported only after the plan-driven backend owns
its stream/event/buffer contract and has a multi-GPU correctness harness.
