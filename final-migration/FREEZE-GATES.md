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

## Backend gate — OPEN (SM89 certified; SM120 plan pending)

The unified SM89/SM120 backend, exact B4-B32 generators, runtime activation
markers, and plan-apply mapping are now part of this delivery. Every
historically positive, numerically valid route is enforced by the four-link
closure contract in `python/cuda_tactic_history.py`. The remaining architecture
gate is a production plan and serial 8,192-row full-FP32 hardware certificate
on SM120.

An independent read-only comparison against the frozen SM89 and SM120 source
trees completed on 2026-08-08 with no remaining four-link blocker. It
materialized 3,738 SM89 coordinates across 20 families and 4,234 SM120
coordinates across 23 families for exact B4-B32; all 62/64 positive-history
records closed.

SM89 is certified on the remote RTX 4090 D. A single-device B12/S2 replay
verified 8,192/8,192 results against the immutable FP32 reference at 3,035.87
physical `nnEval/s`. A two-device, four-lane replay verified 8,192/8,192 at
6,072.97 physical `nnEval/s`; every lane processed work and no external process
used SM time during that measurement. A later 20-pass correctness stress
verified 163,840/163,840 results, but its throughput is explicitly invalid
because an external job began using the second GPU during the run.

The existing RTX 5080 long gate predates the final positive-history union and
is not eligible for migration: it is the known 2,586.58 `nnEval/s` B18 result
from the incomplete search space. SM120 runtime certification must use a new
unified-scan `best-tactic-plan.json`; the production loader correctly refuses
the old non-production plan.

## Frontend gate — OPEN (SM89 integrated and certified)

The previously exercised scheduler is ported. It implements full-batch or
device-idle gathering, fixed physical batch padding, independent persistent
submission workers, pinned H2D/D2H, single-slot input/output-consumed event
handshakes, and receiver-local multi-GPU routing. The SM89 single- and dual-GPU
certificates above exercise this path. SM120 remains gated by its new
production plan rather than by an implicit fallback or synthetic plan.
