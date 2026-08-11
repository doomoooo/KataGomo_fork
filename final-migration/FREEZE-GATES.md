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
- handover
  `/workspace/results/docs/SM89_SM120_AUTOTUNE_HANDOVER_20260807.md`.

The immutable 8192-row FP32 golden was subsequently recovered and bound to the
model and corpus hashes. Candidate output is never promoted to its own
reference; both checked-in production plans passed replay against that golden.

## SM120 scanner gate — OPEN (frozen snapshot)

The user explicitly froze the SM120 implementation under the same bug-fix
allowance. Its immutable snapshot is
`refs/final-migration/frozen-sm120-working` at `335206cb`, merged by
`b45a24c`. The supported domain is SM120 exact B4-B32 with two streams and the
coordinate fat-binary workflow described in the same handover.

## Backend gate — OPEN (SM89 and SM120 certified)

The unified SM89/SM120 backend, exact B4-B32 generators, runtime activation
markers, and plan-apply mapping are now part of this delivery. Every
historically positive, numerically valid route is enforced by the four-link
closure contract in `python/cuda_tactic_history.py`.

An independent read-only comparison against the frozen SM89 and SM120 source
trees completed on 2026-08-08. A subsequent coupling audit merged overlapping
catalogs, restored missing joint candidates, and removed candidates that could
rewrite an earlier decision. The resulting full B4-B32 domain materializes
3,564 SM89 coordinates and 3,944 SM120 coordinates. Both architectures expose
19 implementation catalogs organized into 10 ownership-closed decision groups.
All 60/63 positive-history records close
the implementation, scan, activation, and plan-apply links.

SM89 is certified on the remote RTX 4090 D. A single-device B12/S2 replay
verified 8,192/8,192 results against the immutable FP32 reference at 3,035.87
physical `nnEval/s`. A two-device, four-lane replay verified 8,192/8,192 at
6,072.97 physical `nnEval/s`; every lane processed work and no external process
used SM time during that measurement. A later 20-pass correctness stress
verified 163,840/163,840 results, but its throughput is explicitly invalid
because an external job began using the second GPU during the run.

The post-audit RTX 5080 B16/S2 production plan passed its stable long gate at
`2836.211933` physical `nnEval/s` and then passed the 8,192-row FP32 replay.
The RTX 4090 D B12/S2 production plan passed at `3110.690824` physical
`nnEval/s` and passed the same replay contract. Both plans are checked in under
`final-migration/plans/`; the production loader refuses a different GPU
fingerprint, model, shape, batch, stream topology, or tactic mapping.

## Frontend gate — OPEN (SM89 and SM120 integrated and certified)

The previously exercised scheduler is ported. It implements full-batch or
device-idle gathering, fixed physical batch padding, independent persistent
submission workers, pinned H2D/D2H, single-slot input/output-consumed event
handshakes, and receiver-local multi-GPU routing. The SM89 single- and dual-GPU
certificates exercise this path. A normal RTX 5080 GTP startup also loaded the
production plan on both lanes, observed every selected post-launch marker, and
started the event-gated single-slot scheduler without fallback.
