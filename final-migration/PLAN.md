# Final migration plan

## Invariants

1. Start from the latest fetched official `master`, not from an optimization
   branch.
2. Never mutate an active optimization worktree or results directory.
3. Preserve correctness before performance. Optimized results must be checked
   against the existing offline FP32 reference corpus once its canonical
   location and checksum are recorded.
4. A plan is data, not implicit compile-time state. The backend must reject an
   unsupported or incompatible component instead of silently falling back.
5. All global plans, reminders, inventories, and audit records live below this
   directory.

## Phase 1: reproducible environment (implemented)

Deliverables:

- one fresh-Ubuntu entry point;
- exact system/Python/resolved-source dependency inventory;
- latest-source local builds for optimizer components, with local bundle
  seeding and explicit GitHub/proxy behavior;
- a hash-checked, non-invasive tar distribution for future target machines;
- compile smokes for CUDA, cuBLAS, cuDNN frontend, CUTLASS/CuTe, Triton,
  TileLang, and FlashAttention Python/CuTe imports;
- a clean KataGo CUDA-backend build;
- machine-readable and human-readable audit records.

Acceptance:

- rerunning setup is idempotent;
- no dependency is satisfied by an unexplained ambient path;
- a new driver installation is reported as requiring reboot;
- the CUDA build completes and its executable answers `version`;
- runtime library paths show which system and Python CUDA stacks are in use;
- a packaged deployment can extract the CUDA backend and its user-space runtime
  into one empty prefix without APT or system writes; archived source-built
  wheels can optionally be installed offline when the Python ABI matches.

The current host passed these acceptance checks. A true blank-host run remains
a release qualification whenever a new distribution is cut, because the
development path intentionally resolves then-current upstream sources and
NVIDIA packages.

## Phase 2: plan scanner (frozen)

Do not implement this phase until the owning develop session explicitly freezes
the 4090 scanner scripts and publishes the freeze tuple listed in
`FREEZE-GATES.md`. SM120 observations may be inventoried but are not an API.

After unfreeze, normalize the scanner output into a versioned plan schema with:

- device identity and compatibility constraints;
- batch/shape domain;
- selected implementation for every backend component;
- artifact hashes and source revisions;
- workspace and launch requirements;
- validation metrics and explicit fallback policy.

## Phase 3: plan-driven CUDA backend

Reconstruct behavior from history, tests, reports, and frozen plans. Remove
experimentation scaffolding and consolidate resource ownership. The maintained
backend must cover every component representable by the plan schema and make
stream/event/buffer ownership explicit.

Required properties include externally owned inference streams, asynchronous
pinned H2D/D2H, input-consumed and output-consumed event handshakes, safe
single-slot reuse where supported, CUDA Graph compatibility boundaries, and
multi-GPU isolation. Unsupported plan content is a startup error, not a hidden
fallback.

## Phase 4: frontend integration

Port the already exercised scheduler changes after the backend contract is
stable. Preserve batch-aware dispatch, fixed/exact batch semantics, independent
stream submission, multi-GPU request routing, and correctness guardrails.

The throughput metric for padded fixed batches is inferred evaluations times
the launched batch size. Search visits must remain strictly greater than real
neural evaluations and must not be conflated with padded inference work.

## Phase 5: archive

Organize the final retained material by architecture, component, stage, and
status. Store manifests and hashes in Git; store large payloads in the persistent
data area and reference them from the manifest. Never archive build caches or
known-dead experimental output merely because it exists.
