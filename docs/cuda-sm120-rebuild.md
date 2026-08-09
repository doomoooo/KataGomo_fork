# CUDA SM120 backend

The rebuild is now maintained by the unified SM89/SM120 tactic workflow. The
authoritative usage document is `docs/cuda-tactic-workflow.md`; the
machine-readable positive-history contract is
`python/cuda_tactic_history.py`; and the reconciled history audit is
`final-migration/OPTIMIZATION_HISTORY_AUDIT_20260808.md`.

## Architecture

`ComputeHandle` builds the official `Model` once. On SM120 it also creates an
`Sm120Backend::Sm120Model`, which owns AOT registries, transformed/shared
weights, persisting-L2 state, and custom kernels. The official model supplies
the single forward traversal and calls thin operator hooks through
`CudaHandles`; selected SM120 tactics replace those boundaries without a
second model traversal or a second buffer layout.

The backend covers the retained QKV, RoPE, attention, FFN, residual projection,
normalization, pointwise, outer projection, mask, initial layer, and head
routes. Exact-batch generated artifacts are linked into an all-family fat
registry. A plan-explicit generated tactic must be present and execute; a
missing entry, unmet packed-path precondition, or launch failure is an error,
not a silent fallback.

## Verification contract

- Generate and scan every exact batch B4-B32; B13 and B19 have no privileged
  status.
- Compare padded throughput as physical `infer_calls * batch_size` nnEval/s.
- Require the exact activation markers for every selected family in both the
  family scan and final joint long gate.
- Bind generated source/object/metadata hashes and launch symbols to the linked
  binary through the artifact bundle.
- Certify the final plan against 8,192 rows of immutable full-FP32 reference
  outputs, checking every output head and stopping on the first failure.
- During every benchmark, record SM occupancy with `nvidia-smi pmon`; reject
  overlapping compute work while ignoring memory-only processes.

Environment creation, source generation, compilation, scan, long gate, and
plan construction are orchestrated by `final-migration/autotune/autotune.py`.
