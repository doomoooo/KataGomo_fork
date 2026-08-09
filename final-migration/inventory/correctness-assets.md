# Correctness assets

Captured read-only on 2026-08-07. The input corpus remains persistent, but the
historical `.krnn` files were deliberately removed during the later workspace
cleanup. The golden row below records its former immutable identity; it is not
an assertion that the file is still available.

## Canonical offline FP32 pair

| Role | Persistent path | Size | SHA-256 |
| --- | --- | ---: | --- |
| 8,192 prepared 19x19 inputs | `/workspace/trainingdata/accuracy/2026-08-01-19x19-8192-seed20260803-full19.npz` | 151 MiB | `0b2f2838df51ff98847f5bf595f9670350e993c5e178a92855c21e80e75762c5` |
| full-FP32 raw output golden (removed after comparison) | `/workspace/trainingdata/accuracy/replay-fixed-fp32-full19.krnn` | 425 MiB | `4603a430367f5dd67c02ba02b71692f9a17ce24acae08d354152dd8376f229b3` |

The baseline report identifies the golden as CUDA full-FP32 from revision
`847e78a44d402df200eabbbe0506776a45607c29-dirty-cuda`. It must remain an
external immutable reference: a candidate backend must never regenerate its
own expected result.

Availability was rechecked on 2026-08-08 under `/workspace`, `/data`, both
validation hosts' persistent data roots, and `/mnt/CacheSSD`; no retained copy
was found. Consequently scan plans must retain `production_ready=false` until
a byte-identical file with the recorded SHA-256 is supplied again.

## Existing GTP-shaped guard

The implementation currently lives in the externally owned event-pipeline
worktree:

- `cpp/tests/testnngtpharness.cpp`: prepared-corpus loader, KRNN golden loader,
  request producer, asynchronous verifier, fail-fast behavior, and per-slot
  accounting;
- `cpp/command/runtests.cpp`: `runnngtpstresstest` command wiring;
- `docs/rtx4090-gtp-integration-report.md`: accepted single/dual-card evidence.

It exercises the evaluator's actual request queue and event scheduler. Inputs
are unpacked once before timing, request threads run continuously without a
per-pass barrier, and an unbounded CPU verifier sidecar checks results. The
default producer concurrency is:

```text
batchSize * (inferenceSlots + 1) + 32
```

The verifier atomically stops new submission on the first mismatch and reports
pass, corpus row, head, and element. Comparisons are after softmax for policy
and value, after sigmoid for ownership, and raw for the six score outputs.
Current per-result gates are:

| Head | max absolute | RMSE |
| --- | ---: | ---: |
| policy probability | 0.025 | 0.002 |
| value probability | 0.05 | 0.04 |
| score raw | 0.60 | 0.30 |
| ownership probability | 0.025 | 0.006 |

After completion it also asserts total evaluator rows, unchanged slot
topology, nonzero work on every inference slot, and equality between summed
per-slot rows and evaluator rows.

## Existing evidence, not yet migrated

The report records:

- single 4090, B13/S2: all 8,192 rows passed at 3,451.41 requests/s and
  3,461.52 physical aligned nnEval/s, exactly 4,096 real rows per slot;
- dual 4090, B13/S4: all rows passed at 6,968.23 requests/s and 6,988.65
  aligned nnEval/s, all four slots exercised;
- dual-card long run: 100 corpus repetitions, 819,200 results passed in
  120.92 seconds, 6,774.85 requests/s and 6,828.35 aligned nnEval/s, maximum
  verifier backlog 32.

Those GPU runs used the development-only external coordination wrapper. They prove the prior
behavior, not the current official-master migration. The harness must later be
ported without replacing the evaluator under test, and the same file hashes
must be checked before a final correctness run.
