# Dual RTX 4090 event-pipeline fix validation

Date: 2026-08-06

## Failure and fix

The two-GPU event-gated GTP topology originally failed in the asynchronous
submit path with `CUBLAS_STATUS_INTERNAL_ERROR`. CUDA's current device is
host-thread-local, but per-slot submission workers could issue work for GPU 1
while retaining GPU 0 as their current device.

The fix makes the opaque CUDA compute stream device-aware, records the owning
device on every compute handle, and restores that device at every asynchronous
pipeline backend entry point and before resource destruction. It does not
change the accepted SM89 kernels, stream topology, fixed-B13 policy, or the
single-slot event protocol.

## Correctness

- The minimal asynchronous `[GPU0, GPU1]` reproducer completed without a CUDA
  or cuBLAS error.
- The deployment topology `[GPU0, GPU0, GPU1, GPU1]` completed real GTP.
- GPU 0 versus GPU 1 replay output is bit-identical for all compared policy,
  value, score, and ownership fields.
- GPU 0 versus the historical accepted eager replay output is also
  bit-identical.
- `katago runtests` passed.
- `katago runnnlayertests` tested and passed 28 configurations on RTX 4090.

## Request supply and throughput

All throughput below uses physical fixed-B13 work (`infer launches * 13`) for
the aligned comparison. GPU application clock locks were reset and all runs
used `gpu-lock` for the two physical RTX 4090s.

The four-slot synthetic event-pipeline ceiling measured 6721.803 physical
nnEval/s. The real GTP thread sweep measured:

| Search threads | visits/s | real nnEval/s | physical nnEval/s | vs. synthetic sample |
|---:|---:|---:|---:|---:|
| 112 | 9054.48 | 6685.43 | 6790.68 | +1.03% |
| 120 | 8862.99 | 6707.42 | 6816.55 | +1.41% |
| 124 | 8802.79 | 6717.95 | 6815.90 | +1.40% |

The physical column is computed directly from `nnBatches/s * 13`; the
displayed real `nnEval/s` excludes padded rows. The approximately 1% crossing
of the earlier synthetic sample is normal boost and measurement phase
variance. More importantly, 120 and 124 threads give the same physical
throughput to 0.01%, so request production is sufficient and the backend is
the limiting stage. `visits/s` remains strictly above real `nnEval/s`, as
expected because not every visit produces a new neural-network evaluation.

Single-GPU real GTP measured 3398.93 real nnEval/s and 263.64 batches/s, or
3427.32 physical fixed-B13 nnEval/s. This does not regress the prior Stage 71
approximately-3397 physical nnEval/s mean.

## Nsight Systems timeline

The dual-card q1 trace contains two compute streams per device:

| CUDA device | compute streams | trace span (ms) | union kernel busy (ms) | union busy |
|---:|---|---:|---:|---:|
| 0 | 164, 166 | 918.189 | 910.720 | 99.19% |
| 1 | 165, 167 | 881.724 | 876.345 | 99.39% |

H2D and D2H appear on separate copy streams for every lane (device 0 streams
296/298 and 299/301; device 1 streams 297/300 and 302/303), rather than on the
compute streams. The trace therefore confirms the intended two-compute-stream
per GPU topology and copy-engine separation after the device-binding fix.

Artifacts include `gpu0-vs-historical.json`, `gpu1-vs-gpu0.json`, the
`supply/` sweep, and `nsys/dual-event-q1.nsys-rep` plus its SQLite export.
