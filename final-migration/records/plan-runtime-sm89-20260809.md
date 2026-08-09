# SM89 plan runtime validation, 2026-08-09

Target: `wangyize@10.101.3.169`, RTX 4090 D, CUDA-visible physical devices 0
and 5. Plan: `sm89-rtx4090-1a068fd146ad0776`, exact B12, two streams per
device, eager event pipeline. Plan file SHA-256:
`57aba0d9f5ff009f0103fe792766bd3fe065d156c13396cb99bc40b5488f9edb`.

The production GTP path loaded the plan, activated its exact tactics, and
returned a legal `genmove`. Long search-shaped measurements used physical
`nnEval/s = launched_batches * 12 / wall_seconds`; visits were kept separate.
CUDA Graph was functional but slower than eager submission on this target.

| topology | requests verified | physical nnEval/s | result |
| --- | ---: | ---: | --- |
| one device, two lanes | 8,192 | 3,035.87 | pass |
| two devices, four lanes | 8,192 | 6,072.97 | pass |
| two devices, four lanes, 20 passes | 163,840 | invalid | correctness pass; external GPU5 SM work appeared |

The valid dual-device run distributed rows 2047/2052/2041/2052 across the four
lanes. Low-frequency `nvidia-smi pmon` samples showed no external non-zero SM
PID on devices 0 or 5. The 20-pass run later detected external Python and
ffmpeg SM activity on device 5; its 5,666.80 measured value is intentionally
not performance evidence.

The first dual-device attempt exposed a missing receiver-device selection in
the persistent host submission worker. GPU1's cuBLAS handle was invoked while
GPU0 was current and failed at `model.linear_global` with
`CUBLAS_STATUS_INTERNAL_ERROR`. `launchEventPipelineInference` now makes the
handle's owning device current before any stream, event, copy, graph, or cuBLAS
submission. The corrected 1,024-row smoke reached 6,016.95 physical
`nnEval/s`, followed by the full certificate above.

Remote logs are retained in the persistent directory
`/mnt/CacheSSD/wangyize/katago-plan-runtime-20260809/logs/`.
