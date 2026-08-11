# RTX 4090 explicit-stream nsys verification

Date: 2026-08-06 UTC

## Result

- Workload: exact B13, 19x19, two NN server threads, 20 measured iterations after 10 warmups.
- GPU clock: fixed at 2400 MHz.
- GPU access: `gpu-lock with --gpu 0`.
- Stage 68 explicit-stream binary SHA-256: `4685776d9db45fce392555c2131f5d403a84ccada38bf08c303eda0626fcaa5f`.
- Profiled throughput: **3440.839 nnEval/s**.
- Unprofiled Stage 68 S2 repeats: **3458.389, 3434.821, 3433.794 nnEval/s**.
- The accepted CUTLASS dual-GEMM, fused QKV+RoPE, and FlashAttention both16 kernels are present.
- CUDA Runtime calls have no nonzero return codes.

## Stream placement

All steady-state forward kernels are on exactly two explicitly created non-blocking streams:

| stream | creation flag | kernel launches | kernel time |
| --- | --- | ---: | ---: |
| 81 | non-blocking | 9446 | 196.229 ms |
| 82 | non-blocking | 9446 | 198.649 ms |

Streams 15 and 16 only contain construction/warmup work. No steady-state forward kernel is on a default or per-thread-default stream.

For the final measured S2 interval, the union of kernel intervals from streams 81 and 82 is:

- interval: 151.135 ms
- at least one compute kernel active: 150.553 ms
- kernel-busy fraction: **99.615%**
- total gaps: 0.581 ms
- maximum gap: 23.2 us
- gaps at least 10 us: 1
- gaps at least 50 us: 0

## Why the earlier 3653 nnEval/s result was invalid

The earlier PTDS trace used two CUDA streams per logical NN server from the same host thread:

| host thread | AOT/library stream | generic-helper PTDS stream |
| --- | --- | --- |
| 291946594141747 | 84 (6560 launches) | 118 (2886 launches) |
| 291946594141748 | 83 (6560 launches) | 106 (2886 launches) |

The AOT/CUTLASS/cuBLAS/cuDNN work used non-blocking streams 83/84, while RMSNorm, scale/bias, pooling, copies, and other generic helpers used PTDS streams 106/118. The trace contains no `cudaStreamWaitEvent` calls ordering either pair. Those helper kernels consume and produce tensors in the same forward, so their apparent overlap with the AOT/library stream is a data race. The resulting 3653 nnEval/s measurement is not a valid inference schedule.

Stage 68 puts all operations of each logical server on one explicit stream. Its approximately 3440 nnEval/s result is both correctly ordered and consistent with the accepted 3400+ optimization result.

## Artifacts

- `external-s2.nsys-rep`
- `external-s2.sqlite`
- `external-s2.log`
- `inputs.sha256`
