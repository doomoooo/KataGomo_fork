# RTX 4090 event-pipeline graph-node verification

Date: 2026-08-06 UTC

- Integration revision: `c4b92e3`
- Optimizer stream-interface checkpoint: `bd6b8a6`
- Workload: 19x19, FP16 NHWC, exact B13, S2, t96, three GTP benchmark positions
- GPU access: `gpu-lock with --profile --gpu 0`
- Profiled throughput: 3786.08 visits/s, 3234.71 real nnEval/s
- Unprofiled ten-position throughput: 3660.30 visits/s, 3308.06 real nnEval/s

The trace contains 427 CUDA Graph launches with exactly 295 forward kernel
nodes each. All graph kernels use explicit non-blocking compute streams 15 and
51. H2D uses streams 48/84 and D2H uses streams 49/85; the copy streams contain
no kernels. No steady forward kernel uses a legacy or per-thread default
stream, and no CUDA Runtime API returned a nonzero status.

H2D overlap with compute is 822/854 copies and 96.507% by duration. D2H overlap
is 2045/2135 copies and 96.758% by duration. The two compute streams are both
active for 62.8-63.7% of collective busy time in continuously supplied
regions.

Continuously supplied compute regions:

| span (ms) | compute-kernel union busy | both streams / union busy |
| ---: | ---: | ---: |
| 405.360 | 99.995278% | 62.826% |
| 190.995 | 99.995878% | 63.662% |
| 519.734 | 99.996250% | 63.649% |
| 482.973 | 99.998066% | 63.292% |

The largest included collective kernel gap is below one microsecond. Position
restart/ramp and final drain are outside those regions. A separate 5.445 ms
gap in the first position correlates with unusually slow node-traced
`cudaGraphLaunch` host calls of 6.91 and 9.63 ms; it is kept visible as a
profiler/submission outlier.

Graph capture makes two external event waits and two external event records
per handle. Runtime records show the matching input-ready, input-consumed,
apply-complete, output-consumed, and output-ready cadence, validating the
single-device-slot reuse contract.

Artifacts:

- `gtp-c4b92e3-v2.nsys-rep`
- `gtp-c4b92e3-v2.sqlite`
- `gtp-c4b92e3-v2.log`
