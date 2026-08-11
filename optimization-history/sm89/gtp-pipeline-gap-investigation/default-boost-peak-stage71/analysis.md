# RTX 4090 default-Boost peak retest after Stage 71

Date: 2026-08-06 19:27 UTC

## Configuration

- GPU 0: NVIDIA GeForce RTX 4090
- Exact 19x19, FP16 NHWC, physical B13, S2, search t96
- Event-gated asynchronous pipeline enabled
- CUDA Graph disabled (`cudaEventPipelineUseGraph=false`)
- `CUDA_DEVICE_MAX_CONNECTIONS` unset; equal default stream priorities
- Both graphics and memory application-clock locks reset before the run
- Default 450 W power limit; all work executed under `gpu-lock with --gpu 0`
- One 1200-iteration direct pass discarded as thermal/plan warmup
- Decision order: direct, GTP, GTP, direct, direct, GTP

The aligned GTP metric remains `nnBatches/s * 13`. `visits/s` is reported
separately and is not interpreted as physical neural-network work.

## Results

### Direct common-wall physical B13 throughput

| sample | physical nnEval/s |
|---|---:|
| direct-a | 3390.919 |
| direct-b | 3417.270 |
| direct-c | 3419.155 |
| mean | 3409.115 |
| median | 3417.270 |
| best | 3419.155 |

The discarded warmup pass measured 3416.913 physical nnEval/s.

### Real GTP eager pipeline

| sample | visits/s | real nnEval/s | nnBatches/s | physical B13 nnEval/s |
|---|---:|---:|---:|---:|
| gtp-a | 4719.33 | 3374.08 | 261.45 | 3398.85 |
| gtp-b | 4740.91 | 3367.06 | 260.99 | 3392.87 |
| gtp-c | 4730.60 | 3371.23 | 261.51 | 3399.63 |
| mean | 4730.28 | 3370.79 | 261.32 | 3397.12 |

Peak observed real-GTP physical throughput is **3399.63 nnEval/s**. The mean
direct-to-GTP physical residual is 11.998 rows/s, or 0.353% of GTP. The lower
real `nnEval/s` is expected from the small amount of row padding; visits/s is
strictly higher because many visits do not cause a real NN evaluation.

Relative to the Stage 71 fixed 2205/10251 MHz attribution regime, mean direct
throughput increased from 3242.370 to 3409.115 (+5.143%), and mean GTP physical
throughput increased from 3236.025 to 3397.117 (+4.978%).

## Telemetry

For samples with GPU utilization at least 90%:

- GPU utilization averaged 99.9-100.0% in every decision segment.
- Mean SM clock was 2357-2396 MHz; observed loaded range was 2295-2595 MHz.
- Memory clock was 10251 MHz under load.
- Temperature remained 68-75 C during decision samples.
- The software power-cap reason was active under sustained load at the default
  450 W limit.
- No software thermal slowdown, hardware thermal slowdown, or hardware power
  brake was observed.

Therefore the unlocked steady-state ceiling is power-limited rather than
thermal-throttled, and the eager two-stream GTP frontend keeps the GPU fully
busy while reaching approximately 3400 physical B13 nnEval/s.

Artifacts: `direct-{a,b,c}.json`, `gtp-{a,b,c}.summary.txt`, `telemetry.csv`,
`markers.log`. The exact runner is the parent-directory script
`run-default-boost-peak-stage71.sh`.
