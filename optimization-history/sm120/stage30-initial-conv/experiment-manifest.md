# Stage 30 initial-convolution experiment manifest

- Result window: 2026-08-06 07:49 through 08:10:53 UTC.
- Repository: `/workspace/katago`, revision
  `090caa6115c2ae86a75839d1b4fddeacd23d7444`, dirty optimization tree.
- No repository source, runtime option, or default was modified by this task.
- Target: RTX 5090 D, CUDA device 2 / NVML GPU 1,
  UUID `GPU-a05bf432-3f6b-83c1-98b1-aec8f8a6fd69`, 170 SMs, driver 595.80.
- Toolchain: CUDA 13.2.86, cuDNN header/runtime 9.25.0,
  cudnn_frontend 1.24.0.
- Workload: fixed 19x19/B13, FP16 NHWC X/W/Y, FP32 convolution compute,
  C22->C768 3x3 pad-1 cross-correlation, beta 0, one non-blocking stream.
- Timing: deterministic seed 20260806, four alternating-order legs, 100
  warmups and 5,000 measured calls per finalist leg.
- Standalone probe source SHA256:
  `e4d662e102def9f51176dea03b18ffee4063755d62aff88280d469641570760f`.
- Standalone probe binary SHA256:
  `badcd6f03474668c17fdda52c430dcf5bc1efe787f314cc9d0cdd7093c2db5fa`.
- Full enumeration JSON SHA256:
  `d74a6d121efa75111813ca956a496e624d798f3e0479772a62ebd12bcbbfbb9f`.
- Winner long JSON SHA256:
  `d805e609e559f3bbf55b38dd1fa187bc438c9cf4f59ddbf629d523b9e1652350`.
- Winner Nsys SHA256:
  `35a976999e67b88b576b655cc7d0d5cf47d9a0afca2c9a8e21416ee29af6717d`.
- Winner NCU SHA256:
  `5ed5c21f737a5c8df8faa8a48f7528129c8ff1ee75ee2d63a72b49023b0d2645`.
- Legacy NCU SHA256:
  `fb4328bea7c675036e1b0417337316698b9aff4d5e393e7a9f3f6eb34c96f06d`.

All GPU executions used:

```text
source /workspace/container-setup/nvidia-env.sh && gpu-lock with --gpu 2 -- ...
```

NCU/Nsys used `gpu-lock with --profile --gpu 2`; Nsys additionally set
`DEBUGINFOD_URLS=` and `DEBUGINFOD_TIMEOUT=1`.

The first probe build failed before GPU execution because the vendored frontend
header references NVRTC helpers; `-lnvrtc -lcuda` fixed the standalone link.
That negative result is retained in `build-attempts.md`.
