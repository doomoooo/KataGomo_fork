# Stage 28 outer-projection experiment manifest

- Timestamp range: 2026-08-06 06:14:56 through 06:37:00 UTC.
- Repository: `/workspace/katago`, revision
  `090caa6115c2ae86a75839d1b4fddeacd23d7444`, dirty optimization tree.
- Target: fixed 19x19, B13, FP16, two independent CUDA streams with private
  but value-identical weights.
- Contract: `M4693 N384 K768 beta=0`; expand:
  `M4693 N768 K384 beta=1`.
- GPU: CUDA device 2, NVML GPU 1, NVIDIA GeForce RTX 5090 D, 32607 MiB,
  driver 595.80.
- CUDA toolkit: 13.2.86; Nsys 2026.1.3; NCU 2026.2.1.
- Harness SHA256:
  `7bc75c399767d54c1ac5be71184588624791e2af664a8661686a3c9d98a65e85`.
- Confirmed contract binary SHA256:
  `c1df1bfae110690361b3bf97c3fe11d6064009d0ae6f5b112cfdec175a3da958`.

All GPU executions used:

```text
source /workspace/container-setup/nvidia-env.sh && gpu-lock with --gpu 2 -- ...
```

NCU/Nsys used `gpu-lock with --profile --gpu 2`; Nsys additionally set
`DEBUGINFOD_URLS=` and `DEBUGINFOD_TIMEOUT=1`.

No repository source or default was changed. Full-network and full-accuracy
gates were intentionally skipped because no candidate passed exact S2 micro.
