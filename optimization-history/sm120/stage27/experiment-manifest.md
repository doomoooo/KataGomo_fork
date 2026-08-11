# Stage 27 experiment manifest

- Result timestamps: 2026-08-06 05:37:52 through 06:44:35 UTC.
- Repository: `/workspace/katago`, revision
  `090caa6115c2ae86a75839d1b4fddeacd23d7444`, dirty optimization tree.
- CUDA executable SHA256:
  `f538073ebec4297d35ebec10dcbda66ed32081ec2e1af5ba430ad75d80fee4e0`.
- Model: `b11c768h12nbt3tflrs-fson-silu.bin.gz`, SHA256
  `1881600caab9e9d85a3dd6a019e9b8e7d2c237b5f984e13ed49a8645be3077c6`.
- Shape/precision/topology: 19x19, B13, FP16, two CUDA streams, private but
  value-identical out-projection weights, `M4693 N384 K384`, residual
  epilogue. The original sweep's shared-weight assumption was corrected in a
  finalist rerun and fresh Nsys traces.
- GPU: CUDA device 2, NVML GPU 1, NVIDIA GeForce RTX 5090 D,
  UUID `GPU-a05bf432-3f6b-83c1-98b1-aec8f8a6fd69`, driver 595.80.
- Search script SHA256:
  `e7215a33918d217a982d2d64e0abbec2a1e24ad3d57ff0c25f230b3197864c87`.
- Exact cuBLAS micro source SHA256:
  `b2e5939f41745032d805d8d4908a0ac78575fb86da5f246adb2bb0b61795149a`.
- Exact cuBLAS micro binary SHA256:
  `290aafcdf8ae68111a7417db56fb2dc0a7b8a0d9eac0d55be0fadc68096a7d91`.
- Generated M96N128 candidate source SHA256:
  `ba172100de98ab925b100dc3dbac6e4ebe051a84311fdf2fd7e6e56bc6e0ea66`.

All GPU executions used:

```text
source /workspace/container-setup/nvidia-env.sh && gpu-lock with --gpu 2 -- ...
```

NCU/Nsys used `gpu-lock with --profile --gpu 2`; Nsys additionally set
`DEBUGINFOD_URLS=` and `DEBUGINFOD_TIMEOUT=1`.

The search changed no repository default or shared C++ source. No whole-network
benchmark or full accuracy run was performed because the direct S2 boundary
gate failed by 63.3% after the ownership correction.
