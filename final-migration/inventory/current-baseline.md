# Current baseline inventory

Captured on 2026-08-07 UTC. This is an observation, not a freeze declaration.

## Official integration base

- Remote: `https://github.com/lightvector/KataGo.git`
- Commit: `6a1fc5de9fc253723ac475a0683bf0b9d9b7bd19`
- Commit date: 2026-08-05
- Description: `v1.17.2`
- Local branch: `final-migration`
- Local worktree: `/workspace/katago-final-migration`

The repository's original `origin` fetch refspec only fetched tag `v1.17.1`.
The official branch was therefore refreshed explicitly with:

```text
git fetch origin refs/heads/master:refs/remotes/origin/master
```

Future refreshes must repeat an explicit branch fetch or repair the refspec.

## Host

- Ubuntu 24.04.4 LTS, kernel 6.8.0-124
- GCC/G++ 13.3.0, CMake 3.28.3, Ninja 1.11.1
- Python 3.12.3
- NVIDIA driver reported by the host: 595.80
- CUDA toolkit: 13.2.2, nvcc 13.2.86
- cuBLAS: 13.4.1.3
- cuDNN system packages: 9.25.0.15 for CUDA 13
- TensorRT 10.16.1.11 is present on the captured host but is explicitly outside
  final-migration scope.
- Nsight Systems: 2026.1.3; Nsight Compute: 2026.2.1

The NVIDIA repository candidate observed later the same day was CUDA 13.3.1
and open driver 610.57.04. Fresh setup resolves current meta packages; this
shared host remains on its operational 13.2.2/595.80 stack so active optimizer
sessions are not disturbed.

Visible GPUs at capture time:

- one RTX 5090D, compute capability 12.0;
- two RTX 4090, compute capability 8.9.

CUDA runtime numbering and `nvidia-smi` numbering have differed on this host.
GPU jobs must continue to use the existing lock/device-map wrapper rather than
assuming index identity.

## Disk risk

`/workspace` had about 102 GiB free while 98% full. Do not duplicate existing
results trees, Python environments, generated AOT bundles, or build trees.
Archive large payloads by manifest into persistent storage. Keep only compact
metadata, patches, scripts, and reports in Git.

## Externally owned worktrees

| Role | Path | Observed branch/commit |
| --- | --- | --- |
| active SM120/scanner | `/workspace/katago` | `benchmarknn` / `ed509a1` |
| 4090 optimization | `/workspace/katago-4090` | `4090-opt` / `bd6b8a6` |
| frontend integration | `/workspace/katago-final-frontend-integration` | historical integration snapshot |
| SM120 accepted snapshot | `/workspace/katago-sm120-accepted` | historical accepted snapshot |
| event pipeline | `/workspace/katago-dual-gpu-event-pipeline` | `cafeeae` |

These paths may contain dirty or conflicted state owned by other sessions. They
are read-only to final migration.

## Optimization result roots

- SM120 history: `/workspace/results/sm120/HISTORY.md`
- recent SM120 stage reports: `stage55-cute-packed-qkv-rope`,
  `stage56-fa4-n64`, and `stage57-fa4-n96`
- generated cross-batch plans: `/workspace/results/sm120/cross-batch-search/`
- SM89 history: `/workspace/results/sm89/HISTORY.md`

Generated plan filenames currently include SM120 tactic plans and joint fixed
plans. They are evidence for schema discovery only until their owner freezes
them.
