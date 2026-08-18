#!/usr/bin/env python3
"""Launch one B29 native SwiGLU inside an explicit CUDA profiler window."""

from __future__ import annotations

import argparse
import ctypes
import json
import pathlib

try:
    import sm103_cudnn_oss_b29 as base
    import sm103_cudnn_oss_b29_no_ab12 as no_ab12
except ModuleNotFoundError:
    from python import sm103_cudnn_oss_b29 as base
    from python import sm103_cudnn_oss_b29_no_ab12 as no_ab12


def profile_once(
    *, library_path: pathlib.Path, device: int, warmup: int, launches: int
) -> dict[str, object]:
    if not library_path.is_file():
        raise RuntimeError(f"native library is missing: {library_path}")
    if device < 0 or warmup < 1 or launches < 1:
        raise RuntimeError("invalid device or launch count")

    import torch

    torch.cuda.set_device(device)
    if tuple(torch.cuda.get_device_capability(device)) != (10, 3):
        raise RuntimeError("profile requires exact SM103")
    tensors = base._allocate_gpu_benchmark_inputs(torch, device, 20260818)
    problem = tensors["problem"]
    ab12 = torch.empty_strided(
        (problem.m, problem.n_packed, 1),
        (problem.n_packed, 1, problem.m * problem.n_packed),
        dtype=torch.float16,
        device=f"cuda:{device}",
    )
    output = torch.empty_strided(
        (problem.m, problem.n_output, 1),
        (problem.n_output, 1, problem.m * problem.n_output),
        dtype=torch.float16,
        device=f"cuda:{device}",
    )
    library = no_ab12._load_native_library(library_path)
    status = ctypes.c_int32(-999)
    context = library.katagoCudnnOssB29Create(device, ctypes.byref(status))
    if not context or status.value != 0:
        raise RuntimeError(f"native context create failed: {status.value}")

    def launch() -> None:
        stream = torch.cuda.current_stream(device)
        launch_status = library.katagoCudnnOssB29Launch(
            context,
            ctypes.c_void_p(tensors["a_tensor"].data_ptr()),
            ctypes.c_void_p(tensors["b_tensor"].data_ptr()),
            ctypes.c_void_p(ab12.data_ptr()),
            ctypes.c_void_p(output.data_ptr()),
            ctypes.c_float(1.0),
            ctypes.c_void_p(stream.cuda_stream),
            problem.m,
            problem.k,
            problem.n_packed,
            problem.n_output,
            1,
        )
        if launch_status != 0:
            raise RuntimeError(f"native launch failed: {launch_status}")

    try:
        for _ in range(warmup):
            launch()
        torch.cuda.synchronize(device)
        torch.cuda.cudart().cudaProfilerStart()
        for _ in range(launches):
            launch()
        torch.cuda.synchronize(device)
        torch.cuda.cudart().cudaProfilerStop()
    finally:
        try:
            torch.cuda.synchronize(device)
        finally:
            library.katagoCudnnOssB29Destroy(context)

    return {
        "status": "passed",
        "library": str(library_path.resolve()),
        "device": device,
        "warmup": warmup,
        "profiled_launches": launches,
        "problem": {"m": problem.m, "k": problem.k, "n_packed": problem.n_packed},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=pathlib.Path, required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--launches", type=int, default=1)
    parser.add_argument("--allow-gpu", action="store_true")
    args = parser.parse_args()
    if not args.allow_gpu:
        raise RuntimeError("profiling requires explicit --allow-gpu")
    print(
        json.dumps(
            profile_once(
                library_path=args.library,
                device=args.device,
                warmup=args.warmup,
                launches=args.launches,
            ),
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
