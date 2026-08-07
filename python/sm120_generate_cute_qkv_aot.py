#!/usr/bin/env python3
"""Generate the exact-B1..B32 packed-QKV CuTe AOT search candidate.

This generator does not allocate or compute on a GPU. It supplies static CUDA
DLPack descriptors backed by inert host-side metadata so CuTe can specialize
the kernel types and emit an SM120 object. When no explicit scheduler value is
provided, the target device's SM count is queried through the CUDA runtime and
baked into this compile-time AOT object. The generated object is newly
compiled from the pinned CUTLASS source; no historical object is copied.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import importlib.util
import importlib.metadata
import json
import os
import pathlib
import re
import subprocess
import sys

os.environ.setdefault("CUTE_DSL_ARCH", "sm_120")
if "CUDA_TOOLKIT_PATH" not in os.environ and "CUDA_HOME" in os.environ:
    os.environ["CUDA_TOOLKIT_PATH"] = os.environ["CUDA_HOME"]

import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
from cuda.bindings import driver as cuda
from cutlass.cute.runtime import from_dlpack

try:
    from sm120_device import query_cuda_device
except ModuleNotFoundError:  # imported as ``python.sm120_generate_cute_qkv_aot``
    from python.sm120_device import query_cuda_device


CANDIDATE_ID = "qkv-m128-n128-k64-s2-cute-atom4x2-packed"
CUTLASS_COMMIT = "e05f953a5b3d38adc240df2ff928e0421c2abba3"
DENSE_GEMM_SHA256 = "613052799aff35d5564d49c8bbb4bbac2e22bc58cb3e27499c4c9c3ee95c6e03"
SEQUENCE = 361
INPUT_CHANNELS = 384
OUTPUT_CHANNELS = 3 * 384
TILE = (128, 128, 64)
ATOM_LAYOUT = (4, 2, 1)


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_output(root: pathlib.Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(root), *args], check=True, text=True,
        capture_output=True,
    ).stdout.strip()


def package_version(*names: str) -> str | None:
    """Return the first installed distribution name used by this toolchain."""
    for name in names:
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            continue
    return None


class _DLDevice(ctypes.Structure):
    _fields_ = [("device_type", ctypes.c_int), ("device_id", ctypes.c_int)]


class _DLDataType(ctypes.Structure):
    _fields_ = [
        ("code", ctypes.c_uint8),
        ("bits", ctypes.c_uint8),
        ("lanes", ctypes.c_uint16),
    ]


class _DLTensor(ctypes.Structure):
    _fields_ = [
        ("data", ctypes.c_void_p),
        ("device", _DLDevice),
        ("ndim", ctypes.c_int),
        ("dtype", _DLDataType),
        ("shape", ctypes.POINTER(ctypes.c_int64)),
        ("strides", ctypes.POINTER(ctypes.c_int64)),
        ("byte_offset", ctypes.c_uint64),
    ]


_DL_DELETER = ctypes.CFUNCTYPE(None, ctypes.c_void_p)


class _DLManagedTensor(ctypes.Structure):
    _fields_ = [
        ("dl_tensor", _DLTensor),
        ("manager_ctx", ctypes.c_void_p),
        ("deleter", _DL_DELETER),
    ]


_PY_CAPSULE_NEW = ctypes.pythonapi.PyCapsule_New
_PY_CAPSULE_NEW.restype = ctypes.py_object
_PY_CAPSULE_NEW.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p]
_STATIC_SPEC_KEEPALIVE: list["StaticCudaTensorSpec"] = []


class StaticCudaTensorSpec:
    """DLPack producer describing CUDA storage without touching a CUDA device."""

    def __init__(
        self, shape: tuple[int, ...], strides: tuple[int, ...], address: int,
        device_id: int = 0,
    ) -> None:
        if len(shape) != len(strides):
            raise ValueError("shape/stride rank mismatch")
        self._shape = (ctypes.c_int64 * len(shape))(*shape)
        self._strides = (ctypes.c_int64 * len(strides))(*strides)
        # DLPack: kDLCUDA=2, kDLFloat=2. The inert pointer is only a type token;
        # generated code is never invoked by this Python process.
        tensor = _DLTensor(
            ctypes.c_void_p(address), _DLDevice(2, device_id), len(shape),
            _DLDataType(2, 16, 1), self._shape, self._strides, 0,
        )
        self._managed = _DLManagedTensor(tensor, None, _DL_DELETER())

    def __dlpack__(self, stream: int | None = None):
        del stream
        return _PY_CAPSULE_NEW(
            ctypes.addressof(self._managed), b"dltensor", None,
        )

    def __dlpack_device__(self) -> tuple[int, int]:
        return (2, self._managed.dl_tensor.device.device_id)


class FixedAtomLayoutGemm:
    """Factory wrapper kept outside the JIT source transformation."""

    @staticmethod
    def make(sm120_gemm_kernel):
        class _Kernel(sm120_gemm_kernel):
            def __init__(self) -> None:
                # Match the retained historical packed-QKV candidate.
                super().__init__(cutlass.Float16, TILE)
                self.atom_layout = ATOM_LAYOUT
                self.num_mma_warps = (
                    ATOM_LAYOUT[0] * ATOM_LAYOUT[1] * ATOM_LAYOUT[2]
                )
                self.threads_per_cta = (
                    self.num_mma_warps + 1
                ) * self.num_threads_per_warp
                self.epilog_sync_barrier = pipeline.NamedBarrier(
                    barrier_id=2,
                    num_threads=(
                        self.num_mma_warps * self.num_threads_per_warp
                    ),
                )

        return _Kernel()


def make_tensor(
    shape: tuple[int, ...], strides: tuple[int, ...], address: int,
    device_id: int,
) -> cute.Tensor:
    spec = StaticCudaTensorSpec(shape, strides, address, device_id)
    _STATIC_SPEC_KEEPALIVE.append(spec)
    return from_dlpack(
        spec,
        assumed_align=16,
        enable_tvm_ffi=False,
    )


def load_sm120_gemm_kernel(
    dense_path: pathlib.Path, output_dir: pathlib.Path,
):
    """Load a pinned SM120 example with its two unsupported SM120 hints off.

    CUTLASS v4.6.1's example emits setmaxregister directives that libNVVM
    rejects for SM120. The historical KataGo source made exactly the same
    source-level adjustment. Materializing the patched Python next to the AOT
    outputs keeps the generated object independently auditable.
    """
    source = dense_path.read_text()
    replacements = {
        "            cute.arch.setmaxregister_increase(self.mma_register_requirement)":
            "            pass  # SM120 rejects setmaxregister increase",
        "            cute.arch.setmaxregister_decrease(self.load_register_requirement)":
            "            pass  # SM120 rejects setmaxregister decrease",
    }
    for before, after in replacements.items():
        if source.count(before) != 1:
            raise RuntimeError(f"unexpected pinned dense source near: {before}")
        source = source.replace(before, after)
    patched_path = output_dir / "dense_gemm_sm120_patched.py"
    patched_path.write_text(source)
    module_spec = importlib.util.spec_from_file_location(
        "katago_sm120_pinned_dense_gemm", patched_path,
    )
    if module_spec is None or module_spec.loader is None:
        raise RuntimeError(f"cannot load {patched_path}")
    module = importlib.util.module_from_spec(module_spec)
    sys.modules[module_spec.name] = module
    module_spec.loader.exec_module(module)
    return module.Sm120GemmKernel, patched_path


def render_bridge(
    artifact_stem: str, batch: int, candidate_id: str,
    fat_symbol_token: str | None = None,
) -> str:
    prefix = artifact_stem
    if fat_symbol_token is None:
        exports = f'''extern "C" int sm120_search_qkv_batch() {{ return {batch}; }}
extern "C" const char* sm120_search_qkv_id() {{ return "{candidate_id}"; }}
extern "C" int sm120_search_qkv_packed() {{ return 1; }}

extern "C" cudaError_t sm120_search_qkv_launch'''
    else:
        exports = (
            'extern "C" cudaError_t sm120_search_qkv_fat_launch_'
            f'{fat_symbol_token}'
        )
    return f'''#include "{artifact_stem}.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <mutex>

namespace {{

{prefix}_Kernel_Module_t module = {{}};
std::once_flag loadOnce;

}} // namespace

{exports}(
  const half* input, const half* weights, half* output, cudaStream_t stream
) {{
  std::call_once(loadOnce, []() {{ {prefix}_Kernel_Module_Load(&module); }});
  {prefix}_Tensor_a_arg_t a = {{const_cast<half*>(input)}};
  {prefix}_Tensor_b_arg_t b = {{const_cast<half*>(weights)}};
  {prefix}_Tensor_c_arg_t c = {{output}};
  int32_t status = cute_dsl_{prefix}_wrapper(&module, &a, &b, &c, stream);
  return status == 0 ? cudaPeekAtLastError() : cudaErrorUnknown;
}}
'''


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, required=True)
    parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    parser.add_argument("--artifact-stem", default="sm120_qkv_cute_active")
    parser.add_argument("--bridge-path", type=pathlib.Path, required=True)
    parser.add_argument("--candidate-id", default=CANDIDATE_ID)
    parser.add_argument("--fat-symbol-token")
    parser.add_argument(
        "--cutlass-root", type=pathlib.Path,
        default=pathlib.Path(
            os.environ.get("CUTLASS_ROOT", "/workspace/third_party/cutlass")
        ),
    )
    parser.add_argument(
        "--device", type=int, default=None,
        help=(
            "target CUDA runtime ordinal used to query the default SM count; "
            "required when --max-active-clusters is omitted"
        ),
    )
    parser.add_argument("--max-active-clusters", type=int, default=None)
    args = parser.parse_args()
    if not 1 <= args.batch <= 32:
        parser.error("--batch must be in B1..B32")
    if args.candidate_id != CANDIDATE_ID:
        parser.error(f"only {CANDIDATE_ID} is supported")
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", args.artifact_stem):
        parser.error("--artifact-stem must be a C identifier")
    if args.fat_symbol_token and not re.fullmatch(
        r"[A-Za-z_][A-Za-z0-9_]*", args.fat_symbol_token
    ):
        parser.error("--fat-symbol-token must be a C identifier")
    if args.max_active_clusters is not None and args.max_active_clusters <= 0:
        parser.error("--max-active-clusters must be positive")
    device_properties = None
    if args.device is not None:
        device_properties = query_cuda_device(args.device)
    if args.max_active_clusters is None and device_properties is None:
        parser.error("--device is required when --max-active-clusters is omitted")
    if args.max_active_clusters is None:
        max_active_clusters = int(device_properties["multiprocessor_count"])
        max_active_clusters_source = "cudaDevAttrMultiProcessorCount"
    else:
        max_active_clusters = args.max_active_clusters
        max_active_clusters_source = "explicit_candidate"

    cutlass_root = args.cutlass_root.resolve()
    actual_commit = git_output(cutlass_root, "rev-parse", "HEAD")
    if actual_commit != CUTLASS_COMMIT:
        raise RuntimeError(
            f"CUTLASS commit mismatch: {actual_commit} != {CUTLASS_COMMIT}"
        )
    if git_output(cutlass_root, "status", "--short"):
        raise RuntimeError("CUTLASS source must be clean")
    dense_path = cutlass_root / (
        "examples/python/CuTeDSL/cute/blackwell_geforce/kernel/"
        "dense_gemm/dense_gemm.py"
    )
    dense_sha256 = sha256_file(dense_path)
    if dense_sha256 != DENSE_GEMM_SHA256:
        raise RuntimeError(
            f"dense_gemm.py hash mismatch: {dense_sha256} != {DENSE_GEMM_SHA256}"
        )
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    Sm120GemmKernel, patched_dense_path = load_sm120_gemm_kernel(
        dense_path, output_dir,
    )

    rows = args.batch * SEQUENCE
    device_id = args.device if args.device is not None else 0
    a = make_tensor(
        (rows, INPUT_CHANNELS, 1),
        (INPUT_CHANNELS, 1, rows * INPUT_CHANNELS),
        0x10000, device_id,
    )
    # Weight storage is [K,N] row-major; GEMM's logical B is [N,K,1].
    b = make_tensor(
        (OUTPUT_CHANNELS, INPUT_CHANNELS, 1),
        (1, OUTPUT_CHANNELS, OUTPUT_CHANNELS * INPUT_CHANNELS),
        0x20000, device_id,
    )
    # This row-major [token,1152] result is packed [token,3,384].
    c = make_tensor(
        (rows, OUTPUT_CHANNELS, 1),
        (OUTPUT_CHANNELS, 1, rows * OUTPUT_CHANNELS),
        0x30000, device_id,
    )
    gemm = FixedAtomLayoutGemm.make(Sm120GemmKernel)
    @cute.jit
    def launch(
        a_arg: cute.Tensor,
        b_arg: cute.Tensor,
        c_arg: cute.Tensor,
        stream: cuda.CUstream,
    ):
        gemm(a_arg, b_arg, c_arg, max_active_clusters, stream)

    compiled = cute.compile(
        launch, a, b, c,
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=False),
    )
    compiled.export_to_c(str(output_dir), file_name=args.artifact_stem)

    bridge_path = args.bridge_path.resolve()
    if bridge_path.parent != output_dir:
        raise ValueError("bridge and AOT artifacts must share --output-dir")
    bridge = render_bridge(
        args.artifact_stem, args.batch, args.candidate_id,
        args.fat_symbol_token,
    )
    bridge_path.write_text(bridge)

    artifact_base = output_dir / args.artifact_stem
    metadata = {
        "schema": 1,
        "candidate_id": args.candidate_id,
        "batch": args.batch,
        "fixed_shape": {
            "board": [19, 19],
            "rows": rows,
            "input_channels": INPUT_CHANNELS,
            "output_channels": OUTPUT_CHANNELS,
        },
        "layout": "packed-token-qkv",
        "fat_symbol_token": args.fat_symbol_token,
        "launch_symbol": (
            f"sm120_search_qkv_fat_launch_{args.fat_symbol_token}"
            if args.fat_symbol_token else "sm120_search_qkv_launch"
        ),
        "tile": list(TILE),
        "atom_layout": list(ATOM_LAYOUT),
        "max_active_clusters": max_active_clusters,
        "max_active_clusters_source": max_active_clusters_source,
        "device_properties": device_properties,
        "gpu_used_for_generation": False,
        "provenance": {
            "generator_sha256": sha256_file(pathlib.Path(__file__).resolve()),
            "cutlass_commit": actual_commit,
            "dense_gemm_sha256": dense_sha256,
            "patched_dense_gemm_sha256": sha256_file(patched_dense_path),
            "python": sys.version.split()[0],
            "nvidia_cutlass_dsl": importlib.metadata.version(
                "nvidia-cutlass-dsl"
            ),
            "nvidia_cutlass_dsl_libs_cu12": importlib.metadata.version(
                "nvidia-cutlass-dsl-libs-cu12"
            ),
            "dsl_cuda_version": str(cutlass.CUDA_VERSION),
            "cuda_python": package_version("cuda-python", "cuda-bindings"),
            "cuda_bindings": package_version("cuda-bindings"),
            "cuda_toolkit_path": os.environ.get("CUDA_TOOLKIT_PATH", ""),
            "nvcc_version": subprocess.run(
                ["nvcc", "--version"], check=True, text=True,
                capture_output=True,
            ).stdout.strip().splitlines()[-1],
        },
        "sha256": {
            "header": sha256_file(artifact_base.with_suffix(".h")),
            "object": sha256_file(artifact_base.with_suffix(".o")),
            "bridge": hashlib.sha256(bridge.encode()).hexdigest(),
        },
    }
    metadata_path = artifact_base.with_suffix(".json")
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    print(metadata_path)


if __name__ == "__main__":
    main()
