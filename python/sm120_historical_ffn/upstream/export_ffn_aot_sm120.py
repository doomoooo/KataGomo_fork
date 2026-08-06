#!/usr/bin/env python3
"""Export the verified fixed-B19 TileLang FFN kernel as a CUDA AOT object."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import onnx_kernels
from tune_ffn_swiglu import intrinsic_config


ROWS = 19 * 361
COLUMNS = 1152
INNER = 384
KERNEL_NAME = "katago_ffn_tilelang_sm120_b19_s361_kernel"
OBJECT_NAME = "katago_ffn_tilelang_sm120_b19_s361.o"
SOURCE_NAME = "katago_ffn_tilelang_sm120_b19_s361.cu"
HEADER_NAME = "katago_ffn_tilelang_sm120.h"

HEADER = """#pragma once

#include <cuda_runtime_api.h>

extern "C" void* katago_create_ffn_tilelang_sm120_b19(
  const void* linear1_weights,
  const void* gate_weights
);

extern "C" void katago_destroy_ffn_tilelang_sm120_b19(void* handle);

extern "C" int katago_launch_ffn_tilelang_sm120_b19(
  void* handle,
  const void* input,
  void* output,
  cudaStream_t stream
);
"""

WRAPPER = r'''

#include "katago_ffn_tilelang_sm120.h"

#include <new>

namespace {

struct KataGoFFNTileLangHandle {
  const half_t* linear1Weights;
  const half_t* gateWeights;
};

} // namespace

extern "C" void* katago_create_ffn_tilelang_sm120_b19(
  const void* linear1_weights,
  const void* gate_weights
) {
  cudaError_t status = cudaFuncSetAttribute(
    KATAGO_KERNEL_NAME,
    cudaFuncAttributeMaxDynamicSharedMemorySize,
    KATAGO_DYNAMIC_SMEM);
  if(status != cudaSuccess)
    return nullptr;
  return new (std::nothrow) KataGoFFNTileLangHandle{
    static_cast<const half_t*>(linear1_weights),
    static_cast<const half_t*>(gate_weights)
  };
}

extern "C" void katago_destroy_ffn_tilelang_sm120_b19(void* handle) {
  delete static_cast<KataGoFFNTileLangHandle*>(handle);
}

extern "C" int katago_launch_ffn_tilelang_sm120_b19(
  void* handle,
  const void* input,
  void* output,
  cudaStream_t stream
) {
  if(handle == nullptr || input == nullptr || output == nullptr)
    return static_cast<int>(cudaErrorInvalidValue);
  const KataGoFFNTileLangHandle* state =
    static_cast<const KataGoFFNTileLangHandle*>(handle);
  KATAGO_KERNEL_NAME<<<dim3(18, 54, 1), dim3(128, 1, 1), KATAGO_DYNAMIC_SMEM, stream>>>(
    static_cast<const half_t*>(input),
    state->gateWeights,
    state->linear1Weights,
    static_cast<half_t*>(output));
  return static_cast<int>(cudaPeekAtLastError());
}
'''


def write_if_changed(path: Path, contents: str) -> None:
    if path.is_file() and path.read_text() == contents:
        return
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(contents)
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/data/wangyize/katago/build/ffn-tilelang-aot-sm120"),
    )
    parser.add_argument(
        "--cuda-root",
        type=Path,
        default=Path("/data/wangyize/katago/opt/cuda-13.2"),
    )
    parser.add_argument("--stages", type=int, choices=(1, 2), default=2)
    parser.add_argument("--min-blocks-per-sm", type=int, choices=(3, 4), default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = intrinsic_config(
        128,
        64,
        32,
        args.stages,
        epilogue="swiglu_tanh_f16x2",
        min_blocks_per_sm=args.min_blocks_per_sm,
    )
    kernel = onnx_kernels.ffn_swiglu_intrinsics(
        ROWS,
        COLUMNS,
        INNER,
        128,
        64,
        32,
        args.stages,
        "float16",
        False,
        2,
        2,
        "swiglu_tanh_f16x2",
        False,
        args.min_blocks_per_sm,
    )
    source = kernel.get_kernel_source()
    expected_signature = (
        'extern "C" __global__ void __launch_bounds__'
        f'(128, {args.min_blocks_per_sm}) main_kernel'
    )
    if expected_signature not in source:
        raise RuntimeError("generated kernel does not have the verified launch bounds")
    if "dim3(18, 54" in source:
        raise RuntimeError("expected device-only source, got a host wrapper")
    if source.count("main_kernel") != 2:
        raise RuntimeError("unexpected generated kernel symbol count")
    source = source.replace("main_kernel", KERNEL_NAME)
    dynamic_shared_memory_bytes = 16384 * args.stages
    source += (
        WRAPPER.replace("KATAGO_KERNEL_NAME", KERNEL_NAME)
        .replace("KATAGO_DYNAMIC_SMEM", str(dynamic_shared_memory_bytes))
    )

    tilelang_root = Path(__import__("tilelang").__file__).resolve().parent
    args.output_dir.mkdir(parents=True, exist_ok=True)
    source_path = args.output_dir / SOURCE_NAME
    header_path = args.output_dir / HEADER_NAME
    object_path = args.output_dir / OBJECT_NAME
    write_if_changed(source_path, source)
    write_if_changed(header_path, HEADER)

    command = [
        args.cuda_root / "bin/nvcc",
        "-std=c++20",
        "-O3",
        "-w",
        "-Xcudafe",
        "--diag_suppress=177",
        "-lineinfo",
        "-gencode",
        "arch=compute_120,code=sm_120",
        f"-I{tilelang_root / '3rdparty/cutlass/include'}",
        f"-I{tilelang_root / 'src'}",
        f"-I{args.output_dir}",
        "-c",
        source_path,
        "-o",
        object_path,
    ]
    subprocess.run([str(item) for item in command], check=True)

    metadata = {
        "schemaVersion": 1,
        "shape": {"M": ROWS, "N": COLUMNS, "K": INNER},
        "config": config.__dict__,
        "kernelName": KERNEL_NAME,
        "grid": [18, 54, 1],
        "block": [128, 1, 1],
        "dynamicSharedMemoryBytes": dynamic_shared_memory_bytes,
        "sourceSha256": hashlib.sha256(source.encode()).hexdigest(),
        "objectSha256": hashlib.sha256(object_path.read_bytes()).hexdigest(),
        "compileCommand": [str(item) for item in command],
    }
    write_if_changed(
        args.output_dir / "katago_ffn_tilelang_sm120_b19_s361.json",
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
    )
    print(object_path)


if __name__ == "__main__":
    main()
