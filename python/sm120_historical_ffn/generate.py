#!/usr/bin/env python3
"""Generate the historical exact-batch SM120 tanh-half2 FFN source.

This is deliberately a source-only generator.  It imports the frozen historical
TileLang schedule, emits a CUDA translation unit for B1..B32, and never imports
PyTorch or selects a CUDA device.  Acceptance remains natural whole-graph S2.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import sys
from typing import Any


# Make accidental runtime GPU use impossible before TileLang is imported.  CUDA
# source generation for an explicit sm_120 target does not require a device.
os.environ["CUDA_VISIBLE_DEVICES"] = ""

ROOT = Path(__file__).resolve().parent
UPSTREAM = ROOT / "upstream"
MANIFEST_PATH = ROOT / "manifest.json"
sys.path.insert(0, str(ROOT.parent))

from sm120_fat_scan import (  # noqa: E402
    isolate_tilelang_debug_symbols,
    launch_symbol,
    validate_symbol_token,
)

BOARD_AREA = 19 * 19
COLUMNS = 1152
INNER = 384
BLOCK_M = 128
BLOCK_N = 64
BLOCK_K = 32
STAGES = 2
THREADS = 128
MIN_BLOCKS_PER_SM = 3
DYNAMIC_SMEM_BYTES = 32768
CANDIDATE_ID = "ffn-m128-n64-k32-s2-mb3-tanh-half2"
SOURCE_NAME = f"ffn-{CANDIDATE_ID}.cu"
METADATA_NAME = f"ffn-{CANDIDATE_ID}.json"

CANDIDATE = {
    "id": CANDIDATE_ID,
    "m": BLOCK_M,
    "n": BLOCK_N,
    "k": BLOCK_K,
    "stages": STAGES,
    "threads": THREADS,
    "min_blocks": MIN_BLOCKS_PER_SM,
    "a_fragment_reuse": False,
    "swiglu": "tanh_half2",
    "implementation": "historical_tilelang",
}

WRAPPER = r'''

#include <cuda_runtime_api.h>

extern "C" int sm120_search_ffn_batch() { return KATAGO_BATCH; }
extern "C" const char* sm120_search_ffn_id() {
  return "ffn-m128-n64-k32-s2-mb3-tanh-half2";
}

extern "C" cudaError_t sm120_search_ffn_launch(
  const half* input, const half* linear_weights, const half* gate_weights,
  half* output, cudaStream_t stream
) {
  KATAGO_KERNEL_NAME<<<
    dim3(18, KATAGO_GRID_Y, 1), dim3(128, 1, 1), 32768, stream>>>(
      reinterpret_cast<const half_t*>(input),
      reinterpret_cast<const half_t*>(gate_weights),
      reinterpret_cast<const half_t*>(linear_weights),
      reinterpret_cast<half_t*>(output));
  return cudaPeekAtLastError();
}
'''

FAT_WRAPPER = r'''

#include <cuda_runtime_api.h>

extern "C" cudaError_t KATAGO_FAT_LAUNCH_SYMBOL(
  const half* input, const half* linear_weights, const half* gate_weights,
  half* output, cudaStream_t stream
) {
  KATAGO_KERNEL_NAME<<<
    dim3(18, KATAGO_GRID_Y, 1), dim3(128, 1, 1), 32768, stream>>>(
      reinterpret_cast<const half_t*>(input),
      reinterpret_cast<const half_t*>(gate_weights),
      reinterpret_cast<const half_t*>(linear_weights),
      reinterpret_cast<half_t*>(output));
  return cudaPeekAtLastError();
}
'''


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_path(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def write_if_changed(path: Path, contents: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file() and path.read_text(encoding="utf-8") == contents:
        return
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(contents, encoding="utf-8")
    temporary.replace(path)


def load_manifest() -> dict[str, Any]:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    for name, expected in manifest["historicalSources"].items():
        actual = sha256_path(UPSTREAM / name)
        if actual != expected:
            raise RuntimeError(
                f"historical source drift for {name}: expected {expected}, got {actual}"
            )
    return manifest


def tilelang_tree_hash(tilelang_root: Path) -> str:
    """Hash the installed TileLang tree with a path-stable canonical encoding."""

    records = bytearray()
    files = (
        path
        for path in tilelang_root.rglob("*")
        if path.is_file() and "__pycache__" not in path.parts
    )
    for path in sorted(
        files, key=lambda item: item.relative_to(tilelang_root).as_posix()
    ):
        relative = path.relative_to(tilelang_root).as_posix()
        records.extend(relative.encode("utf-8"))
        records.extend(b"  ")
        records.extend(sha256_path(path).encode("ascii"))
        records.extend(b"\n")
    return sha256_bytes(bytes(records))


def verify_dependencies(manifest: dict[str, Any]) -> dict[str, Any]:
    distribution = importlib.metadata.distribution("tilelang")
    tilelang_root = Path(distribution.locate_file("tilelang")).resolve()
    actual_tree = tilelang_tree_hash(tilelang_root)
    expected_tree = manifest["dependencies"]["tilelangPackageTreeSha256"]
    if actual_tree != expected_tree:
        raise RuntimeError(
            "TileLang package tree drift: "
            f"expected {expected_tree}, got {actual_tree} at {tilelang_root}"
        )
    actual_version = distribution.version
    expected_version = manifest["dependencies"]["tilelangVersion"]
    if actual_version != expected_version:
        raise RuntimeError(
            f"TileLang version drift: expected {expected_version}, got {actual_version}"
        )
    versions = {}
    for name in ("tilelang", "apache-tvm-ffi", "numpy"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return {
        "tilelangPackageRoot": str(tilelang_root),
        "tilelangPackageTreeSha256": actual_tree,
        "versions": versions,
    }


def load_historical_kernels():
    module_path = UPSTREAM / "onnx_kernels.py"
    spec = importlib.util.spec_from_file_location(
        "sm120_historical_ffn_onnx_kernels", module_path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load historical kernel module: {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def matching_brace(lines: list[str], start: int) -> int:
    depth = 0
    for index in range(start, len(lines)):
        depth += lines[index].count("{") - lines[index].count("}")
        if depth == 0:
            return index
    raise RuntimeError(f"unmatched generated brace at line {start + 1}")


def restrict_kernel_to_sm120(source: str, kernel_name: str) -> str:
    """Allow the candidate TU to coexist in KataGo's multi-arch fat binary."""

    lines = source.splitlines(keepends=True)
    definition = next(
        index
        for index, line in enumerate(lines)
        if kernel_name in line
        and "__global__" in line
        and line.rstrip().endswith("{")
    )
    end = matching_brace(lines, definition)
    lines.insert(
        definition + 1,
        "#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1200\n",
    )
    lines.insert(end + 1, "#endif\n")
    return "".join(lines)


def validate_historical_device_source(
    source: str, *, batch: int, kernel_name: str, manifest: dict[str, Any]
) -> dict[str, Any]:
    rows = batch * BOARD_AREA
    # TileLang's simplifier divides the output-row predicate by the greatest
    # common vector factor for even exact batches (for example B2 emits
    # `< 361` over paired rows rather than `< 722`).
    predicate_bound = rows // math.gcd(rows, BLOCK_M)
    expected_signature = (
        'extern "C" __global__ void __launch_bounds__(128, 3) '
        f"{kernel_name}(const half_t* __restrict__ data, "
        "const half_t* __restrict__ gate_weight, "
        "const half_t* __restrict__ linear_weight, "
        "half_t* __restrict__ output)"
    )
    required = (
        expected_signature,
        "tilelang_h2tanh_approx",
        "h2tanh_approx(input.value)",
        "tl::fma2",
        "tl::mul2",
        "kFloat16, 16, 8, 16",
        f"< {predicate_bound})",
    )
    missing = [marker for marker in required if marker not in source]
    if missing:
        raise RuntimeError(f"generated source lost historical markers: {missing}")
    forbidden = ("expf(", "T.exp", "/ (1.0 +")
    present = [marker for marker in forbidden if marker in source]
    if present:
        raise RuntimeError(
            "generated source is not the tanh-half2 path; forbidden exp markers: "
            f"{present}"
        )
    if source.count(kernel_name) != 2:
        raise RuntimeError("unexpected generated kernel symbol count")
    source_hash = sha256_bytes(source.encode("utf-8"))
    if batch == 19:
        expected = manifest["golden"]["b19RenamedDeviceSourceSha256"]
        if source_hash != expected:
            raise RuntimeError(
                "B19 historical device source is not byte reproducible: "
                f"expected {expected}, got {source_hash}"
            )
    return {
        "sourceSha256": source_hash,
        "sourceBytes": len(source.encode("utf-8")),
        "usesHalf2TanhApprox": True,
        "usesScalarExpf": False,
        "fp16MmaInstructionCount": source.count(
            "kFloat16, 16, 8, 16"
        ),
    }


def generate_device_source(
    kernels: Any, batch: int, manifest: dict[str, Any]
) -> tuple[str, str, dict[str, Any]]:
    import tvm

    rows = batch * BOARD_AREA
    # The historical function decorates its nested PrimFunc with
    # execution_backend="cython".  Calling that decorator constructs a runtime
    # adapter and probes a CUDA device after source generation.  Replace only
    # that nested decorator with an identity long enough to recover the exact
    # PrimFunc, then invoke TileLang's own lowering pipeline without host codegen
    # or device compilation.  This emits the same device source without CUDA
    # runtime initialization.
    original_jit = kernels.tilelang.jit

    def source_only_jit(*_args: Any, **_kwargs: Any):
        def decorate(builder: Any) -> Any:
            return builder

        return decorate

    kernels.tilelang.jit = source_only_jit
    try:
        prim_func = kernels.ffn_swiglu_intrinsics(
            rows,
            COLUMNS,
            INNER,
            BLOCK_M,
            BLOCK_N,
            BLOCK_K,
            STAGES,
            "float16",
            False,
            2,
            2,
            "swiglu_tanh_f16x2",
            False,
            MIN_BLOCKS_PER_SM,
        )
    finally:
        kernels.tilelang.jit = original_jit

    pass_configs = dict(kernels.FLASH_ATTENTION_PASS_CONFIGS)
    pass_configs[
        kernels.tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED
    ] = True
    target = tvm.target.Target(kernels.TARGET)
    with tvm.transform.PassContext(
        opt_level=3, config=pass_configs
    ), target:
        artifact = kernels.tilelang.lower(
            prim_func,
            target=target,
            enable_host_codegen=False,
            enable_device_compile=False,
        )
    source = artifact.kernel_source
    expected_original = (
        'extern "C" __global__ void __launch_bounds__(128, 3) main_kernel'
    )
    if expected_original not in source or source.count("main_kernel") != 2:
        raise RuntimeError("historical TileLang codegen changed its kernel signature")
    kernel_name = f"katago_ffn_tilelang_sm120_b{batch}_s361_kernel"
    source = source.replace("main_kernel", kernel_name)
    evidence = validate_historical_device_source(
        source, batch=batch, kernel_name=kernel_name, manifest=manifest
    )
    return source, kernel_name, evidence


def append_search_wrapper(
    device_source: str, *, batch: int, kernel_name: str,
    fat_symbol_token: str | None = None,
) -> str:
    grid_y = (batch * BOARD_AREA + BLOCK_M - 1) // BLOCK_M
    guarded = restrict_kernel_to_sm120(device_source, kernel_name)
    if fat_symbol_token is None:
        template = WRAPPER.replace("KATAGO_BATCH", str(batch))
    else:
        validate_symbol_token(fat_symbol_token)
        guarded = isolate_tilelang_debug_symbols(guarded, fat_symbol_token)
        template = FAT_WRAPPER.replace(
            "KATAGO_FAT_LAUNCH_SYMBOL",
            launch_symbol("ffn", fat_symbol_token),
        )
    wrapper = (
        template.replace("KATAGO_GRID_Y", str(grid_y))
        .replace("KATAGO_KERNEL_NAME", kernel_name)
    )
    return guarded + wrapper


def validate_space(path: Path, batch: int, candidate_id: str) -> None:
    space = json.loads(path.read_text(encoding="utf-8"))
    if space.get("schema") != 2:
        raise ValueError("--space must use schema 2")
    batch_space = next(
        (item for item in space.get("batches", []) if item.get("batch") == batch),
        None,
    )
    if batch_space is None:
        raise ValueError(f"B{batch} is outside --space")
    candidate = next(
        (item for item in batch_space.get("ffn", []) if item.get("id") == candidate_id),
        None,
    )
    if candidate is None:
        raise ValueError(f"{candidate_id} is outside B{batch} --space")
    expected = dict(CANDIDATE)
    implementation = candidate.get("implementation")
    expected.pop("implementation")
    if any(
        candidate.get(key, THREADS if key == "threads" else None) != value
        for key, value in expected.items()
    ) or (
        implementation not in ("tilelang", "historical_tilelang")
    ):
        raise ValueError(
            f"historical candidate definition drift in B{batch}: {candidate}"
        )


def metadata_for(
    *,
    batch: int,
    source_path: Path,
    source: str,
    device_evidence: dict[str, Any],
    manifest: dict[str, Any],
    dependency_evidence: dict[str, Any],
    fat_symbol_token: str | None,
) -> dict[str, Any]:
    rows = batch * BOARD_AREA
    grid_y = (rows + BLOCK_M - 1) // BLOCK_M
    return {
        "schema": 2,
        "family": "ffn",
        "candidate": CANDIDATE,
        "batch": batch,
        "tokens": rows,
        "fixedBoard": [19, 19],
        "shape": {"M": rows, "N": COLUMNS, "K": INNER},
        "launch": {
            "grid": [18, grid_y, 1],
            "block": [THREADS, 1, 1],
            "dynamicSharedMemoryBytes": DYNAMIC_SMEM_BYTES,
            "launchBounds": [THREADS, MIN_BLOCKS_PER_SM],
        },
        "arithmetic": {
            "input": "float16",
            "accumulation": "float16",
            "epilogue": "swiglu_tanh_f16x2",
            "tanhIntrinsic": "CUDA h2tanh_approx",
            "scalarExpf": False,
            "output": "float16",
            "operationOrder": "half2 linear * (half2_fma(h2tanh_approx(half2_linear * 0.5), 0.5, 0.5)) * half2 gate",
        },
        "abi": {
            "batchSymbol": (
                None if fat_symbol_token else "sm120_search_ffn_batch"
            ),
            "idSymbol": None if fat_symbol_token else "sm120_search_ffn_id",
            "launchSymbol": (
                launch_symbol("ffn", fat_symbol_token)
                if fat_symbol_token else "sm120_search_ffn_launch"
            ),
            "launchArguments": [
                "const half* input[M,K]",
                "const half* linear_weights[K,N]",
                "const half* gate_weights[K,N]",
                "half* output[M,N]",
                "cudaStream_t stream",
            ],
            "generatedKernelArgumentOrder": [
                "input",
                "gate_weights",
                "linear_weights",
                "output",
            ],
            "singleActiveTranslationUnit": fat_symbol_token is None,
        },
        "fat_symbol_token": fat_symbol_token,
        "launch_symbol": (
            launch_symbol("ffn", fat_symbol_token)
            if fat_symbol_token else "sm120_search_ffn_launch"
        ),
        "source": str(source_path.resolve()),
        "sourceSha256": sha256_bytes(source.encode("utf-8")),
        "deviceSource": device_evidence,
        "generatorSha256": sha256_path(Path(__file__).resolve()),
        "historicalSources": manifest["historicalSources"],
        "dependencyEvidence": dependency_evidence,
        "compileContract": {
            "language": "CUDA C++20",
            "target": "sm_120",
            "requiredIncludeRoots": [
                "${tilelangPackageRoot}/3rdparty/cutlass/include",
                "${tilelangPackageRoot}/src",
            ],
            "suggestedFlags": [
                "-std=c++20",
                "-O3",
                "-w",
                "-Xcudafe",
                "--diag_suppress=177",
                "-lineinfo",
                "-gencode",
                "arch=compute_120,code=sm_120",
            ],
        },
        "codegenUsesGpu": False,
        "sourceOnlyLowering": True,
        "acceptanceMetric": "natural whole-graph S2 total throughput",
    }


def emit_one(
    *,
    kernels: Any,
    batch: int,
    output_dir: Path,
    source_path: Path | None,
    space: Path | None,
    candidate_id: str,
    manifest: dict[str, Any],
    dependency_evidence: dict[str, Any],
    fat_symbol_token: str | None = None,
) -> dict[str, Any]:
    if not 1 <= batch <= 32:
        raise ValueError(f"batch must be in B1..B32, got B{batch}")
    if candidate_id != CANDIDATE_ID:
        raise ValueError(
            f"this frozen generator only emits {CANDIDATE_ID}, got {candidate_id}"
        )
    if space is not None:
        validate_space(space, batch, candidate_id)
    device_source, kernel_name, device_evidence = generate_device_source(
        kernels, batch, manifest
    )
    source = append_search_wrapper(
        device_source, batch=batch, kernel_name=kernel_name,
        fat_symbol_token=fat_symbol_token,
    )
    actual_source_path = (
        source_path.resolve()
        if source_path is not None
        else output_dir / SOURCE_NAME
    )
    metadata_path = output_dir / METADATA_NAME
    write_if_changed(actual_source_path, source)
    metadata = metadata_for(
        batch=batch,
        source_path=actual_source_path,
        source=source,
        device_evidence=device_evidence,
        manifest=manifest,
        dependency_evidence=dependency_evidence,
        fat_symbol_token=fat_symbol_token,
    )
    write_if_changed(
        metadata_path,
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
    )
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--batch", type=int)
    group.add_argument("--all-batches", action="store_true")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--source-path",
        type=Path,
        help="stable active-slot path; valid only with one --batch",
    )
    parser.add_argument("--space", type=Path)
    parser.add_argument("--family", choices=("ffn",), default="ffn")
    parser.add_argument("--candidate-id", default=CANDIDATE_ID)
    parser.add_argument("--fat-symbol-token")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.fat_symbol_token:
        validate_symbol_token(args.fat_symbol_token)
    if args.all_batches and args.source_path is not None:
        raise ValueError("--source-path cannot be combined with --all-batches")
    if args.all_batches and args.fat_symbol_token is not None:
        raise ValueError("--fat-symbol-token requires one exact --batch")
    manifest = load_manifest()
    dependency_evidence = verify_dependencies(manifest)
    kernels = load_historical_kernels()
    batches = range(1, 33) if args.all_batches else (args.batch,)
    generated = []
    for batch in batches:
        output_dir = (
            args.output_dir / f"b{batch}-ffn"
            if args.all_batches
            else args.output_dir
        )
        generated.append(
            emit_one(
                kernels=kernels,
                batch=batch,
                output_dir=output_dir,
                source_path=args.source_path,
                space=args.space,
                candidate_id=args.candidate_id,
                manifest=manifest,
                dependency_evidence=dependency_evidence,
                fat_symbol_token=args.fat_symbol_token,
            )
        )
    summary = {
        "schema": 1,
        "candidateId": CANDIDATE_ID,
        "batches": [item["batch"] for item in generated],
        "artifacts": [
            {
                "batch": item["batch"],
                "source": item["source"],
                "sourceSha256": item["sourceSha256"],
            }
            for item in generated
        ],
        "codegenUsesGpu": False,
    }
    if args.all_batches:
        write_if_changed(
            args.output_dir / "manifest-b1-b32.json",
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
        )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
