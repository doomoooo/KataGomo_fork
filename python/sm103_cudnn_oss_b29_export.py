#!/usr/bin/env python3
"""Export the exact B29 cuDNN OSS SwiGLU kernel through a native C ABI.

Without ``--export`` this module is device-free and emits only an export plan.
Actual CuTe compilation, linking, and launch validation require both ``--export``
and ``--allow-gpu``.  The generated object embeds the SM103 cubin and links to
the static CUDA-dialect runtime; Python and TVM-FFI are not runtime dependencies.
"""

from __future__ import annotations

import argparse
import ctypes
import datetime
import hashlib
import importlib.metadata
import json
import os
import pathlib
import subprocess
import sys
import time
from typing import Any

try:
    import sm103_cudnn_oss_b29 as candidate
except ModuleNotFoundError:
    from python import sm103_cudnn_oss_b29 as candidate


ARTIFACT_STEM = "katago_sm103_b29_cudnn_swiglu"
EXPORT_FUNCTION_PREFIX = ARTIFACT_STEM
UPSTREAM_NUMERIC_SEMANTICS = "upstream-fp32-projection"
VARIANT_A_NUMERIC_SEMANTICS = "projection-fp16-roundtrip"
VARIANT_B_NUMERIC_SEMANTICS = "projection-fp16-roundtrip-precise-math"
VARIANT_C_NUMERIC_SEMANTICS = "projection-fp16-roundtrip-newton1"
NO_AB12_NUMERIC_SEMANTICS = "projection-fp16-roundtrip-no-ab12"
ROUNDTRIP_NUMERIC_SEMANTICS = (
    VARIANT_A_NUMERIC_SEMANTICS,
    VARIANT_B_NUMERIC_SEMANTICS,
    VARIANT_C_NUMERIC_SEMANTICS,
    NO_AB12_NUMERIC_SEMANTICS,
)
NUMERIC_SEMANTICS_CHOICES = (
    UPSTREAM_NUMERIC_SEMANTICS,
    *ROUNDTRIP_NUMERIC_SEMANTICS,
)
BRIDGE_HEADER_RELATIVE = pathlib.Path(
    "cpp/neuralnet/cudnn_oss_b29_aot_bridge.h"
)
BRIDGE_SOURCE_RELATIVE = pathlib.Path(
    "cpp/neuralnet/cudnn_oss_b29_aot_bridge.cpp"
)


class CudnnOssExportError(RuntimeError):
    """Raised when an AOT artifact cannot be generated or verified safely."""


def _sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _repo_root() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parents[1]


def _candidate_for_numeric_semantics(numeric_semantics: str) -> Any:
    if numeric_semantics == UPSTREAM_NUMERIC_SEMANTICS:
        return candidate
    if numeric_semantics == VARIANT_A_NUMERIC_SEMANTICS:
        try:
            import sm103_cudnn_oss_b29_roundtrip as roundtrip
        except ModuleNotFoundError:
            from python import sm103_cudnn_oss_b29_roundtrip as roundtrip
        return roundtrip
    if numeric_semantics == VARIANT_B_NUMERIC_SEMANTICS:
        try:
            import sm103_cudnn_oss_b29_precise as precise
        except ModuleNotFoundError:
            from python import sm103_cudnn_oss_b29_precise as precise
        return precise
    if numeric_semantics == VARIANT_C_NUMERIC_SEMANTICS:
        try:
            import sm103_cudnn_oss_b29_newton as newton
        except ModuleNotFoundError:
            from python import sm103_cudnn_oss_b29_newton as newton
        return newton
    if numeric_semantics == NO_AB12_NUMERIC_SEMANTICS:
        try:
            import sm103_cudnn_oss_b29_no_ab12 as no_ab12
        except ModuleNotFoundError:
            from python import sm103_cudnn_oss_b29_no_ab12 as no_ab12
        return no_ab12
    raise CudnnOssExportError(
        f"unsupported numeric semantics {numeric_semantics!r}; expected one of "
        f"{NUMERIC_SEMANTICS_CHOICES!r}"
    )


def _file_evidence(path: pathlib.Path, repo_root: pathlib.Path) -> dict[str, Any]:
    return {
        "path": str(path.relative_to(repo_root)),
        "sha256": _sha256(path),
        "bytes": path.stat().st_size,
    }


def build_export_plan(
    repo_root: pathlib.Path | None = None,
    numeric_semantics: str = UPSTREAM_NUMERIC_SEMANTICS,
) -> dict[str, Any]:
    """Describe the AOT boundary without importing any GPU Python package."""

    if repo_root is None:
        repo_root = _repo_root()
    selected_candidate = _candidate_for_numeric_semantics(numeric_semantics)
    provider = candidate.inspect_installed_provider()
    kernel_manifest = selected_candidate.validate_candidate_manifest(
        selected_candidate.build_candidate_manifest(
            provider=provider, repo_root=repo_root
        )
    )
    bridge_header = repo_root / BRIDGE_HEADER_RELATIVE
    bridge_source = repo_root / BRIDGE_SOURCE_RELATIVE
    if not bridge_header.is_file() or not bridge_source.is_file():
        raise CudnnOssExportError("checked-in AOT bridge sources are missing")
    payload = {
        "schema": 1,
        "kind": "katago-sm103-b29-cudnn-oss-aot-export-plan",
        "candidate_id": selected_candidate.CANDIDATE_ID,
        "numeric_semantics_selector": numeric_semantics,
        "numeric_semantics": kernel_manifest["operation"].get(
            "numeric_semantics",
            {
                "id": UPSTREAM_NUMERIC_SEMANTICS,
                "accumulation": "tcgen05 FP32",
                "projection_boundary": "none before fast SwiGLU",
            },
        ),
        "artifact_stem": ARTIFACT_STEM,
        "compile_target": "sm_103a",
        "compile_options": ["--gpu-arch=sm_103a"],
        "export_mode": "cute_native_c_header_and_pic_object",
        "tvm_ffi": False,
        "runtime_requires_python": False,
        "runtime_requires_tvm_ffi": False,
        "runtime_requires_cuda_13": True,
        "kernel_manifest_provider": kernel_manifest["provider"],
        "bridge_sources": [
            _file_evidence(bridge_header, repo_root),
            _file_evidence(bridge_source, repo_root),
        ],
        "expected_artifacts": [
            f"{ARTIFACT_STEM}.h",
            f"{ARTIFACT_STEM}.o",
            f"lib{ARTIFACT_STEM}.so",
            "aot-manifest.json",
        ],
        "c_abi": {
            "create": "katagoCudnnOssB29Create",
            "launch": "katagoCudnnOssB29Launch",
            "destroy": "katagoCudnnOssB29Destroy",
            "fixed_problem": "R10469 K384 N2304->1152 FP16",
            "caller_owned_buffers": ["A", "packed_B", "AB12", "C"],
        },
        "production_ready": False,
    }
    if numeric_semantics in ROUNDTRIP_NUMERIC_SEMANTICS:
        payload["derivative"] = kernel_manifest["static_support"]["derivative"]
        payload["expected_artifacts"].extend(
            [
                selected_candidate.DERIVATIVE_FILENAME,
                selected_candidate.DERIVATIVE_PROVENANCE_FILENAME,
            ]
        )
    return payload


def _ensure_export_target(output_dir: pathlib.Path, force: bool) -> pathlib.Path:
    resolved = output_dir.resolve()
    if resolved in (pathlib.Path("/"), _repo_root()):
        raise CudnnOssExportError(f"unsafe AOT output directory: {resolved}")
    known = (
        resolved / f"{ARTIFACT_STEM}.h",
        resolved / f"{ARTIFACT_STEM}.o",
        resolved / f"lib{ARTIFACT_STEM}.so",
        resolved / "aot-manifest.json",
    )
    existing = [str(path) for path in known if path.exists()]
    if existing and not force:
        raise CudnnOssExportError(
            "refusing to overwrite existing AOT artifacts: " + ", ".join(existing)
        )
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def _site_packages_root(distribution_name: str) -> pathlib.Path:
    distribution = importlib.metadata.distribution(distribution_name)
    return pathlib.Path(distribution.locate_file("")).resolve()


def _artifact_evidence(path: pathlib.Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "sha256": _sha256(path),
        "bytes": path.stat().st_size,
    }


def _link_bridge(
    *,
    output_dir: pathlib.Path,
    object_path: pathlib.Path,
    cxx: str,
) -> tuple[pathlib.Path, list[str], pathlib.Path]:
    repo_root = _repo_root()
    site_packages = _site_packages_root("nvidia-cutlass-dsl")
    runtime_archive = (
        site_packages
        / "nvidia_cutlass_dsl/cu13/lib/libcuda_dialect_runtime_static.a"
    )
    cuda_root = site_packages / "nvidia/cu13"
    bridge_source = repo_root / BRIDGE_SOURCE_RELATIVE
    library_path = output_dir / f"lib{ARTIFACT_STEM}.so"
    for required in (
        runtime_archive,
        cuda_root / "include/cuda_runtime.h",
        cuda_root / "lib/libcudart.so",
        bridge_source,
    ):
        if not required.exists():
            raise CudnnOssExportError(f"AOT link dependency is missing: {required}")
    command = [
        cxx,
        "-std=c++17",
        "-O2",
        "-Wall",
        "-Wextra",
        "-Wpedantic",
        "-Werror",
        "-fPIC",
        "-shared",
        f"-I{output_dir}",
        f"-I{repo_root / 'cpp/neuralnet'}",
        f"-I{cuda_root / 'include'}",
        str(bridge_source),
        str(object_path),
        str(runtime_archive),
        f"-L{cuda_root / 'lib'}",
        "-lcudart",
        "-ldl",
        "-pthread",
        "-Wl,-z,defs",
        "-o",
        str(library_path),
    ]
    subprocess.run(command, check=True, text=True)
    return library_path, command, runtime_archive


def _validate_bridge_launch(
    *,
    library_path: pathlib.Path,
    device: int,
    seed: int,
    numeric_semantics: str = UPSTREAM_NUMERIC_SEMANTICS,
) -> dict[str, Any]:
    import torch

    torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = False
    tensors = candidate._allocate_gpu_benchmark_inputs(torch, device, seed)
    selected_candidate = _candidate_for_numeric_semantics(numeric_semantics)
    signal_scaling: dict[str, float] | None = None
    if numeric_semantics in ROUNDTRIP_NUMERIC_SEMANTICS:
        # Use exactly the same deterministic, non-vacuous probe as the
        # standalone Variant-A validator.  In particular, validating a nearly
        # zero random output with a loose allclose tolerance is not evidence
        # that either C or the mandatory AB12 projection scratch is correct.
        signal_scaling = selected_candidate.strengthen_probe_signal(
            torch, tensors
        )
    problem: candidate.DenseSwiGLUProblem = tensors["problem"]
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
    # Finish fixture packing before native module/context creation.  The
    # sentinel fills below are then issued after Create on the exact stream
    # passed to Launch, so context/stream transitions cannot reorder them.
    torch.cuda.synchronize(device)
    library = ctypes.CDLL(str(library_path.resolve()), mode=ctypes.RTLD_LOCAL)
    library.katagoCudnnOssB29Create.argtypes = [
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_int32),
    ]
    library.katagoCudnnOssB29Create.restype = ctypes.c_void_p
    library.katagoCudnnOssB29Launch.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_float,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
    ]
    library.katagoCudnnOssB29Launch.restype = ctypes.c_int32
    library.katagoCudnnOssB29Destroy.argtypes = [ctypes.c_void_p]
    library.katagoCudnnOssB29Destroy.restype = None

    status = ctypes.c_int32(-999)
    context = library.katagoCudnnOssB29Create(device, ctypes.byref(status))
    if not context or status.value != 0:
        raise CudnnOssExportError(
            f"C ABI context creation failed with status {status.value}"
        )
    try:
        stream = torch.cuda.current_stream(device)
        # A deterministic non-finite sentinel makes a missing/ragged-tail
        # store a hard failure instead of comparing allocator residue.
        with torch.cuda.stream(stream):
            ab12.fill_(float("nan"))
            output.fill_(float("nan"))
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
            raise CudnnOssExportError(
                f"C ABI launch failed with status {launch_status}"
            )
        torch.cuda.synchronize(device)
        actual = output[:, :, 0].float()
        if numeric_semantics in ROUNDTRIP_NUMERIC_SEMANTICS:
            reference = selected_candidate.projection_fp16_roundtrip_reference(
                torch,
                tensors["input_2d"],
                tensors["linear1"],
                tensors["linear_gate"],
            )
            try:
                if numeric_semantics == NO_AB12_NUMERIC_SEMANTICS:
                    tight_summary = selected_candidate.build_gpu_correctness_summary(
                        torch,
                        actual_output=actual,
                        reference_output=reference,
                        ab12_untouched=bool(torch.isnan(ab12).all().item()),
                    )
                else:
                    packed_reference = (
                        tensors["input_2d"].float()
                        @ tensors["b_tensor"][:, :, 0].float().transpose(0, 1)
                    ).half()
                    tight_summary = selected_candidate.build_gpu_correctness_summary(
                        torch,
                        actual_output=actual,
                        reference_output=reference,
                        actual_ab12=ab12[:, :, 0],
                        reference_ab12=packed_reference,
                    )
            except RuntimeError as error:
                raise CudnnOssExportError(
                    "C ABI projection-roundtrip tight correctness failed: "
                    f"{error}"
                ) from error
            return {
                "status": "passed",
                "numeric_semantics_selector": numeric_semantics,
                "same_seed": seed,
                "signal_scaling": signal_scaling,
                "tight_correctness": tight_summary,
                "ab12_shape": list(ab12.shape),
                "output_shape": list(output.shape),
            }
        else:
            linear1_projection = (
                tensors["input_2d"].float()
                @ tensors["linear1"].float().transpose(0, 1)
            )
            gate_projection = (
                tensors["input_2d"].float()
                @ tensors["linear_gate"].float().transpose(0, 1)
            )
            reference = (
                torch.nn.functional.silu(linear1_projection) * gate_projection
            )
        difference = (actual - reference).abs()
        max_abs = float(difference.max().item())
        max_rel = float(
            (difference / reference.abs().clamp_min(1.0e-3)).max().item()
        )
        passed = bool(torch.allclose(actual, reference, atol=2.0e-2, rtol=5.0e-2))
        if not passed:
            raise CudnnOssExportError(
                f"C ABI correctness failed: max_abs={max_abs}, max_rel={max_rel}"
            )
        return {
            "status": "passed",
            "numeric_semantics_selector": numeric_semantics,
            "max_abs_error": max_abs,
            "max_rel_error": max_rel,
            "ab12_shape": list(ab12.shape),
            "output_shape": list(output.shape),
        }
    finally:
        # Launch is asynchronous.  Keep the CUDA module/context alive until
        # all work that can reference it has completed, including exceptional
        # paths after a successful launch.
        try:
            torch.cuda.synchronize(device)
        finally:
            library.katagoCudnnOssB29Destroy(context)


def _validate_bridge_launch_isolated(
    *,
    library_path: pathlib.Path,
    device: int,
    seed: int,
    numeric_semantics: str,
) -> dict[str, Any]:
    """Validate in a fresh process where the linked DSO is the only module.

    ``cute.compile`` loads a JIT module into the exporting process.  Loading an
    exported DSO with the same generated symbol prefix in that process makes
    launch validation vulnerable to symbol/module aliasing.  A fresh child has
    no JIT module and therefore exercises exactly the standalone C++ runtime.
    """

    command = [
        sys.executable,
        str(pathlib.Path(__file__).resolve()),
        "--validate-library",
        str(library_path.resolve()),
        "--allow-gpu",
        "--device",
        str(device),
        "--seed",
        str(seed),
        "--numeric-semantics",
        numeric_semantics,
    ]
    completed = subprocess.run(
        command,
        cwd=_repo_root(),
        check=False,
        text=True,
        capture_output=True,
    )
    if completed.returncode != 0:
        raise CudnnOssExportError(
            "isolated C ABI launch validation failed:\n" + completed.stderr
        )
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise CudnnOssExportError(
            "isolated C ABI launch validation emitted invalid JSON"
        ) from error
    if not isinstance(payload, dict) or payload.get("status") != "passed":
        raise CudnnOssExportError(
            "isolated C ABI launch validation did not pass"
        )
    return payload


def export_aot(
    *,
    allow_gpu: bool,
    device: int,
    output_dir: pathlib.Path,
    force: bool = False,
    cxx: str = "g++",
    validate_launch: bool = True,
    seed: int = 20260818,
    numeric_semantics: str = UPSTREAM_NUMERIC_SEMANTICS,
) -> dict[str, Any]:
    """Compile, export, link, and optionally launch-check the native bridge."""

    if not allow_gpu:
        raise CudnnOssExportError(
            "AOT export requires the explicit --allow-gpu acknowledgement"
        )
    if type(device) is not int or device < 0:  # noqa: E721
        raise CudnnOssExportError("device must be a non-negative integer")
    selected_candidate = _candidate_for_numeric_semantics(numeric_semantics)
    plan = build_export_plan(numeric_semantics=numeric_semantics)
    target = _ensure_export_target(output_dir, force)

    import torch
    import cutlass
    import cutlass.cute as cute
    from cutlass.cute.runtime import make_fake_stream
    from cudnn.datatypes import _convert_to_cutlass_data_type
    from cudnn.gemm.cutedsl.dense.swiglu.api import GemmSwigluSm100
    from cudnn.gemm.cutedsl.dense.swiglu.dense_gemm_persistent_swiglu import (
        PersistentDenseGemmKernel as UpstreamPersistentDenseGemmKernel,
    )

    torch.cuda.set_device(device)
    torch.cuda.init()
    capability = tuple(torch.cuda.get_device_capability(device))
    if capability != candidate.COMPUTE_CAPABILITY:
        raise CudnnOssExportError(
            f"AOT export requires exact SM103, got {capability!r}"
        )
    problem = candidate.DenseSwiGLUProblem()

    def meta_tensor(shape: tuple[int, ...], stride: tuple[int, ...]) -> Any:
        return torch.empty_strided(
            shape,
            stride,
            dtype=torch.float16,
            device="meta",
        )

    contract = problem.tensor_contract
    a = meta_tensor(tuple(contract["a"]["shape"]), tuple(contract["a"]["stride"]))
    b = meta_tensor(tuple(contract["b"]["shape"]), tuple(contract["b"]["stride"]))
    ab12 = meta_tensor(
        tuple(contract["ab12"]["shape"]), tuple(contract["ab12"]["stride"])
    )
    c = meta_tensor(tuple(contract["c"]["shape"]), tuple(contract["c"]["stride"]))
    api = GemmSwigluSm100(
        a,
        b,
        ab12,
        c,
        acc_dtype=torch.float32,
        mma_tiler_mn=candidate.MMA_TILER_MN,
        cluster_shape_mn=candidate.CLUSTER_SHAPE_MN,
    )
    if not api.check_support():
        raise CudnnOssExportError("upstream API rejected the exact B29 signature")
    derivative_evidence: dict[str, Any] | None = None
    derivative_artifacts: dict[str, Any] | None = None
    if numeric_semantics in ROUNDTRIP_NUMERIC_SEMANTICS:
        (
            kernel_class,
            derivative,
            derivative_source,
            derivative_provenance,
        ) = selected_candidate.load_derivative_kernel_class(target)
        derivative_evidence = derivative.to_dict()
        derivative_artifacts = {
            "source": _artifact_evidence(derivative_source),
            "provenance": _artifact_evidence(derivative_provenance),
        }
    else:
        kernel_class = UpstreamPersistentDenseGemmKernel
    gemm = kernel_class(
        acc_dtype=_convert_to_cutlass_data_type(torch.float32),
        use_2cta_instrs=False,
        mma_tiler_mn=candidate.MMA_TILER_MN,
        cluster_shape_mn=candidate.CLUSTER_SHAPE_MN,
    )
    max_active_clusters = cutlass.utils.HardwareInfo().get_max_active_clusters(1)
    fake_stream = make_fake_stream(use_tvm_ffi_env_stream=False)
    compile_started = time.perf_counter()
    compiled = cute.compile(
        gemm,
        a=api._make_fake_cute_tensor_from_desc(api.a_desc),
        b=api._make_fake_cute_tensor_from_desc(api.b_desc),
        ab12=api._make_fake_cute_tensor_from_desc(api.ab12_desc),
        c=api._make_fake_cute_tensor_from_desc(api.c_desc),
        alpha=1.0,
        max_active_clusters=max_active_clusters,
        stream=fake_stream,
        options="--gpu-arch=sm_103a",
    )
    compile_seconds = time.perf_counter() - compile_started
    compiled.export_to_c(
        file_path=str(target),
        file_name=ARTIFACT_STEM,
        function_prefix=EXPORT_FUNCTION_PREFIX,
    )
    header_path = target / f"{ARTIFACT_STEM}.h"
    object_path = target / f"{ARTIFACT_STEM}.o"
    if not header_path.is_file() or not object_path.is_file():
        raise CudnnOssExportError("export_to_c did not emit the expected artifacts")
    header = header_path.read_text(encoding="utf-8")
    expected_tokens = (
        f"{ARTIFACT_STEM}_Kernel_Module_Load",
        f"cute_dsl_{ARTIFACT_STEM}_wrapper",
        f"{ARTIFACT_STEM}_Tensor_a_t",
        f"{ARTIFACT_STEM}_Tensor_b_t",
        f"{ARTIFACT_STEM}_Tensor_ab12_t",
        f"{ARTIFACT_STEM}_Tensor_c_t",
        "void *args[7]",
    )
    missing_tokens = [token for token in expected_tokens if token not in header]
    if missing_tokens:
        raise CudnnOssExportError(
            "generated C ABI changed; missing: " + ", ".join(missing_tokens)
        )

    library_path, link_command, runtime_archive = _link_bridge(
        output_dir=target,
        object_path=object_path,
        cxx=cxx,
    )
    dynamic_symbols = subprocess.run(
        ["nm", "-D", str(library_path)],
        check=True,
        text=True,
        capture_output=True,
    ).stdout
    for symbol in (
        "katagoCudnnOssB29Create",
        "katagoCudnnOssB29Launch",
        "katagoCudnnOssB29Destroy",
    ):
        if symbol not in dynamic_symbols:
            raise CudnnOssExportError(f"linked bridge is missing symbol {symbol}")

    launch_validation: dict[str, Any]
    if validate_launch:
        launch_validation = _validate_bridge_launch_isolated(
            library_path=library_path,
            device=device,
            seed=seed,
            numeric_semantics=numeric_semantics,
        )
    else:
        launch_validation = {"status": "not_run"}

    payload = {
        **plan,
        "kind": "katago-sm103-b29-cudnn-oss-aot-artifact",
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z"),
        "device": {
            "ordinal": device,
            "name": torch.cuda.get_device_name(device),
            "compute_capability": list(capability),
        },
        "compile_seconds": compile_seconds,
        "max_active_clusters": max_active_clusters,
        "artifacts": {
            "header": _artifact_evidence(header_path),
            "object": _artifact_evidence(object_path),
            "bridge_shared_library": _artifact_evidence(library_path),
        },
        "link": {
            "command": link_command,
            "static_cuda_dialect_runtime": _artifact_evidence(runtime_archive),
            "python_runtime_dependency": False,
            "tvm_ffi_runtime_dependency": False,
        },
        "launch_validation": launch_validation,
        "production_ready": False,
    }
    if derivative_evidence is not None and derivative_artifacts is not None:
        payload["derivative"] = {
            "evidence": derivative_evidence,
            "artifacts": derivative_artifacts,
        }
    manifest_path = target / "aot-manifest.json"
    manifest_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export", action="store_true")
    parser.add_argument("--validate-library", type=pathlib.Path)
    parser.add_argument("--allow-gpu", action="store_true")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260818)
    parser.add_argument("--output-dir", type=pathlib.Path)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--cxx", default=os.environ.get("CXX", "g++"))
    parser.add_argument(
        "--numeric-semantics",
        choices=NUMERIC_SEMANTICS_CHOICES,
        default=UPSTREAM_NUMERIC_SEMANTICS,
    )
    parser.add_argument(
        "--validate-launch",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--manifest-output", type=pathlib.Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.validate_library is not None:
        if args.export or args.output_dir is not None or args.force:
            raise CudnnOssExportError(
                "--validate-library cannot be combined with export options"
            )
        if not args.allow_gpu:
            raise CudnnOssExportError(
                "--validate-library requires explicit --allow-gpu"
            )
        payload = _validate_bridge_launch(
            library_path=args.validate_library,
            device=args.device,
            seed=args.seed,
            numeric_semantics=args.numeric_semantics,
        )
    elif args.export:
        if args.output_dir is None:
            raise CudnnOssExportError("--export requires --output-dir")
        payload = export_aot(
            allow_gpu=args.allow_gpu,
            device=args.device,
            output_dir=args.output_dir,
            force=args.force,
            cxx=args.cxx,
            validate_launch=args.validate_launch,
            seed=args.seed,
            numeric_semantics=args.numeric_semantics,
        )
    else:
        if args.allow_gpu:
            raise CudnnOssExportError(
                "--allow-gpu requires --export or --validate-library"
            )
        if args.output_dir is not None or args.force:
            raise CudnnOssExportError(
                "--output-dir/--force are valid only with --export"
            )
        payload = build_export_plan(numeric_semantics=args.numeric_semantics)
    serialized = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.manifest_output is not None:
        args.manifest_output.write_text(serialized, encoding="utf-8")
    sys.stdout.write(serialized)
    return 0


__all__ = (
    "ARTIFACT_STEM",
    "CudnnOssExportError",
    "EXPORT_FUNCTION_PREFIX",
    "NUMERIC_SEMANTICS_CHOICES",
    "NO_AB12_NUMERIC_SEMANTICS",
    "UPSTREAM_NUMERIC_SEMANTICS",
    "VARIANT_A_NUMERIC_SEMANTICS",
    "VARIANT_B_NUMERIC_SEMANTICS",
    "VARIANT_C_NUMERIC_SEMANTICS",
    "build_export_plan",
    "build_parser",
    "export_aot",
)


if __name__ == "__main__":
    raise SystemExit(main())
