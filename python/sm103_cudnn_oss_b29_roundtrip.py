#!/usr/bin/env python3
"""B29 cuDNN OSS SwiGLU Variant A: FP16 projection round-trip.

This module is device-free unless ``--aot-benchmark --allow-gpu`` is selected.
It derives one kernel source file from the exact audited cuDNN Frontend 1.27.0
Apache-2.0 source.  Derivation is fail-closed: the upstream file SHA-256 and the
single replaced epilogue block must both match, and site-packages is never
modified.

Variant A changes one numerical factor only.  tcgen05 still accumulates in
FP32, but each gate/linear1 projection is rounded FP32 -> FP16 (round-to-nearest
even, the CUTLASS DSL ``TensorSSA.to`` default) -> FP32 before the unchanged
fast exp2+rcp SwiGLU epilogue.  The same FP16 values are written to AB12.
"""

from __future__ import annotations

import argparse
import ctypes
import datetime
from dataclasses import asdict, dataclass
import hashlib
import importlib.util
import json
import math
import pathlib
import statistics
import sys
from types import ModuleType
from typing import Any

try:
    import sm103_cudnn_oss_b29 as upstream_candidate
except ModuleNotFoundError:
    from python import sm103_cudnn_oss_b29 as upstream_candidate


CANDIDATE_ID = (
    "cudnn-fe-1_27-oss-dense-gemm-swiglu-proj-fp16-roundtrip-b29"
)
NUMERIC_SEMANTICS_ID = "projection-fp16-rne-roundtrip-before-fast-swiglu"
UPSTREAM_KERNEL_RELATIVE_PATH = (
    "cudnn/gemm/cutedsl/dense/swiglu/dense_gemm_persistent_swiglu.py"
)
UPSTREAM_KERNEL_SHA256 = upstream_candidate.SOURCE_IDENTITIES[
    UPSTREAM_KERNEL_RELATIVE_PATH
]
UPSTREAM_DISTRIBUTION = upstream_candidate.PROVIDER_DISTRIBUTION
UPSTREAM_VERSION = upstream_candidate.PROVIDER_VERSION
UPSTREAM_LICENSE = "Apache-2.0"
UPSTREAM_PROJECT = "NVIDIA cuDNN Frontend"
DERIVATIVE_FILENAME = "dense_gemm_persistent_swiglu_variant_a.py"
DERIVATIVE_PROVENANCE_FILENAME = "variant-a-provenance.json"
PROBE_INPUT_SCALE = 4.0
PROBE_WEIGHT_SCALE = 4.0
CORRECTNESS_LIMITS = {
    "reference_max_abs_minimum": 1.0e-2,
    "reference_rms_minimum": 1.0e-3,
    "output_max_abs_maximum": 5.0e-4,
    "output_rmse_maximum": 5.0e-5,
    "ab12_max_abs_maximum": 5.0e-4,
    "ab12_rmse_maximum": 5.0e-5,
}

# Keep the match intentionally narrow.  These exact lines occur once in the
# audited source, so an upstream edit or an accidentally broadened patch fails.
UPSTREAM_EPILOGUE_BLOCK = """\
                    acc_vec0 = acc_vec0 * alpha
                    acc_vec1 = acc_vec1 * alpha
                    # Use exp2 with log2(e) conversion since cute.math.exp is not available
"""

VARIANT_A_EPILOGUE_BLOCK = """\
                    acc_vec0 = acc_vec0 * alpha
                    acc_vec1 = acc_vec1 * alpha

                    # KataGo B29 Variant A: reproduce the legacy FP16 GEMM
                    # projection boundary before the otherwise unchanged
                    # FP32 fast-exp2/rcp SwiGLU epilogue. TensorSSA.to uses
                    # arith.truncf's default round-to-nearest-even mode.
                    acc_vec0_ab12 = acc_vec0.to(self.ab12_dtype)
                    acc_vec1_ab12 = acc_vec1.to(self.ab12_dtype)
                    acc_vec0 = acc_vec0_ab12.to(self.acc_dtype)
                    acc_vec1 = acc_vec1_ab12.to(self.acc_dtype)
                    # Use exp2 with log2(e) conversion since cute.math.exp is not available
"""

UPSTREAM_AB12_CAST_BLOCK = """\
                    acc_vec0 = (acc_vec0).to(self.ab12_dtype)
                    acc_vec1 = (acc_vec1).to(self.ab12_dtype)

                    tRS_rAB12.store(acc_vec0)  # both of them are pure Gemm Output.
"""

VARIANT_A_AB12_REUSE_BLOCK = """\
                    acc_vec0 = acc_vec0_ab12
                    acc_vec1 = acc_vec1_ab12

                    tRS_rAB12.store(acc_vec0)  # both of them are pure Gemm Output.
"""


class CudnnOssRoundtripError(RuntimeError):
    """Raised when Variant A cannot be derived, loaded, or measured safely."""


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


PATCH_SPEC_SHA256 = _sha256_bytes(
    (
        UPSTREAM_EPILOGUE_BLOCK
        + "\0"
        + VARIANT_A_EPILOGUE_BLOCK
        + "\0"
        + UPSTREAM_AB12_CAST_BLOCK
        + "\0"
        + VARIANT_A_AB12_REUSE_BLOCK
    ).encode("utf-8")
)


@dataclass(frozen=True)
class DerivativeEvidence:
    upstream_project: str
    upstream_distribution: str
    upstream_version: str
    upstream_relative_path: str
    upstream_installed_path: str
    upstream_sha256: str
    upstream_license: str
    patch_spec_sha256: str
    derivative_sha256: str
    numeric_semantics_id: str
    site_packages_modified: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _provider_kernel_path(
    provider: upstream_candidate.ProviderEvidence,
) -> pathlib.Path:
    if not provider.verified:
        raise CudnnOssRoundtripError(
            "exact cuDNN Frontend provider evidence is required for Variant A"
        )
    for source in provider.sources:
        if source.relative_path == UPSTREAM_KERNEL_RELATIVE_PATH:
            if source.installed_path is None:
                break
            return pathlib.Path(source.installed_path)
    raise CudnnOssRoundtripError("audited upstream SwiGLU kernel path is missing")


def derive_variant_a_source(upstream_source: bytes) -> bytes:
    """Apply the two exact Variant A substitutions to audited source bytes."""

    actual_sha256 = _sha256_bytes(upstream_source)
    if actual_sha256 != UPSTREAM_KERNEL_SHA256:
        raise CudnnOssRoundtripError(
            "upstream kernel SHA-256 mismatch: expected "
            f"{UPSTREAM_KERNEL_SHA256}, got {actual_sha256}"
        )
    try:
        source = upstream_source.decode("utf-8")
    except UnicodeDecodeError as error:
        raise CudnnOssRoundtripError("upstream kernel is not UTF-8") from error

    replacements = (
        (UPSTREAM_EPILOGUE_BLOCK, VARIANT_A_EPILOGUE_BLOCK),
        (UPSTREAM_AB12_CAST_BLOCK, VARIANT_A_AB12_REUSE_BLOCK),
    )
    for old, new in replacements:
        count = source.count(old)
        if count != 1:
            raise CudnnOssRoundtripError(
                f"Variant A patch context must occur exactly once, got {count}"
            )
        source = source.replace(old, new, 1)

    # These assertions make the one-factor intent machine-checkable.  The
    # upstream fast activation instructions and both AB12 stores remain intact.
    required_unchanged = (
        "cute.math.exp2(-1 * acc_vec1 * LOG2_E, True)",
        "cute.arch.rcp_approx(res[i])",
        "tRS_rAB12.store(acc_vec0)",
        "tRS_rAB12_1.store(acc_vec1)",
        "acc_vec_c = (acc_vec0 * gate).to(self.c_dtype)",
    )
    missing = [token for token in required_unchanged if token not in source]
    if missing:
        raise CudnnOssRoundtripError(
            "Variant A derivation lost required upstream operations: "
            + ", ".join(missing)
        )
    return source.encode("utf-8")


def inspect_derivative(
    provider: upstream_candidate.ProviderEvidence | None = None,
) -> tuple[bytes, DerivativeEvidence]:
    """Verify the installed source and return the deterministic derivative."""

    if provider is None:
        provider = upstream_candidate.inspect_installed_provider()
    upstream_path = _provider_kernel_path(provider)
    if not upstream_path.is_file():
        raise CudnnOssRoundtripError(
            f"upstream kernel is missing: {upstream_path}"
        )
    upstream_source = upstream_path.read_bytes()
    derivative = derive_variant_a_source(upstream_source)
    evidence = DerivativeEvidence(
        upstream_project=UPSTREAM_PROJECT,
        upstream_distribution=UPSTREAM_DISTRIBUTION,
        upstream_version=UPSTREAM_VERSION,
        upstream_relative_path=UPSTREAM_KERNEL_RELATIVE_PATH,
        upstream_installed_path=str(upstream_path.resolve()),
        upstream_sha256=_sha256_bytes(upstream_source),
        upstream_license=UPSTREAM_LICENSE,
        patch_spec_sha256=PATCH_SPEC_SHA256,
        derivative_sha256=_sha256_bytes(derivative),
        numeric_semantics_id=NUMERIC_SEMANTICS_ID,
    )
    return derivative, evidence


def materialize_derivative(
    output_dir: pathlib.Path,
    *,
    provider: upstream_candidate.ProviderEvidence | None = None,
) -> tuple[pathlib.Path, pathlib.Path, DerivativeEvidence]:
    """Write the verified derivative outside site-packages, never overwriting drift."""

    derivative, evidence = inspect_derivative(provider)
    resolved = output_dir.resolve()
    if resolved == pathlib.Path("/"):
        raise CudnnOssRoundtripError("unsafe derivative output directory")
    upstream_path = pathlib.Path(evidence.upstream_installed_path).resolve()
    site_packages_root = upstream_path
    for _ in pathlib.PurePosixPath(UPSTREAM_KERNEL_RELATIVE_PATH).parts:
        site_packages_root = site_packages_root.parent
    if resolved == site_packages_root or site_packages_root in resolved.parents:
        raise CudnnOssRoundtripError(
            "derivative output must remain outside the resolved site-packages root"
        )
    resolved.mkdir(parents=True, exist_ok=True)
    source_path = resolved / DERIVATIVE_FILENAME
    provenance_path = resolved / DERIVATIVE_PROVENANCE_FILENAME
    expected_provenance = (
        json.dumps(evidence.to_dict(), indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    for path, expected in (
        (source_path, derivative),
        (provenance_path, expected_provenance),
    ):
        if path.exists() and path.read_bytes() != expected:
            raise CudnnOssRoundtripError(
                f"refusing to overwrite mismatched derivative artifact: {path}"
            )
        if not path.exists():
            path.write_bytes(expected)
    if _sha256(pathlib.Path(evidence.upstream_installed_path)) != evidence.upstream_sha256:
        raise CudnnOssRoundtripError("site-packages source changed during derivation")
    return source_path, provenance_path, evidence


def load_derivative_kernel_class(
    output_dir: pathlib.Path,
) -> tuple[type[Any], DerivativeEvidence, pathlib.Path, pathlib.Path]:
    """Materialize and import the GPU derivative only after explicit GPU use."""

    source_path, provenance_path, evidence = materialize_derivative(output_dir)
    module_name = "katago_sm103_cudnn_oss_swiglu_variant_a"
    spec = importlib.util.spec_from_file_location(module_name, source_path)
    if spec is None or spec.loader is None:
        raise CudnnOssRoundtripError("could not construct derivative module spec")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    kernel_class = getattr(module, "PersistentDenseGemmKernel", None)
    if not isinstance(kernel_class, type):
        raise CudnnOssRoundtripError("derivative kernel class is missing")
    return kernel_class, evidence, source_path, provenance_path


def numeric_semantics() -> dict[str, Any]:
    return {
        "id": NUMERIC_SEMANTICS_ID,
        "accumulation": "tcgen05 FP32 (unchanged)",
        "projection_boundary": (
            "gate and linear1: FP32 -> FP16 round-to-nearest-even -> FP32"
        ),
        "boundary_position": "after alpha and before SwiGLU",
        "activation": "unchanged upstream fast exp2(..., True) + rcp_approx",
        "ab12": "the same rounded FP16 projections are stored; AB12 is retained",
        "output": "FP16",
        "changed_factors": ["projection FP16 round-trip"],
    }


def build_candidate_manifest(
    provider: upstream_candidate.ProviderEvidence | None = None,
    baseline_path: pathlib.Path | None = None,
    repo_root: pathlib.Path | None = None,
) -> dict[str, Any]:
    """Build the distinct CPU-only Variant A contract."""

    if provider is None:
        provider = upstream_candidate.inspect_installed_provider()
    base = upstream_candidate.validate_candidate_manifest(
        upstream_candidate.build_candidate_manifest(
            provider=provider,
            baseline_path=baseline_path,
            repo_root=repo_root,
        )
    )
    _, derivative = inspect_derivative(provider)
    base["kind"] = "katago-sm103-b29-isolated-kernel-candidate-variant-a"
    base["candidate_id"] = CANDIDATE_ID
    base["operation"]["numeric_semantics"] = numeric_semantics()
    base["static_support"]["derivative"] = derivative.to_dict()
    base["static_support"]["status"] = "cpu_contract_and_derivative_verified"
    base["correctness"]["required_reference"] = (
        "(SiLU((linear1 FP32 GEMM).half().float()) * "
        "((linearGate FP32 GEMM).half().float())).half().float()"
    )
    base["benchmark"]["artifact_runtime"] = "native C ABI only"
    return base


def validate_candidate_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    if manifest.get("kind") != (
        "katago-sm103-b29-isolated-kernel-candidate-variant-a"
    ):
        raise CudnnOssRoundtripError("unexpected Variant A manifest kind")
    if manifest.get("candidate_id") != CANDIDATE_ID:
        raise CudnnOssRoundtripError("Variant A candidate identity changed")
    provider = upstream_candidate.inspect_installed_provider()
    fresh_base = upstream_candidate.validate_candidate_manifest(
        upstream_candidate.build_candidate_manifest(provider=provider)
    )
    for field in (
        "schema",
        "provider",
        "anchor",
        "fixed_baseline_control",
        "target",
        "problem",
        "cpp_bridge",
    ):
        if manifest.get(field) != fresh_base.get(field):
            raise CudnnOssRoundtripError(
                f"Variant A base field changed: {field}"
            )

    operation = manifest.get("operation")
    if not isinstance(operation, dict) or operation.get("numeric_semantics") != (
        numeric_semantics()
    ):
        raise CudnnOssRoundtripError("Variant A numeric semantics changed")
    base_operation = dict(operation)
    base_operation.pop("numeric_semantics", None)
    if base_operation != fresh_base["operation"]:
        raise CudnnOssRoundtripError("Variant A base operation changed")
    support = manifest.get("static_support")
    if not isinstance(support, dict) or support.get("status") != (
        "cpu_contract_and_derivative_verified"
    ):
        raise CudnnOssRoundtripError("Variant A derivative is not verified")
    base_support = dict(support)
    base_support.pop("derivative", None)
    base_support["status"] = fresh_base["static_support"]["status"]
    if base_support != fresh_base["static_support"]:
        raise CudnnOssRoundtripError("Variant A base support contract changed")
    base_correctness = dict(manifest.get("correctness", {}))
    base_correctness["required_reference"] = fresh_base["correctness"][
        "required_reference"
    ]
    if base_correctness != fresh_base["correctness"]:
        raise CudnnOssRoundtripError("Variant A base correctness contract changed")
    base_benchmark = dict(manifest.get("benchmark", {}))
    base_benchmark.pop("artifact_runtime", None)
    if base_benchmark != fresh_base["benchmark"]:
        raise CudnnOssRoundtripError("Variant A base benchmark contract changed")
    derivative = support.get("derivative")
    _, fresh_derivative = inspect_derivative(provider)
    if (
        not isinstance(derivative, dict)
        or derivative != fresh_derivative.to_dict()
    ):
        raise CudnnOssRoundtripError("Variant A provenance changed")
    if manifest.get("production_ready") is not False:
        raise CudnnOssRoundtripError("isolated Variant A cannot be production-ready")
    # Every base field and the complete derivative evidence are identity, not
    # advisory metadata.  The targeted checks above retain useful diagnostics;
    # this final equality also rejects unknown fields and any future base field
    # that was not yet added to the explicit checks.
    expected = build_candidate_manifest(provider=provider)
    if manifest != expected:
        raise CudnnOssRoundtripError("full Variant A manifest identity changed")
    return manifest


def projection_fp16_roundtrip_reference(
    torch: ModuleType,
    input_2d: Any,
    linear1: Any,
    linear_gate: Any,
) -> Any:
    """Reference with the two explicit legacy FP16 projection boundaries."""

    linear1_projection = (
        input_2d.float() @ linear1.float().transpose(0, 1)
    ).half().float()
    gate_projection = (
        input_2d.float() @ linear_gate.float().transpose(0, 1)
    ).half().float()
    return (
        torch.nn.functional.silu(linear1_projection) * gate_projection
    ).half().float()


def strengthen_probe_signal(torch: ModuleType, tensors: dict[str, Any]) -> dict[str, float]:
    """Scale A and both packed/unpacked weights coherently for a useful signal."""

    with torch.no_grad():
        tensors["input_2d"].mul_(PROBE_INPUT_SCALE)
        tensors["a_tensor"][:, :, 0].copy_(tensors["input_2d"])
        tensors["linear1"].mul_(PROBE_WEIGHT_SCALE)
        tensors["linear_gate"].mul_(PROBE_WEIGHT_SCALE)
        tensors["b_tensor"].mul_(PROBE_WEIGHT_SCALE)
    return {
        "input_scale_from_base_fixture": PROBE_INPUT_SCALE,
        "weight_scale_from_base_fixture": PROBE_WEIGHT_SCALE,
        "projection_scale_from_base_fixture": (
            PROBE_INPUT_SCALE * PROBE_WEIGHT_SCALE
        ),
    }


def _tensor_error_metrics(torch: ModuleType, actual: Any, reference: Any) -> dict[str, float]:
    difference = actual.float() - reference.float()
    absolute = difference.abs()
    return {
        "max_abs_error": float(absolute.max().item()),
        "rmse": float(difference.square().mean().sqrt().item()),
        "max_rel_error": float(
            (absolute / reference.float().abs().clamp_min(1.0e-3)).max().item()
        ),
    }


def validate_correctness_summary(summary: dict[str, Any]) -> dict[str, Any]:
    """Apply non-vacuous signal and tight error limits without needing torch."""

    signal = summary.get("reference_signal")
    output = summary.get("output")
    ab12 = summary.get("ab12_vs_fp32_gemm_half")
    if not all(isinstance(item, dict) for item in (signal, output, ab12)):
        raise CudnnOssRoundtripError("correctness summary fields are missing")
    checks = {
        "reference_max_abs_signal": (
            signal.get("max_abs", 0.0)
            >= CORRECTNESS_LIMITS["reference_max_abs_minimum"]
        ),
        "reference_rms_signal": (
            signal.get("rms", 0.0)
            >= CORRECTNESS_LIMITS["reference_rms_minimum"]
        ),
        "output_max_abs": (
            output.get("max_abs_error", math.inf)
            <= CORRECTNESS_LIMITS["output_max_abs_maximum"]
        ),
        "output_rmse": (
            output.get("rmse", math.inf)
            <= CORRECTNESS_LIMITS["output_rmse_maximum"]
        ),
        "ab12_max_abs": (
            ab12.get("max_abs_error", math.inf)
            <= CORRECTNESS_LIMITS["ab12_max_abs_maximum"]
        ),
        "ab12_rmse": (
            ab12.get("rmse", math.inf)
            <= CORRECTNESS_LIMITS["ab12_rmse_maximum"]
        ),
    }
    numeric_values = (
        signal.get("max_abs"),
        signal.get("rms"),
        output.get("max_abs_error"),
        output.get("rmse"),
        ab12.get("max_abs_error"),
        ab12.get("rmse"),
    )
    if not all(
        isinstance(value, (int, float)) and math.isfinite(float(value))
        for value in numeric_values
    ):
        raise CudnnOssRoundtripError("correctness summary contains non-finite values")
    summary["limits"] = dict(CORRECTNESS_LIMITS)
    summary["checks"] = checks
    summary["passed"] = all(checks.values())
    if not summary["passed"]:
        failed = [name for name, passed in checks.items() if not passed]
        raise CudnnOssRoundtripError(
            "Variant A tight correctness gate failed: "
            + ", ".join(failed)
            + "; metrics="
            + json.dumps(
                {
                    "reference_signal": signal,
                    "output": output,
                    "ab12_vs_fp32_gemm_half": ab12,
                },
                sort_keys=True,
            )
        )
    return summary


def build_gpu_correctness_summary(
    torch: ModuleType,
    *,
    actual_output: Any,
    reference_output: Any,
    actual_ab12: Any,
    reference_ab12: Any,
) -> dict[str, Any]:
    reference_float = reference_output.float()
    summary = {
        "reference_signal": {
            "max_abs": float(reference_float.abs().max().item()),
            "rms": float(reference_float.square().mean().sqrt().item()),
        },
        "output": _tensor_error_metrics(torch, actual_output, reference_output),
        "ab12_vs_fp32_gemm_half": _tensor_error_metrics(
            torch, actual_ab12, reference_ab12
        ),
    }
    return validate_correctness_summary(summary)


def _load_native_library(library_path: pathlib.Path) -> Any:
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
    return library


def authenticate_aot_library(library_path: pathlib.Path) -> dict[str, Any]:
    """Fail closed unless a sibling AOT manifest authenticates this Variant A DSO."""

    resolved = library_path.resolve()
    if not resolved.is_file():
        raise CudnnOssRoundtripError(f"AOT library is missing: {resolved}")
    manifest_path = resolved.parent / "aot-manifest.json"
    if not manifest_path.is_file():
        raise CudnnOssRoundtripError(
            f"authenticated AOT manifest is missing: {manifest_path}"
        )
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CudnnOssRoundtripError("AOT manifest is unreadable") from error
    _, fresh_derivative = inspect_derivative()
    fresh_candidate = validate_candidate_manifest(build_candidate_manifest())
    actual_library_sha256 = _sha256(resolved)
    artifacts = manifest.get("artifacts")
    bridge = (
        artifacts.get("bridge_shared_library")
        if isinstance(artifacts, dict)
        else None
    )
    derivative = manifest.get("derivative")
    derivative_evidence = (
        derivative.get("evidence") if isinstance(derivative, dict) else None
    )
    derivative_artifacts = (
        derivative.get("artifacts") if isinstance(derivative, dict) else None
    )

    def artifact_matches(record: Any, expected_path: pathlib.Path) -> bool:
        if not isinstance(record, dict) or not expected_path.is_file():
            return False
        try:
            recorded_path = pathlib.Path(record.get("path", "")).resolve()
        except (OSError, RuntimeError, TypeError, ValueError):
            return False
        return (
            recorded_path == expected_path.resolve()
            and record.get("sha256") == _sha256(expected_path)
            and record.get("bytes") == expected_path.stat().st_size
        )

    derivative_source_path = resolved.parent / DERIVATIVE_FILENAME
    derivative_provenance_path = (
        resolved.parent / DERIVATIVE_PROVENANCE_FILENAME
    )
    source_record = (
        derivative_artifacts.get("source")
        if isinstance(derivative_artifacts, dict)
        else None
    )
    provenance_record = (
        derivative_artifacts.get("provenance")
        if isinstance(derivative_artifacts, dict)
        else None
    )
    provenance_payload: Any = None
    if derivative_provenance_path.is_file():
        try:
            provenance_payload = json.loads(
                derivative_provenance_path.read_text(encoding="utf-8")
            )
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            provenance_payload = None
    checks = {
        "kind": manifest.get("kind")
        == "katago-sm103-b29-cudnn-oss-aot-artifact",
        "candidate_id": manifest.get("candidate_id") == CANDIDATE_ID,
        "numeric_semantics_selector": manifest.get("numeric_semantics_selector")
        == "projection-fp16-roundtrip",
        "numeric_semantics": manifest.get("numeric_semantics")
        == numeric_semantics(),
        "compile_target": manifest.get("compile_target") == "sm_103a",
        "compile_options": manifest.get("compile_options")
        == ["--gpu-arch=sm_103a"],
        "derivative_evidence": derivative_evidence
        == fresh_derivative.to_dict(),
        "derivative_source": artifact_matches(
            source_record, derivative_source_path
        )
        and _sha256(derivative_source_path) == fresh_derivative.derivative_sha256,
        "derivative_provenance": artifact_matches(
            provenance_record, derivative_provenance_path
        )
        and provenance_payload == fresh_derivative.to_dict(),
        "kernel_manifest_provider": manifest.get("kernel_manifest_provider")
        == fresh_candidate["provider"],
        "shared_library_sibling": artifact_matches(bridge, resolved),
        "shared_library_sha256": isinstance(bridge, dict)
        and bridge.get("sha256") == actual_library_sha256,
        "tight_launch_validation": (
            isinstance(manifest.get("launch_validation"), dict)
            and manifest["launch_validation"].get("status") == "passed"
            and isinstance(
                manifest["launch_validation"].get("tight_correctness"), dict
            )
            and manifest["launch_validation"]["tight_correctness"].get(
                "passed"
            )
            is True
        ),
        "no_timing_identity": "timing_record" not in manifest,
        "nonproduction": manifest.get("production_ready") is False,
    }
    if not all(checks.values()):
        failed = [name for name, passed in checks.items() if not passed]
        raise CudnnOssRoundtripError(
            "AOT library authentication failed: " + ", ".join(failed)
        )
    return {
        "manifest_path": str(manifest_path),
        "manifest_sha256": _sha256(manifest_path),
        "library_path": str(resolved),
        "library_sha256": actual_library_sha256,
        "checks": checks,
    }


def _timing_summary(samples: list[float], iterations: int, streams: int) -> dict[str, Any]:
    iteration_ms = [sample * 1000.0 / iterations for sample in samples]
    median_ms = statistics.median(iteration_ms)
    return {
        "cuda_event_seconds_samples": samples,
        "milliseconds_per_concurrent_iteration_samples": iteration_ms,
        "median_stream_call_wall_milliseconds": median_ms,
        "median_effective_milliseconds_per_call": median_ms / streams,
        "calls_per_iteration": streams,
        "calls_per_second": 1000.0 * streams / median_ms,
        "relative_spread": (max(iteration_ms) - min(iteration_ms)) / median_ms,
    }


def benchmark_aot(
    *,
    allow_gpu: bool,
    device: int,
    library_path: pathlib.Path,
    warmup: int = 100,
    iterations: int = 1000,
    repeats: int = 5,
    seed: int = 20260818,
) -> dict[str, Any]:
    """Correctness-check and time a separately exported Variant A C ABI."""

    if not allow_gpu:
        raise CudnnOssRoundtripError(
            "AOT benchmark requires the explicit --allow-gpu acknowledgement"
        )
    for name, value in (
        ("device", device),
        ("warmup", warmup),
        ("iterations", iterations),
        ("repeats", repeats),
        ("seed", seed),
    ):
        if type(value) is not int:  # noqa: E721 - reject bool
            raise CudnnOssRoundtripError(f"{name} must be an integer")
    if device < 0 or warmup < 1 or iterations < 1 or repeats < 1:
        raise CudnnOssRoundtripError("invalid device or timing count")
    if not library_path.is_file():
        raise CudnnOssRoundtripError(f"AOT library is missing: {library_path}")

    validate_candidate_manifest(build_candidate_manifest())
    authentication = authenticate_aot_library(library_path)
    import torch

    if tuple(torch.cuda.get_device_capability(device)) != (
        upstream_candidate.COMPUTE_CAPABILITY
    ):
        raise CudnnOssRoundtripError("AOT benchmark requires exact SM103")
    torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = False
    tensors = upstream_candidate._allocate_gpu_benchmark_inputs(torch, device, seed)
    signal_scaling = strengthen_probe_signal(torch, tensors)
    problem: upstream_candidate.DenseSwiGLUProblem = tensors["problem"]

    def allocate_ab12() -> Any:
        return torch.empty_strided(
            (problem.m, problem.n_packed, 1),
            (problem.n_packed, 1, problem.m * problem.n_packed),
            dtype=torch.float16,
            device=f"cuda:{device}",
        )

    def allocate_c() -> Any:
        return torch.empty_strided(
            (problem.m, problem.n_output, 1),
            (problem.n_output, 1, problem.m * problem.n_output),
            dtype=torch.float16,
            device=f"cuda:{device}",
        )

    def open_instance(path: pathlib.Path, label: str) -> dict[str, Any]:
        if not path.is_file():
            raise CudnnOssRoundtripError(f"{label} AOT library is missing: {path}")
        # Allocate every caller-owned launch buffer before module/context
        # creation so any later exception has no post-Create allocation edge.
        ab12 = [allocate_ab12(), allocate_ab12()]
        outputs = [allocate_c(), allocate_c()]
        library = _load_native_library(path)
        status = ctypes.c_int32(-999)
        context = library.katagoCudnnOssB29Create(device, ctypes.byref(status))
        if not context or status.value != 0:
            raise CudnnOssRoundtripError(
                f"{label} C ABI context creation failed with status {status.value}"
            )
        return {
            "label": label,
            "path": path,
            "library": library,
            "context": context,
            "ab12": ab12,
            "outputs": outputs,
        }

    def launch(instance: dict[str, Any], index: int) -> tuple[Any, Any]:
        stream = torch.cuda.current_stream(device)
        launch_status = instance["library"].katagoCudnnOssB29Launch(
            instance["context"],
            ctypes.c_void_p(tensors["a_tensor"].data_ptr()),
            ctypes.c_void_p(tensors["b_tensor"].data_ptr()),
            ctypes.c_void_p(instance["ab12"][index].data_ptr()),
            ctypes.c_void_p(instance["outputs"][index].data_ptr()),
            ctypes.c_float(1.0),
            ctypes.c_void_p(stream.cuda_stream),
            problem.m,
            problem.k,
            problem.n_packed,
            problem.n_output,
            1,
        )
        if launch_status != 0:
            raise CudnnOssRoundtripError(
                f"C ABI launch failed with status {launch_status}"
            )
        return instance["ab12"][index], instance["outputs"][index]

    variant_instance: dict[str, Any] | None = None
    try:
        torch.cuda.synchronize(device)
        variant_instance = open_instance(library_path, "Variant A")
        variant_instance["ab12"][0].fill_(float("nan"))
        variant_instance["outputs"][0].fill_(float("nan"))
        launch(variant_instance, 0)
        torch.cuda.synchronize(device)
        reference = projection_fp16_roundtrip_reference(
            torch,
            tensors["input_2d"],
            tensors["linear1"],
            tensors["linear_gate"],
        )
        packed_reference = (
            tensors["input_2d"].float()
            @ tensors["b_tensor"][:, :, 0].float().transpose(0, 1)
        ).half()
        variant_summary = build_gpu_correctness_summary(
            torch,
            actual_output=variant_instance["outputs"][0][:, :, 0],
            reference_output=reference,
            actual_ab12=variant_instance["ab12"][0][:, :, 0],
            reference_ab12=packed_reference,
        )
        correctness = {
            "status": "passed",
            "reference_semantics": NUMERIC_SEMANTICS_ID,
            "same_seed": seed,
            "signal_scaling": signal_scaling,
            "variant_a_vs_official_half_boundary_reference": variant_summary,
        }

        # Every timing buffer is allocated before any event is recorded.  The
        # two FP16 GEMM destinations are the official projection boundary;
        # widening, SiLU/multiply, and the one final FP16 store are timed work.
        control_buffers: list[dict[str, Any]] = []
        for _ in range(2):
            control_buffers.append(
                {
                    "linear1_half": allocate_c()[:, :, 0],
                    "gate_half": allocate_c()[:, :, 0],
                    "linear1_float": torch.empty(
                        (problem.m, problem.n_output),
                        dtype=torch.float32,
                        device=f"cuda:{device}",
                    ),
                    "gate_float": torch.empty(
                        (problem.m, problem.n_output),
                        dtype=torch.float32,
                        device=f"cuda:{device}",
                    ),
                    "sigmoid": torch.empty(
                        (problem.m, problem.n_output),
                        dtype=torch.float32,
                        device=f"cuda:{device}",
                    ),
                    "activation": torch.empty(
                        (problem.m, problem.n_output),
                        dtype=torch.float32,
                        device=f"cuda:{device}",
                    ),
                    "output_float": torch.empty(
                        (problem.m, problem.n_output),
                        dtype=torch.float32,
                        device=f"cuda:{device}",
                    ),
                    "output_half": allocate_c()[:, :, 0],
                }
            )

        def torch_control(index: int) -> Any:
            buffers = control_buffers[index]
            torch.mm(
                tensors["input_2d"],
                tensors["linear1"].transpose(0, 1),
                out=buffers["linear1_half"],
            )
            torch.mm(
                tensors["input_2d"],
                tensors["linear_gate"].transpose(0, 1),
                out=buffers["gate_half"],
            )
            buffers["linear1_float"].copy_(buffers["linear1_half"])
            buffers["gate_float"].copy_(buffers["gate_half"])
            torch.sigmoid(buffers["linear1_float"], out=buffers["sigmoid"])
            torch.mul(
                buffers["linear1_float"],
                buffers["sigmoid"],
                out=buffers["activation"],
            )
            torch.mul(
                buffers["activation"],
                buffers["gate_float"],
                out=buffers["output_float"],
            )
            buffers["output_half"].copy_(buffers["output_float"])
            return buffers["output_half"]

        def measure(operation: Any, stream_count: int) -> dict[str, Any]:
            streams = [torch.cuda.Stream(device=device) for _ in range(stream_count)]
            coordinator = torch.cuda.Stream(device=device)
            live: list[Any] = [None] * stream_count
            for _ in range(warmup):
                for index, stream in enumerate(streams):
                    with torch.cuda.stream(stream):
                        live[index] = operation(index)
            torch.cuda.synchronize(device)
            samples: list[float] = []
            for _ in range(repeats):
                torch.cuda.synchronize(device)
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                if stream_count == 1:
                    with torch.cuda.stream(streams[0]):
                        start.record()
                        for _ in range(iterations):
                            live[0] = operation(0)
                        end.record()
                else:
                    done = [torch.cuda.Event() for _ in range(stream_count)]
                    with torch.cuda.stream(coordinator):
                        start.record()
                    for stream in streams:
                        stream.wait_event(start)
                    for _ in range(iterations):
                        for index, stream in enumerate(streams):
                            with torch.cuda.stream(stream):
                                live[index] = operation(index)
                    for event, stream in zip(done, streams, strict=True):
                        event.record(stream)
                    with torch.cuda.stream(coordinator):
                        for event in done:
                            coordinator.wait_event(event)
                        end.record()
                end.synchronize()
                samples.append(start.elapsed_time(end) / 1000.0)
            return _timing_summary(samples, iterations, stream_count)

        timings: dict[str, Any] = {}
        for stream_count in (1, 2):
            mode = f"s{stream_count}"
            timings[mode] = {
                "variant_a_native_c_abi": measure(
                    lambda index: launch(variant_instance, index), stream_count
                ),
                "torch_semantic_order_control_not_performance_floor": measure(
                    torch_control, stream_count
                ),
            }
            candidate_ms = timings[mode]["variant_a_native_c_abi"][
                "median_stream_call_wall_milliseconds"
            ]
            control_ms = timings[mode][
                "torch_semantic_order_control_not_performance_floor"
            ]["median_stream_call_wall_milliseconds"]
            timings[mode]["variant_a_native_c_abi"]["speedup_vs_torch_control"] = (
                control_ms / candidate_ms
            )
    finally:
        if variant_instance is not None:
            try:
                torch.cuda.synchronize(device)
            finally:
                variant_instance["library"].katagoCudnnOssB29Destroy(
                    variant_instance["context"]
                )

    return {
        "schema": 1,
        "kind": "katago-sm103-b29-cudnn-oss-variant-a-aot-timing",
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z"),
        "candidate_id": CANDIDATE_ID,
        "numeric_semantics": numeric_semantics(),
        "device": {
            "ordinal": device,
            "name": torch.cuda.get_device_name(device),
            "compute_capability": list(torch.cuda.get_device_capability(device)),
        },
        "library": {
            "path": str(library_path.resolve()),
            "sha256": _sha256(library_path),
            "bytes": library_path.stat().st_size,
        },
        "authentication": authentication,
        "method": {
            "clock": "CUDA coordinator-stream events spanning all worker streams",
            "warmup_iterations": warmup,
            "timed_iterations": iterations,
            "repeats": repeats,
            "allocation": "all native and torch-control buffers preallocated before timing",
            "torch_control_numeric_order": (
                "FP16-output GEMM projections; widen FP32; FP32 sigmoid/multiply; "
                "one final FP16 store"
            ),
            "torch_control_role": (
                "semantic/order control only, not a performance floor; explicit "
                "widen/sigmoid/multiply/cast launches replace the official custom SwiGLU"
            ),
            "s2": "two independent CUDA streams and per-stream AB12/C buffers",
        },
        "correctness": correctness,
        "timings": timings,
        "production_ready": False,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=pathlib.Path)
    parser.add_argument("--materialize-dir", type=pathlib.Path)
    parser.add_argument("--aot-benchmark", action="store_true")
    parser.add_argument("--allow-gpu", action="store_true")
    parser.add_argument("--library", type=pathlib.Path)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--repeats", type=int, default=5)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.aot_benchmark:
        if args.library is None:
            raise CudnnOssRoundtripError("--aot-benchmark requires --library")
        if args.materialize_dir is not None:
            raise CudnnOssRoundtripError(
                "--materialize-dir is not valid with --aot-benchmark"
            )
        payload = benchmark_aot(
            allow_gpu=args.allow_gpu,
            device=args.device,
            library_path=args.library,
            warmup=args.warmup,
            iterations=args.iterations,
            repeats=args.repeats,
        )
    else:
        if args.allow_gpu or args.library is not None:
            raise CudnnOssRoundtripError(
                "--allow-gpu/--library require --aot-benchmark"
            )
        payload = validate_candidate_manifest(build_candidate_manifest())
        if args.materialize_dir is not None:
            source_path, provenance_path, evidence = materialize_derivative(
                args.materialize_dir
            )
            payload["materialized_derivative"] = {
                **evidence.to_dict(),
                "source_path": str(source_path),
                "provenance_path": str(provenance_path),
            }
    serialized = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.write_text(serialized, encoding="utf-8")
    sys.stdout.write(serialized)
    return 0


__all__ = (
    "CANDIDATE_ID",
    "CudnnOssRoundtripError",
    "NUMERIC_SEMANTICS_ID",
    "PATCH_SPEC_SHA256",
    "UPSTREAM_KERNEL_SHA256",
    "benchmark_aot",
    "build_candidate_manifest",
    "derive_variant_a_source",
    "inspect_derivative",
    "load_derivative_kernel_class",
    "materialize_derivative",
    "numeric_semantics",
    "projection_fp16_roundtrip_reference",
    "validate_candidate_manifest",
)


if __name__ == "__main__":
    raise SystemExit(main())
