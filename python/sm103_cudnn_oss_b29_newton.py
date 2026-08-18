#!/usr/bin/env python3
"""B29 cuDNN OSS SwiGLU Variant C: one reciprocal Newton refinement.

Variant C is a fail-closed derivative of the exact audited Variant-A source.
It keeps the SM103 tcgen05 GEMM, tiling, grid/cluster schedule, FP16 projection
round-trip, AB12/C stores, fast ``exp2``, and native C ABI unchanged.  The only
new operation is one FP32 Newton-Raphson refinement of the upstream reciprocal
approximation before the gate multiply::

    r0 = rcp_approx(den)
    r1 = r0 * (2 - den * r0)

Import, source inspection, and manifest construction are device-free.  GPU
compilation is owned by ``sm103_cudnn_oss_b29_export.py``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import importlib.util
import json
import pathlib
import sys
from typing import Any

try:
    import sm103_cudnn_oss_b29 as upstream_candidate
    import sm103_cudnn_oss_b29_roundtrip as variant_a
except ModuleNotFoundError:
    from python import sm103_cudnn_oss_b29 as upstream_candidate
    from python import sm103_cudnn_oss_b29_roundtrip as variant_a


CANDIDATE_ID = (
    "cudnn-fe-1_27-oss-dense-gemm-swiglu-proj-fp16-roundtrip-"
    "newton1-b29"
)
NUMERIC_SEMANTICS_SELECTOR = "projection-fp16-roundtrip-newton1"
NUMERIC_SEMANTICS_ID = (
    "projection-fp16-rne-roundtrip-fast-exp2-rcp-approx-newton1-swiglu"
)
EXPECTED_VARIANT_A_DERIVATIVE_SHA256 = (
    "99247c64d70a5f0b14ff75c08ba8d28fde31f159248e1e86c934cec6152777bc"
)
DERIVATIVE_FILENAME = "dense_gemm_persistent_swiglu_variant_c.py"
DERIVATIVE_PROVENANCE_FILENAME = "variant-c-provenance.json"


VARIANT_A_FAST_MATH_BLOCK = """\
                    # Use exp2 with log2(e) conversion since cute.math.exp is not available
                    # exp(x) = 2^(x * log2(e))
                    gate_rcp = (1 + cute.math.exp2(-1 * acc_vec1 * LOG2_E, True)).to(self.acc_dtype)

                    res = cute.make_rmem_tensor(gate_rcp.shape, cutlass.Float32)
                    res.store(gate_rcp)
                    for i in cutlass.range_constexpr(cute.size(res.shape)):
                        res[i] = cute.arch.rcp_approx(res[i])

                    gate = res.load()
                    gate = gate * acc_vec1
"""


VARIANT_C_NEWTON_MATH_BLOCK = """\
                    # Use exp2 with log2(e) conversion since cute.math.exp is not available
                    # exp(x) = 2^(x * log2(e))
                    gate_rcp = (1 + cute.math.exp2(-1 * acc_vec1 * LOG2_E, True)).to(self.acc_dtype)

                    res = cute.make_rmem_tensor(gate_rcp.shape, cutlass.Float32)
                    res.store(gate_rcp)
                    for i in cutlass.range_constexpr(cute.size(res.shape)):
                        # KataGo B29 Variant C: preserve the fast reciprocal seed,
                        # then apply exactly one FP32 Newton refinement.
                        denominator = res[i]
                        reciprocal = cute.arch.rcp_approx(denominator)
                        res[i] = reciprocal * (2.0 - denominator * reciprocal)

                    gate = res.load()
                    gate = gate * acc_vec1
"""


class CudnnOssNewtonError(RuntimeError):
    """Raised when Variant C provenance or derivation is invalid."""


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


PATCH_SPEC_SHA256 = _sha256_bytes(
    (VARIANT_A_FAST_MATH_BLOCK + "\0" + VARIANT_C_NEWTON_MATH_BLOCK).encode(
        "utf-8"
    )
)


@dataclass(frozen=True)
class DerivativeEvidence:
    parent_candidate_id: str
    parent_numeric_semantics_id: str
    parent_derivative_sha256: str
    parent_patch_spec_sha256: str
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


def derive_variant_c_source(variant_a_source: bytes) -> bytes:
    """Replace exactly Variant A's fast-math block with Newton1 math."""

    actual = _sha256_bytes(variant_a_source)
    if actual != EXPECTED_VARIANT_A_DERIVATIVE_SHA256:
        raise CudnnOssNewtonError(
            "Variant-A derivative SHA-256 mismatch: expected "
            f"{EXPECTED_VARIANT_A_DERIVATIVE_SHA256}, got {actual}"
        )
    try:
        source = variant_a_source.decode("utf-8")
    except UnicodeDecodeError as error:
        raise CudnnOssNewtonError("Variant-A derivative is not UTF-8") from error
    count = source.count(VARIANT_A_FAST_MATH_BLOCK)
    if count != 1:
        raise CudnnOssNewtonError(
            f"Variant-C Newton context must occur exactly once, got {count}"
        )
    source = source.replace(
        VARIANT_A_FAST_MATH_BLOCK, VARIANT_C_NEWTON_MATH_BLOCK, 1
    )

    required_once = (
        "acc_vec0_ab12 = acc_vec0.to(self.ab12_dtype)",
        "acc_vec1_ab12 = acc_vec1.to(self.ab12_dtype)",
        "cute.math.exp2(-1 * acc_vec1 * LOG2_E, True)",
        "cute.arch.rcp_approx(denominator)",
        "reciprocal * (2.0 - denominator * reciprocal)",
        "acc_vec_c = (acc_vec0 * gate).to(self.c_dtype)",
        "tRS_rAB12.store(acc_vec0)",
        "tRS_rAB12_1.store(acc_vec1)",
        "tRS_rC.store(acc_vec_c)",
    )
    invalid_counts = {
        token: source.count(token)
        for token in required_once
        if source.count(token) != 1
    }
    if invalid_counts:
        raise CudnnOssNewtonError(
            "Variant-C cumulative operation counts changed: "
            + ", ".join(
                f"{token!r}={count}" for token, count in invalid_counts.items()
            )
        )
    forbidden = (
        "cute.math.exp(-acc_vec1",
        "fastmath=False",
        "acc_vec1 / gate_denominator",
        "arith.divf",
    )
    present = [token for token in forbidden if token in source]
    if present:
        raise CudnnOssNewtonError(
            "Variant-C derivation retained a precise-math slow path: "
            + ", ".join(present)
        )
    return source.encode("utf-8")


def inspect_derivative(
    provider: upstream_candidate.ProviderEvidence | None = None,
) -> tuple[bytes, DerivativeEvidence]:
    parent_source, parent = variant_a.inspect_derivative(provider)
    if parent.derivative_sha256 != EXPECTED_VARIANT_A_DERIVATIVE_SHA256:
        raise CudnnOssNewtonError("audited Variant-A evidence changed")
    derivative = derive_variant_c_source(parent_source)
    evidence = DerivativeEvidence(
        parent_candidate_id=variant_a.CANDIDATE_ID,
        parent_numeric_semantics_id=variant_a.NUMERIC_SEMANTICS_ID,
        parent_derivative_sha256=parent.derivative_sha256,
        parent_patch_spec_sha256=parent.patch_spec_sha256,
        upstream_project=parent.upstream_project,
        upstream_distribution=parent.upstream_distribution,
        upstream_version=parent.upstream_version,
        upstream_relative_path=parent.upstream_relative_path,
        upstream_installed_path=parent.upstream_installed_path,
        upstream_sha256=parent.upstream_sha256,
        upstream_license=parent.upstream_license,
        patch_spec_sha256=PATCH_SPEC_SHA256,
        derivative_sha256=_sha256_bytes(derivative),
        numeric_semantics_id=NUMERIC_SEMANTICS_ID,
    )
    return derivative, evidence


def _site_packages_root(evidence: DerivativeEvidence) -> pathlib.Path:
    root = pathlib.Path(evidence.upstream_installed_path).resolve()
    for _ in pathlib.PurePosixPath(variant_a.UPSTREAM_KERNEL_RELATIVE_PATH).parts:
        root = root.parent
    return root


def materialize_derivative(
    output_dir: pathlib.Path,
    *,
    provider: upstream_candidate.ProviderEvidence | None = None,
) -> tuple[pathlib.Path, pathlib.Path, DerivativeEvidence]:
    derivative, evidence = inspect_derivative(provider)
    resolved = output_dir.resolve()
    if resolved == pathlib.Path("/"):
        raise CudnnOssNewtonError("unsafe derivative output directory")
    site_packages = _site_packages_root(evidence)
    if resolved == site_packages or site_packages in resolved.parents:
        raise CudnnOssNewtonError(
            "Variant-C derivative output must remain outside site-packages"
        )
    resolved.mkdir(parents=True, exist_ok=True)
    source_path = resolved / DERIVATIVE_FILENAME
    provenance_path = resolved / DERIVATIVE_PROVENANCE_FILENAME
    provenance = (
        json.dumps(evidence.to_dict(), indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    for path, expected in (
        (source_path, derivative),
        (provenance_path, provenance),
    ):
        if path.exists() and path.read_bytes() != expected:
            raise CudnnOssNewtonError(
                f"refusing to overwrite mismatched Variant-C artifact: {path}"
            )
        if not path.exists():
            path.write_bytes(expected)
    if _sha256(pathlib.Path(evidence.upstream_installed_path)) != (
        evidence.upstream_sha256
    ):
        raise CudnnOssNewtonError("site-packages source changed during derivation")
    return source_path, provenance_path, evidence


def load_derivative_kernel_class(
    output_dir: pathlib.Path,
) -> tuple[type[Any], DerivativeEvidence, pathlib.Path, pathlib.Path]:
    source_path, provenance_path, evidence = materialize_derivative(output_dir)
    module_name = "katago_sm103_cudnn_oss_swiglu_variant_c"
    spec = importlib.util.spec_from_file_location(module_name, source_path)
    if spec is None or spec.loader is None:
        raise CudnnOssNewtonError("could not construct Variant-C module spec")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    kernel_class = getattr(module, "PersistentDenseGemmKernel", None)
    if not isinstance(kernel_class, type):
        raise CudnnOssNewtonError("Variant-C derivative kernel class is missing")
    return kernel_class, evidence, source_path, provenance_path


def numeric_semantics() -> dict[str, Any]:
    return {
        "id": NUMERIC_SEMANTICS_ID,
        "accumulation": "tcgen05 FP32 (unchanged)",
        "projection_boundary": (
            "gate and linear1: FP32 -> FP16 round-to-nearest-even -> FP32 "
            "(unchanged from Variant A)"
        ),
        "boundary_position": "after alpha and before SwiGLU (unchanged)",
        "activation": (
            "fast exp2 + one rcp_approx seed + exactly one FP32 Newton "
            "reciprocal refinement"
        ),
        "ab12": "same rounded FP16 projections; AB12 retained unchanged",
        "output": "FP16",
        "parent_candidate_id": variant_a.CANDIDATE_ID,
        "changed_factors_from_variant_a": [
            "one FP32 Newton refinement after the unchanged rcp_approx seed"
        ],
    }


def build_candidate_manifest(
    provider: upstream_candidate.ProviderEvidence | None = None,
    baseline_path: pathlib.Path | None = None,
    repo_root: pathlib.Path | None = None,
) -> dict[str, Any]:
    if provider is None:
        provider = upstream_candidate.inspect_installed_provider()
    parent = variant_a.validate_candidate_manifest(
        variant_a.build_candidate_manifest(
            provider=provider,
            baseline_path=baseline_path,
            repo_root=repo_root,
        )
    )
    _, derivative = inspect_derivative(provider)
    parent["kind"] = "katago-sm103-b29-isolated-kernel-candidate-variant-c"
    parent["candidate_id"] = CANDIDATE_ID
    parent["operation"]["numeric_semantics"] = numeric_semantics()
    parent["static_support"]["derivative"] = derivative.to_dict()
    parent["static_support"]["status"] = (
        "cpu_contract_and_cumulative_derivative_verified"
    )
    parent["correctness"]["required_reference"] = (
        "precise FP32 SiLU of FP16-rounded linear1 projection multiplied by "
        "the FP16-rounded gate projection, then FP16 output"
    )
    parent["benchmark"]["artifact_runtime"] = "native C ABI only"
    return parent


def validate_candidate_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    if manifest.get("kind") != (
        "katago-sm103-b29-isolated-kernel-candidate-variant-c"
    ):
        raise CudnnOssNewtonError("unexpected Variant-C manifest kind")
    if manifest.get("candidate_id") != CANDIDATE_ID:
        raise CudnnOssNewtonError("Variant-C candidate identity changed")
    provider = upstream_candidate.inspect_installed_provider()
    expected = build_candidate_manifest(provider=provider)
    if manifest != expected:
        raise CudnnOssNewtonError("full Variant-C manifest identity changed")
    if manifest.get("production_ready") is not False:
        raise CudnnOssNewtonError("isolated Variant C cannot be production-ready")
    return manifest


# Variant C intentionally shares Variant A's official half-boundary reference,
# signal-strengthening fixture, and tight output/AB12 correctness gate.
projection_fp16_roundtrip_reference = variant_a.projection_fp16_roundtrip_reference
strengthen_probe_signal = variant_a.strengthen_probe_signal
build_gpu_correctness_summary = variant_a.build_gpu_correctness_summary
validate_correctness_summary = variant_a.validate_correctness_summary
CORRECTNESS_LIMITS = variant_a.CORRECTNESS_LIMITS


__all__ = (
    "CANDIDATE_ID",
    "CORRECTNESS_LIMITS",
    "CudnnOssNewtonError",
    "DERIVATIVE_FILENAME",
    "DERIVATIVE_PROVENANCE_FILENAME",
    "EXPECTED_VARIANT_A_DERIVATIVE_SHA256",
    "NUMERIC_SEMANTICS_ID",
    "NUMERIC_SEMANTICS_SELECTOR",
    "PATCH_SPEC_SHA256",
    "VARIANT_A_FAST_MATH_BLOCK",
    "VARIANT_C_NEWTON_MATH_BLOCK",
    "build_candidate_manifest",
    "build_gpu_correctness_summary",
    "derive_variant_c_source",
    "inspect_derivative",
    "load_derivative_kernel_class",
    "materialize_derivative",
    "numeric_semantics",
    "projection_fp16_roundtrip_reference",
    "strengthen_probe_signal",
    "validate_candidate_manifest",
    "validate_correctness_summary",
)
