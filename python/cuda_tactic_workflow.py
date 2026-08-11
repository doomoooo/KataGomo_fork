#!/usr/bin/env python3
"""Unified SM89/SM120 exact-batch tactic scanning and plan generation.

This file deliberately lives outside the CUDA runtime. It is the only
optimization workflow boundary maintained by final-migration:

  space   materialize the candidates that may be scanned
  scan    run the normal whole-graph ``benchmarknn`` for every candidate
  plan    select the best *long, stable* candidate per family and batch
  validate check a plan against a receiver's space/model/config
  apply   render the per-batch config needed to bypass the search stage

The plan is an execution input, not a claim that a kernel is correct.  A
candidate can carry an optional correctness record; ``production_ready`` is
only true when every selected candidate has an explicit correctness pass.
``ready_for_scan_bypass`` is the weaker, intended handoff gate: it means that
the complete requested candidate coverage has been measured with long stable
whole-graph throughput.

No CUDA/Python package is required to use ``space``, ``plan``, ``validate`` or
``apply``.  ``scan`` only needs the repository's benchmarknn executable.
"""

from __future__ import annotations

import argparse
import datetime as _datetime
import functools
import hashlib
import importlib
import json
import math
import os
import pathlib
import platform
import re
import signal
import shlex
import shutil
import statistics
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Iterable, Sequence
from typing import Any

try:
    from cuda_tactic_history import validate_positive_history_closure
except ModuleNotFoundError:
    from python.cuda_tactic_history import validate_positive_history_closure


SCHEMA = 1
SPACE_KIND = "cuda-tactic-search-space"
PLAN_KIND = "cuda-tactic-plan"
RESULT_KIND = "cuda-tactic-scan"
ARTIFACT_BUNDLE_KIND = "cuda-tactic-artifact-bundle"
BASELINE_SCAN_KIND = "cuda-stable-optimized-batch-prescan"
SM_CONFLICT_RETRY_SECONDS = 30.0
# A family is an implementation catalog. A decision group is the actual
# ordered coordinate: every runtime dependency/overlap stays inside one group,
# and no later group may own the same config key. Bundle setup comes first;
# finer components may then explicitly refine individual keys without erasing
# an unrelated component of the bundle.
SM89_DECISION_GROUPS = (
    ("wide_projection", "qkv_rope", "dual_ffn"),
    (
        "fused_residual", "linear2", "outproj", "postconv_bn", "pointwise",
    ),
    ("rmsnorm",),
    ("fa4",),
    ("preconv",),
    ("l2",),
    ("weight_sharing",),
    ("initial_conv",),
    ("wide_head", "initial_global", "policy_p1", "head_bn"),
    ("value_terminal",),
)
SM120_DECISION_GROUPS = (
    ("fa4", "wide_projection", "qkv_rope", "dual_ffn"),
    ("fused_residual", "linear2", "outproj"),
    ("postconv_bn", "preconv", "pointwise"),
    ("wide_head", "policy_p1", "head_bn"),
    ("rmsnorm",),
    ("l2",),
    ("weight_sharing",),
    ("initial_conv",),
    ("initial_global",),
    ("value_terminal",),
)
SM89_FAMILIES = tuple(
    family for group in SM89_DECISION_GROUPS for family in group
)
SM120_FAMILIES = tuple(
    family for group in SM120_DECISION_GROUPS for family in group
)
# The remaining shared keys are deliberate whole-boundary decisions. This is
# an exhaustive allow-list, not a compatibility mechanism: adding a second
# owner to any other runtime key is a search-space error. FA is intentionally
# absent because its tile/accumulator winner must never be rewritten by a QKV
# or RoPE bundle.
EXPECTED_CROSS_FAMILY_OWNERS = {
    "sm89": {
        "cudaPolicyP1RowsPerBlockSm89": ("wide_head", "policy_p1"),
        "cudaUseFusedResidual": ("fused_residual", "linear2", "outproj"),
        "cudaUseHeadBNHalfToFloat": ("wide_head", "head_bn"),
        "cudaUseInitialGlobalMatMulAdd": ("wide_head", "initial_global"),
        "cudaUseLinear2PostBNSiluSm89": ("linear2", "pointwise"),
        "cudaUsePostConvBNSiluSm89": ("postconv_bn", "pointwise"),
        "cudaUseWideFFN": ("wide_projection", "dual_ffn"),
        "cudaUseWideHeadProjection": ("wide_head", "policy_p1"),
        "cudaUseWideQKV": ("wide_projection", "qkv_rope"),
    },
    "sm120": {
        "cudaFusedFFNAotTacticSm120": ("wide_projection", "dual_ffn"),
        "cudaOuterProjectionDownTacticSm120": ("postconv_bn", "preconv"),
        "cudaQKVRopeAotTacticSm120": ("wide_projection", "qkv_rope"),
        "cudaUseBatchSharedRoPE": ("wide_projection", "qkv_rope"),
        "cudaUseBatchSharedRoPEUnrolledSm120": (
            "wide_projection", "qkv_rope",
        ),
        "cudaUseFusedFFN": ("wide_projection", "dual_ffn"),
        "cudaUseFusedPolicyP1": ("wide_head", "policy_p1"),
        "cudaUseFusedQKRoPE": ("wide_projection", "qkv_rope"),
        "cudaUseFusedQKRoPEHalf2Sm120": (
            "wide_projection", "qkv_rope",
        ),
        "cudaUseFusedResidualGemmSm120": (
            "fused_residual", "linear2", "outproj",
        ),
        "cudaUseHeadBNHalfToFloat": ("wide_head", "head_bn"),
        "cudaUseQKVGemmAot": ("wide_projection", "qkv_rope"),
        "cudaUseQKVStridedSm120": (
            "wide_projection", "qkv_rope",
        ),
        "cudaUseWideFFNSingleGemm": (
            "wide_projection", "dual_ffn",
        ),
        "cudaUseWideQKV": ("wide_projection", "qkv_rope"),
        "cudaWideQKVAotTacticSm120": (
            "wide_projection", "qkv_rope",
        ),
    },
}
ALL_FAMILIES = tuple(dict.fromkeys((*SM89_FAMILIES, *SM120_FAMILIES)))
SM89_RUNTIME_CONFIG_KEYS = frozenset({
    "cudaFusedFFNAotTacticSm89",
    "cudaDualFfnCutlassTacticSm89",
    "cudaFlashAttentionTacticSm89",
    "cudaLinear2AotTacticSm89",
    "cudaLinear2CutlassTacticSm89",
    "cudaOutProjCutlassTacticSm89",
    "cudaPreConvCutlassTacticSm89",
    "cudaPostConvCutlassTacticSm89",
    "cudaPersistingL2HitRatioSm89",
    "cudaPlainQKVVariantSm89",
    "cudaPolicyP1RowsPerBlockSm89",
    "cudaRMSNormRowsPerBlockSm89",
    "cudaRoPEBatchGroupSm89",
    "cudaShareModelWeights",
    "cudaUseFusedQKRoPE",
    "cudaUseFusedResidual",
    "cudaUseFusedValueTerminalSm89",
    "cudaUseHeadBNHalfToFloat",
    "cudaUseInitialConvFrontend",
    "cudaUseInitialGlobalMatMulAdd",
    "cudaUseLinear2PostBNSiluSm89",
    "cudaUsePersistingL2Inner",
    "cudaUsePersistingL2Trunk",
    "cudaUsePostConvBNSiluSm89",
    "cudaUsePrecomputedQKRoPESm89",
    "cudaUseQKVRoPEGemmSm89",
    "cudaUseRMSNormOpt",
    "cudaUseScaleBiasSiluVec4C384Sm89",
    "cudaUseScaleBiasSiluVec8Sm89",
    "cudaUseScaleBiasSiluVec8C384Sm89",
    "cudaUseSplitQKVRoPEGemmSm89",
    "cudaUseWideFFN",
    "cudaUseWideHeadProjection",
    "cudaUseWideQKV",
})
SM120_RUNTIME_CONFIG_KEYS = frozenset({
    "cudaShareModelWeights",
    "cudaFlashAttentionAotTacticSm120",
    "cudaFlashAttentionSm120Accum",
    "cudaFusedFFNAotTacticSm120",
    "cudaLinear2AotTacticSm120",
    "cudaOutProjectionAotTacticSm120",
    "cudaPersistingL2HitRatioSm120",
    "cudaAffineSiluTacticSm120",
    "cudaUseBatchSharedRoPE",
    "cudaUseBatchSharedRoPEUnrolledSm120",
    "cudaUseFlashAttentionSm120",
    "cudaUseFusedFFN",
    "cudaUseFusedPolicyP1",
    "cudaUseFusedQKRoPE",
    "cudaUseFusedQKRoPEHalf2Sm120",
    "cudaUseFusedResidual",
    "cudaUseFusedResidualGemmSm120",
    "cudaUseHeadBNHalfToFloat",
    "cudaInitialConvFrontendPlanSm120",
    "cudaUseInitialGlobalMatMulAdd",
    "cudaUseLinear2ResidualAot",
    "cudaOuterProjectionDownTacticSm120",
    "cudaOuterProjectionUpTacticSm120",
    "cudaUsePostConvBNSiluSm120",
    "cudaUseOutProjectionResidualAot",
    "cudaUsePersistingL2Inner",
    "cudaUsePersistingL2Trunk",
    "cudaUseQKVGemmAot",
    "cudaQKVRopeAotTacticSm120",
    "cudaUseQKVStridedSm120",
    "cudaRMSNormTacticSm120",
    "cudaUseSwiGLU1152Sm120",
    "cudaUseWideFFNSingleGemm",
    "cudaWideHeadProjectionTacticSm120",
    "cudaUseWideQKV",
    "cudaWideQKVAotTacticSm120",
    "cudaUseFusedValueTerminalSm120",
})
SM89_RUNTIME_BASELINE: dict[str, object] = {
    "cudaFusedFFNAotTacticSm89": "disabled",
    "cudaDualFfnCutlassTacticSm89": "disabled",
    "cudaFlashAttentionTacticSm89": "disabled",
    "cudaLinear2AotTacticSm89": "disabled",
    "cudaLinear2CutlassTacticSm89": "disabled",
    "cudaOutProjCutlassTacticSm89": "disabled",
    "cudaPreConvCutlassTacticSm89": "disabled",
    "cudaPostConvCutlassTacticSm89": "disabled",
    "cudaPersistingL2HitRatioSm89": 1.0,
    "cudaPlainQKVVariantSm89": 0,
    "cudaPolicyP1RowsPerBlockSm89": 0,
    "cudaRMSNormRowsPerBlockSm89": 4,
    "cudaRoPEBatchGroupSm89": 1,
    "cudaShareModelWeights": False,
    "cudaUseFusedQKRoPE": False,
    "cudaUseFusedResidual": False,
    "cudaUseFusedValueTerminalSm89": False,
    "cudaUseHeadBNHalfToFloat": False,
    "cudaUseInitialConvFrontend": False,
    "cudaUseInitialGlobalMatMulAdd": False,
    "cudaUseLinear2PostBNSiluSm89": False,
    "cudaUsePersistingL2Inner": False,
    "cudaUsePersistingL2Trunk": False,
    "cudaUsePostConvBNSiluSm89": False,
    "cudaUsePrecomputedQKRoPESm89": False,
    "cudaUseQKVRoPEGemmSm89": False,
    "cudaUseRMSNormOpt": False,
    "cudaUseScaleBiasSiluVec4C384Sm89": False,
    "cudaUseScaleBiasSiluVec8Sm89": False,
    "cudaUseScaleBiasSiluVec8C384Sm89": False,
    "cudaUseSplitQKVRoPEGemmSm89": False,
    "cudaUseWideFFN": False,
    "cudaUseWideHeadProjection": False,
    "cudaUseWideQKV": False,
}
SM120_RUNTIME_BASELINE: dict[str, object] = {
    "cudaShareModelWeights": False,
    "cudaFlashAttentionAotTacticSm120": "disabled",
    "cudaFlashAttentionSm120Accum": "none",
    "cudaFusedFFNAotTacticSm120": "disabled",
    "cudaLinear2AotTacticSm120": "disabled",
    "cudaOutProjectionAotTacticSm120": "disabled",
    "cudaPersistingL2HitRatioSm120": 1.0,
    "cudaAffineSiluTacticSm120": "disabled",
    "cudaUseBatchSharedRoPE": False,
    "cudaUseBatchSharedRoPEUnrolledSm120": False,
    "cudaUseFlashAttentionSm120": False,
    "cudaUseFusedFFN": False,
    "cudaUseFusedPolicyP1": False,
    "cudaUseFusedQKRoPE": False,
    "cudaUseFusedQKRoPEHalf2Sm120": False,
    "cudaUseFusedResidual": False,
    "cudaUseFusedResidualGemmSm120": False,
    "cudaUseHeadBNHalfToFloat": False,
    "cudaInitialConvFrontendPlanSm120": "disabled",
    "cudaUseInitialGlobalMatMulAdd": False,
    "cudaUseLinear2ResidualAot": False,
    "cudaOuterProjectionDownTacticSm120": "disabled",
    "cudaOuterProjectionUpTacticSm120": "disabled",
    "cudaUsePostConvBNSiluSm120": False,
    "cudaUseOutProjectionResidualAot": False,
    "cudaUsePersistingL2Inner": False,
    "cudaUsePersistingL2Trunk": False,
    "cudaUseQKVGemmAot": False,
    "cudaQKVRopeAotTacticSm120": "disabled",
    "cudaUseQKVStridedSm120": False,
    "cudaRMSNormTacticSm120": "disabled",
    "cudaUseSwiGLU1152Sm120": False,
    "cudaUseWideFFNSingleGemm": False,
    "cudaWideHeadProjectionTacticSm120": "disabled",
    "cudaUseWideQKV": False,
    "cudaWideQKVAotTacticSm120": "disabled",
    "cudaUseFusedValueTerminalSm120": False,
}
if set(SM89_RUNTIME_BASELINE) != set(SM89_RUNTIME_CONFIG_KEYS):
    raise RuntimeError("SM89 runtime baseline does not cover its config-key contract")
if set(SM120_RUNTIME_BASELINE) != set(SM120_RUNTIME_CONFIG_KEYS):
    raise RuntimeError("SM120 runtime baseline does not cover its config-key contract")
MIN_LONG_ITERATIONS = 1000
MIN_STABLE_SAMPLES = 2
MIN_DISCOVERY_ITERATIONS = 100
MIN_DISCOVERY_WARMUP = 50
DEFAULT_MAX_RELATIVE_SPREAD = 0.10
DEFAULT_MIN_DISCOVERY_IMPROVEMENT_FRACTION = 0.001

ARCHITECTURES: dict[str, dict[str, Any]] = {
    "sm89": {
        "compute_capability": [8, 9],
        "gpu_classes": ("rtx4090", "rtx4080", "sm89"),
        "precision": "FP16/NHWC",
        "families": SM89_FAMILIES,
    },
    "sm120": {
        "compute_capability": [12, 0],
        "gpu_classes": ("rtx5080", "rtx5090d", "sm120"),
        "precision": "FP16/NHWC",
        "families": SM120_FAMILIES,
    },
}
GPU_CLASS_ARCH = {
    gpu_class: architecture
    for architecture, value in ARCHITECTURES.items()
    for gpu_class in value["gpu_classes"]
}


def utc_now() -> str:
    return _datetime.datetime.now(_datetime.timezone.utc).isoformat()


def canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def cuda_compute_capability(properties: dict[str, object]) -> list[int] | None:
    value = properties.get("compute_capability")
    if (
        isinstance(value, list) and len(value) == 2 and
        all(isinstance(item, int) for item in value)
    ):
        return list(value)
    major = properties.get("computeCapabilityMajor")
    minor = properties.get("computeCapabilityMinor")
    if isinstance(major, int) and isinstance(minor, int):
        return [major, minor]
    return None


CUDA_PLAN_DEVICE_IDENTITY_FIELDS = (
    "name",
    "computeCapabilityMajor",
    "computeCapabilityMinor",
    "multiProcessorCount",
    "totalGlobalMem",
    "maxThreadsPerBlock",
    "maxThreadsPerMultiProcessor",
    "regsPerMultiprocessor",
    "sharedMemPerBlockOptin",
    "sharedMemPerMultiprocessor",
    "l2CacheSize",
    "memoryBusWidth",
    "asyncEngineCount",
    "concurrentKernels",
)


def cuda_plan_device_identity(properties: dict[str, object]) -> dict[str, object]:
    """Normalize CUDA query and recorded-plan device fields to one contract."""
    attributes = properties.get("attributes", {})
    if not isinstance(attributes, dict):
        attributes = {}

    def value(*names: str) -> object:
        for name in names:
            if name in properties:
                return properties[name]
            if name in attributes:
                return attributes[name]
        return None

    compute_capability = cuda_compute_capability(properties)
    identity: dict[str, object] = {
        "name": value("name"),
        "computeCapabilityMajor": (
            compute_capability[0] if compute_capability is not None else None
        ),
        "computeCapabilityMinor": (
            compute_capability[1] if compute_capability is not None else None
        ),
        "multiProcessorCount": value("multiProcessorCount"),
        "totalGlobalMem": value("totalGlobalMem"),
        "maxThreadsPerBlock": value("maxThreadsPerBlock"),
        "maxThreadsPerMultiProcessor": value(
            "maxThreadsPerMultiProcessor", "maxThreadsPerMultiprocessor",
        ),
        "regsPerMultiprocessor": value("regsPerMultiprocessor"),
        "sharedMemPerBlockOptin": value(
            "sharedMemPerBlockOptin", "maxSharedMemoryPerBlockOptin",
        ),
        "sharedMemPerMultiprocessor": value(
            "sharedMemPerMultiprocessor", "sharedMemoryPerMultiprocessor",
        ),
        "l2CacheSize": value("l2CacheSize"),
        "memoryBusWidth": value("memoryBusWidth"),
        "asyncEngineCount": value("asyncEngineCount"),
        "concurrentKernels": value("concurrentKernels"),
    }
    missing = [
        field for field in CUDA_PLAN_DEVICE_IDENTITY_FIELDS
        if identity.get(field) is None
    ]
    if missing:
        raise ValueError(
            "CUDA device identity is incomplete: " + ", ".join(missing)
        )
    if not isinstance(identity["name"], str) or not identity["name"]:
        raise ValueError("CUDA device identity has an invalid GPU name")
    for field in CUDA_PLAN_DEVICE_IDENTITY_FIELDS:
        if field == "name":
            continue
        raw = identity[field]
        if field == "concurrentKernels":
            identity[field] = bool(raw)
        elif isinstance(raw, bool) or not isinstance(raw, int) or raw < 0:
            raise ValueError(f"CUDA device identity has invalid {field}: {raw!r}")
    return identity


def nvcc_arch_flag(compute_capability: object) -> str:
    if (
        not isinstance(compute_capability, list)
        or len(compute_capability) != 2
        or any(not isinstance(value, int) or value < 0 for value in compute_capability)
    ):
        raise ValueError(f"invalid CUDA compute capability: {compute_capability!r}")
    major, minor = compute_capability
    return f"-arch=sm_{major}{minor}"


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


@functools.lru_cache(maxsize=128)
def _sha256_file_version(path_text: str, size: int, mtime_ns: int) -> str:
    del size, mtime_ns  # They are cache-key version fields.
    digest = hashlib.sha256()
    path = pathlib.Path(path_text)
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_file(path: pathlib.Path) -> str:
    resolved = path.resolve()
    stat = resolved.stat()
    return _sha256_file_version(str(resolved), stat.st_size, stat.st_mtime_ns)


def workflow_implementation_identity() -> dict[str, object]:
    repo = pathlib.Path(__file__).resolve().parents[1]
    paths = (
        pathlib.Path(__file__).resolve(),
        repo / "python/portable_cuda_device.py",
        repo / "python/portable_fat_scan.py",
    )
    files = {
        str(path.relative_to(repo)): sha256_file(path)
        for path in paths if path.is_file()
    }
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo, text=True,
        capture_output=True, check=False,
    )
    return {
        "files": files,
        "git_head": revision.stdout.strip() if revision.returncode == 0 else None,
    }


def parse_int_set(value: str) -> list[int]:
    result: set[int] = set()
    for token in value.split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            first_text, last_text = token.split("-", 1)
            first, last = int(first_text), int(last_text)
            if last < first:
                raise ValueError(f"invalid descending range: {token}")
            result.update(range(first, last + 1))
        else:
            result.add(int(token))
    values = sorted(result)
    if not values or values[0] < 1:
        raise ValueError("integer sets must contain positive integers")
    return values


def parse_key_values(value: str | None) -> dict[str, str]:
    """Parse the comma-separated syntax accepted by -override-config."""
    result: dict[str, str] = {}
    if not value:
        return result
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"config override is missing '=': {item}")
        key, raw = item.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"config override has an empty key: {item}")
        result[key] = raw.strip()
    return result


def config_value(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return format(value, ".12g")
    return str(value)


def config_string(values: dict[str, object]) -> str:
    return ",".join(f"{key}={config_value(values[key])}" for key in sorted(values))


def canonical_architecture(architecture: str | None, gpu_class: str | None) -> str:
    if architecture:
        architecture = architecture.lower()
        if architecture not in ARCHITECTURES:
            raise ValueError(f"architecture must be one of {tuple(ARCHITECTURES)}")
        if gpu_class and gpu_class.lower() in GPU_CLASS_ARCH:
            expected = GPU_CLASS_ARCH[gpu_class.lower()]
            if expected != architecture:
                raise ValueError(f"GPU class {gpu_class} belongs to {expected}, not {architecture}")
        return architecture
    if gpu_class and gpu_class.lower() in GPU_CLASS_ARCH:
        return GPU_CLASS_ARCH[gpu_class.lower()]
    raise ValueError("one of --architecture or a known --gpu-class is required")


def validate_gpu_class(architecture: str, gpu_class: str) -> None:
    if gpu_class not in ARCHITECTURES[architecture]["gpu_classes"]:
        raise ValueError(f"GPU class {gpu_class} is not valid for {architecture}")


def architecture_families(architecture: str) -> tuple[str, ...]:
    if architecture not in ARCHITECTURES:
        raise ValueError(f"unknown CUDA architecture: {architecture}")
    return tuple(ARCHITECTURES[architecture]["families"])


def architecture_decision_groups(
    architecture: str,
) -> tuple[tuple[str, ...], ...]:
    if architecture == "sm89":
        groups = SM89_DECISION_GROUPS
    elif architecture == "sm120":
        groups = SM120_DECISION_GROUPS
    else:
        raise ValueError(f"unknown CUDA architecture: {architecture}")
    if tuple(family for group in groups for family in group) != architecture_families(
        architecture
    ):
        raise ValueError(f"{architecture} decision groups do not flatten to families")
    return groups


def runtime_tactic_baseline(architecture: str) -> dict[str, object]:
    if architecture == "sm89":
        return dict(SM89_RUNTIME_BASELINE)
    if architecture == "sm120":
        return dict(SM120_RUNTIME_BASELINE)
    raise ValueError(f"unknown CUDA architecture: {architecture}")


def space_families(space: dict[str, object]) -> tuple[str, ...]:
    architecture = str(space.get("architecture"))
    expected = architecture_families(architecture)
    actual = space.get("families")
    if actual != list(expected):
        raise ValueError(
            f"search-space family contract differs from {architecture}: "
            f"{actual} != {list(expected)}"
        )
    return expected


def candidate(candidate_id: str, implementation: str = "config", **values: object) -> dict[str, object]:
    return {"id": candidate_id, "implementation": implementation, **values}


def artifact_candidate_identity(value: dict[str, object]) -> dict[str, object]:
    """Return candidate fields that can affect generated source/object code.

    ``config`` contains runtime dispatch/control-flow overrides. For example,
    Linear2 requires fused-residual routing, but adding that flag does not
    alter its already generated GEMM translation unit. All generator inputs
    and resource metadata remain identity fields.
    """
    return {key: item for key, item in value.items() if key != "config"}


def _config_candidate(
    family: str, batch: int, candidate_id: str, **config: object
) -> dict[str, object]:
    return candidate(
        candidate_id, "config", batch=batch, history_family=family, config=config,
    )


def _aot_candidate(
    architecture: str,
    family: str,
    batch: int,
    candidate_id: str,
    generator: str,
    **parameters: object,
) -> dict[str, object]:
    config_keys = {
        "dual_ffn": f"cudaFusedFFNAotTactic{architecture.title()}",
        "linear2": f"cudaLinear2AotTactic{architecture.title()}",
    }
    if family not in config_keys:
        raise ValueError(f"{family} has no linked SM89 AOT registry")
    config: dict[str, object] = {config_keys[family]: candidate_id}
    config.update({
        "dual_ffn": {
            "cudaUseWideFFN": True,
            "cudaDualFfnCutlassTacticSm89": "disabled",
        },
        "linear2": {
            "cudaUseFusedResidual": True,
            "cudaLinear2CutlassTacticSm89": "disabled",
            "cudaUseLinear2PostBNSiluSm89": False,
        },
    }[family])
    return candidate(
        candidate_id,
        generator,
        batch=batch,
        tokens=batch * 361,
        exact_batch_aot=True,
        requires_artifact=True,
        generator=generator,
        config=config,
        **parameters,
    )


def _fallback_candidate(architecture: str, family: str, batch: int) -> dict[str, object]:
    key = {
        "dual_ffn": f"cudaFusedFFNAotTactic{architecture.title()}",
        "linear2": f"cudaLinear2AotTactic{architecture.title()}",
    }[family]
    config: dict[str, object] = {key: "disabled"}
    config.update({
        "dual_ffn": {
            "cudaDualFfnCutlassTacticSm89": "disabled",
        },
        "linear2": {
            "cudaLinear2CutlassTacticSm89": "disabled",
            "cudaUseLinear2PostBNSiluSm89": False,
        },
    }[family])
    return candidate(
        f"{family}-fallback", "fallback",
        batch=batch, config=config, tier="control",
    )


def _gemm_candidates(architecture: str, family: str, batch: int) -> list[dict[str, object]]:
    # 4090 portability document section "固定 GEMM AOT": M64/128,
    # N64/128, stages 3-5, swizzle 1/2.  Keep a pruned neighborhood around
    # the historically successful points rather than taking the 24-point
    # Cartesian product for every operator.
    result = [_fallback_candidate(architecture, family, batch)]
    if family == "dual_ffn":
        for tactic in (
            "m128-n64-k32-w64-n32-s3-sw2-exp",
            "m128-n64-k32-w64-n32-s3-sw4-exp",
            "m128-n64-k32-w64-n32-s3-sw2-tanh-half2",
        ):
            result.append(_config_candidate(
                family, batch, f"dual-cutlass-{tactic}",
                cudaFusedFFNAotTacticSm89="disabled",
                cudaUseWideFFN=True,
                cudaDualFfnCutlassTacticSm89=tactic,
            ))
        # Stage 8/62 center plus its occupancy/resource neighborhood.
        shapes = [
            (128, 64, 32, 2, 3),
            (64, 64, 32, 2, 4),
            (128, 64, 32, 3, 2),
            (64, 64, 32, 3, 2),
            (128, 64, 64, 2, 1),
            (64, 64, 64, 2, 2),
        ]
        for tile_m, tile_n, tile_k, stages, min_blocks in shapes:
            candidate_id = (
                f"dual_ffn-m{tile_m}-n{tile_n}-k{tile_k}-"
                f"s{stages}-mb{min_blocks}-exp"
            )
            result.append(_aot_candidate(
                architecture, family, batch, candidate_id, "tilelang_gemm",
                m=tile_m, n=tile_n, k=tile_k, stages=stages,
                min_blocks=min_blocks, threads=128, epilogue="swiglu-exp",
                a_fragment_reuse=True,
            ))
            result[-1].setdefault("config", {})[
                "cudaDualFfnCutlassTacticSm89"] = "disabled"
        return result
    if family == "linear2":
        cutlass_tactics = (
            "m128-n128-k32-w64-n32-s3-sw1",
            "m128-n128-k32-w64-n32-s4-sw1",
            "m128-n128-k32-w64-n64-s3-sw1",
            "m128-n128-k32-w64-n64-s4-sw1",
            "m128-n128-k32-w64-n64-s5-sw1",
            "m128-n128-k32-w64-n64-s6-sw1",
        )
        for tactic in cutlass_tactics:
            result.append(_config_candidate(
                family, batch, f"linear2-cutlass-{tactic}",
                cudaLinear2AotTacticSm89="disabled",
                cudaUseFusedResidual=True,
                cudaLinear2CutlassTacticSm89=tactic,
                cudaUseLinear2PostBNSiluSm89=False,
            ))
        result.append(_config_candidate(
            family, batch,
            "linear2-cutlass-m128-n128-k32-w64-n64-s4-sw1-postbn",
            cudaLinear2AotTacticSm89="disabled",
            cudaUseFusedResidual=True,
            cudaLinear2CutlassTacticSm89=
                "m128-n128-k32-w64-n64-s4-sw1",
            cudaUseLinear2PostBNSiluSm89=True,
        ))
        for tile_m, tile_n, tile_k, stages, min_blocks, smem in (
            (128, 128, 32, 4, 1, 65536),
            (128, 128, 32, 3, 2, 49152),
            (128, 96, 32, 4, 2, None),
            (64, 128, 32, 4, 2, None),
        ):
            candidate_id = (
                f"linear2-m{tile_m}-n{tile_n}-k{tile_k}-"
                f"s{stages}-mb{min_blocks}"
            )
            result.append(_aot_candidate(
                architecture, family, batch, candidate_id, "tilelang_gemm",
                m=tile_m, n=tile_n, k=tile_k, stages=stages,
                min_blocks=min_blocks, threads=128,
                dynamic_smem_bytes=smem, epilogue="beta1-residual",
            ))
            result[-1].setdefault("config", {})[
                "cudaLinear2CutlassTacticSm89"] = "disabled"
        return result
    raise ValueError(f"{family} has no SM89 GEMM tactic registry")


def _history_candidates(architecture: str, family: str, batch: int) -> list[dict[str, object]]:
    if family in ("dual_ffn", "linear2"):
        return _gemm_candidates(architecture, family, batch)
    toggle_keys = {
        "fused_residual": "cudaUseFusedResidual",
        "initial_conv": "cudaUseInitialConvFrontend",
        "initial_global": "cudaUseInitialGlobalMatMulAdd",
        "head_bn": "cudaUseHeadBNHalfToFloat",
        "value_terminal": "cudaUseFusedValueTerminalSm89",
        "weight_sharing": "cudaShareModelWeights",
    }
    if family in toggle_keys:
        key = toggle_keys[family]
        on_config: dict[str, object] = {key: True}
        return [
            _config_candidate(family, batch, f"{family}-off", **{key: False}),
            _config_candidate(family, batch, f"{family}-on", **on_config),
        ]
    if family == "wide_projection":
        return [
            _config_candidate(
                family, batch, "wide-projection-off",
                cudaUseWideQKV=False,
                cudaUseWideFFN=False,
            ),
            _config_candidate(
                family, batch, "wide-projection-both",
                cudaUseWideQKV=True,
                cudaUseWideFFN=True,
            ),
        ]
    if family == "policy_p1":
        values = [
            _config_candidate(
                family, batch, "policy-p1-disabled",
                cudaPolicyP1RowsPerBlockSm89=0,
                # Wide-head storage is consumed by the fused P1 route. A
                # disabled P1 control must disable that producer as well.
                cudaUseWideHeadProjection=False,
            ),
            _config_candidate(
                family, batch, "policy-p1-block96x1",
                cudaPolicyP1RowsPerBlockSm89=1,
            ),
            _config_candidate(
                family, batch, "policy-p1-block96x5",
                cudaPolicyP1RowsPerBlockSm89=5,
            ),
        ]
        return values
    if family == "rmsnorm":
        return [
            _config_candidate(
                family, batch, "rmsnorm-off",
                cudaUseRMSNormOpt=False,
                cudaRMSNormRowsPerBlockSm89=4,
            ),
            _config_candidate(
                family, batch, "rmsnorm-warps4",
                cudaUseRMSNormOpt=True,
                cudaRMSNormRowsPerBlockSm89=4,
            ),
            _config_candidate(
                family, batch, "rmsnorm-warps8",
                cudaUseRMSNormOpt=True,
                cudaRMSNormRowsPerBlockSm89=8,
            ),
        ]
    if family == "outproj":
        values = [_config_candidate(
            family, batch, "outproj-off",
            cudaOutProjCutlassTacticSm89="disabled",
        )]
        for tactic in (
            "m128-n128-k32-w64-n32-s2-sw1",
            "m128-n128-k32-w64-n32-s3-sw1",
            "m128-n128-k32-w64-n32-s4-sw1",
            "m128-n128-k32-w64-n64-s3-sw1",
            "m128-n128-k32-w64-n64-s4-sw1",
        ):
            values.append(_config_candidate(
                family, batch, f"outproj-cutlass-{tactic}",
                cudaUseFusedResidual=True,
                cudaOutProjCutlassTacticSm89=tactic,
            ))
        return values
    if family == "preconv":
        values = [_config_candidate(
            family, batch, "preconv-off",
            cudaPreConvCutlassTacticSm89="disabled",
        )]
        for tactic in (
            "m128-n128-k32-w64-n32-s3-sw1",
            "m128-n128-k32-w64-n32-s4-sw1",
            "m128-n128-k32-w64-n64-s3-sw1",
            "m128-n128-k32-w64-n64-s4-sw1",
            "m128-n128-k32-w64-n64-s5-sw1",
            "m128-n128-k32-w64-n64-s6-sw1",
        ):
            values.append(_config_candidate(
                family, batch, f"preconv-cutlass-{tactic}",
                cudaPreConvCutlassTacticSm89=tactic,
            ))
        return values
    if family == "qkv_rope":
        reset = {
            "cudaUseFusedQKRoPE": False,
            "cudaUsePrecomputedQKRoPESm89": False,
            "cudaUseQKVRoPEGemmSm89": False,
            "cudaUseSplitQKVRoPEGemmSm89": False,
            "cudaPlainQKVVariantSm89": 0,
            "cudaRoPEBatchGroupSm89": 1,
        }
        values = [
            _config_candidate(family, batch, "qkv-rope-official", **reset),
            _config_candidate(
                family, batch, "qkv-rope-fused", **{
                    **reset, "cudaUseFusedQKRoPE": True,
                },
            ),
            _config_candidate(
                family, batch, "qkv-rope-precomputed", **{
                    **reset, "cudaUseFusedQKRoPE": True,
                    "cudaUsePrecomputedQKRoPESm89": True,
                },
            ),
            _config_candidate(
                family, batch, "qkv-rope-gemm-epilogue", **{
                    **reset, "cudaUseWideQKV": True,
                    "cudaUseQKVRoPEGemmSm89": True,
                },
            ),
            _config_candidate(
                family, batch, "qkv-rope-gemm-epilogue-precomputed", **{
                    **reset, "cudaUseWideQKV": True,
                    "cudaUseFusedQKRoPE": True,
                    "cudaUsePrecomputedQKRoPESm89": True,
                    "cudaUseQKVRoPEGemmSm89": True,
                },
            ),
        ]
        for group in sorted({2, 3, 4, 7, 13, batch}):
            values.append(_config_candidate(
                family, batch, f"qkv-rope-group-{group}", **{
                    **reset, "cudaUseFusedQKRoPE": True,
                    "cudaRoPEBatchGroupSm89": group,
                },
            ))
        for variant in (0, 1):
            values.append(_config_candidate(
                family, batch, f"qkv-rope-gemm-split-v{variant}", **{
                    **reset, "cudaUseWideQKV": True,
                    "cudaUseFusedQKRoPE": True,
                    "cudaUseQKVRoPEGemmSm89": True,
                    "cudaUseSplitQKVRoPEGemmSm89": True,
                    "cudaPlainQKVVariantSm89": variant,
                },
            ))
        return values
    if family == "fa4":
        return [
            _config_candidate(
                family, batch, "fa4-off",
                cudaFlashAttentionTacticSm89="disabled",
            ),
            _config_candidate(
                family, batch, "fa4-d32-m128-n112-w4-pack0-fp32",
                cudaFlashAttentionTacticSm89="d32-m128-n112-w4-pack0-fp32",
            ),
            _config_candidate(
                family, batch, "fa4-d32-m128-n96-w4-pack0-fp32",
                cudaFlashAttentionTacticSm89="d32-m128-n96-w4-pack0-fp32",
            ),
            _config_candidate(
                family, batch, "fa4-d32-m64-n96-w4-pack1-fp32",
                cudaFlashAttentionTacticSm89="d32-m64-n96-w4-pack1-fp32",
            ),
            _config_candidate(
                family, batch, "fa4-d32-m64-n96-w4-pack0-fp32",
                cudaFlashAttentionTacticSm89="d32-m64-n96-w4-pack0-fp32",
            ),
            _config_candidate(
                family, batch, "fa4-d32-m64-n96-w4-pack0-both16",
                cudaFlashAttentionTacticSm89="d32-m64-n96-w4-pack0-both16",
            ),
        ]
    if family == "postconv_bn":
        values = [_config_candidate(
            family, batch, "postconv-off",
            cudaPostConvCutlassTacticSm89="disabled",
            cudaUsePostConvBNSiluSm89=False,
        )]
        for tactic in (
            "m128-n128-k32-w64-n32-s2-sw1",
            "m128-n128-k32-w64-n32-s3-sw1",
            "m128-n128-k32-w64-n32-s3-sw2",
            "m128-n128-k32-w64-n64-s3-sw1",
            "m128-n128-k32-w64-n64-s3-sw2",
            "m128-n128-k32-w64-n64-s3-sw4",
            "m128-n256-k32-w64-n64-s2-sw2",
            "m256-n128-k32-w64-n64-s2-sw1",
            "m256-n128-k32-w64-n64-s2-sw2",
        ):
            values.append(_config_candidate(
                family, batch, f"postconv-cutlass-{tactic}",
                cudaPostConvCutlassTacticSm89=tactic,
                cudaUsePostConvBNSiluSm89=False,
            ))
        values.append(_config_candidate(
            family, batch,
            "postconv-cutlass-m128-n128-k32-w64-n64-s3-sw1-bn-silu",
            cudaPostConvCutlassTacticSm89=
                "m128-n128-k32-w64-n64-s3-sw1",
            cudaUsePostConvBNSiluSm89=True,
        ))
        return values
    if family == "pointwise":
        values = [
            _config_candidate(
                family, batch, "pointwise-off",
                cudaUseScaleBiasSiluVec8Sm89=False,
                cudaUseScaleBiasSiluVec8C384Sm89=False,
                cudaUseScaleBiasSiluVec4C384Sm89=False,
            ),
            _config_candidate(
                family, batch, "pointwise-c768-vec8",
                cudaUseScaleBiasSiluVec8Sm89=True,
                cudaUseScaleBiasSiluVec8C384Sm89=False,
                cudaUseScaleBiasSiluVec4C384Sm89=False,
            ),
            _config_candidate(
                family, batch, "pointwise-c384-vec8",
                cudaUseScaleBiasSiluVec8Sm89=False,
                cudaUseScaleBiasSiluVec8C384Sm89=True,
                cudaUseScaleBiasSiluVec4C384Sm89=False,
            ),
            _config_candidate(
                family, batch, "pointwise-c384-vec4",
                cudaUseScaleBiasSiluVec8Sm89=False,
                cudaUseScaleBiasSiluVec8C384Sm89=False,
                cudaUseScaleBiasSiluVec4C384Sm89=True,
            ),
            _config_candidate(
                family, batch, "pointwise-c768-vec8-c384-vec8",
                cudaUseScaleBiasSiluVec8Sm89=True,
                cudaUseScaleBiasSiluVec8C384Sm89=True,
                cudaUseScaleBiasSiluVec4C384Sm89=False,
            ),
            _config_candidate(
                family, batch, "pointwise-c768-vec8-c384-vec4",
                cudaUseScaleBiasSiluVec8Sm89=True,
                cudaUseScaleBiasSiluVec8C384Sm89=False,
                cudaUseScaleBiasSiluVec4C384Sm89=True,
            ),
        ]
        # The accepted postconv+BN+SiLU boundary removes every C384 affine
        # SiLU launch. A standalone C384 tactic must therefore own that
        # boundary explicitly; otherwise it is a no-op candidate that can
        # never provide runtime activation evidence. C768 remains independent.
        for value in values:
            if "c384" in str(value["id"]):
                value["config"]["cudaUseLinear2PostBNSiluSm89"] = False
                value["config"]["cudaUsePostConvBNSiluSm89"] = False
                value["supersedes"] = ["postconv_bn"]
                value["overrides_keys"] = [
                    "cudaUseLinear2PostBNSiluSm89",
                ]
        return values
    if family == "l2":
        values = [_config_candidate(
            family, batch, f"l2-b{batch}-off",
            cudaUsePersistingL2Trunk=False,
            cudaUsePersistingL2Inner=False,
            cudaPersistingL2HitRatioSm89=1.0,
        )]
        ratio_key = f"cudaPersistingL2HitRatio{architecture.title()}"
        for trunk, inner in ((True, False), (False, True), (True, True)):
            scope = "trunk-inner" if trunk and inner else ("trunk" if trunk else "inner")
            for ratio in (0.5, 0.75, 1.0):
                value = _config_candidate(
                    family, batch,
                    f"l2-b{batch}-{scope}-r{str(ratio).replace('.', 'p')}",
                    **{
                        "cudaUsePersistingL2Trunk": trunk,
                        "cudaUsePersistingL2Inner": inner,
                        ratio_key: ratio,
                    },
                )
                value["actual_grant_limited"] = True
                values.append(value)
        return values
    if family == "wide_head":
        values = [
            _config_candidate(
                family, batch, "wide-head-off", cudaUseWideHeadProjection=False,
            ),
            _config_candidate(
                family, batch, "wide-head-on",
                cudaPolicyP1RowsPerBlockSm89=5,
                cudaUseWideHeadProjection=True,
            ),
            _config_candidate(
                family, batch, "wide-head-stage52-intrinsic-bundle",
                cudaUseInitialGlobalMatMulAdd=True,
                cudaUseHeadBNHalfToFloat=True,
                cudaPolicyP1RowsPerBlockSm89=5,
                cudaUseWideHeadProjection=True,
            ),
        ]
        return values
    raise ValueError(f"unsupported tactic family: {family}")


def _sm89_candidates(family: str, batch: int) -> list[dict[str, object]]:
    # Every coordinate must be allowed to retain the state inherited from the
    # accepted config and earlier family winners. Without this explicit no-op,
    # a family whose whole local neighborhood regresses is forced to accept
    # the least-bad regression (observed at B15 qkv_rope).
    values = [
        _config_candidate(family, batch, f"{family}-keep-incumbent"),
        *_history_candidates("sm89", family, batch),
    ]
    for value in values:
        config = candidate_config(family, value)
        # These later boundaries deliberately take ownership of one key from
        # an earlier, still otherwise-effective family.  Keep that partial
        # ownership explicit so plan construction cannot silently depend on
        # dict update order.
        partial_overrides = {
            "qkv_rope": {"cudaUseWideQKV"},
            "dual_ffn": {"cudaUseWideFFN"},
            "linear2": {
                "cudaUseFusedResidual",
            },
            "outproj": {"cudaUseFusedResidual"},
            "initial_global": {"cudaUseInitialGlobalMatMulAdd"},
            "policy_p1": {
                "cudaPolicyP1RowsPerBlockSm89",
                "cudaUseWideHeadProjection",
            },
            "head_bn": {"cudaUseHeadBNHalfToFloat"},
        }
        overridden_keys = sorted(
            set(config) & partial_overrides.get(family, set())
        )
        if overridden_keys:
            value["overrides_keys"] = overridden_keys
        markers: list[str] = []
        for key, item in config.items():
            if key in {
                "cudaPersistingL2HitRatioSm89",
            }:
                continue
            if key in {
                "cudaDualFfnCutlassTacticSm89",
                "cudaFusedFFNAotTacticSm89",
                "cudaFlashAttentionTacticSm89",
                "cudaLinear2CutlassTacticSm89",
                "cudaLinear2AotTacticSm89",
                "cudaOutProjCutlassTacticSm89",
                "cudaPreConvCutlassTacticSm89",
                "cudaPostConvCutlassTacticSm89",
            }:
                if isinstance(item, str) and item != "disabled":
                    markers.append(
                        "SM89 backend: runtime tactic active: " +
                        f"{key}={item}"
                    )
                continue
            if key in {
                "cudaPlainQKVVariantSm89", "cudaRoPEBatchGroupSm89",
                "cudaRMSNormRowsPerBlockSm89", "cudaPolicyP1RowsPerBlockSm89",
            }:
                if key == "cudaRMSNormRowsPerBlockSm89":
                    if item == 8:
                        markers.append(
                            "SM89 backend: runtime tactic active: " + key + "=8"
                        )
                elif key == "cudaPolicyP1RowsPerBlockSm89":
                    if item in (1, 5):
                        markers.append(
                            "SM89 backend: runtime tactic active: " +
                            f"{key}={item}"
                        )
                elif isinstance(item, int) and item not in (0, 1):
                    markers.append(
                        "SM89 backend: runtime tactic active: " + key
                    )
                elif key == "cudaPlainQKVVariantSm89" and item == 1:
                    markers.append(
                        "SM89 backend: runtime tactic active: " + key
                    )
                continue
            if item is True:
                markers.append(
                    "SM89 backend: runtime tactic active: " + key
                )
        if markers:
            value["activation_markers"] = sorted(set(markers))
    return values


def _sm120_value(
    family: str,
    batch: int,
    candidate_id: str,
    implementation: str,
    config: dict[str, object],
    **parameters: object,
) -> dict[str, object]:
    generated = {
        "tilelang": "tilelang",
        "historical_tilelang": "historical_tilelang",
        "fa4_cute": "fa4_cute",
    }
    if implementation == "cute":
        generated[implementation] = {
            "qkv_rope": "cute_qkv_rope",
            "dual_ffn": "cute_fused_ffn",
        }[family]
    if implementation in generated:
        parameters["requires_artifact"] = True
        parameters.setdefault("generator", generated[implementation])
        parameters.pop("prelinked_artifact", None)
    return candidate(
        candidate_id,
        implementation,
        batch=batch,
        history_family=family,
        config=config,
        **parameters,
    )


def _sm120_toggle(
    family: str,
    batch: int,
    key: str,
    *,
    marker: str | None = None,
) -> list[dict[str, object]]:
    enabled = _sm120_value(
        family, batch, f"{family}-on", "builtin", {key: True},
    )
    if marker is not None:
        enabled["activation_markers"] = [marker]
    return [
        _sm120_value(
            family, batch, f"{family}-off", "fallback", {key: False},
        ),
        enabled,
    ]


SM120_WIDE_QKV_ROUTES: tuple[tuple[str, str, str, dict[str, object]], ...] = (
    (
        "wide_qkv-fallback-three-gemm", "fallback", "planar", {},
    ),
    (
        "wide_qkv-strided-batched", "builtin", "planar", {},
    ),
    (
        "wide_qkv-m128-n128-k64-s2-tilelang-planar", "tilelang", "planar",
        {"m": 128, "n": 128, "k": 64, "stages": 2,
         "threads": 128, "min_blocks": 3},
    ),
    (
        "wide_qkv-m128-n128-k32-s3-tilelang-planar", "tilelang", "planar",
        {"m": 128, "n": 128, "k": 32, "stages": 3,
         "threads": 128, "min_blocks": 3},
    ),
    (
        "wide_qkv-m64-n128-k32-s3-tilelang-planar", "tilelang", "planar",
        {"m": 64, "n": 128, "k": 32, "stages": 3,
         "threads": 128, "min_blocks": 3},
    ),
    (
        "wide_qkv-m128-n128-k64-s2-cute-atom2x2-packed", "cute", "packed",
        {"m": 128, "n": 128, "k": 64, "stages": 2,
         "threads": 160, "copy_atom": "2x2"},
    ),
    (
        "wide_qkv-m128-n128-k64-s2-cute-atom4x2-packed", "cute", "packed",
        {"m": 128, "n": 128, "k": 64, "stages": 2,
         "threads": 288, "copy_atom": "4x2"},
    ),
)


def _sm120_qkv_route_config(candidate_id: str) -> dict[str, object]:
    if candidate_id == "wide_qkv-fallback-three-gemm":
        return {
            "cudaUseQKVGemmAot": False,
            "cudaUseQKVStridedSm120": False,
            "cudaWideQKVAotTacticSm120": "disabled",
        }
    if candidate_id == "wide_qkv-strided-batched":
        return {
            "cudaUseQKVGemmAot": False,
            "cudaUseQKVStridedSm120": True,
            "cudaWideQKVAotTacticSm120": "disabled",
        }
    return {
        "cudaUseWideQKV": True,
        "cudaUseQKVGemmAot": True,
        "cudaUseQKVStridedSm120": False,
        "cudaWideQKVAotTacticSm120": candidate_id,
    }


def _sm120_qkv_route_marker(candidate_id: str) -> str | None:
    if candidate_id == "wide_qkv-fallback-three-gemm":
        return None
    if candidate_id == "wide_qkv-strided-batched":
        return "SM120 backend: strided-batched QKV projection active"
    return "SM120 backend: wide QKV AOT active, tactic=" + candidate_id


def _sm120_candidates(
    family: str, batch: int, gpu_class: str,
) -> list[dict[str, object]]:
    keep = _config_candidate(family, batch, f"{family}-keep-incumbent")

    if family == "fused_residual":
        return [keep, *_sm120_toggle(
            family, batch, "cudaUseFusedResidualGemmSm120",
            marker="SM120 backend: GEMM beta residual fusion active",
        )]

    if family == "rmsnorm":
        return [
            keep,
            _sm120_value(
                family, batch, "rmsnorm-off", "fallback",
                {"cudaRMSNormTacticSm120": "disabled"},
            ),
            _sm120_value(
                family, batch, "rmsnorm-ordered-ept3", "builtin",
                {"cudaRMSNormTacticSm120": "ordered-ept3"},
                activation_markers=[
                    "SM120 backend: ordered-EPT3 C384 RMSNorm active"
                ],
            ),
            _sm120_value(
                family, batch, "rmsnorm-one-warp", "builtin",
                {"cudaRMSNormTacticSm120": "one-warp-exact"},
                activation_markers=[
                    "SM120 backend: one-warp C384 RMSNorm active"
                ],
            ),
            _sm120_value(
                family, batch, "rmsnorm-vec8", "builtin",
                {"cudaRMSNormTacticSm120": "warp4-vec8"},
                activation_markers=[
                    "SM120 backend: vec8 C384 RMSNorm active"
                ],
            ),
        ]

    if family == "qkv_rope":
        fused_aot_id = (
            "qkv-packed-cute-precomputed-rope-static-register-"
            "both16-epilogue"
        )
        reset = {
            "cudaUseFusedQKRoPE": True,
            "cudaUseFusedQKRoPEHalf2Sm120": False,
            "cudaUseBatchSharedRoPE": False,
            "cudaUseBatchSharedRoPEUnrolledSm120": False,
            "cudaQKVRopeAotTacticSm120": "disabled",
        }
        # QKV projection and RoPE used to be two sequential families even
        # though the latter enumerated and superseded every QKV route again.
        # Keep the official-RoPE controls and all fused/packed combinations in
        # one boundary coordinate so a measured QKV winner cannot be silently
        # replaced later in the same batch.
        values = []
        for qkv_id, implementation, output, parameters in SM120_WIDE_QKV_ROUTES:
            config = _sm120_qkv_route_config(qkv_id)
            markers = []
            qkv_marker = _sm120_qkv_route_marker(qkv_id)
            if qkv_marker is not None:
                markers.append(qkv_marker)
            extra: dict[str, object] = {}
            if implementation == "cute":
                extra["generator"] = "cute_qkv"
            if implementation in {"tilelang", "cute"}:
                extra["artifact_family"] = "wide_qkv"
            if output == "packed":
                config.update({
                    "cudaUseFusedQKRoPE": True,
                    "cudaUseFusedQKRoPEHalf2Sm120": False,
                    "cudaUseBatchSharedRoPE": True,
                    "cudaUseBatchSharedRoPEUnrolledSm120": False,
                })
                markers.append(
                    "SM120 backend: batch-shared fused Q/K RoPE active"
                )
                extra["requires"] = {"fa4.supports_packed": True}
            values.append(_sm120_value(
                family, batch, qkv_id, implementation, config,
                output=output,
                qkv_variant=qkv_id,
                rope_variant=("batch-shared" if output == "packed" else "official"),
                activation_markers=markers,
                prelinked_artifact=True,
                **extra,
                **parameters,
            ))
        rope_modes = {
            "scalar": (
                {}, "SM120 backend: fused Q/K learnable RoPE active",
            ),
            "half2": (
                {"cudaUseFusedQKRoPEHalf2Sm120": True},
                "SM120 backend: half2 fused Q/K RoPE active",
            ),
            "batch-shared": (
                {"cudaUseBatchSharedRoPE": True},
                "SM120 backend: batch-shared fused Q/K RoPE active",
            ),
            "batch-shared-unrolled": (
                {
                    "cudaUseBatchSharedRoPE": True,
                    "cudaUseBatchSharedRoPEUnrolledSm120": True,
                },
                "SM120 backend: unrolled packed batch-shared fused Q/K RoPE active",
            ),
        }
        legacy_route = "wide_qkv-fallback-three-gemm"
        legacy_ids = {
            (legacy_route, "scalar"): "qkv-rope-fused-scalar",
            (legacy_route, "half2"): "qkv-rope-fused-half2",
            (legacy_route, "batch-shared"): "qkv-rope-batch-shared",
            (
                "wide_qkv-m128-n128-k64-s2-cute-atom4x2-packed",
                "batch-shared-unrolled",
            ): "qkv-rope-batch-shared-unrolled",
        }
        for qkv_id, qkv_implementation, output, _ in SM120_WIDE_QKV_ROUTES:
            # Packed+batch-shared is already the base packed candidate above.
            modes = (
                ("scalar", "half2", "batch-shared")
                if output == "planar" else
                ("batch-shared-unrolled",)
            )
            for rope_mode in modes:
                candidate_id = legacy_ids.get(
                    (qkv_id, rope_mode),
                    f"qkv-rope-{rope_mode}-with-{qkv_id}",
                )
                rope_config, rope_marker = rope_modes[rope_mode]
                config = {
                    **reset,
                    **_sm120_qkv_route_config(qkv_id),
                    **rope_config,
                }
                markers = [rope_marker]
                requires = (
                    {"fa4.supports_packed": True}
                    if output == "packed" else {}
                )
                qkv_marker = _sm120_qkv_route_marker(qkv_id)
                if qkv_marker is not None:
                    markers.insert(0, qkv_marker)
                artifact_dependencies = []
                if qkv_implementation in {"tilelang", "cute"}:
                    artifact_dependencies.append({
                        "family": "qkv_rope", "candidate_id": qkv_id,
                    })
                values.append(_sm120_value(
                    family, batch, candidate_id, "builtin_bundle", config,
                    qkv_variant=qkv_id,
                    rope_variant=rope_mode,
                    requires=requires,
                    artifact_dependencies=artifact_dependencies,
                    activation_markers=markers,
                ))
        values.append(_sm120_value(
            family, batch, fused_aot_id, "cute",
            {
                **reset,
                "cudaUseWideQKV": True,
                "cudaUseQKVGemmAot": True,
                "cudaUseQKVStridedSm120": False,
                "cudaWideQKVAotTacticSm120": "disabled",
                "cudaQKVRopeAotTacticSm120": fused_aot_id,
            },
            exact_batch_aot=True,
            packed_output=True,
            rope_epilogue="fp16-register-fragment",
            requires_artifact=True,
            generator="cute_qkv_rope",
            requires={"fa4.supports_packed": True},
            activation_markers=[
                "SM120 backend: packed QKV+RoPE AOT active, tactic=" +
                fused_aot_id,
            ],
        ))
        for value in values:
            overridden = sorted(
                set(candidate_config(family, value)) & {
                    "cudaUseWideQKV", "cudaUseQKVGemmAot",
                    "cudaUseQKVStridedSm120", "cudaWideQKVAotTacticSm120",
                    "cudaUseFusedQKRoPE", "cudaUseFusedQKRoPEHalf2Sm120",
                    "cudaUseBatchSharedRoPE",
                    "cudaUseBatchSharedRoPEUnrolledSm120",
                    "cudaQKVRopeAotTacticSm120",
                }
            )
            if overridden:
                value["overrides_keys"] = overridden
        return [keep, *values]

    if family == "fa4":
        values = []
        # Accumulator policy and N tile both changed winners during the 5080
        # and 5090D histories. They are independent coordinates: every exact
        # batch must be allowed to rediscover any precision-valid combination.
        for accumulation in ("fp32", "qk16", "pv16", "both16"):
            for tile_n in (64, 96, 128):
                candidate_id = (
                    f"fa4-b{batch}-s361-h12-d32-tm128-tn{tile_n}-"
                    f"s1-{accumulation}"
                )
                values.append(_sm120_value(
                    family, batch, candidate_id, "fa4_cute",
                    {
                        "cudaUseFlashAttentionSm120": True,
                        "cudaFlashAttentionSm120Accum": accumulation,
                        "cudaFlashAttentionAotTacticSm120": candidate_id,
                    },
                    seq_len=361, heads=12, head_dim=32, tile_m=128,
                    tile_n=tile_n, num_stages=1,
                    accumulation=accumulation,
                    supports_packed=True,
                    exact_shape_aot=True, requires_artifact=True,
                    generator="fa4_cute",
                    activation_markers=[
                        "SM120 backend: FA4 AOT active, tactic=" + candidate_id
                    ],
                ))
        values.append(_sm120_value(
            family, batch, "fa4-official-attention", "fallback",
            {
                "cudaUseFlashAttentionSm120": False,
            },
            supports_packed=False,
        ))
        return [keep, *values]

    if family == "dual_ffn":
        # These were previously three sequential families that repeatedly
        # rewrote cudaUseWideFFNSingleGemm. They are one mutually-exclusive
        # FFN boundary and are therefore scanned as one coordinate.
        values = [
            _sm120_value(
                family, batch, "wide_ffn-single-projection", "builtin",
                {
                    "cudaUseFusedFFN": False,
                    "cudaFusedFFNAotTacticSm120": "disabled",
                    "cudaUseWideFFNSingleGemm": True,
                    "cudaUseSwiGLU1152Sm120": False,
                },
                activation_markers=[
                    "SM120 backend: single-wide FFN projection active"
                ],
            ),
            _sm120_value(
                family, batch, "swiglu-on", "builtin",
                {
                    "cudaUseFusedFFN": False,
                    "cudaFusedFFNAotTacticSm120": "disabled",
                    "cudaUseWideFFNSingleGemm": False,
                    "cudaUseSwiGLU1152Sm120": True,
                },
                activation_markers=[
                    "SM120 backend: contiguous half8 C1152 SwiGLU active"
                ],
            ),
        ]
        cutlass_shared_a_id = (
            "dual_ffn-cutlass-shared-a-m128-n64-k32-s3-swizzle2"
        )
        values.append(_sm120_value(
            family, batch, cutlass_shared_a_id, "builtin_cutlass",
            {
                "cudaUseFusedFFN": True,
                "cudaFusedFFNAotTacticSm120": cutlass_shared_a_id,
            },
            m=128, n=64, k=32, stages=3, swizzle=2,
            shared_a=True, dynamic_batch=True,
            activation_markers=[
                "SM120 backend: CUTLASS shared-A dual FFN active, tactic=" +
                cutlass_shared_a_id
            ],
        ))
        native_max_active_clusters = {
            "rtx5080": 168,
            "rtx5090d": 340,
            "sm120": 168,
        }[gpu_class]
        # Stage47's accepted 5090D coordinate used grid340. Keep both explicit
        # persistent-grid limits in every SM120 scan: the plan, not a GPU-name
        # conditional, chooses the winner and can reproduce either result.
        for max_active_clusters in dict.fromkeys((native_max_active_clusters, 168, 340)):
            cute_id = (
                "dual_ffn-cute-m128-n64x2-k32-ab2-epi4-"
                f"grid{max_active_clusters}"
            )
            values.append(_sm120_value(
                family, batch, cute_id, "cute",
                {
                    "cudaUseFusedFFN": True,
                    "cudaFusedFFNAotTacticSm120": cute_id,
                },
                m=128, n=128, effective_n=64, k=32,
                ab_stages=2, epilogue_stages=4,
                max_active_clusters=max_active_clusters,
                paired_weights=True, swiglu="exp", exact_batch_aot=True,
                requires_artifact=True, generator="cute_fused_ffn",
                activation_markers=[
                    "SM120 backend: fused FFN AOT active, tactic=" + cute_id
                ],
            ))
        shapes = (
            (128, 64, 32, 2, 3),
            (64, 64, 32, 2, 4),
            (128, 64, 32, 3, 2),
            (64, 64, 32, 3, 2),
            (128, 64, 64, 2, 1),
            (64, 64, 64, 2, 2),
        )
        original_exp_id = "dual_ffn-m128-n64-k32-s2-mb3-exp"
        values.append(_sm120_value(
            family, batch, original_exp_id, "tilelang",
            {
                "cudaUseFusedFFN": True,
                "cudaFusedFFNAotTacticSm120": original_exp_id,
            },
            m=128, n=64, k=32, stages=2, threads=128, min_blocks=3,
            a_fragment_reuse=False, swiglu="exp",
            prelinked_artifact=True,
            activation_markers=[
                "SM120 backend: fused FFN AOT active, tactic=" +
                original_exp_id
            ],
        ))
        for tile_m, tile_n, tile_k, stages, min_blocks in shapes:
            candidate_id = (
                f"dual_ffn-m{tile_m}-n{tile_n}-k{tile_k}-"
                f"s{stages}-mb{min_blocks}-areuse-exp"
            )
            values.append(_sm120_value(
                family, batch, candidate_id, "tilelang",
                {
                    "cudaUseFusedFFN": True,
                    "cudaFusedFFNAotTacticSm120": candidate_id,
                },
                m=tile_m, n=tile_n, k=tile_k, stages=stages,
                min_blocks=min_blocks, a_fragment_reuse=True, swiglu="exp",
                prelinked_artifact=True,
                activation_markers=[
                    "SM120 backend: fused FFN AOT active, tactic=" + candidate_id
                ],
            ))
        historical_id = "dual_ffn-m128-n64-k32-s2-mb3-tanh-half2"
        values.append(_sm120_value(
            family, batch, historical_id, "historical_tilelang",
            {
                "cudaUseFusedFFN": True,
                "cudaFusedFFNAotTacticSm120": historical_id,
            },
            m=128, n=64, k=32, stages=2, min_blocks=3,
            a_fragment_reuse=False, swiglu="tanh_half2",
            prelinked_artifact=True,
            activation_markers=[
                "SM120 backend: fused FFN AOT active, tactic=" + historical_id
            ],
        ))
        values.append(_sm120_value(
            family, batch, "dual_ffn-fallback-cublas-swiglu", "fallback",
            {
                "cudaUseFusedFFN": False,
                "cudaFusedFFNAotTacticSm120": "disabled",
                "cudaUseWideFFNSingleGemm": False,
                "cudaUseSwiGLU1152Sm120": False,
            },
        ))
        for value in values:
            if str(value.get("id", "")).startswith("dual_ffn-"):
                value["config"]["cudaUseWideFFNSingleGemm"] = False
                value["config"]["cudaUseSwiGLU1152Sm120"] = False
            overridden = sorted(
                set(candidate_config(family, value)) & {
                    "cudaUseFusedFFN", "cudaFusedFFNAotTacticSm120",
                    "cudaUseWideFFNSingleGemm",
                }
            )
            if overridden:
                value["overrides_keys"] = overridden
        return [keep, *values]

    if family == "wide_projection":
        return [
            keep,
            _sm120_value(
                family, batch, "wide-projections-s1-bundle", "builtin_bundle",
                {
                    "cudaUseWideFFNSingleGemm": True,
                    "cudaUseFusedFFN": False,
                    "cudaFusedFFNAotTacticSm120": "disabled",
                    "cudaUseWideQKV": False,
                    "cudaUseQKVGemmAot": False,
                    "cudaUseQKVStridedSm120": True,
                    "cudaWideQKVAotTacticSm120": "disabled",
                    "cudaUseFusedQKRoPE": False,
                    "cudaUseFusedQKRoPEHalf2Sm120": False,
                    "cudaUseBatchSharedRoPE": False,
                    "cudaUseBatchSharedRoPEUnrolledSm120": False,
                    "cudaQKVRopeAotTacticSm120": "disabled",
                },
                activation_markers=[
                    "SM120 backend: strided-batched QKV projection active",
                    "SM120 backend: single-wide FFN projection active",
                ],
                activation_marker_keys={
                    "SM120 backend: strided-batched QKV projection active": [
                        "cudaUseQKVStridedSm120",
                    ],
                    "SM120 backend: single-wide FFN projection active": [
                        "cudaUseWideFFNSingleGemm",
                    ],
                },
            ),
        ]

    if family == "linear2":
        values = []
        shapes = (
            ("linear2-m256-n64-k32-s4-mb1-tilelang-80k", "tilelang", 256, 64, 32, 4, 128, 1, 81920),
            ("linear2-m128-n128-k32-s2-t128-mb3-tilelang-32k", "tilelang", 128, 128, 32, 2, 128, 3, 32768),
            ("linear2-m128-n128-k32-s3-t128-mb3-tilelang-49k", "tilelang", 128, 128, 32, 3, 128, 3, 49152),
            ("linear2-m128-n128-k32-s3-t256-mb3-tilelang-49k", "tilelang", 128, 128, 32, 3, 256, 3, 49152),
            ("linear2-m128-n128-k32-s4-tilelang-64k", "tilelang", 128, 128, 32, 4, 128, 3, 65536),
            ("linear2-m128-n128-k64-s2-t128-mb3-tilelang-64k", "tilelang", 128, 128, 64, 2, 128, 3, 65536),
            ("linear2-m128-n64-k32-s3-t128-mb3-tilelang-36k", "tilelang", 128, 64, 32, 3, 128, 3, 36864),
            ("linear2-m64-n128-k32-s3-t128-mb4-tilelang-36k", "tilelang", 64, 128, 32, 3, 128, 4, 36864),
            ("linear2-m128-n128-k32-s3-mb2-tilelang-49k", "tilelang", 128, 128, 32, 3, 128, 2, 49152),
            ("linear2-m128-n96-k32-s4-tilelang", "tilelang", 128, 96, 32, 4, 128, 3, None),
            ("linear2-m128-n128-k32-s3-cutlass", "builtin_cutlass", 128, 128, 32, 3, 128, 2, None),
        )
        for candidate_id, implementation, tile_m, tile_n, tile_k, stages, threads, min_blocks, smem in shapes:
            values.append(_sm120_value(
                family, batch, candidate_id, implementation,
                {
                    "cudaUseFusedResidual": True,
                    "cudaUseFusedResidualGemmSm120": True,
                    "cudaUseLinear2ResidualAot": True,
                    "cudaLinear2AotTacticSm120": candidate_id,
                },
                m=tile_m, n=tile_n, k=tile_k, stages=stages,
                threads=threads, min_blocks=min_blocks,
                dynamic_smem_bytes=smem,
                exact_batch_runtime=implementation == "builtin_cutlass",
                prelinked_artifact=implementation == "tilelang",
                overrides_keys=["cudaUseFusedResidualGemmSm120"],
                activation_markers=[
                    "SM120 backend: linear2 residual AOT active, tactic=" + candidate_id
                ],
            ))
        values.append(_sm120_value(
            family, batch, "linear2-fallback-cublas-beta1", "fallback",
            {
                "cudaUseLinear2ResidualAot": False,
                "cudaLinear2AotTacticSm120": "disabled",
            },
        ))
        return [keep, *values]

    if family == "outproj":
        values = []
        shapes = (
            ("outproj-m128-n128-k32-s3-cutlass", "builtin_cutlass", 128, 128, 32, 3, 128, 2, None),
            ("outproj-m128-n128-k32-s3-t128-mb3-tilelang-49k", "tilelang", 128, 128, 32, 3, 128, 3, 49152),
            ("outproj-m128-n128-k32-s4-tilelang-64k", "tilelang", 128, 128, 32, 4, 128, 3, 65536),
            ("outproj-m128-n128-k64-s2-t128-mb3-tilelang-64k", "tilelang", 128, 128, 64, 2, 128, 3, 65536),
            ("outproj-m128-n64-k32-s3-t128-mb3-tilelang-36k", "tilelang", 128, 64, 32, 3, 128, 3, 36864),
            ("outproj-m64-n128-k32-s3-t128-mb4-tilelang-36k", "tilelang", 64, 128, 32, 3, 128, 4, 36864),
            ("outproj-m128-n128-k32-s3-mb2-tilelang-49k", "tilelang", 128, 128, 32, 3, 128, 2, 49152),
        )
        for candidate_id, implementation, tile_m, tile_n, tile_k, stages, threads, min_blocks, smem in shapes:
            values.append(_sm120_value(
                family, batch, candidate_id, implementation,
                {
                    "cudaUseFusedResidualGemmSm120": True,
                    "cudaUseOutProjectionResidualAot": True,
                    "cudaOutProjectionAotTacticSm120": candidate_id,
                },
                m=tile_m, n=tile_n, k=tile_k, stages=stages,
                threads=threads, min_blocks=min_blocks,
                dynamic_smem_bytes=smem,
                exact_batch_runtime=implementation == "builtin_cutlass",
                prelinked_artifact=implementation == "tilelang",
                overrides_keys=["cudaUseFusedResidualGemmSm120"],
                activation_markers=[
                    "SM120 backend: out-projection residual AOT active, tactic=" + candidate_id
                ],
            ))
        values.append(_sm120_value(
            family, batch, "outproj-fallback-cublas-beta1", "fallback",
            {
                "cudaUseOutProjectionResidualAot": False,
                "cudaOutProjectionAotTacticSm120": "disabled",
            },
        ))
        return [keep, *values]

    if family == "preconv":
        values = [
            keep,
            _sm120_value(
                family, batch, "preconv-off", "fallback",
                {"cudaOuterProjectionDownTacticSm120": "disabled"},
            ),
            _sm120_value(
                family, batch, "preconv-cutlass-warp64x64", "builtin_cutlass",
                {"cudaOuterProjectionDownTacticSm120": "warp64x64"},
                activation_markers=[
                    "SM120 backend: C768->C384 outer projection CUTLASS active, tactic=warp64x64"
                ],
            ),
            _sm120_value(
                family, batch, "preconv-cutlass-warp64x32", "builtin_cutlass",
                {"cudaOuterProjectionDownTacticSm120": "warp64x32"},
                activation_markers=[
                    "SM120 backend: C768->C384 outer projection CUTLASS active, tactic=warp64x32"
                ],
            ),
        ]
        for value in values:
            if "cudaOuterProjectionDownTacticSm120" in candidate_config(
                family, value,
            ):
                value["overrides_keys"] = [
                    "cudaOuterProjectionDownTacticSm120"
                ]
        return values

    if family == "postconv_bn":
        return [
            keep,
            _sm120_value(
                family, batch, "postconv-off", "fallback",
                {
                    "cudaOuterProjectionUpTacticSm120": "disabled",
                    "cudaUsePostConvBNSiluSm120": False,
                },
            ),
            _sm120_value(
                family, batch,
                "outer-projection-cutlass-warp64x64-bundle",
                "builtin_cutlass",
                {
                    "cudaOuterProjectionDownTacticSm120": "warp64x64",
                    "cudaOuterProjectionUpTacticSm120": "warp64x64",
                    "cudaUsePostConvBNSiluSm120": False,
                },
                activation_markers=[
                    "SM120 backend: C768->C384 outer projection CUTLASS active, tactic=warp64x64",
                    "SM120 backend: C384->C768 outer projection+residual CUTLASS active, tactic=warp64x64",
                ],
                activation_marker_keys={
                    "SM120 backend: C768->C384 outer projection CUTLASS active, tactic=warp64x64": [
                        "cudaOuterProjectionDownTacticSm120",
                    ],
                },
            ),
            _sm120_value(
                family, batch, "postconv-cutlass-warp64x64", "builtin_cutlass",
                {
                    "cudaOuterProjectionUpTacticSm120": "warp64x64",
                    "cudaUsePostConvBNSiluSm120": False,
                },
                activation_markers=[
                    "SM120 backend: C384->C768 outer projection+residual CUTLASS active, tactic=warp64x64"
                ],
            ),
            _sm120_value(
                family, batch, "postconv-cutlass-warp64x32", "builtin_cutlass",
                {
                    "cudaOuterProjectionUpTacticSm120": "warp64x32",
                    "cudaUsePostConvBNSiluSm120": False,
                },
                activation_markers=[
                    "SM120 backend: C384->C768 outer projection+residual CUTLASS active, tactic=warp64x32"
                ],
            ),
            _sm120_value(
                family, batch, "postconv-cutlass-bn-silu", "builtin_cutlass",
                {
                    "cudaOuterProjectionUpTacticSm120": "disabled",
                    "cudaUsePostConvBNSiluSm120": True,
                },
                activation_markers=[
                    "SM120 backend: postConv residual + following C768 affine SiLU active"
                ],
            ),
        ]

    if family == "pointwise":
        return [
            keep,
            _sm120_value(
                family, batch, "pointwise-off", "fallback",
                {"cudaAffineSiluTacticSm120": "disabled"},
            ),
            _sm120_value(
                family, batch, "pointwise-half2", "builtin",
                {"cudaAffineSiluTacticSm120": "half2"},
                activation_markers=[
                    "SM120 backend: half2 C384/C768 affine SiLU active"
                ],
            ),
            _sm120_value(
                family, batch, "pointwise-half2x3", "builtin",
                {"cudaAffineSiluTacticSm120": "half2x3"},
                activation_markers=[
                    "SM120 backend: half2x3 C384/C768 affine SiLU active"
                ],
            ),
            _sm120_value(
                family, batch, "pointwise-flat-vec8-c768", "builtin",
                {"cudaAffineSiluTacticSm120": "flat-vec8-c768"},
                activation_markers=[
                    "SM120 backend: flat vec8 C768 affine SiLU active"
                ],
            ),
        ]

    if family == "l2":
        values = [_sm120_value(
            family, batch, "l2-off", "fallback",
            {
                "cudaUsePersistingL2Trunk": False,
                "cudaUsePersistingL2Inner": False,
            },
        )]
        for trunk, inner in ((True, False), (False, True), (True, True)):
            scope = (
                "trunk-inner" if trunk and inner else
                ("trunk" if trunk else "inner")
            )
            for ratio in (0.5, 0.75, 1.0):
                markers = []
                if trunk:
                    markers.append("SM120 backend: persisting-L2 C768 trunk active")
                if inner:
                    markers.append("SM120 backend: persisting-L2 C384 inner active")
                values.append(_sm120_value(
                    family, batch,
                    f"l2-{scope}-ratio-{str(ratio).replace('.', 'p')}",
                    "builtin",
                    {
                        "cudaUsePersistingL2Trunk": trunk,
                        "cudaUsePersistingL2Inner": inner,
                        "cudaPersistingL2HitRatioSm120": ratio,
                    },
                    trunk=trunk, inner=inner, hit_ratio=ratio,
                    actual_grant_limited=True, activation_markers=markers,
                ))
        return [keep, *values]

    if family == "weight_sharing":
        return [keep, *_sm120_toggle(
            family, batch, "cudaShareModelWeights",
            marker="SM120 backend: per-device model-weight sharing active",
        )]

    if family == "initial_conv":
        return [
            keep,
            _sm120_value(
                family, batch, "initial-conv-disabled", "fallback",
                {"cudaInitialConvFrontendPlanSm120": "disabled"},
            ),
            _sm120_value(
                family, batch, "initial-conv-eng45-tile0-stages2", "cudnn_frontend",
                {"cudaInitialConvFrontendPlanSm120": "eng45-tile0-stages2"},
                activation_markers=[
                    "SM120 backend: initial-conv frontend eng45/tile0/stages2 active"
                ],
            ),
            _sm120_value(
                family, batch,
                "initial-conv-eng47-k2-2-k6-1-k13-1-k14-0-k22-2",
                "cudnn_frontend",
                {"cudaInitialConvFrontendPlanSm120":
                 "eng47-k2-2-k6-1-k13-1-k14-0-k22-2"},
                activation_markers=[
                    "SM120 backend: initial-conv frontend eng47/k2=2/k6=1/k13=1/k14=0/k22=2 active"
                ],
            ),
        ]
    if family == "initial_global":
        return [keep, *_sm120_toggle(
            family, batch, "cudaUseInitialGlobalMatMulAdd",
            marker="SM120 backend: fused global-feature matmul+broadcast add active",
        )]
    if family == "policy_p1":
        values = [keep, *_sm120_toggle(
            family, batch, "cudaUseFusedPolicyP1",
            marker="SM120 backend: fused 19x19 policy P1 active",
        )]
        for value in values:
            if "cudaUseFusedPolicyP1" in candidate_config(family, value):
                value["overrides_keys"] = ["cudaUseFusedPolicyP1"]
            if candidate_config(family, value).get("cudaUseFusedPolicyP1") is False:
                # The SM120 wide-head route is constructed only when fused P1
                # and direct-FP32 head BN are both active. Turning either
                # prerequisite off removes that route rather than merely
                # changing one independent pointwise kernel.
                value["supersedes"] = ["wide_head"]
        return values
    if family == "wide_head":
        return [
            keep,
            _sm120_value(
                family, batch, "wide-head-off", "fallback",
                {"cudaWideHeadProjectionTacticSm120": "disabled"},
            ),
            _sm120_value(
                family, batch, "wide-head-full-c384", "builtin_cutlass",
                {
                    "cudaWideHeadProjectionTacticSm120": "full-c384",
                    "cudaUseFusedPolicyP1": True,
                    "cudaUseHeadBNHalfToFloat": True,
                },
                activation_markers=[
                    "SM120 backend: full C384 no-split wide head projection active"
                ],
            ),
            _sm120_value(
                family, batch, "wide-head-partial-c288-g1-v1", "builtin_cutlass",
                {
                    "cudaWideHeadProjectionTacticSm120": "partial-c288-g1-v1",
                    "cudaUseFusedPolicyP1": True,
                    "cudaUseHeadBNHalfToFloat": True,
                },
                activation_markers=[
                    "SM120 backend: partial C288 no-split g1+v1 head active"
                ],
            ),
        ]
    if family == "head_bn":
        values = [keep, *_sm120_toggle(
            family, batch, "cudaUseHeadBNHalfToFloat",
            marker="SM120 backend: head BN direct FP32 output active",
        )]
        for value in values:
            if "cudaUseHeadBNHalfToFloat" in candidate_config(family, value):
                value["overrides_keys"] = ["cudaUseHeadBNHalfToFloat"]
            if candidate_config(family, value).get("cudaUseHeadBNHalfToFloat") is False:
                value["supersedes"] = ["wide_head"]
        return values
    if family == "value_terminal":
        return [keep, *_sm120_toggle(
            family, batch, "cudaUseFusedValueTerminalSm120",
            marker="SM120 backend: fused value/score terminal active",
        )]
    raise ValueError(f"unsupported SM120 tactic family: {family}")


def default_candidates(
    architecture: str, family: str, batch: int, gpu_class: str,
) -> list[dict[str, object]]:
    if architecture == "sm89":
        return _sm89_candidates(family, batch)
    if architecture == "sm120":
        return _sm120_candidates(family, batch, gpu_class)
    raise ValueError(f"unsupported architecture: {architecture}")


def positive_history_seed_candidate_ids(
    architecture: str, gpu_class: str, batch: int,
) -> dict[str, str]:
    """Return an explicit known-good whole-graph restart when one is frozen.

    This is deliberately a map of ordinary search-space candidates, not a
    hidden override string.  Every component is therefore measured, carries
    its normal activation proof, can be replaced by coordinate search, and is
    serialized through the normal plan-apply mapping.  Exact-batch IDs are
    materialized for every requested batch; there is no privileged B19 launch.
    """
    if architecture == "sm89" and gpu_class == "rtx4090":
        return {
            "fa4": "fa4-d32-m64-n96-w4-pack0-both16",
            "wide_projection": "wide-projection-both",
            "qkv_rope": "qkv-rope-gemm-epilogue",
            "dual_ffn": (
                "dual-cutlass-m128-n64-k32-w64-n32-s3-sw2-tanh-half2"
            ),
            "fused_residual": "fused_residual-on",
            "linear2": (
                "linear2-cutlass-m128-n128-k32-w64-n64-s3-sw1"
            ),
            "outproj": (
                "outproj-cutlass-m128-n128-k32-w64-n64-s3-sw1"
            ),
            "postconv_bn": (
                "postconv-cutlass-m128-n128-k32-w64-n64-s3-sw1"
            ),
            "pointwise": "pointwise-c768-vec8-c384-vec8",
            "rmsnorm": "rmsnorm-warps4",
            "preconv": (
                "preconv-cutlass-m128-n128-k32-w64-n64-s3-sw1"
            ),
            "l2": f"l2-b{batch}-trunk-inner-r1p0",
            "weight_sharing": "weight_sharing-on",
            "initial_conv": "initial_conv-off",
            "wide_head": "wide-head-on",
            "initial_global": "initial_global-on",
            "policy_p1": "policy-p1-block96x5",
            "head_bn": "head_bn-off",
            "value_terminal": "value_terminal-off",
        }
    if architecture == "sm120" and gpu_class == "rtx5080":
        return {
            "fa4": (
                f"fa4-b{batch}-s361-h12-d32-tm128-tn64-s1-both16"
            ),
            "wide_projection": "wide_projection-keep-incumbent",
            "qkv_rope": "qkv-rope-batch-shared-unrolled",
            "dual_ffn": "dual_ffn-m128-n64-k32-s2-mb3-tanh-half2",
            "fused_residual": "fused_residual-on",
            "linear2": "linear2-m128-n128-k32-s3-cutlass",
            "outproj": "outproj-m128-n128-k32-s3-cutlass",
            "postconv_bn": "postconv-cutlass-warp64x32",
            "preconv": "preconv-cutlass-warp64x64",
            "pointwise": "pointwise-half2",
            "wide_head": "wide-head-full-c384",
            "policy_p1": "policy_p1-on",
            "head_bn": "head_bn-on",
            "rmsnorm": "rmsnorm-vec8",
            "l2": "l2-trunk-inner-ratio-1p0",
            "weight_sharing": "weight_sharing-on",
            "initial_conv": "initial-conv-eng45-tile0-stages2",
            "initial_global": "initial_global-on",
            "value_terminal": "value_terminal-off",
        }
    return {}


def stable_prescan_candidate_ids(
    architecture: str, gpu_class: str, batch: int,
) -> dict[str, str]:
    """Return the artifact-free optimized graph used to rank exact batches.

    The pre-scan intentionally runs before exact-batch source generation.  Its
    graph may therefore use only implementations compiled into the base CUDA
    backend.  It is explicit, fail-closed, and substantially closer to the
    eventual optimized graph than disabling the custom backend altogether.
    """
    if architecture == "sm89" and gpu_class == "rtx4090":
        # Every selected implementation is batch-generic on SM89, so the
        # certified B12 graph itself is a valid B4-B32 ranking baseline.
        return positive_history_seed_candidate_ids(
            architecture, gpu_class, batch=batch,
        )
    if architecture == "sm120" and gpu_class in {"rtx5080", "rtx5090d"}:
        return {
            "fa4": "fa4-official-attention",
            "qkv_rope": (
                "qkv-rope-batch-shared-with-wide_qkv-strided-batched"
            ),
            "dual_ffn": (
                "dual_ffn-cutlass-shared-a-m128-n64-k32-s3-swizzle2"
            ),
            "fused_residual": "fused_residual-on",
            "linear2": "linear2-m128-n128-k32-s3-cutlass",
            "outproj": "outproj-m128-n128-k32-s3-cutlass",
            "postconv_bn": "postconv-cutlass-warp64x64",
            "preconv": "preconv-cutlass-warp64x32",
            "pointwise": "pointwise-half2",
            "wide_head": "wide-head-full-c384",
            "policy_p1": "policy_p1-on",
            "head_bn": "head_bn-on",
            "rmsnorm": "rmsnorm-vec8",
            "l2": "l2-trunk-inner-ratio-1p0",
            "weight_sharing": "weight_sharing-on",
            "initial_conv": "initial-conv-eng45-tile0-stages2",
            "initial_global": "initial_global-on",
            "value_terminal": "value_terminal-off",
        }
    raise ValueError(
        f"no stable pre-scan baseline for {architecture}/{gpu_class}"
    )


def deduplicate_candidates(values: Iterable[dict[str, object]]) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    seen: set[str] = set()
    for value in values:
        candidate_id = value.get("id")
        if not isinstance(candidate_id, str) or not candidate_id:
            raise ValueError("every tactic candidate requires a non-empty id")
        if candidate_id in seen:
            continue
        seen.add(candidate_id)
        result.append(value)
    return result


def load_candidate_files(paths: Sequence[str]) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    for path_text in paths:
        path = pathlib.Path(path_text)
        payload = json.loads(path.read_text())
        entries = payload.get("entries") if isinstance(payload, dict) else payload
        if not isinstance(entries, list):
            raise ValueError(f"candidate file must be a list or {{entries: [...]}}: {path}")
        for entry in entries:
            if not isinstance(entry, dict):
                raise ValueError(f"candidate entry is not an object: {path}")
            family = entry.get("family")
            value = entry.get("candidate", entry)
            if family not in ALL_FAMILIES or not isinstance(value, dict):
                raise ValueError(f"candidate entry needs family and candidate: {path}")
            batches = entry.get("batches")
            if batches is None:
                batches = ["all"]
            elif isinstance(batches, str):
                batches = parse_int_set(batches)
            else:
                batches = [int(item) for item in batches]
            result.append({
                "family": family,
                "batches": batches,
                "candidate": value,
                "source_file": str(path.resolve()),
            })
    return result


def materialize_space(
    architecture: str,
    gpu_class: str,
    device: int,
    batches: Sequence[int],
    streams: int,
    extra_paths: Sequence[str] = (),
    extra_topology: str | None = None,
    device_properties: dict[str, object] | None = None,
) -> dict[str, object]:
    if architecture not in ARCHITECTURES:
        raise ValueError(f"architecture must be one of {tuple(ARCHITECTURES)}")
    validate_gpu_class(architecture, gpu_class)
    if device < 0:
        raise ValueError("CUDA device ordinal must be non-negative")
    if streams < 1:
        raise ValueError("streams must be positive")
    expected_compute_capability = ARCHITECTURES[architecture]["compute_capability"]
    compute_capability = expected_compute_capability
    if device_properties is not None:
        compute_capability = device_properties.get("compute_capability")
        if compute_capability != expected_compute_capability:
            raise ValueError(
                "CUDA-reported compute capability does not match requested "
                f"architecture: {compute_capability} != {expected_compute_capability}"
            )
    extra = load_candidate_files(extra_paths)
    target_families = architecture_families(architecture)
    batch_payloads: list[dict[str, object]] = []
    for batch in sorted(set(int(item) for item in batches)):
        if batch < 1:
            raise ValueError("batch values must be positive")
        batch_space: dict[str, object] = {"batch": batch, "tokens": batch * 361}
        for family in target_families:
            values = default_candidates(architecture, family, batch, gpu_class)
            for entry in extra:
                entry_batches = entry["batches"]
                applies = "all" in entry_batches or batch in entry_batches
                if applies and entry["family"] == family:
                    values.append(entry["candidate"])
            values = deduplicate_candidates(values)
            runtime_keys = (
                SM89_RUNTIME_CONFIG_KEYS
                if architecture == "sm89" else SM120_RUNTIME_CONFIG_KEYS
            )
            for value in values:
                unknown = sorted(set(candidate_config(family, value)) - runtime_keys)
                if unknown:
                    raise ValueError(
                        f"candidate uses unparsed {architecture.upper()} config keys: "
                        f"{family}/B{batch}/"
                        f"{value.get('id')}: {unknown}"
                    )
                validate_candidate_execution_contract(
                    architecture, family, batch, value,
                )
            batch_space[family] = values
        validate_cross_family_config_ownership(
            architecture, batch, batch_space,
        )
        for family in target_families:
            for value in batch_space[family]:
                dependencies = value.get("artifact_dependencies", [])
                if not isinstance(dependencies, list):
                    raise ValueError(
                        f"{architecture}/{family}/B{batch}/{value.get('id')} "
                        "has malformed artifact_dependencies"
                    )
                for dependency in dependencies:
                    if not isinstance(dependency, dict):
                        raise ValueError("artifact dependency is not an object")
                    dependency_family = str(dependency.get("family", ""))
                    dependency_id = str(dependency.get("candidate_id", ""))
                    if dependency_family not in target_families:
                        raise ValueError(
                            f"artifact dependency has unknown family: {dependency}"
                        )
                    dependency_candidates = {
                        str(item["id"]): item
                        for item in batch_space[dependency_family]
                    }
                    target = dependency_candidates.get(dependency_id)
                    if target is None or not target.get("requires_artifact"):
                        raise ValueError(
                            "artifact dependency does not name a generated "
                            f"candidate: {dependency}"
                        )
        batch_payloads.append(batch_space)
    runtime_keys = (
        SM89_RUNTIME_CONFIG_KEYS
        if architecture == "sm89" else SM120_RUNTIME_CONFIG_KEYS
    )
    positive_history_closure = validate_positive_history_closure(
        pathlib.Path(__file__).resolve().parents[1],
        architecture,
        {
            int(item["batch"]): {
                family: item[family]
                for family in target_families
            }
            for item in batch_payloads
        },
        runtime_keys,
    )
    topology = {
        "streams": streams,
        "device_ordinals": [device] * streams,
        "config_overrides": parse_key_values(extra_topology),
        "stream_ownership": "benchmarknn creates one externally-owned compute stream per server thread",
    }
    return {
        "schema": SCHEMA,
        "kind": SPACE_KIND,
        "generated_utc": utc_now(),
        "architecture": architecture,
        "compute_capability": compute_capability,
        "gpu_class": gpu_class,
        "device_ordinal": device,
        "cuda_device_properties_at_space_generation": device_properties,
        "fixed_board": [19, 19],
        "precision": ARCHITECTURES[architecture]["precision"],
        "families": list(target_families),
        "decision_groups": [
            list(group) for group in architecture_decision_groups(architecture)
        ],
        "streams": streams,
        "topology": topology,
        "batch_policy": "only explicitly materialized batches; no anchor or plateau pruning",
        "candidate_policy": {
            "accepted_history_points_define_local_search_neighborhoods_not_winners": True,
            "every_family_is_materialized_for_every_requested_batch": True,
            "every_family_has_an_explicit_keep_incumbent_candidate": True,
            "batch_13_has_no_anchor_or_special_case": True,
            "external_candidate_manifests_are_part_of_the_search_space": True,
            "aot_artifacts_must_be_replayed_or_present_before_production_use": True,
            "historically_positive_routes_require_four_link_closure": True,
        },
        "positive_history_closure": positive_history_closure,
        "history_recipe": {
            "sources": (
                [
                    "/workspace/results/4090/HISTORY.md",
                    "/workspace/4090-optimization-portability.md",
                ]
                if architecture == "sm89" else
                [
                    "SM89_SM120_AUTOTUNE_HANDOVER_20260807.md",
                    "retained SM120 optimization commits",
                ]
            ),
            "execution_order": list(target_families),
            "decision_groups": [
                list(group)
                for group in architecture_decision_groups(architecture)
            ],
            "search_semantics": (
                "accepted-history-seeded coordinate search with accumulated "
                "winners and a non-regressing incumbent at every stage"
            ),
            "positive_history_contract_sha256": positive_history_closure[
                "contract_sha256"
            ],
            "positive_history_record_ids": positive_history_closure[
                "record_ids"
            ],
            "candidate_payload_is_authoritative": True,
            "notes": [
                "No batch is an anchor or a privileged specialization.",
                "Every listed historical route has backend, scan, activation, and plan-apply proofs.",
                "Candidate axes are read from each exact-batch payload; this metadata does not duplicate them.",
            ],
        },
        "batches": batch_payloads,
        "candidate_files": [str(pathlib.Path(path).resolve()) for path in extra_paths],
    }


def make_generation_plan(
    space_path: pathlib.Path,
    *,
    phase: str = "full",
    families: Sequence[str] | None = None,
) -> dict[str, object]:
    space = read_json(space_path)
    if space.get("schema") != SCHEMA or space.get("kind") != SPACE_KIND:
        raise ValueError("generation-plan requires a CUDA tactic search space")
    if phase not in ("seed", "full"):
        raise ValueError("generation phase must be seed or full")
    target_families = space_families(space)
    requested = list(dict.fromkeys(families or target_families))
    if not requested or any(family not in target_families for family in requested):
        raise ValueError(f"invalid generation families: {requested}")
    closure = space.get("positive_history_closure")
    if not isinstance(closure, dict) or not closure.get("complete"):
        raise ValueError("search space lacks a complete positive-history closure")
    complete = phase == "full" and requested == list(target_families)
    tasks: list[dict[str, object]] = []
    coverage: dict[str, dict[str, int]] = {family: {} for family in requested}
    for batch, batch_space in sorted(space_batches(space).items()):
        for family in requested:
            values = list(candidate_map(space, family, batch).values())
            generated = [value for value in values if value.get("requires_artifact")]
            if phase == "seed" and generated:
                # Seed every family at every batch with the historical center;
                # this is a pipeline check, never a winner shortcut.
                generated = [generated[0]]
            coverage[family][str(batch)] = len(generated)
            for value in generated:
                tasks.append({
                    "task_key": f"{space['architecture']}/{family}/B{batch}/{value['id']}",
                    "architecture": space["architecture"],
                    "compute_capability": space["compute_capability"],
                    "gpu_class": space["gpu_class"],
                    "device_ordinal": space["device_ordinal"],
                    "streams": space["streams"],
                    "batch": batch,
                    "tokens": batch * 361,
                    "family": family,
                    "candidate_id": value["id"],
                    "candidate": value,
                    "generator": value["generator"],
                    "output_subdir": f"{family}/b{batch}/{value['id']}",
                    "gates": [
                        "generator correctness",
                        "single-stream local timing for pruning only",
                        "natural whole-graph S2 discovery",
                        "long stable S2 validation before plan",
                    ],
                })
    return {
        "schema": 1,
        "kind": "cuda-tactic-generation-plan",
        "generated_utc": utc_now(),
        "phase": phase,
        "complete_history_coverage": complete,
        "eligible_for_whole_graph_scan": complete,
        "positive_history_closure": closure,
        "source_space": str(space_path.resolve()),
        "space_sha256": sha256_file(space_path),
        "architecture": space["architecture"],
        "gpu_class": space["gpu_class"],
        "batches": sorted(space_batches(space)),
        "families": requested,
        "batch_13_special_case": False,
        "coverage": coverage,
        "tasks": tasks,
    }


def write_json(path: pathlib.Path, payload: object, *, compact: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    encoded = (
        json.dumps(payload, sort_keys=True, separators=(",", ":"))
        if compact else json.dumps(payload, indent=2, sort_keys=True)
    )
    temporary.write_text(encoded + "\n", encoding="utf-8")
    temporary.replace(path)


def read_json(path: pathlib.Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root is not an object: {path}")
    return payload


def _build_sm120_coordinate_artifact_bundle(
    space_path: pathlib.Path,
    space: dict[str, object],
    binary: pathlib.Path,
    manifest_path: pathlib.Path,
) -> dict[str, object]:
    """Normalize the all-family SM120 fat build into the common proof schema."""
    manifest = read_json(manifest_path)
    if (
        manifest.get("kind") != "sm120-coordinate-fat-bundle" or
        not manifest.get("complete")
    ):
        raise ValueError(f"incomplete SM120 coordinate bundle: {manifest_path}")
    if manifest.get("space_sha256") != sha256_file(space_path):
        raise ValueError("SM120 coordinate bundle search-space hash mismatch")
    if manifest.get("binary_sha256") != sha256_file(binary):
        raise ValueError("SM120 coordinate bundle does not prove the selected binary")
    configure = manifest.get("commands", {}).get("configure", [])
    if (
        not isinstance(configure, list) or
        "-DKATAGO_CUDA_ARCHITECTURES=120" not in configure
    ):
        raise ValueError("SM120 coordinate bundle lacks an exact sm120 build command")
    expected = {
        (family, batch, str(value["id"])): value
        for batch in sorted(space_batches(space))
        for family in space_families(space)
        for value in candidate_map(space, family, batch).values()
        if value.get("requires_artifact")
    }
    nm = subprocess.run(
        ["nm", "-a", str(binary)], text=True, capture_output=True, check=False,
    )
    if nm.returncode != 0:
        raise ValueError(f"nm could not inspect linked binary: {nm.stderr.strip()}")
    checked: dict[tuple[str, int, str], dict[str, object]] = {}
    raw_entries = manifest.get("entries", [])
    if not isinstance(raw_entries, list):
        raise ValueError("SM120 coordinate bundle entries must be a list")
    for item in raw_entries:
        if not isinstance(item, dict):
            raise ValueError("SM120 coordinate bundle contains a non-object entry")
        key = (
            str(item.get("family")), int(item.get("batch", -1)),
            str(item.get("candidate_id")),
        )
        if key not in expected:
            raise ValueError(f"unexpected SM120 coordinate artifact: {key}")
        if key in checked:
            raise ValueError(f"duplicate SM120 coordinate artifact: {key}")
        candidate = item.get("candidate")
        if (
            not isinstance(candidate, dict) or
            artifact_candidate_identity(candidate) !=
                artifact_candidate_identity(expected[key])
        ):
            raise ValueError(f"SM120 coordinate candidate drift: {key}")
        files: dict[str, dict[str, object]] = {}
        for name in ("source", "metadata"):
            path_text = item.get(name)
            recorded_hash = item.get(f"{name}_sha256")
            if not path_text or not recorded_hash:
                raise ValueError(f"SM120 coordinate artifact lacks {name}: {key}")
            path = pathlib.Path(str(path_text)).resolve()
            if not path.is_file() or sha256_file(path) != recorded_hash:
                raise ValueError(f"SM120 coordinate {name} hash mismatch: {key}")
            files[name] = {"path": str(path), "sha256": recorded_hash}
        object_hash = item.get("object_sha256")
        if object_hash is not None:
            object_path = pathlib.Path(str(item.get("object", ""))).resolve()
            if not object_path.is_file() or sha256_file(object_path) != object_hash:
                raise ValueError(f"SM120 coordinate object hash mismatch: {key}")
            files["object"] = {"path": str(object_path), "sha256": object_hash}
        launch_symbol = str(item.get("launch_symbol", ""))
        if not launch_symbol or launch_symbol not in nm.stdout:
            raise ValueError(f"SM120 coordinate launcher is not linked: {key}")
        checked[key] = {
            "family": key[0],
            "batch": key[1],
            "candidate_id": key[2],
            "status": "linked",
            "launch_symbol": launch_symbol,
            "source_sha256": item["source_sha256"],
            "object_sha256": object_hash,
            "metadata_sha256": item["metadata_sha256"],
            "files": files,
            "correctness": None,
            "generation_command": item.get("generation_command"),
        }
    missing = sorted(set(expected) - set(checked))
    if missing:
        preview = ", ".join(f"{f}/B{b}/{c}" for f, b, c in missing[:8])
        raise ValueError(
            f"SM120 coordinate bundle is missing {len(missing)} entries: {preview}"
        )
    closure = space["positive_history_closure"]
    return {
        "schema": SCHEMA,
        "kind": ARTIFACT_BUNDLE_KIND,
        "generated_utc": utc_now(),
        "complete_history_coverage": True,
        "positive_history_closure": closure,
        "space": str(space_path.resolve()),
        "space_sha256": sha256_file(space_path),
        "architecture": space["architecture"],
        "gpu_class": space["gpu_class"],
        "linked_binary": str(binary.resolve()),
        "linked_binary_sha256": sha256_file(binary),
        "link_proof": "every generated extern-C launch symbol is present in nm -a output",
        "source_manifests": [{
            "path": str(manifest_path.resolve()),
            "sha256": sha256_file(manifest_path),
            "kind": "sm120-coordinate-fat-bundle",
            "entry_count": len(checked),
            "configure_command": configure,
        }],
        "entries": [checked[key] for key in sorted(checked)],
    }


def build_artifact_bundle(
    space_path: pathlib.Path,
    binary: pathlib.Path,
    manifest_paths: Sequence[pathlib.Path],
) -> dict[str, object]:
    """Combine generated family manifests and prove their launchers are linked."""
    space = read_json(space_path)
    if space.get("schema") != SCHEMA or space.get("kind") != SPACE_KIND:
        raise ValueError("artifact-bundle requires a CUDA tactic search space")
    closure = space.get("positive_history_closure")
    if not isinstance(closure, dict) or not closure.get("complete"):
        raise ValueError("search space lacks a complete positive-history closure")
    if not binary.is_file():
        raise ValueError(f"linked binary does not exist: {binary}")
    if len(manifest_paths) == 1:
        manifest_kind = read_json(manifest_paths[0]).get("kind")
        if manifest_kind == "sm120-coordinate-fat-bundle":
            if space.get("architecture") != "sm120":
                raise ValueError("SM120 coordinate bundle used with a non-SM120 space")
            return _build_sm120_coordinate_artifact_bundle(
                space_path, space, binary, manifest_paths[0],
            )
    space_sha256 = sha256_file(space_path)
    expected = {
        (family, batch, str(value["id"])): value
        for batch in sorted(space_batches(space))
        for family in space_families(space)
        for value in candidate_map(space, family, batch).values()
        if value.get("requires_artifact")
    }
    if not expected:
        if manifest_paths:
            for manifest_path in manifest_paths:
                manifest = read_json(manifest_path)
                if not manifest.get("complete") or manifest.get("entries") != []:
                    raise ValueError(
                        "static-only artifact bundle received a non-empty manifest"
                    )
        return {
            "schema": SCHEMA,
            "kind": ARTIFACT_BUNDLE_KIND,
            "generated_utc": utc_now(),
            "complete_history_coverage": True,
            "positive_history_closure": closure,
            "space": str(space_path.resolve()),
            "space_sha256": sha256_file(space_path),
            "architecture": space["architecture"],
            "gpu_class": space["gpu_class"],
            "linked_binary": str(binary.resolve()),
            "linked_binary_sha256": sha256_file(binary),
            "link_proof": "plan-restricted space requires no generated AOT launcher",
            "source_manifests": [
                {
                    "path": str(path.resolve()),
                    "sha256": sha256_file(path),
                    "entry_count": 0,
                }
                for path in manifest_paths
            ],
            "entries": [],
        }
    nm = subprocess.run(
        ["nm", "-a", str(binary)], text=True, capture_output=True, check=False,
    )
    if nm.returncode != 0:
        raise ValueError(f"nm could not inspect linked binary: {nm.stderr.strip()}")
    entries: dict[tuple[str, int, str], dict[str, object]] = {}
    source_manifests: list[dict[str, object]] = []
    for manifest_path in manifest_paths:
        manifest = read_json(manifest_path)
        family = str(manifest.get("family"))
        if family not in space_families(space) or not manifest.get("complete"):
            raise ValueError(f"incomplete or unsupported generation manifest: {manifest_path}")
        manifest_space_sha256 = str(manifest.get("space_sha256", ""))
        space_binding: dict[str, object] = {
            "kind": "exact_search_space",
            "source_space_sha256": manifest_space_sha256,
            "target_space_sha256": space_sha256,
        }
        if manifest_space_sha256 != space_sha256:
            # Adding non-AOT controls (for example keep-incumbent) must not
            # force hundreds of byte-identical TileLang TUs to be regenerated.
            # Reuse is legal only when the complete generated-candidate
            # projection for this family is exactly equal in the old and new
            # spaces; source/object/metadata hashes and linked symbols are
            # still checked below.
            manifest_space_path = pathlib.Path(str(manifest.get("space", ""))).resolve()
            if (
                not manifest_space_path.is_file() or
                sha256_file(manifest_space_path) != manifest_space_sha256
            ):
                raise ValueError(
                    f"generation manifest source space is unavailable: {manifest_path}"
                )
            manifest_space = read_json(manifest_space_path)
            current_projection = {
                (batch, candidate_id): artifact_candidate_identity(value)
                for (candidate_family, batch, candidate_id), value in expected.items()
                if candidate_family == family
            }
            source_projection = {
                (batch, str(value["id"])): artifact_candidate_identity(value)
                for batch in sorted(space_batches(manifest_space))
                for value in candidate_map(manifest_space, family, batch).values()
                if value.get("requires_artifact")
            }
            if source_projection != current_projection:
                raise ValueError(
                    "generation manifest artifact candidate projection differs "
                    f"from the current search space: {manifest_path}"
                )
            space_binding.update({
                "kind": "exact_artifact_candidate_projection",
                "source_space": str(manifest_space_path),
                "reason": (
                    "all generated candidate parameters are identical; only "
                    "non-artifact search controls changed"
                ),
            })
        source_manifests.append({
            "path": str(manifest_path.resolve()),
            "sha256": sha256_file(manifest_path),
            "family": family,
            "entry_count": len(manifest.get("entries", [])),
            "space_binding": space_binding,
        })
        for item in manifest.get("entries", []):
            if not isinstance(item, dict):
                raise ValueError(f"non-object generation entry: {manifest_path}")
            key = (family, int(item.get("batch", -1)), str(item.get("candidate_id")))
            if key not in expected:
                continue
            if key in entries:
                raise ValueError(f"duplicate generated artifact: {key}")
            checked_files: dict[str, object] = {}
            for name in ("source", "object", "metadata"):
                path_text = item.get(name)
                recorded_hash = item.get(f"{name}_sha256")
                if not path_text or not recorded_hash:
                    raise ValueError(f"generated artifact lacks {name} evidence: {key}")
                path = pathlib.Path(str(path_text)).resolve()
                if not path.is_file() or sha256_file(path) != recorded_hash:
                    raise ValueError(f"generated artifact {name} hash mismatch: {key}")
                checked_files[name] = {"path": str(path), "sha256": recorded_hash}
            launch_symbol = str(item.get("launch_symbol", ""))
            if not launch_symbol or launch_symbol not in nm.stdout:
                raise ValueError(f"generated launcher is absent from linked binary: {key}")
            metadata = read_json(pathlib.Path(str(item["metadata"])).resolve())
            if item.get("space_sha256") != manifest_space_sha256:
                raise ValueError(f"generation entry search-space hash mismatch: {key}")
            if (
                metadata.get("space_sha256") != manifest_space_sha256
                or metadata.get("family") != family
                or int(metadata.get("batch", -1)) != key[1]
                or not isinstance(metadata.get("candidate"), dict)
                or artifact_candidate_identity(metadata["candidate"]) !=
                    artifact_candidate_identity(expected[key])
                or metadata.get("architecture") != space.get("architecture")
                or metadata.get("fixed_board") != [19, 19]
            ):
                raise ValueError(f"generated metadata does not match the search entry: {key}")
            generation_environment = metadata.get("generation_environment")
            if (
                not isinstance(generation_environment, dict)
                or generation_environment.get("compute_capability") != space.get("compute_capability")
            ):
                raise ValueError(f"generated artifact was not verified on the target architecture: {key}")
            compile_command = item.get("compile_command")
            expected_arch = nvcc_arch_flag(space.get("compute_capability"))
            if (
                not isinstance(compile_command, list)
                or expected_arch not in compile_command
            ):
                raise ValueError(
                    "generated artifact compile command does not target "
                    f"{space.get('architecture')}: {key}"
                )
            correctness = metadata.get("correctness_against_torch")
            if correctness is not None and not isinstance(correctness, dict):
                raise ValueError(f"generated artifact has malformed correctness evidence: {key}")
            entries[key] = {
                "family": family,
                "batch": key[1],
                "candidate_id": key[2],
                "status": "linked",
                "launch_symbol": launch_symbol,
                "source_sha256": item["source_sha256"],
                "object_sha256": item["object_sha256"],
                "metadata_sha256": item["metadata_sha256"],
                "compile_command": compile_command,
                "files": checked_files,
                "correctness": (
                    {"status": "passed", **correctness}
                    if isinstance(correctness, dict) else None
                ),
                "generation_environment": generation_environment,
                "generation_command": metadata.get("generation_command"),
            }
    missing = sorted(set(expected) - set(entries))
    if missing:
        preview = ", ".join(f"{f}/B{b}/{c}" for f, b, c in missing[:8])
        raise ValueError(
            f"generation manifests are missing {len(missing)} AOT entries: {preview}"
        )
    return {
        "schema": SCHEMA,
        "kind": ARTIFACT_BUNDLE_KIND,
        "generated_utc": utc_now(),
        "complete_history_coverage": True,
        "positive_history_closure": closure,
        "space": str(space_path.resolve()),
        "space_sha256": space_sha256,
        "architecture": space["architecture"],
        "gpu_class": space["gpu_class"],
        "linked_binary": str(binary.resolve()),
        "linked_binary_sha256": sha256_file(binary),
        "link_proof": "every generated extern-C launch symbol is present in nm -a output",
        "source_manifests": source_manifests,
        "entries": [entries[key] for key in sorted(entries)],
    }


def validate_artifact_bundle(
    bundle_path: pathlib.Path,
    *,
    space_path: pathlib.Path,
    space: dict[str, object],
    binary: pathlib.Path,
    required: Sequence[tuple[str, int, str]],
) -> tuple[dict[tuple[str, int, str], dict[str, object]], dict[str, object]]:
    """Verify auditable generation/link evidence for every selected AOT entry."""
    bundle = read_json(bundle_path)
    if bundle.get("schema") != SCHEMA or bundle.get("kind") != ARTIFACT_BUNDLE_KIND:
        raise ValueError("--artifact-bundle is not a CUDA tactic artifact bundle")
    if not bundle.get("complete_history_coverage", False):
        raise ValueError("artifact bundle is not a complete full-history generation")
    space_closure = space.get("positive_history_closure")
    bundle_closure = bundle.get("positive_history_closure")
    if (
        not isinstance(space_closure, dict) or
        not space_closure.get("complete") or
        not isinstance(bundle_closure, dict) or
        bundle_closure.get("contract_sha256") != space_closure.get("contract_sha256") or
        bundle_closure.get("record_ids") != space_closure.get("record_ids")
    ):
        raise ValueError("artifact bundle positive-history closure differs from --space")
    if bundle.get("space_sha256") != sha256_file(space_path):
        raise ValueError("artifact bundle search-space hash does not match --space")
    if bundle.get("architecture") != space.get("architecture"):
        raise ValueError("artifact bundle architecture does not match --space")
    if bundle.get("gpu_class") != space.get("gpu_class"):
        raise ValueError("artifact bundle GPU class does not match --space")
    binary_sha256 = sha256_file(binary)
    if bundle.get("linked_binary_sha256") != binary_sha256:
        raise ValueError("artifact bundle does not prove the selected binary link")
    entries = bundle.get("entries")
    if not isinstance(entries, list):
        raise ValueError("artifact bundle entries must be a list")
    by_key: dict[tuple[str, int, str], dict[str, object]] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("artifact bundle contains a non-object entry")
        key = (
            str(entry.get("family")),
            int(entry.get("batch", -1)),
            str(entry.get("candidate_id")),
        )
        if key in by_key:
            raise ValueError(f"duplicate artifact bundle entry: {key}")
        if entry.get("status") != "linked":
            raise ValueError(f"artifact bundle entry is not linked: {key}")
        if not entry.get("source_sha256") and not entry.get("object_sha256"):
            raise ValueError(f"artifact bundle entry has no source/object hash: {key}")
        by_key[key] = entry
    missing = sorted(set(required) - set(by_key))
    if missing:
        preview = ", ".join(f"{f}/B{b}/{c}" for f, b, c in missing[:8])
        raise ValueError(
            f"artifact bundle is missing {len(missing)} selected AOT entries: {preview}"
        )
    metadata = {
        "path": str(bundle_path.resolve()),
        "sha256": sha256_file(bundle_path),
        "linked_binary_sha256": binary_sha256,
        "required_entry_count": len(required),
    }
    return by_key, metadata


def space_batches(space: dict[str, object]) -> dict[int, dict[str, object]]:
    result: dict[int, dict[str, object]] = {}
    for item in space.get("batches", []):
        if not isinstance(item, dict):
            raise ValueError("search space contains a non-object batch")
        batch = int(item["batch"])
        if batch in result:
            raise ValueError(f"search space contains duplicate B{batch}")
        result[batch] = item
    return result


def candidate_map(space: dict[str, object], family: str, batch: int) -> dict[str, dict[str, object]]:
    if family not in space_families(space):
        raise ValueError(f"unsupported tactic family: {family}")
    batch_space = space_batches(space).get(batch)
    if batch_space is None:
        raise ValueError(f"search space has no B{batch}")
    values = batch_space.get(family, [])
    if not isinstance(values, list):
        raise ValueError(f"search space family {family}/B{batch} is not a list")
    result = {}
    for item in values:
        if not isinstance(item, dict) or not item.get("id"):
            raise ValueError(f"invalid candidate in {family}/B{batch}")
        result[str(item["id"])] = item
    return result


def restrict_space_to_plan(
    space: dict[str, object], plan: dict[str, object], batch: int,
) -> dict[str, object]:
    """Keep only one plan's selected generated tactics and dependencies."""
    validate_plan(plan, space=space, batches=[batch])
    restricted = json.loads(canonical_json(space))
    source_batches = space_batches(space)
    if batch not in source_batches:
        raise ValueError(f"search space has no B{batch}")
    plan_families = plan.get("families", {})
    if not isinstance(plan_families, dict):
        raise ValueError("plan has no family map")

    needed: dict[str, set[str]] = {
        family: set() for family in space_families(space)
    }
    pending: list[tuple[str, str]] = []
    for family in space_families(space):
        family_payload = plan_families.get(family)
        entries = (
            family_payload.get("batches", {})
            if isinstance(family_payload, dict) else {}
        )
        entry = entries.get(str(batch)) if isinstance(entries, dict) else None
        if not isinstance(entry, dict) or not entry.get("candidate_id"):
            raise ValueError(f"plan has no {family}/B{batch} candidate")
        candidate_id = str(entry["candidate_id"])
        current = candidate_map(space, family, batch).get(candidate_id)
        if current is None or current != entry.get("candidate"):
            raise ValueError(
                f"plan candidate differs from receiver space: {family}/B{batch}"
            )
        pending.append((family, candidate_id))

    while pending:
        family, candidate_id = pending.pop()
        if candidate_id in needed[family]:
            continue
        candidate = candidate_map(space, family, batch).get(candidate_id)
        if candidate is None:
            raise ValueError(
                f"plan artifact dependency is absent: {family}/B{batch}/{candidate_id}"
            )
        needed[family].add(candidate_id)
        dependencies = candidate.get("artifact_dependencies", [])
        if not isinstance(dependencies, list):
            raise ValueError(
                f"candidate has malformed artifact dependencies: {family}/{candidate_id}"
            )
        for dependency in dependencies:
            if not isinstance(dependency, dict):
                raise ValueError("artifact dependency is not an object")
            dependency_family = str(dependency.get("family", ""))
            dependency_id = str(dependency.get("candidate_id", ""))
            if dependency_family not in needed or not dependency_id:
                raise ValueError(f"invalid artifact dependency: {dependency}")
            pending.append((dependency_family, dependency_id))

    only_batch = space_batches(restricted)[batch]
    for family, candidate_ids in needed.items():
        values = only_batch.get(family, [])
        assert isinstance(values, list)
        only_batch[family] = [
            value for value in values
            if isinstance(value, dict) and str(value.get("id")) in candidate_ids
        ]
        if len(only_batch[family]) != len(candidate_ids):
            raise ValueError(
                f"failed to retain every plan candidate for {family}/B{batch}"
            )
    restricted["batches"] = [only_batch]
    candidate_policy = restricted.get("candidate_policy", {})
    if not isinstance(candidate_policy, dict):
        candidate_policy = {}
    candidate_policy.update({
        "build_only_plan_restricted": True,
        "build_only_plan_id": plan.get("plan_id"),
        "build_only_exact_batch": batch,
    })
    restricted["candidate_policy"] = candidate_policy
    return restricted


def candidate_config(family: str, value: dict[str, object]) -> dict[str, object]:
    config = value.get("config", value.get("config_overrides", {}))
    if config is None:
        return {}
    if not isinstance(config, dict):
        raise ValueError(f"candidate {value.get('id')} has a non-object config")
    return {str(key): item for key, item in config.items()}


def tactic_overrides(family: str, value: dict[str, object]) -> dict[str, object]:
    if family not in ALL_FAMILIES:
        raise ValueError(f"unsupported tactic family: {family}")
    return dict(candidate_config(family, value))


def validate_candidate_execution_contract(
    architecture: str,
    family: str,
    batch: int,
    value: dict[str, object],
) -> None:
    """Reject scanner entries that cannot close the runtime/plan loop."""
    candidate_id = str(value.get("id", ""))
    config = candidate_config(family, value)
    if tactic_overrides(family, value) != config:
        raise ValueError(
            f"plan apply loses config for {architecture}/{family}/B{batch}/"
            f"{candidate_id}"
        )
    supersedes = value.get("supersedes", [])
    if not isinstance(supersedes, list) or not all(
        isinstance(item, str) and item for item in supersedes
    ):
        raise ValueError(
            f"{architecture}/{family}/B{batch}/{candidate_id} has malformed "
            "supersedes metadata"
        )
    family_order = architecture_families(architecture)
    for superseded in supersedes:
        if superseded not in family_order or family_order.index(superseded) >= family_order.index(family):
            raise ValueError(
                f"{architecture}/{family}/B{batch}/{candidate_id} may only "
                f"supersede an earlier family, got {superseded}"
            )
    overrides_keys = value.get("overrides_keys", [])
    if not isinstance(overrides_keys, list) or not all(
        isinstance(item, str) and item for item in overrides_keys
    ) or len(set(overrides_keys)) != len(overrides_keys):
        raise ValueError(
            f"{architecture}/{family}/B{batch}/{candidate_id} has malformed "
            "overrides_keys metadata"
        )
    unknown_overrides = sorted(set(overrides_keys) - set(config))
    if unknown_overrides:
        raise ValueError(
            f"{architecture}/{family}/B{batch}/{candidate_id} declares config "
            f"keys it does not apply: {unknown_overrides}"
        )
    active = any(item is True for item in config.values()) or any(
        isinstance(item, str) and item not in {"", "disabled", "auto"}
        for item in config.values()
    ) or any(
        isinstance(item, int) and not isinstance(item, bool) and
        ((key == "cudaPlainQKVVariantSm89" and item > 0) or
         (key == "cudaRoPEBatchGroupSm89" and item > 1))
        for key, item in config.items()
    )
    if active and not activation_markers(value):
        raise ValueError(
            f"active candidate lacks runtime activation evidence: "
            f"{architecture}/{family}/B{batch}/{candidate_id}"
        )
    if value.get("requires_artifact") and not value.get("generator"):
        raise ValueError(
            f"AOT candidate lacks a generator mapping: "
            f"{architecture}/{family}/B{batch}/{candidate_id}"
        )


def candidate_compatibility(
    value: dict[str, object],
    selected: dict[str, dict[str, object]],
) -> tuple[bool, str | None]:
    """Check declarative cross-family requirements for one coordinate.

    Requirements use canonical family fields, for example
    ``{"fa4.supports_packed": true}``. An incompatible candidate is explicit
    scan evidence, not a silently omitted candidate and not a failed kernel.
    """
    requirements = value.get("requires", {})
    if not isinstance(requirements, dict):
        return False, "candidate.requires is not an object"
    for path, expected in requirements.items():
        if not isinstance(path, str) or "." not in path:
            return False, f"invalid requirement path: {path!r}"
        family, field = path.split(".", 1)
        current = selected.get(family)
        if current is None:
            return False, f"requirement refers to unselected family: {family}"
        actual = current.get(field)
        if actual != expected:
            return False, f"requires {path}={expected}, current={actual}"
    return True, None


def runtime_supersedes(
    family: str, value: dict[str, object],
) -> list[str]:
    """Return the candidate's explicit whole-boundary ownership."""
    supersedes = value.get("supersedes", [])
    if not isinstance(supersedes, list):
        raise ValueError(f"candidate {value.get('id')} has malformed supersedes")
    return list(dict.fromkeys(str(previous) for previous in supersedes))


def effective_candidate_map(
    selected: dict[str, dict[str, object]],
) -> tuple[dict[str, dict[str, object]], dict[str, str]]:
    """Resolve explicit whole-boundary bundles in architecture family order."""
    effective: dict[str, dict[str, object]] = {}
    superseded_by: dict[str, str] = {}
    for family, value in selected.items():
        for previous in runtime_supersedes(family, value):
            effective.pop(str(previous), None)
            superseded_by[str(previous)] = family
        effective[family] = value
    return effective, superseded_by


def resolve_candidate_config_state(
    selected: dict[str, dict[str, object]],
) -> tuple[
    dict[str, dict[str, object]], dict[str, str], dict[str, object],
    dict[str, dict[str, str]],
]:
    """Resolve bundles and explicit partial-key ownership in family order."""
    effective, superseded_by = effective_candidate_map(selected)
    applied: dict[str, object] = {}
    owners: dict[str, str] = {}
    overridden_by: dict[str, dict[str, str]] = {}
    for family, value in selected.items():
        supersedes = set(runtime_supersedes(family, value))
        overrides_keys = set(value.get("overrides_keys", []))
        for key, item in tactic_overrides(family, value).items():
            previous = owners.get(key)
            if previous is not None and previous != family:
                if previous not in supersedes and key not in overrides_keys:
                    raise ValueError(
                        "selected family configs have an undeclared ownership "
                        f"change: {previous}->{family}/{key}"
                    )
                overridden_by.setdefault(previous, {})[key] = family
            applied[key] = item
            owners[key] = family
    for family, value in effective.items():
        for key, expected_value in tactic_overrides(family, value).items():
            if applied.get(key) != expected_value:
                owner = overridden_by.get(family, {}).get(key)
                if owner is None:
                    raise ValueError(
                        "selected family configs conflict after plan apply: "
                        f"{family}/{key}={expected_value!r}, "
                        f"effective={applied.get(key)!r}"
                    )
    return effective, superseded_by, applied, overridden_by


def validate_cross_family_config_ownership(
    architecture: str,
    batch: int,
    batch_space: dict[str, object],
) -> None:
    """Require every cross-family config-key owner change to be declared."""
    exclusive_keys = {
        "sm89": {
            "cudaFlashAttentionTacticSm89": "fa4",
        },
        "sm120": {
            "cudaUseFlashAttentionSm120": "fa4",
            "cudaFlashAttentionSm120Accum": "fa4",
            "cudaFlashAttentionAotTacticSm120": "fa4",
        },
    }[architecture]
    prior_owners: dict[str, set[str]] = {}
    for family in architecture_families(architecture):
        values = batch_space.get(family)
        if not isinstance(values, list):
            raise ValueError(f"missing candidate list for {family}/B{batch}")
        for value in values:
            if not isinstance(value, dict):
                raise ValueError(f"malformed candidate for {family}/B{batch}")
            supersedes = set(value.get("supersedes", []))
            overrides_keys = set(value.get("overrides_keys", []))
            for key in candidate_config(family, value):
                exclusive_owner = exclusive_keys.get(key)
                if exclusive_owner is not None and family != exclusive_owner:
                    raise ValueError(
                        "exclusive tactic axis is owned by another family: "
                        f"{architecture}/{family}/B{batch}/{value.get('id')}/"
                        f"{key}, owner={exclusive_owner}"
                    )
                owners = prior_owners.get(key, set())
                if (
                    owners and key not in overrides_keys and
                    not owners.issubset(supersedes)
                ):
                    raise ValueError(
                        "cross-family config ownership is implicit: "
                        f"{architecture}/{family}/B{batch}/{value.get('id')}/"
                        f"{key}, earlier owners={sorted(owners)}"
                    )
            for key in overrides_keys:
                if not prior_owners.get(key):
                    raise ValueError(
                        "candidate declares a partial-key override without an "
                        f"earlier owner: {architecture}/{family}/B{batch}/"
                        f"{value.get('id')}/{key}"
                    )
        for value in values:
            assert isinstance(value, dict)
            for key in candidate_config(family, value):
                prior_owners.setdefault(key, set()).add(family)
    family_order = architecture_families(architecture)
    actual_cross_owners = {
        key: tuple(family for family in family_order if family in owners)
        for key, owners in prior_owners.items()
        if len(owners) > 1
    }
    expected_cross_owners = EXPECTED_CROSS_FAMILY_OWNERS[architecture]
    if actual_cross_owners != expected_cross_owners:
        missing = sorted(set(expected_cross_owners) - set(actual_cross_owners))
        unexpected = sorted(set(actual_cross_owners) - set(expected_cross_owners))
        changed = sorted(
            key for key in set(actual_cross_owners) & set(expected_cross_owners)
            if actual_cross_owners[key] != expected_cross_owners[key]
        )
        raise ValueError(
            "cross-family ownership contract changed; merge the axis or update "
            "the explicit joint-boundary contract: "
            f"missing={missing}, unexpected={unexpected}, changed={changed}"
        )
    group_index = {
        family: index
        for index, group in enumerate(architecture_decision_groups(architecture))
        for family in group
    }
    leaked = {
        key: owners
        for key, owners in actual_cross_owners.items()
        if len({group_index[family] for family in owners}) != 1
    }
    if leaked:
        raise ValueError(
            "runtime config ownership crosses decision groups: " +
            repr(leaked)
        )
    for family in architecture_families(architecture):
        values = batch_space[family]
        assert isinstance(values, list)
        for value in values:
            assert isinstance(value, dict)
            requirements = value.get("requires", {})
            if not isinstance(requirements, dict):
                continue
            for path in requirements:
                required_family = str(path).split(".", 1)[0]
                if (
                    required_family not in group_index or
                    group_index[required_family] != group_index[family]
                ):
                    raise ValueError(
                        "candidate dependency crosses decision groups: "
                        f"{architecture}/{family}/B{batch}/{value.get('id')}/"
                        f"{path}"
                    )


def activation_markers(value: dict[str, object]) -> list[str]:
    markers = value.get("activation_markers", [])
    if not isinstance(markers, list) or not all(
        isinstance(marker, str) and marker for marker in markers
    ):
        raise ValueError(
            f"candidate {value.get('id')} has malformed activation markers"
        )
    return markers


def effective_activation_markers(
    value: dict[str, object], overridden_keys: Iterable[str] = (),
) -> list[str]:
    """Drop only markers for config keys explicitly owned by a later family."""
    ignored = set(overridden_keys)
    marker_keys = value.get("activation_marker_keys", {})
    if not isinstance(marker_keys, dict):
        raise ValueError(
            f"candidate {value.get('id')} has malformed activation_marker_keys"
        )
    return [
        marker for marker in activation_markers(value)
        if not (
            any(key in marker for key in ignored) or
            bool(ignored & set(marker_keys.get(marker, [])))
        )
    ]


def require_activation_markers(
    value: dict[str, object], output: str,
    overridden_keys: Iterable[str] = (),
) -> None:
    missing = [
        marker for marker in effective_activation_markers(value, overridden_keys)
        if marker not in output
    ]
    if missing:
        raise RuntimeError(
            f"requested tactic {value.get('id')} did not acknowledge activation: "
            + "; ".join(missing)
        )


def topology_overrides(
    architecture: str,
    device: int,
    streams: int,
    space: dict[str, object] | None = None,
) -> dict[str, object]:
    values: dict[str, object] = {
        "numNNServerThreadsPerModel": streams,
    }
    for index in range(streams):
        values[f"cudaDeviceToUseThread{index}"] = device
    if space is not None:
        topology = space.get("topology", {})
        if isinstance(topology, dict):
            extra = topology.get("config_overrides", {})
            if isinstance(extra, dict):
                values.update(extra)
    if architecture not in ARCHITECTURES:
        raise ValueError(f"unsupported architecture: {architecture}")
    if architecture == "sm89":
        values["cudaSm89Backend"] = True
        values["cudaSm89Forward"] = True
    if architecture == "sm120":
        values["cudaSm120Backend"] = True
        values["cudaPersistingL2StreamsSm120"] = streams
    return values


def combined_overrides(
    space: dict[str, object],
    architecture: str,
    device: int,
    streams: int,
    family: str,
    value: dict[str, object],
    extra: str | None = None,
) -> dict[str, object]:
    result = runtime_tactic_baseline(architecture)
    result.update(parse_key_values(extra))
    result.update(topology_overrides(architecture, device, streams, space))
    result.update(tactic_overrides(family, value))
    return result


def result_metric(record: dict[str, object]) -> float:
    keys = (
        "combinedNNEvalsPerSec",
        "combined_nn_evals_per_sec",
        "nn_evals_per_sec",
        "nnEvalPerSec",
        "nnEval/s",
    )
    for key in keys:
        value = record.get(key)
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            return float(value)
    raise ValueError("benchmark JSON has no finite combined throughput metric")


def last_json_object(text: str) -> dict[str, object]:
    for line in reversed(text.splitlines()):
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            value = json.loads(line)
            if isinstance(value, dict):
                return value
    raise ValueError("benchmark output did not contain a JSON object")


def summarize_samples(
    samples: Iterable[float],
    *,
    iterations: int,
    warmup: int,
    max_relative_spread: float = DEFAULT_MAX_RELATIVE_SPREAD,
) -> dict[str, object]:
    values = [float(value) for value in samples]
    if not values or not all(math.isfinite(value) for value in values):
        raise ValueError("throughput samples must be non-empty finite numbers")
    median = statistics.median(values)
    relative_spread = (
        math.inf if median == 0 else (max(values) - min(values)) / abs(median)
    )
    long_enough = int(iterations) >= MIN_LONG_ITERATIONS
    enough_samples = len(values) >= MIN_STABLE_SAMPLES
    stable = long_enough and enough_samples and relative_spread <= max_relative_spread
    return {
        "nn_evals_per_sec_median": median,
        "nn_evals_per_sec_min": min(values),
        "nn_evals_per_sec_max": max(values),
        "nn_evals_per_sec_samples": values,
        "measurement_iterations": int(iterations),
        "measurement_warmup": int(warmup),
        "measurement_sample_count": len(values),
        "measurement_relative_spread": relative_spread,
        "measurement_max_relative_spread": max_relative_spread,
        "measurement_kind": "long_stable" if stable else (
            "long_unstable" if long_enough else "short_scan"
        ),
        "stable_long_nn_evals_per_sec": median if stable else None,
    }


def stable_metric(row: dict[str, object]) -> float | None:
    value = row.get("stable_long_nn_evals_per_sec")
    iterations = row.get("measurement_iterations", row.get("iterations"))
    sample_count = row.get("measurement_sample_count")
    kind = row.get("measurement_kind")
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        return None
    if not isinstance(iterations, (int, float)) or int(iterations) < MIN_LONG_ITERATIONS:
        return None
    if not isinstance(sample_count, (int, float)) or int(sample_count) < MIN_STABLE_SAMPLES:
        return None
    if kind != "long_stable":
        return None
    return float(value)


def choose_history_stage_winner(
    rows: Sequence[dict[str, object]],
    incumbent_candidate_id: str,
    metric: Any,
    min_improvement_fraction: float,
) -> tuple[dict[str, object], dict[str, object]]:
    """Choose a winner only after measuring the current accumulated state."""
    incumbents = [
        row for row in rows
        if row.get("candidate_id") == incumbent_candidate_id
    ]
    if len(incumbents) != 1:
        raise ValueError(
            "history coordinate must measure its incumbent exactly once: "
            f"{incumbent_candidate_id}"
        )
    incumbent = incumbents[0]
    incumbent_value = float(metric(incumbent))
    best = max(
        rows,
        key=lambda row: (
            float(metric(row)),
            row.get("candidate_id") == incumbent_candidate_id,
        ),
    )
    best_value = float(metric(best))
    required = incumbent_value * (1.0 + min_improvement_fraction)
    if best.get("candidate_id") != incumbent_candidate_id and best_value < required:
        best = incumbent
    return best, incumbent


def refinement_top_candidates(
    rows: Sequence[dict[str, object]],
    incumbent_candidate_id: str,
    limit: int,
) -> list[dict[str, object]]:
    """Return the first-pass top-K while always retaining the incumbent.

    The second discovery pass runs on the already improved whole graph.  Its
    candidate set must remain a deterministic projection of the first pass so
    a resume cannot silently change the search domain after some refined rows
    have been written.
    """
    if limit < 1:
        raise ValueError("refinement top-K must be positive")
    measured = [
        row for row in rows
        if row.get("status") == "measured" and
        isinstance(row.get("nn_evals_per_sec_median"), (int, float)) and
        math.isfinite(float(row["nn_evals_per_sec_median"]))
    ]
    # Once the incumbent is a concrete tactic, the empty keep row is not an
    # alternative implementation. Treating it as one silently restored the
    # runtime baseline during refinement (for example disabling the accepted
    # initial-global fusion). The concrete incumbent is forced below instead.
    if not incumbent_candidate_id.endswith("-keep-incumbent"):
        measured = [
            row for row in measured
            if not str(row.get("candidate_id", "")).endswith(
                "-keep-incumbent"
            )
        ]
    by_id = {str(row.get("candidate_id")): row for row in measured}
    if len(by_id) != len(measured):
        raise ValueError("first-pass refinement rows contain duplicate candidates")
    if incumbent_candidate_id not in by_id:
        raise ValueError(
            "refinement incumbent is absent from the first pass: "
            f"{incumbent_candidate_id}"
        )
    ranked = sorted(
        measured,
        key=lambda row: (
            -float(row["nn_evals_per_sec_median"]),
            str(row.get("candidate_id")),
        ),
    )
    selected = ranked[:limit]
    if incumbent_candidate_id not in {
        str(row.get("candidate_id")) for row in selected
    }:
        selected[-1] = by_id[incumbent_candidate_id]
    return selected


def canonical_refinement_rows(
    first_pass_rows: Sequence[dict[str, object]],
    refinement_rows: Sequence[dict[str, object]],
) -> list[dict[str, object]]:
    """Replace retested first-pass rows without emitting duplicate keys.

    The complete first pass is already retained as its own result file.  The
    refined result is the canonical input to the long gate and plan builder,
    so each family/batch/candidate key must describe exactly one measured
    graph state there.
    """
    refined_by_key: dict[tuple[str, int, str], dict[str, object]] = {}
    for row in refinement_rows:
        key = _row_key(str(row.get("family")), row)
        if key in refined_by_key:
            raise ValueError(f"duplicate refinement row: {key}")
        refined_by_key[key] = row
    result: list[dict[str, object]] = []
    first_keys: set[tuple[str, int, str]] = set()
    for row in first_pass_rows:
        key = _row_key(str(row.get("family")), row)
        if key in first_keys:
            raise ValueError(f"duplicate first-pass refinement row: {key}")
        first_keys.add(key)
        result.append(refined_by_key.get(key, row))
    unexpected = sorted(set(refined_by_key) - first_keys)
    if unexpected:
        raise ValueError(f"refinement rows are absent from first pass: {unexpected[:4]}")
    return result


def mark_superseded_refinement_winner(
    first_pass_rows: Sequence[dict[str, object]],
    refinement_rows: Sequence[dict[str, object]],
    *,
    family: str,
    batch: int,
    candidate_id: str,
    superseding_family: str,
    min_improvement_fraction: float,
) -> None:
    """Retain one catalog coordinate when another family owns its runtime keys."""
    marked = False
    for row in (*first_pass_rows, *refinement_rows):
        if row.get("family") != family or int(row.get("batch", -1)) != batch:
            continue
        is_winner = str(row.get("candidate_id")) == candidate_id
        row["history_stage_winner"] = is_winner
        row["history_final_joint"] = False
        if is_winner:
            row["history_superseded_by"] = superseding_family
            incumbent_id = row.get("history_incumbent_candidate_id")
            recorded_gain = row.get(
                "history_improvement_fraction_vs_incumbent"
            )
            if not isinstance(incumbent_id, str) or not isinstance(
                recorded_gain, (int, float)
            ):
                metric = row.get("nn_evals_per_sec_median")
                if not isinstance(metric, (int, float)):
                    raise ValueError(
                        "superseded catalog winner has no measured throughput: "
                        f"{family}/B{batch}/{candidate_id}"
                    )
                row["history_incumbent_candidate_id"] = candidate_id
                row["history_incumbent_nn_evals_per_sec"] = float(metric)
                row["history_selection_nn_evals_per_sec"] = float(metric)
                row["history_accepted_change"] = False
                row["history_min_improvement_fraction"] = (
                    min_improvement_fraction
                )
                row["history_improvement_fraction_vs_incumbent"] = 0.0
            marked = True
        else:
            row.pop("history_superseded_by", None)
    if not marked:
        raise ValueError(
            "refinement cannot retain the superseded catalog winner: "
            f"{family}/B{batch}/{candidate_id}"
        )


def refinement_sweep_limit_can_resume(
    previous_limit: object, requested_limit: int,
) -> bool:
    """Allow a completed/checkpointed refinement to grow, never to shrink."""
    return (
        type(previous_limit) is int and
        1 <= previous_limit <= requested_limit
    )


def require_stable_metric(row: dict[str, object]) -> float:
    value = stable_metric(row)
    if value is None:
        raise ValueError(
            "final plan/report requires measurement_kind=long_stable, "
            f"at least {MIN_LONG_ITERATIONS} iterations, and "
            f"at least {MIN_STABLE_SAMPLES} samples"
        )
    return value


def _run_capture(command: Sequence[str], cwd: pathlib.Path | None = None, timeout: int = 30) -> str | None:
    try:
        completed = subprocess.run(
            list(command), cwd=str(cwd) if cwd else None,
            text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            timeout=timeout, check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if completed.returncode != 0:
        return completed.stdout.strip() or None
    return completed.stdout.strip() or None


def _module_version(name: str) -> str | None:
    try:
        module = importlib.import_module(name)
    except Exception:  # optional package, including broken CUDA imports
        return None
    value = getattr(module, "__version__", None)
    return str(value) if value is not None else None


def _relevant_environment() -> dict[str, str]:
    prefixes = (
        "CUDA", "CUDNN", "CUTLASS", "TILELANG", "TRITON", "TORCH",
        "NVIDIA", "CMAKE", "CC", "CXX", "OMP", "CUDA_VISIBLE_DEVICES",
    )
    result: dict[str, str] = {}
    for key, value in os.environ.items():
        if not any(key == prefix or key.startswith(prefix + "_") for prefix in prefixes):
            continue
        if any(token in key.upper() for token in ("KEY", "TOKEN", "SECRET", "PASSWORD")):
            result[key] = "<redacted>"
        else:
            result[key] = value
    return dict(sorted(result.items()))


def _compile_metadata(
    repo: pathlib.Path, binary: pathlib.Path | None = None,
) -> dict[str, object]:
    build_dirs: list[pathlib.Path] = []
    if binary is not None:
        build_dirs.append(binary.resolve().parent)
    build_dirs.extend((repo / "build-cuda", repo / "build"))
    build_dirs = list(dict.fromkeys(build_dirs))
    candidates = [directory / "compile_commands.json" for directory in build_dirs]
    compile_path = next((path for path in candidates if path.is_file()), None)
    result: dict[str, object] = {}
    if compile_path is not None:
        result["compile_commands_path"] = str(compile_path.resolve())
        result["compile_commands_sha256"] = sha256_file(compile_path)
        try:
            commands = json.loads(compile_path.read_text(encoding="utf-8"))
            if isinstance(commands, list):
                result["compile_commands"] = commands
        except (OSError, json.JSONDecodeError):
            pass
    cache_candidates = [directory / "CMakeCache.txt" for directory in build_dirs]
    cache_path = next((path for path in cache_candidates if path.is_file()), None)
    if cache_path is not None:
        exact_keys = {
            "CMAKE_BUILD_TYPE", "CMAKE_CXX_COMPILER", "CMAKE_CXX_FLAGS",
            "CMAKE_CUDA_COMPILER", "CMAKE_CUDA_FLAGS", "CMAKE_CUDA_ARCHITECTURES",
            "CMAKE_GENERATOR", "CMAKE_TOOLCHAIN_FILE", "CUDAToolkit_ROOT",
            "CUTLASS_DIR", "CUDA_TOOLKIT_ROOT_DIR", "CUDNN_INCLUDE_DIR",
            "CUDNN_LIBRARY", "USE_BACKEND",
        }
        prefixes = (
            "CMAKE_CUDA_", "CMAKE_CXX_", "CUDA_", "CUDNN_", "SM89_", "SM120_",
        )
        cache: dict[str, str] = {}
        for line in cache_path.read_text(encoding="utf-8", errors="replace").splitlines():
            if not line or line.startswith("#") or "=" not in line:
                continue
            left, value = line.split("=", 1)
            key = left.split(":", 1)[0]
            if key in exact_keys or key.startswith(prefixes):
                cache[key] = value
        result["cmake_cache_path"] = str(cache_path.resolve())
        result["cmake_cache_sha256"] = sha256_file(cache_path)
        result["cmake_cache"] = cache
        third_party: dict[str, object] = {}
        flash_root = cache.get("SM89_FLASH_ATTN_ROOT")
        if flash_root:
            root = pathlib.Path(flash_root)
            third_party["flash_attention"] = {
                "path": str(root.resolve()),
                "git_revision": _run_capture(["git", "-C", str(root), "rev-parse", "HEAD"]),
                "git_status_short": _run_capture(["git", "-C", str(root), "status", "--short"]),
            }
        tilelang_root = cache.get("SM89_TACTIC_TILELANG_ROOT")
        if tilelang_root:
            root = pathlib.Path(tilelang_root)
            third_party["tilelang"] = {"path": str(root.resolve())}
            cutlass = root / "3rdparty" / "cutlass"
            if cutlass.is_dir():
                third_party["tilelang_cutlass"] = {"path": str(cutlass.resolve())}
        if third_party:
            result["third_party_sources"] = third_party
    return result


def collect_provenance(
    repo: pathlib.Path,
    *,
    binary: pathlib.Path | None = None,
    config: pathlib.Path | None = None,
    model: pathlib.Path | None = None,
    device: int | None = None,
    command: Sequence[str] | None = None,
) -> dict[str, object]:
    git_revision = _run_capture(["git", "rev-parse", "HEAD"], repo)
    git_status = _run_capture(["git", "status", "--short"], repo)
    git_diff_stat = _run_capture(["git", "diff", "--stat"], repo)
    git_submodules = _run_capture(["git", "submodule", "status", "--recursive"], repo)
    versions: dict[str, object] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "packages": {
            name: version for name in (
                "numpy", "torch", "cupy", "triton", "tilelang", "cutlass",
            ) if (version := _module_version(name)) is not None
        },
    }
    pip_freeze = _run_capture([sys.executable, "-m", "pip", "freeze"], timeout=60)
    if pip_freeze is not None:
        versions["pip_freeze"] = pip_freeze.splitlines()
    tools: dict[str, object] = {}
    for name, cmd in (
        ("nvidia_smi", ["nvidia-smi", "--query-gpu=index,name,driver_version,memory.total,compute_cap", "--format=csv,noheader"]),
        ("nvidia_smi_q", ["nvidia-smi", "-q"]),
        ("nvcc", ["nvcc", "--version"]),
        ("cudnn_ldconfig", ["bash", "-lc", "ldconfig -p 2>/dev/null | rg -i 'cudnn|cuda|cublas' || true"]),
        ("cudnn_packages", ["bash", "-lc", "dpkg-query -W 'libcudnn*' 2>/dev/null || true"]),
        ("cmake", ["cmake", "--version"]),
        ("cxx", ["c++", "--version"]),
    ):
        value = _run_capture(cmd, timeout=60)
        if value is not None:
            tools[name] = value[:200000]
    torch_info: dict[str, object] = {}
    try:
        torch = importlib.import_module("torch")
        torch_info["version"] = getattr(torch, "__version__", None)
        torch_info["cuda_version"] = getattr(getattr(torch, "version", None), "cuda", None)
        cuda = getattr(torch, "cuda", None)
        if cuda is not None:
            torch_info["cuda_available"] = bool(cuda.is_available())
            torch_info["device_count"] = int(cuda.device_count())
            backends = getattr(torch, "backends", None)
            cudnn = getattr(backends, "cudnn", None) if backends else None
            if cudnn is not None:
                torch_info["cudnn_version"] = cudnn.version()
                torch_info["cudnn_enabled"] = bool(cudnn.enabled)
    except Exception:
        pass
    result: dict[str, object] = {
        "schema": 1,
        "captured_utc": utc_now(),
        "repository": str(repo.resolve()),
        "git": {
            "revision": git_revision,
            "status_short": git_status or "",
            "diff_stat": git_diff_stat or "",
            "dirty": bool(git_status),
            "submodules": git_submodules or "",
        },
        "versions": versions,
        "torch": torch_info,
        "tools": tools,
        "environment": _relevant_environment(),
        "compile": _compile_metadata(repo, binary),
    }
    if binary is not None and binary.is_file():
        result["tools"]["binary_ldd"] = _run_capture(["ldd", str(binary)], timeout=60)
    if device is not None:
        result["cuda_device_ordinal"] = device
    if command is not None:
        result["command"] = list(command)
    files: dict[str, object] = {}
    for name, path in (("binary", binary), ("config", config), ("model", model)):
        if path is None:
            continue
        path = path.resolve()
        item: dict[str, object] = {"path": str(path), "exists": path.is_file()}
        if path.is_file():
            item["sha256"] = sha256_file(path)
            if name == "config":
                text_value = path.read_text(encoding="utf-8", errors="replace")
                if len(text_value) <= 1024 * 1024:
                    item["text"] = text_value
        files[name] = item
    result["files"] = files
    return result


def _space_identity(space: dict[str, object]) -> dict[str, object]:
    return {
        key: space.get(key) for key in (
            "architecture", "compute_capability", "gpu_class", "fixed_board",
            "precision", "families", "streams", "topology", "batches",
        )
    }


def _result_file_metadata(path: pathlib.Path, payload: dict[str, object]) -> dict[str, object]:
    return {
        "name": path.name,
        "sha256": sha256_file(path),
        "family": payload.get("family"),
        "rows": len(payload.get("rows", [])) if isinstance(payload.get("rows"), list) else 0,
        "finished_utc": payload.get("finished_utc"),
    }


def _portable_correctness_evidence(value: object) -> object:
    if not isinstance(value, dict):
        return value
    # The report content hashes make the evidence addressable. Its producer
    # path is scan-host metadata and must never become a receiver dependency.
    return {
        key: item for key, item in value.items()
        if key != "comparison"
    }


def absolute_paths_in_json(value: object, path: str = "$") -> list[str]:
    found: list[str] = []
    if isinstance(value, dict):
        for key, item in value.items():
            found.extend(absolute_paths_in_json(item, f"{path}.{key}"))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            found.extend(absolute_paths_in_json(item, f"{path}[{index}]"))
    elif isinstance(value, str) and pathlib.PurePosixPath(value).is_absolute():
        found.append(path)
    return found


def portable_plan_payload(payload: dict[str, object]) -> dict[str, object]:
    """Remove scan-host state and derive a location-independent plan ID."""
    plan = json.loads(canonical_json(payload))
    families = plan.get("families", {})
    if isinstance(families, dict):
        for family_entry in families.values():
            if not isinstance(family_entry, dict):
                continue
            family_entry.pop("space_path_at_scan", None)
            family_batches = family_entry.get("batches", {})
            if not isinstance(family_batches, dict):
                continue
            for batch_entry in family_batches.values():
                if not isinstance(batch_entry, dict):
                    continue
                batch_entry.pop("command", None)
                batch_entry.pop("source_result_path_at_scan", None)
                batch_entry["correctness"] = _portable_correctness_evidence(
                    batch_entry.get("correctness")
                )

    final_joint = plan.get("final_joint", {})
    if isinstance(final_joint, dict):
        for batch_entry in final_joint.values():
            if not isinstance(batch_entry, dict):
                continue
            batch_entry.pop("command", None)
            batch_entry["correctness"] = _portable_correctness_evidence(
                batch_entry.get("correctness")
            )

    source_results = plan.get("source_results", [])
    if isinstance(source_results, list):
        for result in source_results:
            if isinstance(result, dict):
                result.pop("path", None)

    plan["reproducibility"] = {
        "notes": [
            "Full commands and environment snapshots remain in the content-addressed scan results, not in this runtime plan.",
            "The receiving device must match architecture/GPU class and stream topology; producer device ordinal may change.",
        ],
    }

    batches = [int(item) for item in plan.get("batches", [])]
    family_identity: dict[str, object] = {}
    if isinstance(families, dict):
        for family, family_entry in families.items():
            if not isinstance(family_entry, dict):
                continue
            family_batches = family_entry.get("batches", {})
            family_identity[str(family)] = {
                "space_sha256": family_entry.get("space_sha256"),
                "selected": {
                    batch: (
                        family_batches.get(str(batch), {}).get("candidate_id")
                        if isinstance(family_batches, dict) and
                        isinstance(family_batches.get(str(batch)), dict)
                        else None
                    )
                    for batch in batches
                },
            }
    closure = plan.get("positive_history_closure", {})
    plan_identity = {
        "target": plan.get("target"),
        "positive_history_contract_sha256": (
            closure.get("contract_sha256")
            if isinstance(closure, dict) else None
        ),
        "batches": batches,
        "families": family_identity,
        "final_joint": final_joint,
    }
    plan_hash = sha256_bytes(canonical_json(plan_identity).encode("utf-8"))
    target = plan.get("target", {})
    if not isinstance(target, dict):
        raise ValueError("plan target is not an object")
    plan["plan_sha256"] = plan_hash
    plan["plan_id"] = (
        f"{target.get('architecture')}-{target.get('gpu_class')}-{plan_hash[:16]}"
    )
    absolute_paths = absolute_paths_in_json(plan)
    if absolute_paths:
        raise ValueError(
            "portable plan contains absolute paths: " +
            ", ".join(absolute_paths[:8])
        )
    return plan


def _row_key(family: str, row: dict[str, object]) -> tuple[str, int, str]:
    return family, int(row["batch"]), str(row["candidate_id"])


def _row_is_newer(row: dict[str, object], previous: dict[str, object]) -> bool:
    return str(row.get("finished_utc", "")) >= str(previous.get("finished_utc", ""))


def build_plan(
    result_paths: Sequence[pathlib.Path],
    space_path: pathlib.Path,
    families: Sequence[str],
    batches: Sequence[int],
    *,
    allow_partial: bool = False,
) -> dict[str, object]:
    space = read_json(space_path)
    if space.get("schema") != SCHEMA or space.get("kind") != SPACE_KIND:
        raise ValueError("plan requires a cuda-tactic-search-space schema-1 file")
    positive_history_closure = space.get("positive_history_closure")
    if (
        not isinstance(positive_history_closure, dict) or
        not positive_history_closure.get("complete")
    ):
        raise ValueError("search space lacks a complete positive-history closure")
    architecture = str(space.get("architecture"))
    gpu_class = str(space.get("gpu_class"))
    if architecture not in ARCHITECTURES:
        raise ValueError(f"unknown architecture in search space: {architecture}")
    validate_gpu_class(architecture, gpu_class)
    target_families = space_families(space)
    requested_set = set(families)
    if not requested_set or any(
        family not in target_families for family in requested_set
    ):
        raise ValueError(f"invalid tactic families: {list(families)}")
    requested_families = [
        family for family in target_families if family in requested_set
    ]
    required_families = list(target_families)
    unscanned_families = sorted(set(required_families) - set(requested_families))
    requested_batches = sorted(set(int(item) for item in batches))
    expected_streams = int(space.get("streams", -1))
    rows_by_key: dict[tuple[str, int, str], dict[str, object]] = {}
    result_metadata: list[dict[str, object]] = []
    model_hashes: set[str] = set()
    config_hashes: set[str] = set()
    target_devices: set[int] = set()
    cuda_capabilities_at_scan: dict[str, dict[str, object]] = {}
    for path in result_paths:
        payload = read_json(path)
        if payload.get("schema") != SCHEMA or payload.get("kind") != RESULT_KIND:
            raise ValueError(f"unsupported scan result: {path}")
        if payload.get("architecture") != architecture or payload.get("gpu_class") != gpu_class:
            raise ValueError(f"scan result target does not match search space: {path}")
        streams = int(payload.get("streams", -1))
        if streams != expected_streams:
            raise ValueError(f"scan result stream topology does not match search space: {path}")
        if payload.get("device_ordinal") is not None:
            target_devices.add(int(payload["device_ordinal"]))
        payload_capabilities = payload.get("cuda_device_capabilities", [])
        if isinstance(payload_capabilities, list):
            for capability in payload_capabilities:
                if isinstance(capability, dict):
                    cuda_capabilities_at_scan[canonical_json(capability)] = capability
        result_metadata.append(_result_file_metadata(path, payload))
        identity = payload.get("identity", {})
        if isinstance(identity, dict):
            if identity.get("model_sha256"):
                model_hashes.add(str(identity["model_sha256"]))
            if identity.get("config_sha256"):
                config_hashes.add(str(identity["config_sha256"]))
        payload_family = payload.get("family")
        # A multi-family scan records an empty top-level family.  Treat that
        # the same as null so its per-row family labels are still consumed.
        if payload_family not in (None, "") and payload_family not in requested_families:
            continue
        rows = payload.get("rows", [])
        if not isinstance(rows, list):
            raise ValueError(f"scan result rows are not a list: {path}")
        for row in rows:
            if not isinstance(row, dict) or "candidate_id" not in row:
                continue
            family = str(row.get("family", payload_family))
            if family not in requested_families:
                continue
            key = _row_key(family, row)
            if key[1] not in requested_batches:
                continue
            previous = rows_by_key.get(key)
            if previous is None or _row_is_newer(row, previous):
                item = dict(row)
                item["_source_result"] = str(path.resolve())
                rows_by_key[key] = item
    if not result_metadata:
        raise ValueError("no scan result files were supplied")
    batch_map = space_batches(space)
    selected_families: dict[str, dict[str, object]] = {}
    coverage: dict[str, dict[str, object]] = {}
    missing: list[dict[str, object]] = []
    for family in requested_families:
        family_batches: dict[str, object] = {}
        family_coverage: dict[str, object] = {}
        for batch in requested_batches:
            expected = candidate_map(space, family, batch)
            covered_rows = {
                candidate_id: rows_by_key[(family, batch, candidate_id)]
                for candidate_id in expected
                if (family, batch, candidate_id) in rows_by_key
            }
            observed = {
                candidate_id: row
                for candidate_id, row in covered_rows.items()
                if row.get("status") == "measured"
            }
            stable: list[tuple[float, str, dict[str, object]]] = []
            for candidate_id, row in observed.items():
                metric = stable_metric(row)
                if metric is not None:
                    stable.append((metric, candidate_id, row))
            stable.sort(key=lambda item: (-item[0], item[1]))
            missing_ids = sorted(set(expected) - set(covered_rows))
            invalid_status_ids = sorted(
                candidate_id for candidate_id, row in covered_rows.items()
                if row.get("status") != "measured"
            )
            history_winners = [
                (candidate_id, row) for candidate_id, row in observed.items()
                if row.get("history_stage_winner") is True
            ]
            history_evidence_error = None
            if len(history_winners) == 1:
                winner_id, winner_row = history_winners[0]
                incumbent_id = winner_row.get(
                    "history_incumbent_candidate_id"
                )
                accepted_change = winner_id != incumbent_id
                recorded_gain = winner_row.get(
                    "history_improvement_fraction_vs_incumbent"
                )
                minimum_gain = winner_row.get("history_min_improvement_fraction")
                if (
                    not isinstance(incumbent_id, str) or
                    incumbent_id not in expected or
                    winner_row.get("history_accepted_change") is not accepted_change or
                    not isinstance(recorded_gain, (int, float)) or
                    not isinstance(minimum_gain, (int, float)) or
                    (accepted_change and float(recorded_gain) < float(minimum_gain)) or
                    (not accepted_change and float(recorded_gain) != 0.0)
                ):
                    history_evidence_error = (
                        "winner lacks non-regressing measured-incumbent evidence"
                    )
            family_coverage[str(batch)] = {
                "expected_count": len(expected),
                "observed_count": len(observed),
                "stable_long_count": len(stable),
                "missing_candidate_ids": missing_ids,
                "invalid_status_candidate_ids": invalid_status_ids,
                "history_stage_winner_count": len(history_winners),
                "history_evidence_error": history_evidence_error,
            }
            history_error = (
                history_evidence_error
                if history_evidence_error is not None else
                None if len(history_winners) == 1 else
                f"expected one long-stable accumulated-history winner, got {len(history_winners)}"
            )
            if missing_ids or invalid_status_ids or not observed or history_error:
                missing.append({
                    "family": family,
                    "batch": batch,
                    "missing_candidate_ids": missing_ids,
                    "invalid_status_candidate_ids": invalid_status_ids,
                    "history_error": history_error,
                })
            if not observed or history_error:
                continue
            candidate_id, row = history_winners[0]
            stable_value = stable_metric(row)
            discovery_value = row.get("nn_evals_per_sec_median")
            selected_candidate = expected[candidate_id]
            recorded_candidate = row.get("candidate")
            if recorded_candidate is not None and recorded_candidate != selected_candidate:
                raise ValueError(f"candidate parameters differ from space for {family}/B{batch}/{candidate_id}")
            family_batches[str(batch)] = {
                "candidate_id": candidate_id,
                "candidate": selected_candidate,
                "implementation": row.get("implementation", selected_candidate.get("implementation")),
                "stable_long_nn_evals_per_sec": stable_value,
                "discovery_nn_evals_per_sec": discovery_value,
                "nn_evals_per_sec_samples": row.get("nn_evals_per_sec_samples", []),
                "measurement_iterations": row.get("measurement_iterations"),
                "measurement_warmup": row.get("measurement_warmup"),
                "measurement_sample_count": row.get("measurement_sample_count"),
                "measurement_kind": row.get("measurement_kind", "long_stable"),
                "measurement_relative_spread": row.get("measurement_relative_spread"),
                "history_base_overrides": row.get("history_base_overrides"),
                "history_accumulated_overrides": row.get("history_accumulated_overrides"),
                "correctness": _portable_correctness_evidence(
                    row.get("correctness")
                ),
                "binary_sha256": row.get("binary_sha256"),
                "source_result": pathlib.Path(str(row["_source_result"])).name,
            }
        selected_families[family] = {
            "space_sha256": sha256_file(space_path),
            "batches": family_batches,
        }
        coverage[family] = family_coverage
    for batch in requested_batches:
        selected_for_batch: dict[str, dict[str, object]] = {}
        for family in requested_families:
            entry = selected_families[family]["batches"].get(str(batch))
            if isinstance(entry, dict) and isinstance(entry.get("candidate"), dict):
                selected_for_batch[family] = entry["candidate"]
        (
            effective, superseded_by, _applied, overridden_by,
        ) = resolve_candidate_config_state(selected_for_batch)
        for family in requested_families:
            entry = selected_families[family]["batches"].get(str(batch))
            if not isinstance(entry, dict):
                continue
            entry["effective"] = family in effective
            entry["superseded_by"] = superseded_by.get(family)
            entry["overridden_keys"] = overridden_by.get(family, {})
    final_joint: dict[str, object] = {}
    for batch in requested_batches:
        joint_rows = [
            row for (family_name, row_batch, _), row in rows_by_key.items()
            if row_batch == batch and family_name in requested_families and
            row.get("history_final_joint") is True and stable_metric(row) is not None
        ]
        if len(joint_rows) != 1:
            missing.append({
                "family": "__final_joint__",
                "batch": batch,
                "missing_candidate_ids": [],
                "not_long_stable_candidate_ids": [],
                "history_error": (
                    f"expected one final long-stable joint row, got {len(joint_rows)}"
                ),
            })
            continue
        row = joint_rows[0]
        final_joint[str(batch)] = {
            "stable_long_nn_evals_per_sec": stable_metric(row),
            "nn_evals_per_sec_samples": row.get("nn_evals_per_sec_samples", []),
            "measurement_iterations": row.get("measurement_iterations"),
            "measurement_warmup": row.get("measurement_warmup"),
            "measurement_sample_count": row.get("measurement_sample_count"),
            "measurement_kind": row.get("measurement_kind"),
            "measurement_relative_spread": row.get("measurement_relative_spread"),
            "family": row.get("family"),
            "candidate_id": row.get("candidate_id"),
            "accumulated_overrides": row.get("history_accumulated_overrides"),
            "binary_sha256": row.get("binary_sha256"),
            "correctness": _portable_correctness_evidence(
                row.get("correctness")
            ),
            "source_result": pathlib.Path(str(row["_source_result"])).name,
        }
    if len(model_hashes) > 1 or len(config_hashes) > 1:
        raise ValueError("scan result files contain mixed model/config hashes")
    identity_missing: list[str] = []
    if not model_hashes:
        identity_missing.append("model_sha256")
    if not config_hashes:
        identity_missing.append("config_sha256")
    if unscanned_families:
        identity_missing.append(
            "unscanned_families=" + ",".join(unscanned_families)
        )
    ready = not missing and not identity_missing
    if not ready and not allow_partial:
        preview = ", ".join(f"{item['family']}/B{item['batch']}" for item in missing[:8])
        if identity_missing:
            preview = ", ".join([*identity_missing, preview] if preview else identity_missing)
        raise ValueError(f"scan coverage is incomplete; first gaps: {preview}")
    scan_devices = sorted(target_devices)
    preferred_device = int(space.get("device_ordinal", 0))
    target_device = (
        preferred_device if preferred_device in target_devices
        else (scan_devices[0] if scan_devices else preferred_device)
    )
    reported_compute_capabilities = {
        tuple(value)
        for capability in cuda_capabilities_at_scan.values()
        if (value := cuda_compute_capability(capability)) is not None
    }
    if len(reported_compute_capabilities) > 1:
        raise ValueError("scan results contain mixed CUDA compute capabilities")
    target_compute_capability = (
        list(next(iter(reported_compute_capabilities)))
        if reported_compute_capabilities else list(space["compute_capability"])
    )
    if target_compute_capability != space.get("compute_capability"):
        raise ValueError(
            "CUDA-reported scan capability does not match the search space"
        )
    target = {
        "architecture": architecture,
        "compute_capability": target_compute_capability,
        "gpu_class": gpu_class,
        "device_ordinal_at_scan": target_device,
        "device_ordinals_at_scan": scan_devices or [target_device],
        "fixed_board": space.get("fixed_board", [19, 19]),
        "precision": space.get("precision"),
        "streams": expected_streams,
        "model_sha256": next(iter(model_hashes), None),
        "config_sha256": next(iter(config_hashes), None),
        "cuda_device_capabilities_at_scan": [
            cuda_capabilities_at_scan[key] for key in sorted(cuda_capabilities_at_scan)
        ],
    }
    plan_identity = {
        "target": target,
        "positive_history_contract_sha256": positive_history_closure.get(
            "contract_sha256"
        ),
        "batches": requested_batches,
        "families": {
            family: {
                "space_sha256": selected_families[family]["space_sha256"],
                "selected": {
                    batch: selected_families[family]["batches"].get(str(batch), {}).get("candidate_id")
                    for batch in requested_batches
                },
            }
            for family in requested_families
        },
        "final_joint": final_joint,
    }
    plan_hash = sha256_bytes(canonical_json(plan_identity).encode("utf-8"))
    production_ready = ready and all(
        isinstance(entry, dict) and
        isinstance(entry.get("correctness"), dict) and
        entry["correctness"].get("status") == "passed"
        for entry in final_joint.values()
    )
    return portable_plan_payload({
        "schema": SCHEMA,
        "kind": PLAN_KIND,
        "plan_id": f"{architecture}-{gpu_class}-{plan_hash[:16]}",
        "plan_sha256": plan_hash,
        "generated_utc": utc_now(),
        "status": "complete_long_stable" if ready else "partial_or_unstable",
        "ready_for_scan_bypass": ready,
        "production_ready": production_ready,
        "positive_history_closure": positive_history_closure,
        "selection": {
            "metric": "stable long natural whole-graph combined nnEval/s",
            "method": "history-ordered accumulated coordinate winners; final joint long-stable row",
            "minimum_iterations": MIN_LONG_ITERATIONS,
            "minimum_samples": MIN_STABLE_SAMPLES,
            "maximum_relative_spread": DEFAULT_MAX_RELATIVE_SPREAD,
            "short_scan_values_are_never_final": True,
        },
        "target": target,
        "batches": requested_batches,
        "families": selected_families,
        "final_joint": final_joint,
        "coverage": coverage,
        "missing": missing,
        "identity_missing": identity_missing,
        "source_results": result_metadata,
        "reproducibility": {
            "notes": [
                "Full commands and environment snapshots remain in the content-addressed scan results, not in this runtime plan.",
                "The receiving device must match architecture/GPU class and stream topology; producer device ordinal may change.",
            ],
        },
        "apply": {
            "topology": topology_overrides(architecture, int(target_device or 0), expected_streams, space),
            "per_batch_tactic_overrides": {
                str(batch): render_plan_overrides(
                    selected_families, batch, architecture=architecture,
                    include_topology=False,
                )
                for batch in requested_batches
            },
        },
    })


def render_plan_overrides(
    families: dict[str, object], batch: int, *, architecture: str,
    include_topology: bool = False,
    topology: dict[str, object] | None = None,
) -> str:
    values = runtime_tactic_baseline(architecture)
    if include_topology and topology:
        values.update(topology)
    for family, family_payload in families.items():
        if family not in ALL_FAMILIES:
            raise ValueError(f"unsupported tactic family: {family}")
        if not isinstance(family_payload, dict):
            continue
        entries = family_payload.get("batches", {})
        if not isinstance(entries, dict):
            continue
        entry = entries.get(str(batch))
        if isinstance(entry, dict) and isinstance(entry.get("candidate"), dict):
            values.update(tactic_overrides(family, entry["candidate"]))
    return config_string(values)


def load_plan(path: pathlib.Path) -> dict[str, object]:
    payload = read_json(path)
    if payload.get("schema") != SCHEMA or payload.get("kind") != PLAN_KIND:
        raise ValueError(f"unsupported CUDA tactic plan: {path}")
    return payload


def validate_plan(
    plan: dict[str, object],
    *,
    space: dict[str, object] | None = None,
    space_path: pathlib.Path | None = None,
    model: pathlib.Path | None = None,
    config: pathlib.Path | None = None,
    architecture: str | None = None,
    gpu_class: str | None = None,
    streams: int | None = None,
    batches: Sequence[int] | None = None,
    families: Sequence[str] | None = None,
    device_properties: dict[str, object] | None = None,
) -> dict[str, object]:
    if plan.get("schema") != SCHEMA or plan.get("kind") != PLAN_KIND:
        raise ValueError("unsupported CUDA tactic plan")
    plan_closure = plan.get("positive_history_closure")
    if not isinstance(plan_closure, dict) or not plan_closure.get("complete"):
        raise ValueError("plan lacks a complete positive-history closure")
    target = plan.get("target", {})
    if not isinstance(target, dict):
        raise ValueError("plan has no target")
    plan_arch = str(target.get("architecture"))
    plan_gpu = str(target.get("gpu_class"))
    if plan_arch not in ARCHITECTURES:
        raise ValueError(f"plan has unknown architecture: {plan_arch}")
    validate_gpu_class(plan_arch, plan_gpu)
    target_families = architecture_families(plan_arch)
    requested_set = set(families or target_families)
    if any(family not in target_families for family in requested_set):
        raise ValueError(f"unsupported tactic families: {sorted(requested_set)}")
    requested_families = tuple(
        family for family in target_families if family in requested_set
    )
    if architecture and architecture != plan_arch:
        raise ValueError(f"plan architecture mismatch: {plan_arch} != {architecture}")
    if gpu_class and gpu_class != plan_gpu:
        raise ValueError(f"plan GPU class mismatch: {plan_gpu} != {gpu_class}")
    if streams is not None and int(target.get("streams", -1)) != streams:
        raise ValueError("plan stream topology mismatch")
    if target.get("compute_capability") != ARCHITECTURES[plan_arch]["compute_capability"]:
        raise ValueError("plan compute capability is inconsistent with its architecture")
    if device_properties is not None:
        producer_devices = target.get("cuda_device_capabilities_at_scan", [])
        producer = (
            producer_devices[0]
            if isinstance(producer_devices, list) and producer_devices else None
        )
        if not isinstance(producer, dict):
            raise ValueError("plan producer device identity is unavailable")
        producer_identity = cuda_plan_device_identity(producer)
        receiver_identity = cuda_plan_device_identity(device_properties)
        for field in CUDA_PLAN_DEVICE_IDENTITY_FIELDS:
            expected = producer_identity[field]
            actual = receiver_identity[field]
            if actual != expected:
                raise ValueError(
                    f"CUDA receiver {field} does not match the plan: "
                    f"{actual!r} != {expected!r}"
                )
    if target.get("fixed_board") != [19, 19]:
        raise ValueError("CUDA tactic plans currently require 19x19")
    if not plan.get("ready_for_scan_bypass", False):
        raise ValueError("plan is partial/unstable and cannot bypass the scan")
    if model is not None and target.get("model_sha256"):
        if sha256_file(model.resolve()) != target["model_sha256"]:
            raise ValueError("plan model SHA-256 does not match receiver model")
    if config is not None and target.get("config_sha256"):
        if sha256_file(config.resolve()) != target["config_sha256"]:
            raise ValueError("plan config SHA-256 does not match receiver config")
    if space is None and space_path is not None:
        space = read_json(space_path)
    if space is not None:
        if space.get("architecture") != plan_arch or space.get("gpu_class") != plan_gpu:
            raise ValueError("plan target does not match receiver search space")
        if int(space.get("streams", -1)) != int(target.get("streams", -2)):
            raise ValueError("plan and search-space stream topology differ")
        space_closure = space.get("positive_history_closure")
        plan_closure = plan.get("positive_history_closure")
        if (
            not isinstance(space_closure, dict) or
            not space_closure.get("complete") or
            not isinstance(plan_closure, dict) or
            plan_closure.get("contract_sha256") !=
                space_closure.get("contract_sha256") or
            plan_closure.get("record_ids") != space_closure.get("record_ids")
        ):
            raise ValueError("plan positive-history closure differs from search space")
        if space_path is not None:
            expected_sha = sha256_file(space_path.resolve())
            for family in requested_families:
                family_payload = plan.get("families", {}).get(family, {})
                if isinstance(family_payload, dict) and family_payload.get("space_sha256") not in (None, expected_sha):
                    raise ValueError(f"plan search-space hash mismatch for {family}")
    selected_batches = sorted(set(int(item) for item in (batches or plan.get("batches", []))))
    checked: dict[str, object] = {}
    for family in requested_families:
        if family not in target_families:
            raise ValueError(f"unsupported tactic family: {family}")
        family_payload = plan.get("families", {}).get(family)
        if not isinstance(family_payload, dict):
            raise ValueError(f"plan has no family {family}")
        entries = family_payload.get("batches", {})
        if not isinstance(entries, dict):
            raise ValueError(f"plan family {family} has no batch map")
        for batch in selected_batches:
            entry = entries.get(str(batch))
            if not isinstance(entry, dict) or not entry.get("candidate_id"):
                raise ValueError(f"plan has no {family}/B{batch} entry")
            if space is not None:
                current = candidate_map(space, family, batch).get(str(entry["candidate_id"]))
                if current is None:
                    raise ValueError(f"plan tactic is absent from receiver space: {family}/B{batch}")
                if entry.get("candidate") != current:
                    raise ValueError(f"plan candidate parameters differ from receiver space: {family}/B{batch}")
        checked[family] = selected_batches
    if requested_families == target_families:
        apply_payload = plan.get("apply", {})
        per_batch_apply = (
            apply_payload.get("per_batch_tactic_overrides", {})
            if isinstance(apply_payload, dict) else {}
        )
        family_payloads = plan.get("families", {})
        if not isinstance(per_batch_apply, dict) or not isinstance(
            family_payloads, dict
        ):
            raise ValueError("plan has malformed apply metadata")
        for batch in selected_batches:
            selected_for_batch: dict[str, dict[str, object]] = {}
            for family in target_families:
                family_payload = family_payloads[family]
                assert isinstance(family_payload, dict)
                entry = family_payload["batches"][str(batch)]
                assert isinstance(entry, dict)
                candidate_value = entry["candidate"]
                assert isinstance(candidate_value, dict)
                selected_for_batch[family] = candidate_value
            (
                effective, superseded_by, applied, overridden_by,
            ) = resolve_candidate_config_state(selected_for_batch)
            for family in target_families:
                entry = family_payloads[family]["batches"][str(batch)]
                assert isinstance(entry, dict)
                if entry.get("effective") is not (family in effective):
                    raise ValueError(
                        f"plan effective-family metadata differs at {family}/B{batch}"
                    )
                if entry.get("superseded_by") != superseded_by.get(family):
                    raise ValueError(
                        f"plan supersession metadata differs at {family}/B{batch}"
                    )
                if entry.get("overridden_keys") != overridden_by.get(family, {}):
                    raise ValueError(
                        f"plan key-ownership metadata differs at {family}/B{batch}"
                    )
            expected_values = runtime_tactic_baseline(plan_arch)
            expected_values.update(applied)
            expected_apply = config_string(expected_values)
            if per_batch_apply.get(str(batch)) != expected_apply:
                raise ValueError(
                    f"plan apply mapping differs from selected tactics at B{batch}"
                )
    final_joint = plan.get("final_joint")
    if not isinstance(final_joint, dict):
        raise ValueError("plan has no final joint long-gate results")
    final_metrics: dict[str, float] = {}
    for batch in selected_batches:
        entry = final_joint.get(str(batch))
        if not isinstance(entry, dict):
            raise ValueError(f"plan has no final joint B{batch} result")
        final_metrics[str(batch)] = require_stable_metric(entry)
    warnings = [
        "recorded driver/CUDA/cuDNN/package versions are compatibility evidence, not exact-match requirements",
    ]
    if not plan.get("production_ready", False):
        warnings.append(
            "production_ready is false: selected candidates still need an explicit correctness.status=passed record"
        )
    return {
        "valid": True,
        "architecture": plan_arch,
        "gpu_class": plan_gpu,
        "streams": int(target["streams"]),
        "batches": selected_batches,
        "families": checked,
        "final_joint_nn_evals_per_sec": final_metrics,
        "production_ready": bool(plan.get("production_ready", False)),
        "warnings": warnings,
    }


def _parse_benchmark_record(stdout: str) -> dict[str, object]:
    try:
        return last_json_object(stdout)
    except (ValueError, json.JSONDecodeError):
        matches = re.findall(r"combined throughput:\s*([0-9.eE+\-]+)\s+nnEval/s", stdout)
        if not matches:
            matches = re.findall(r"([0-9.eE+\-]+)\s+nnEval/s", stdout)
        if not matches:
            raise ValueError("benchmark output has no combined nnEval/s metric")
        return {"combinedNNEvalsPerSec": float(matches[-1])}


def _timeout_text(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def _active_sm_pids(device: int) -> set[int]:
    """Return PIDs with non-zero SM activity in one pmon sample."""
    try:
        sample = subprocess.run(
            ["nvidia-smi", "pmon", "-i", str(device), "-c", "1", "-s", "u"],
            text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            check=False, timeout=3,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError(f"unable to sample GPU SM occupancy: {exc}") from exc
    if sample.returncode != 0:
        detail = sample.stderr.strip() or sample.stdout.strip()
        raise RuntimeError(f"nvidia-smi pmon failed: {detail}")
    active: set[int] = set()
    for line in sample.stdout.splitlines():
        fields = line.split()
        if len(fields) < 4 or not fields[0].isdigit() or not fields[1].isdigit():
            continue
        try:
            sm = float(fields[3])
        except ValueError:
            continue
        pid = int(fields[1])
        # pmon can retain one final utilization sample after a short-lived
        # benchmark process has exited.  It is not competing work once the
        # process no longer exists, and treating it as such can abort the next
        # candidate in the same scan.
        if sm > 0.0 and pathlib.Path(f"/proc/{pid}").exists():
            active.add(pid)
    return active


class _GpuOccupancyMonitor:
    def __init__(self, device: int, process: subprocess.Popen[str]):
        self.device = device
        self.process = process
        self.process_start_time = _process_start_time(process.pid)
        self.process_group = os.getpgid(process.pid)
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.foreign_pids: set[int] = set()
        self.samples = 0
        self.error: str | None = None

    def start(self) -> None:
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        self.thread.join(timeout=4)

    def _run(self) -> None:
        while not self.stop_event.is_set():
            try:
                active = _active_sm_pids(self.device)
                self.samples += 1
                for pid in active:
                    same_group = pid == self.process.pid
                    if not same_group:
                        try:
                            same_group = os.getpgid(pid) == self.process_group
                        except ProcessLookupError:
                            same_group = False
                    if not same_group and not _is_recent_completed_benchmark(pid):
                        self.foreign_pids.add(pid)
                if self.foreign_pids:
                    os.killpg(self.process_group, signal.SIGTERM)
                    return
            except Exception as exc:  # fail closed for an unobservable GPU
                self.error = str(exc)
                try:
                    os.killpg(self.process_group, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                return
            self.stop_event.wait(0.25)

    def evidence(self) -> dict[str, object]:
        return {
            "samples": self.samples,
            "foreign_active_sm_pids": sorted(self.foreign_pids),
            "error": self.error,
        }


_COMPLETED_BENCHMARKS_LOCK = threading.Lock()
_COMPLETED_BENCHMARKS: dict[int, tuple[str | None, float]] = {}


def _process_start_time(pid: int) -> str | None:
    try:
        # Field 22 is the kernel start time. Splitting after the final ') '
        # avoids spaces and parentheses in the process comm field.
        tail = pathlib.Path(f"/proc/{pid}/stat").read_text().rpartition(") ")[2]
        return tail.split()[19]
    except (OSError, IndexError):
        return None


def _remember_completed_benchmark(pid: int, start_time: str | None) -> None:
    now = time.monotonic()
    with _COMPLETED_BENCHMARKS_LOCK:
        expired = [
            old_pid for old_pid, (_, deadline) in _COMPLETED_BENCHMARKS.items()
            if deadline <= now
        ]
        for old_pid in expired:
            _COMPLETED_BENCHMARKS.pop(old_pid, None)
        _COMPLETED_BENCHMARKS[pid] = (start_time, now + 10.0)


def _is_recent_completed_benchmark(pid: int) -> bool:
    with _COMPLETED_BENCHMARKS_LOCK:
        identity = _COMPLETED_BENCHMARKS.get(pid)
    if identity is None or identity[1] <= time.monotonic():
        return False
    return _process_start_time(pid) == identity[0]


def _wait_for_gpu_sm_idle(
    device: int, *, retry_seconds: float = SM_CONFLICT_RETRY_SECONDS,
) -> list[dict[str, object]]:
    conflicts: list[dict[str, object]] = []
    while True:
        active = {
            pid for pid in _active_sm_pids(device)
            if not _is_recent_completed_benchmark(pid)
        }
        if active:
            # Confirm once quickly because pmon can retain a final sample from
            # a process that has just exited. Persistent work then uses a
            # deliberately low-frequency retry loop.
            time.sleep(0.5)
            active = {
                pid for pid in _active_sm_pids(device)
                if not _is_recent_completed_benchmark(pid)
            }
        if not active:
            return conflicts
        event = {
            "detected_utc": utc_now(),
            "foreign_active_sm_pids": sorted(active),
            "retry_seconds": retry_seconds,
        }
        conflicts.append(event)
        print(
            "[cuda-tactic] GPU " + str(device) +
            " has external SM work from PID(s) " +
            ",".join(str(pid) for pid in sorted(active)) +
            f"; retrying in {retry_seconds:g}s",
            file=sys.stderr, flush=True,
        )
        time.sleep(retry_seconds)


def _run_benchmark_once_with_occupancy(
    command: Sequence[str], *, device: int, timeout: int,
) -> tuple[subprocess.CompletedProcess[str], bool, dict[str, object]]:
    try:
        process = subprocess.Popen(
            list(command), text=True, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, start_new_session=True,
        )
    except OSError as exc:
        raise RuntimeError(f"unable to start benchmark: {exc}") from exc
    monitor = _GpuOccupancyMonitor(device, process)
    monitor.start()
    timed_out = False
    try:
        stdout, stderr = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        timed_out = True
        try:
            os.killpg(monitor.process_group, signal.SIGTERM)
        except ProcessLookupError:
            pass
        stdout, stderr = process.communicate()
        stdout = _timeout_text(stdout or exc.stdout)
        stderr = _timeout_text(stderr or exc.stderr)
    finally:
        monitor.stop()
        _remember_completed_benchmark(process.pid, monitor.process_start_time)
    evidence = monitor.evidence()
    if monitor.error:
        stderr = (stderr or "") + "\nGPU occupancy monitor: " + monitor.error
    if monitor.foreign_pids:
        stderr = (stderr or "") + "\nGPU occupancy monitor detected external SM work"
    return subprocess.CompletedProcess(
        command, process.returncode, stdout or "", stderr or "",
    ), timed_out, evidence


def _run_benchmark_with_occupancy(
    command: Sequence[str], *, device: int, timeout: int,
) -> tuple[subprocess.CompletedProcess[str], bool, dict[str, object]]:
    conflicts: list[dict[str, object]] = []
    while True:
        conflicts.extend(_wait_for_gpu_sm_idle(device))
        completed, timed_out, evidence = _run_benchmark_once_with_occupancy(
            command, device=device, timeout=timeout,
        )
        foreign = evidence.get("foreign_active_sm_pids", [])
        if not foreign:
            evidence["sm_conflict_retries"] = conflicts
            return completed, timed_out, evidence
        event = {
            "detected_utc": utc_now(),
            "foreign_active_sm_pids": list(foreign),
            "retry_seconds": SM_CONFLICT_RETRY_SECONDS,
            "during_benchmark": True,
        }
        conflicts.append(event)
        print(
            "[cuda-tactic] external SM work interrupted the benchmark on GPU " +
            str(device) + "; discarding the measurement and retrying in " +
            f"{SM_CONFLICT_RETRY_SECONDS:g}s",
            file=sys.stderr, flush=True,
        )
        time.sleep(SM_CONFLICT_RETRY_SECONDS)


def scan_command(
    space: dict[str, object],
    architecture: str,
    device: int,
    streams: int,
    family: str,
    batch: int,
    value: dict[str, object],
    *,
    binary: str,
    config: str,
    model: str,
    iterations: int,
    warmup: int,
    extra_override: str | None,
    runner: Sequence[str],
) -> tuple[list[str], dict[str, object]]:
    overrides = combined_overrides(
        space, architecture, device, streams, family, value, extra_override
    )
    command = list(runner) + [
        binary, "benchmarknn",
        "-config", config,
        "-override-config", config_string(overrides),
        "-model", model,
        "-iterations", str(iterations),
        "-warmup", str(warmup),
        "-batch-size", str(batch),
        "-boardsize", "19",
        "-json",
    ]
    return command, overrides


def official_fallback_overrides(
    architecture: str, device: int, streams: int, batch: int,
) -> dict[str, object]:
    """Return an exact-batch topology with both custom backends disabled."""
    if architecture not in ARCHITECTURES:
        raise ValueError(f"unsupported architecture: {architecture}")
    values: dict[str, object] = {
        "cudaSm89Backend": False,
        "cudaSm89Forward": False,
        "cudaSm120Backend": False,
        "cudaDisableWarmup": True,
        "cudaWarmupOnlyMaxBatchSize": True,
        "nnMaxBatchSize": batch,
        "numNNServerThreadsPerModel": streams,
    }
    for index in range(streams):
        values[f"cudaDeviceToUseThread{index}"] = device
    return values


def stable_optimized_prescan_state(
    architecture: str, gpu_class: str, device: int, streams: int, batch: int,
) -> tuple[dict[str, object], dict[str, str], list[str]]:
    """Materialize the self-contained, artifact-free batch-ranking graph."""
    requested = stable_prescan_candidate_ids(architecture, gpu_class, batch)
    unknown = sorted(set(requested) - set(architecture_families(architecture)))
    if unknown:
        raise ValueError(f"pre-scan baseline has unknown families: {unknown}")
    selected: dict[str, dict[str, object]] = {}
    for family in architecture_families(architecture):
        candidate_id = requested.get(family)
        if candidate_id is None:
            continue
        choices = {
            str(value["id"]): value
            for value in default_candidates(
                architecture, family, batch, gpu_class,
            )
        }
        if candidate_id not in choices:
            raise ValueError(
                f"stable pre-scan candidate is absent from B{batch}: "
                f"{family}/{candidate_id}"
            )
        value = choices[candidate_id]
        dependencies = value.get("artifact_dependencies", [])
        if value.get("requires_artifact") or dependencies:
            raise ValueError(
                "stable pre-scan must not depend on exact-batch artifacts: "
                f"{family}/{candidate_id}"
            )
        if candidate_id.endswith("-keep-incumbent"):
            raise ValueError(
                f"stable pre-scan is not self-contained: {family}/{candidate_id}"
            )
        selected[family] = value

    effective, _, applied, overridden_by = resolve_candidate_config_state(selected)
    markers: list[str] = []
    for family, value in effective.items():
        markers.extend(effective_activation_markers(
            value, overridden_by.get(family, {}),
        ))
    markers = list(dict.fromkeys(markers))

    values = runtime_tactic_baseline(architecture)
    values.update(applied)
    values.update(topology_overrides(architecture, device, streams))
    values.update({
        "cudaSm89Backend": architecture == "sm89",
        "cudaSm89Forward": architecture == "sm89",
        "cudaSm120Backend": architecture == "sm120",
        "cudaDisableWarmup": True,
        "cudaWarmupOnlyMaxBatchSize": True,
        "nnMaxBatchSize": batch,
    })
    return values, {
        family: str(value["id"]) for family, value in selected.items()
    }, markers


def run_stable_optimized_batch_prescan(args: argparse.Namespace) -> None:
    """Rank exact batches on an explicit, stable optimized CUDA graph."""
    architecture = canonical_architecture(args.architecture, args.gpu_class)
    gpu_class = str(args.gpu_class)
    validate_gpu_class(architecture, gpu_class)
    if args.streams < 1:
        raise ValueError("--streams must be positive")
    if args.iterations < MIN_DISCOVERY_ITERATIONS:
        raise ValueError(
            f"optimized prescan requires at least {MIN_DISCOVERY_ITERATIONS} iterations"
        )
    if args.warmup < MIN_DISCOVERY_WARMUP:
        raise ValueError(
            f"optimized prescan requires at least {MIN_DISCOVERY_WARMUP} warmups"
        )
    if args.repeats < MIN_STABLE_SAMPLES:
        raise ValueError(
            f"optimized prescan requires at least {MIN_STABLE_SAMPLES} repeats"
        )
    if args.top_batches < 1:
        raise ValueError("--top-batches must be positive")
    batches = parse_int_set(args.batches)
    if args.top_batches > len(batches):
        raise ValueError("--top-batches exceeds the prescan batch domain")

    binary = pathlib.Path(args.binary).resolve()
    config = pathlib.Path(args.config).resolve()
    model = pathlib.Path(args.model).resolve()
    for path, label in ((binary, "binary"), (config, "config"), (model, "model")):
        if not path.is_file():
            raise ValueError(f"{label} does not exist: {path}")
    try:
        from portable_cuda_device import query_cuda_device
    except ModuleNotFoundError:
        from python.portable_cuda_device import query_cuda_device
    device_properties = query_cuda_device(args.device)
    expected_cc = ARCHITECTURES[architecture]["compute_capability"]
    if cuda_compute_capability(device_properties) != expected_cc:
        raise ValueError("CUDA prescan device capability does not match architecture")

    output = pathlib.Path(args.output).resolve()
    raw_dir = pathlib.Path(args.raw_dir).resolve()
    raw_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for batch in batches:
        overrides, baseline_candidates, activation_proof = (
            stable_optimized_prescan_state(
                architecture, gpu_class, args.device, args.streams, batch,
            )
        )
        command = [
            str(binary), "benchmarknn", "-config", str(config),
            "-override-config", config_string(overrides),
            "-model", str(model), "-iterations", str(args.iterations),
            "-warmup", str(args.warmup), "-batch-size", str(batch),
            "-boardsize", "19", "-json",
        ]
        samples: list[float] = []
        runs: list[dict[str, object]] = []
        for repeat in range(args.repeats):
            completed = None
            occupancy: dict[str, object] = {}
            attempts: list[dict[str, object]] = []
            for attempt in range(args.max_attempts):
                completed, timed_out, occupancy = _run_benchmark_with_occupancy(
                    command, device=args.device, timeout=args.timeout_seconds,
                )
                stem = f"optimized-baseline-b{batch}-r{repeat}-a{attempt}"
                stdout_path = raw_dir / f"{stem}.out"
                stderr_path = raw_dir / f"{stem}.err"
                stdout_path.write_text(completed.stdout, encoding="utf-8")
                stderr_path.write_text(completed.stderr, encoding="utf-8")
                attempts.append({
                    "attempt": attempt, "returncode": completed.returncode,
                    "timed_out": timed_out, "stdout": str(stdout_path),
                    "stderr": str(stderr_path),
                })
                foreign = occupancy.get("foreign_active_sm_pids", [])
                monitor_error = occupancy.get("error")
                if (
                    completed.returncode == 0 and not foreign and
                    not monitor_error
                ):
                    break
            assert completed is not None
            if completed.returncode != 0:
                raise RuntimeError(
                    f"optimized baseline benchmark failed for B{batch}; "
                    f"see {attempts[-1]['stderr']}"
                )
            combined_output = completed.stdout + "\n" + completed.stderr
            missing_markers = [
                marker for marker in activation_proof
                if marker not in combined_output
            ]
            if missing_markers:
                raise RuntimeError(
                    f"optimized baseline silently fell back for B{batch}: "
                    + "; ".join(missing_markers)
                )
            record = _parse_benchmark_record(completed.stdout)
            throughput = result_metric(record)
            samples.append(throughput)
            runs.append({
                "repeat": repeat, "throughput": throughput,
                "benchmark": record, "attempts": attempts,
                "gpu_occupancy": occupancy,
                "activation_markers": activation_proof,
            })
        summary = summarize_samples(
            samples, iterations=args.iterations, warmup=args.warmup,
            max_relative_spread=args.max_relative_spread,
        )
        spread = summary.get("measurement_relative_spread")
        if not isinstance(spread, (int, float)) or spread > args.max_relative_spread:
            raise RuntimeError(
                f"optimized baseline B{batch} is unstable: relative spread={spread}"
            )
        row = {
            "batch": batch, "status": "measured", "command": command,
            "overrides": overrides, "runs": runs, **summary,
            "baseline_candidates": baseline_candidates,
        }
        rows.append(row)
        print(
            f"optimized baseline B{batch}: "
            f"{row['nn_evals_per_sec_median']:.3f} nnEval/s",
            flush=True,
        )
    ranked = sorted(
        rows,
        key=lambda row: (-float(row["nn_evals_per_sec_median"]), int(row["batch"])),
    )
    selected = sorted(int(row["batch"]) for row in ranked[:args.top_batches])
    payload = {
        "schema": SCHEMA,
        "kind": BASELINE_SCAN_KIND,
        "created_utc": utc_now(),
        "architecture": architecture,
        "gpu_class": gpu_class,
        "compute_capability": expected_cc,
        "device_ordinal": args.device,
        "streams": args.streams,
        "requested_batches": batches,
        "top_batch_count": args.top_batches,
        "baseline": {
            "policy": "artifact-free certified optimized graph",
            "candidate_ids_by_batch": {
                str(batch): stable_prescan_candidate_ids(
                    architecture, gpu_class, batch,
                )
                for batch in batches
            },
            "official_fallback_is_selector": False,
        },
        "measurement_request": {
            "iterations": args.iterations,
            "warmup": args.warmup,
            "repeats": args.repeats,
        },
        "selected_batches": selected,
        "ranking": [int(row["batch"]) for row in ranked],
        "rows": rows,
        "identity": {
            "binary_sha256": sha256_file(binary),
            "config_sha256": sha256_file(config),
            "model_sha256": sha256_file(model),
        },
        "device": device_properties,
    }
    write_json(output, payload)
    print(json.dumps({"output": str(output), "selected_batches": selected}))


def run_scan(args: argparse.Namespace) -> None:
    space_path = pathlib.Path(args.space).resolve()
    space = read_json(space_path)
    if space.get("schema") != SCHEMA or space.get("kind") != SPACE_KIND:
        raise ValueError("scan requires a cuda-tactic-search-space file")
    architecture = str(space["architecture"])
    if args.architecture and args.architecture != architecture:
        raise ValueError("--architecture does not match the search space")
    gpu_class = str(space["gpu_class"])
    target_families = space_families(space)
    device = int(args.device if args.device is not None else space.get("device_ordinal", 0))
    streams = int(space["streams"])
    device_properties = None
    if not args.dry_run:
        try:
            from portable_cuda_device import query_cuda_device
        except ModuleNotFoundError:
            from python.portable_cuda_device import query_cuda_device
        device_properties = query_cuda_device(device)
        if cuda_compute_capability(device_properties) != space.get("compute_capability"):
            raise ValueError(
                "CUDA-reported scan device capability does not match the search space"
            )
    if args.streams is not None and int(args.streams) != streams:
        raise ValueError("--streams does not match the search space")
    batches = parse_int_set(args.batches) if args.batches else sorted(space_batches(space))
    families = [item.strip() for item in args.families.split(",") if item.strip()]
    if not families:
        families = list(target_families)
    if any(item not in target_families for item in families):
        raise ValueError(f"invalid families: {families}")
    if args.phase == "long" and (
        args.iterations < MIN_LONG_ITERATIONS or args.repeats < MIN_STABLE_SAMPLES
    ):
        raise ValueError(
            "long scan phase requires at least "
            f"{MIN_LONG_ITERATIONS} iterations and {MIN_STABLE_SAMPLES} repeats"
        )
    if (
        args.phase == "discovery" and not args.dry_run and
        (args.iterations < MIN_DISCOVERY_ITERATIONS or args.warmup < MIN_DISCOVERY_WARMUP)
    ):
        raise ValueError(
            "discovery requires at least "
            f"{MIN_DISCOVERY_ITERATIONS} iterations and "
            f"{MIN_DISCOVERY_WARMUP} warmups to stabilize GPU clocks"
        )
    if args.max_attempts < 1:
        raise ValueError("--max-attempts must be positive")
    if not 0.0 <= args.min_improvement_fraction < 1.0:
        raise ValueError("--min-improvement-fraction must be in [0,1)")
    if args.timeout_seconds <= 0:
        raise ValueError("--timeout-seconds must be positive")
    for batch in batches:
        if batch not in space_batches(space):
            raise ValueError(f"B{batch} is absent from the search space")
    artifact_candidates = [
        (family, batch, value["id"])
        for family in families
        for batch in batches
        for value in candidate_map(space, family, batch).values()
        if value.get("requires_artifact")
    ]
    binary = pathlib.Path(args.binary).resolve()
    config = pathlib.Path(args.config).resolve()
    model = pathlib.Path(args.model).resolve()
    model_identity = pathlib.Path(
        args.model_identity if args.model_identity else args.model
    ).resolve()
    if not args.dry_run:
        for path, label in (
            (binary, "binary"), (config, "config"), (model, "model"),
            (model_identity, "model identity"),
        ):
            if not path.is_file():
                raise ValueError(f"{label} does not exist: {path}")
    artifact_evidence: dict[tuple[str, int, str], dict[str, object]] = {}
    artifact_bundle_metadata: dict[str, object] | None = None
    if artifact_candidates and not args.dry_run:
        if not args.artifact_bundle:
            preview = ", ".join(
                f"{family}/B{batch}/{candidate_id}"
                for family, batch, candidate_id in artifact_candidates[:4]
            )
            raise ValueError(
                "AOT candidates require --artifact-bundle with complete generation "
                f"and linked-binary evidence (first candidates: {preview})"
            )
        artifact_evidence, artifact_bundle_metadata = validate_artifact_bundle(
            pathlib.Path(args.artifact_bundle).resolve(),
            space_path=space_path, space=space, binary=binary,
            required=artifact_candidates,
        )
    current_binary_sha256 = sha256_file(binary) if binary.is_file() else None
    current_config_sha256 = sha256_file(config) if config.is_file() else None
    current_execution_model_sha256 = sha256_file(model) if model.is_file() else None
    current_identity_model_sha256 = (
        sha256_file(model_identity) if model_identity.is_file() else None
    )
    output = pathlib.Path(args.output).resolve()
    raw_dir = pathlib.Path(args.raw_dir).resolve() if args.raw_dir else output.parent / f"{output.stem}-raw"
    runner = shlex.split(args.runner) if args.runner else []
    rows: list[dict[str, object]] = []
    started = utc_now()
    implementation_identity = workflow_implementation_identity()
    resumed_compatible = False
    if args.resume and output.is_file():
        previous = read_json(output)
        previous_identity = previous.get("identity", {})
        if (
            previous.get("space_sha256") == sha256_file(space_path) and
            previous.get("implementation_identity") == implementation_identity and
            isinstance(previous_identity, dict) and
            previous_identity.get("config_sha256") == current_config_sha256 and
            previous_identity.get("execution_model_sha256") == current_execution_model_sha256 and
            previous_identity.get("model_sha256") == current_identity_model_sha256
        ):
            rows = [row for row in previous.get("rows", []) if isinstance(row, dict)]
            resumed_compatible = True
    provenance = collect_provenance(
        pathlib.Path(__file__).resolve().parents[1], binary=binary, config=config, model=model,
        device=device,
    ) if not args.dry_run else {"schema": 1, "captured_utc": utc_now(), "dry_run": True}
    raw_dir.mkdir(parents=True, exist_ok=True)
    persisted_rows = (
        json.dumps(rows, sort_keys=True, separators=(",", ":"))
        if resumed_compatible else None
    )

    def checkpoint_if_changed() -> None:
        nonlocal persisted_rows
        serialized = json.dumps(rows, sort_keys=True, separators=(",", ":"))
        if serialized == persisted_rows:
            return
        _write_scan_payload(
            output, space_path, space, architecture, gpu_class, device,
            streams, args, started, provenance, artifact_bundle_metadata,
            rows, device_properties, implementation_identity,
        )
        persisted_rows = serialized

    for batch in batches:
        # Seed coordinate search from the accepted configuration file. The
        # previous all-off reset destroyed interactions between already-
        # accepted history stages and made the entire curve regress. Every
        # family still scans its explicit off control and all real variants;
        # after the final family the accumulated overrides are self-contained.
        accumulated = runtime_tactic_baseline(architecture)
        accumulated.update(parse_key_values(args.override_config))
        # Make every exact-batch implementation build for the batch currently
        # being scanned.
        accumulated["nnMaxBatchSize"] = batch
        # Each subprocess measures one exact batch. Compiling lazy SDPA graphs
        # for 1..B on every candidate only adds setup time; the target-B graph
        # is still compiled before benchmarknn's own warmup/timed passes.
        accumulated["cudaWarmupOnlyMaxBatchSize"] = True
        accumulated["cudaDisableWarmup"] = True
        selected_candidates: dict[str, dict[str, object]] = {}
        for family_index, family in enumerate(families):
            base_overrides = config_string(accumulated)
            stage_rows: list[dict[str, object]] = []
            for value in candidate_map(space, family, batch).values():
                key = (family, batch, str(value["id"]))
                compatible, incompatibility = candidate_compatibility(
                    value, selected_candidates,
                )
                if not compatible:
                    raise ValueError(
                        "search-space candidate has an unresolved runtime "
                        f"dependency for {family}/B{batch}/{value['id']}: "
                        f"{incompatibility}; encode the dependency in the "
                        "candidate config instead of declaring unsupported"
                    )
                command, overrides = scan_command(
                    space, architecture, device, streams, family, batch, value,
                    binary=str(binary), config=str(config), model=str(model),
                    iterations=args.iterations, warmup=args.warmup,
                    extra_override=base_overrides, runner=runner,
                )
                previous = next((
                    row for row in rows
                    if row.get("status") == "measured" and
                    str(row.get("family")) == family and
                    int(row.get("batch", -1)) == batch and
                    str(row.get("candidate_id")) == str(value["id"]) and
                    row.get("candidate") == value and
                    row.get("history_base_overrides") == base_overrides and
                    row.get("overrides") == overrides and
                    row.get("command") == command and
                    row.get("binary_sha256") == current_binary_sha256 and
                    row.get("config_sha256") == current_config_sha256 and
                    int(row.get("measurement_iterations", -1)) == args.iterations and
                    int(row.get("measurement_warmup", -1)) == args.warmup and
                    int(row.get("measurement_sample_count", -1)) == args.repeats
                ), None)
                if previous is not None:
                    stage_rows.append(previous)
                    continue
                if args.dry_run:
                    row = {
                        "family": family, "batch": batch, "candidate_id": value["id"],
                        "candidate": value, "implementation": value.get("implementation"),
                        "status": "planned", "command": command, "overrides": overrides,
                        "history_family_index": family_index,
                        "history_base_overrides": base_overrides,
                        "config_sha256": current_config_sha256,
                        "finished_utc": utc_now(),
                    }
                    rows.append(row)
                    stage_rows.append(row)
                    continue
                samples: list[float] = []
                run_records: list[dict[str, object]] = []
                for repeat in range(args.repeats):
                    attempt_records: list[dict[str, object]] = []
                    completed = None
                    stdout_path = None
                    stderr_path = None
                    occupancy_evidence: dict[str, object] = {}
                    for attempt in range(args.max_attempts):
                        completed, timed_out, occupancy_evidence = (
                            _run_benchmark_with_occupancy(
                                command, device=device, timeout=args.timeout_seconds,
                            )
                        )
                        stem = re.sub(
                            r"[^A-Za-z0-9_.-]+", "_",
                            f"{family}-b{batch}-{value['id']}-r{repeat}-a{attempt}",
                        )
                        stdout_path = raw_dir / f"{stem}.out"
                        stderr_path = raw_dir / f"{stem}.err"
                        stdout_path.write_text(completed.stdout, encoding="utf-8")
                        stderr_path.write_text(completed.stderr, encoding="utf-8")
                        attempt_records.append({
                            "attempt": attempt,
                            "returncode": completed.returncode,
                            "timed_out": timed_out,
                            "stdout": str(stdout_path),
                            "stderr": str(stderr_path),
                            "gpu_occupancy": occupancy_evidence,
                        })
                        if completed.returncode == 0:
                            break
                    assert completed is not None and stdout_path is not None and stderr_path is not None
                    if completed.returncode != 0:
                        row = {
                            "family": family, "batch": batch, "candidate_id": value["id"],
                            "candidate": value, "status": "failed", "command": command,
                            "overrides": overrides,
                            "history_family_index": family_index,
                            "history_base_overrides": base_overrides,
                            "returncode": completed.returncode,
                            "attempts": attempt_records,
                            "binary_sha256": current_binary_sha256,
                            "config_sha256": current_config_sha256,
                            "finished_utc": utc_now(),
                        }
                        rows.append(row)
                        checkpoint_if_changed()
                        raise RuntimeError(
                            f"benchmark failed for {family}/B{batch}/{value['id']} "
                            f"after {args.max_attempts} attempts; see {stderr_path}"
                        )
                    require_activation_markers(
                        value, completed.stdout + "\n" + completed.stderr,
                    )
                    record = _parse_benchmark_record(completed.stdout)
                    throughput = result_metric(record)
                    samples.append(throughput)
                    run_records.append({
                        "repeat": repeat, "throughput": throughput,
                        "benchmark": record, "stdout": str(stdout_path),
                        "stderr": str(stderr_path), "attempts": attempt_records,
                        "gpu_occupancy": occupancy_evidence,
                    })
                row = {
                    "family": family, "batch": batch, "candidate_id": value["id"],
                    "candidate": value, "implementation": value.get("implementation"),
                    "status": "measured", "command": command, "overrides": overrides,
                    "history_family_index": family_index,
                    "history_base_overrides": base_overrides,
                    "finished_utc": utc_now(),
                    "binary_sha256": current_binary_sha256,
                    "config_sha256": current_config_sha256,
                    "correctness": artifact_evidence.get(key, {}).get(
                        "correctness", value.get("correctness")
                    ),
                    "artifact_evidence": artifact_evidence.get(key),
                    "runs": run_records,
                    **summarize_samples(
                        samples, iterations=args.iterations, warmup=args.warmup,
                        max_relative_spread=args.max_relative_spread,
                    ),
                }
                rows = [
                    old for old in rows
                    if not (old.get("family") == family and int(old.get("batch", -1)) == batch and old.get("candidate_id") == value["id"])
                ]
                rows.append(row)
                stage_rows.append(row)
                # A single exact batch can still contain hundreds of
                # candidates. Persist every completed measurement so an
                # external-SM abort or host interruption resumes at the next
                # candidate rather than repeating the entire batch.
                checkpoint_if_changed()
                metric = row.get("stable_long_nn_evals_per_sec")
                print(f"{family} B{batch} {value['id']}: {metric if metric is not None else row['nn_evals_per_sec_median']:.3f} nnEval/s ({row['measurement_kind']})", flush=True)
            if args.dry_run:
                # A dry-run plans commands only. Choose the first entry solely
                # to make later-stage command contexts deterministic.
                winner = stage_rows[0]
            else:
                def stage_metric(row: dict[str, object]) -> float:
                    if args.phase == "long":
                        metric = stable_metric(row)
                        if metric is None:
                            raise ValueError(
                                f"history stage is not long-stable: {family}/B{batch}/"
                                f"{row.get('candidate_id')}"
                            )
                        return metric
                    metric = row.get("nn_evals_per_sec_median")
                    if not isinstance(metric, (int, float)) or not math.isfinite(float(metric)):
                        raise ValueError(
                            f"history stage has no discovery metric: {family}/B{batch}/"
                            f"{row.get('candidate_id')}"
                        )
                    return float(metric)

                incumbent_id = f"{family}-keep-incumbent"
                winner, incumbent = choose_history_stage_winner(
                    stage_rows, incumbent_id, stage_metric,
                    args.min_improvement_fraction,
                )
                winner["history_incumbent_candidate_id"] = incumbent_id
                winner["history_incumbent_nn_evals_per_sec"] = stage_metric(incumbent)
                winner["history_accepted_change"] = (
                    winner.get("candidate_id") != incumbent_id
                )
                winner["history_min_improvement_fraction"] = (
                    args.min_improvement_fraction
                )
                winner["history_improvement_fraction_vs_incumbent"] = (
                    stage_metric(winner) / stage_metric(incumbent) - 1.0
                )
            for row in stage_rows:
                row["history_stage_winner"] = row is winner
                row["history_final_joint"] = (
                    row is winner and family_index + 1 == len(families)
                )
            winner_candidate = winner.get("candidate")
            if not isinstance(winner_candidate, dict):
                raise ValueError(f"history stage winner has no candidate: {family}/B{batch}")
            selected_candidates[family] = winner_candidate
            accumulated.update(tactic_overrides(family, winner_candidate))
            winner["history_accumulated_overrides"] = config_string(accumulated)
        # Atomic batch-level checkpoint. On an unexpected interruption only
        # the current batch is repeated; explicit candidate failures still
        # checkpoint immediately above with their logs and return code.
        checkpoint_if_changed()
    checkpoint_if_changed()
    print(json.dumps({"output": str(output), "rows": len(rows), "dry_run": args.dry_run}))


def run_refine(args: argparse.Namespace) -> None:
    """Retest each family's first-pass top-K on the improved whole graph."""
    if args.top_k < 1:
        raise ValueError("--top-k must be positive")
    if args.iterations < MIN_DISCOVERY_ITERATIONS:
        raise ValueError(
            f"refinement requires at least {MIN_DISCOVERY_ITERATIONS} iterations"
        )
    if args.warmup < MIN_DISCOVERY_WARMUP:
        raise ValueError(
            f"refinement requires at least {MIN_DISCOVERY_WARMUP} warmups"
        )
    if args.repeats < 1 or args.max_attempts < 1:
        raise ValueError("refinement repeats and max attempts must be positive")
    if args.max_sweeps < 1:
        raise ValueError("--max-sweeps must be positive")
    if args.resweep_top_k < 1 or args.resweep_top_k > args.top_k:
        raise ValueError("--resweep-top-k must be in [1,--top-k]")
    if args.confirmation_iterations < MIN_DISCOVERY_ITERATIONS:
        raise ValueError(
            f"--confirmation-iterations must be at least {MIN_DISCOVERY_ITERATIONS}"
        )
    if not 0.0 <= args.min_improvement_fraction < 1.0:
        raise ValueError("--min-improvement-fraction must be in [0,1)")

    space_path = pathlib.Path(args.space).resolve()
    discovery_path = pathlib.Path(args.discovery).resolve()
    space = read_json(space_path)
    discovery = read_json(discovery_path)
    if space.get("schema") != SCHEMA or space.get("kind") != SPACE_KIND:
        raise ValueError("refine requires a cuda-tactic-search-space file")
    if discovery.get("kind") != RESULT_KIND:
        raise ValueError("refine input is not a scan result")
    if discovery.get("space_sha256") != sha256_file(space_path):
        raise ValueError("refine input does not match --space")
    architecture = str(space["architecture"])
    gpu_class = str(space["gpu_class"])
    families = space_families(space)
    streams = int(space["streams"])
    device = int(args.device if args.device is not None else space.get("device_ordinal", 0))
    batches = parse_int_set(args.batches) if args.batches else sorted(space_batches(space))
    for batch in batches:
        if batch not in space_batches(space):
            raise ValueError(f"B{batch} is absent from the search space")

    binary = pathlib.Path(args.binary).resolve()
    config = pathlib.Path(args.config).resolve()
    model = pathlib.Path(args.model).resolve()
    model_identity = pathlib.Path(
        args.model_identity if args.model_identity else args.model
    ).resolve()
    for path, label in (
        (binary, "binary"), (config, "config"), (model, "model"),
        (model_identity, "model identity"),
    ):
        if not path.is_file():
            raise ValueError(f"refine {label} does not exist: {path}")
    expected_identity = {
        "model_sha256": sha256_file(model_identity),
        "execution_model_sha256": sha256_file(model),
        "config_sha256": sha256_file(config),
    }
    discovery_identity = discovery.get("identity", {})
    if not isinstance(discovery_identity, dict) or any(
        discovery_identity.get(key) != value
        for key, value in expected_identity.items()
    ):
        raise ValueError("refine binary inputs differ from first-pass discovery")
    binary_sha256 = sha256_file(binary)
    first_pass_rows = [
        row for row in discovery.get("rows", [])
        if isinstance(row, dict) and row.get("status") == "measured"
    ]
    if not first_pass_rows or any(
        row.get("binary_sha256") != binary_sha256 for row in first_pass_rows
    ):
        raise ValueError("refine binary differs from first-pass discovery")

    try:
        from portable_cuda_device import query_cuda_device
    except ModuleNotFoundError:
        from python.portable_cuda_device import query_cuda_device
    device_properties = query_cuda_device(device)
    if cuda_compute_capability(device_properties) != space.get("compute_capability"):
        raise ValueError("refine device capability does not match the search space")

    artifact_required = [
        (family, batch, candidate_id)
        for batch in batches
        for family in families
        for candidate_id, value in candidate_map(space, family, batch).items()
        if value.get("requires_artifact")
    ]
    artifact_evidence: dict[tuple[str, int, str], dict[str, object]] = {}
    if artifact_required:
        if not args.artifact_bundle:
            raise ValueError("refinement of AOT candidates requires --artifact-bundle")
        artifact_evidence, _ = validate_artifact_bundle(
            pathlib.Path(args.artifact_bundle).resolve(),
            space_path=space_path, space=space, binary=binary,
            required=artifact_required,
        )

    source_sha256 = sha256_file(discovery_path)
    output = pathlib.Path(args.output).resolve()
    raw_dir = pathlib.Path(args.raw_dir).resolve() if args.raw_dir else (
        output.parent / f"{output.stem}-raw"
    )
    raw_dir.mkdir(parents=True, exist_ok=True)
    runner = shlex.split(args.runner) if args.runner else []
    refinement_rows: list[dict[str, object]] = []
    if args.resume and output.is_file():
        previous = read_json(output)
        metadata = previous.get("refinement", {})
        if (
            isinstance(metadata, dict) and
            metadata.get("source_discovery_sha256") == source_sha256 and
            metadata.get("top_k") == args.top_k and
            metadata.get("iterations") == args.iterations and
            metadata.get("warmup") == args.warmup and
            metadata.get("repeats") == args.repeats and
            refinement_sweep_limit_can_resume(
                metadata.get("max_sweeps"), args.max_sweeps,
            ) and
            metadata.get("resweep_top_k") == args.resweep_top_k and
            # A checkpoint written before adaptive ABBA was added still has
            # valid broad-scan rows. Reuse those exact command matches and add
            # confirmation evidence only at the decision boundary.
            metadata.get("confirmation_iterations") in (
                None, args.confirmation_iterations,
            ) and
            previous.get("space_sha256") == sha256_file(space_path)
        ):
            refinement_rows = [
                row for row in previous.get("rows", [])
                if isinstance(row, dict) and row.get("refinement_pass") == 2
            ]

    base_rows: list[dict[str, object]] = []
    for original in first_pass_rows:
        row = dict(original)
        row["first_pass_history_stage_winner"] = bool(
            row.get("history_stage_winner")
        )
        row["first_pass_history_final_joint"] = bool(
            row.get("history_final_joint")
        )
        row["history_stage_winner"] = False
        row["history_final_joint"] = False
        row.pop("history_accumulated_overrides", None)
        row["refinement_pass"] = 1
        base_rows.append(row)

    started = utc_now()
    scan_parameters = discovery.get("scan_parameters", {})
    source_override_config = (
        str(scan_parameters.get("override_config", ""))
        if isinstance(scan_parameters, dict) else ""
    )

    def write_checkpoint(complete: bool) -> None:
        payload = dict(discovery)
        payload["started_utc"] = started
        payload["finished_utc"] = utc_now()
        payload["rows"] = canonical_refinement_rows(
            base_rows, refinement_rows,
        )
        payload["refinement"] = {
            "schema": 1,
            "complete": complete,
            "source_discovery": str(discovery_path),
            "source_discovery_sha256": source_sha256,
            "semantics": "first_pass_top_k_retested_on_improved_whole_graph",
            "top_k": args.top_k,
            "iterations": args.iterations,
            "warmup": args.warmup,
            "repeats": args.repeats,
            "max_sweeps": args.max_sweeps,
            "resweep_top_k": args.resweep_top_k,
            "confirmation_iterations": args.confirmation_iterations,
            "min_improvement_fraction": args.min_improvement_fraction,
            "implementation_identity": workflow_implementation_identity(),
        }
        write_json(output, payload, compact=True)

    first_by_key = {
        (str(row.get("family")), int(row.get("batch", -1)),
         str(row.get("candidate_id"))): row
        for row in first_pass_rows
    }
    for batch in batches:
        selected: dict[str, dict[str, object]] = {}
        for family in families:
            expected = candidate_map(space, family, batch)
            missing = [
                candidate_id for candidate_id in expected
                if (family, batch, candidate_id) not in first_by_key
            ]
            if missing:
                raise ValueError(
                    f"first-pass coverage is incomplete for {family}/B{batch}: "
                    f"{missing[:4]}"
                )
            winners = [
                first_by_key[(family, batch, candidate_id)]
                for candidate_id in expected
                if first_by_key[(family, batch, candidate_id)].get(
                    "history_stage_winner"
                ) is True
            ]
            if len(winners) != 1:
                raise ValueError(
                    f"first pass has {len(winners)} winners for {family}/B{batch}"
                )
            selected[family] = expected[str(winners[0]["candidate_id"])]

        seed_ids = positive_history_seed_candidate_ids(
            architecture, gpu_class, batch,
        )
        for family, candidate_id in seed_ids.items():
            if family not in selected:
                raise ValueError(
                    f"positive-history seed has unknown family: {family}"
                )
            expected = candidate_map(space, family, batch)
            if candidate_id not in expected:
                raise ValueError(
                    "positive-history seed is absent from the materialized "
                    f"space: {family}/B{batch}/{candidate_id}"
                )
            selected[family] = expected[candidate_id]

        completed_sweeps = 0
        for sweep in range(1, args.max_sweeps + 1):
            changed_families: list[str] = []
            for family_index, family in enumerate(families):
                expected = candidate_map(space, family, batch)
                incumbent_id = str(selected[family]["id"])
                current_effective, superseded_by = effective_candidate_map(
                    selected
                )
                if family not in current_effective:
                    superseding_family = superseded_by.get(family)
                    if superseding_family is None:
                        raise ValueError(
                            "refinement dropped a family without an explicit "
                            f"superseding owner: {family}/B{batch}"
                        )
                    # A plan still records exactly one coordinate for every
                    # implementation catalog. Keep the selected coordinate as
                    # the catalog winner, while the joint ownership resolver
                    # records that it is inactive in this graph. The row keeps
                    # its original measurement rather than fabricating a
                    # redundant timing for an implementation that cannot run.
                    mark_superseded_refinement_winner(
                        base_rows, refinement_rows,
                        family=family, batch=batch,
                        candidate_id=incumbent_id,
                        superseding_family=superseding_family,
                        min_improvement_fraction=(
                            args.min_improvement_fraction
                        ),
                    )
                    print(
                        f"refine {family} B{batch}: skipped because the "
                        "current joint boundary explicitly supersedes it "
                        f"via {superseding_family}",
                        flush=True,
                    )
                    continue
                first_family_rows = [
                    first_by_key[(family, batch, candidate_id)]
                    for candidate_id in expected
                ]
                top_rows = refinement_top_candidates(
                    first_family_rows, incumbent_id,
                    args.top_k if sweep == 1 else args.resweep_top_k,
                )
                top_ids = [str(row["candidate_id"]) for row in top_rows]
                stage_rows: list[dict[str, object]] = []
                for candidate_id in top_ids:
                    value = expected[candidate_id]
                    tentative = dict(selected)
                    tentative[family] = value
                    compatible = True
                    incompatibility = None
                    for selected_family, selected_value in tentative.items():
                        compatible, incompatibility = candidate_compatibility(
                            selected_value, tentative,
                        )
                        if not compatible:
                            incompatibility = (
                                f"{selected_family}: {incompatibility}"
                            )
                            break
                    if not compatible:
                        print(
                            f"refine {family} B{batch} {candidate_id}: "
                            f"incompatible with improved graph ({incompatibility})",
                            flush=True,
                        )
                        continue
                    effective, _, applied, overridden_by = (
                        resolve_candidate_config_state(tentative)
                    )
                    full_state = runtime_tactic_baseline(architecture)
                    full_state.update(parse_key_values(source_override_config))
                    full_state.update(applied)
                    full_state["nnMaxBatchSize"] = batch
                    full_state["cudaWarmupOnlyMaxBatchSize"] = True
                    full_state["cudaDisableWarmup"] = True
                    # The resolved full graph already includes the current
                    # candidate and every later owner. Reapplying a family here
                    # would resurrect keys superseded by another family.
                    overrides = dict(full_state)
                    overrides.update(
                        topology_overrides(
                            architecture, device, streams, space,
                        )
                    )
                    command = list(runner) + [
                        str(binary), "benchmarknn",
                        "-config", str(config),
                        "-override-config", config_string(overrides),
                        "-model", str(model),
                        "-iterations", str(args.iterations),
                        "-warmup", str(args.warmup),
                        "-batch-size", str(batch),
                        "-boardsize", "19",
                        "-json",
                    ]
                    previous = next((
                        row for row in refinement_rows
                        if row.get("status") == "measured" and
                        row.get("family") == family and
                        int(row.get("batch", -1)) == batch and
                        row.get("candidate_id") == candidate_id and
                        row.get("candidate") == value and
                        row.get("command") == command and
                        row.get("overrides") == overrides and
                        row.get("binary_sha256") == binary_sha256 and
                        int(row.get("measurement_iterations", -1)) == args.iterations and
                        int(row.get("measurement_warmup", -1)) == args.warmup and
                        int(row.get("measurement_sample_count", -1)) == args.repeats
                    ), None)
                    if previous is not None:
                        stage_rows.append(previous)
                        continue
                    samples: list[float] = []
                    run_records: list[dict[str, object]] = []
                    for repeat in range(args.repeats):
                        completed = None
                        stdout_path = None
                        stderr_path = None
                        attempt_records: list[dict[str, object]] = []
                        occupancy_evidence: dict[str, object] = {}
                        for attempt in range(args.max_attempts):
                            completed, timed_out, occupancy_evidence = (
                                _run_benchmark_with_occupancy(
                                    command, device=device,
                                    timeout=args.timeout_seconds,
                                )
                            )
                            stem = re.sub(
                                r"[^A-Za-z0-9_.-]+", "_",
                                f"refine-s{sweep}-{family}-b{batch}-{candidate_id}-r{repeat}-a{attempt}",
                            )
                            stdout_path = raw_dir / f"{stem}.out"
                            stderr_path = raw_dir / f"{stem}.err"
                            stdout_path.write_text(
                                completed.stdout, encoding="utf-8",
                            )
                            stderr_path.write_text(
                                completed.stderr, encoding="utf-8",
                            )
                            attempt_records.append({
                                "attempt": attempt,
                                "returncode": completed.returncode,
                                "timed_out": timed_out,
                                "stdout": str(stdout_path),
                                "stderr": str(stderr_path),
                                "gpu_occupancy": occupancy_evidence,
                            })
                            if completed.returncode == 0:
                                break
                        assert completed is not None
                        assert stdout_path is not None
                        assert stderr_path is not None
                        if completed.returncode != 0:
                            write_checkpoint(False)
                            raise RuntimeError(
                                f"refinement failed for {family}/B{batch}/"
                                f"{candidate_id} after {args.max_attempts} "
                                f"attempts; see {stderr_path}"
                            )
                        combined_output = (
                            completed.stdout + "\n" + completed.stderr
                        )
                        for active_family, active_value in effective.items():
                            # Compatibility for result files materialized
                            # before policy/head-BN explicitly superseded the
                            # wide-head route.
                            if (
                                architecture == "sm120" and
                                active_family == "wide_head" and
                                (
                                    applied.get("cudaUseFusedPolicyP1") is False or
                                    applied.get("cudaUseHeadBNHalfToFloat") is False
                                )
                            ):
                                continue
                            require_activation_markers(
                                active_value, combined_output,
                                overridden_by.get(active_family, {}),
                            )
                        record = _parse_benchmark_record(completed.stdout)
                        throughput = result_metric(record)
                        samples.append(throughput)
                        run_records.append({
                            "repeat": repeat,
                            "throughput": throughput,
                            "benchmark": record,
                            "stdout": str(stdout_path),
                            "stderr": str(stderr_path),
                            "attempts": attempt_records,
                            "gpu_occupancy": occupancy_evidence,
                        })
                    row = {
                        "family": family,
                        "batch": batch,
                        "candidate_id": candidate_id,
                        "candidate": value,
                        "implementation": value.get("implementation"),
                        "status": "measured",
                        "command": command,
                        "overrides": overrides,
                        "history_base_overrides": config_string(full_state),
                        "history_stage_winner": False,
                        "history_final_joint": False,
                        "refinement_pass": 2,
                        "refinement_sweep": sweep,
                        "refinement_top_k": args.top_k,
                        "finished_utc": utc_now(),
                        "binary_sha256": binary_sha256,
                        "config_sha256": sha256_file(config),
                        "correctness": artifact_evidence.get(
                            (family, batch, candidate_id), {}
                        ).get("correctness", value.get("correctness")),
                        "runs": run_records,
                        **summarize_samples(
                            samples, iterations=args.iterations,
                            warmup=args.warmup,
                            max_relative_spread=args.max_relative_spread,
                        ),
                    }
                    refinement_rows = [
                        old for old in refinement_rows
                        if not (
                            old.get("family") == family and
                            int(old.get("batch", -1)) == batch and
                            old.get("candidate_id") == candidate_id
                        )
                    ]
                    refinement_rows.append(row)
                    stage_rows.append(row)
                    write_checkpoint(False)
                    print(
                        f"refine sweep{sweep} {family} B{batch} "
                        f"{candidate_id}: "
                        f"{row['nn_evals_per_sec_median']:.3f} nnEval/s",
                        flush=True,
                    )
                incumbent_rows = [
                    row for row in stage_rows
                    if row.get("candidate_id") == incumbent_id
                ]
                if len(incumbent_rows) != 1:
                    raise ValueError(
                        "refinement did not measure its incumbent exactly "
                        f"once: {family}/B{batch}/{incumbent_id}"
                    )
                incumbent = incumbent_rows[0]
                provisional = max(
                    stage_rows,
                    key=lambda row: (
                        float(row["nn_evals_per_sec_median"]),
                        row.get("candidate_id") == incumbent_id,
                    ),
                )
                required = float(incumbent["nn_evals_per_sec_median"]) * (
                    1.0 + args.min_improvement_fraction
                )
                best = incumbent
                incumbent_selection_metric = float(
                    incumbent["nn_evals_per_sec_median"]
                )
                best_selection_metric = incumbent_selection_metric
                if (
                    provisional.get("candidate_id") != incumbent_id and
                    float(provisional["nn_evals_per_sec_median"]) >= required
                ):
                    pair_rows = {
                        "incumbent": incumbent,
                        "challenger": provisional,
                    }
                    pair_samples: dict[str, list[float]] = {
                        "incumbent": [], "challenger": [],
                    }
                    pair_runs: list[dict[str, object]] = []
                    for sequence, label in enumerate((
                        "incumbent", "challenger", "challenger", "incumbent",
                    )):
                        pair_row = pair_rows[label]
                        confirm_command = list(pair_row["command"])
                        iterations_index = confirm_command.index("-iterations") + 1
                        confirm_command[iterations_index] = str(
                            args.confirmation_iterations
                        )
                        completed = None
                        stdout_path = None
                        stderr_path = None
                        occupancy_evidence: dict[str, object] = {}
                        attempt_records: list[dict[str, object]] = []
                        for attempt in range(args.max_attempts):
                            completed, timed_out, occupancy_evidence = (
                                _run_benchmark_with_occupancy(
                                    confirm_command, device=device,
                                    timeout=args.timeout_seconds,
                                )
                            )
                            stem = re.sub(
                                r"[^A-Za-z0-9_.-]+", "_",
                                f"confirm-s{sweep}-{family}-b{batch}-"
                                f"{label}-q{sequence}-a{attempt}",
                            )
                            stdout_path = raw_dir / f"{stem}.out"
                            stderr_path = raw_dir / f"{stem}.err"
                            stdout_path.write_text(
                                completed.stdout, encoding="utf-8",
                            )
                            stderr_path.write_text(
                                completed.stderr, encoding="utf-8",
                            )
                            attempt_records.append({
                                "attempt": attempt,
                                "returncode": completed.returncode,
                                "timed_out": timed_out,
                                "stdout": str(stdout_path),
                                "stderr": str(stderr_path),
                                "gpu_occupancy": occupancy_evidence,
                            })
                            if completed.returncode == 0:
                                break
                        assert completed is not None
                        assert stdout_path is not None
                        assert stderr_path is not None
                        if completed.returncode != 0:
                            write_checkpoint(False)
                            raise RuntimeError(
                                f"ABBA confirmation failed for {family}/B{batch}/"
                                f"{label}; see {stderr_path}"
                            )
                        throughput = result_metric(
                            _parse_benchmark_record(completed.stdout)
                        )
                        pair_samples[label].append(throughput)
                        pair_runs.append({
                            "sequence": sequence,
                            "label": label,
                            "candidate_id": pair_row["candidate_id"],
                            "throughput": throughput,
                            "command": confirm_command,
                            "stdout": str(stdout_path),
                            "stderr": str(stderr_path),
                            "attempts": attempt_records,
                            "gpu_occupancy": occupancy_evidence,
                        })
                    incumbent_selection_metric = statistics.mean(
                        pair_samples["incumbent"]
                    )
                    challenger_selection_metric = statistics.mean(
                        pair_samples["challenger"]
                    )
                    confirmation = {
                        "schema": 1,
                        "order": [
                            "incumbent", "challenger", "challenger", "incumbent",
                        ],
                        "iterations": args.confirmation_iterations,
                        "warmup": args.warmup,
                        "incumbent_candidate_id": incumbent_id,
                        "challenger_candidate_id": provisional["candidate_id"],
                        "incumbent_samples": pair_samples["incumbent"],
                        "challenger_samples": pair_samples["challenger"],
                        "incumbent_mean_nn_evals_per_sec": (
                            incumbent_selection_metric
                        ),
                        "challenger_mean_nn_evals_per_sec": (
                            challenger_selection_metric
                        ),
                        "runs": pair_runs,
                    }
                    incumbent["selection_confirmation"] = confirmation
                    provisional["selection_confirmation"] = confirmation
                    if challenger_selection_metric >= (
                        incumbent_selection_metric *
                        (1.0 + args.min_improvement_fraction)
                    ):
                        best = provisional
                        best_selection_metric = challenger_selection_metric
                    else:
                        best_selection_metric = incumbent_selection_metric
                    print(json.dumps({
                        "refinement_confirmation": family,
                        "batch": batch,
                        "sweep": sweep,
                        "incumbent": incumbent_id,
                        "challenger": provisional["candidate_id"],
                        "incumbent_mean": incumbent_selection_metric,
                        "challenger_mean": challenger_selection_metric,
                        "accepted": best is provisional,
                    }), flush=True)
                for row in refinement_rows:
                    if (
                        row.get("family") == family and
                        int(row.get("batch", -1)) == batch
                    ):
                        row["history_stage_winner"] = row is best
                        row["history_final_joint"] = False
                winner_id = str(best["candidate_id"])
                if winner_id != incumbent_id:
                    changed_families.append(family)
                selected[family] = expected[winner_id]
                _, _, selected_applied, _ = resolve_candidate_config_state(selected)
                accumulated = runtime_tactic_baseline(architecture)
                accumulated.update(parse_key_values(source_override_config))
                accumulated.update(selected_applied)
                accumulated["nnMaxBatchSize"] = batch
                accumulated["cudaWarmupOnlyMaxBatchSize"] = True
                accumulated["cudaDisableWarmup"] = True
                best["history_accumulated_overrides"] = config_string(accumulated)
                best["history_incumbent_candidate_id"] = incumbent_id
                best["history_incumbent_nn_evals_per_sec"] = float(
                    incumbent_selection_metric
                )
                best["history_selection_nn_evals_per_sec"] = float(
                    best_selection_metric
                )
                best["history_accepted_change"] = winner_id != incumbent_id
                best["history_min_improvement_fraction"] = (
                    args.min_improvement_fraction
                )
                best["history_improvement_fraction_vs_incumbent"] = (
                    float(best_selection_metric) /
                    float(incumbent_selection_metric) - 1.0
                )
                best["history_final_joint"] = family_index + 1 == len(families)
                write_checkpoint(False)
            completed_sweeps = sweep
            print(json.dumps({
                "refinement_sweep": sweep,
                "batch": batch,
                "changed_families": changed_families,
            }), flush=True)
            if not changed_families:
                break
        if completed_sweeps == args.max_sweeps and changed_families:
            print(
                f"refine B{batch}: reached --max-sweeps={args.max_sweeps} "
                f"with changes in {','.join(changed_families)}",
                flush=True,
            )
    write_checkpoint(True)
    print(json.dumps({
        "output": str(output),
        "first_pass_rows": len(base_rows),
        "refinement_rows": len(refinement_rows),
        "top_k": args.top_k,
        "max_sweeps": args.max_sweeps,
        "resweep_top_k": args.resweep_top_k,
        "confirmation_iterations": args.confirmation_iterations,
    }))


def run_gate(args: argparse.Namespace) -> None:
    """Long-stability gate for the final accumulated discovery winner."""
    if args.iterations < MIN_LONG_ITERATIONS or args.repeats < MIN_STABLE_SAMPLES:
        raise ValueError(
            f"gate requires at least {MIN_LONG_ITERATIONS} iterations and "
            f"{MIN_STABLE_SAMPLES} repeats"
        )
    if args.max_attempts < 1:
        raise ValueError("--max-attempts must be positive")
    if args.timeout_seconds <= 0:
        raise ValueError("--timeout-seconds must be positive")
    space_path = pathlib.Path(args.space).resolve()
    space = read_json(space_path)
    discovery_path = pathlib.Path(args.discovery).resolve()
    discovery = read_json(discovery_path)
    if discovery.get("kind") != RESULT_KIND:
        raise ValueError("gate discovery input is not a scan result")
    if discovery.get("space_sha256") != sha256_file(space_path):
        raise ValueError("gate discovery input does not match --space")
    architecture = str(space["architecture"])
    gpu_class = str(space["gpu_class"])
    target_families = space_families(space)
    device = int(args.device if args.device is not None else space.get("device_ordinal", 0))
    streams = int(space["streams"])
    try:
        from portable_cuda_device import query_cuda_device
    except ModuleNotFoundError:
        from python.portable_cuda_device import query_cuda_device
    device_properties = query_cuda_device(device)
    if cuda_compute_capability(device_properties) != space.get("compute_capability"):
        raise ValueError(
            "CUDA-reported gate device capability does not match the search space"
        )
    batches = parse_int_set(args.batches) if args.batches else sorted(space_batches(space))
    discovery_rows = [
        row for row in discovery.get("rows", [])
        if isinstance(row, dict) and row.get("status") == "measured"
    ]
    by_key = {
        (str(row.get("family")), int(row.get("batch", -1)), str(row.get("candidate_id"))): row
        for row in discovery_rows
    }
    selected_aot: list[tuple[str, int, str]] = []
    selected_candidates_by_batch: dict[int, dict[str, dict[str, object]]] = {}
    overridden_keys_by_batch: dict[int, dict[str, dict[str, str]]] = {}
    final_rows: dict[int, dict[str, object]] = {}
    for batch in batches:
        selected_for_batch: dict[str, dict[str, object]] = {}
        for family in target_families:
            expected = candidate_map(space, family, batch)
            missing = [
                candidate_id for candidate_id in expected
                if (family, batch, candidate_id) not in by_key
            ]
            if missing:
                raise ValueError(
                    f"discovery coverage is incomplete for {family}/B{batch}: {missing[:4]}"
                )
            winners = [
                by_key[(family, batch, candidate_id)] for candidate_id in expected
                if by_key[(family, batch, candidate_id)].get("status") == "measured" and
                by_key[(family, batch, candidate_id)].get("history_stage_winner") is True
            ]
            if len(winners) != 1:
                raise ValueError(
                    f"discovery has {len(winners)} history winners for {family}/B{batch}"
                )
            winner_id = str(winners[0]["candidate_id"])
            selected_for_batch[family] = expected[winner_id]
        effective_for_batch, _, _, overridden_by = resolve_candidate_config_state(
            selected_for_batch
        )
        selected_candidates_by_batch[batch] = effective_for_batch
        overridden_keys_by_batch[batch] = overridden_by
        for family, selected in effective_for_batch.items():
            selected_id = str(selected["id"])
            if selected.get("requires_artifact"):
                selected_aot.append((family, batch, selected_id))
            for dependency in selected.get("artifact_dependencies", []):
                selected_aot.append((
                    str(dependency["family"]), batch,
                    str(dependency["candidate_id"]),
                ))
        final = [
            row for row in discovery_rows
            if int(row.get("batch", -1)) == batch and
            row.get("status") == "measured" and
            row.get("history_final_joint") is True
        ]
        if len(final) != 1 or not final[0].get("history_accumulated_overrides"):
            raise ValueError(f"discovery has no unique final joint state for B{batch}")
        final_rows[batch] = final[0]

    binary = pathlib.Path(args.binary).resolve()
    config = pathlib.Path(args.config).resolve()
    model = pathlib.Path(args.model).resolve()
    model_identity = pathlib.Path(
        args.model_identity if args.model_identity else args.model
    ).resolve()
    for path, label in (
        (binary, "binary"), (config, "config"), (model, "model"),
        (model_identity, "model identity"),
    ):
        if not path.is_file():
            raise ValueError(f"gate {label} does not exist: {path}")
    artifact_evidence: dict[tuple[str, int, str], dict[str, object]] = {}
    artifact_bundle_metadata: dict[str, object] | None = None
    if selected_aot:
        if not args.artifact_bundle:
            raise ValueError("selected final tactics require --artifact-bundle")
        artifact_evidence, artifact_bundle_metadata = validate_artifact_bundle(
            pathlib.Path(args.artifact_bundle).resolve(),
            space_path=space_path, space=space, binary=binary, required=selected_aot,
        )

    output = pathlib.Path(args.output).resolve()
    raw_dir = pathlib.Path(args.raw_dir).resolve() if args.raw_dir else output.parent / f"{output.stem}-raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    runner = shlex.split(args.runner) if args.runner else []
    rows: list[dict[str, object]] = []
    provenance = collect_provenance(
        pathlib.Path(__file__).resolve().parents[1], binary=binary,
        config=config, model=model, device=device,
    )
    started = utc_now()
    implementation_identity = workflow_implementation_identity()
    for batch in batches:
        source = final_rows[batch]
        overrides = parse_key_values(str(source["history_accumulated_overrides"]))
        overrides.update(topology_overrides(architecture, device, streams, space))
        overrides["cudaWarmupOnlyMaxBatchSize"] = True
        overrides["cudaDisableWarmup"] = True
        command = runner + [
            str(binary), "benchmarknn", "-config", str(config),
            "-override-config", config_string(overrides),
            "-model", str(model), "-iterations", str(args.iterations),
            "-warmup", str(args.warmup), "-batch-size", str(batch),
            "-boardsize", "19", "-json",
        ]
        samples: list[float] = []
        run_records: list[dict[str, object]] = []
        for repeat in range(args.repeats):
            attempt_records: list[dict[str, object]] = []
            completed = None
            stdout_path = None
            stderr_path = None
            occupancy_evidence: dict[str, object] = {}
            for attempt in range(args.max_attempts):
                completed, timed_out, occupancy_evidence = (
                    _run_benchmark_with_occupancy(
                        command, device=device, timeout=args.timeout_seconds,
                    )
                )
                stem = f"final-joint-b{batch}-r{repeat}-a{attempt}"
                stdout_path = raw_dir / f"{stem}.out"
                stderr_path = raw_dir / f"{stem}.err"
                stdout_path.write_text(completed.stdout, encoding="utf-8")
                stderr_path.write_text(completed.stderr, encoding="utf-8")
                attempt_records.append({
                    "attempt": attempt,
                    "returncode": completed.returncode,
                    "timed_out": timed_out,
                    "stdout": str(stdout_path),
                    "stderr": str(stderr_path),
                })
                if completed.returncode == 0:
                    break
            assert completed is not None and stdout_path is not None and stderr_path is not None
            if completed.returncode != 0:
                raise RuntimeError(
                    f"final joint gate failed for B{batch} after "
                    f"{args.max_attempts} attempts; see {stderr_path}"
                )
            combined_output = completed.stdout + "\n" + completed.stderr
            for family, selected in selected_candidates_by_batch[batch].items():
                require_activation_markers(
                    selected, combined_output,
                    overridden_keys_by_batch[batch].get(family, {}),
                )
            record = _parse_benchmark_record(completed.stdout)
            throughput = result_metric(record)
            samples.append(throughput)
            run_records.append({
                "repeat": repeat, "throughput": throughput,
                "benchmark": record, "stdout": str(stdout_path),
                "stderr": str(stderr_path), "attempts": attempt_records,
                "gpu_occupancy": occupancy_evidence,
            })
        row = dict(source)
        row.update({
            "status": "measured",
            "command": command,
            "overrides": overrides,
            "finished_utc": utc_now(),
            "binary_sha256": sha256_file(binary),
            "history_stage_winner": True,
            "history_final_joint": True,
            "history_long_gate": True,
            "history_incumbent_candidate_id": source.get(
                "history_incumbent_candidate_id"
            ),
            "history_incumbent_nn_evals_per_sec": source.get(
                "history_incumbent_nn_evals_per_sec"
            ),
            "history_accepted_change": source.get("history_accepted_change"),
            "history_min_improvement_fraction": source.get(
                "history_min_improvement_fraction"
            ),
            "history_improvement_fraction_vs_incumbent": source.get(
                "history_improvement_fraction_vs_incumbent"
            ),
            "discovery_result": str(discovery_path),
            "runs": run_records,
            **summarize_samples(
                samples, iterations=args.iterations, warmup=args.warmup,
                max_relative_spread=args.max_relative_spread,
            ),
        })
        rows.append(row)
        metric = stable_metric(row)
        if metric is None:
            raise RuntimeError(
                f"final joint B{batch} did not pass the long-stability gate"
            )
        print(f"final joint B{batch}: {metric:.3f} nnEval/s (long_stable)", flush=True)
    # Reuse the result schema so plan can merge discovery coverage with these
    # newer rows for the same final candidate IDs.
    args.phase = "long"
    args.families = ",".join(target_families)
    args.override_config = ""
    _write_scan_payload(
        output, space_path, space, architecture, gpu_class, device, streams,
        args, started, provenance, artifact_bundle_metadata, rows,
        device_properties, implementation_identity,
    )
    print(json.dumps({"output": str(output), "rows": len(rows)}))


def _write_scan_payload(
    output: pathlib.Path,
    space_path: pathlib.Path,
    space: dict[str, object],
    architecture: str,
    gpu_class: str,
    device: int,
    streams: int,
    args: argparse.Namespace,
    started: str,
    provenance: dict[str, object],
    artifact_bundle_metadata: dict[str, object] | None,
    rows: list[dict[str, object]],
    device_properties: dict[str, object] | None,
    implementation_identity: dict[str, object],
) -> None:
    execution_model_path = pathlib.Path(args.model).resolve()
    identity_model_path = pathlib.Path(
        args.model_identity if getattr(args, "model_identity", None) else args.model
    ).resolve()
    config_path = pathlib.Path(args.config).resolve()
    identity = {
        # The compressed source model remains the portable identity while an
        # equivalent uncompressed copy may be used to avoid repeated inflate
        # cost in thousands of short-lived discovery subprocesses.
        "model_sha256": (
            sha256_file(identity_model_path) if identity_model_path.is_file() else None
        ),
        "identity_model_path": str(identity_model_path),
        "execution_model_sha256": (
            sha256_file(execution_model_path) if execution_model_path.is_file() else None
        ),
        "execution_model_path": str(execution_model_path),
        "config_sha256": sha256_file(config_path) if config_path.is_file() else None,
    }
    cuda_device_capabilities: list[dict[str, object]] = []
    seen_capabilities: set[str] = set()
    for row in rows:
        runs = row.get("runs", [])
        if not isinstance(runs, list):
            continue
        for run in runs:
            benchmark = run.get("benchmark", {}) if isinstance(run, dict) else {}
            devices = benchmark.get("cudaDevices", []) if isinstance(benchmark, dict) else []
            if not isinstance(devices, list):
                continue
            for capability in devices:
                if not isinstance(capability, dict):
                    continue
                key = canonical_json(capability)
                if key not in seen_capabilities:
                    seen_capabilities.add(key)
                    cuda_device_capabilities.append(capability)
    payload = {
        "schema": SCHEMA,
        "kind": RESULT_KIND,
        "started_utc": started,
        "finished_utc": utc_now(),
        "architecture": architecture,
        "compute_capability": (
            cuda_compute_capability(device_properties)
            if device_properties is not None else space.get("compute_capability")
        ),
        "gpu_class": gpu_class,
        "device_ordinal": device,
        "streams": streams,
        "fixed_board": [19, 19],
        "precision": space.get("precision"),
        "space": str(space_path),
        "space_sha256": sha256_file(space_path),
        "family": None if "," in args.families else args.families,
        "identity": identity,
        "scan_parameters": {
            "search_semantics": "accepted_history_seeded_accumulated_coordinate",
            "family_order": [item.strip() for item in args.families.split(",") if item.strip()],
            "phase": args.phase,
            "iterations": args.iterations, "warmup": args.warmup,
            "repeats": args.repeats,
            "max_attempts": getattr(args, "max_attempts", 1),
            "timeout_seconds": getattr(args, "timeout_seconds", None),
            "max_relative_spread": args.max_relative_spread,
            "min_improvement_fraction": getattr(
                args, "min_improvement_fraction",
                DEFAULT_MIN_DISCOVERY_IMPROVEMENT_FRACTION,
            ),
            "runner": shlex.split(args.runner) if args.runner else [],
            "override_config": args.override_config or "",
        },
        "artifact_bundle": artifact_bundle_metadata,
        "implementation_identity": implementation_identity,
        "cuda_device_capabilities": cuda_device_capabilities,
        "cuda_device_properties_at_scan_start": device_properties,
        "provenance": provenance,
        "rows": rows,
    }
    # Scan payloads can contain thousands of commands and benchmark records.
    # Compact encoding keeps family-level atomic checkpoints cheap enough that
    # serialization does not steal time from the GPU search.
    write_json(output, payload, compact=True)


def command_space(args: argparse.Namespace) -> None:
    architecture = canonical_architecture(args.architecture, args.gpu_class)
    gpu_class = args.gpu_class or ARCHITECTURES[architecture]["gpu_classes"][0]
    try:
        from portable_cuda_device import query_cuda_device
    except ModuleNotFoundError:
        from python.portable_cuda_device import query_cuda_device
    device_properties = query_cuda_device(args.device)
    payload = materialize_space(
        architecture, gpu_class, args.device, parse_int_set(args.batches),
        args.streams, args.candidate_file, args.topology_override,
        device_properties,
    )
    if args.output:
        write_json(pathlib.Path(args.output).resolve(), payload)
        print(json.dumps({"output": str(pathlib.Path(args.output).resolve()), "batches": payload["batches"] and len(payload["batches"])}))
    else:
        print(json.dumps(payload, indent=2, sort_keys=True))


def command_generation_plan(args: argparse.Namespace) -> None:
    families = [item.strip() for item in args.families.split(",") if item.strip()]
    payload = make_generation_plan(
        pathlib.Path(args.space).resolve(), phase=args.phase, families=families,
    )
    if args.output:
        write_json(pathlib.Path(args.output).resolve(), payload)
    else:
        print(json.dumps(payload, indent=2, sort_keys=True))
    print(json.dumps({
        "phase": payload["phase"],
        "batches": len(payload["batches"]),
        "families": len(payload["families"]),
        "tasks": len(payload["tasks"]),
        "output": str(pathlib.Path(args.output).resolve()) if args.output else None,
    }))


def command_artifact_bundle(args: argparse.Namespace) -> None:
    output = pathlib.Path(args.output).resolve()
    payload = build_artifact_bundle(
        pathlib.Path(args.space).resolve(),
        pathlib.Path(args.binary).resolve(),
        [pathlib.Path(item).resolve() for item in args.manifests],
    )
    write_json(output, payload)
    print(json.dumps({
        "output": str(output),
        "entries": len(payload["entries"]),
        "linked_binary_sha256": payload["linked_binary_sha256"],
    }))


def command_certify(args: argparse.Namespace) -> None:
    gate_path = pathlib.Path(args.gate).resolve()
    payload = read_json(gate_path)
    if payload.get("kind") != RESULT_KIND:
        raise ValueError("certify requires a long-gate scan result")
    reports: dict[int, pathlib.Path] = {}
    for item in args.comparison:
        if "=" not in item:
            raise ValueError("--comparison must use BATCH=PATH")
        batch_text, path_text = item.split("=", 1)
        batch = int(batch_text)
        if batch in reports:
            raise ValueError(f"duplicate accuracy comparison for B{batch}")
        reports[batch] = pathlib.Path(path_text).resolve()
    thresholds = {
        "minimum_rows": 8192,
        "minimum_policy_top1_vs_reference": 0.995,
        "maximum_weighted_p0loss_delta": 0.001,
        "maximum_policy_probability_rmse": 0.001,
        "maximum_value_outcome_rmse": 0.01,
        "maximum_score_mean_rmse": 0.01,
        "maximum_ownership_sigmoid_rmse": 0.001,
        "maximum_request_policy_probability_abs": 0.025,
        "maximum_request_policy_probability_rmse": 0.002,
        "maximum_request_value_probability_abs": 0.06,
        "maximum_request_value_probability_rmse": 0.05,
        "maximum_request_score_raw_abs": 0.60,
        "maximum_request_score_raw_rmse": 0.30,
        "maximum_request_ownership_probability_abs": 0.025,
        "maximum_request_ownership_probability_rmse": 0.006,
    }
    certified = 0
    reference_hashes: set[str] = set()
    corpus_hashes: set[str] = set()
    model_hashes: set[str] = set()
    rows = payload.get("rows", [])
    if not isinstance(rows, list):
        raise ValueError("gate result rows are not a list")
    gate_batches = {
        int(row.get("batch", -1))
        for row in rows
        if isinstance(row, dict) and row.get("history_long_gate") is True
    }
    batches_text = getattr(args, "batches", None)
    target_batches = parse_int_set(batches_text) if batches_text else sorted(gate_batches)
    unknown_batches = sorted(set(target_batches) - gate_batches)
    if unknown_batches:
        raise ValueError(
            f"certification batches are absent from the long gate: {unknown_batches}"
        )
    missing_reports = sorted(set(target_batches) - reports.keys())
    unexpected_reports = sorted(reports.keys() - set(target_batches))
    if missing_reports:
        raise ValueError(f"missing --comparison for gate B{missing_reports[0]}")
    if unexpected_reports:
        raise ValueError(
            f"comparison batches were not selected for certification: {unexpected_reports}"
        )
    for row in rows:
        if not isinstance(row, dict) or row.get("history_long_gate") is not True:
            continue
        batch = int(row.get("batch", -1))
        if batch not in target_batches:
            continue
        report_path = reports.get(batch)
        assert report_path is not None
        report = read_json(report_path)
        reference_sha256 = str(report.get("referenceSha256", ""))
        candidate_sha256 = str(report.get("candidateSha256", ""))
        if not re.fullmatch(r"[0-9a-f]{64}", reference_sha256):
            raise ValueError(
                f"accuracy comparison lacks an immutable reference SHA-256: B{batch}"
            )
        if not re.fullmatch(r"[0-9a-f]{64}", candidate_sha256):
            raise ValueError(
                f"accuracy comparison lacks a candidate SHA-256: B{batch}"
            )
        if int(report.get("exactBatch", -1)) != batch:
            raise ValueError(
                f"accuracy comparison is not bound to exact B{batch}"
            )
        if (
            int(report.get("candidateMaxBatchSize", -1)) != batch or
            report.get("candidateFixedBatchTailPadding") is not True or
            report.get("referenceFixedBatchTailPadding") is not True or
            report.get("inputAndTargetSectionsByteExact") is not True
        ):
            raise ValueError(
                f"accuracy comparison lacks fixed-batch/input identity evidence: B{batch}"
            )
        if report.get("candidateBinarySha256") != row.get("binary_sha256"):
            raise ValueError(
                f"accuracy comparison binary differs from long gate: B{batch}"
            )
        if report.get("candidateOverrides") != row.get("overrides"):
            raise ValueError(
                f"accuracy comparison overrides differ from long gate: B{batch}"
            )
        corpus_sha256 = str(report.get("corpusSha256", ""))
        model_sha256 = str(report.get("modelSha256", ""))
        if not re.fullmatch(r"[0-9a-f]{64}", corpus_sha256):
            raise ValueError(f"accuracy comparison lacks corpus identity: B{batch}")
        if not re.fullmatch(r"[0-9a-f]{64}", model_sha256):
            raise ValueError(f"accuracy comparison lacks model identity: B{batch}")
        gate_identity = payload.get("identity", {})
        if (
            not isinstance(gate_identity, dict) or
            gate_identity.get("model_sha256") != model_sha256
        ):
            raise ValueError(
                f"accuracy comparison model differs from long gate: B{batch}"
            )
        reference_hashes.add(reference_sha256)
        corpus_hashes.add(corpus_sha256)
        model_hashes.add(model_sha256)
        policy = report.get("policy", {})
        value = report.get("value", {})
        score = report.get("score", {})
        ownership = report.get("ownership", {})
        request_gate = report.get("requestGate", {})
        request_policy = request_gate.get("policyProbability", {})
        request_value = request_gate.get("valueProbability", {})
        request_score = request_gate.get("scoreRaw", {})
        request_ownership = request_gate.get("ownershipProbability", {})
        p0_delta = abs(
            float(policy.get("p0lossCandidateWeighted", math.inf)) -
            float(policy.get("p0lossReferenceWeighted", -math.inf))
        )
        checks = {
            "rows": int(report.get("numRows", 0)) >= thresholds["minimum_rows"],
            "policy_top1": float(policy.get("top1VsReference", -math.inf)) >= thresholds["minimum_policy_top1_vs_reference"],
            "weighted_p0loss_delta": p0_delta <= thresholds["maximum_weighted_p0loss_delta"],
            "policy_probability_rmse": float(policy.get("probabilityRmse", math.inf)) <= thresholds["maximum_policy_probability_rmse"],
            "value_outcome_rmse": float(value.get("outcomeRmse", math.inf)) <= thresholds["maximum_value_outcome_rmse"],
            "score_mean_rmse": float(score.get("meanRmse", math.inf)) <= thresholds["maximum_score_mean_rmse"],
            "ownership_sigmoid_rmse": float(ownership.get("sigmoidRmse", math.inf)) <= thresholds["maximum_ownership_sigmoid_rmse"],
            "request_policy_probability_abs": float(request_policy.get("maximumAbs", math.inf)) <= thresholds["maximum_request_policy_probability_abs"],
            "request_policy_probability_rmse": float(request_policy.get("maximumRmse", math.inf)) <= thresholds["maximum_request_policy_probability_rmse"],
            "request_value_probability_abs": float(request_value.get("maximumAbs", math.inf)) <= thresholds["maximum_request_value_probability_abs"],
            "request_value_probability_rmse": float(request_value.get("maximumRmse", math.inf)) <= thresholds["maximum_request_value_probability_rmse"],
            "request_score_raw_abs": float(request_score.get("maximumAbs", math.inf)) <= thresholds["maximum_request_score_raw_abs"],
            "request_score_raw_rmse": float(request_score.get("maximumRmse", math.inf)) <= thresholds["maximum_request_score_raw_rmse"],
            "request_ownership_probability_abs": float(request_ownership.get("maximumAbs", math.inf)) <= thresholds["maximum_request_ownership_probability_abs"],
            "request_ownership_probability_rmse": float(request_ownership.get("maximumRmse", math.inf)) <= thresholds["maximum_request_ownership_probability_rmse"],
        }
        status = "passed" if all(checks.values()) else "failed"
        row["correctness"] = {
            "status": status,
            "kind": "8192-row all-head FP32-reference replay",
            "comparison": str(report_path),
            "comparison_sha256": sha256_file(report_path),
            "reference_sha256": reference_sha256,
            "candidate_sha256": candidate_sha256,
            "thresholds": thresholds,
            "checks": checks,
            "metrics": {
                "policy_top1_vs_reference": policy.get("top1VsReference"),
                "weighted_p0loss_delta": p0_delta,
                "policy_probability_rmse": policy.get("probabilityRmse"),
                "value_outcome_rmse": value.get("outcomeRmse"),
                "score_mean_rmse": score.get("meanRmse"),
                "ownership_sigmoid_rmse": ownership.get("sigmoidRmse"),
                "request_gate": request_gate,
            },
        }
        row["finished_utc"] = utc_now()
        if status != "passed":
            failed = ", ".join(name for name, passed in checks.items() if not passed)
            raise ValueError(f"accuracy certification failed for B{batch}: {failed}")
        certified += 1
    if len(reference_hashes) != 1 or len(corpus_hashes) != 1 or len(model_hashes) != 1:
        raise ValueError(
            "accuracy comparisons do not share one immutable reference, "
            "corpus, and model"
        )
    if certified != len(reports):
        raise ValueError(
            f"certified {certified} gate rows from {len(reports)} comparison reports"
        )
    payload["finished_utc"] = utc_now()
    payload["accuracy_certification"] = {
        "status": "passed", "thresholds": thresholds,
        "batches": sorted(reports),
        "reference_sha256": next(iter(reference_hashes)),
        "corpus_sha256": next(iter(corpus_hashes)),
        "model_sha256": next(iter(model_hashes)),
    }
    output = pathlib.Path(args.output).resolve()
    write_json(output, payload)
    print(json.dumps({"output": str(output), "certified_batches": certified}))


def command_plan(args: argparse.Namespace) -> None:
    families = [item.strip() for item in args.families.split(",") if item.strip()]
    space_path = pathlib.Path(args.space).resolve()
    if not families:
        families = list(space_families(read_json(space_path)))
    payload = build_plan(
        [pathlib.Path(item).resolve() for item in args.results],
        space_path, families,
        parse_int_set(args.batches), allow_partial=args.allow_partial,
    )
    write_json(pathlib.Path(args.output).resolve(), payload)
    print(json.dumps({
        "output": str(pathlib.Path(args.output).resolve()),
        "plan_id": payload["plan_id"],
        "ready_for_scan_bypass": payload["ready_for_scan_bypass"],
        "production_ready": payload["production_ready"],
        "missing_groups": len(payload["missing"]),
    }))


def command_validate(args: argparse.Namespace) -> None:
    plan_path = pathlib.Path(args.plan).resolve()
    plan = load_plan(plan_path)
    space_path = pathlib.Path(args.space).resolve() if args.space else None
    device_properties = None
    if args.device is not None:
        try:
            from portable_cuda_device import query_cuda_device
        except ModuleNotFoundError:
            from python.portable_cuda_device import query_cuda_device
        device_properties = query_cuda_device(args.device)
    result = validate_plan(
        plan,
        space_path=space_path,
        model=pathlib.Path(args.model).resolve() if args.model else None,
        config=pathlib.Path(args.config).resolve() if args.config else None,
        architecture=args.architecture, gpu_class=args.gpu_class,
        streams=args.streams,
        batches=parse_int_set(args.batches) if args.batches else None,
        families=[item.strip() for item in args.families.split(",") if item.strip()],
        device_properties=device_properties,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


def command_apply(args: argparse.Namespace) -> None:
    plan = load_plan(pathlib.Path(args.plan).resolve())
    batches = parse_int_set(args.batches) if args.batches else [int(item) for item in plan.get("batches", [])]
    families = [item.strip() for item in args.families.split(",") if item.strip()]
    target = plan["target"]
    if not families:
        families = list(architecture_families(str(target["architecture"])))
    device = int(args.device if args.device is not None else target.get("device_ordinal_at_scan", 0))
    try:
        from portable_cuda_device import query_cuda_device
    except ModuleNotFoundError:
        from python.portable_cuda_device import query_cuda_device
    validate_plan(
        plan, batches=batches, families=families,
        device_properties=query_cuda_device(device),
    )
    streams = int(target["streams"])
    topology = topology_overrides(
        str(target["architecture"]), device, streams
    )
    result: dict[str, object] = {
        "schema": 1,
        "kind": "cuda-tactic-application",
        "plan_id": plan.get("plan_id"),
        "architecture": target["architecture"],
        "gpu_class": target["gpu_class"],
        "device_ordinal": device,
        "streams": streams,
        "batches": {},
    }
    for batch in batches:
        selected: dict[str, object] = {}
        family_map = plan.get("families", {})
        for family in families:
            entry = family_map[family]["batches"][str(batch)]
            selected[family] = {
                "candidate_id": entry["candidate_id"],
                "stable_long_nn_evals_per_sec": entry["stable_long_nn_evals_per_sec"],
                "candidate": entry["candidate"],
            }
        tactic_values = runtime_tactic_baseline(str(target["architecture"]))
        for family in families:
            entry = family_map[family]["batches"][str(batch)]
            tactic_values.update(tactic_overrides(family, entry["candidate"]))
        all_values = dict(topology)
        all_values.update(tactic_values)
        result["batches"][str(batch)] = {
            "selected": selected,
            "topology_overrides": config_string(topology),
            "tactic_overrides": config_string(tactic_values),
            "override_config": config_string(all_values),
        }
    if args.output:
        write_json(pathlib.Path(args.output).resolve(), result)
    print(json.dumps(result, indent=2, sort_keys=True))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    space = sub.add_parser("space", help="materialize an SM89/SM120 device/batch search space")
    space.add_argument("--architecture", choices=tuple(ARCHITECTURES))
    space.add_argument("--gpu-class")
    space.add_argument("--device", type=int, default=0)
    space.add_argument("--batches", default="4-32")
    space.add_argument("--streams", type=int, default=2)
    space.add_argument("--candidate-file", action="append", default=[])
    space.add_argument("--topology-override")
    space.add_argument("--output")
    space.set_defaults(function=command_space)

    generation = sub.add_parser(
        "generation-plan",
        help="materialize per-batch generator tasks from the optimization history",
    )
    generation.add_argument("--space", required=True)
    generation.add_argument("--phase", choices=("seed", "full"), default="full")
    generation.add_argument("--families", default="")
    generation.add_argument("--output")
    generation.set_defaults(function=command_generation_plan)

    artifact = sub.add_parser(
        "artifact-bundle",
        help="prove generated AOT sources/objects are present in the linked binary",
    )
    artifact.add_argument("--space", required=True)
    artifact.add_argument("--binary", required=True)
    artifact.add_argument("--manifests", nargs="*", default=[])
    artifact.add_argument("--output", required=True)
    artifact.set_defaults(function=command_artifact_bundle)

    prescan = sub.add_parser(
        "baseline-prescan",
        help="rank exact batches with an artifact-free stable optimized graph",
    )
    prescan.add_argument("--architecture", choices=tuple(ARCHITECTURES), required=True)
    prescan.add_argument("--gpu-class", required=True)
    prescan.add_argument("--device", type=int, default=0)
    prescan.add_argument("--streams", type=int, default=2)
    prescan.add_argument("--batches", default="4-32")
    prescan.add_argument("--top-batches", type=int, default=3)
    prescan.add_argument("--binary", required=True)
    prescan.add_argument("--config", required=True)
    prescan.add_argument("--model", required=True)
    prescan.add_argument("--iterations", type=int, default=200)
    prescan.add_argument("--warmup", type=int, default=50)
    prescan.add_argument("--repeats", type=int, default=2)
    prescan.add_argument("--max-attempts", type=int, default=2)
    prescan.add_argument("--timeout-seconds", type=float, default=120.0)
    prescan.add_argument(
        "--max-relative-spread", type=float,
        default=DEFAULT_MAX_RELATIVE_SPREAD,
    )
    prescan.add_argument("--raw-dir", required=True)
    prescan.add_argument("--output", required=True)
    prescan.set_defaults(function=run_stable_optimized_batch_prescan)

    certify = sub.add_parser(
        "certify", help="attach an accepted 8192-row FP32 replay to long-gate rows",
    )
    certify.add_argument("--gate", required=True)
    certify.add_argument(
        "--comparison", action="append", required=True, metavar="BATCH=PATH",
    )
    certify.add_argument(
        "--batches",
        help=(
            "certify only this gate subset; without this option every long-gate "
            "batch still requires a comparison"
        ),
    )
    certify.add_argument("--output", required=True)
    certify.set_defaults(function=command_certify)

    scan = sub.add_parser("scan", help="scan candidates with whole-graph benchmarknn")
    scan.add_argument("--space", required=True)
    scan.add_argument("--binary", required=True)
    scan.add_argument("--config", required=True)
    scan.add_argument("--model", required=True)
    scan.add_argument(
        "--model-identity",
        help="portable source-model identity when --model is an equivalent execution copy",
    )
    scan.add_argument("--output", required=True)
    scan.add_argument("--raw-dir")
    scan.add_argument("--architecture")
    scan.add_argument("--device", type=int)
    scan.add_argument("--streams", type=int)
    scan.add_argument("--batches")
    scan.add_argument("--families", default="")
    scan.add_argument("--phase", choices=("discovery", "long"), default="long")
    scan.add_argument("--iterations", type=int, default=MIN_LONG_ITERATIONS)
    scan.add_argument("--warmup", type=int, default=50)
    scan.add_argument("--repeats", type=int, default=MIN_STABLE_SAMPLES)
    scan.add_argument("--max-attempts", type=int, default=2)
    scan.add_argument(
        "--timeout-seconds", type=float, default=60.0,
        help="terminate and retry a benchmark subprocess that stops making progress",
    )
    scan.add_argument("--max-relative-spread", type=float, default=DEFAULT_MAX_RELATIVE_SPREAD)
    scan.add_argument(
        "--min-improvement-fraction", type=float,
        default=DEFAULT_MIN_DISCOVERY_IMPROVEMENT_FRACTION,
        help=(
            "retain the measured incumbent unless a candidate exceeds it by "
            "this fraction (default: 0.001)"
        ),
    )
    scan.add_argument("--override-config")
    scan.add_argument("--runner", help="optional command prefix, parsed with shlex")
    scan.add_argument("--resume", action="store_true")
    scan.add_argument("--dry-run", action="store_true")
    scan.add_argument(
        "--artifact-bundle",
        help="complete generation/link manifest whose binary hash matches --binary",
    )
    scan.set_defaults(function=run_scan)

    refine = sub.add_parser(
        "refine",
        help="retest each family's first-pass top-K on the improved whole graph",
    )
    refine.add_argument("--space", required=True)
    refine.add_argument("--discovery", required=True)
    refine.add_argument("--binary", required=True)
    refine.add_argument("--config", required=True)
    refine.add_argument("--model", required=True)
    refine.add_argument("--model-identity")
    refine.add_argument("--output", required=True)
    refine.add_argument("--raw-dir")
    refine.add_argument("--device", type=int)
    refine.add_argument("--batches")
    refine.add_argument("--top-k", type=int, default=10)
    refine.add_argument("--max-sweeps", type=int, default=3)
    refine.add_argument("--resweep-top-k", type=int, default=3)
    refine.add_argument("--confirmation-iterations", type=int, default=300)
    refine.add_argument("--iterations", type=int, default=MIN_DISCOVERY_ITERATIONS)
    refine.add_argument("--warmup", type=int, default=MIN_DISCOVERY_WARMUP)
    refine.add_argument("--repeats", type=int, default=1)
    refine.add_argument("--max-attempts", type=int, default=2)
    refine.add_argument("--timeout-seconds", type=float, default=60.0)
    refine.add_argument(
        "--max-relative-spread", type=float,
        default=DEFAULT_MAX_RELATIVE_SPREAD,
    )
    refine.add_argument(
        "--min-improvement-fraction", type=float,
        default=DEFAULT_MIN_DISCOVERY_IMPROVEMENT_FRACTION,
    )
    refine.add_argument("--runner")
    refine.add_argument("--resume", action="store_true")
    refine.add_argument("--artifact-bundle")
    refine.set_defaults(function=run_refine)

    gate = sub.add_parser(
        "gate", help="long-stability gate for discovery's final accumulated joint winner",
    )
    gate.add_argument("--space", required=True)
    gate.add_argument("--discovery", required=True)
    gate.add_argument("--binary", required=True)
    gate.add_argument("--config", required=True)
    gate.add_argument("--model", required=True)
    gate.add_argument(
        "--model-identity",
        help="portable source-model identity when --model is an equivalent execution copy",
    )
    gate.add_argument("--output", required=True)
    gate.add_argument("--raw-dir")
    gate.add_argument("--device", type=int)
    gate.add_argument("--batches")
    gate.add_argument("--iterations", type=int, default=MIN_LONG_ITERATIONS)
    gate.add_argument("--warmup", type=int, default=50)
    gate.add_argument("--repeats", type=int, default=MIN_STABLE_SAMPLES)
    gate.add_argument("--max-attempts", type=int, default=2)
    gate.add_argument(
        "--timeout-seconds", type=float, default=60.0,
        help="terminate and retry a benchmark subprocess that stops making progress",
    )
    gate.add_argument("--max-relative-spread", type=float, default=DEFAULT_MAX_RELATIVE_SPREAD)
    gate.add_argument("--runner", help="optional command prefix, parsed with shlex")
    gate.add_argument("--artifact-bundle")
    gate.set_defaults(function=run_gate)

    plan = sub.add_parser("plan", help="select stable long winners and write a portable plan")
    plan.add_argument("--space", required=True)
    plan.add_argument("--results", nargs="+", required=True)
    plan.add_argument("--output", required=True)
    plan.add_argument("--batches", required=True)
    plan.add_argument("--families", default="")
    plan.add_argument("--allow-partial", action="store_true")
    plan.set_defaults(function=command_plan)

    validate = sub.add_parser("validate", help="validate a plan on a receiving environment")
    validate.add_argument("--plan", required=True)
    validate.add_argument("--space")
    validate.add_argument("--model")
    validate.add_argument("--config")
    validate.add_argument("--architecture")
    validate.add_argument("--gpu-class")
    validate.add_argument("--streams", type=int)
    validate.add_argument("--device", type=int)
    validate.add_argument("--batches")
    validate.add_argument("--families", default="")
    validate.set_defaults(function=command_validate)

    apply = sub.add_parser("apply", help="render plan overrides for one or more batches")
    apply.add_argument("--plan", required=True)
    apply.add_argument("--batches")
    apply.add_argument("--families", default="")
    apply.add_argument("--device", type=int)
    apply.add_argument("--output")
    apply.set_defaults(function=command_apply)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        args.function(args)
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
        print(f"cuda_tactic_workflow: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
