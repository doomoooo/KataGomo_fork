#!/usr/bin/env python3
"""Cross-device/batch tactic scanning and portable plan generation.

This file deliberately lives outside the CUDA runtime. It is the SM89
RTX 4090/4080 workflow boundary that final-migration can consume later:

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
import shlex
import shutil
import statistics
import subprocess
import sys
import tempfile
from collections.abc import Iterable, Sequence
from typing import Any


SCHEMA = 1
SPACE_KIND = "portable-tactic-search-space"
PLAN_KIND = "portable-tactic-plan"
RESULT_KIND = "portable-tactic-scan"
ARTIFACT_BUNDLE_KIND = "portable-tactic-artifact-bundle"
FAMILIES = (
    "wide_qkv",
    "wide_ffn",
    "fused_residual",
    "rmsnorm",
    "exact_mask",
    "qkv_rope",
    "fa4",
    "dual_ffn",
    "linear2",
    "outproj",
    "preconv",
    "postconv_bn",
    "pointwise",
    "l2",
    "initial_conv",
    "initial_global",
    "policy_p1",
    "wide_head",
    "head_bn",
    "value_terminal",
)
SM89_RUNTIME_CONFIG_KEYS = frozenset({
    "cudaFusedFFNAotTacticSm89",
    "cudaLinear2AotTacticSm89",
    "cudaPersistingL2HitRatioSm89",
    "cudaPlainQKVVariantSm89",
    "cudaRoPEBatchGroupSm89",
    "cudaUseDualGemmSwiGLUHalf2TanhSm89",
    "cudaUseDualGemmSwiGLUSm89",
    "cudaUseExactMaskElisionSm89",
    "cudaUseFlashAttentionBoth16Sm89",
    "cudaUseFlashAttentionSm89",
    "cudaUseFusedPolicyP1",
    "cudaUseFusedQKRoPE",
    "cudaUseFusedResidual",
    "cudaUseFusedValueTerminalSm89",
    "cudaUseHeadBNHalfToFloat",
    "cudaUseInitialConvFrontend",
    "cudaUseInitialGlobalMatMulAdd",
    "cudaUseLinear2GemmSm89",
    "cudaUseLinear2PostBNSiluSm89",
    "cudaUseOutProjGemmSm89",
    "cudaUsePersistingL2Inner",
    "cudaUsePersistingL2Trunk",
    "cudaUsePostConvBNSiluSm89",
    "cudaUsePostConvGemmSm89",
    "cudaUsePreConvGemmSm89",
    "cudaUsePrecomputedQKRoPESm89",
    "cudaUseQKVRoPEGemmSm89",
    "cudaUseRMSNormOpt",
    "cudaUseScaleBiasSiluVec4C384Sm89",
    "cudaUseScaleBiasSiluVec8Sm89",
    "cudaUseSplitQKVRoPEGemmSm89",
    "cudaUseWideFFN",
    "cudaUseWideHeadProjection",
    "cudaUseWideQKV",
})
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
            "cudaUseDualGemmSwiGLUSm89": True,
            "cudaUseDualGemmSwiGLUHalf2TanhSm89": False,
        },
        "linear2": {
            "cudaUseFusedResidual": True,
            "cudaUseLinear2GemmSm89": True,
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
            "cudaUseDualGemmSwiGLUSm89": False,
            "cudaUseDualGemmSwiGLUHalf2TanhSm89": False,
        },
        "linear2": {
            "cudaUseLinear2GemmSm89": False,
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
        result.extend([
            _config_candidate(
                family, batch, "dual-compiled-exp",
                cudaFusedFFNAotTacticSm89="disabled",
                cudaUseWideFFN=True,
                cudaUseDualGemmSwiGLUSm89=True,
                cudaUseDualGemmSwiGLUHalf2TanhSm89=False,
            ),
            _config_candidate(
                family, batch, "dual-compiled-half2-tanh",
                cudaFusedFFNAotTacticSm89="disabled",
                cudaUseWideFFN=True,
                cudaUseDualGemmSwiGLUSm89=True,
                cudaUseDualGemmSwiGLUHalf2TanhSm89=True,
            ),
        ])
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
        return result
    if family == "linear2":
        result.extend([
            _config_candidate(
                family, batch, "linear2-compiled-residual",
                cudaLinear2AotTacticSm89="disabled",
                cudaUseFusedResidual=True,
                cudaUseLinear2GemmSm89=True,
                cudaUseLinear2PostBNSiluSm89=False,
            ),
            _config_candidate(
                family, batch, "linear2-compiled-postbn",
                cudaLinear2AotTacticSm89="disabled",
                cudaUseFusedResidual=True,
                cudaUseLinear2GemmSm89=True,
                cudaUseLinear2PostBNSiluSm89=True,
            ),
        ])
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
        return result
    raise ValueError(f"{family} has no SM89 GEMM tactic registry")


def _history_candidates(architecture: str, family: str, batch: int) -> list[dict[str, object]]:
    if family in ("dual_ffn", "linear2"):
        return _gemm_candidates(architecture, family, batch)
    toggle_keys = {
        "wide_qkv": "cudaUseWideQKV",
        "wide_ffn": "cudaUseWideFFN",
        "fused_residual": "cudaUseFusedResidual",
        "rmsnorm": "cudaUseRMSNormOpt",
        "exact_mask": "cudaUseExactMaskElisionSm89",
        "outproj": "cudaUseOutProjGemmSm89",
        "preconv": "cudaUsePreConvGemmSm89",
        "initial_conv": "cudaUseInitialConvFrontend",
        "initial_global": "cudaUseInitialGlobalMatMulAdd",
        "policy_p1": "cudaUseFusedPolicyP1",
        "head_bn": "cudaUseHeadBNHalfToFloat",
        "value_terminal": "cudaUseFusedValueTerminalSm89",
    }
    if family in toggle_keys:
        key = toggle_keys[family]
        on_config: dict[str, object] = {key: True}
        if family == "outproj":
            on_config["cudaUseFusedResidual"] = True
        return [
            _config_candidate(family, batch, f"{family}-off", **{key: False}),
            _config_candidate(family, batch, f"{family}-on", **on_config),
        ]
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
        ]
        for group in (2, 3, 4, 7, 13):
            values.append(_config_candidate(
                family, batch, f"qkv-rope-group-{group}", **{
                    **reset, "cudaUseFusedQKRoPE": True,
                    "cudaRoPEBatchGroupSm89": group,
                },
            ))
        for variant in (0, 1, 2):
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
        # This branch links the historical M64/N96 FA4 implementation. Other
        # tile shapes must not be advertised until they have distinct linked
        # launchers. Both accumulator policies are real runtime choices.
        return [
            _config_candidate(
                family, batch, "fa4-off",
                cudaUseFlashAttentionSm89=False,
                cudaUseFlashAttentionBoth16Sm89=False,
            ),
            _config_candidate(
                family, batch, "fa4-n96-fp32",
                cudaUseFlashAttentionSm89=True,
                cudaUseFlashAttentionBoth16Sm89=False,
            ),
            _config_candidate(
                family, batch, "fa4-n96-both16",
                cudaUseFlashAttentionSm89=True,
                cudaUseFlashAttentionBoth16Sm89=True,
            ),
        ]
    if family == "postconv_bn":
        return [
            _config_candidate(
                family, batch, "postconv-off",
                cudaUsePostConvGemmSm89=False,
                cudaUsePostConvBNSiluSm89=False,
            ),
            _config_candidate(
                family, batch, "postconv-gemm",
                cudaUsePostConvGemmSm89=True,
                cudaUsePostConvBNSiluSm89=False,
            ),
            _config_candidate(
                family, batch, "postconv-gemm-bn-silu",
                cudaUsePostConvGemmSm89=True,
                cudaUsePostConvBNSiluSm89=True,
            ),
        ]
    if family == "pointwise":
        return [
            _config_candidate(
                family, batch, "pointwise-off",
                cudaUseScaleBiasSiluVec8Sm89=False,
                cudaUseScaleBiasSiluVec4C384Sm89=False,
            ),
            _config_candidate(
                family, batch, "pointwise-c768-vec8",
                cudaUseScaleBiasSiluVec8Sm89=True,
                cudaUseScaleBiasSiluVec4C384Sm89=False,
            ),
            _config_candidate(
                family, batch, "pointwise-c384-vec4",
                cudaUseScaleBiasSiluVec8Sm89=False,
                cudaUseScaleBiasSiluVec4C384Sm89=True,
            ),
            _config_candidate(
                family, batch, "pointwise-c768-vec8-c384-vec4",
                cudaUseScaleBiasSiluVec8Sm89=True,
                cudaUseScaleBiasSiluVec4C384Sm89=True,
            ),
        ]
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
        return [
            _config_candidate(
                family, batch, "wide-head-off", cudaUseWideHeadProjection=False,
            ),
            _config_candidate(
                family, batch, "wide-head-on",
                cudaUseFusedPolicyP1=True,
                cudaUseWideHeadProjection=True,
            ),
        ]
    raise ValueError(f"unsupported tactic family: {family}")


def _sm89_candidates(family: str, batch: int) -> list[dict[str, object]]:
    # Every coordinate must be allowed to retain the state inherited from the
    # accepted config and earlier family winners. Without this explicit no-op,
    # a family whose whole local neighborhood regresses is forced to accept
    # the least-bad regression (observed at B15 qkv_rope).
    return [
        _config_candidate(family, batch, f"{family}-keep-incumbent"),
        *_history_candidates("sm89", family, batch),
    ]


def default_candidates(architecture: str, family: str, batch: int) -> list[dict[str, object]]:
    if architecture == "sm89":
        return _sm89_candidates(family, batch)
    raise ValueError(f"unsupported architecture: {architecture}")


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
            if family not in FAMILIES or not isinstance(value, dict):
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
    batch_payloads: list[dict[str, object]] = []
    for batch in sorted(set(int(item) for item in batches)):
        if batch < 1:
            raise ValueError("batch values must be positive")
        batch_space: dict[str, object] = {"batch": batch, "tokens": batch * 361}
        for family in FAMILIES:
            values = default_candidates(architecture, family, batch)
            for entry in extra:
                entry_batches = entry["batches"]
                applies = "all" in entry_batches or batch in entry_batches
                if applies and entry["family"] == family:
                    values.append(entry["candidate"])
            values = deduplicate_candidates(values)
            for value in values:
                unknown = sorted(set(candidate_config(family, value)) - SM89_RUNTIME_CONFIG_KEYS)
                if unknown:
                    raise ValueError(
                        f"candidate uses unparsed SM89 config keys: {family}/B{batch}/"
                        f"{value.get('id')}: {unknown}"
                    )
            batch_space[family] = values
        batch_payloads.append(batch_space)
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
        "families": list(FAMILIES),
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
        },
        "history_recipe": {
            "sources": [
                "/workspace/results/4090/HISTORY.md",
                "/workspace/4090-optimization-portability.md",
            ],
            "execution_order": list(FAMILIES),
            "search_semantics": (
                "accepted-history-seeded coordinate search with accumulated "
                "winners and a non-regressing incumbent at every stage"
            ),
            "fixed_gemm_aot": {
                "families": ["dual_ffn", "linear2"],
                "threadblock_m": [64, 128],
                "threadblock_n": [64, 128],
                "pipeline_stages": [3, 4, 5],
                "policy": "pruned neighborhood, not full Cartesian product",
            },
            "flash_attention": {
                "linked_tile_m": 64,
                "linked_tile_n": 96,
                "warps": [4],
                "accumulation": ["fp32", "both16"],
                "runtime_num_sms": True,
            },
            "qkv_rope": {
                "paths": ["official", "fused", "precomputed", "gemm-epilogue", "gemm-split"],
                "batch_groups": [2, 3, 4, 7, 13],
                "plain_variants": [0, 1, 2],
            },
            "pointwise": {"c768_vector_width": 8, "c384_vector_width": 4},
            "policy_p1": {"block_xy": [96, 5]},
            "persisting_l2": {
                "scopes": ["off", "trunk", "inner", "trunk-inner"],
                "hit_ratio": [0.5, 0.75, 1.0],
            },
            "initial_conv": {"engine": 45, "tile_size": 0, "stages": 2},
        },
        "batches": batch_payloads,
        "candidate_files": [str(pathlib.Path(path).resolve()) for path in extra_paths],
    }


def make_generation_plan(
    space_path: pathlib.Path,
    *,
    phase: str = "full",
    families: Sequence[str] = FAMILIES,
) -> dict[str, object]:
    space = read_json(space_path)
    if space.get("schema") != SCHEMA or space.get("kind") != SPACE_KIND:
        raise ValueError("generation-plan requires a portable tactic search space")
    if phase not in ("seed", "full"):
        raise ValueError("generation phase must be seed or full")
    requested = list(dict.fromkeys(families))
    if not requested or any(family not in FAMILIES for family in requested):
        raise ValueError(f"invalid generation families: {requested}")
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
        "kind": "portable-tactic-generation-plan",
        "generated_utc": utc_now(),
        "phase": phase,
        "complete_history_coverage": phase == "full",
        "eligible_for_whole_graph_scan": phase == "full",
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


def build_artifact_bundle(
    space_path: pathlib.Path,
    binary: pathlib.Path,
    manifest_paths: Sequence[pathlib.Path],
) -> dict[str, object]:
    """Combine generated family manifests and prove their launchers are linked."""
    space = read_json(space_path)
    if space.get("schema") != SCHEMA or space.get("kind") != SPACE_KIND:
        raise ValueError("artifact-bundle requires a portable tactic search space")
    if not binary.is_file():
        raise ValueError(f"linked binary does not exist: {binary}")
    space_sha256 = sha256_file(space_path)
    expected = {
        (family, batch, str(value["id"])): value
        for batch in sorted(space_batches(space))
        for family in FAMILIES
        for value in candidate_map(space, family, batch).values()
        if value.get("requires_artifact")
    }
    if not expected:
        raise ValueError("search space contains no generated AOT candidates")
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
        if family not in FAMILIES or not manifest.get("complete"):
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
            if not isinstance(compile_command, list) or "-arch=sm_89" not in compile_command:
                raise ValueError(f"generated artifact lacks an SM89 compile command: {key}")
            correctness = metadata.get("correctness_against_torch")
            if not isinstance(correctness, dict):
                raise ValueError(f"generated artifact lacks correctness evidence: {key}")
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
                "correctness": {"status": "passed", **correctness},
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
        raise ValueError("--artifact-bundle is not a portable tactic artifact bundle")
    if not bundle.get("complete_history_coverage", False):
        raise ValueError("artifact bundle is not a complete full-history generation")
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
    if family not in FAMILIES:
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


def candidate_config(family: str, value: dict[str, object]) -> dict[str, object]:
    config = value.get("config", value.get("config_overrides", {}))
    if config is None:
        return {}
    if not isinstance(config, dict):
        raise ValueError(f"candidate {value.get('id')} has a non-object config")
    return {str(key): item for key, item in config.items()}


def tactic_overrides(family: str, value: dict[str, object]) -> dict[str, object]:
    # ``family`` is currently used for validation/documentation.  Keeping the
    # conversion in one place lets final-migration add a new AOT registry key
    # without changing scan, plan, or apply semantics.
    if family not in FAMILIES:
        raise ValueError(f"unsupported tactic family: {family}")
    config = dict(candidate_config(family, value))
    # Older materialized spaces predate this explicit dependency. Every
    # residual linear2 implementation (compiled or generated) accumulates into
    # trunkBuf and therefore must select the fused-residual control flow.
    if family == "linear2" and value.get("id") != "linear2-fallback":
        config["cudaUseFusedResidual"] = True
    return config


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
    # Keep architecture explicit in manifests, but this branch only owns SM89.
    if architecture not in ARCHITECTURES:
        raise ValueError(f"unsupported architecture: {architecture}")
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
    result = parse_key_values(extra)
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
        prefixes = ("CMAKE_CUDA_", "CMAKE_CXX_", "CUDA_", "CUDNN_", "SM89_")
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
        "path": str(path.resolve()),
        "name": path.name,
        "sha256": sha256_file(path),
        "family": payload.get("family"),
        "rows": len(payload.get("rows", [])) if isinstance(payload.get("rows"), list) else 0,
        "finished_utc": payload.get("finished_utc"),
    }


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
        raise ValueError("plan requires a portable-tactic-search-space schema-1 file")
    architecture = str(space.get("architecture"))
    gpu_class = str(space.get("gpu_class"))
    if architecture not in ARCHITECTURES:
        raise ValueError(f"unknown architecture in search space: {architecture}")
    validate_gpu_class(architecture, gpu_class)
    requested_families = list(dict.fromkeys(families))
    if not requested_families or any(family not in FAMILIES for family in requested_families):
        raise ValueError(f"invalid tactic families: {requested_families}")
    required_families = [
        str(item) for item in space.get("families", FAMILIES)
    ]
    unscanned_families = sorted(set(required_families) - set(requested_families))
    requested_batches = sorted(set(int(item) for item in batches))
    expected_streams = int(space.get("streams", -1))
    rows_by_key: dict[tuple[str, int, str], dict[str, object]] = {}
    result_metadata: list[dict[str, object]] = []
    provenance: list[dict[str, object]] = []
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
        if isinstance(payload.get("provenance"), dict):
            provenance.append(payload["provenance"])
        payload_family = payload.get("family")
        if payload_family is not None and payload_family not in requested_families:
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
            observed = {
                candidate_id: rows_by_key[(family, batch, candidate_id)]
                for candidate_id in expected
                if (family, batch, candidate_id) in rows_by_key and
                rows_by_key[(family, batch, candidate_id)].get("status") == "measured"
            }
            stable: list[tuple[float, str, dict[str, object]]] = []
            for candidate_id, row in observed.items():
                metric = stable_metric(row)
                if metric is not None:
                    stable.append((metric, candidate_id, row))
            stable.sort(key=lambda item: (-item[0], item[1]))
            missing_ids = sorted(set(expected) - set(observed))
            history_winners = [
                (candidate_id, row) for candidate_id, row in observed.items()
                if row.get("history_stage_winner") is True
            ]
            history_evidence_error = None
            if len(history_winners) == 1:
                winner_id, winner_row = history_winners[0]
                incumbent_id = f"{family}-keep-incumbent"
                accepted_change = winner_id != incumbent_id
                recorded_gain = winner_row.get(
                    "history_improvement_fraction_vs_incumbent"
                )
                minimum_gain = winner_row.get("history_min_improvement_fraction")
                if (
                    winner_row.get("history_incumbent_candidate_id") != incumbent_id or
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
                "history_stage_winner_count": len(history_winners),
                "history_evidence_error": history_evidence_error,
            }
            history_error = (
                history_evidence_error
                if history_evidence_error is not None else
                None if len(history_winners) == 1 else
                f"expected one long-stable accumulated-history winner, got {len(history_winners)}"
            )
            if missing_ids or not observed or history_error:
                missing.append({
                    "family": family,
                    "batch": batch,
                    "missing_candidate_ids": missing_ids,
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
                "correctness": row.get("correctness"),
                "binary_sha256": row.get("binary_sha256"),
                "command": row.get("command"),
                "source_result": pathlib.Path(str(row["_source_result"])).name,
                "source_result_path_at_scan": row["_source_result"],
            }
        selected_families[family] = {
            "space_sha256": sha256_file(space_path),
            "space_path_at_scan": str(space_path.resolve()),
            "batches": family_batches,
        }
        coverage[family] = family_coverage
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
            "correctness": row.get("correctness"),
            "command": row.get("command"),
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
    return {
        "schema": SCHEMA,
        "kind": PLAN_KIND,
        "plan_id": f"{architecture}-{gpu_class}-{plan_hash[:16]}",
        "plan_sha256": plan_hash,
        "generated_utc": utc_now(),
        "status": "complete_long_stable" if ready else "partial_or_unstable",
        "ready_for_scan_bypass": ready,
        "production_ready": production_ready,
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
            "provenance_snapshots": provenance,
            "notes": [
                "Environment and tool versions are evidence and reproducibility metadata, not strict equality gates.",
                "Discovery may be partitioned across identical GPU-class devices; every local ordinal is recorded.",
                "The receiving device must match architecture/GPU class and stream topology; device ordinal may change.",
                "Run correctness replay before production use unless every selected row carries correctness.status=passed.",
            ],
        },
        "apply": {
            "topology": topology_overrides(architecture, int(target_device or 0), expected_streams, space),
            "per_batch_tactic_overrides": {
                str(batch): render_plan_overrides(
                    selected_families, batch, include_topology=False
                )
                for batch in requested_batches
            },
        },
    }


def render_plan_overrides(
    families: dict[str, object], batch: int, *, include_topology: bool = False,
    topology: dict[str, object] | None = None,
) -> str:
    values: dict[str, object] = {}
    if include_topology and topology:
        values.update(topology)
    for family in FAMILIES:
        family_payload = families.get(family)
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
        raise ValueError(f"unsupported portable tactic plan: {path}")
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
    families: Sequence[str] = FAMILIES,
    device_properties: dict[str, object] | None = None,
) -> dict[str, object]:
    if plan.get("schema") != SCHEMA or plan.get("kind") != PLAN_KIND:
        raise ValueError("unsupported portable tactic plan")
    target = plan.get("target", {})
    if not isinstance(target, dict):
        raise ValueError("plan has no target")
    plan_arch = str(target.get("architecture"))
    plan_gpu = str(target.get("gpu_class"))
    if plan_arch not in ARCHITECTURES:
        raise ValueError(f"plan has unknown architecture: {plan_arch}")
    validate_gpu_class(plan_arch, plan_gpu)
    if architecture and architecture != plan_arch:
        raise ValueError(f"plan architecture mismatch: {plan_arch} != {architecture}")
    if gpu_class and gpu_class != plan_gpu:
        raise ValueError(f"plan GPU class mismatch: {plan_gpu} != {gpu_class}")
    if streams is not None and int(target.get("streams", -1)) != streams:
        raise ValueError("plan stream topology mismatch")
    if target.get("compute_capability") != ARCHITECTURES[plan_arch]["compute_capability"]:
        raise ValueError("plan compute capability is inconsistent with its architecture")
    if device_properties is not None:
        actual_compute_capability = cuda_compute_capability(device_properties)
        if actual_compute_capability != target.get("compute_capability"):
            raise ValueError(
                "CUDA-reported receiver capability does not match the plan: "
                f"{actual_compute_capability} != {target.get('compute_capability')}"
            )
    if target.get("fixed_board") != [19, 19]:
        raise ValueError("portable tactic plans currently require 19x19")
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
        if space_path is not None:
            expected_sha = sha256_file(space_path.resolve())
            for family in families:
                family_payload = plan.get("families", {}).get(family, {})
                if isinstance(family_payload, dict) and family_payload.get("space_sha256") not in (None, expected_sha):
                    raise ValueError(f"plan search-space hash mismatch for {family}")
    selected_batches = sorted(set(int(item) for item in (batches or plan.get("batches", []))))
    checked: dict[str, object] = {}
    for family in families:
        if family not in FAMILIES:
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


def run_scan(args: argparse.Namespace) -> None:
    space_path = pathlib.Path(args.space).resolve()
    space = read_json(space_path)
    if space.get("schema") != SCHEMA or space.get("kind") != SPACE_KIND:
        raise ValueError("scan requires a portable-tactic-search-space file")
    architecture = str(space["architecture"])
    if args.architecture and args.architecture != architecture:
        raise ValueError("--architecture does not match the search space")
    gpu_class = str(space["gpu_class"])
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
    if not families or any(item not in FAMILIES for item in families):
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
    provenance = collect_provenance(
        pathlib.Path(__file__).resolve().parents[1], binary=binary, config=config, model=model,
        device=device,
    ) if not args.dry_run else {"schema": 1, "captured_utc": utc_now(), "dry_run": True}
    raw_dir.mkdir(parents=True, exist_ok=True)
    for batch in batches:
        # Seed coordinate search from the accepted configuration file. The
        # previous all-off reset destroyed interactions between already-
        # accepted history stages and made the entire curve regress. Every
        # family still scans its explicit off control and all real variants;
        # after the final family the accumulated overrides are self-contained.
        accumulated = parse_key_values(args.override_config)
        # Make every exact-batch implementation build for the batch currently
        # being scanned.
        accumulated["nnMaxBatchSize"] = batch
        # Each subprocess measures one exact batch. Compiling lazy SDPA graphs
        # for 1..B on every candidate only adds setup time; the target-B graph
        # is still compiled before benchmarknn's own warmup/timed passes.
        accumulated["cudaWarmupOnlyMaxBatchSize"] = True
        accumulated["cudaDisableWarmup"] = True
        for family_index, family in enumerate(families):
            base_overrides = config_string(accumulated)
            stage_rows: list[dict[str, object]] = []
            for value in candidate_map(space, family, batch).values():
                key = (family, batch, str(value["id"]))
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
                    for attempt in range(args.max_attempts):
                        timed_out = False
                        try:
                            completed = subprocess.run(
                                command, text=True, capture_output=True, check=False,
                                timeout=args.timeout_seconds,
                            )
                        except subprocess.TimeoutExpired as exc:
                            timed_out = True
                            completed = subprocess.CompletedProcess(
                                command, 124, _timeout_text(exc.stdout),
                                _timeout_text(exc.stderr),
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
                        _write_scan_payload(
                            output, space_path, space, architecture, gpu_class, device,
                            streams, args, started, provenance,
                            artifact_bundle_metadata, rows, device_properties,
                            implementation_identity,
                        )
                        raise RuntimeError(
                            f"benchmark failed for {family}/B{batch}/{value['id']} "
                            f"after {args.max_attempts} attempts; see {stderr_path}"
                        )
                    record = _parse_benchmark_record(completed.stdout)
                    throughput = result_metric(record)
                    samples.append(throughput)
                    run_records.append({
                        "repeat": repeat, "throughput": throughput,
                        "benchmark": record, "stdout": str(stdout_path),
                        "stderr": str(stderr_path), "attempts": attempt_records,
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
            accumulated.update(tactic_overrides(family, winner_candidate))
            winner["history_accumulated_overrides"] = config_string(accumulated)
        # Atomic batch-level checkpoint. On an unexpected interruption only
        # the current batch is repeated; explicit candidate failures still
        # checkpoint immediately above with their logs and return code.
        _write_scan_payload(
            output, space_path, space, architecture, gpu_class, device,
            streams, args, started, provenance,
            artifact_bundle_metadata, rows, device_properties,
            implementation_identity,
        )
    _write_scan_payload(
        output, space_path, space, architecture, gpu_class, device,
        streams, args, started, provenance, artifact_bundle_metadata, rows,
        device_properties, implementation_identity,
    )
    print(json.dumps({"output": str(output), "rows": len(rows), "dry_run": args.dry_run}))


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
    final_rows: dict[int, dict[str, object]] = {}
    for batch in batches:
        for family in FAMILIES:
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
                if by_key[(family, batch, candidate_id)].get("history_stage_winner") is True
            ]
            if len(winners) != 1:
                raise ValueError(
                    f"discovery has {len(winners)} history winners for {family}/B{batch}"
                )
            winner_id = str(winners[0]["candidate_id"])
            if expected[winner_id].get("requires_artifact"):
                selected_aot.append((family, batch, winner_id))
        final = [
            row for row in discovery_rows
            if int(row.get("batch", -1)) == batch and
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
            for attempt in range(args.max_attempts):
                timed_out = False
                try:
                    completed = subprocess.run(
                        command, text=True, capture_output=True, check=False,
                        timeout=args.timeout_seconds,
                    )
                except subprocess.TimeoutExpired as exc:
                    timed_out = True
                    completed = subprocess.CompletedProcess(
                        command, 124, _timeout_text(exc.stdout),
                        _timeout_text(exc.stderr),
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
            record = _parse_benchmark_record(completed.stdout)
            throughput = result_metric(record)
            samples.append(throughput)
            run_records.append({
                "repeat": repeat, "throughput": throughput,
                "benchmark": record, "stdout": str(stdout_path),
                "stderr": str(stderr_path), "attempts": attempt_records,
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
    args.families = ",".join(FAMILIES)
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
    }
    certified = 0
    rows = payload.get("rows", [])
    if not isinstance(rows, list):
        raise ValueError("gate result rows are not a list")
    for row in rows:
        if not isinstance(row, dict) or row.get("history_long_gate") is not True:
            continue
        batch = int(row.get("batch", -1))
        report_path = reports.get(batch)
        if report_path is None:
            raise ValueError(f"missing --comparison for gate B{batch}")
        report = read_json(report_path)
        policy = report.get("policy", {})
        value = report.get("value", {})
        score = report.get("score", {})
        ownership = report.get("ownership", {})
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
        }
        status = "passed" if all(checks.values()) else "failed"
        row["correctness"] = {
            "status": status,
            "kind": "8192-row all-head FP32-reference replay",
            "comparison": str(report_path),
            "comparison_sha256": sha256_file(report_path),
            "thresholds": thresholds,
            "checks": checks,
            "metrics": {
                "policy_top1_vs_reference": policy.get("top1VsReference"),
                "weighted_p0loss_delta": p0_delta,
                "policy_probability_rmse": policy.get("probabilityRmse"),
                "value_outcome_rmse": value.get("outcomeRmse"),
                "score_mean_rmse": score.get("meanRmse"),
                "ownership_sigmoid_rmse": ownership.get("sigmoidRmse"),
            },
        }
        row["finished_utc"] = utc_now()
        if status != "passed":
            failed = ", ".join(name for name, passed in checks.items() if not passed)
            raise ValueError(f"accuracy certification failed for B{batch}: {failed}")
        certified += 1
    if certified != len(reports):
        raise ValueError(
            f"certified {certified} gate rows from {len(reports)} comparison reports"
        )
    payload["finished_utc"] = utc_now()
    payload["accuracy_certification"] = {
        "status": "passed", "thresholds": thresholds, "batches": sorted(reports),
    }
    output = pathlib.Path(args.output).resolve()
    write_json(output, payload)
    print(json.dumps({"output": str(output), "certified_batches": certified}))


def command_plan(args: argparse.Namespace) -> None:
    families = [item.strip() for item in args.families.split(",") if item.strip()]
    payload = build_plan(
        [pathlib.Path(item).resolve() for item in args.results],
        pathlib.Path(args.space).resolve(), families,
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
    topology = {
        "numNNServerThreadsPerModel": streams,
        **{f"cudaDeviceToUseThread{i}": device for i in range(streams)},
    }
    result: dict[str, object] = {
        "schema": 1,
        "kind": "portable-tactic-application",
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
        tactic_values: dict[str, object] = {}
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
    space = sub.add_parser("space", help="materialize an SM89 device/batch search space")
    space.add_argument("--architecture", choices=tuple(ARCHITECTURES))
    space.add_argument("--gpu-class")
    space.add_argument("--device", type=int, default=0)
    space.add_argument("--batches", default="1-32")
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
    generation.add_argument("--families", default=",".join(FAMILIES))
    generation.add_argument("--output")
    generation.set_defaults(function=command_generation_plan)

    artifact = sub.add_parser(
        "artifact-bundle",
        help="prove generated AOT sources/objects are present in the linked binary",
    )
    artifact.add_argument("--space", required=True)
    artifact.add_argument("--binary", required=True)
    artifact.add_argument("--manifests", nargs="+", required=True)
    artifact.add_argument("--output", required=True)
    artifact.set_defaults(function=command_artifact_bundle)

    certify = sub.add_parser(
        "certify", help="attach an accepted 8192-row FP32 replay to long-gate rows",
    )
    certify.add_argument("--gate", required=True)
    certify.add_argument(
        "--comparison", action="append", required=True, metavar="BATCH=PATH",
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
    scan.add_argument("--families", default=",".join(FAMILIES))
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
    plan.add_argument("--families", default=",".join(FAMILIES))
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
    validate.add_argument("--families", default=",".join(FAMILIES))
    validate.set_defaults(function=command_validate)

    apply = sub.add_parser("apply", help="render plan overrides for one or more batches")
    apply.add_argument("--plan", required=True)
    apply.add_argument("--batches")
    apply.add_argument("--families", default=",".join(FAMILIES))
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
        print(f"portable_tactic_workflow: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
