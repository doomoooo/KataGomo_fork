#!/usr/bin/env python3
"""Device-free contract and opt-in probe for cuDNN FE OSS GEMM+SwiGLU.

The default path only reads installed package metadata and Python source files.
It never imports torch, cudnn, CUDA bindings, or CUTLASS.  GPU compilation and
execution are available only through the explicit ``--gpu-probe --allow-gpu``
pair and are intentionally separate from KataGo's production dispatch.
"""

from __future__ import annotations

import argparse
import datetime
from dataclasses import asdict, dataclass
import hashlib
import importlib.metadata
import json
import pathlib
import statistics
import sys
import time
from typing import Any

try:
    from sm103_b29_contract import (
        DEVELOPMENT_BATCH,
        DEVELOPMENT_ROWS,
        DEVELOPMENT_STREAMS,
        FIXED_BASELINE_BACKEND,
        FIXED_BASELINE_BINARY_SHA256,
        FIXED_BASELINE_CONFIG_SHA256,
        FIXED_BASELINE_NN_EVALS_PER_SEC,
        FIXED_BASELINE_SAMPLES,
        build_b29_development_manifest,
    )
    from sm103_contract import ACCELERATED_TARGET, COMPUTE_CAPABILITY
except ModuleNotFoundError:
    from python.sm103_b29_contract import (
        DEVELOPMENT_BATCH,
        DEVELOPMENT_ROWS,
        DEVELOPMENT_STREAMS,
        FIXED_BASELINE_BACKEND,
        FIXED_BASELINE_BINARY_SHA256,
        FIXED_BASELINE_CONFIG_SHA256,
        FIXED_BASELINE_NN_EVALS_PER_SEC,
        FIXED_BASELINE_SAMPLES,
        build_b29_development_manifest,
    )
    from python.sm103_contract import ACCELERATED_TARGET, COMPUTE_CAPABILITY


PROVIDER_DISTRIBUTION = "nvidia-cudnn-frontend"
PROVIDER_VERSION = "1.27.0"
CANDIDATE_ID = "cudnn-fe-1_27-oss-dense-gemm-swiglu-fp16-b29"

INPUT_CHANNELS = 384
OUTPUT_CHANNELS = 1152
PACKED_CHANNELS = OUTPUT_CHANNELS * 2
BATCH_DIMENSION = 1
FP16_BYTES = 2
MMA_TILER_MN = (128, 128)
CLUSTER_SHAPE_MN = (1, 1)
EPILOGUE_CHANNEL_GROUP = 32

SOURCE_IDENTITIES = {
    "cudnn/gemm/cutedsl/dense/swiglu/api.py": (
        "369173403e3b0c8fc845416d59123782a978d051b858de28bc883167f0e563cc"
    ),
    "cudnn/gemm/cutedsl/dense/swiglu/dense_gemm_persistent_swiglu.py": (
        "9eba76ddf512e9fe6074c7cd1b5fe25ebfe389655f8a22a4ba9b0ad5571b8652"
    ),
    "include/cudnn_frontend_version.h": (
        "1850d166698c9637ef7409aaffb5ec96c9e04f902065bbff459602a2efc0e0ab"
    ),
}


class CudnnOssCandidateError(ValueError):
    """Raised when the isolated candidate cannot be represented safely."""


def _plain_int(name: str, value: object) -> int:
    if type(value) is not int:  # noqa: E721 - bool must not pass as an integer
        raise CudnnOssCandidateError(f"{name} must be an integer, got {value!r}")
    return value


def _sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class SourceEvidence:
    relative_path: str
    expected_sha256: str
    installed_path: str | None
    actual_sha256: str | None

    @property
    def verified(self) -> bool:
        return self.actual_sha256 == self.expected_sha256

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["verified"] = self.verified
        return payload


@dataclass(frozen=True)
class ProviderEvidence:
    distribution: str
    expected_version: str
    installed_version: str | None
    sources: tuple[SourceEvidence, ...]
    errors: tuple[str, ...] = ()

    @property
    def verified(self) -> bool:
        return (
            self.installed_version == self.expected_version
            and len(self.sources) == len(SOURCE_IDENTITIES)
            and all(source.verified for source in self.sources)
            and not self.errors
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "distribution": self.distribution,
            "expected_version": self.expected_version,
            "installed_version": self.installed_version,
            "sources": [source.to_dict() for source in self.sources],
            "errors": list(self.errors),
            "verified": self.verified,
        }


def inspect_installed_provider() -> ProviderEvidence:
    """Verify the exact audited 1.27 wheel without importing its modules."""

    errors: list[str] = []
    try:
        distribution = importlib.metadata.distribution(PROVIDER_DISTRIBUTION)
    except importlib.metadata.PackageNotFoundError:
        return ProviderEvidence(
            distribution=PROVIDER_DISTRIBUTION,
            expected_version=PROVIDER_VERSION,
            installed_version=None,
            sources=(),
            errors=(f"missing distribution {PROVIDER_DISTRIBUTION}",),
        )

    installed_version = distribution.version
    if installed_version != PROVIDER_VERSION:
        errors.append(
            f"provider version mismatch: expected {PROVIDER_VERSION}, "
            f"got {installed_version}"
        )

    sources: list[SourceEvidence] = []
    for relative_path, expected_sha256 in SOURCE_IDENTITIES.items():
        installed_path = pathlib.Path(distribution.locate_file(relative_path))
        actual_sha256 = _sha256(installed_path) if installed_path.is_file() else None
        if actual_sha256 is None:
            errors.append(f"provider source is missing: {relative_path}")
        elif actual_sha256 != expected_sha256:
            errors.append(f"provider source identity changed: {relative_path}")
        sources.append(
            SourceEvidence(
                relative_path=relative_path,
                expected_sha256=expected_sha256,
                installed_path=str(installed_path.resolve())
                if installed_path.exists()
                else None,
                actual_sha256=actual_sha256,
            )
        )

    return ProviderEvidence(
        distribution=PROVIDER_DISTRIBUTION,
        expected_version=PROVIDER_VERSION,
        installed_version=installed_version,
        sources=tuple(sources),
        errors=tuple(errors),
    )


@dataclass(frozen=True)
class DenseSwiGLUProblem:
    batch: int = DEVELOPMENT_BATCH
    streams: int = DEVELOPMENT_STREAMS
    m: int = DEVELOPMENT_ROWS
    k: int = INPUT_CHANNELS
    n_packed: int = PACKED_CHANNELS
    n_output: int = OUTPUT_CHANNELS
    batch_dimension: int = BATCH_DIMENSION
    dtype: str = "float16"
    accumulator_dtype: str = "float32"
    ab12_dtype: str = "float16"
    mma_tiler_mn: tuple[int, int] = MMA_TILER_MN
    cluster_shape_mn: tuple[int, int] = CLUSTER_SHAPE_MN

    def __post_init__(self) -> None:
        for name in (
            "batch",
            "streams",
            "m",
            "k",
            "n_packed",
            "n_output",
            "batch_dimension",
        ):
            _plain_int(name, getattr(self, name))
        expected = (
            DEVELOPMENT_BATCH,
            DEVELOPMENT_STREAMS,
            DEVELOPMENT_ROWS,
            INPUT_CHANNELS,
            PACKED_CHANNELS,
            OUTPUT_CHANNELS,
            BATCH_DIMENSION,
        )
        actual = (
            self.batch,
            self.streams,
            self.m,
            self.k,
            self.n_packed,
            self.n_output,
            self.batch_dimension,
        )
        if actual != expected:
            raise CudnnOssCandidateError(
                f"candidate must remain exact B29 dense SwiGLU {expected!r}, got {actual!r}"
            )
        if self.n_packed != 2 * self.n_output:
            raise CudnnOssCandidateError("packed GEMM width must be exactly 2*Noutput")
        if self.dtype != "float16" or self.ab12_dtype != "float16":
            raise CudnnOssCandidateError("B29 candidate requires FP16 inputs and outputs")
        if self.accumulator_dtype != "float32":
            raise CudnnOssCandidateError("B29 candidate requires FP32 accumulation")
        if self.mma_tiler_mn != MMA_TILER_MN:
            raise CudnnOssCandidateError(
                f"unreviewed MMA tile {self.mma_tiler_mn!r}; expected {MMA_TILER_MN!r}"
            )
        if self.cluster_shape_mn != CLUSTER_SHAPE_MN:
            raise CudnnOssCandidateError(
                "the isolated control requires a 1CTA (1,1) cluster"
            )
        # The regular (non-block-scaled) kernel permits a ragged M tail.  Its
        # contiguous FP16 dimensions still need 16-byte / eight-element alignment.
        aligned_dimensions = (self.k, self.n_packed, self.n_output)
        if any(dimension % 8 != 0 for dimension in aligned_dimensions):
            raise CudnnOssCandidateError(
                "K, Npacked, and Noutput must be eight-element aligned for FP16 TMA"
            )
        if self.n_output % EPILOGUE_CHANNEL_GROUP != 0:
            raise CudnnOssCandidateError(
                "Noutput must be divisible by the audited 32-channel epilogue group"
            )

    @property
    def tensor_contract(self) -> dict[str, Any]:
        return {
            "a": {
                "meaning": "RMSNorm output in NHWC-flattened row-major form",
                "shape": [self.m, self.k, self.batch_dimension],
                "stride": [self.k, 1, self.m * self.k],
            },
            "b": {
                "meaning": "one-time repacked linearGate/linear1 weights",
                "shape": [self.n_packed, self.k, self.batch_dimension],
                "stride": [1, self.n_packed, self.n_packed * self.k],
            },
            "ab12": {
                "meaning": "mandatory full GEMM output scratch",
                "shape": [self.m, self.n_packed, self.batch_dimension],
                "stride": [self.n_packed, 1, self.m * self.n_packed],
                "bytes": self.m * self.n_packed * FP16_BYTES,
            },
            "c": {
                "meaning": "SwiGLU output in KataGo FFN channel order",
                "shape": [self.m, self.n_output, self.batch_dimension],
                "stride": [self.n_output, 1, self.m * self.n_output],
                "bytes": self.m * self.n_output * FP16_BYTES,
            },
        }

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["mma_tiler_mn"] = list(self.mma_tiler_mn)
        payload["cluster_shape_mn"] = list(self.cluster_shape_mn)
        payload["tensor_contract"] = self.tensor_contract
        return payload


def packed_weight_row(projection: str, output_channel: object) -> int:
    """Map KataGo's two weight matrices into the OSS kernel's N packing.

    The kernel consumes adjacent 32-channel accumulator subtiles as
    ``acc0 * SiLU(acc1)``.  KataGo computes
    ``SiLU(linear1) * linearGate``, so each packed pair must contain
    ``linearGate`` first and ``linear1`` second.
    """

    channel = _plain_int("output_channel", output_channel)
    if not 0 <= channel < OUTPUT_CHANNELS:
        raise CudnnOssCandidateError(
            f"output_channel must be in [0, {OUTPUT_CHANNELS}), got {channel}"
        )
    if projection not in ("linear_gate", "linear1"):
        raise CudnnOssCandidateError(
            "projection must be 'linear_gate' or 'linear1'"
        )
    group, offset = divmod(channel, EPILOGUE_CHANNEL_GROUP)
    pair_base = group * (2 * EPILOGUE_CHANNEL_GROUP)
    if projection == "linear1":
        pair_base += EPILOGUE_CHANNEL_GROUP
    return pair_base + offset


def _inspect_cpp_hook(repo_root: pathlib.Path) -> dict[str, Any]:
    backend = repo_root / "cpp" / "neuralnet" / "cudabackend.cpp"
    sm103_header = repo_root / "cpp" / "neuralnet" / "cudabackend_sm103.h"
    backend_text = backend.read_text(encoding="utf-8") if backend.is_file() else ""
    sm103_text = sm103_header.read_text(encoding="utf-8") if sm103_header.is_file() else ""
    reusable_call_site = all(
        token in backend_text
        for token in (
            "sm120FFNSingleGemm",
            "linear1.matBuf",
            "linearGate->matBuf",
            "wideFFNBuf.buf",
            "ffnBuf.buf",
        )
    )
    sm103_is_scaffold = (
        "future SM103 hook" in sm103_text and "no optimized kernels yet" in sm103_text
    )
    return {
        "integration_ready": False,
        "reusable_ffn_call_site_present": reusable_call_site,
        "sm103_backend_is_scaffold": sm103_is_scaffold,
        "existing_hook": "Sm120FFNSingleGemmFn (architecture-owned; not wired to SM103)",
        "minimum_bridge": [
            "export the CuTe-DSL compiled kernel through a stable C ABI/AOT launcher",
            "add an SM103-owned callback with the existing wide FFN argument contract",
            "pack linearGate then linear1 in 32-channel pairs once per model weight set",
            "reuse wideFFNBuf as mandatory AB12 scratch and ffnBuf as C",
            "accept only B29/R10469/K384/N2304->1152/FP16/sm_103a and fail closed otherwise",
        ],
    }


def build_candidate_manifest(
    provider: ProviderEvidence | None = None,
    baseline_path: pathlib.Path | None = None,
    repo_root: pathlib.Path | None = None,
) -> dict[str, Any]:
    """Build a CPU-only manifest for the exact isolated B29 candidate."""

    if provider is None:
        provider = inspect_installed_provider()
    if repo_root is None:
        repo_root = pathlib.Path(__file__).resolve().parents[1]
    problem = DenseSwiGLUProblem()
    anchor = build_b29_development_manifest(baseline_path)
    if (
        anchor.get("batch") != DEVELOPMENT_BATCH
        or anchor.get("streams") != DEVELOPMENT_STREAMS
        or anchor.get("rows") != DEVELOPMENT_ROWS
        or anchor.get("accelerated_target") != ACCELERATED_TARGET
        or not anchor.get("batch_selection_fixed")
        or anchor.get("production_ready")
    ):
        raise CudnnOssCandidateError("central B29 anchor no longer matches this candidate")

    eligible = provider.verified
    blockers = [] if eligible else list(provider.errors) or ["provider evidence failed"]
    return {
        "schema": 1,
        "kind": "katago-sm103-b29-isolated-kernel-candidate",
        "candidate_id": CANDIDATE_ID,
        "provider": provider.to_dict(),
        "anchor": anchor,
        "fixed_baseline_control": {
            "backend": FIXED_BASELINE_BACKEND,
            "binary_sha256": FIXED_BASELINE_BINARY_SHA256,
            "config_sha256": FIXED_BASELINE_CONFIG_SHA256,
            "nn_evals_per_sec_median": FIXED_BASELINE_NN_EVALS_PER_SEC,
            "sample_count": FIXED_BASELINE_SAMPLES,
            "measurement_iterations_per_sample_minimum": 1000,
        },
        "target": {
            "architecture": "sm103",
            "compute_capability": list(COMPUTE_CAPABILITY),
            "compile_target": ACCELERATED_TARGET,
        },
        "operation": {
            "family": "dense-gemm-swiglu",
            "api_module": "cudnn.gemm.cutedsl.dense.swiglu",
            "api_class": "GemmSwigluSm100",
            "api_wrapper": "gemm_swiglu_wrapper_sm100",
            "kernel_class": "PersistentDenseGemmKernel",
            "equation": "C = linearGate(A) * SiLU(linear1(A))",
            "kernel_accumulator_equation": "acc0 * SiLU(acc1)",
            "weight_pair_order": ["linear_gate", "linear1"],
            "weight_pair_channels": EPILOGUE_CHANNEL_GROUP,
            "quantized": False,
        },
        "problem": problem.to_dict(),
        "static_support": {
            "status": "cpu_contract_verified" if eligible else "blocked",
            "dense_shape_representable": True,
            "ragged_m_tail_supported": True,
            "m_tile_alignment_required": False,
            "provider_source_verified": provider.verified,
            "eligible_for_isolated_gpu_probe": eligible,
            "blockers": blockers,
            "note": (
                "upstream check_support touches torch.cuda; the CPU harness therefore "
                "validates its exact audited source instead of calling it"
            ),
        },
        "cpp_bridge": _inspect_cpp_hook(repo_root),
        "correctness": {
            "status": "not_run",
            "gpu_execution": False,
            "required_reference": "FP32 SiLU(linear1) * linearGate",
        },
        "benchmark": {
            "status": "not_run",
            "required_batch": DEVELOPMENT_BATCH,
            "required_streams": DEVELOPMENT_STREAMS,
        },
        "production_ready": False,
    }


def validate_candidate_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    """Fail closed on mutation or unsupported provider evidence."""

    if manifest.get("schema") != 1 or manifest.get("kind") != (
        "katago-sm103-b29-isolated-kernel-candidate"
    ):
        raise CudnnOssCandidateError("unexpected candidate manifest schema")
    if manifest.get("candidate_id") != CANDIDATE_ID:
        raise CudnnOssCandidateError("candidate identity changed")
    if manifest.get("production_ready") is not False:
        raise CudnnOssCandidateError("an isolated CPU contract cannot be production-ready")
    target = manifest.get("target")
    if not isinstance(target, dict) or target.get("compile_target") != ACCELERATED_TARGET:
        raise CudnnOssCandidateError("candidate target must remain exact sm_103a")
    baseline = manifest.get("fixed_baseline_control")
    expected_baseline = {
        "backend": FIXED_BASELINE_BACKEND,
        "binary_sha256": FIXED_BASELINE_BINARY_SHA256,
        "config_sha256": FIXED_BASELINE_CONFIG_SHA256,
        "nn_evals_per_sec_median": FIXED_BASELINE_NN_EVALS_PER_SEC,
        "sample_count": FIXED_BASELINE_SAMPLES,
        "measurement_iterations_per_sample_minimum": 1000,
    }
    if baseline != expected_baseline:
        raise CudnnOssCandidateError("fixed B29 baseline identity changed")
    problem = manifest.get("problem")
    if not isinstance(problem, dict):
        raise CudnnOssCandidateError("candidate problem is missing")
    DenseSwiGLUProblem(
        batch=problem.get("batch"),
        streams=problem.get("streams"),
        m=problem.get("m"),
        k=problem.get("k"),
        n_packed=problem.get("n_packed"),
        n_output=problem.get("n_output"),
        batch_dimension=problem.get("batch_dimension"),
        dtype=problem.get("dtype"),
        accumulator_dtype=problem.get("accumulator_dtype"),
        ab12_dtype=problem.get("ab12_dtype"),
        mma_tiler_mn=tuple(problem.get("mma_tiler_mn", ())),
        cluster_shape_mn=tuple(problem.get("cluster_shape_mn", ())),
    )
    provider = manifest.get("provider")
    if not isinstance(provider, dict) or provider.get("verified") is not True:
        raise CudnnOssCandidateError("exact cuDNN Frontend provider source is not verified")
    support = manifest.get("static_support")
    if (
        not isinstance(support, dict)
        or support.get("status") != "cpu_contract_verified"
        or support.get("eligible_for_isolated_gpu_probe") is not True
    ):
        raise CudnnOssCandidateError("dense OSS candidate is not eligible for a GPU probe")
    return manifest


def run_gpu_probe(
    *,
    allow_gpu: bool,
    device: int,
    seed: int = 20260818,
) -> dict[str, Any]:
    """Compile and correctness-check the exact candidate after explicit opt-in."""

    if not allow_gpu:
        raise CudnnOssCandidateError(
            "GPU probe requires the explicit --allow-gpu acknowledgement"
        )
    manifest = validate_candidate_manifest(build_candidate_manifest())

    # Deliberately lazy: the default manifest path never imports this stack.
    import torch
    from cudnn.gemm.cutedsl.dense.swiglu import gemm_swiglu_wrapper_sm100

    if type(device) is not int or device < 0:  # noqa: E721
        raise CudnnOssCandidateError("device must be a non-negative integer")
    if not torch.cuda.is_available():
        raise CudnnOssCandidateError("CUDA is unavailable")
    capability = tuple(torch.cuda.get_device_capability(device))
    if capability != COMPUTE_CAPABILITY:
        raise CudnnOssCandidateError(
            f"GPU probe requires exact SM103, got compute capability {capability!r}"
        )

    problem = DenseSwiGLUProblem()
    with torch.cuda.device(device):
        torch.manual_seed(seed)
        input_2d = (
            torch.randn(
                (problem.m, problem.k),
                dtype=torch.float16,
                device=f"cuda:{device}",
            )
            * 0.02
        )
        linear1 = (
            torch.randn(
                (problem.n_output, problem.k),
                dtype=torch.float16,
                device=f"cuda:{device}",
            )
            * 0.02
        )
        linear_gate = (
            torch.randn(
                (problem.n_output, problem.k),
                dtype=torch.float16,
                device=f"cuda:{device}",
            )
            * 0.02
        )

        a_tensor = torch.empty_strided(
            (problem.m, problem.k, 1),
            (problem.k, 1, problem.m * problem.k),
            dtype=torch.float16,
            device=f"cuda:{device}",
        )
        a_tensor[:, :, 0].copy_(input_2d)
        b_tensor = torch.empty_strided(
            (problem.n_packed, problem.k, 1),
            (1, problem.n_packed, problem.n_packed * problem.k),
            dtype=torch.float16,
            device=f"cuda:{device}",
        )
        for start in range(0, problem.n_output, EPILOGUE_CHANNEL_GROUP):
            stop = start + EPILOGUE_CHANNEL_GROUP
            gate_destination = packed_weight_row("linear_gate", start)
            linear1_destination = packed_weight_row("linear1", start)
            b_tensor[
                gate_destination : gate_destination + EPILOGUE_CHANNEL_GROUP,
                :,
                0,
            ].copy_(linear_gate[start:stop])
            b_tensor[
                linear1_destination : linear1_destination + EPILOGUE_CHANNEL_GROUP,
                :,
                0,
            ].copy_(linear1[start:stop])

        torch.cuda.synchronize(device)
        started = time.perf_counter()
        result = gemm_swiglu_wrapper_sm100(
            a_tensor,
            b_tensor,
            c_major="n",
            ab12_dtype=torch.float16,
            c_dtype=torch.float16,
            acc_dtype=torch.float32,
            mma_tiler_mn=MMA_TILER_MN,
            cluster_shape_mn=CLUSTER_SHAPE_MN,
        )
        torch.cuda.synchronize(device)
        elapsed_seconds = time.perf_counter() - started

        actual = result["c_tensor"][:, :, 0].float()
        linear1_projection = input_2d.float() @ linear1.float().transpose(0, 1)
        gate_projection = input_2d.float() @ linear_gate.float().transpose(0, 1)
        reference = torch.nn.functional.silu(linear1_projection) * gate_projection
        difference = (actual - reference).abs()
        max_abs = float(difference.max().item())
        max_rel = float(
            (difference / reference.abs().clamp_min(1.0e-3)).max().item()
        )
        passed = bool(torch.allclose(actual, reference, atol=2.0e-2, rtol=5.0e-2))
        if not passed:
            raise CudnnOssCandidateError(
                f"GPU correctness probe failed: max_abs={max_abs}, max_rel={max_rel}"
            )

    return {
        "schema": 1,
        "kind": "katago-sm103-b29-cudnn-oss-gpu-probe",
        "candidate_id": CANDIDATE_ID,
        "device": device,
        "compute_capability": list(capability),
        "elapsed_seconds_including_first_compile": elapsed_seconds,
        "max_abs_error": max_abs,
        "max_rel_error": max_rel,
        "correctness_passed": passed,
        "ab12_shape": list(result["ab12_tensor"].shape),
        "output_shape": list(result["c_tensor"].shape),
        "production_ready": False,
        "manifest": manifest,
    }


def _allocate_gpu_benchmark_inputs(torch: Any, device: int, seed: int) -> dict[str, Any]:
    """Allocate one exact B29 problem after the caller explicitly enabled GPU use."""

    problem = DenseSwiGLUProblem()
    torch.manual_seed(seed)
    input_2d = (
        torch.randn(
            (problem.m, problem.k),
            dtype=torch.float16,
            device=f"cuda:{device}",
        )
        * 0.02
    )
    linear1 = (
        torch.randn(
            (problem.n_output, problem.k),
            dtype=torch.float16,
            device=f"cuda:{device}",
        )
        * 0.02
    )
    linear_gate = (
        torch.randn(
            (problem.n_output, problem.k),
            dtype=torch.float16,
            device=f"cuda:{device}",
        )
        * 0.02
    )
    a_tensor = torch.empty_strided(
        (problem.m, problem.k, 1),
        (problem.k, 1, problem.m * problem.k),
        dtype=torch.float16,
        device=f"cuda:{device}",
    )
    a_tensor[:, :, 0].copy_(input_2d)
    b_tensor = torch.empty_strided(
        (problem.n_packed, problem.k, 1),
        (1, problem.n_packed, problem.n_packed * problem.k),
        dtype=torch.float16,
        device=f"cuda:{device}",
    )
    for start in range(0, problem.n_output, EPILOGUE_CHANNEL_GROUP):
        stop = start + EPILOGUE_CHANNEL_GROUP
        gate_destination = packed_weight_row("linear_gate", start)
        linear1_destination = packed_weight_row("linear1", start)
        b_tensor[
            gate_destination : gate_destination + EPILOGUE_CHANNEL_GROUP,
            :,
            0,
        ].copy_(linear_gate[start:stop])
        b_tensor[
            linear1_destination : linear1_destination + EPILOGUE_CHANNEL_GROUP,
            :,
            0,
        ].copy_(linear1[start:stop])
    return {
        "problem": problem,
        "input_2d": input_2d,
        "linear1": linear1,
        "linear_gate": linear_gate,
        "a_tensor": a_tensor,
        "b_tensor": b_tensor,
    }


def _timing_summary(
    elapsed_seconds: list[float],
    *,
    iterations: int,
    stream_count: int,
) -> dict[str, Any]:
    iteration_ms = [elapsed * 1000.0 / iterations for elapsed in elapsed_seconds]
    effective_call_ms = [value / stream_count for value in iteration_ms]
    median_iteration_ms = statistics.median(iteration_ms)
    median_effective_call_ms = statistics.median(effective_call_ms)
    relative_spread = (
        max(iteration_ms) - min(iteration_ms)
    ) / median_iteration_ms
    return {
        "host_wall_seconds_samples": elapsed_seconds,
        "milliseconds_per_iteration_samples": iteration_ms,
        "effective_milliseconds_per_call_samples": effective_call_ms,
        "median_concurrent_iteration_milliseconds": median_iteration_ms,
        # Each stream has one call in a concurrent iteration.  Its completion
        # latency is the paired wall time, not the throughput-amortized /S value.
        "median_stream_call_wall_milliseconds": median_iteration_ms,
        "median_effective_milliseconds_per_call": median_effective_call_ms,
        "calls_per_iteration": stream_count,
        "calls_per_second": 1000.0 / median_effective_call_ms,
        "relative_spread": relative_spread,
        "stable_within_ten_percent": relative_spread <= 0.10,
    }


def _measure_gpu_operation(
    torch: Any,
    *,
    device: int,
    stream_count: int,
    warmup: int,
    iterations: int,
    repeats: int,
    operation: Any,
) -> dict[str, Any]:
    streams = [torch.cuda.Stream(device=device) for _ in range(stream_count)]
    live_outputs: list[Any] = [None] * stream_count
    torch.cuda.synchronize(device)
    for _ in range(warmup):
        for index, stream in enumerate(streams):
            with torch.cuda.stream(stream):
                live_outputs[index] = operation(index)
    torch.cuda.synchronize(device)

    elapsed_samples: list[float] = []
    for _ in range(repeats):
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        for _ in range(iterations):
            for index, stream in enumerate(streams):
                with torch.cuda.stream(stream):
                    live_outputs[index] = operation(index)
        torch.cuda.synchronize(device)
        elapsed_samples.append(time.perf_counter() - started)
    return _timing_summary(
        elapsed_samples,
        iterations=iterations,
        stream_count=stream_count,
    )


def benchmark_gpu_candidate(
    *,
    allow_gpu: bool,
    device: int,
    warmup: int = 100,
    iterations: int = 1000,
    repeats: int = 5,
    seed: int = 20260818,
) -> dict[str, Any]:
    """Benchmark S1/S2 wrapper, preallocated API, and two PyTorch controls."""

    if not allow_gpu:
        raise CudnnOssCandidateError(
            "GPU benchmark requires the explicit --allow-gpu acknowledgement"
        )
    for name, value in (
        ("device", device),
        ("warmup", warmup),
        ("iterations", iterations),
        ("repeats", repeats),
        ("seed", seed),
    ):
        _plain_int(name, value)
    if device < 0 or warmup < 1 or iterations < 1 or repeats < 1:
        raise CudnnOssCandidateError(
            "device must be non-negative and timing counts must be positive"
        )

    # This performs the exact correctness gate and populates the wrapper's
    # compiled-kernel cache before timing cached calls.
    correctness = run_gpu_probe(allow_gpu=True, device=device, seed=seed)

    import torch
    from cudnn.gemm.cutedsl.dense.swiglu import (
        GemmSwigluSm100,
        gemm_swiglu_wrapper_sm100,
    )

    capability = tuple(torch.cuda.get_device_capability(device))
    if capability != COMPUTE_CAPABILITY:
        raise CudnnOssCandidateError(
            f"GPU benchmark requires exact SM103, got {capability!r}"
        )

    with torch.cuda.device(device):
        tensors = _allocate_gpu_benchmark_inputs(torch, device, seed)
        problem: DenseSwiGLUProblem = tensors["problem"]
        input_2d = tensors["input_2d"]
        linear1 = tensors["linear1"]
        linear_gate = tensors["linear_gate"]
        a_tensor = tensors["a_tensor"]
        b_tensor = tensors["b_tensor"]

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

        # Two buffers permit S2 execution without output races.  The direct API
        # accepts caller-owned AB12/C tensors even though the high-level wrapper
        # allocates both on every invocation.
        direct_ab12 = [allocate_ab12(), allocate_ab12()]
        direct_c = [allocate_c(), allocate_c()]
        direct_api = GemmSwigluSm100(
            sample_a=a_tensor,
            sample_b=b_tensor,
            sample_ab12=direct_ab12[0],
            sample_c=direct_c[0],
            alpha=1.0,
            acc_dtype=torch.float32,
            mma_tiler_mn=MMA_TILER_MN,
            cluster_shape_mn=CLUSTER_SHAPE_MN,
        )
        if not direct_api.check_support():
            raise CudnnOssCandidateError("direct preallocated API rejected B29")
        torch.cuda.synchronize(device)
        compile_started = time.perf_counter()
        direct_api.compile()
        torch.cuda.synchronize(device)
        direct_compile_seconds = time.perf_counter() - compile_started

        def cudnn_wrapper(_: int) -> Any:
            return gemm_swiglu_wrapper_sm100(
                a_tensor,
                b_tensor,
                c_major="n",
                ab12_dtype=torch.float16,
                c_dtype=torch.float16,
                acc_dtype=torch.float32,
                mma_tiler_mn=MMA_TILER_MN,
                cluster_shape_mn=CLUSTER_SHAPE_MN,
            )

        def cudnn_preallocated(index: int) -> Any:
            direct_api.execute(
                a_tensor=a_tensor,
                b_tensor=b_tensor,
                ab12_tensor=direct_ab12[index],
                c_tensor=direct_c[index],
                alpha=1.0,
            )
            return direct_ab12[index], direct_c[index]

        def torch_two_gemm(_: int) -> Any:
            linear1_projection = input_2d @ linear1.transpose(0, 1)
            gate_projection = input_2d @ linear_gate.transpose(0, 1)
            return torch.nn.functional.silu(linear1_projection) * gate_projection

        def torch_packed_gemm(_: int) -> Any:
            packed_projection = input_2d @ b_tensor[:, :, 0].transpose(0, 1)
            grouped = packed_projection.view(
                problem.m,
                problem.n_output // EPILOGUE_CHANNEL_GROUP,
                2 * EPILOGUE_CHANNEL_GROUP,
            )
            gate_projection = grouped[:, :, :EPILOGUE_CHANNEL_GROUP]
            linear1_projection = grouped[:, :, EPILOGUE_CHANNEL_GROUP:]
            return (
                torch.nn.functional.silu(linear1_projection) * gate_projection
            ).reshape(problem.m, problem.n_output)

        operations = {
            "cudnn_oss_cached_wrapper": cudnn_wrapper,
            "cudnn_oss_preallocated_direct_api": cudnn_preallocated,
            "torch_two_gemm_swiglu": torch_two_gemm,
            "torch_packed_gemm_reorder_swiglu": torch_packed_gemm,
        }
        timings: dict[str, Any] = {}
        for stream_count in (1, 2):
            mode = f"s{stream_count}"
            timings[mode] = {
                name: _measure_gpu_operation(
                    torch,
                    device=device,
                    stream_count=stream_count,
                    warmup=warmup,
                    iterations=iterations,
                    repeats=repeats,
                    operation=operation,
                )
                for name, operation in operations.items()
            }
            control_ms = timings[mode]["torch_two_gemm_swiglu"][
                "median_stream_call_wall_milliseconds"
            ]
            for candidate_name in (
                "cudnn_oss_cached_wrapper",
                "cudnn_oss_preallocated_direct_api",
            ):
                candidate_ms = timings[mode][candidate_name][
                    "median_stream_call_wall_milliseconds"
                ]
                timings[mode][candidate_name]["speedup_vs_torch_two_gemm"] = (
                    control_ms / candidate_ms
                )
                timings[mode][candidate_name][
                    "latency_reduction_vs_torch_two_gemm_fraction"
                ] = 1.0 - candidate_ms / control_ms

    ab12_bytes = problem.m * problem.n_packed * FP16_BYTES
    c_bytes = problem.m * problem.n_output * FP16_BYTES
    return {
        "schema": 1,
        "kind": "katago-sm103-b29-cudnn-oss-timing",
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z"),
        "candidate_id": CANDIDATE_ID,
        "device": device,
        "device_name": torch.cuda.get_device_name(device),
        "compute_capability": list(capability),
        "software": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "torch_cuda_build": torch.version.cuda,
            "torch_loaded_cudnn": torch.backends.cudnn.version(),
            "nvidia_cudnn_frontend": importlib.metadata.version(
                PROVIDER_DISTRIBUTION
            ),
            "nvidia_cutlass_dsl": importlib.metadata.version(
                "nvidia-cutlass-dsl"
            ),
        },
        "problem": problem.to_dict(),
        "method": {
            "clock": "steady-state amortized host time.perf_counter with device synchronization around each repeat",
            "warmup_iterations": warmup,
            "timed_iterations": iterations,
            "repeats": repeats,
            "s1_calls_per_iteration": 1,
            "s2_calls_per_iteration": 2,
            "s2_execution": "two independent torch CUDA streams with shared read-only inputs and per-stream outputs",
            "s2_latency_definition": (
                "median_stream_call_wall_milliseconds is the two-stream concurrent "
                "iteration wall time; median_effective_milliseconds_per_call divides "
                "that time by two only for throughput accounting"
            ),
            "scope": (
                "cached wrapper timing includes Python dispatch and per-call torch.empty_strided; "
                "preallocated timing includes direct API execute only"
            ),
        },
        "compile_and_correctness": {
            "wrapper_first_compile_and_execute_seconds": correctness[
                "elapsed_seconds_including_first_compile"
            ],
            "direct_api_compile_seconds_after_wrapper_cache": direct_compile_seconds,
            "max_abs_error": correctness["max_abs_error"],
            "max_rel_error": correctness["max_rel_error"],
            "passed": correctness["correctness_passed"],
        },
        "allocation_and_write_contract": {
            "wrapper_allocates_outputs_per_call": True,
            "direct_api_accepts_preallocated_outputs": True,
            "ab12_is_mandatory": True,
            "ab12_bytes_per_call": ab12_bytes,
            "c_bytes_per_call": c_bytes,
            "total_explicit_output_bytes_per_call": ab12_bytes + c_bytes,
            "s2_preallocated_output_bytes": 2 * (ab12_bytes + c_bytes),
            "ab12_is_written_to_gmem": True,
            "source_evidence": (
                "dense_gemm_persistent_swiglu.py stores both AB12 accumulator "
                "subtiles at audited source lines 1062-1110"
            ),
            "katago_consumes_ab12": False,
        },
        "timings": timings,
        "production_ready": False,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=pathlib.Path)
    parser.add_argument("--output", type=pathlib.Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--gpu-probe", action="store_true")
    mode.add_argument("--gpu-benchmark", action="store_true")
    parser.add_argument("--allow-gpu", action="store_true")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--repeats", type=int, default=5)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.allow_gpu and not (args.gpu_probe or args.gpu_benchmark):
        raise CudnnOssCandidateError(
            "--allow-gpu is valid only with --gpu-probe or --gpu-benchmark"
        )
    if args.gpu_benchmark:
        payload = benchmark_gpu_candidate(
            allow_gpu=args.allow_gpu,
            device=args.device,
            warmup=args.warmup,
            iterations=args.iterations,
            repeats=args.repeats,
        )
    elif args.gpu_probe:
        payload = run_gpu_probe(allow_gpu=args.allow_gpu, device=args.device)
    else:
        payload = build_candidate_manifest(baseline_path=args.baseline)
        validate_candidate_manifest(payload)
    serialized = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.write_text(serialized, encoding="utf-8")
    sys.stdout.write(serialized)
    return 0


__all__ = (
    "CANDIDATE_ID",
    "CLUSTER_SHAPE_MN",
    "CudnnOssCandidateError",
    "DenseSwiGLUProblem",
    "EPILOGUE_CHANNEL_GROUP",
    "MMA_TILER_MN",
    "PACKED_CHANNELS",
    "PROVIDER_DISTRIBUTION",
    "PROVIDER_VERSION",
    "ProviderEvidence",
    "SOURCE_IDENTITIES",
    "SourceEvidence",
    "benchmark_gpu_candidate",
    "build_candidate_manifest",
    "build_parser",
    "inspect_installed_provider",
    "packed_weight_row",
    "run_gpu_probe",
    "validate_candidate_manifest",
)


if __name__ == "__main__":
    raise SystemExit(main())
