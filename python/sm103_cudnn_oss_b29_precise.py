#!/usr/bin/env python3
"""B29 cuDNN OSS SwiGLU Variant B: FP16 projections plus precise math.

Variant B is cumulative on the exact audited Variant-A derivative.  It keeps
tcgen05 FP32 accumulation, the explicit FP16 projection round-trip, tiling,
cluster/grid scheduling, AB12/C stores, and the native C ABI unchanged.  The
only new factor is the SwiGLU math sequence: non-fast ``cute.math.exp`` and
precise FP32 division replace fast exp2 plus ``rcp_approx``.

The default path is CPU-only.  GPU compilation is owned by
``sm103_cudnn_oss_b29_export.py`` and standalone execution additionally
requires ``--aot-benchmark --allow-gpu``.
"""

from __future__ import annotations

import argparse
import ctypes
import datetime
from dataclasses import asdict, dataclass
import hashlib
import importlib.util
import json
import pathlib
import statistics
import sys
from types import ModuleType
from typing import Any

try:
    import sm103_cudnn_oss_b29 as upstream_candidate
    import sm103_cudnn_oss_b29_roundtrip as variant_a
except ModuleNotFoundError:
    from python import sm103_cudnn_oss_b29 as upstream_candidate
    from python import sm103_cudnn_oss_b29_roundtrip as variant_a


CANDIDATE_ID = (
    "cudnn-fe-1_27-oss-dense-gemm-swiglu-proj-fp16-roundtrip-"
    "precise-math-b29"
)
NUMERIC_SEMANTICS_SELECTOR = "projection-fp16-roundtrip-precise-math"
NUMERIC_SEMANTICS_ID = (
    "projection-fp16-rne-roundtrip-nonfast-exp-precise-div-swiglu"
)
EXPECTED_VARIANT_A_DERIVATIVE_SHA256 = (
    "99247c64d70a5f0b14ff75c08ba8d28fde31f159248e1e86c934cec6152777bc"
)
DERIVATIVE_FILENAME = "dense_gemm_persistent_swiglu_variant_b.py"
DERIVATIVE_PROVENANCE_FILENAME = "variant-b-provenance.json"

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

VARIANT_B_PRECISE_MATH_BLOCK = """\
                    # KataGo B29 Variant B: keep Variant A's rounded FP16
                    # projections, but use non-fast exp and precise FP32
                    # division for the SwiGLU sigmoid.
                    gate_denominator = (
                        1.0 + cute.math.exp(-acc_vec1, fastmath=False)
                    ).to(self.acc_dtype)
                    gate = (acc_vec1 / gate_denominator).to(self.acc_dtype)
"""


class CudnnOssPreciseError(RuntimeError):
    """Raised when Variant B provenance, execution, or timing is invalid."""


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


PATCH_SPEC_SHA256 = _sha256_bytes(
    (VARIANT_A_FAST_MATH_BLOCK + "\0" + VARIANT_B_PRECISE_MATH_BLOCK).encode(
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


def derive_variant_b_source(variant_a_source: bytes) -> bytes:
    actual = _sha256_bytes(variant_a_source)
    if actual != EXPECTED_VARIANT_A_DERIVATIVE_SHA256:
        raise CudnnOssPreciseError(
            "Variant-A derivative SHA-256 mismatch: expected "
            f"{EXPECTED_VARIANT_A_DERIVATIVE_SHA256}, got {actual}"
        )
    try:
        source = variant_a_source.decode("utf-8")
    except UnicodeDecodeError as error:
        raise CudnnOssPreciseError("Variant-A derivative is not UTF-8") from error
    count = source.count(VARIANT_A_FAST_MATH_BLOCK)
    if count != 1:
        raise CudnnOssPreciseError(
            f"Variant-B precise-math context must occur exactly once, got {count}"
        )
    source = source.replace(
        VARIANT_A_FAST_MATH_BLOCK, VARIANT_B_PRECISE_MATH_BLOCK, 1
    )
    required = (
        "acc_vec0_ab12 = acc_vec0.to(self.ab12_dtype)",
        "acc_vec1_ab12 = acc_vec1.to(self.ab12_dtype)",
        "cute.math.exp(-acc_vec1, fastmath=False)",
        "gate = (acc_vec1 / gate_denominator).to(self.acc_dtype)",
        "acc_vec_c = (acc_vec0 * gate).to(self.c_dtype)",
        "tRS_rAB12.store(acc_vec0)",
        "tRS_rAB12_1.store(acc_vec1)",
        "tRS_rC.store(acc_vec_c)",
    )
    missing = [token for token in required if token not in source]
    if missing:
        raise CudnnOssPreciseError(
            "Variant-B derivation lost required cumulative operations: "
            + ", ".join(missing)
        )
    forbidden = (
        "cute.math.exp2(-1 * acc_vec1 * LOG2_E, True)",
        "cute.arch.rcp_approx(res[i])",
    )
    present = [token for token in forbidden if token in source]
    if present:
        raise CudnnOssPreciseError(
            "Variant-B derivation retained fast math: " + ", ".join(present)
        )
    return source.encode("utf-8")


def inspect_derivative(
    provider: upstream_candidate.ProviderEvidence | None = None,
) -> tuple[bytes, DerivativeEvidence]:
    variant_a_source, parent = variant_a.inspect_derivative(provider)
    if parent.derivative_sha256 != EXPECTED_VARIANT_A_DERIVATIVE_SHA256:
        raise CudnnOssPreciseError("audited Variant-A evidence changed")
    derivative = derive_variant_b_source(variant_a_source)
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
    for _ in pathlib.PurePosixPath(
        variant_a.UPSTREAM_KERNEL_RELATIVE_PATH
    ).parts:
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
        raise CudnnOssPreciseError("unsafe derivative output directory")
    site_packages = _site_packages_root(evidence)
    if resolved == site_packages or site_packages in resolved.parents:
        raise CudnnOssPreciseError(
            "Variant-B derivative output must remain outside site-packages"
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
            raise CudnnOssPreciseError(
                f"refusing to overwrite mismatched Variant-B artifact: {path}"
            )
        if not path.exists():
            path.write_bytes(expected)
    if _sha256(pathlib.Path(evidence.upstream_installed_path)) != (
        evidence.upstream_sha256
    ):
        raise CudnnOssPreciseError("site-packages source changed during derivation")
    return source_path, provenance_path, evidence


def load_derivative_kernel_class(
    output_dir: pathlib.Path,
) -> tuple[type[Any], DerivativeEvidence, pathlib.Path, pathlib.Path]:
    source_path, provenance_path, evidence = materialize_derivative(output_dir)
    module_name = "katago_sm103_cudnn_oss_swiglu_variant_b"
    spec = importlib.util.spec_from_file_location(module_name, source_path)
    if spec is None or spec.loader is None:
        raise CudnnOssPreciseError("could not construct Variant-B module spec")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    kernel_class = getattr(module, "PersistentDenseGemmKernel", None)
    if not isinstance(kernel_class, type):
        raise CudnnOssPreciseError("Variant-B derivative kernel class is missing")
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
            "non-fast cute.math.exp and precise arith.divf in FP32"
        ),
        "ab12": "same rounded FP16 projections; AB12 retained unchanged",
        "output": "FP16",
        "parent_candidate_id": variant_a.CANDIDATE_ID,
        "changed_factors_from_variant_a": [
            "fast exp2 -> non-fast exp",
            "rcp_approx multiply -> precise FP32 divide",
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
    parent["kind"] = "katago-sm103-b29-isolated-kernel-candidate-variant-b"
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
        "katago-sm103-b29-isolated-kernel-candidate-variant-b"
    ):
        raise CudnnOssPreciseError("unexpected Variant-B manifest kind")
    if manifest.get("candidate_id") != CANDIDATE_ID:
        raise CudnnOssPreciseError("Variant-B candidate identity changed")
    provider = upstream_candidate.inspect_installed_provider()
    expected = build_candidate_manifest(provider=provider)
    if manifest != expected:
        raise CudnnOssPreciseError("full Variant-B manifest identity changed")
    if manifest.get("production_ready") is not False:
        raise CudnnOssPreciseError("isolated Variant B cannot be production-ready")
    return manifest


projection_fp16_roundtrip_reference = variant_a.projection_fp16_roundtrip_reference
strengthen_probe_signal = variant_a.strengthen_probe_signal
build_gpu_correctness_summary = variant_a.build_gpu_correctness_summary
CORRECTNESS_LIMITS = variant_a.CORRECTNESS_LIMITS


def authenticate_aot_library(library_path: pathlib.Path) -> dict[str, Any]:
    resolved = library_path.resolve()
    if not resolved.is_file():
        raise CudnnOssPreciseError(f"Variant-B AOT library is missing: {resolved}")
    manifest_path = resolved.parent / "aot-manifest.json"
    if not manifest_path.is_file():
        raise CudnnOssPreciseError(
            f"Variant-B sibling manifest is missing: {manifest_path}"
        )
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CudnnOssPreciseError("Variant-B AOT manifest is unreadable") from error
    _, derivative = inspect_derivative()
    fresh_manifest = validate_candidate_manifest(build_candidate_manifest())

    def artifact_matches(record: Any, expected_path: pathlib.Path) -> bool:
        if not isinstance(record, dict) or not expected_path.is_file():
            return False
        try:
            recorded = pathlib.Path(record.get("path", "")).resolve()
        except (OSError, RuntimeError, TypeError, ValueError):
            return False
        return (
            recorded == expected_path.resolve()
            and record.get("sha256") == _sha256(expected_path)
            and record.get("bytes") == expected_path.stat().st_size
        )

    artifacts = manifest.get("artifacts")
    bridge = (
        artifacts.get("bridge_shared_library")
        if isinstance(artifacts, dict)
        else None
    )
    derivative_manifest = manifest.get("derivative")
    derivative_artifacts = (
        derivative_manifest.get("artifacts")
        if isinstance(derivative_manifest, dict)
        else None
    )
    source_path = resolved.parent / DERIVATIVE_FILENAME
    provenance_path = resolved.parent / DERIVATIVE_PROVENANCE_FILENAME
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
    try:
        provenance_payload = json.loads(provenance_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        provenance_payload = None
    launch_validation = manifest.get("launch_validation")
    tight = (
        launch_validation.get("tight_correctness")
        if isinstance(launch_validation, dict)
        else None
    )
    checks = {
        "kind": manifest.get("kind")
        == "katago-sm103-b29-cudnn-oss-aot-artifact",
        "candidate_id": manifest.get("candidate_id") == CANDIDATE_ID,
        "numeric_semantics_selector": manifest.get("numeric_semantics_selector")
        == NUMERIC_SEMANTICS_SELECTOR,
        "numeric_semantics": manifest.get("numeric_semantics")
        == numeric_semantics(),
        "compile_target": manifest.get("compile_target") == "sm_103a",
        "compile_options": manifest.get("compile_options")
        == ["--gpu-arch=sm_103a"],
        "kernel_manifest_provider": manifest.get("kernel_manifest_provider")
        == fresh_manifest["provider"],
        "derivative_evidence": isinstance(derivative_manifest, dict)
        and derivative_manifest.get("evidence") == derivative.to_dict(),
        "derivative_source": artifact_matches(source_record, source_path)
        and _sha256(source_path) == derivative.derivative_sha256,
        "derivative_provenance": artifact_matches(
            provenance_record, provenance_path
        )
        and provenance_payload == derivative.to_dict(),
        "shared_library_sibling": artifact_matches(bridge, resolved),
        "tight_launch_validation": isinstance(launch_validation, dict)
        and launch_validation.get("status") == "passed"
        and isinstance(tight, dict)
        and tight.get("passed") is True,
        "no_timing_identity": "timing_record" not in manifest,
        "nonproduction": manifest.get("production_ready") is False,
    }
    if not all(checks.values()):
        failed = [name for name, passed in checks.items() if not passed]
        raise CudnnOssPreciseError(
            "Variant-B AOT authentication failed: " + ", ".join(failed)
        )
    return {
        "manifest_path": str(manifest_path),
        "manifest_sha256": _sha256(manifest_path),
        "library_path": str(resolved),
        "library_sha256": _sha256(resolved),
        "checks": checks,
    }


def _timing_summary(
    samples: list[float], iterations: int, streams: int
) -> dict[str, Any]:
    iteration_ms = [sample * 1000.0 / iterations for sample in samples]
    median = statistics.median(iteration_ms)
    return {
        "cuda_event_seconds_samples": samples,
        "milliseconds_per_concurrent_iteration_samples": iteration_ms,
        "median_stream_call_wall_milliseconds": median,
        "median_effective_milliseconds_per_call": median / streams,
        "calls_per_iteration": streams,
        "calls_per_second": 1000.0 * streams / median,
        "relative_spread": (max(iteration_ms) - min(iteration_ms)) / median,
    }


def benchmark_aot(
    *,
    allow_gpu: bool,
    device: int,
    library_path: pathlib.Path,
    warmup: int = 20000,
    iterations: int = 1000,
    repeats: int = 5,
    seed: int = 20260818,
) -> dict[str, Any]:
    if not allow_gpu:
        raise CudnnOssPreciseError(
            "Variant-B AOT benchmark requires explicit --allow-gpu"
        )
    for name, value in (
        ("device", device),
        ("warmup", warmup),
        ("iterations", iterations),
        ("repeats", repeats),
        ("seed", seed),
    ):
        if type(value) is not int:  # noqa: E721
            raise CudnnOssPreciseError(f"{name} must be an integer")
    if device < 0 or warmup < 1 or iterations < 1 or repeats < 1:
        raise CudnnOssPreciseError("invalid Variant-B device or timing count")
    validate_candidate_manifest(build_candidate_manifest())
    authentication = authenticate_aot_library(library_path)

    import torch

    if tuple(torch.cuda.get_device_capability(device)) != (
        upstream_candidate.COMPUTE_CAPABILITY
    ):
        raise CudnnOssPreciseError("Variant-B benchmark requires exact SM103")
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

    ab12 = [allocate_ab12(), allocate_ab12()]
    outputs = [allocate_c(), allocate_c()]
    torch.cuda.synchronize(device)
    library = variant_a._load_native_library(library_path)
    status = ctypes.c_int32(-999)
    context = library.katagoCudnnOssB29Create(device, ctypes.byref(status))
    if not context or status.value != 0:
        raise CudnnOssPreciseError(
            f"Variant-B context creation failed with status {status.value}"
        )

    def launch(index: int) -> tuple[Any, Any]:
        stream = torch.cuda.current_stream(device)
        launch_status = library.katagoCudnnOssB29Launch(
            context,
            ctypes.c_void_p(tensors["a_tensor"].data_ptr()),
            ctypes.c_void_p(tensors["b_tensor"].data_ptr()),
            ctypes.c_void_p(ab12[index].data_ptr()),
            ctypes.c_void_p(outputs[index].data_ptr()),
            ctypes.c_float(1.0),
            ctypes.c_void_p(stream.cuda_stream),
            problem.m,
            problem.k,
            problem.n_packed,
            problem.n_output,
            1,
        )
        if launch_status != 0:
            raise CudnnOssPreciseError(
                f"Variant-B launch failed with status {launch_status}"
            )
        return ab12[index], outputs[index]

    try:
        ab12[0].fill_(float("nan"))
        outputs[0].fill_(float("nan"))
        launch(0)
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
        correctness = build_gpu_correctness_summary(
            torch,
            actual_output=outputs[0][:, :, 0],
            reference_output=reference,
            actual_ab12=ab12[0][:, :, 0],
            reference_ab12=packed_reference,
        )

        def measure(stream_count: int) -> dict[str, Any]:
            streams = [torch.cuda.Stream(device=device) for _ in range(stream_count)]
            coordinator = torch.cuda.Stream(device=device)
            live: list[Any] = [None] * stream_count
            for _ in range(warmup):
                for index, stream in enumerate(streams):
                    with torch.cuda.stream(stream):
                        live[index] = launch(index)
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
                            live[0] = launch(0)
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
                                live[index] = launch(index)
                    for event, stream in zip(done, streams, strict=True):
                        event.record(stream)
                    with torch.cuda.stream(coordinator):
                        for event in done:
                            coordinator.wait_event(event)
                        end.record()
                end.synchronize()
                samples.append(start.elapsed_time(end) / 1000.0)
            return _timing_summary(samples, iterations, stream_count)

        timings = {f"s{count}": measure(count) for count in (1, 2)}
    finally:
        try:
            torch.cuda.synchronize(device)
        finally:
            library.katagoCudnnOssB29Destroy(context)

    return {
        "schema": 1,
        "kind": "katago-sm103-b29-cudnn-oss-variant-b-aot-timing",
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
            "allocation": "all A/B/AB12/C buffers preallocated",
            "s2": "two independent streams and per-stream AB12/C buffers",
            "single_dso": True,
        },
        "correctness": {
            "status": "passed",
            "reference_semantics": NUMERIC_SEMANTICS_ID,
            "same_seed": seed,
            "signal_scaling": signal_scaling,
            "variant_b_vs_precise_half_boundary_reference": correctness,
        },
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
    parser.add_argument("--warmup", type=int, default=20000)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260818)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.aot_benchmark:
        if args.library is None:
            raise CudnnOssPreciseError("--aot-benchmark requires --library")
        if args.materialize_dir is not None:
            raise CudnnOssPreciseError(
                "--materialize-dir is invalid with --aot-benchmark"
            )
        payload = benchmark_aot(
            allow_gpu=args.allow_gpu,
            device=args.device,
            library_path=args.library,
            warmup=args.warmup,
            iterations=args.iterations,
            repeats=args.repeats,
            seed=args.seed,
        )
    else:
        if args.allow_gpu or args.library is not None:
            raise CudnnOssPreciseError(
                "--allow-gpu/--library require --aot-benchmark"
            )
        payload = validate_candidate_manifest(build_candidate_manifest())
        if args.materialize_dir is not None:
            source, provenance, evidence = materialize_derivative(
                args.materialize_dir
            )
            payload["materialized_derivative"] = {
                **evidence.to_dict(),
                "source_path": str(source),
                "provenance_path": str(provenance),
            }
    serialized = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.write_text(serialized, encoding="utf-8")
    sys.stdout.write(serialized)
    return 0


__all__ = (
    "CANDIDATE_ID",
    "CORRECTNESS_LIMITS",
    "CudnnOssPreciseError",
    "DERIVATIVE_FILENAME",
    "DERIVATIVE_PROVENANCE_FILENAME",
    "EXPECTED_VARIANT_A_DERIVATIVE_SHA256",
    "NUMERIC_SEMANTICS_ID",
    "NUMERIC_SEMANTICS_SELECTOR",
    "PATCH_SPEC_SHA256",
    "authenticate_aot_library",
    "benchmark_aot",
    "build_candidate_manifest",
    "build_gpu_correctness_summary",
    "derive_variant_b_source",
    "inspect_derivative",
    "load_derivative_kernel_class",
    "materialize_derivative",
    "numeric_semantics",
    "projection_fp16_roundtrip_reference",
    "strengthen_probe_signal",
    "validate_candidate_manifest",
)


if __name__ == "__main__":
    raise SystemExit(main())
