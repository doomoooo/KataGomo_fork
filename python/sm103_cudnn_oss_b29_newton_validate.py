#!/usr/bin/env python3
"""Validate and stably time the isolated SM103 B29 Newton1 AOT kernel.

The default invocation is device-free and only prints the CLI help.  A CUDA
launch requires ``--benchmark --allow-gpu``.  The resulting JSON authenticates
the Variant-C artifact, repeats the tight output/AB12 check, records S1/S2
coordinator-event timing, and audits the embedded cubin for the precise-divide
slow path that made Variant B unsuitable for the dual-stream graph.
"""

from __future__ import annotations

import argparse
import ctypes
import datetime
import hashlib
import json
import pathlib
import re
import statistics
import subprocess
import sys
import tempfile
from typing import Any

try:
    import sm103_cudnn_oss_b29 as base
    import sm103_cudnn_oss_b29_export as exporter
    import sm103_cudnn_oss_b29_newton as variant
    import sm103_cudnn_oss_b29_roundtrip as variant_a
except ModuleNotFoundError:
    from python import sm103_cudnn_oss_b29 as base
    from python import sm103_cudnn_oss_b29_export as exporter
    from python import sm103_cudnn_oss_b29_newton as variant
    from python import sm103_cudnn_oss_b29_roundtrip as variant_a


class NewtonValidationError(RuntimeError):
    """Raised when an artifact, SASS audit, launch, or timing is invalid."""


def _sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_record(path: pathlib.Path) -> dict[str, Any]:
    resolved = path.resolve()
    return {
        "path": str(resolved),
        "sha256": _sha256(resolved),
        "bytes": resolved.stat().st_size,
    }


def authenticate_artifact(
    library_path: pathlib.Path, object_path: pathlib.Path
) -> dict[str, Any]:
    """Fail closed on the complete identities needed by the validator."""

    library = library_path.resolve()
    object_file = object_path.resolve()
    manifest_path = library.parent / "aot-manifest.json"
    for label, path in (
        ("library", library),
        ("object", object_file),
        ("manifest", manifest_path),
    ):
        if not path.is_file():
            raise NewtonValidationError(f"Variant-C {label} is missing: {path}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise NewtonValidationError("Variant-C manifest is unreadable") from error

    derivative_source = library.parent / variant.DERIVATIVE_FILENAME
    derivative_provenance = (
        library.parent / variant.DERIVATIVE_PROVENANCE_FILENAME
    )
    _, fresh_derivative = variant.inspect_derivative()
    fresh_manifest = variant.validate_candidate_manifest(
        variant.build_candidate_manifest()
    )

    def artifact_matches(record: Any, expected: pathlib.Path) -> bool:
        if not isinstance(record, dict) or not expected.is_file():
            return False
        try:
            recorded_path = pathlib.Path(record.get("path", "")).resolve()
        except (OSError, RuntimeError, TypeError, ValueError):
            return False
        return (
            recorded_path == expected.resolve()
            and record.get("sha256") == _sha256(expected)
            and record.get("bytes") == expected.stat().st_size
        )

    artifacts = manifest.get("artifacts")
    derivative = manifest.get("derivative")
    derivative_artifacts = (
        derivative.get("artifacts") if isinstance(derivative, dict) else None
    )
    launch = manifest.get("launch_validation")
    tight = launch.get("tight_correctness") if isinstance(launch, dict) else None
    checks = {
        "kind": manifest.get("kind")
        == "katago-sm103-b29-cudnn-oss-aot-artifact",
        "candidate_id": manifest.get("candidate_id") == variant.CANDIDATE_ID,
        "selector": manifest.get("numeric_semantics_selector")
        == variant.NUMERIC_SEMANTICS_SELECTOR,
        "numeric_semantics": manifest.get("numeric_semantics")
        == variant.numeric_semantics(),
        "compile_target": manifest.get("compile_target") == "sm_103a",
        "compile_options": manifest.get("compile_options")
        == ["--gpu-arch=sm_103a"],
        "provider": manifest.get("kernel_manifest_provider")
        == fresh_manifest["provider"],
        "library": isinstance(artifacts, dict)
        and artifact_matches(artifacts.get("bridge_shared_library"), library),
        "object": isinstance(artifacts, dict)
        and artifact_matches(artifacts.get("object"), object_file),
        "derivative_evidence": isinstance(derivative, dict)
        and derivative.get("evidence") == fresh_derivative.to_dict(),
        "derivative_source": isinstance(derivative_artifacts, dict)
        and artifact_matches(derivative_artifacts.get("source"), derivative_source),
        "derivative_provenance": isinstance(derivative_artifacts, dict)
        and artifact_matches(
            derivative_artifacts.get("provenance"), derivative_provenance
        ),
        "tight_launch_validation": isinstance(launch, dict)
        and launch.get("status") == "passed"
        and isinstance(tight, dict)
        and tight.get("passed") is True,
        "nonproduction": manifest.get("production_ready") is False,
    }
    if not all(checks.values()):
        failed = [name for name, passed in checks.items() if not passed]
        raise NewtonValidationError(
            "Variant-C artifact authentication failed: " + ", ".join(failed)
        )
    return {
        "manifest": _file_record(manifest_path),
        "library": _file_record(library),
        "object": _file_record(object_file),
        "checks": checks,
    }


def _sass_counts(sass: str) -> dict[str, int]:
    """Count instruction mnemonics, ignoring symbol/comment text."""

    instruction_lines = [line for line in sass.splitlines() if "/*" in line]
    return {
        "call": sum(bool(re.search(r"\bCALL(?:\.|\s)", line)) for line in instruction_lines),
        "div": sum(bool(re.search(r"\bDIV(?:\.|\s)", line)) for line in instruction_lines),
        "mufu_ex2": sum("MUFU.EX2" in line for line in instruction_lines),
        "mufu_rcp": sum("MUFU.RCP" in line for line in instruction_lines),
        "fma": sum(
            bool(re.search(r"\b(?:FMA|FFMA)(?:\.|\s)", line))
            for line in instruction_lines
        ),
    }


def inspect_embedded_sass(
    object_path: pathlib.Path,
    *,
    objcopy: str = "objcopy",
    cuobjdump: str = "cuobjdump",
) -> dict[str, Any]:
    """Extract the embedded cubin and prove there is no divide slow path."""

    object_file = object_path.resolve()
    if not object_file.is_file():
        raise NewtonValidationError(f"Variant-C object is missing: {object_file}")
    with tempfile.TemporaryDirectory(prefix="katago-newton-sass-") as temporary:
        cubin = pathlib.Path(temporary) / "kernel.cubin"
        extract = subprocess.run(
            [objcopy, "--dump-section", f".lrodata={cubin}", str(object_file)],
            check=False,
            text=True,
            capture_output=True,
        )
        if extract.returncode != 0 or not cubin.is_file():
            raise NewtonValidationError(
                "could not extract embedded cubin: " + extract.stderr.strip()
            )
        sass_run = subprocess.run(
            [cuobjdump, "--dump-sass", "--gpu-architecture", "sm_103a", str(cubin)],
            check=False,
            text=True,
            capture_output=True,
        )
        symbols_run = subprocess.run(
            [cuobjdump, "--dump-elf-symbols", str(cubin)],
            check=False,
            text=True,
            capture_output=True,
        )
        if sass_run.returncode != 0 or symbols_run.returncode != 0:
            raise NewtonValidationError("cuobjdump could not inspect embedded cubin")
        counts = _sass_counts(sass_run.stdout)
        symbols = symbols_run.stdout
        slowpath_symbols = sorted(
            set(
                line.strip()
                for line in symbols.splitlines()
                if "div" in line.lower() or "slowpath" in line.lower()
            )
        )
        guardrail_symbols = sorted(
            set(
                line.strip()
                for line in symbols.splitlines()
                if "tcgen05_guardrail_trap" in line
            )
        )
        checks = {
            "sm103a_elf": "arch = sm_103a" in sass_run.stdout,
            "no_div_instruction": counts["div"] == 0,
            "no_div_or_slowpath_symbol": not slowpath_symbols,
            "only_two_guardrail_calls": counts["call"] == 2
            and len(guardrail_symbols) == 2,
            "fast_exp2_present": counts["mufu_ex2"] > 0,
            "reciprocal_seed_present": counts["mufu_rcp"] > 0,
            "newton_fma_present": counts["fma"] > 0,
        }
        if not all(checks.values()):
            failed = [name for name, passed in checks.items() if not passed]
            raise NewtonValidationError(
                "Variant-C SASS audit failed: " + ", ".join(failed)
            )
        return {
            "embedded_cubin_sha256": _sha256(cubin),
            "embedded_cubin_bytes": cubin.stat().st_size,
            "counts": counts,
            "guardrail_symbols": guardrail_symbols,
            "slowpath_symbols": slowpath_symbols,
            "checks": checks,
        }


def _timing_summary(samples_ms: list[float], streams: int) -> dict[str, Any]:
    median = statistics.median(samples_ms)
    return {
        "milliseconds_per_concurrent_iteration_samples": samples_ms,
        "median_stream_call_wall_milliseconds": median,
        "median_effective_milliseconds_per_call": median / streams,
        "calls_per_iteration": streams,
        "calls_per_second": 1000.0 * streams / median,
        "relative_spread": (max(samples_ms) - min(samples_ms)) / median,
    }


def benchmark(
    *,
    allow_gpu: bool,
    library_path: pathlib.Path,
    object_path: pathlib.Path,
    device: int = 0,
    warmup: int = 20000,
    iterations: int = 1000,
    repeats: int = 5,
    seed: int = 20260818,
) -> dict[str, Any]:
    if not allow_gpu:
        raise NewtonValidationError(
            "Variant-C benchmark requires explicit --allow-gpu"
        )
    for name, value in (
        ("device", device),
        ("warmup", warmup),
        ("iterations", iterations),
        ("repeats", repeats),
        ("seed", seed),
    ):
        if type(value) is not int:  # noqa: E721 - deliberately reject bool
            raise NewtonValidationError(f"{name} must be an integer")
    if device < 0 or warmup < 1 or iterations < 1 or repeats < 1:
        raise NewtonValidationError("invalid Variant-C device or timing count")

    authentication = authenticate_artifact(library_path, object_path)
    sass = inspect_embedded_sass(object_path)

    import torch

    if tuple(torch.cuda.get_device_capability(device)) != base.COMPUTE_CAPABILITY:
        raise NewtonValidationError("Variant-C benchmark requires exact SM103")
    torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = False
    tensors = base._allocate_gpu_benchmark_inputs(torch, device, seed)
    signal_scaling = variant.strengthen_probe_signal(torch, tensors)
    problem: base.DenseSwiGLUProblem = tensors["problem"]

    def allocate_ab12() -> Any:
        return torch.empty_strided(
            (problem.m, problem.n_packed, 1),
            (problem.n_packed, 1, problem.m * problem.n_packed),
            dtype=torch.float16,
            device=f"cuda:{device}",
        )

    def allocate_output() -> Any:
        return torch.empty_strided(
            (problem.m, problem.n_output, 1),
            (problem.n_output, 1, problem.m * problem.n_output),
            dtype=torch.float16,
            device=f"cuda:{device}",
        )

    ab12 = [allocate_ab12(), allocate_ab12()]
    outputs = [allocate_output(), allocate_output()]
    library = variant_a._load_native_library(library_path)
    status = ctypes.c_int32(-999)
    context = library.katagoCudnnOssB29Create(device, ctypes.byref(status))
    if not context or status.value != 0:
        raise NewtonValidationError(
            f"Variant-C context creation failed with status {status.value}"
        )

    def launch(index: int) -> None:
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
            raise NewtonValidationError(
                f"Variant-C launch failed with status {launch_status}"
            )

    def measure(stream_count: int) -> dict[str, Any]:
        streams = [torch.cuda.Stream(device=device) for _ in range(stream_count)]
        coordinator = torch.cuda.Stream(device=device)
        for _ in range(warmup):
            for index, stream in enumerate(streams):
                with torch.cuda.stream(stream):
                    launch(index)
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
                        launch(0)
                    end.record()
            else:
                done = [torch.cuda.Event() for _ in streams]
                with torch.cuda.stream(coordinator):
                    start.record()
                for stream in streams:
                    stream.wait_event(start)
                for _ in range(iterations):
                    for index, stream in enumerate(streams):
                        with torch.cuda.stream(stream):
                            launch(index)
                for event, stream in zip(done, streams, strict=True):
                    event.record(stream)
                with torch.cuda.stream(coordinator):
                    for event in done:
                        coordinator.wait_event(event)
                    end.record()
            end.synchronize()
            samples.append(start.elapsed_time(end) / iterations)
        return _timing_summary(samples, stream_count)

    try:
        ab12[0].fill_(float("nan"))
        outputs[0].fill_(float("nan"))
        launch(0)
        torch.cuda.synchronize(device)
        reference = variant.projection_fp16_roundtrip_reference(
            torch,
            tensors["input_2d"],
            tensors["linear1"],
            tensors["linear_gate"],
        )
        packed_reference = (
            tensors["input_2d"].float()
            @ tensors["b_tensor"][:, :, 0].float().transpose(0, 1)
        ).half()
        correctness = variant.build_gpu_correctness_summary(
            torch,
            actual_output=outputs[0][:, :, 0],
            reference_output=reference,
            actual_ab12=ab12[0][:, :, 0],
            reference_ab12=packed_reference,
        )
        timings = {"s1": measure(1), "s2": measure(2)}
    finally:
        try:
            torch.cuda.synchronize(device)
        finally:
            library.katagoCudnnOssB29Destroy(context)

    return {
        "schema": 1,
        "kind": "katago-sm103-b29-cudnn-oss-variant-c-aot-validation",
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z"),
        "candidate_id": variant.CANDIDATE_ID,
        "numeric_semantics": variant.numeric_semantics(),
        "device": {
            "ordinal": device,
            "name": torch.cuda.get_device_name(device),
            "compute_capability": list(torch.cuda.get_device_capability(device)),
        },
        "authentication": authentication,
        "sass": sass,
        "correctness": {
            "status": "passed",
            "same_seed": seed,
            "signal_scaling": signal_scaling,
            "tight_correctness": correctness,
        },
        "method": {
            "clock": "CUDA coordinator-stream events spanning all worker streams",
            "warmup_iterations": warmup,
            "timed_iterations": iterations,
            "repeats": repeats,
            "allocation": "all A/B/AB12/C buffers preallocated before timing",
            "s2": "two independent CUDA streams and per-stream AB12/C buffers",
        },
        "timings": timings,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--allow-gpu", action="store_true")
    parser.add_argument("--library", type=pathlib.Path)
    parser.add_argument("--object", dest="object_path", type=pathlib.Path)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=20000)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260818)
    parser.add_argument("--output", type=pathlib.Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.benchmark:
        if args.allow_gpu or args.library or args.object_path or args.output:
            raise NewtonValidationError(
                "GPU/artifact/output options require --benchmark"
            )
        build_parser().print_help()
        return 0
    if args.library is None or args.object_path is None:
        raise NewtonValidationError("--benchmark requires --library and --object")
    payload = benchmark(
        allow_gpu=args.allow_gpu,
        library_path=args.library,
        object_path=args.object_path,
        device=args.device,
        warmup=args.warmup,
        iterations=args.iterations,
        repeats=args.repeats,
        seed=args.seed,
    )
    serialized = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.write_text(serialized, encoding="utf-8")
    sys.stdout.write(serialized)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
