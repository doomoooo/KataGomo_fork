#!/usr/bin/env python3
"""Scan exact-batch throughput for KataGo's official CUDA or TensorRT backend.

The scanner intentionally does not enable any repository-specific CUDA
architecture backend.  It is a reusable control measurement for a short
batch sweep and, with 1000 or more iterations and at least two repeats, a
long-stability confirmation.

Every benchmark is observed by the SM-occupancy guard from
``cuda_tactic_workflow``.  Measurements interrupted by another process using
the selected GPU's SMs are discarded by that guard.  All subprocess output,
commands, overrides, input hashes, device identity, samples, and rankings are
written to the result bundle.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import pathlib
import re
import subprocess
import sys
from collections.abc import Sequence
from typing import Any

try:
    from cuda_tactic_workflow import (
        MIN_LONG_ITERATIONS,
        MIN_STABLE_SAMPLES,
        _parse_benchmark_record,
        _run_benchmark_with_occupancy,
        config_string,
        parse_int_set,
        result_metric,
        sha256_file,
        summarize_samples,
        utc_now,
        write_json,
    )
except ModuleNotFoundError:  # pragma: no cover - package-style import
    from python.cuda_tactic_workflow import (
        MIN_LONG_ITERATIONS,
        MIN_STABLE_SAMPLES,
        _parse_benchmark_record,
        _run_benchmark_with_occupancy,
        config_string,
        parse_int_set,
        result_metric,
        sha256_file,
        summarize_samples,
        utc_now,
        write_json,
    )


SCHEMA = 1
RESULT_KIND = "official-backend-baseline"
BACKENDS = ("cuda", "tensorrt")


def _query_cuda_device(device: int) -> dict[str, Any]:
    try:
        from portable_cuda_device import query_cuda_device
    except ModuleNotFoundError:  # pragma: no cover - package-style import
        from python.portable_cuda_device import query_cuda_device
    return query_cuda_device(device)


def official_backend_overrides(
    backend: str, device: int, streams: int, batch: int,
) -> dict[str, object]:
    """Return an explicit, exact-batch official-backend configuration."""
    if backend not in BACKENDS:
        raise ValueError(f"--backend must be one of {BACKENDS}")
    if device < 0:
        raise ValueError("--device must be non-negative")
    if streams < 1:
        raise ValueError("--streams must be positive")
    if batch < 1:
        raise ValueError("batch sizes must be positive")

    values: dict[str, object] = {
        "nnMaxBatchSize": batch,
        "numNNServerThreadsPerModel": streams,
        "requireMaxBoardSize": True,
        "useFP16": True,
    }
    if backend == "cuda":
        values.update({
            "cudaSm89Backend": False,
            "cudaSm89Forward": False,
            "cudaSm120Backend": False,
            "cudaUseNHWC": True,
            "cudaWarmupOnlyMaxBatchSize": True,
        })
        for thread in range(streams):
            values[f"cudaDeviceToUseThread{thread}"] = device
    else:
        values.update({
            "trtDisableOnnx": False,
            "trtTransformerNHWC": True,
        })
        for thread in range(streams):
            values[f"trtDeviceToUseThread{thread}"] = device
    return values


def benchmark_command(
    *,
    binary: pathlib.Path,
    config: pathlib.Path,
    model: pathlib.Path,
    overrides: dict[str, object],
    batch: int,
    iterations: int,
    warmup: int,
) -> list[str]:
    """Build the normal whole-graph ``benchmarknn`` invocation."""
    return [
        str(binary),
        "benchmarknn",
        "-config",
        str(config),
        "-override-config",
        config_string(overrides),
        "-model",
        str(model),
        "-iterations",
        str(iterations),
        "-warmup",
        str(warmup),
        "-batch-size",
        str(batch),
        "-boardsize",
        "19",
        "-json",
    ]


def _validate_args(args: argparse.Namespace) -> list[int]:
    if args.device < 0:
        raise ValueError("--device must be non-negative")
    if args.streams < 1:
        raise ValueError("--streams must be positive")
    if args.iterations < 1:
        raise ValueError("--iterations must be positive")
    if args.warmup < 0:
        raise ValueError("--warmup must be non-negative")
    if args.repeats < 1:
        raise ValueError("--repeats must be positive")
    if args.timeout <= 0 or not math.isfinite(args.timeout):
        raise ValueError("--timeout must be a positive finite number")
    if args.max_attempts < 1:
        raise ValueError("--max-attempts must be positive")
    if (
        args.iterations >= MIN_LONG_ITERATIONS
        and args.repeats < MIN_STABLE_SAMPLES
    ):
        raise ValueError(
            f"a {MIN_LONG_ITERATIONS}-iteration long confirmation requires "
            f"at least {MIN_STABLE_SAMPLES} repeats"
        )
    return parse_int_set(args.batches)


def _input_path(value: str, label: str, *, executable: bool = False) -> pathlib.Path:
    path = pathlib.Path(value).expanduser().resolve()
    if not path.is_file():
        raise ValueError(f"{label} does not exist: {path}")
    if executable and not os.access(path, os.X_OK):
        raise ValueError(f"{label} is not executable: {path}")
    return path


def _safe_stem(backend: str, batch: int, repeat: int, attempt: int) -> str:
    return re.sub(
        r"[^A-Za-z0-9_.-]+",
        "_",
        f"official-{backend}-b{batch}-r{repeat}-a{attempt}",
    )


def _write_raw_attempt(
    raw_dir: pathlib.Path,
    *,
    backend: str,
    batch: int,
    repeat: int,
    attempt: int,
    stdout: str,
    stderr: str,
) -> tuple[pathlib.Path, pathlib.Path]:
    stem = _safe_stem(backend, batch, repeat, attempt)
    stdout_path = raw_dir / f"{stem}.out"
    stderr_path = raw_dir / f"{stem}.err"
    stdout_path.write_text(stdout, encoding="utf-8")
    stderr_path.write_text(stderr, encoding="utf-8")
    return stdout_path, stderr_path


def _ranking(rows: Sequence[dict[str, object]]) -> list[dict[str, object]]:
    measured = [row for row in rows if row.get("status") == "measured"]
    ordered = sorted(
        measured,
        key=lambda row: (
            -float(row["nn_evals_per_sec_median"]),
            int(row["batch"]),
        ),
    )
    ranking: list[dict[str, object]] = []
    for rank, row in enumerate(ordered, start=1):
        row["rank"] = rank
        ranking.append({
            "rank": rank,
            "batch": int(row["batch"]),
            "nn_evals_per_sec_median": float(
                row["nn_evals_per_sec_median"]
            ),
        })
    return ranking


def _new_payload(
    args: argparse.Namespace,
    *,
    binary: pathlib.Path,
    config: pathlib.Path,
    model: pathlib.Path,
    batches: list[int],
    device_identity: dict[str, object],
) -> dict[str, object]:
    return {
        "schema": SCHEMA,
        "kind": RESULT_KIND,
        "status": "running",
        "created_utc": utc_now(),
        "backend": args.backend,
        "device_ordinal": args.device,
        "streams": args.streams,
        "requested_batches": batches,
        "measurement_mode": (
            "long_confirmation"
            if args.iterations >= MIN_LONG_ITERATIONS
            else "short_scan"
        ),
        "measurement_request": {
            "iterations": args.iterations,
            "warmup": args.warmup,
            "repeats": args.repeats,
            "timeout_seconds": args.timeout,
            "max_attempts": args.max_attempts,
        },
        "inputs": {
            "binary": str(binary),
            "config": str(config),
            "model": str(model),
            "home_data_dir": args.home_data_dir,
        },
        "identity": {
            "binary_sha256": sha256_file(binary),
            "config_sha256": sha256_file(config),
            "model_sha256": sha256_file(model),
        },
        "device": device_identity,
        "rows": [],
        "ranking": [],
    }


def run_baseline(args: argparse.Namespace) -> dict[str, object]:
    """Run a serial exact-batch scan and return the persisted result bundle.

    A partial bundle with ``status=failed`` is written before a benchmark or
    stability error is raised.  Callers must therefore check both the process
    exit status and the bundle status; partial data is diagnostic evidence,
    never a passing baseline.
    """
    batches = _validate_args(args)
    binary = _input_path(args.binary, "binary", executable=True)
    config = _input_path(args.config, "config")
    model = _input_path(args.model, "model")
    output = pathlib.Path(args.output).expanduser().resolve()
    raw_dir = pathlib.Path(args.raw_dir).expanduser().resolve()
    raw_dir.mkdir(parents=True, exist_ok=True)
    if args.home_data_dir is not None:
        home_data_dir = pathlib.Path(args.home_data_dir).expanduser().resolve()
        home_data_dir.mkdir(parents=True, exist_ok=True)
        args.home_data_dir = str(home_data_dir)

    device_identity = _query_cuda_device(args.device)
    if not isinstance(device_identity, dict):
        raise RuntimeError("CUDA device query did not return an identity object")
    payload = _new_payload(
        args,
        binary=binary,
        config=config,
        model=model,
        batches=batches,
        device_identity=device_identity,
    )
    rows = payload["rows"]
    assert isinstance(rows, list)
    write_json(output, payload)

    def fail(message: str, row: dict[str, object]) -> None:
        row["status"] = "failed"
        row["error"] = message
        row["finished_utc"] = utc_now()
        rows.append(row)
        payload["status"] = "failed"
        payload["error"] = message
        payload["failed_utc"] = utc_now()
        payload["ranking"] = _ranking(rows)
        write_json(output, payload)
        raise RuntimeError(message)

    # Deliberately serial: an exact batch owns the selected GPU until every
    # requested repeat is complete, then the next batch starts.
    for batch in batches:
        overrides = official_backend_overrides(
            args.backend, args.device, args.streams, batch,
        )
        if args.home_data_dir is not None:
            overrides["homeDataDir"] = args.home_data_dir
        command = benchmark_command(
            binary=binary,
            config=config,
            model=model,
            overrides=overrides,
            batch=batch,
            iterations=args.iterations,
            warmup=args.warmup,
        )
        row: dict[str, object] = {
            "batch": batch,
            "status": "running",
            "command": command,
            "overrides": overrides,
            "runs": [],
        }
        samples: list[float] = []
        run_records = row["runs"]
        assert isinstance(run_records, list)

        for repeat in range(args.repeats):
            attempt_records: list[dict[str, object]] = []
            run_record: dict[str, object] = {
                "repeat": repeat,
                "status": "running",
                "attempts": attempt_records,
            }
            throughput: float | None = None
            benchmark_record: dict[str, object] | None = None

            for attempt in range(args.max_attempts):
                completed: subprocess.CompletedProcess[str] | None = None
                timed_out = False
                occupancy: dict[str, object] = {}
                invocation_error: str | None = None
                try:
                    completed, timed_out, occupancy = (
                        _run_benchmark_with_occupancy(
                            command,
                            device=args.device,
                            timeout=args.timeout,
                        )
                    )
                except (OSError, RuntimeError, subprocess.SubprocessError) as exc:
                    invocation_error = str(exc)

                stdout = completed.stdout if completed is not None else ""
                stderr = completed.stderr if completed is not None else ""
                if invocation_error is not None:
                    stderr = (stderr + "\n" + invocation_error).lstrip("\n")
                stdout_path, stderr_path = _write_raw_attempt(
                    raw_dir,
                    backend=args.backend,
                    batch=batch,
                    repeat=repeat,
                    attempt=attempt,
                    stdout=stdout,
                    stderr=stderr,
                )
                returncode = completed.returncode if completed is not None else None
                attempt_record: dict[str, object] = {
                    "attempt": attempt,
                    "returncode": returncode,
                    "timed_out": timed_out,
                    "stdout": str(stdout_path),
                    "stderr": str(stderr_path),
                    "gpu_occupancy": occupancy,
                }
                if invocation_error is not None:
                    attempt_record["error"] = invocation_error
                attempt_records.append(attempt_record)

                foreign = occupancy.get("foreign_active_sm_pids", [])
                monitor_error = occupancy.get("error")
                clean_process = (
                    completed is not None
                    and completed.returncode == 0
                    and not timed_out
                    and not foreign
                    and not monitor_error
                    and invocation_error is None
                )
                if not clean_process:
                    reasons = []
                    if invocation_error:
                        reasons.append(invocation_error)
                    if completed is not None and completed.returncode != 0:
                        reasons.append(f"returncode={completed.returncode}")
                    if timed_out:
                        reasons.append("timed out")
                    if foreign:
                        reasons.append(
                            "external SM work from PID(s) "
                            + ",".join(str(pid) for pid in foreign)
                        )
                    if monitor_error:
                        reasons.append(f"occupancy monitor: {monitor_error}")
                    attempt_record["discarded"] = True
                    attempt_record["discard_reason"] = "; ".join(reasons)
                    continue

                try:
                    benchmark_record = _parse_benchmark_record(stdout)
                    throughput = result_metric(benchmark_record)
                except (ValueError, json.JSONDecodeError) as exc:
                    attempt_record["discarded"] = True
                    attempt_record["discard_reason"] = str(exc)
                    continue
                attempt_record["discarded"] = False
                break

            if throughput is None or benchmark_record is None:
                run_record["status"] = "failed"
                run_records.append(run_record)
                fail(
                    f"official {args.backend} baseline failed for B{batch} "
                    f"repeat {repeat} after {args.max_attempts} attempts",
                    row,
                )

            samples.append(throughput)
            final_attempt = attempt_records[-1]
            run_record.update({
                "status": "measured",
                "throughput": throughput,
                "benchmark": benchmark_record,
                "stdout": final_attempt["stdout"],
                "stderr": final_attempt["stderr"],
                "gpu_occupancy": final_attempt["gpu_occupancy"],
            })
            run_records.append(run_record)

        summary = summarize_samples(
            samples,
            iterations=args.iterations,
            warmup=args.warmup,
        )
        row.update(summary)
        if (
            args.iterations >= MIN_LONG_ITERATIONS
            and summary.get("stable_long_nn_evals_per_sec") is None
        ):
            fail(
                f"official {args.backend} long confirmation is unstable for "
                f"B{batch}: relative spread="
                f"{summary.get('measurement_relative_spread')}",
                row,
            )
        row["status"] = "measured"
        row["finished_utc"] = utc_now()
        rows.append(row)
        payload["ranking"] = _ranking(rows)
        write_json(output, payload)
        print(
            f"official {args.backend} B{batch}: "
            f"{float(row['nn_evals_per_sec_median']):.3f} nnEval/s",
            flush=True,
        )

    payload["status"] = "completed"
    payload["finished_utc"] = utc_now()
    payload["ranking"] = _ranking(rows)
    write_json(output, payload)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=BACKENDS, required=True)
    parser.add_argument("--binary", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--streams", type=int, default=2)
    parser.add_argument("--batches", default="4-32")
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--output", required=True)
    parser.add_argument("--raw-dir", required=True)
    parser.add_argument(
        "--home-data-dir",
        help="managed KataGo home/cache directory recorded in every override",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=120.0,
        help="seconds allowed for one benchmark subprocess",
    )
    parser.add_argument("--max-attempts", type=int, default=2)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        run_baseline(args)
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
        print(f"official_backend_baseline: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
