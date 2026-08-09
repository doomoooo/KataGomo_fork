#!/usr/bin/env python3
"""Replay generated packed-QKV artifacts against the local cuBLAS reference."""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import pathlib
import shlex
import subprocess
import sys

try:
    from cuda_tactic_workflow import parse_int_set, sha256_file, utc_now
except ModuleNotFoundError:  # imported as ``python.sm120_replay...`` in tests
    from python.cuda_tactic_workflow import parse_int_set, sha256_file, utc_now


CANDIDATE_ID = "wide_qkv-m128-n128-k64-s2-cute-atom4x2-packed"


def write_json(path: pathlib.Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def run_checked(command: list[str], log_prefix: pathlib.Path) -> str:
    completed = subprocess.run(command, text=True, capture_output=True)
    log_prefix.parent.mkdir(parents=True, exist_ok=True)
    log_prefix.with_suffix(".out").write_text(completed.stdout)
    log_prefix.with_suffix(".err").write_text(completed.stderr)
    if completed.returncode != 0:
        raise RuntimeError(
            f"command failed ({completed.returncode}); see "
            f"{log_prefix.with_suffix('.err')}"
        )
    return completed.stdout


def last_json(text: str) -> dict:
    for line in reversed(text.splitlines()):
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            return json.loads(line)
    raise RuntimeError("correctness replay emitted no JSON")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default=".")
    parser.add_argument("--artifact-root", required=True)
    parser.add_argument("--batches", default="4-32")
    parser.add_argument("--device", type=int, required=True)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--nvcc", default="nvcc")
    parser.add_argument("--runner", default="")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    repo = pathlib.Path(args.repo).resolve()
    artifact_root = pathlib.Path(args.artifact_root).resolve()
    batches = parse_int_set(args.batches)
    runner = shlex.split(args.runner)
    output = pathlib.Path(args.output).resolve()
    logs = output.parent / f"{output.stem}-logs"
    regime = {
        "artifact_root": str(artifact_root),
        "batches": batches,
        "device": args.device,
        "warmup": args.warmup,
        "iterations": args.iterations,
        "nvcc": args.nvcc,
        "runner": runner,
        "generator": str(repo / "python/sm120_generate_cute_qkv_aot.py"),
        "generator_sha256": sha256_file(
            repo / "python/sm120_generate_cute_qkv_aot.py"
        ),
    }
    if output.exists():
        payload = json.loads(output.read_text())
        if payload.get("regime") != regime:
            raise ValueError("replay output exists with a different regime")
    else:
        payload = {
            "schema": 1,
            "kind": "sm120-cute-qkv-correctness-replay",
            "started_utc": utc_now(),
            "candidate_id": CANDIDATE_ID,
            "regime": regime,
            "rows": [],
            "acceptance": {
                "reference": "cuBLAS FP16 GEMM with matching row-major storage",
                "metric": "max_abs and rmse",
                "timing_is_not_an_acceptance_gate": True,
            },
        }
    completed = {
        int(row["batch"]) for row in payload.get("rows", [])
        if row.get("status") == "correct"
    }
    for batch in batches:
        if batch in completed:
            continue
        artifact_dir = (
            artifact_root / f"b{batch}" / CANDIDATE_ID
        )
        stem = artifact_dir / "sm120_qkv_cute_active"
        metadata_path = stem.with_suffix(".json")
        required = [
            stem.with_suffix(".cu"),
            stem.with_suffix(".h"),
            stem.with_suffix(".o"),
            metadata_path,
        ]
        row = {
            "batch": batch,
            "candidate_id": CANDIDATE_ID,
            "artifact_dir": str(artifact_dir),
            "started_utc": utc_now(),
        }
        try:
            if any(not path.is_file() for path in required):
                raise FileNotFoundError(
                    ", ".join(str(path) for path in required if not path.is_file())
                )
            command = runner + [
                sys.executable,
                str(repo / "python/sm120_measure_cute_qkv_local.py"),
                "--repo", str(repo),
                "--artifact-dir", str(artifact_dir),
                "--artifact-stem", "sm120_qkv_cute_active",
                "--device", str(args.device),
                "--batch", str(batch),
                "--warmup", str(args.warmup),
                "--iterations", str(args.iterations),
                "--nvcc", args.nvcc,
            ]
            row["command"] = command
            measurement = last_json(
                run_checked(command, logs / f"b{batch}-replay")
            )
            metadata = json.loads(metadata_path.read_text())
            correctness = measurement.get("correctness_against_cublas")
            if not isinstance(correctness, dict):
                raise RuntimeError("replay JSON has no correctness_against_cublas")
            row.update({
                "status": "correct",
                "finished_utc": utc_now(),
                "measurement": measurement,
                "correctness": correctness,
                "metadata_sha256": sha256_file(metadata_path),
                "artifact_sha256": {
                    path.suffix.lstrip("."): sha256_file(path)
                    for path in required if path.suffix != ".json"
                },
            })
            print(
                f"B{batch} correctness: max_abs={correctness.get('max_abs')} "
                f"rmse={correctness.get('rmse')}",
                flush=True,
            )
        except Exception as error:
            row.update({
                "status": "failed",
                "finished_utc": utc_now(),
                "error": str(error),
            })
            print(f"B{batch} correctness FAILED: {error}", flush=True)
        payload["rows"].append(row)
        payload["finished_utc"] = utc_now()
        write_json(output, payload)


if __name__ == "__main__":
    main()
