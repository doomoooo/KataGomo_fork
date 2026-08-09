#!/usr/bin/env python3
"""Measure one generated packed-QKV CuTe object without GPU counters."""

from __future__ import annotations

import argparse
import json
import pathlib
import subprocess


def last_json(text: str) -> dict:
    for line in reversed(text.splitlines()):
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            return json.loads(line)
    raise RuntimeError("local packed-QKV benchmark emitted no JSON")


def run_checked(command: list[str], log_prefix: pathlib.Path) -> subprocess.CompletedProcess:
    result = subprocess.run(command, text=True, capture_output=True)
    log_prefix.parent.mkdir(parents=True, exist_ok=True)
    log_prefix.with_suffix(".out").write_text(result.stdout)
    log_prefix.with_suffix(".err").write_text(result.stderr)
    if result.returncode != 0:
        raise RuntimeError(
            f"command failed ({result.returncode}); see {log_prefix.with_suffix('.err')}"
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default=".")
    parser.add_argument("--artifact-dir", required=True)
    parser.add_argument("--artifact-stem", default="sm120_qkv_cute_active")
    parser.add_argument("--device", type=int, required=True)
    parser.add_argument("--batch", type=int, required=True)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--nvcc", default="nvcc")
    args = parser.parse_args()

    repo = pathlib.Path(args.repo).resolve()
    artifact_dir = pathlib.Path(args.artifact_dir).resolve()
    stem = artifact_dir / args.artifact_stem
    bridge = stem.with_suffix(".cu")
    object_path = stem.with_suffix(".o")
    metadata_path = stem.with_suffix(".json")
    for path in (bridge, object_path, metadata_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    metadata = json.loads(metadata_path.read_text())
    if metadata.get("batch") != args.batch:
        raise ValueError("artifact batch does not match --batch")
    executable = artifact_dir / "local-qkv-bench"
    compile_command = [
        args.nvcc, "-std=c++17", "-O3", "-arch=sm_120",
        "-I", str(artifact_dir),
        str(repo / "cpp/neuralnet/sm120_local_qkv_bench.cu"),
        str(bridge),
        str(repo / "cpp/neuralnet/fa4_aot/fa4_cuda_bridge.cpp"),
        str(object_path),
        "-lcublas", "-lcuda", "-o", str(executable),
    ]
    run_checked(compile_command, artifact_dir / "local-qkv-compile")
    result = run_checked(
        [
            str(executable), str(args.device), str(args.batch),
            str(args.warmup), str(args.iterations),
        ],
        artifact_dir / "local-qkv-run",
    )
    measurement = last_json(result.stdout)
    metadata["local_measurement"] = {
        **measurement,
        "metric": "single-stream CUDA-event kernel latency",
        "gpu_performance_counters_required": False,
        "ncu_output_parsed": False,
        "compile_command": compile_command,
    }
    temporary = metadata_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(metadata, indent=2) + "\n")
    temporary.replace(metadata_path)
    print(json.dumps(measurement))


if __name__ == "__main__":
    main()
