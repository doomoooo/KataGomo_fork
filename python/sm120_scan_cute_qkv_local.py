#!/usr/bin/env python3
"""Generate and locally time packed-QKV for every explicit exact batch."""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import pathlib
import subprocess
import sys


CANDIDATE_ID = "qkv-m128-n128-k64-s2-cute-atom4x2-packed"


def utc_now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_int_set(value: str) -> list[int]:
    result = set()
    for token in value.split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            first, last = (int(item) for item in token.split("-", 1))
            result.update(range(first, last + 1))
        else:
            result.add(int(token))
    values = sorted(result)
    if not values or values[0] < 1:
        raise ValueError("batch set must contain positive integers")
    return values


def run_checked(command: list[str], log_prefix: pathlib.Path) -> None:
    result = subprocess.run(command, text=True, capture_output=True)
    log_prefix.parent.mkdir(parents=True, exist_ok=True)
    log_prefix.with_suffix(".out").write_text(result.stdout)
    log_prefix.with_suffix(".err").write_text(result.stderr)
    if result.returncode != 0:
        raise RuntimeError(
            f"command failed ({result.returncode}); see {log_prefix.with_suffix('.err')}"
        )


def write_manifest(path: pathlib.Path, payload: dict) -> None:
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default=".")
    parser.add_argument("--space", required=True)
    parser.add_argument("--batches", default="4-32")
    parser.add_argument("--device", type=int, required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--cutlass-root", required=True)
    parser.add_argument("--nvcc", default="nvcc")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--reuse-existing", action="store_true")
    args = parser.parse_args()

    repo = pathlib.Path(args.repo).resolve()
    space_path = pathlib.Path(args.space).resolve()
    space = json.loads(space_path.read_text())
    batch_spaces = {item["batch"]: item for item in space["batches"]}
    batches = parse_int_set(args.batches)
    missing = sorted(set(batches) - set(batch_spaces))
    if missing:
        raise ValueError(f"batches outside search space: {missing}")
    output_dir = pathlib.Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.json"
    payload = {
        "schema": 1,
        "started_utc": utc_now(),
        "family": "qkv",
        "fixed_board": [19, 19],
        "space": str(space_path),
        "space_sha256": sha256_file(space_path),
        "counter_policy": {
            "gpu_performance_counters_required": False,
            "ncu_output_parsed": False,
        },
        "entries": [],
    }
    if args.reuse_existing and manifest_path.is_file():
        previous = json.loads(manifest_path.read_text())
        if previous.get("space_sha256") != payload["space_sha256"]:
            raise ValueError("cannot reuse packed-QKV results from another space")
        payload = previous
    completed = {item["batch"] for item in payload["entries"]}

    for batch in batches:
        if batch in completed:
            continue
        candidate = next(
            item for item in batch_spaces[batch]["qkv"]
            if item["id"] == CANDIDATE_ID
        )
        batch_dir = output_dir / f"b{batch}"
        stem = f"sm120_qkv_cute_b{batch}"
        bridge = batch_dir / f"{stem}.cu"
        generator_command = [
            args.python,
            str(repo / "python/sm120_generate_cute_qkv_aot.py"),
            "--batch", str(batch),
            "--output-dir", str(batch_dir),
            "--artifact-stem", stem,
            "--bridge-path", str(bridge),
            "--candidate-id", CANDIDATE_ID,
            "--device", str(args.device),
            "--cutlass-root", str(pathlib.Path(args.cutlass_root).resolve()),
        ]
        if "max_active_clusters" in candidate:
            generator_command.extend([
                "--max-active-clusters", str(candidate["max_active_clusters"]),
            ])
        run_checked(generator_command, batch_dir / "generate")
        measure_command = [
            sys.executable,
            str(repo / "python/sm120_measure_cute_qkv_local.py"),
            "--repo", str(repo),
            "--artifact-dir", str(batch_dir),
            "--artifact-stem", stem,
            "--device", str(args.device),
            "--batch", str(batch),
            "--warmup", str(args.warmup),
            "--iterations", str(args.iterations),
            "--nvcc", args.nvcc,
        ]
        run_checked(measure_command, batch_dir / "measure")
        metadata_path = batch_dir / f"{stem}.json"
        metadata = json.loads(metadata_path.read_text())
        payload["entries"].append({
            "batch": batch,
            "candidate_id": CANDIDATE_ID,
            "candidate": candidate,
            "source": str(bridge),
            "source_sha256": sha256_file(bridge),
            "metadata": str(metadata_path),
            "metadata_sha256": sha256_file(metadata_path),
        })
        payload["finished_utc"] = utc_now()
        write_manifest(manifest_path, payload)
        print(
            f"B{batch} {CANDIDATE_ID}: "
            f"{metadata['local_measurement']['s1_us_median']:.3f} us",
            flush=True,
        )
    print(json.dumps({"manifest": str(manifest_path), "entries": len(payload["entries"])}))


if __name__ == "__main__":
    main()
