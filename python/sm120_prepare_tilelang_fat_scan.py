#!/usr/bin/env python3
"""Generate one linkable exact-batch TileLang family bundle.

All requested TileLang ``(batch, tactic ID)`` translation units are generated
before a registry is emitted.  Configure KataGo with the manifest's registry
and source list, then build once; runtime selection remains an exact batch/ID
lookup and whole-graph measurements no longer rebuild or relink per candidate.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import pathlib
import subprocess
import sys

from sm120_fat_scan import select_tilelang_requests, write_registry


def parse_int_set(value: str) -> list[int]:
    result: set[int] = set()
    for token in value.split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            first, last = (int(item) for item in token.split("-", 1))
            if last < first:
                raise ValueError(f"invalid descending range: {token}")
            result.update(range(first, last + 1))
        else:
            result.add(int(token))
    values = sorted(result)
    if not values or values[0] < 1:
        raise ValueError("batch set must contain positive integers")
    return values


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def utc_now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def run_generator(command: list[str], log_prefix: pathlib.Path) -> None:
    completed = subprocess.run(command, text=True, capture_output=True)
    log_prefix.parent.mkdir(parents=True, exist_ok=True)
    log_prefix.with_suffix(".out").write_text(completed.stdout)
    log_prefix.with_suffix(".err").write_text(completed.stderr)
    if completed.returncode != 0:
        raise RuntimeError(
            f"candidate generation failed ({completed.returncode}); "
            f"see {log_prefix.with_suffix('.err')}"
        )


def reusable_entry(
    previous: dict | None,
    source_path: pathlib.Path,
    metadata_path: pathlib.Path,
    request: dict,
    generator_sha256: str,
    space_sha256: str,
) -> bool:
    if previous is None or not source_path.is_file() or not metadata_path.is_file():
        return False
    metadata = json.loads(metadata_path.read_text())
    return (
        previous.get("generator_sha256") == generator_sha256
        and previous.get("space_sha256") == space_sha256
        and previous.get("source_sha256") == sha256_file(source_path)
        and previous.get("metadata_sha256") == sha256_file(metadata_path)
        and metadata.get("batch") == request["batch"]
        and metadata.get("candidate", {}).get("id") == request["candidate_id"]
        and metadata.get("fat_symbol_token") == request["symbol_token"]
        and metadata.get("launch_symbol") == request["launch_symbol"]
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--space", required=True)
    parser.add_argument("--family", choices=("ffn", "qkv", "linear2"), required=True)
    parser.add_argument("--batches", default="1-32")
    parser.add_argument("--candidate-ids", default="")
    parser.add_argument("--device", type=int, required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--generator", default="")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--s1-warmup", type=int, default=5)
    parser.add_argument("--s1-iterations", type=int, default=20)
    parser.add_argument(
        "--reuse-existing",
        action="store_true",
        help="reuse a TU only when generator, space, source, and metadata hashes match",
    )
    args = parser.parse_args()

    space_path = pathlib.Path(args.space).resolve()
    space = json.loads(space_path.read_text())
    batches = parse_int_set(args.batches)
    candidate_ids = [item for item in args.candidate_ids.split(",") if item]
    requests = select_tilelang_requests(space, args.family, batches, candidate_ids)
    output_dir = pathlib.Path(args.output_dir).resolve()
    sources_dir = output_dir / "sources"
    metadata_dir = output_dir / "metadata"
    logs_dir = output_dir / "logs"
    registry_path = output_dir / f"sm120_search_{args.family}_fat_registry.cu"
    manifest_path = output_dir / "manifest.json"
    generator_path = pathlib.Path(args.generator).resolve() if args.generator else (
        pathlib.Path(__file__).resolve().parent / "sm120_generate_tilelang_aot.py"
    )
    generator_sha256 = sha256_file(generator_path)
    space_sha256 = sha256_file(space_path)
    previous_entries = {}
    if args.reuse_existing and manifest_path.is_file():
        previous_manifest = json.loads(manifest_path.read_text())
        previous_entries = {
            (item["batch"], item["candidate_id"]): item
            for item in previous_manifest.get("entries", [])
        }

    started = utc_now()
    entries = []
    for index, request in enumerate(requests, start=1):
        token = request["symbol_token"]
        candidate_dir = metadata_dir / token
        source_path = sources_dir / f"{token}.cu"
        metadata_path = candidate_dir / f'{args.family}-{request["candidate_id"]}.json'
        previous = previous_entries.get((request["batch"], request["candidate_id"]))
        reused = args.reuse_existing and reusable_entry(
            previous, source_path, metadata_path, request,
            generator_sha256, space_sha256,
        )
        if not reused:
            command = [
                args.python,
                str(generator_path),
                "--space", str(space_path),
                "--family", args.family,
                "--candidate-id", request["candidate_id"],
                "--batch", str(request["batch"]),
                "--device", str(args.device),
                "--output-dir", str(candidate_dir),
                "--source-path", str(source_path),
                "--fat-symbol-token", token,
                "--s1-warmup", str(args.s1_warmup),
                "--s1-iterations", str(args.s1_iterations),
            ]
            run_generator(command, logs_dir / f"{index:04d}-{token}")
        metadata = json.loads(metadata_path.read_text())
        source_sha256 = sha256_file(source_path)
        if metadata.get("source_sha256") != source_sha256:
            raise RuntimeError(f"source hash mismatch for {source_path}")
        entries.append({
            **request,
            "source": str(source_path),
            "source_sha256": source_sha256,
            "metadata": str(metadata_path),
            "metadata_sha256": sha256_file(metadata_path),
            "generator_sha256": generator_sha256,
            "space_sha256": space_sha256,
            "reused": reused,
        })
        print(
            f'prepared B{request["batch"]} {request["candidate_id"]}'
            + (" (reused)" if reused else ""),
            flush=True,
        )

    write_registry(registry_path, args.family, entries)
    payload = {
        "schema": 1,
        "started_utc": started,
        "finished_utc": utc_now(),
        "family": args.family,
        "fixed_board": [19, 19],
        "batch_policy": "all explicitly requested batches; exact batch+ID dispatch",
        "space": str(space_path),
        "space_sha256": space_sha256,
        "generator": str(generator_path),
        "generator_sha256": generator_sha256,
        "registry_source": str(registry_path),
        "registry_sha256": sha256_file(registry_path),
        "sources": [item["source"] for item in entries],
        "entries": entries,
    }
    temporary = manifest_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(manifest_path)
    print(json.dumps({
        "manifest": str(manifest_path),
        "family": args.family,
        "entries": len(entries),
        "registry_source": str(registry_path),
    }))


if __name__ == "__main__":
    main()
