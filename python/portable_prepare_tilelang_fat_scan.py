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

from portable_fat_scan import select_tilelang_requests, write_registry


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


def run_command(command: list[str], log_prefix: pathlib.Path, label: str) -> None:
    completed = subprocess.run(command, text=True, capture_output=True)
    log_prefix.parent.mkdir(parents=True, exist_ok=True)
    log_prefix.with_suffix(".out").write_text(completed.stdout)
    log_prefix.with_suffix(".err").write_text(completed.stderr)
    if completed.returncode != 0:
        raise RuntimeError(
            f"{label} failed ({completed.returncode}); "
            f"see {log_prefix.with_suffix('.err')}"
        )


def reusable_entry(
    previous: dict | None,
    source_path: pathlib.Path,
    object_path: pathlib.Path,
    metadata_path: pathlib.Path,
    request: dict,
    generator_sha256: str,
    space_sha256: str,
    compile_objects: bool,
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
        and (
            not compile_objects
            or (
                object_path.is_file()
                and previous.get("object_sha256") == sha256_file(object_path)
            )
        )
    )


def recover_existing_entry(
    source_path: pathlib.Path,
    metadata_path: pathlib.Path,
    request: dict,
    generator_sha256: str,
    space_sha256: str,
) -> dict | None:
    """Recover a hash-valid TU when an older run died before its manifest."""
    if not source_path.is_file() or not metadata_path.is_file():
        return None
    metadata = json.loads(metadata_path.read_text())
    source_sha256 = sha256_file(source_path)
    if (
        metadata.get("source_sha256") != source_sha256
        or metadata.get("batch") != request["batch"]
        or metadata.get("candidate", {}).get("id") != request["candidate_id"]
        or metadata.get("fat_symbol_token") != request["symbol_token"]
        or metadata.get("launch_symbol") != request["launch_symbol"]
    ):
        return None
    return {
        "generator_sha256": generator_sha256,
        "space_sha256": space_sha256,
        "source_sha256": source_sha256,
        "metadata_sha256": sha256_file(metadata_path),
        "recovered_without_prior_manifest": True,
    }


def write_checkpoint(
    manifest_path: pathlib.Path,
    registry_path: pathlib.Path,
    family: str,
    entries: list[dict],
    requested_count: int,
    started: str,
    space_path: pathlib.Path,
    space_sha256: str,
    generator_path: pathlib.Path,
    generator_sha256: str,
) -> None:
    write_registry(registry_path, family, entries)
    complete = len(entries) == requested_count
    payload = {
        "schema": 1,
        "started_utc": started,
        "updated_utc": utc_now(),
        "finished_utc": utc_now() if complete else None,
        "complete": complete,
        "requested_entry_count": requested_count,
        "family": family,
        "fixed_board": [19, 19],
        "batch_policy": "all explicitly requested batches; exact batch+ID dispatch",
        "space": str(space_path),
        "space_sha256": space_sha256,
        "generator": str(generator_path),
        "generator_sha256": generator_sha256,
        "registry_source": str(registry_path),
        "registry_sha256": sha256_file(registry_path),
        "sources": [item["source"] for item in entries],
        "objects": [item["object"] for item in entries if item.get("object")],
        "entries": entries,
    }
    temporary = manifest_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(manifest_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--space", required=True)
    parser.add_argument(
        "--family", choices=("dual_ffn", "linear2"), required=True
    )
    parser.add_argument("--batches", default="1-32")
    parser.add_argument("--candidate-ids", default="")
    parser.add_argument("--device", type=int, required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--generator", default="")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--s1-warmup", type=int, default=5)
    parser.add_argument("--s1-iterations", type=int, default=20)
    parser.add_argument("--nvcc", default="nvcc")
    parser.add_argument(
        "--compile-objects", action=argparse.BooleanOptionalAction, default=True,
        help="compile every generated TU for sm_89 and record the exact command/hash",
    )
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
    objects_dir = output_dir / "objects"
    metadata_dir = output_dir / "metadata"
    logs_dir = output_dir / "logs"
    registry_path = output_dir / f"portable_search_{args.family}_fat_registry.cu"
    manifest_path = output_dir / "manifest.json"
    generator_path = pathlib.Path(args.generator).resolve() if args.generator else (
        pathlib.Path(__file__).resolve().parent / "portable_generate_tilelang_aot.py"
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
        object_path = objects_dir / f"{token}.o"
        metadata_path = candidate_dir / f'{args.family}-{request["candidate_id"]}.json'
        previous = previous_entries.get((request["batch"], request["candidate_id"]))
        if args.reuse_existing and previous is None:
            previous = recover_existing_entry(
                source_path, metadata_path, request,
                generator_sha256, space_sha256,
            )
        reused = args.reuse_existing and reusable_entry(
            previous, source_path, object_path, metadata_path, request,
            generator_sha256, space_sha256, args.compile_objects,
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
            run_command(
                command, logs_dir / f"{index:04d}-{token}-generate", "generation"
            )
        metadata = json.loads(metadata_path.read_text())
        source_sha256 = sha256_file(source_path)
        if metadata.get("source_sha256") != source_sha256:
            raise RuntimeError(f"source hash mismatch for {source_path}")
        compile_command = None
        object_sha256 = None
        if args.compile_objects and not reused:
            tilelang_root = pathlib.Path(
                metadata["generation_environment"]["tilelang_root"]
            )
            object_path.parent.mkdir(parents=True, exist_ok=True)
            compile_command = [
                args.nvcc, "-std=c++17", "-arch=sm_89", "-O3",
                "--use_fast_math", "--expt-relaxed-constexpr",
                "--expt-extended-lambda", "-lineinfo",
                f"-I{tilelang_root / 'src'}",
                f"-I{tilelang_root / '3rdparty' / 'cutlass' / 'include'}",
                "-c", str(source_path), "-o", str(object_path),
            ]
            run_command(
                compile_command, logs_dir / f"{index:04d}-{token}-compile",
                "SM89 compilation",
            )
        elif args.compile_objects:
            compile_command = previous.get("compile_command") if previous else None
        if args.compile_objects:
            if not object_path.is_file():
                raise RuntimeError(f"object was not produced: {object_path}")
            object_sha256 = sha256_file(object_path)
        entries.append({
            **request,
            "source": str(source_path),
            "source_sha256": source_sha256,
            "object": str(object_path) if args.compile_objects else None,
            "object_sha256": object_sha256,
            "compile_command": compile_command,
            "metadata": str(metadata_path),
            "metadata_sha256": sha256_file(metadata_path),
            "generator_sha256": generator_sha256,
            "space_sha256": space_sha256,
            "reused": reused,
            "recovered_without_prior_manifest": bool(
                previous and previous.get("recovered_without_prior_manifest")
            ),
        })
        write_checkpoint(
            manifest_path, registry_path, args.family, entries, len(requests),
            started, space_path, space_sha256, generator_path,
            generator_sha256,
        )
        print(
            f'prepared B{request["batch"]} {request["candidate_id"]}'
            + (" (reused)" if reused else ""),
            flush=True,
        )

    payload = json.loads(manifest_path.read_text())
    print(json.dumps({
        "manifest": str(manifest_path),
        "family": args.family,
        "entries": len(entries),
        "registry_source": str(registry_path),
        "compiled_objects": args.compile_objects,
    }))


if __name__ == "__main__":
    main()
