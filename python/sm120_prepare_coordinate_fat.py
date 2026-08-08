#!/usr/bin/env python3
"""Prepare and build one SM120 whole-coordinate fat binary.

The ordinary active-slot workflow rebuilds KataGo for every candidate.  This
preparer combines all exact ``(batch, tactic ID)`` implementations needed by
coordinate search into one executable.  Runtime config selects the requested
tactic, so discovery and the later joint gate reuse one binary SHA.

Existing hash-validated TileLang fat manifests are reused by exact candidate
projection. Historical FFN, CuTe QKV, and FA4 candidates are generated with
unique symbols and cached in this bundle directory.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import pathlib
import subprocess
import sys
import time
from collections.abc import Iterable

try:
    from sm120_fat_scan import launch_symbol, symbol_token, write_registry
    from sm120_run_tactic_search import parse_int_set
except ModuleNotFoundError:
    from python.sm120_fat_scan import launch_symbol, symbol_token, write_registry
    from python.sm120_run_tactic_search import parse_int_set


FAT_FAMILIES = ("ffn", "qkv", "linear2", "fa4")
TILELANG_FAMILIES = ("ffn", "qkv", "linear2")


def utc_now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: pathlib.Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def run_logged(
    command: list[str], log_prefix: pathlib.Path, *, env: dict[str, str] | None = None,
) -> dict:
    log_prefix.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    completed = subprocess.run(
        command, text=True, capture_output=True, env=env,
    )
    elapsed = time.monotonic() - started
    log_prefix.with_suffix(".out").write_text(completed.stdout)
    log_prefix.with_suffix(".err").write_text(completed.stderr)
    if completed.returncode != 0:
        raise RuntimeError(
            f"command failed ({completed.returncode}); see "
            f"{log_prefix.with_suffix('.err')}"
        )
    return {"command": command, "elapsed_seconds": elapsed}


def batch_map(space: dict) -> dict[int, dict]:
    return {int(item["batch"]): item for item in space.get("batches", [])}


def implementation(candidate: dict) -> str:
    return candidate.get("implementation", "tilelang")


def requests_for(
    space: dict, batches: Iterable[int], family: str,
    implementations: set[str],
) -> list[dict]:
    by_batch = batch_map(space)
    requests = []
    for batch in batches:
        if batch not in by_batch:
            raise ValueError(f"search space has no B{batch}")
        for candidate in by_batch[batch].get(family, []):
            if implementation(candidate) not in implementations:
                continue
            token = symbol_token(family, batch, candidate["id"])
            requests.append({
                "family": family,
                "batch": batch,
                "candidate_id": candidate["id"],
                "candidate": candidate,
                "implementation": implementation(candidate),
                "symbol_token": token,
                "launch_symbol": launch_symbol(family, token),
            })
    return requests


def exact_key(item: dict) -> tuple[int, str]:
    return int(item["batch"]), str(item["candidate_id"])


def validate_file(path_value: str, expected_sha256: str, label: str) -> pathlib.Path:
    path = pathlib.Path(path_value).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    actual = sha256_file(path)
    if actual != expected_sha256:
        raise ValueError(f"{label} hash mismatch: {actual} != {expected_sha256}")
    return path


def load_tilelang_entries(
    manifest_path: pathlib.Path, family: str, space: dict, batches: list[int],
) -> list[dict]:
    manifest = json.loads(manifest_path.read_text())
    if not manifest.get("complete") or manifest.get("family") != family:
        raise ValueError(f"incomplete or wrong-family manifest: {manifest_path}")
    expected = {
        exact_key(item): item
        for item in requests_for(space, batches, family, {"tilelang"})
    }
    available = {
        exact_key(item): item for item in manifest.get("entries", [])
    }
    missing = sorted(set(expected) - set(available))
    if missing:
        raise ValueError(f"TileLang manifest misses {family} entries: {missing}")
    result = []
    for key, request in expected.items():
        prior = available[key]
        if prior.get("candidate") != request["candidate"]:
            raise ValueError(
                f"TileLang candidate drift for {family}/B{key[0]}/{key[1]}"
            )
        source = validate_file(
            prior["source"], prior["source_sha256"], "TileLang source",
        )
        metadata = validate_file(
            prior["metadata"], prior["metadata_sha256"], "TileLang metadata",
        )
        if (
            prior.get("symbol_token") != request["symbol_token"] or
            prior.get("launch_symbol") != request["launch_symbol"]
        ):
            raise ValueError(f"TileLang fat ABI drift for {family}/{key}")
        result.append({
            **request,
            "source": str(source),
            "source_sha256": sha256_file(source),
            "metadata": str(metadata),
            "metadata_sha256": sha256_file(metadata),
            "source_manifest": str(manifest_path),
            "source_manifest_sha256": sha256_file(manifest_path),
            "reused": True,
        })
    return result


def reusable_generated(
    metadata_path: pathlib.Path, request: dict, files: dict[str, pathlib.Path],
) -> dict | None:
    if not metadata_path.is_file() or not all(path.is_file() for path in files.values()):
        return None
    metadata = json.loads(metadata_path.read_text())
    if (
        int(metadata.get("batch", -1)) != request["batch"] or
        metadata.get("candidate_id", metadata.get("candidate", {}).get("id"))
            != request["candidate_id"] or
        metadata.get("fat_symbol_token") != request["symbol_token"] or
        metadata.get("launch_symbol") != request["launch_symbol"]
    ):
        return None
    if metadata.get("sourceSha256") and "source" in files:
        if metadata["sourceSha256"] != sha256_file(files["source"]):
            return None
    recorded = metadata.get("sha256", {})
    for name, metadata_name in (
        ("source", "bridge"), ("header", "header"), ("object", "object"),
    ):
        if name in files and recorded.get(metadata_name):
            if recorded[metadata_name] != sha256_file(files[name]):
                return None
    return metadata


def generated_entry(
    request: dict, metadata_path: pathlib.Path, files: dict[str, pathlib.Path],
    command: list[str] | None,
) -> dict:
    return {
        **request,
        **{name: str(path.resolve()) for name, path in files.items()},
        **{f"{name}_sha256": sha256_file(path) for name, path in files.items()},
        "metadata": str(metadata_path.resolve()),
        "metadata_sha256": sha256_file(metadata_path),
        "generation_command": command,
        "reused": command is None,
    }


def prepare_historical_ffn(
    args: argparse.Namespace, request: dict, output_dir: pathlib.Path,
    logs: pathlib.Path,
) -> dict:
    token = request["symbol_token"]
    directory = output_dir / "historical-ffn" / token
    source = directory / f"{token}.cu"
    metadata = directory / f"ffn-{request['candidate_id']}.json"
    files = {"source": source}
    if reusable_generated(metadata, request, files):
        return generated_entry(request, metadata, files, None)
    command = [
        args.generator_python,
        str(args.repo / "python/sm120_historical_ffn/generate.py"),
        "--batch", str(request["batch"]),
        "--space", str(args.space),
        "--output-dir", str(directory),
        "--source-path", str(source),
        "--candidate-id", request["candidate_id"],
        "--fat-symbol-token", token,
    ]
    run_logged(command, logs / f"historical-{token}")
    if not reusable_generated(metadata, request, files):
        raise RuntimeError(f"historical generator emitted invalid artifacts for {token}")
    return generated_entry(request, metadata, files, command)


def prepare_cute_qkv(
    args: argparse.Namespace, request: dict, output_dir: pathlib.Path,
    logs: pathlib.Path,
) -> dict:
    token = request["symbol_token"]
    directory = output_dir / "qkv-cute" / token
    bridge = directory / f"{token}.cpp"
    header = directory / f"{token}.h"
    obj = directory / f"{token}.o"
    metadata = directory / f"{token}.json"
    files = {"source": bridge, "header": header, "object": obj}
    if reusable_generated(metadata, request, files):
        return generated_entry(request, metadata, files, None)
    command = [
        args.generator_python,
        str(args.repo / "python/sm120_generate_cute_qkv_aot.py"),
        "--batch", str(request["batch"]),
        "--output-dir", str(directory),
        "--artifact-stem", token,
        "--bridge-path", str(bridge),
        "--candidate-id", request["candidate_id"],
        "--fat-symbol-token", token,
        "--device", str(args.device),
        "--cutlass-root", str(args.cutlass_root),
    ]
    max_clusters = request["candidate"].get("max_active_clusters")
    if max_clusters is not None:
        command.extend(["--max-active-clusters", str(max_clusters)])
    run_logged(command, logs / f"qkv-cute-{token}")
    if not reusable_generated(metadata, request, files):
        raise RuntimeError(f"CuTe QKV generator emitted invalid artifacts for {token}")
    return generated_entry(request, metadata, files, command)


def prepare_fa4(
    args: argparse.Namespace, request: dict, output_dir: pathlib.Path,
    logs: pathlib.Path,
) -> dict:
    token = request["symbol_token"]
    directory = output_dir / "fa4" / token
    bridge = directory / f"{token}.cpp"
    header = directory / f"{token}.h"
    obj = directory / f"{token}.o"
    metadata = directory / f"{token}.json"
    files = {"source": bridge, "header": header, "object": obj}
    if reusable_generated(metadata, request, files):
        return generated_entry(request, metadata, files, None)
    candidate = request["candidate"]
    command = [
        args.fa4_python,
        str(args.repo / "cpp/neuralnet/fa4_aot/build_aot.py"),
        "--batch", str(request["batch"]),
        "--device", str(args.device),
        "--output-dir", str(directory),
        "--artifact-stem", token,
        "--symbol-prefix", token,
        "--candidate-id", request["candidate_id"],
        "--bridge-path", str(bridge),
        "--fat-symbol-token", token,
        "--tile-m", str(candidate["tile_m"]),
        "--tile-n", str(candidate["tile_n"]),
        "--num-stages", str(candidate["num_stages"]),
    ]
    env = dict(os.environ)
    env.update({"FA4_QK_ACC": "fp16", "FA4_PV_ACC": "fp16"})
    run_logged(command, logs / f"fa4-{token}", env=env)
    if not reusable_generated(metadata, request, files):
        raise RuntimeError(f"FA4 generator emitted invalid artifacts for {token}")
    return generated_entry(request, metadata, files, command)


def validate_coverage(space: dict, batches: list[int], entries: list[dict]) -> None:
    actual = {(item["family"], *exact_key(item)) for item in entries}
    expected = set()
    for family in FAT_FAMILIES:
        for item in requests_for(
            space, batches, family,
            {"tilelang", "historical_tilelang", "cute", "fa4_cute"},
        ):
            expected.add((family, *exact_key(item)))
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    if missing or extra:
        raise ValueError(f"fat bundle coverage mismatch; missing={missing}, extra={extra}")


def cmake_list(values: Iterable[str]) -> str:
    return ";".join(values)


def build_commands(
    args: argparse.Namespace, entries: list[dict], registries: dict[str, pathlib.Path],
) -> tuple[list[str], list[str]]:
    by_family = {
        family: [item for item in entries if item["family"] == family]
        for family in FAT_FAMILIES
    }
    tile_sources = {
        family: [
            item["source"] for item in by_family[family]
            if item["implementation"] in ("tilelang", "historical_tilelang")
        ]
        for family in TILELANG_FAMILIES
    }
    qkv_cute = [
        item for item in by_family["qkv"] if item["implementation"] == "cute"
    ]
    fa4 = by_family["fa4"]
    configure = [
        "cmake", "-S", str(args.repo / "cpp"), "-B", str(args.build_dir),
        "-DUSE_BACKEND=CUDA", "-DCMAKE_BUILD_TYPE=Release",
        "-DKATAGO_CUDA_ARCHITECTURES=120",
        *args.cmake_arg,
        f"-DSM120_SEARCH_FFN_FAT_SOURCES={cmake_list(tile_sources['ffn'])}",
        f"-DSM120_SEARCH_QKV_FAT_SOURCES={cmake_list(tile_sources['qkv'])}",
        f"-DSM120_SEARCH_LINEAR2_FAT_SOURCES={cmake_list(tile_sources['linear2'])}",
        f"-DSM120_SEARCH_FFN_FAT_REGISTRY_SOURCE={registries['ffn']}",
        f"-DSM120_SEARCH_QKV_FAT_REGISTRY_SOURCE={registries['qkv']}",
        f"-DSM120_SEARCH_LINEAR2_FAT_REGISTRY_SOURCE={registries['linear2']}",
        f"-DSM120_SEARCH_QKV_FAT_BRIDGE_SOURCES={cmake_list(item['source'] for item in qkv_cute)}",
        f"-DSM120_SEARCH_QKV_FAT_OBJECTS={cmake_list(item['object'] for item in qkv_cute)}",
        f"-DSM120_SEARCH_FA4_FAT_SOURCES={cmake_list(item['source'] for item in fa4)}",
        f"-DSM120_SEARCH_FA4_FAT_OBJECTS={cmake_list(item['object'] for item in fa4)}",
        f"-DSM120_SEARCH_FA4_FAT_REGISTRY_SOURCE={registries['fa4']}",
    ]
    build = ["cmake", "--build", str(args.build_dir), f"-j{args.jobs}"]
    return configure, build


def load_coordinate_fat_bundle(
    manifest_path: pathlib.Path, space: dict, space_path: pathlib.Path,
    batches: list[int],
) -> dict:
    manifest_path = manifest_path.resolve()
    manifest = json.loads(manifest_path.read_text())
    if (
        manifest.get("schema") != 1 or
        manifest.get("kind") != "sm120-coordinate-fat-bundle" or
        not manifest.get("complete")
    ):
        raise ValueError(f"incomplete coordinate fat bundle: {manifest_path}")
    if sorted(manifest.get("batches", [])) != sorted(batches):
        raise ValueError("fat bundle batch coverage differs from this scan")
    entries = manifest.get("entries", [])
    validate_coverage(space, batches, entries)
    for item in entries:
        for field in ("source", "metadata"):
            validate_file(item[field], item[f"{field}_sha256"], field)
        for field in ("header", "object"):
            if field in item:
                validate_file(item[field], item[f"{field}_sha256"], field)
    binary = validate_file(
        manifest["binary"], manifest["binary_sha256"], "fat binary",
    )
    # Space-level metadata may grow, but every implementation-bearing exact
    # candidate is checked byte-for-byte by validate_coverage and the entries.
    current = {
        (family, item["batch"], candidate["id"]): candidate
        for item in space.get("batches", [])
        for family in FAT_FAMILIES
        for candidate in item.get(family, [])
        if implementation(candidate) != "fallback"
    }
    for item in entries:
        key = (item["family"], item["batch"], item["candidate_id"])
        if current.get(key) != item.get("candidate"):
            raise ValueError(f"fat bundle candidate drift: {key}")
    manifest["_path"] = str(manifest_path)
    manifest["_sha256"] = sha256_file(manifest_path)
    manifest["_binary"] = str(binary)
    manifest["_entries"] = {
        (item["family"], item["batch"], item["candidate_id"]): item
        for item in entries
    }
    manifest["loaded_space"] = str(space_path.resolve())
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=pathlib.Path, default=".")
    parser.add_argument("--space", type=pathlib.Path, required=True)
    parser.add_argument("--batches", default="4-32")
    parser.add_argument("--device", type=int, required=True)
    parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    parser.add_argument("--build-dir", type=pathlib.Path, required=True)
    parser.add_argument("--jobs", type=int, default=len(os.sched_getaffinity(0)))
    parser.add_argument(
        "--cmake-arg", action="append", default=[],
        help=(
            "extra configure argument recorded in the bundle manifest; repeat "
            "as --cmake-arg=-DNAME=VALUE for non-system CUDA/cuDNN toolchains"
        ),
    )
    parser.add_argument("--generator-python", default=sys.executable)
    parser.add_argument("--fa4-python", default=sys.executable)
    parser.add_argument("--cutlass-root", type=pathlib.Path, required=True)
    parser.add_argument("--tilelang-ffn-manifest", type=pathlib.Path, required=True)
    parser.add_argument("--tilelang-qkv-manifest", type=pathlib.Path, required=True)
    parser.add_argument(
        "--tilelang-linear2-manifest", type=pathlib.Path, required=True,
    )
    args = parser.parse_args()
    args.repo = args.repo.resolve()
    args.space = args.space.resolve()
    args.output_dir = args.output_dir.resolve()
    args.build_dir = args.build_dir.resolve()
    args.cutlass_root = args.cutlass_root.resolve()
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    batches = parse_int_set(args.batches)
    space = json.loads(args.space.read_text())
    output_dir = args.output_dir
    logs = output_dir / "logs"
    started = utc_now()

    entries = []
    for family, manifest_path in (
        ("ffn", args.tilelang_ffn_manifest),
        ("qkv", args.tilelang_qkv_manifest),
        ("linear2", args.tilelang_linear2_manifest),
    ):
        entries.extend(load_tilelang_entries(
            manifest_path.resolve(), family, space, batches,
        ))
    for request in requests_for(
        space, batches, "ffn", {"historical_tilelang"},
    ):
        entries.append(prepare_historical_ffn(args, request, output_dir, logs))
        print(f"prepared historical FFN B{request['batch']}", flush=True)
    for request in requests_for(space, batches, "qkv", {"cute"}):
        entries.append(prepare_cute_qkv(args, request, output_dir, logs))
        print(f"prepared CuTe QKV B{request['batch']}", flush=True)
    for request in requests_for(space, batches, "fa4", {"fa4_cute"}):
        entries.append(prepare_fa4(args, request, output_dir, logs))
        print(f"prepared FA4 {request['candidate_id']}", flush=True)
    validate_coverage(space, batches, entries)

    registries = {}
    registry_dir = output_dir / "registries"
    for family in FAT_FAMILIES:
        suffix = ".cpp" if family == "fa4" else ".cu"
        path = registry_dir / f"sm120_search_{family}_fat_registry{suffix}"
        write_registry(path, family, [
            item for item in entries if item["family"] == family
        ])
        registries[family] = path.resolve()

    configure, build = build_commands(args, entries, registries)
    payload = {
        "schema": 1,
        "kind": "sm120-coordinate-fat-bundle",
        "started_utc": started,
        "updated_utc": utc_now(),
        "complete": False,
        "space": str(args.space),
        "space_sha256": sha256_file(args.space),
        "batches": batches,
        "device_used_for_generation": args.device,
        "entries": entries,
        "registries": {
            family: {"path": str(path), "sha256": sha256_file(path)}
            for family, path in registries.items()
        },
        "commands": {"configure": configure, "build": build},
        "build_dir": str(args.build_dir),
    }
    manifest_path = output_dir / "manifest.json"
    write_json(manifest_path, payload)
    configure_result = run_logged(configure, logs / "configure")
    build_result = run_logged(build, logs / "build")
    binary = args.build_dir / "katago"
    if not binary.is_file():
        raise RuntimeError(f"build did not produce {binary}")
    payload.update({
        "updated_utc": utc_now(),
        "finished_utc": utc_now(),
        "complete": True,
        "binary": str(binary.resolve()),
        "binary_sha256": sha256_file(binary),
        "timing": {"configure": configure_result, "build": build_result},
    })
    write_json(manifest_path, payload)
    print(json.dumps({
        "manifest": str(manifest_path),
        "binary": str(binary),
        "binary_sha256": payload["binary_sha256"],
        "entries": len(entries),
    }))


if __name__ == "__main__":
    main()
