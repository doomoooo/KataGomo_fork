#!/usr/bin/env python3
"""Execute resumable exact-19x19 SM120 tactic searches with whole-graph S2.

The build directory is configured once against stable active-slot source paths.
Each generated TileLang candidate overwrites only its family slot, so subsequent
builds compile one CUDA translation unit and relink instead of reconfiguring the
entire KataGo fat binary. Candidate acceptance data always comes from the normal
``benchmarknn`` two-server graph; no local homogeneous/mixed S2 proxy is used.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import pathlib
import shlex
import statistics
import subprocess
import sys


SLOT_FAMILIES = ("ffn", "qkv", "linear2")
SEARCH_FAMILIES = SLOT_FAMILIES + ("fa4", "l2")
SLOT_CACHE_KEYS = {
    "ffn": "SM120_SEARCH_FFN_SOURCE",
    "qkv": "SM120_SEARCH_QKV_SOURCE",
    "linear2": "SM120_SEARCH_LINEAR2_SOURCE",
}
FAT_SOURCE_CACHE_KEYS = {
    "ffn": "SM120_SEARCH_FFN_FAT_SOURCES",
    "qkv": "SM120_SEARCH_QKV_FAT_SOURCES",
    "linear2": "SM120_SEARCH_LINEAR2_FAT_SOURCES",
}
FAT_REGISTRY_CACHE_KEYS = {
    "ffn": "SM120_SEARCH_FFN_FAT_REGISTRY_SOURCE",
    "qkv": "SM120_SEARCH_QKV_FAT_REGISTRY_SOURCE",
    "linear2": "SM120_SEARCH_LINEAR2_FAT_REGISTRY_SOURCE",
}
TILELANG_FAMILIES = tuple(FAT_SOURCE_CACHE_KEYS)
TACTIC_CONFIG_KEYS = {
    "ffn": "cudaFusedFFNAotTacticSm120",
    "qkv": "cudaWideQKVAotTacticSm120",
    "linear2": "cudaLinear2AotTacticSm120",
}


def utc_now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def parse_int_set(value: str) -> list[int]:
    result: list[int] = []
    for token in value.split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            first, last = (int(item) for item in token.split("-", 1))
            result.extend(range(first, last + 1))
        else:
            result.append(int(token))
    result = sorted(set(result))
    if not result or result[0] < 1:
        raise ValueError("batch set must contain positive integers")
    return result


def last_json_object(text: str) -> dict:
    for line in reversed(text.splitlines()):
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            return json.loads(line)
    raise RuntimeError("benchmark output did not contain a JSON object")


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_checked(command: list[str], log_path: pathlib.Path) -> subprocess.CompletedProcess:
    completed = subprocess.run(command, text=True, capture_output=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.with_suffix(".out").write_text(completed.stdout)
    log_path.with_suffix(".err").write_text(completed.stderr)
    if completed.returncode != 0:
        raise RuntimeError(f"command failed ({completed.returncode}); see {log_path}.err")
    return completed


def write_result(path: pathlib.Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def initialize_active_slots(
    repo: pathlib.Path, active_dir: pathlib.Path, target_family: str,
    keep_other_slots: bool, force_all_stubs: bool = False,
) -> dict[str, pathlib.Path]:
    active_dir.mkdir(parents=True, exist_ok=True)
    result = {}
    for family in SLOT_FAMILIES:
        target = active_dir / f"active-{family}.cu"
        stub = repo / "cpp/neuralnet/tilelang_aot" / f"sm120_search_{family}_stub.cu"
        if (
            force_all_stubs
            or family == target_family
            or not target.exists()
            or (family != target_family and not keep_other_slots)
            or target.read_text().startswith(
            '#include "../cudabackend_sm120_kernels.h"'
            )
        ):
            target.write_text(stub.read_text())
        result[family] = target
    return result


def prepare_fat_bundle(
    repo: pathlib.Path, space_path: pathlib.Path, family: str,
    batches: str, candidate_ids: str, device: int, output_dir: pathlib.Path,
    s1_warmup: int, s1_iterations: int, runner: list[str],
    log_path: pathlib.Path, reuse_existing: bool,
) -> dict:
    command = runner + [
        sys.executable,
        str(repo / "python/sm120_prepare_tilelang_fat_scan.py"),
        "--space", str(space_path),
        "--family", family,
        "--batches", batches,
        "--candidate-ids", candidate_ids,
        "--device", str(device),
        "--output-dir", str(output_dir),
        "--s1-warmup", str(s1_warmup),
        "--s1-iterations", str(s1_iterations),
    ]
    if reuse_existing:
        command.append("--reuse-existing")
    run_checked(command, log_path)
    manifest_path = output_dir / "manifest.json"
    if not manifest_path.is_file():
        raise RuntimeError(f"fat-scan preparer did not produce {manifest_path}")
    return json.loads(manifest_path.read_text())


def configure_build(
    repo: pathlib.Path, build_dir: pathlib.Path, active: dict[str, pathlib.Path],
    jobs: int, logs: pathlib.Path, cmake_args: list[str],
) -> pathlib.Path:
    command = [
        "cmake", "-S", str(repo / "cpp"), "-B", str(build_dir),
        "-DUSE_BACKEND=CUDA", "-DCMAKE_BUILD_TYPE=Release",
    ]
    command.extend(
        f"-D{SLOT_CACHE_KEYS[family]}={path}" for family, path in active.items()
    )
    command.extend(cmake_args)
    command.append("-DSM120_SEARCH_QKV_OBJECT=")
    run_checked(command, logs / "configure")
    run_checked(["cmake", "--build", str(build_dir), f"-j{jobs}"], logs / "initial-build")
    binary = build_dir / "katago"
    if not binary.is_file():
        raise RuntimeError(f"build did not produce {binary}")
    return binary


def configure_qkv_slot(
    repo: pathlib.Path, build_dir: pathlib.Path, active: dict[str, pathlib.Path],
    qkv_source: pathlib.Path, qkv_object: pathlib.Path | None, jobs: int,
    logs: pathlib.Path, cmake_args: list[str], log_name: str,
) -> None:
    command = [
        "cmake", "-S", str(repo / "cpp"), "-B", str(build_dir),
        "-DUSE_BACKEND=CUDA", "-DCMAKE_BUILD_TYPE=Release",
    ]
    for family, path in active.items():
        source = qkv_source if family == "qkv" else path
        command.append(f"-D{SLOT_CACHE_KEYS[family]}={source}")
    command.extend(cmake_args)
    command.append(
        "-DSM120_SEARCH_QKV_OBJECT="
        + ("" if qkv_object is None else str(qkv_object))
    )
    run_checked(command, logs / f"{log_name}-configure")
    run_checked(
        ["cmake", "--build", str(build_dir), f"-j{jobs}"],
        logs / f"{log_name}-build",
    )


def candidate_override(family: str, candidate_value: dict) -> str:
    if candidate_value.get("implementation") == "fallback":
        return "disabled"
    return candidate_value["id"]


def full_override(
    family: str, candidate_value: dict, device: int, streams: int, extra: str,
    isolate_family: bool = False,
) -> str:
    values = [
        f"numNNServerThreadsPerModel={streams}",
        f"cudaPersistingL2StreamsSm120={streams}",
    ]
    if isolate_family and family != "l2":
        values.extend([
            "cudaUsePersistingL2Trunk=false",
            "cudaUsePersistingL2Inner=false",
        ])
    values.extend(f"cudaDeviceToUseThread{index}={device}" for index in range(streams))
    for other_family, key in TACTIC_CONFIG_KEYS.items():
        if other_family == family:
            requested = candidate_override(family, candidate_value)
        else:
            requested = "disabled" if isolate_family else "auto"
        values.append(f"{key}={requested}")
    if family == "l2":
        for key, value in candidate_value["config"].items():
            if isinstance(value, bool):
                value = str(value).lower()
            values.append(f"{key}={value}")
    if family == "fa4":
        if candidate_value.get("implementation") == "fallback":
            values.append("cudaUseFlashAttentionSm120=false")
        else:
            values.extend([
                "cudaUseFlashAttentionSm120=true",
                "cudaFlashAttentionSm120Accum=both16",
                f"cudaFlashAttentionAotTacticSm120={candidate_value['id']}",
            ])
    if family == "qkv" and candidate_value.get("output") == "packed":
        values.extend([
            "cudaUseFusedQKRoPE=true",
            "cudaUseBatchSharedRoPE=true",
            "cudaUseFusedQKRoPEHalf2Sm120=false",
            "cudaUseFlashAttentionSm120=true",
            "cudaFlashAttentionSm120Accum=both16",
        ])
    if extra.strip():
        values.extend(item.strip() for item in extra.split(",") if item.strip())
    return ",".join(values)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--space", required=True)
    parser.add_argument("--family", choices=SEARCH_FAMILIES, required=True)
    parser.add_argument("--repo", default=".")
    parser.add_argument("--build-dir", required=True)
    parser.add_argument("--active-source-dir", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--device", type=int, required=True)
    parser.add_argument("--batches", default="1-32")
    parser.add_argument("--streams", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=80)
    parser.add_argument("--warmup", type=int, default=15)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--s1-warmup", type=int, default=5)
    parser.add_argument("--s1-iterations", type=int, default=20)
    parser.add_argument("--fa4-python", default=sys.executable)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--cmake-arg", action="append", default=[])
    parser.add_argument("--runner", default="")
    parser.add_argument("--override-config", default="")
    parser.add_argument("--candidate-ids", default="")
    parser.add_argument("--keep-other-slots", action="store_true")
    parser.add_argument(
        "--isolate-family",
        action="store_true",
        help=(
            "diagnostic only: disable other AOT families and L2; the default "
            "keeps the accepted common graph active"
        ),
    )
    parser.add_argument(
        "--fat-scan",
        action="store_true",
        help="generate/link every selected TileLang batch+tactic once",
    )
    parser.add_argument(
        "--reuse-fat-generated",
        action="store_true",
        help="reuse hash-validated generated TUs from a prior fat scan",
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.fat_scan and args.family not in TILELANG_FAMILIES:
        parser.error("--fat-scan currently supports only FFN, QKV, and linear2")

    repo = pathlib.Path(args.repo).resolve()
    build_dir = pathlib.Path(args.build_dir).resolve()
    active_dir = pathlib.Path(args.active_source_dir).resolve()
    output = pathlib.Path(args.output).resolve()
    logs = output.parent / f"{output.stem}-logs"
    generated = output.parent / f"{output.stem}-generated"
    runner = shlex.split(args.runner)
    space_path = pathlib.Path(args.space).resolve()
    space = json.loads(space_path.read_text())
    if space.get("schema") != 2:
        raise ValueError("--space must be schema 2")
    batches = parse_int_set(args.batches)
    allowed_ids = {item for item in args.candidate_ids.split(",") if item}
    batch_spaces = {item["batch"]: item for item in space["batches"]}
    missing = sorted(set(batches) - set(batch_spaces))
    if missing:
        raise ValueError(f"batches outside materialized space: {missing}")

    common_graph_policy = (
        "isolated diagnostic" if args.isolate_family
        else "preserve config and exact-batch automatic winners"
    )
    regime = {
        "config": str(pathlib.Path(args.config).resolve()),
        "config_sha256": sha256_file(pathlib.Path(args.config).resolve()),
        "model": str(pathlib.Path(args.model).resolve()),
        "model_sha256": sha256_file(pathlib.Path(args.model).resolve()),
        "cuda_device_ordinal": args.device,
        "streams": args.streams,
        "iterations": args.iterations,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "runner": runner,
        "extra_override_config": args.override_config,
        "common_graph_policy": common_graph_policy,
        "fat_scan": args.fat_scan,
    }

    if output.exists():
        payload = json.loads(output.read_text())
        if payload.get("regime") != regime:
            raise ValueError(
                "refusing to mix tactic results from different measurement "
                f"regimes in {output}"
            )
    else:
        payload = {
            "schema": 1,
            "started_utc": utc_now(),
            "fixed_board": [19, 19],
            "gpu_class": space["gpu_class"],
            "streams": args.streams,
            "family": args.family,
            "space": str(space_path),
            "acceptance_metric": "natural whole-graph S2 total throughput",
            "forbidden_proxy_gates": ["homogeneous local S2", "mixed local S2"],
            "common_graph_policy": common_graph_policy,
            "regime": regime,
            "rows": [],
        }
    completed_keys = {
        (row["batch"], row["candidate_id"])
        for row in payload["rows"] if row.get("status") == "measured"
    }

    active = initialize_active_slots(
        repo, active_dir,
        args.family if args.family in SLOT_FAMILIES else "",
        args.keep_other_slots,
        force_all_stubs=args.fat_scan,
    )
    cmake_args = list(args.cmake_arg)
    fat_cache_values = {
        family: {
            "registry": str(
                repo / "cpp/neuralnet/tilelang_aot"
                / f"sm120_search_{family}_fat_stub.cu"
            ),
            "sources": "",
        }
        for family in TILELANG_FAMILIES
    }
    fat_entries = {}
    if args.fat_scan:
        fat_output_dir = generated / f"fat-{args.family}"
        fat_manifest = prepare_fat_bundle(
            repo, space_path, args.family, args.batches, args.candidate_ids,
            args.device, fat_output_dir, args.s1_warmup,
            args.s1_iterations, runner, logs / "fat-prepare",
            args.reuse_fat_generated,
        )
        fat_entries = {
            (item["batch"], item["candidate_id"]): item
            for item in fat_manifest["entries"]
        }
        fat_cache_values[args.family] = {
            "registry": fat_manifest["registry_source"],
            "sources": ";".join(fat_manifest["sources"]),
        }
        payload["fat_scan_manifest"] = str(
            (fat_output_dir / "manifest.json").resolve()
        )
        payload["build_policy"] = (
            "generate all selected TUs, configure/link once"
        )
    for family in TILELANG_FAMILIES:
        cmake_args.extend([
            f'-D{FAT_REGISTRY_CACHE_KEYS[family]}='
            f'{fat_cache_values[family]["registry"]}',
            f'-D{FAT_SOURCE_CACHE_KEYS[family]}='
            f'{fat_cache_values[family]["sources"]}',
        ])
    fa4_active_dir = active_dir / "fa4"
    fa4_bridge = fa4_active_dir / "active-fa4.cpp"
    fa4_object = fa4_active_dir / "active-fa4.o"
    if args.family == "fa4":
        fa4_active_dir.mkdir(parents=True, exist_ok=True)
        bootstrap = next(
            item for item in batch_spaces[batches[0]]["fa4"]
            if item.get("implementation") != "fallback"
        )
        command = runner + [
            args.fa4_python, str(repo / "cpp/neuralnet/fa4_aot/build_aot.py"),
            "--batch", str(batches[0]),
            "--device", str(args.device),
            "--output-dir", str(fa4_active_dir),
            "--artifact-stem", "active-fa4",
            "--symbol-prefix", "fa4_search",
            "--candidate-id", bootstrap["id"],
            "--bridge-path", str(fa4_bridge),
            "--tile-m", str(bootstrap["tile_m"]),
            "--tile-n", str(bootstrap["tile_n"]),
            "--num-stages", str(bootstrap["num_stages"]),
        ]
        run_checked(command, logs / "bootstrap-fa4")
        cmake_args.extend([
            f"-DSM120_SEARCH_FA4_SOURCE={fa4_bridge}",
            f"-DSM120_SEARCH_FA4_OBJECT={fa4_object}",
        ])
    binary = configure_build(
        repo, build_dir, active, args.jobs, logs, cmake_args,
    )
    write_result(output, payload)
    generator = repo / "python/sm120_generate_tilelang_aot.py"
    cute_qkv_generator = repo / "python/sm120_generate_cute_qkv_aot.py"
    qkv_build_mode = "planar"
    measurement_index = len(payload["rows"])

    for batch in batches:
        candidates = batch_spaces[batch][args.family]
        if allowed_ids:
            candidates = [item for item in candidates if item["id"] in allowed_ids]
        for candidate_value in candidates:
            candidate_id = candidate_value["id"]
            key = (batch, candidate_id)
            if key in completed_keys:
                continue
            measurement_index += 1
            prefix = f"{measurement_index:04d}-b{batch}-{candidate_id}"
            implementation = candidate_value.get("implementation", "tilelang")
            row = {
                "batch": batch,
                "candidate_id": candidate_id,
                "candidate": candidate_value,
                "implementation": implementation,
                "started_utc": utc_now(),
            }
            try:
                if implementation == "tilelang" and args.family in SLOT_FAMILIES:
                    if args.fat_scan:
                        fat_entry = fat_entries.get(key)
                        if fat_entry is None:
                            raise RuntimeError(
                                f"fat bundle is missing B{batch}/{candidate_id}"
                            )
                        row["fat_scan_entry"] = fat_entry
                        row["generator_metadata"] = json.loads(
                            pathlib.Path(fat_entry["metadata"]).read_text()
                        )
                        if args.family == "qkv" and qkv_build_mode != "planar":
                            configure_qkv_slot(
                                repo, build_dir, active, active["qkv"], None,
                                args.jobs, logs, cmake_args, prefix,
                            )
                            qkv_build_mode = "planar"
                    else:
                        candidate_dir = generated / f"b{batch}" / candidate_id
                        command = runner + [
                            sys.executable, str(generator),
                            "--space", str(space_path),
                            "--family", args.family,
                            "--candidate-id", candidate_id,
                            "--batch", str(batch),
                            "--device", str(args.device),
                            "--output-dir", str(candidate_dir),
                            "--source-path", str(active[args.family]),
                            "--s1-warmup", str(args.s1_warmup),
                            "--s1-iterations", str(args.s1_iterations),
                        ]
                        run_checked(command, logs / f"{prefix}-generate")
                        metadata_path = (
                            candidate_dir / f"{args.family}-{candidate_id}.json"
                        )
                        row["generator_metadata"] = json.loads(
                            metadata_path.read_text()
                        )
                        if args.family == "qkv" and qkv_build_mode != "planar":
                            configure_qkv_slot(
                                repo, build_dir, active, active["qkv"], None,
                                args.jobs, logs, cmake_args, prefix,
                            )
                            qkv_build_mode = "planar"
                        else:
                            run_checked(
                                ["cmake", "--build", str(build_dir), f"-j{args.jobs}"],
                                logs / f"{prefix}-build",
                            )
                elif implementation == "historical_tilelang" and args.family == "ffn":
                    candidate_dir = generated / f"b{batch}" / candidate_id
                    command = runner + [
                        sys.executable,
                        str(repo / "python/sm120_historical_ffn/generate.py"),
                        "--batch", str(batch),
                        "--space", str(space_path),
                        "--output-dir", str(candidate_dir),
                        "--source-path", str(active[args.family]),
                        "--candidate-id", candidate_id,
                    ]
                    run_checked(command, logs / f"{prefix}-generate")
                    metadata_path = candidate_dir / f"ffn-{candidate_id}.json"
                    row["generator_metadata"] = json.loads(metadata_path.read_text())
                    run_checked(
                        ["cmake", "--build", str(build_dir), f"-j{args.jobs}"],
                        logs / f"{prefix}-build",
                    )
                elif implementation == "cute" and args.family == "qkv":
                    candidate_dir = generated / f"b{batch}" / candidate_id
                    bridge_path = candidate_dir / "sm120_qkv_cute_active.cu"
                    command = [
                        sys.executable, str(cute_qkv_generator),
                        "--batch", str(batch),
                        "--output-dir", str(candidate_dir),
                        "--bridge-path", str(bridge_path),
                        "--candidate-id", candidate_id,
                        "--max-active-clusters",
                        str(candidate_value["max_active_clusters"]),
                    ]
                    run_checked(command, logs / f"{prefix}-generate")
                    metadata_path = candidate_dir / "sm120_qkv_cute_active.json"
                    row["generator_metadata"] = json.loads(metadata_path.read_text())
                    object_path = candidate_dir / "sm120_qkv_cute_active.o"
                    configure_qkv_slot(
                        repo, build_dir, active, bridge_path, object_path,
                        args.jobs, logs, cmake_args, prefix,
                    )
                    qkv_build_mode = "packed"
                elif args.family == "fa4" and implementation == "fa4_cute":
                    command = runner + [
                        args.fa4_python,
                        str(repo / "cpp/neuralnet/fa4_aot/build_aot.py"),
                        "--batch", str(batch),
                        "--device", str(args.device),
                        "--output-dir", str(fa4_active_dir),
                        "--artifact-stem", "active-fa4",
                        "--symbol-prefix", "fa4_search",
                        "--candidate-id", candidate_id,
                        "--bridge-path", str(fa4_bridge),
                        "--tile-m", str(candidate_value["tile_m"]),
                        "--tile-n", str(candidate_value["tile_n"]),
                        "--num-stages", str(candidate_value["num_stages"]),
                    ]
                    run_checked(command, logs / f"{prefix}-generate")
                    row["generator_metadata"] = json.loads(
                        (fa4_active_dir / "active-fa4.json").read_text()
                    )
                    run_checked(
                        ["cmake", "--build", str(build_dir), f"-j{args.jobs}"],
                        logs / f"{prefix}-build",
                    )
                elif args.family != "l2" and implementation != "fallback":
                    row.update({
                        "status": "unsupported_generator",
                        "finished_utc": utc_now(),
                        "reason": f"reproducible {implementation} generator not wired into this executor",
                    })
                    payload["rows"].append(row)
                    write_result(output, payload)
                    print(f"B{batch} {candidate_id}: unsupported generator", flush=True)
                    continue

                if (
                    args.family == "qkv"
                    and implementation == "fallback"
                    and qkv_build_mode != "planar"
                ):
                    configure_qkv_slot(
                        repo, build_dir, active, active["qkv"], None,
                        args.jobs, logs, cmake_args, prefix,
                    )
                    qkv_build_mode = "planar"

                samples = []
                benchmark_records = []
                override = full_override(
                    args.family, candidate_value, args.device, args.streams,
                    args.override_config, args.isolate_family,
                )
                for repeat in range(args.repeats):
                    command = runner + [
                        str(binary), "benchmarknn",
                        "-config", str(pathlib.Path(args.config).resolve()),
                        "-override-config", override,
                        "-model", str(pathlib.Path(args.model).resolve()),
                        "-iterations", str(args.iterations),
                        "-warmup", str(args.warmup),
                        "-batch-size", str(batch),
                        "-boardsize", "19", "-json",
                    ]
                    completed = run_checked(command, logs / f"{prefix}-r{repeat}")
                    result = last_json_object(completed.stdout)
                    samples.append(result["combinedNNEvalsPerSec"])
                    benchmark_records.append(result)
                row.update({
                    "status": "measured",
                    "finished_utc": utc_now(),
                    "binary_sha256": sha256_file(binary),
                    "nn_evals_per_sec_samples": samples,
                    "nn_evals_per_sec_median": statistics.median(samples),
                    "benchmark_records": benchmark_records,
                    "override_config": override,
                })
                completed_keys.add(key)
                print(
                    f"B{batch} {candidate_id}: {row['nn_evals_per_sec_median']:.3f} nn/s",
                    flush=True,
                )
            except Exception as error:
                row.update({
                    "status": "failed",
                    "finished_utc": utc_now(),
                    "error": str(error),
                })
                print(f"B{batch} {candidate_id}: FAILED: {error}", flush=True)
            payload["rows"].append(row)
            payload["finished_utc"] = utc_now()
            write_result(output, payload)


if __name__ == "__main__":
    main()
