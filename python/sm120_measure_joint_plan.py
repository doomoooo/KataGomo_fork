#!/usr/bin/env python3
"""Measure one exact-batch SM120 tactic plan on the real whole graph.

This runner pins every family in the override and then runs the normal
two-server ``benchmarknn`` graph. A coordinate fat bundle reuses one executable
for every batch; legacy active-slot materialization remains available as a
compatibility fallback. It is intentionally resumable: a completed batch is
retained in the JSON evidence file.

This is a producer-side measurement tool.  The portable plan remains the
source of truth for selection; this file records the joint full-graph evidence
needed before accepting the plan as a smooth deployment curve.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import pathlib
import shlex
import shutil
import subprocess
import sys

try:
    from sm120_benchmark_metrics import benchmark_throughput, summarize_throughput
    from sm120_run_tactic_search import (
        collect_environment,
        implementation_identity,
        last_json_object,
        parse_int_set,
        reproducibility_identity,
        sha256_file,
    )
    from sm120_tactic_plan import (
        FAMILIES, load_plan, validate_coordinate_coverage, validate_plan,
    )
    from sm120_prepare_coordinate_fat import (
        FAT_FAMILIES, load_coordinate_fat_bundle,
    )
except ModuleNotFoundError:  # imported as python.sm120_measure_joint_plan
    from python.sm120_benchmark_metrics import benchmark_throughput, summarize_throughput
    from python.sm120_run_tactic_search import (
        collect_environment,
        implementation_identity,
        last_json_object,
        parse_int_set,
        reproducibility_identity,
        sha256_file,
    )
    from python.sm120_tactic_plan import (
        FAMILIES, load_plan, validate_coordinate_coverage, validate_plan,
    )
    from python.sm120_prepare_coordinate_fat import (
        FAT_FAMILIES, load_coordinate_fat_bundle,
    )


SLOT_FAMILIES = ("ffn", "qkv", "linear2")
DEFAULT_HISTORY_ROOT = (
    "/workspace/results/rebuild/cross-batch-search/"
    "historical-ffn-static-selftest"
)
DEFAULT_QKV_ROOT = (
    "/workspace/results/rebuild/cross-batch-search/"
    "s2-5090d-b4-32-qkv-cute-generated"
)
DEFAULT_SPACE = (
    "/workspace/results/rebuild/cross-batch-search/"
    "space-5090d-b4-32-s2-v6.json"
)
DEFAULT_PLAN = (
    "/workspace/results/rebuild/cross-batch-search/"
    "tactic-plan-5090d-b4-32.json"
)


def utc_now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def write_json(path: pathlib.Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def run_command(
    command: list[str], log_prefix: pathlib.Path, *, check: bool = True,
) -> subprocess.CompletedProcess:
    log_prefix.parent.mkdir(parents=True, exist_ok=True)
    completed = subprocess.run(command, text=True, capture_output=True)
    log_prefix.with_suffix(".out").write_text(completed.stdout)
    log_prefix.with_suffix(".err").write_text(completed.stderr)
    if check and completed.returncode != 0:
        raise RuntimeError(
            f"command failed ({completed.returncode}); see "
            f"{log_prefix.with_suffix('.err')}"
        )
    return completed


def copy_file(source: pathlib.Path, target: pathlib.Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(source)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_file() and source.read_bytes() == target.read_bytes():
        return
    shutil.copyfile(source, target)


def candidate(plan: dict, family: str, batch: int) -> dict:
    entry = plan["families"][family]["batches"].get(str(batch))
    if not isinstance(entry, dict):
        raise ValueError(f"plan has no {family}/B{batch}")
    value = entry.get("candidate")
    if not isinstance(value, dict):
        raise ValueError(f"plan has no candidate parameters for {family}/B{batch}")
    return value


def plan_entry(plan: dict, family: str, batch: int) -> dict:
    return plan["families"][family]["batches"][str(batch)]


def source_from_plan(entry: dict) -> pathlib.Path | None:
    metadata = entry.get("generator_metadata")
    if isinstance(metadata, dict) and metadata.get("source"):
        path = pathlib.Path(metadata["source"])
        if path.is_file():
            return path
    fat_entry = entry.get("fat_scan_entry")
    if isinstance(fat_entry, dict) and fat_entry.get("source"):
        path = pathlib.Path(fat_entry["source"])
        if path.is_file():
            return path
    return None


def prepare_ffn(
    repo: pathlib.Path, plan: dict, batch: int, active: pathlib.Path,
    history_root: pathlib.Path, python_executable: str, runner: list[str],
    logs: pathlib.Path, space: pathlib.Path, device: int,
) -> dict:
    value = candidate(plan, "ffn", batch)
    implementation = value.get("implementation", "tilelang")
    active_source = active / "active-ffn.cu"
    if implementation == "fallback":
        source = repo / "cpp/neuralnet/tilelang_aot/sm120_search_ffn_stub.cu"
        copy_file(source, active_source)
        return {"candidate": value, "source": str(source)}
    if implementation == "historical_tilelang":
        source = history_root / f"b{batch}-ffn" / f"ffn-{value['id']}.cu"
        if source.is_file():
            copy_file(source, active_source)
            return {
                "candidate": value,
                "source": str(source),
                "source_sha256": sha256_file(source),
            }
        output_dir = active / "generated" / f"ffn-b{batch}-{value['id']}"
        command = runner + [
            python_executable,
            str(repo / "python/sm120_historical_ffn/generate.py"),
            "--batch", str(batch), "--space", str(space),
            "--output-dir", str(output_dir),
            "--source-path", str(active_source),
            "--candidate-id", value["id"],
        ]
    elif implementation == "tilelang":
        output_dir = active / "generated" / f"ffn-b{batch}-{value['id']}"
        command = runner + [
            python_executable,
            str(repo / "python/sm120_generate_tilelang_aot.py"),
            "--space", str(space), "--family", "ffn",
            "--candidate-id", value["id"], "--batch", str(batch),
            "--device", str(device), "--output-dir", str(output_dir),
            "--source-path", str(active_source),
        ]
    else:
        raise ValueError(f"unsupported FFN implementation at B{batch}: {implementation}")
    run_command(command, logs / f"b{batch}-ffn-{value['id']}-generate")
    return {
        "candidate": value,
        "source": str(active_source),
        "source_sha256": sha256_file(active_source),
        "generate_command": command,
    }


def prepare_qkv(
    repo: pathlib.Path, plan: dict, batch: int, active: pathlib.Path,
    generated_root: pathlib.Path, python_executable: str,
    runner: list[str], logs: pathlib.Path, space: pathlib.Path,
    device: int, cutlass_root: pathlib.Path,
) -> dict:
    value = candidate(plan, "qkv", batch)
    stub = repo / "cpp/neuralnet/tilelang_aot/sm120_search_qkv_stub.cu"
    active_source = active / "active-qkv.cu"
    # The ordinary slot source is the stub by default.  For CuTe, the
    # configure step swaps it for the generated bridge, while the embedded
    # object supplies the device module implementation.
    copy_file(stub, active_source)
    if value.get("implementation") == "fallback":
        return {"candidate": value, "source": str(stub), "object": None}
    if value.get("implementation") == "cute":
        directory = generated_root / f"b{batch}" / value["id"]
        generated_source = directory / "sm120_qkv_cute_active.cu"
        generated_header = directory / "sm120_qkv_cute_active.h"
        generated_object = directory / "sm120_qkv_cute_active.o"
        if not (generated_source.is_file() and generated_header.is_file() and generated_object.is_file()):
            command = runner + [
                python_executable,
                str(repo / "python/sm120_generate_cute_qkv_aot.py"),
                "--batch", str(batch), "--output-dir", str(directory),
                "--bridge-path", str(generated_source),
                "--candidate-id", value["id"], "--device", str(device),
                "--cutlass-root", str(cutlass_root),
            ]
            if value.get("max_active_clusters") is not None:
                command.extend([
                    "--max-active-clusters", str(value["max_active_clusters"]),
                ])
            run_command(command, logs / f"b{batch}-qkv-{value['id']}-generate")
        copy_file(generated_source, active_source)
        copy_file(generated_header, active / "sm120_qkv_cute_active.h")
        copy_file(generated_object, active / "active-qkv-cute.o")
        result = {
            "candidate": value,
            "source": str(generated_source),
            "object": str(generated_object),
            "bridge_source": str(active_source),
            "source_sha256": sha256_file(generated_source),
            "object_sha256": sha256_file(generated_object),
        }
        if "command" in locals():
            result["generate_command"] = command
        return result
    if value.get("implementation") != "tilelang":
        raise ValueError(f"unsupported QKV implementation at B{batch}")

    # The existing coarse scan source is a fat-launch TU.  Regenerate the
    # selected candidate with the ordinary single-slot ABI for this joint
    # executable instead of trying to reinterpret its registry symbols.
    output_dir = active / "generated" / f"qkv-b{batch}"
    command = runner + [
        python_executable,
        str(repo / "python/sm120_generate_tilelang_aot.py"),
        "--space", str(space), "--family", "qkv",
        "--candidate-id", value["id"], "--batch", str(batch),
        "--device", str(device), "--output-dir", str(output_dir),
        "--source-path", str(active_source),
    ]
    run_command(command, logs / f"b{batch}-qkv-generate")
    return {
        "candidate": value,
        "source": str(active_source),
        "source_sha256": sha256_file(active_source),
        "generate_command": command,
    }


def prepare_linear2(
    repo: pathlib.Path, plan: dict, batch: int, active: pathlib.Path,
    python_executable: str, runner: list[str], logs: pathlib.Path,
    space: pathlib.Path, device: int,
) -> dict:
    value = candidate(plan, "linear2", batch)
    stub = repo / "cpp/neuralnet/tilelang_aot/sm120_search_linear2_stub.cu"
    active_source = active / "active-linear2.cu"
    if value.get("implementation") == "fallback":
        copy_file(stub, active_source)
        return {"candidate": value, "source": str(stub)}
    if value.get("implementation") != "tilelang":
        raise ValueError(f"unsupported Linear2 implementation at B{batch}")
    output_dir = active / "generated" / f"linear2-b{batch}"
    command = runner + [
        python_executable,
        str(repo / "python/sm120_generate_tilelang_aot.py"),
        "--space", str(space), "--family", "linear2",
        "--candidate-id", value["id"], "--batch", str(batch),
        "--device", str(device), "--output-dir", str(output_dir),
        "--source-path", str(active_source),
    ]
    run_command(command, logs / f"b{batch}-linear2-generate")
    return {
        "candidate": value,
        "source": str(active_source),
        "source_sha256": sha256_file(active_source),
        "generate_command": command,
    }


def prepare_fa4(
    repo: pathlib.Path, plan: dict, batch: int, active: pathlib.Path,
    fa4_python: str, runner: list[str], logs: pathlib.Path, device: int,
) -> dict:
    value = candidate(plan, "fa4", batch)
    fa4_dir = active / "fa4"
    bridge = fa4_dir / "active-fa4.cpp"
    obj = fa4_dir / "active-fa4.o"
    if value.get("implementation") == "fallback":
        stub = repo / "cpp/neuralnet/fa4_aot/sm120_search_fa4_stub.cpp"
        copy_file(stub, bridge)
        return {"candidate": value, "source": str(stub), "object": None}
    if value.get("implementation") != "fa4_cute":
        raise ValueError(f"unsupported FA4 implementation at B{batch}")
    command = runner + [
        fa4_python, str(repo / "cpp/neuralnet/fa4_aot/build_aot.py"),
        "--batch", str(batch), "--device", str(device),
        "--output-dir", str(fa4_dir), "--artifact-stem", "active-fa4",
        "--symbol-prefix", "fa4_search", "--candidate-id", value["id"],
        "--bridge-path", str(bridge), "--tile-m", str(value["tile_m"]),
        "--tile-n", str(value["tile_n"]), "--num-stages", str(value["num_stages"]),
    ]
    run_command(command, logs / f"b{batch}-fa4-generate")
    if not obj.is_file():
        raise FileNotFoundError(obj)
    return {
        "candidate": value,
        "source": str(bridge),
        "object": str(obj),
        "source_sha256": sha256_file(bridge),
        "object_sha256": sha256_file(obj),
        "generate_command": command,
    }


def override_for(plan: dict, batch: int, device: int, streams: int) -> str:
    values = [
        f"numNNServerThreadsPerModel={streams}",
        f"cudaPersistingL2StreamsSm120={streams}",
        *(f"cudaDeviceToUseThread{i}={device}" for i in range(streams)),
    ]
    for family, key in (
        ("ffn", "cudaFusedFFNAotTacticSm120"),
        ("qkv", "cudaWideQKVAotTacticSm120"),
        ("linear2", "cudaLinear2AotTacticSm120"),
    ):
        value = candidate(plan, family, batch)
        requested = "disabled" if value.get("implementation") == "fallback" else value["id"]
        values.append(f"{key}={requested}")

    l2 = candidate(plan, "l2", batch)
    for key, value in l2.get("config", {}).items():
        if isinstance(value, bool):
            value = str(value).lower()
        values.append(f"{key}={value}")

    qkv = candidate(plan, "qkv", batch)
    if qkv.get("output") == "packed":
        values.extend([
            "cudaUseFusedQKRoPE=true",
            "cudaUseBatchSharedRoPE=true",
            "cudaUseFusedQKRoPEHalf2Sm120=false",
        ])

    fa4 = candidate(plan, "fa4", batch)
    if fa4.get("implementation") == "fallback":
        values.append("cudaUseFlashAttentionSm120=false")
    else:
        values.extend([
            "cudaUseFlashAttentionSm120=true",
            "cudaFlashAttentionSm120Accum=both16",
            f"cudaFlashAttentionAotTacticSm120={fa4['id']}",
        ])
    return ",".join(values)


def configure_command(
    repo: pathlib.Path, build: pathlib.Path, active: pathlib.Path,
    fa4: dict, qkv: dict,
) -> list[str]:
    source_dir = repo / "cpp/neuralnet/tilelang_aot"
    qkv_source = qkv.get("bridge_source") or str(active / "active-qkv.cu")
    args = [
        "cmake", "-S", str(repo / "cpp"), "-B", str(build),
        "-DUSE_BACKEND=CUDA", "-DCMAKE_BUILD_TYPE=Release",
        "-DKATAGO_CUDA_ARCHITECTURES=120",
        f"-DSM120_SEARCH_FFN_SOURCE={active / 'active-ffn.cu'}",
        f"-DSM120_SEARCH_QKV_SOURCE={qkv_source}",
        f"-DSM120_SEARCH_LINEAR2_SOURCE={active / 'active-linear2.cu'}",
        f"-DSM120_SEARCH_FA4_SOURCE={fa4['source']}",
        f"-DSM120_SEARCH_FA4_OBJECT={fa4.get('object') or ''}",
        f"-DSM120_SEARCH_FFN_FAT_REGISTRY_SOURCE={source_dir / 'sm120_search_ffn_fat_stub.cu'}",
        f"-DSM120_SEARCH_QKV_FAT_REGISTRY_SOURCE={source_dir / 'sm120_search_qkv_fat_stub.cu'}",
        f"-DSM120_SEARCH_LINEAR2_FAT_REGISTRY_SOURCE={source_dir / 'sm120_search_linear2_fat_stub.cu'}",
        "-DSM120_SEARCH_FFN_FAT_SOURCES=",
        "-DSM120_SEARCH_QKV_FAT_SOURCES=",
        "-DSM120_SEARCH_LINEAR2_FAT_SOURCES=",
        "-DSM120_SEARCH_QKV_OBJECT=" + (str(active / "active-qkv-cute.o") if qkv.get("object") else ""),
    ]
    return args


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", default=DEFAULT_PLAN)
    parser.add_argument("--space", default=DEFAULT_SPACE)
    parser.add_argument("--repo", default=".")
    parser.add_argument("--build-dir", default="")
    parser.add_argument("--active-dir", default="")
    parser.add_argument(
        "--fat-bundle", default="",
        help="reuse the coordinate fat binary without per-batch builds",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--device", type=int, required=True)
    parser.add_argument("--batches", default="4-32")
    parser.add_argument("--streams", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--runner", default="")
    parser.add_argument("--generator-python", default="")
    parser.add_argument("--fa4-python", default="")
    parser.add_argument("--historical-root", default=DEFAULT_HISTORY_ROOT)
    parser.add_argument("--qkv-generated-root", default=DEFAULT_QKV_ROOT)
    parser.add_argument(
        "--cutlass-root", default="/workspace/third_party/cutlass",
        help="pinned CUTLASS source used when a CuTe QKV object must be generated",
    )
    args = parser.parse_args()

    repo = pathlib.Path(args.repo).resolve()
    plan_path = pathlib.Path(args.plan).resolve()
    space_path = pathlib.Path(args.space).resolve()
    config_path = pathlib.Path(args.config).resolve()
    model_path = pathlib.Path(args.model).resolve()
    fat_bundle_path = pathlib.Path(args.fat_bundle).resolve() if args.fat_bundle else None
    if fat_bundle_path is None and (not args.build_dir or not args.active_dir):
        parser.error("active-slot mode requires --build-dir and --active-dir")
    build_dir = pathlib.Path(args.build_dir).resolve() if args.build_dir else None
    active_dir = pathlib.Path(args.active_dir).resolve() if args.active_dir else None
    output_path = pathlib.Path(args.output).resolve()
    logs = output_path.parent / f"{output_path.stem}-logs"
    runner = shlex.split(args.runner)
    generator_python = args.generator_python or (
        "/workspace/venv/bin/python3"
        if pathlib.Path("/workspace/venv/bin/python3").is_file()
        else sys.executable
    )
    fa4_python = args.fa4_python or generator_python
    try:
        from sm120_device import query_cuda_device
    except ModuleNotFoundError:
        from python.sm120_device import query_cuda_device
    device_properties = query_cuda_device(args.device)
    batches = parse_int_set(args.batches)
    plan = load_plan(plan_path)
    space = json.loads(space_path.read_text())
    space["_path"] = str(space_path)
    fat_bundle = (
        load_coordinate_fat_bundle(
            fat_bundle_path, space, space_path, batches,
        )
        if fat_bundle_path is not None else None
    )
    for family in FAMILIES:
        validate_plan(
            plan, space, model_path, family, batches, args.streams,
            config_path, require_scan_bypass=False,
            device_properties=device_properties,
        )
    validate_coordinate_coverage(plan, batches)

    environment = collect_environment(
        repo, config_path, model_path,
        {"cutlass": pathlib.Path(args.cutlass_root).resolve()},
        fa4_python, device_properties,
    )
    regime = {
        "plan": str(plan_path), "plan_sha256": sha256_file(plan_path),
        "space": str(space_path), "space_sha256": sha256_file(space_path),
        "config": str(config_path), "config_sha256": sha256_file(config_path),
        "model": str(model_path), "model_sha256": sha256_file(model_path),
        "device": args.device, "streams": args.streams,
        "iterations": args.iterations, "warmup": args.warmup,
        "repeats": args.repeats, "batches": batches, "runner": runner,
        "generator_python": generator_python, "fa4_python": fa4_python,
        "cutlass_root": str(pathlib.Path(args.cutlass_root).resolve()),
        "build_dir": str(build_dir) if build_dir is not None else None,
        "active_dir": str(active_dir) if active_dir is not None else None,
        "fat_bundle": (
            {
                "path": fat_bundle["_path"],
                "sha256": fat_bundle["_sha256"],
                "binary": fat_bundle["_binary"],
                "binary_sha256": fat_bundle["binary_sha256"],
            }
            if fat_bundle is not None else None
        ),
        "cuda_device_properties": device_properties,
        "implementation_identity": implementation_identity(repo),
        "reproducibility_identity": reproducibility_identity(environment),
    }
    if output_path.is_file():
        payload = json.loads(output_path.read_text())
        if payload.get("regime") != regime:
            raise ValueError("existing joint result has a different regime")
    else:
        payload = {
            "schema": 1,
            "kind": "sm120-joint-plan-whole-graph",
            "started_utc": utc_now(),
            "plan_id": plan["plan_id"],
            "regime": regime,
            "rows": [],
            "environment": environment,
            "selection": {
                "metric": "long-stable natural whole-graph benchmarknn combinedNNEvalsPerSec",
                "policy": "one validated plan candidate per family and exact batch",
            },
        }
    completed = {int(row["batch"]) for row in payload.get("rows", []) if row.get("status") == "measured"}
    if active_dir is not None:
        active_dir.mkdir(parents=True, exist_ok=True)
    if build_dir is not None:
        build_dir.mkdir(parents=True, exist_ok=True)

    for batch in batches:
        if batch in completed:
            print(f"B{batch}: already measured", flush=True)
            continue
        row = {
            "batch": batch,
            "started_utc": utc_now(),
            "selected": {
                family: {
                    "candidate_id": plan_entry(plan, family, batch)["candidate_id"],
                    "implementation": candidate(plan, family, batch).get("implementation", "tilelang"),
                }
                for family in FAMILIES
            },
        }
        try:
            if fat_bundle is not None:
                artifacts = {}
                for family in FAT_FAMILIES:
                    value = candidate(plan, family, batch)
                    if value.get("implementation", "tilelang") == "fallback":
                        artifacts[family] = {
                            "candidate": value,
                            "implementation": "fallback",
                            "linked_artifact": None,
                        }
                    else:
                        key = (family, batch, value["id"])
                        artifact = fat_bundle["_entries"].get(key)
                        if artifact is None:
                            raise ValueError(f"fat bundle has no runtime tactic for {key}")
                        artifacts[family] = artifact
                row["artifacts"] = artifacts
                row["commands"] = {
                    "configure": None,
                    "build": None,
                    "fat_bundle": fat_bundle["_path"],
                }
                binary = pathlib.Path(fat_bundle["_binary"])
                binary_sha256 = fat_bundle["binary_sha256"]
            else:
                assert active_dir is not None and build_dir is not None
                ffn = prepare_ffn(
                    repo, plan, batch, active_dir,
                    pathlib.Path(args.historical_root).resolve(), generator_python,
                    runner, logs, space_path, args.device,
                )
                qkv = prepare_qkv(
                    repo, plan, batch, active_dir,
                    pathlib.Path(args.qkv_generated_root).resolve(), generator_python,
                    runner, logs, space_path, args.device,
                    pathlib.Path(args.cutlass_root).resolve(),
                )
                linear2 = prepare_linear2(
                    repo, plan, batch, active_dir, generator_python, runner,
                    logs, space_path, args.device,
                )
                fa4 = prepare_fa4(
                    repo, plan, batch, active_dir, fa4_python, runner, logs,
                    args.device,
                )
                row["artifacts"] = {
                    "ffn": ffn, "qkv": qkv, "linear2": linear2, "fa4": fa4,
                }
                cmake = configure_command(repo, build_dir, active_dir, fa4, qkv)
                build = ["cmake", "--build", str(build_dir), f"-j{args.jobs}"]
                row["commands"] = {"configure": cmake, "build": build}
                run_command(cmake, logs / f"b{batch}-configure")
                run_command(build, logs / f"b{batch}-build")
                binary = build_dir / "katago"
                if not binary.is_file():
                    raise RuntimeError(f"build did not produce {binary}")
                binary_sha256 = sha256_file(binary)
            override = override_for(plan, batch, args.device, args.streams)
            samples = []
            records = []
            for repeat in range(args.repeats):
                command = runner + [
                    str(binary), "benchmarknn", "-config", str(config_path),
                    "-override-config", override, "-model", str(model_path),
                    "-iterations", str(args.iterations), "-warmup", str(args.warmup),
                    "-batch-size", str(batch), "-boardsize", "19", "-json",
                ]
                result = last_json_object(
                    run_command(command, logs / f"b{batch}-benchmark-r{repeat}").stdout
                )
                samples.append(benchmark_throughput(result))
                records.append(result)
            summary = summarize_throughput(
                samples, iterations=args.iterations, warmup=args.warmup,
            )
            row.update({
                "status": "measured", "finished_utc": utc_now(),
                "override_config": override,
                "binary": str(binary), "binary_sha256": binary_sha256,
                "nn_evals_per_sec_samples": samples,
                "benchmark_records": records,
                **summary,
                "aggregate_batch_groups_per_sec": summary["nn_evals_per_sec_median"] / batch,
            })
            row["commands"]["benchmark_template"] = command
            print(
                f"B{batch}: {summary['nn_evals_per_sec_median']:.3f} nnEval/s "
                f"({summary['measurement_kind']})", flush=True,
            )
        except Exception as error:
            row.update({"status": "failed", "finished_utc": utc_now(), "error": str(error)})
            payload["rows"] = [item for item in payload.get("rows", []) if item.get("batch") != batch]
            payload["rows"].append(row)
            write_json(output_path, payload)
            raise
        payload["rows"] = [item for item in payload.get("rows", []) if item.get("batch") != batch]
        payload["rows"].append(row)
        payload["updated_utc"] = utc_now()
        write_json(output_path, payload)

    payload["finished_utc"] = utc_now()
    payload["complete"] = all(
        any(int(row.get("batch", -1)) == batch and row.get("status") == "measured" for row in payload["rows"])
        for batch in batches
    )
    write_json(output_path, payload)
    print(json.dumps({"output": str(output_path), "complete": payload["complete"], "batches": batches}))


if __name__ == "__main__":
    main()
