#!/usr/bin/env python3
"""One entry point for SM89/SM120 optimized-baseline and exact-batch tuning."""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import math
import os
import pathlib
import shlex
import subprocess
import sys
from typing import Any

try:
    from build_parallelism import conservative_build_jobs
except ModuleNotFoundError:  # running from the source tree instead of a release tar
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "python"))
    from build_parallelism import conservative_build_jobs


def run(command: list[str], *, cwd: pathlib.Path, env: dict[str, str]) -> None:
    print("[autotune] +", shlex.join(command), flush=True)
    subprocess.run(command, cwd=cwd, env=env, check=True)


def utc_now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def load_json(path: pathlib.Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def config_value(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def config_string(values: dict[str, object]) -> str:
    return ",".join(
        f"{key}={config_value(values[key])}" for key in sorted(values)
    )


def common_cmake(prefix: pathlib.Path) -> list[str]:
    return [
        f"-DCMAKE_CUDA_COMPILER={prefix / 'cuda/bin/nvcc'}",
        f"-DCUDNN_INCLUDE_DIR={prefix / 'cudnn/include'}",
        f"-DCUDNN_LIBRARY={prefix / 'cudnn/lib/libcudnn.so'}",
        f"-DZLIB_INCLUDE_DIR={prefix / 'native/include'}",
        f"-DZLIB_LIBRARY={prefix / 'native/lib/libz.so'}",
        f"-DKATAGO_TILELANG_ROOT={prefix / 'sources/TileLang'}",
        f"-DKATAGO_CUTLASS_ROOT={prefix / 'sources/cutlass'}",
        "-DNO_GIT_REVISION=1",
    ]


def detect(repo: pathlib.Path, device: int) -> dict[str, Any]:
    sys.path.insert(0, str(repo / "python"))
    from portable_cuda_device import query_cuda_device

    result = query_cuda_device(device)
    cc = tuple(result["compute_capability"])
    if cc == (8, 9):
        workflow = "sm89"
        gpu_class = "rtx4090"
    elif cc == (12, 0):
        workflow = "sm120"
        gpu_class = "rtx5080" if "5080" in result["name"].lower() else "rtx5090d"
    else:
        raise RuntimeError(f"unsupported compute capability {cc}; expected SM89 or SM120")
    return {"schema": 1, "workflow": workflow, "gpu_class": gpu_class, "device": result}


def ensure_file(path: pathlib.Path, label: str) -> None:
    if not path.is_file():
        raise RuntimeError(f"{label} is missing: {path}")


def parse_batch_set(value: str) -> list[int]:
    batches: set[int] = set()
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        if "-" in item:
            first, last = (int(part) for part in item.split("-", 1))
            if last < first:
                raise ValueError(f"invalid descending batch range: {item}")
            batches.update(range(first, last + 1))
        else:
            batches.add(int(item))
    result = sorted(batches)
    if not result or result[0] < 1:
        raise ValueError("batch set must contain positive integers")
    return result


def complete_manifest_for_batches(path: pathlib.Path, batches: str) -> bool:
    """Reject an interrupted or differently scoped fat-bundle checkpoint."""
    if not path.is_file():
        return False
    try:
        payload = load_json(path)
    except (OSError, ValueError, TypeError):
        return False
    return (
        payload.get("complete") is True
        and sorted(payload.get("batches", [])) == parse_batch_set(batches)
    )


def is_build_only_space(path: pathlib.Path) -> bool:
    if not path.is_file():
        return False
    try:
        policy = load_json(path).get("candidate_policy", {})
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return False
    return (
        isinstance(policy, dict) and
        policy.get("build_only_plan_restricted") is True
    )


def has_candidates(
    space_path: pathlib.Path, family: str, implementation: str,
) -> bool:
    space = load_json(space_path)
    search_family = (
        "qkv_rope"
        if space.get("architecture") == "sm120" and family == "wide_qkv"
        else family
    )
    for batch in space.get("batches", []):
        for candidate in batch.get(search_family, []):
            candidate_implementation = candidate.get("implementation")
            if candidate_implementation is None and space.get("architecture") == "sm120":
                candidate_implementation = "tilelang"
            if (
                candidate_implementation == implementation and
                candidate.get("artifact_family", search_family) == family
            ):
                return True
    return False


def write_empty_manifest(
    *, space_path: pathlib.Path, family: str, target: pathlib.Path,
    registry_source: pathlib.Path | None = None,
) -> None:
    target.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "schema": 1,
        "complete": True,
        "requested_entry_count": 0,
        "family": family,
        "space": str(space_path.resolve()),
        "space_sha256": sha256(space_path),
        "sources": [],
        "entries": [],
    }
    if registry_source is not None:
        sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "python"))
        from portable_fat_scan import render_registry

        registry_source.parent.mkdir(parents=True, exist_ok=True)
        registry_source.write_text(render_registry(family, []))
        payload.update({
            "registry_source": str(registry_source.resolve()),
            "registry_sha256": sha256(registry_source),
        })
    (target / "manifest.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )


def tilelang_root_from_manifests(*manifests: dict[str, Any]) -> pathlib.Path:
    roots: set[pathlib.Path] = set()
    for manifest in manifests:
        if manifest.get("complete") is not True:
            raise RuntimeError("cannot configure from an incomplete TileLang manifest")
        for entry in manifest.get("entries", []):
            metadata_path = pathlib.Path(entry["metadata"])
            metadata = load_json(metadata_path)
            root = metadata.get("generation_environment", {}).get("tilelang_root")
            if not root:
                raise RuntimeError(f"TileLang root is missing from {metadata_path}")
            roots.add(pathlib.Path(root).resolve())
    if len(roots) != 1:
        raise RuntimeError(f"fat manifests disagree on TileLang root: {sorted(map(str, roots))}")
    root = roots.pop()
    for relative in ("src/tl_templates/cuda/debug.h", "3rdparty/cutlass/include/cutlass/cutlass.h"):
        ensure_file(root / relative, "TileLang build input")
    return root


def baseline_runtime(paths: dict[str, pathlib.Path]) -> tuple[pathlib.Path, pathlib.Path]:
    out, repo = paths["out"], paths["repo"]
    config = (
        repo / "docs/baseline-configs/bench-cuda-gpu0-4090-s2.cfg"
        if paths["workflow"].name == "sm89"
        else repo / "docs/baseline-configs/bench-cuda-gpu2-5090d-s2.cfg"
    )
    return out / "baseline-prescan-build/katago", config


def prepare_baseline_prescan_binary(
    args: argparse.Namespace,
    paths: dict[str, pathlib.Path],
    env: dict[str, str],
) -> None:
    repo, prefix = paths["repo"], paths["prefix"]
    binary, _ = baseline_runtime(paths)
    build = binary.parent
    architecture = paths["workflow"].name
    configure = [
        "cmake", "-S", str(repo / "cpp"), "-B", str(build), "-G", "Ninja",
        "-DUSE_BACKEND=CUDA", "-DCMAKE_BUILD_TYPE=Release",
        f"-DKATAGO_CUDA_ARCHITECTURES={'89' if architecture == 'sm89' else '120'}",
        *common_cmake(prefix),
    ]
    if architecture == "sm89":
        configure.append(
            f"-DSM89_FLASH_ATTN_ROOT={prefix / 'sources/flash-attention'}"
        )
    if not binary.is_file() or args.force:
        run(configure, cwd=repo, env=env)
        run(
            ["cmake", "--build", str(build), "--parallel", str(args.jobs)],
            cwd=repo, env=env,
        )
    ensure_file(binary, "optimized baseline prescan binary")


def workflow_baseline_prescan(
    args: argparse.Namespace,
    paths: dict[str, pathlib.Path],
    env: dict[str, str],
    requested_batches: str,
) -> list[int]:
    prepare_baseline_prescan_binary(args, paths, env)
    repo, out, python = paths["repo"], paths["out"], paths["python"]
    binary, config = baseline_runtime(paths)
    result = out / "optimized-baseline-batch-prescan.json"
    requested = parse_batch_set(requested_batches)
    top_count = min(args.top_batches, len(requested))
    reusable = False
    if result.is_file() and not args.force:
        payload = load_json(result)
        identity = payload.get("identity", {})
        reusable = (
            payload.get("kind") == "cuda-stable-optimized-batch-prescan" and
            payload.get("architecture") == paths["workflow"].name and
            payload.get("gpu_class") == paths["gpu_class"].name and
            payload.get("requested_batches") == requested and
            payload.get("top_batch_count") == top_count and
            payload.get("device_ordinal") == args.device and
            payload.get("streams") == args.streams and
            payload.get("measurement_request") == {
                "iterations": args.baseline_iterations,
                "warmup": args.warmup,
                "repeats": args.baseline_repeats,
            } and
            isinstance(identity, dict) and
            identity.get("binary_sha256") == sha256(binary) and
            identity.get("config_sha256") == sha256(config) and
            identity.get("model_sha256") == sha256(paths["model"])
        )
    if not reusable:
        run([
            str(python), "python/cuda_tactic_workflow.py", "baseline-prescan",
            "--architecture", paths["workflow"].name,
            "--gpu-class", paths["gpu_class"].name,
            "--device", str(args.device), "--streams", str(args.streams),
            "--batches", requested_batches, "--top-batches", str(top_count),
            "--binary", str(binary), "--config", str(config),
            "--model", str(paths["model"]),
            "--iterations", str(args.baseline_iterations),
            "--warmup", str(args.warmup),
            "--repeats", str(args.baseline_repeats),
            "--raw-dir", str(out / "raw-optimized-baseline-prescan"),
            "--output", str(result),
        ], cwd=repo, env=env)
    payload = load_json(result)
    selected = payload.get("selected_batches")
    if (
        not isinstance(selected, list) or len(selected) != top_count or
        any(not isinstance(batch, int) or batch not in requested for batch in selected)
    ):
        raise RuntimeError("optimized baseline prescan has an invalid top-batch selection")
    print(
        "[autotune] full tactic search batches: " +
        ",".join(f"B{batch}" for batch in selected),
        flush=True,
    )
    return sorted(selected)


def sm89_prepare(args: argparse.Namespace, paths: dict[str, pathlib.Path], env: dict[str, str]) -> None:
    repo, out, python = paths["repo"], paths["out"], paths["python"]
    space = out / "space.json"
    generation = out / "generation-plan.json"
    dual = out / "fat" / "dual-ffn"
    linear = out / "fat" / "linear2"
    build = out / "build"
    binary = build / "katago"
    bundle = out / "artifact-bundle.json"

    if not space.exists() or (args.force and not is_build_only_space(space)):
        run([str(python), "python/cuda_tactic_workflow.py", "space",
             "--architecture", "sm89", "--gpu-class", paths["gpu_class"].name,
             "--device", str(args.device), "--batches", args.batches,
             "--streams", str(args.streams), "--output", str(space)], cwd=repo, env=env)
    if not generation.exists() or args.force:
        run([str(python), "python/cuda_tactic_workflow.py", "generation-plan",
             "--space", str(space), "--phase", "full", "--output", str(generation)], cwd=repo, env=env)

    for family, target in (("dual_ffn", dual), ("linear2", linear)):
        if not has_candidates(space, family, "tilelang_gemm"):
            write_empty_manifest(
                space_path=space, family=family, target=target,
                registry_source=target / f"sm89_search_{family}_fat_registry.cu",
            )
            continue
        command = [str(python), "python/portable_prepare_tilelang_fat_scan.py",
                   "--space", str(space), "--family", family,
                   "--batches", args.batches, "--device", str(args.device),
                   "--output-dir", str(target), "--python", str(python),
                   "--nvcc", str(paths["prefix"] / "cuda/bin/nvcc"),
                   "--compile-objects"]
        if not args.force:
            command.append("--reuse-existing")
        # The generator writes an intentionally incomplete manifest after every
        # candidate. Always enter it so a killed run resumes and closes the
        # exact requested domain instead of mistaking a checkpoint for success.
        run(command, cwd=repo, env=env)

    dual_manifest = load_json(dual / "manifest.json")
    linear_manifest = load_json(linear / "manifest.json")
    if dual_manifest.get("entries") or linear_manifest.get("entries"):
        tilelang_root = tilelang_root_from_manifests(
            dual_manifest, linear_manifest,
        )
    else:
        tilelang_root = paths["prefix"] / "sources/TileLang"
    configure = [
        "cmake", "-S", str(repo / "cpp"), "-B", str(build), "-G", "Ninja",
        "-DUSE_BACKEND=CUDA", "-DCMAKE_BUILD_TYPE=Release",
        "-DKATAGO_CUDA_ARCHITECTURES=89",
        f"-DSM89_FLASH_ATTN_ROOT={paths['prefix'] / 'sources/flash-attention'}",
        f"-DSM89_TACTIC_TILELANG_ROOT={tilelang_root}",
        f"-DSM89_SEARCH_DUAL_FFN_FAT_REGISTRY={dual_manifest['registry_source']}",
        f"-DSM89_SEARCH_DUAL_FFN_FAT_SOURCES={';'.join(dual_manifest['sources'])}",
        f"-DSM89_SEARCH_LINEAR2_FAT_REGISTRY={linear_manifest['registry_source']}",
        f"-DSM89_SEARCH_LINEAR2_FAT_SOURCES={';'.join(linear_manifest['sources'])}",
        *common_cmake(paths["prefix"]),
    ]
    if not binary.exists() or args.force:
        run(configure, cwd=repo, env=env)
        run(["cmake", "--build", str(build), "--parallel", str(args.jobs)], cwd=repo, env=env)
    if not bundle.exists() or args.force:
        run([str(python), "python/cuda_tactic_workflow.py", "artifact-bundle",
             "--space", str(space), "--binary", str(binary), "--manifests",
             str(dual / "manifest.json"), str(linear / "manifest.json"),
             "--output", str(bundle)], cwd=repo, env=env)
    ensure_file(binary, "SM89 fat binary")
    ensure_file(bundle, "SM89 artifact bundle")


def sm120_prepare(args: argparse.Namespace, paths: dict[str, pathlib.Path], env: dict[str, str]) -> None:
    repo, out, python = paths["repo"], paths["out"], paths["python"]
    space = out / "space.json"
    if not space.exists() or (args.force and not is_build_only_space(space)):
        run([str(python), "python/cuda_tactic_workflow.py", "space",
             "--architecture", "sm120", "--gpu-class", paths["gpu_class"].name,
             "--device", str(args.device), "--batches", args.batches,
             "--streams", str(args.streams), "--output", str(space)], cwd=repo, env=env)
    manifests: dict[str, pathlib.Path] = {}
    for family in ("dual_ffn", "wide_qkv", "linear2", "outproj"):
        target = out / "fat" / family
        manifest = target / "manifest.json"
        manifests[family] = manifest
        if has_candidates(space, family, "tilelang"):
            command = [str(python), "python/sm120_prepare_tilelang_fat_scan.py",
                       "--space", str(space), "--family", family,
                       "--batches", args.batches, "--device", str(args.device),
                       "--output-dir", str(target), "--python", str(python)]
            if not args.force:
                command.append("--reuse-existing")
            run(command, cwd=repo, env=env)
        else:
            write_empty_manifest(
                space_path=space, family=family, target=target,
            )
    coordinate = out / "coordinate-fat"
    manifest = coordinate / "manifest.json"
    if not complete_manifest_for_batches(manifest, args.batches) or args.force:
        command = [str(python), "python/sm120_prepare_coordinate_fat.py",
                   "--repo", str(repo), "--space", str(space), "--batches", args.batches,
                   "--device", str(args.device), "--output-dir", str(coordinate),
                   "--build-dir", str(out / "build"), "--jobs", str(args.jobs),
                   "--generator-python", str(python), "--fa4-python", str(python),
                   "--cutlass-root", str(paths["prefix"] / "sources/cutlass"),
                   "--tilelang-dual-ffn-manifest", str(manifests["dual_ffn"]),
                   "--tilelang-wide-qkv-manifest", str(manifests["wide_qkv"]),
                   "--tilelang-linear2-manifest", str(manifests["linear2"]),
                   "--tilelang-outproj-manifest", str(manifests["outproj"])]
        for cmake_arg in common_cmake(paths["prefix"]):
            command.append(f"--cmake-arg={cmake_arg}")
        run(command, cwd=repo, env=env)
    ensure_file(manifest, "SM120 fat bundle")
    bundle = out / "artifact-bundle.json"
    coordinate_payload = load_json(manifest)
    binary = pathlib.Path(coordinate_payload["binary"])
    if not bundle.exists() or args.force:
        run([str(python), "python/cuda_tactic_workflow.py", "artifact-bundle",
             "--space", str(space), "--binary", str(binary),
             "--manifests", str(manifest), "--output", str(bundle)],
            cwd=repo, env=env)
    ensure_file(bundle, "SM120 artifact bundle")


def workflow_runtime(paths: dict[str, pathlib.Path]) -> tuple[pathlib.Path, pathlib.Path, list[str]]:
    repo, out = paths["repo"], paths["out"]
    if paths["workflow"].name == "sm89":
        return (
            out / "build/katago",
            repo / "docs/baseline-configs/bench-cuda-gpu0-4090-s2.cfg",
            ["--artifact-bundle", str(out / "artifact-bundle.json")],
        )
    manifest = load_json(out / "coordinate-fat/manifest.json")
    return (
        pathlib.Path(manifest["binary"]),
        repo / "docs/baseline-configs/bench-cuda-gpu2-5090d-s2.cfg",
        ["--artifact-bundle", str(out / "artifact-bundle.json")],
    )


def workflow_discovery(
    args: argparse.Namespace, paths: dict[str, pathlib.Path], env: dict[str, str],
) -> None:
    repo, out, python = paths["repo"], paths["out"], paths["python"]
    binary, config, artifact_args = workflow_runtime(paths)
    first_pass = out / "discovery-first-pass.json"
    run([
        str(python), "python/cuda_tactic_workflow.py", "scan",
        "--space", str(out / "space.json"), "--binary", str(binary),
        "--config", str(config), "--model", str(paths["model"]),
        "--model-identity", str(paths["model"]), *artifact_args,
        "--device", str(args.device), "--streams", str(args.streams),
        "--batches", args.batches, "--phase", "discovery",
        "--iterations", str(args.discovery_iterations),
        "--warmup", str(args.warmup), "--repeats", "1",
        "--min-improvement-fraction", "0.001", "--resume",
        "--output", str(first_pass),
        "--raw-dir", str(out / "raw-discovery-first-pass"),
    ], cwd=repo, env=env)
    run([
        str(python), "python/cuda_tactic_workflow.py", "refine",
        "--space", str(out / "space.json"),
        "--discovery", str(first_pass),
        "--binary", str(binary), "--config", str(config),
        "--model", str(paths["model"]),
        "--model-identity", str(paths["model"]), *artifact_args,
        "--device", str(args.device), "--batches", args.batches,
        "--top-k", str(args.refinement_top_k),
        "--max-sweeps", str(args.refinement_max_sweeps),
        "--resweep-top-k", str(args.refinement_resweep_top_k),
        "--confirmation-iterations", str(args.refinement_confirmation_iterations),
        "--iterations", str(args.discovery_iterations),
        "--warmup", str(args.warmup), "--repeats", "1",
        "--min-improvement-fraction", "0.001", "--resume",
        "--output", str(out / "discovery.json"),
        "--raw-dir", str(out / "raw-discovery-refinement"),
    ], cwd=repo, env=env)


def workflow_gate(
    args: argparse.Namespace, paths: dict[str, pathlib.Path], env: dict[str, str],
) -> None:
    repo, out, python = paths["repo"], paths["out"], paths["python"]
    binary, config, artifact_args = workflow_runtime(paths)
    gate = out / "long-gate.json"
    run([
        str(python), "python/cuda_tactic_workflow.py", "gate",
        "--space", str(out / "space.json"),
        "--discovery", str(out / "discovery.json"),
        "--binary", str(binary), "--config", str(config),
        "--model", str(paths["model"]), "--model-identity", str(paths["model"]),
        *artifact_args, "--device", str(args.device), "--batches", args.batches,
        "--iterations", str(args.gate_iterations), "--warmup", str(args.warmup),
        "--repeats", str(args.gate_repeats), "--output", str(gate),
        "--raw-dir", str(out / "raw-long"),
    ], cwd=repo, env=env)
    run([
        str(python), "python/cuda_tactic_workflow.py", "plan",
        "--space", str(out / "space.json"), "--results",
        str(out / "discovery.json"), str(gate), "--batches", args.batches,
        "--output", str(out / "tactic-plan.json"),
    ], cwd=repo, env=env)


def accuracy_corpus(paths: dict[str, pathlib.Path]) -> pathlib.Path:
    state = paths["prefix"] / "state/accuracy-corpus.json"
    ensure_file(state, "accuracy corpus state")
    corpus = pathlib.Path(load_json(state)["corpus"]).resolve()
    ensure_file(corpus, "8192-row accuracy corpus")
    return corpus


def replaynn_command(
    binary: pathlib.Path,
    config: pathlib.Path,
    overrides: dict[str, object],
    model: pathlib.Path,
    corpus: pathlib.Path,
    output: pathlib.Path,
    batch: int,
) -> list[str]:
    """Build replaynn argv using KataGo's single-dash command interface."""
    return [
        str(binary), "replaynn", "-config", str(config),
        "-override-config", config_string(overrides),
        "-model", str(model), "-corpus", str(corpus),
        "-output", str(output), "-batch-size", str(batch),
    ]


def workflow_reference(
    args: argparse.Namespace, paths: dict[str, pathlib.Path], env: dict[str, str],
) -> None:
    """Create the immutable reference through the disabled official FP32 path."""
    repo, prefix = paths["repo"], paths["prefix"]
    binary, config, _ = workflow_runtime(paths)
    corpus = accuracy_corpus(paths)
    model_sha256 = sha256(paths["model"])
    corpus_sha256 = sha256(corpus)
    golden = prefix / "assets/replay-fixed-fp32-full19.krnn"
    metadata = prefix / "assets/replay-fixed-fp32-full19.json"
    if golden.is_file() and not args.force:
        ensure_file(metadata, "FP32 reference metadata")
        recorded = load_json(metadata)
        if (
            recorded.get("reference_sha256") != sha256(golden) or
            recorded.get("model_sha256") != model_sha256 or
            recorded.get("corpus_sha256") != corpus_sha256 or
            recorded.get("batch") != 13
        ):
            raise RuntimeError(
                "FP32 reference metadata differs from the current model/corpus"
            )
        print(
            f"[autotune] reusing FP32 reference {golden} sha256={sha256(golden)}",
            flush=True,
        )
        return
    temporary = golden.with_suffix(golden.suffix + ".partial")
    temporary.unlink(missing_ok=True)
    overrides = {
        "cudaDeviceToUseThread0": args.device,
        "cudaSm89Backend": False,
        "cudaSm120Backend": False,
        "nnMaxBatchSize": 13,
        "numNNServerThreadsPerModel": 1,
        "useFP16": False,
    }
    command = replaynn_command(
        binary, config, overrides, paths["model"], corpus, temporary, 13,
    )
    run(command, cwd=repo, env=env)
    os.replace(temporary, golden)
    metadata.write_text(json.dumps({
        "schema": 1,
        "kind": "official-disabled-backend-full-fp32-reference",
        "binary": str(binary),
        "binary_sha256": sha256(binary),
        "model_sha256": model_sha256,
        "corpus_sha256": corpus_sha256,
        "reference": str(golden),
        "reference_sha256": sha256(golden),
        "batch": 13,
        "device": args.device,
        "overrides": overrides,
        "command": command,
    }, indent=2, sort_keys=True) + "\n")
    print(f"[autotune] FP32 reference sha256={sha256(golden)}", flush=True)


def select_best_long_gate_row(
    gate_payload: dict[str, Any], batches: list[int],
) -> tuple[int, dict[str, object], float]:
    rows = [
        row for row in gate_payload.get("rows", [])
        if isinstance(row, dict) and row.get("history_long_gate") is True
    ]
    by_batch: dict[int, dict[str, object]] = {}
    for row in rows:
        batch = int(row["batch"])
        if batch in by_batch:
            raise RuntimeError(f"long gate contains duplicate B{batch} final rows")
        by_batch[batch] = row
    if sorted(by_batch) != batches:
        raise RuntimeError(
            f"long gate batches differ: {sorted(by_batch)} != {batches}"
        )
    measured: list[tuple[float, int]] = []
    for batch in batches:
        row = by_batch[batch]
        metric = row.get("stable_long_nn_evals_per_sec")
        if (
            row.get("status") != "measured" or
            not isinstance(metric, (int, float)) or
            not math.isfinite(float(metric)) or
            float(metric) <= 0.0
        ):
            raise RuntimeError(f"long gate B{batch} lacks a stable positive throughput")
        measured.append((float(metric), batch))
    # A tie selects the smaller exact batch deterministically.
    best_metric, best_batch = max(measured, key=lambda item: (item[0], -item[1]))
    return best_batch, by_batch[best_batch], best_metric


def workflow_accuracy(
    args: argparse.Namespace, paths: dict[str, pathlib.Path], env: dict[str, str],
) -> None:
    """Replay only the fastest long-gate batch and attach its 8192-row certificate."""
    repo, out, python, prefix = (
        paths["repo"], paths["out"], paths["python"], paths["prefix"],
    )
    binary, config, _ = workflow_runtime(paths)
    corpus = accuracy_corpus(paths)
    golden = prefix / "assets/replay-fixed-fp32-full19.krnn"
    ensure_file(golden, "immutable official full-FP32 reference")
    golden_metadata = prefix / "assets/replay-fixed-fp32-full19.json"
    ensure_file(golden_metadata, "immutable official full-FP32 metadata")
    reference_sha256 = sha256(golden)
    binary_sha256 = sha256(binary)
    corpus_sha256 = sha256(corpus)
    model_sha256 = sha256(paths["model"])
    recorded_golden = load_json(golden_metadata)
    if (
        recorded_golden.get("reference_sha256") != reference_sha256 or
        recorded_golden.get("model_sha256") != model_sha256 or
        recorded_golden.get("corpus_sha256") != corpus_sha256 or
        recorded_golden.get("batch") != 13
    ):
        raise RuntimeError(
            "FP32 reference metadata differs from the current model/corpus"
        )
    gate = out / "long-gate.json"
    ensure_file(gate, "long gate")
    gate_payload = load_json(gate)
    batches = parse_batch_set(args.batches)
    best_batch, best_row, best_metric = select_best_long_gate_row(
        gate_payload, batches
    )
    print(
        f"[autotune] fastest long-gate plan is B{best_batch}: "
        f"{best_metric:.3f} nnEval/s; certifying only this plan",
        flush=True,
    )
    accuracy_dir = out / "accuracy"
    accuracy_dir.mkdir(parents=True, exist_ok=True)
    reports: dict[int, pathlib.Path] = {}
    for batch in (best_batch,):
        report = accuracy_dir / f"replay-b{batch}-vs-fp32.json"
        reports[batch] = report
        row_overrides = best_row.get("overrides")
        if not isinstance(row_overrides, dict):
            raise RuntimeError(f"long gate B{batch} has no replayable overrides")
        overrides = dict(row_overrides)
        if int(overrides.get("nnMaxBatchSize", -1)) != batch:
            raise RuntimeError(
                f"long gate B{batch} is not bound to its exact evaluator capacity"
            )
        if report.is_file() and not args.force:
            existing = load_json(report)
            if (
                existing.get("referenceSha256") == reference_sha256 and
                int(existing.get("numRows", 0)) == 8192 and
                existing.get("exactBatch") == batch and
                existing.get("candidateBinarySha256") == binary_sha256 and
                existing.get("candidateOverrides") == overrides and
                existing.get("corpusSha256") == corpus_sha256 and
                existing.get("modelSha256") == model_sha256 and
                existing.get("candidateMaxBatchSize") == batch and
                existing.get("candidateFixedBatchTailPadding") is True and
                existing.get("referenceFixedBatchTailPadding") is True and
                existing.get("inputAndTargetSectionsByteExact") is True and
                set(existing.get("requestGate", {})) == {
                    "policyProbability", "valueProbability", "scoreRaw",
                    "ownershipProbability",
                }
            ):
                print(f"[autotune] reusing accuracy report {report}", flush=True)
                continue
        candidate = accuracy_dir / f"replay-b{batch}.krnn"
        candidate.unlink(missing_ok=True)
        replay = replaynn_command(
            binary, config, overrides, paths["model"], corpus, candidate, batch,
        )
        run(replay, cwd=repo, env=env)
        run([
            str(python), "python/katago/train/compare_replay_krnn.py",
            "--reference", str(golden), "--candidate", str(candidate),
            "--output", str(report),
            "--expected-candidate-batch", str(batch),
        ], cwd=repo, env=env)
        report_payload = load_json(report)
        report_payload.update({
            "candidateBinarySha256": binary_sha256,
            "candidateOverrides": overrides,
            "corpusSha256": corpus_sha256,
            "modelSha256": model_sha256,
        })
        report.write_text(
            json.dumps(report_payload, indent=2, sort_keys=True) + "\n"
        )
        candidate.unlink()
    certified = out / "long-gate-best-certified.json"
    certify = [
        str(python), "python/cuda_tactic_workflow.py", "certify",
        "--gate", str(gate), "--batches", str(best_batch),
    ]
    for batch, report in reports.items():
        certify.extend(["--comparison", f"{batch}={report}"])
    certify.extend(["--output", str(certified)])
    run(certify, cwd=repo, env=env)
    run([
        str(python), "python/cuda_tactic_workflow.py", "plan",
        "--space", str(out / "space.json"), "--results",
        str(out / "discovery.json"), str(certified),
        "--batches", str(best_batch),
        "--output", str(out / "best-tactic-plan.json"),
    ], cwd=repo, env=env)
    best_plan = out / "best-tactic-plan.json"
    (out / "SHA256SUMS").write_text(
        f"{sha256(best_plan)}  best-tactic-plan.json\n"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", type=pathlib.Path)
    parser.add_argument("--repo", type=pathlib.Path)
    parser.add_argument("--output-dir", type=pathlib.Path)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--batches", default="4-32")
    parser.add_argument("--streams", type=int, default=2)
    parser.add_argument("--jobs", type=int, default=conservative_build_jobs())
    parser.add_argument(
        "--phase",
        choices=(
            "detect", "prescan", "prepare", "discovery", "gate",
            "reference", "accuracy", "all",
        ),
        default="all",
    )
    parser.add_argument(
        "--full-batch-scan", action="store_true",
        help="skip optimized baseline prescan and fully optimize every --batches value",
    )
    parser.add_argument(
        "--top-batches", type=int, default=3,
        help="number of optimized-baseline winners receiving full tactic search",
    )
    parser.add_argument("--baseline-iterations", type=int, default=200)
    parser.add_argument("--baseline-repeats", type=int, default=2)
    parser.add_argument("--discovery-iterations", type=int, default=100)
    parser.add_argument(
        "--refinement-top-k", type=int, default=10,
        help=(
            "after the full first pass, retest this many first-pass leaders "
            "per family on the improved whole graph"
        ),
    )
    parser.add_argument(
        "--refinement-max-sweeps", type=int, default=3,
        help="repeat whole-graph Top-K coordinate refinement until unchanged",
    )
    parser.add_argument(
        "--refinement-resweep-top-k", type=int, default=3,
        help="candidate count per family after the first Top-K refinement sweep",
    )
    parser.add_argument(
        "--refinement-confirmation-iterations", type=int, default=300,
        help="timed iterations in each leg of a provisional-winner ABBA check",
    )
    parser.add_argument("--gate-iterations", type=int, default=1000)
    parser.add_argument("--gate-repeats", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=80)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.top_batches < 1:
        parser.error("--top-batches must be positive")
    if args.refinement_top_k < 1:
        parser.error("--refinement-top-k must be positive")
    if args.refinement_max_sweeps < 1:
        parser.error("--refinement-max-sweeps must be positive")
    if not 1 <= args.refinement_resweep_top_k <= args.refinement_top_k:
        parser.error(
            "--refinement-resweep-top-k must be in [1,--refinement-top-k]"
        )
    if args.refinement_confirmation_iterations < 100:
        parser.error("--refinement-confirmation-iterations must be at least 100")
    if args.full_batch_scan and args.phase == "prescan":
        parser.error("--phase prescan cannot be combined with --full-batch-scan")
    script_dir = pathlib.Path(__file__).resolve().parent
    if args.prefix is None:
        pointer = script_dir / "runtime-prefix.txt"
        args.prefix = pathlib.Path(pointer.read_text().strip()) if pointer.exists() else script_dir / "runtime"
    prefix = args.prefix.resolve()
    repo = (args.repo or prefix / "repo").resolve()
    python = prefix / "venv/bin/python"
    model = prefix / "assets/b11c768h12nbt3tflrs-fson-silu.bin.gz"
    for path, label in ((python, "configured Python"), (model, "model")):
        ensure_file(path, label)
    hardware = detect(repo, args.device)
    requested_batches = args.batches
    mode = "full" if args.full_batch_scan else f"top{args.top_batches}"
    out = (
        args.output_dir or prefix / "results" /
        f"{hardware['workflow']}-{mode}-from-{requested_batches}-s{args.streams}-gpu{args.device}"
    ).resolve()
    out.mkdir(parents=True, exist_ok=True)
    (out / "device.json").write_text(json.dumps(hardware, indent=2, sort_keys=True) + "\n")
    print(json.dumps(hardware, indent=2, sort_keys=True), flush=True)
    if args.phase == "detect":
        return 0
    env = dict(os.environ)
    env.update({
        "AUTOTUNE_PREFIX": str(prefix), "CUDA_HOME": str(prefix / "cuda"),
        "CUDA_PATH": str(prefix / "cuda"), "CUDNN_ROOT": str(prefix / "cudnn"),
        "PATH": f"{prefix / 'venv/bin'}:{prefix / 'cuda/bin'}:{env.get('PATH', '')}",
        "LD_LIBRARY_PATH": f"{prefix / 'cudnn/lib'}:{prefix / 'cuda/lib64'}:{prefix / 'native/lib'}:{env.get('LD_LIBRARY_PATH', '')}",
        "CMAKE_PREFIX_PATH": f"{prefix / 'native'}:{env.get('CMAKE_PREFIX_PATH', '')}",
        "XDG_CACHE_HOME": str(prefix / "cache"),
        "CMAKE_BUILD_PARALLEL_LEVEL": str(args.jobs), "MAX_JOBS": str(args.jobs),
    })
    paths = {"prefix": prefix, "repo": repo, "python": python, "model": model,
             "out": out, "gpu_class": pathlib.Path(hardware["gpu_class"]),
             "workflow": pathlib.Path(hardware["workflow"])}
    if args.full_batch_scan:
        selected_batches = parse_batch_set(requested_batches)
    else:
        selected_batches = workflow_baseline_prescan(
            args, paths, env, requested_batches,
        )
    args.batches = ",".join(str(batch) for batch in selected_batches)
    if args.phase == "prescan":
        return 0
    prepare = sm89_prepare if hardware["workflow"] == "sm89" else sm120_prepare
    if args.phase in ("prepare", "all"):
        prepare(args, paths, env)
    if args.phase in ("discovery", "all"):
        workflow_discovery(args, paths, env)
    if args.phase in ("gate", "all"):
        workflow_gate(args, paths, env)
    if args.phase == "reference":
        workflow_reference(args, paths, env)
    if args.phase in ("accuracy", "all"):
        golden = prefix / "assets/replay-fixed-fp32-full19.krnn"
        if args.phase == "accuracy" or golden.is_file():
            workflow_accuracy(args, paths, env)
        else:
            print(
                "[autotune] immutable FP32 reference is absent; skipping accuracy "
                "and leaving production_ready=false",
                flush=True,
            )
    final_plan = (
        out / "best-tactic-plan.json"
        if (out / "best-tactic-plan.json").is_file()
        else out / "tactic-plan.json"
    )
    if final_plan.exists():
        print(f"[autotune] final plan: {final_plan} sha256={sha256(final_plan)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
