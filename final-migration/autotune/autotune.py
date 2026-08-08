#!/usr/bin/env python3
"""One entry point for the frozen SM89 and SM120 B4-B32 workflows."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import shlex
import subprocess
import sys
from typing import Any


def run(command: list[str], *, cwd: pathlib.Path, env: dict[str, str]) -> None:
    print("[autotune] +", shlex.join(command), flush=True)
    subprocess.run(command, cwd=cwd, env=env, check=True)


def load_json(path: pathlib.Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def common_cmake(prefix: pathlib.Path) -> list[str]:
    return [
        f"-DCMAKE_CUDA_COMPILER={prefix / 'cuda/bin/nvcc'}",
        f"-DCUDNN_INCLUDE_DIR={prefix / 'cudnn/include'}",
        f"-DCUDNN_LIBRARY={prefix / 'cudnn/lib/libcudnn.so'}",
        f"-DZLIB_INCLUDE_DIR={prefix / 'native/include'}",
        f"-DZLIB_LIBRARY={prefix / 'native/lib/libz.so'}",
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


def sm89_prepare(args: argparse.Namespace, paths: dict[str, pathlib.Path], env: dict[str, str]) -> None:
    repo, out, python = paths["repo"], paths["out"], paths["python"]
    space = out / "space.json"
    generation = out / "generation-plan.json"
    dual = out / "fat" / "dual-ffn"
    linear = out / "fat" / "linear2"
    build = out / "build"
    binary = build / "katago"
    bundle = out / "artifact-bundle.json"

    if not space.exists() or args.force:
        run([str(python), "python/portable_tactic_workflow.py", "space",
             "--architecture", "sm89", "--gpu-class", paths["gpu_class"].name,
             "--device", str(args.device), "--batches", args.batches,
             "--streams", str(args.streams), "--output", str(space)], cwd=repo, env=env)
    if not generation.exists() or args.force:
        run([str(python), "python/portable_tactic_workflow.py", "generation-plan",
             "--space", str(space), "--phase", "full", "--output", str(generation)], cwd=repo, env=env)

    for family, target in (("dual_ffn", dual), ("linear2", linear)):
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
    tilelang_root = tilelang_root_from_manifests(dual_manifest, linear_manifest)
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
        run([str(python), "python/portable_tactic_workflow.py", "artifact-bundle",
             "--space", str(space), "--binary", str(binary), "--manifests",
             str(dual / "manifest.json"), str(linear / "manifest.json"),
             "--output", str(bundle)], cwd=repo, env=env)
    ensure_file(binary, "SM89 fat binary")
    ensure_file(bundle, "SM89 artifact bundle")


def sm89_discovery(args: argparse.Namespace, paths: dict[str, pathlib.Path], env: dict[str, str]) -> None:
    repo, out, python = paths["repo"], paths["out"], paths["python"]
    result = out / "discovery.json"
    run([str(python), "python/portable_tactic_workflow.py", "scan",
         "--space", str(out / "space.json"), "--binary", str(out / "build/katago"),
         "--config", str(repo / "docs/baseline-configs/bench-cuda-gpu0-4090-s2.cfg"),
         "--model", str(paths["model"]), "--model-identity", str(paths["model"]),
         "--artifact-bundle", str(out / "artifact-bundle.json"),
         "--device", str(args.device), "--streams", str(args.streams),
         "--batches", args.batches, "--phase", "discovery",
         "--iterations", str(args.discovery_iterations), "--warmup", str(args.warmup),
         "--repeats", "1", "--min-improvement-fraction", "0.001", "--resume",
         "--output", str(result), "--raw-dir", str(out / "raw-discovery")], cwd=repo, env=env)


def sm89_gate(args: argparse.Namespace, paths: dict[str, pathlib.Path], env: dict[str, str]) -> None:
    repo, out, python = paths["repo"], paths["out"], paths["python"]
    gate = out / "long-gate.json"
    plan = out / "tactic-plan.json"
    run([str(python), "python/portable_tactic_workflow.py", "gate",
         "--space", str(out / "space.json"), "--discovery", str(out / "discovery.json"),
         "--binary", str(out / "build/katago"),
         "--config", str(repo / "docs/baseline-configs/bench-cuda-gpu0-4090-s2.cfg"),
         "--model", str(paths["model"]), "--model-identity", str(paths["model"]),
         "--artifact-bundle", str(out / "artifact-bundle.json"), "--device", str(args.device),
         "--batches", args.batches, "--iterations", str(args.gate_iterations),
         "--warmup", str(args.warmup), "--repeats", str(args.gate_repeats),
         "--output", str(gate), "--raw-dir", str(out / "raw-long")], cwd=repo, env=env)
    run([str(python), "python/portable_tactic_workflow.py", "plan",
         "--space", str(out / "space.json"), "--results", str(out / "discovery.json"),
         str(gate), "--batches", args.batches, "--output", str(plan)], cwd=repo, env=env)


def sm120_prepare(args: argparse.Namespace, paths: dict[str, pathlib.Path], env: dict[str, str]) -> None:
    repo, out, python = paths["repo"], paths["out"], paths["python"]
    space = out / "space.json"
    if not space.exists() or args.force:
        run([str(python), "python/sm120_tactic_search.py", "space",
             "--gpu-class", paths["gpu_class"].name, "--device", str(args.device),
             "--batches", args.batches, "--streams", str(args.streams),
             "--output", str(space)], cwd=repo, env=env)
    manifests: dict[str, pathlib.Path] = {}
    for family in ("ffn", "qkv", "linear2"):
        target = out / "fat" / family
        manifest = target / "manifest.json"
        manifests[family] = manifest
        command = [str(python), "python/sm120_prepare_tilelang_fat_scan.py",
                   "--space", str(space), "--family", family,
                   "--batches", args.batches, "--device", str(args.device),
                   "--output-dir", str(target), "--python", str(python)]
        if not args.force:
            command.append("--reuse-existing")
        run(command, cwd=repo, env=env)
    coordinate = out / "coordinate-fat"
    manifest = coordinate / "manifest.json"
    if not complete_manifest_for_batches(manifest, args.batches) or args.force:
        command = [str(python), "python/sm120_prepare_coordinate_fat.py",
                   "--repo", str(repo), "--space", str(space), "--batches", args.batches,
                   "--device", str(args.device), "--output-dir", str(coordinate),
                   "--build-dir", str(out / "build"), "--jobs", str(args.jobs),
                   "--generator-python", str(python), "--fa4-python", str(python),
                   "--cutlass-root", str(paths["prefix"] / "sources/cutlass"),
                   "--tilelang-ffn-manifest", str(manifests["ffn"]),
                   "--tilelang-qkv-manifest", str(manifests["qkv"]),
                   "--tilelang-linear2-manifest", str(manifests["linear2"])]
        for cmake_arg in common_cmake(paths["prefix"]):
            command.append(f"--cmake-arg={cmake_arg}")
        run(command, cwd=repo, env=env)
    ensure_file(manifest, "SM120 fat bundle")


def sm120_discovery(args: argparse.Namespace, paths: dict[str, pathlib.Path], env: dict[str, str]) -> None:
    repo, out, python = paths["repo"], paths["out"], paths["python"]
    coordinate = out / "coordinate.json"
    short_plan = out / "selected-plan-short.json"
    if coordinate.exists() and short_plan.exists() and not args.force:
        print("[autotune] reusing completed SM120 coordinate output", flush=True)
        return
    run([str(python), "python/sm120_coordinate_search.py",
         "--space", str(out / "space.json"), "--fat-bundle", str(out / "coordinate-fat/manifest.json"),
         "--output", str(coordinate), "--plan-output", str(short_plan),
         "--config", str(repo / "docs/baseline-configs/bench-cuda-gpu2-5090d-s2.cfg"),
         "--model", str(paths["model"]), "--device", str(args.device),
         "--batches", args.batches, "--streams", str(args.streams), "--passes", "1",
         "--iterations", str(args.discovery_iterations), "--warmup", str(args.warmup),
         "--repeats", "1", "--min-improvement-fraction", "0.001"], cwd=repo, env=env)


def sm120_gate(args: argparse.Namespace, paths: dict[str, pathlib.Path], env: dict[str, str]) -> None:
    repo, out, python = paths["repo"], paths["out"], paths["python"]
    joint = out / "joint-long.json"
    final_plan = out / "tactic-plan.json"
    run([str(python), "python/sm120_measure_joint_plan.py",
         "--plan", str(out / "selected-plan-short.json"), "--space", str(out / "space.json"),
         "--fat-bundle", str(out / "coordinate-fat/manifest.json"), "--output", str(joint),
         "--config", str(repo / "docs/baseline-configs/bench-cuda-gpu2-5090d-s2.cfg"),
         "--model", str(paths["model"]), "--device", str(args.device),
         "--batches", args.batches, "--streams", str(args.streams),
         "--iterations", str(args.gate_iterations), "--warmup", str(args.warmup),
         "--repeats", str(args.gate_repeats)], cwd=repo, env=env)
    run([str(python), "python/sm120_tactic_plan.py", "finalize",
         "--plan", str(out / "selected-plan-short.json"), "--joint-result", str(joint),
         "--space", str(out / "space.json"), "--model", str(paths["model"]),
         "--config", str(repo / "docs/baseline-configs/bench-cuda-gpu2-5090d-s2.cfg"),
         "--batches", args.batches, "--streams", str(args.streams),
         "--output", str(final_plan)], cwd=repo, env=env)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", type=pathlib.Path)
    parser.add_argument("--output-dir", type=pathlib.Path)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--batches", default="4-32")
    parser.add_argument("--streams", type=int, default=2)
    parser.add_argument("--jobs", type=int, default=min(os.cpu_count() or 1, 8))
    parser.add_argument("--phase", choices=("detect", "prepare", "discovery", "gate", "all"), default="all")
    parser.add_argument("--discovery-iterations", type=int, default=100)
    parser.add_argument("--gate-iterations", type=int, default=1000)
    parser.add_argument("--gate-repeats", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    script_dir = pathlib.Path(__file__).resolve().parent
    if args.prefix is None:
        pointer = script_dir / "runtime-prefix.txt"
        args.prefix = pathlib.Path(pointer.read_text().strip()) if pointer.exists() else script_dir / "runtime"
    prefix = args.prefix.resolve()
    repo = prefix / "repo"
    python = prefix / "venv/bin/python"
    model = prefix / "assets/b11c768h12nbt3tflrs-fson-silu.bin.gz"
    for path, label in ((python, "configured Python"), (model, "model")):
        ensure_file(path, label)
    hardware = detect(repo, args.device)
    out = (args.output_dir or prefix / "results" / f"{hardware['workflow']}-{args.batches}-s{args.streams}-gpu{args.device}").resolve()
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
        "XDG_CACHE_HOME": str(prefix / "cache"), "TRITON_HOME": str(prefix / "cache/triton"),
        "TRITON_CACHE_DIR": str(prefix / "cache/triton-runtime"),
        "CMAKE_BUILD_PARALLEL_LEVEL": str(args.jobs), "MAX_JOBS": str(args.jobs),
    })
    paths = {"prefix": prefix, "repo": repo, "python": python, "model": model,
             "out": out, "gpu_class": pathlib.Path(hardware["gpu_class"])}
    prepare = sm89_prepare if hardware["workflow"] == "sm89" else sm120_prepare
    discovery = sm89_discovery if hardware["workflow"] == "sm89" else sm120_discovery
    gate = sm89_gate if hardware["workflow"] == "sm89" else sm120_gate
    if args.phase in ("prepare", "all"):
        prepare(args, paths, env)
    if args.phase in ("discovery", "all"):
        discovery(args, paths, env)
    if args.phase in ("gate", "all"):
        gate(args, paths, env)
    if (out / "tactic-plan.json").exists():
        print(f"[autotune] final plan: {out / 'tactic-plan.json'} sha256={sha256(out / 'tactic-plan.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
