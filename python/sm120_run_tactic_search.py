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
import os
import pathlib
import shlex
import shutil
import statistics
import subprocess
import sys

try:
    from sm120_tactic_plan import (
        candidate_override as plan_candidate_override,
        load_plan,
        validate_plan,
    )
    from sm120_fat_scan import launch_symbol, symbol_token
except ModuleNotFoundError:  # imported as ``python.sm120_run_tactic_search`` in tests
    from python.sm120_tactic_plan import (
        candidate_override as plan_candidate_override,
        load_plan,
        validate_plan,
    )
    from python.sm120_fat_scan import launch_symbol, symbol_token

try:
    from sm120_device import query_cuda_device
except ModuleNotFoundError:  # imported as ``python.sm120_run_tactic_search``
    from python.sm120_device import query_cuda_device


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


def implementation_identity(repo: pathlib.Path) -> dict[str, str | None]:
    """Hash the code that makes completed-row resume decisions meaningful."""
    relative_paths = (
        "cpp/CMakeLists.txt",
        "cpp/command/benchmarknn.cpp",
        "cpp/neuralnet/cudabackend_sm120.cpp",
        "cpp/neuralnet/cudabackend_sm120.h",
        "cpp/neuralnet/cudabackend_sm120_aot_registry.cu",
        "cpp/neuralnet/cudabackend_sm120_kernels.cu",
        "cpp/neuralnet/cudabackend_sm120_kernels.h",
        "cpp/neuralnet/fa4_aot/build_aot.py",
        "python/sm120_benchmark_metrics.py",
        "python/sm120_device.py",
        "python/sm120_coordinate_search.py",
        "python/sm120_fat_scan.py",
        "python/sm120_generate_cute_qkv_aot.py",
        "python/sm120_generate_tilelang_aot.py",
        "python/sm120_historical_ffn/generate.py",
        "python/sm120_historical_ffn/manifest.json",
        "python/sm120_run_tactic_search.py",
        "python/sm120_tactic_search.py",
        "python/sm120_measure_joint_plan.py",
        "python/sm120_tactic_plan.py",
    )
    result = {
        relative: sha256_file(repo / relative)
        for relative in relative_paths
        if (repo / relative).is_file()
    }
    head = capture_command(["git", "-C", str(repo), "rev-parse", "HEAD"])
    result["git_head"] = head.get("stdout", "").strip() or None
    return result


def load_candidate_selection(
    selection_path: pathlib.Path, space_path: pathlib.Path, space: dict,
    family: str, batches: list[int],
) -> dict:
    """Load a per-(family,batch) S1 retention file for the S2 runner."""
    payload = json.loads(selection_path.read_text())
    if payload.get("schema") != 1 or not isinstance(payload.get("groups"), list):
        raise ValueError(f"unsupported candidate selection: {selection_path}")
    allow_empty_groups = payload.get("selection_kind") == "s1_complement"
    selected: dict[int, set[str]] = {}
    for group in payload["groups"]:
        if group.get("family") != family:
            continue
        batch = int(group["batch"])
        retained = group.get("retained")
        if not isinstance(retained, list) or (not retained and not allow_empty_groups):
            raise ValueError(f"selection has no retained candidates for {family}/B{batch}")
        if batch in selected:
            raise ValueError(f"candidate selection has duplicate {family}/B{batch} groups")
        selected[batch] = {str(candidate_id) for candidate_id in retained}

    missing = sorted(set(batches) - set(selected))
    if missing:
        raise ValueError(
            f"candidate selection has no {family} groups for batches {missing}"
        )
    space_sha256 = sha256_file(space_path)
    source_manifest_hashes = []
    for manifest_value in payload.get("source_manifests", []):
        manifest_path = pathlib.Path(manifest_value)
        if not manifest_path.is_file():
            continue
        manifest = json.loads(manifest_path.read_text())
        manifest_space_sha256 = manifest.get("space_sha256")
        if manifest_space_sha256 and manifest_space_sha256 != space_sha256:
            raise ValueError(
                "candidate selection search-space hash does not match the current space: "
                f"{manifest_path}"
            )
        source_manifest_hashes.append({
            "path": str(manifest_path.resolve()),
            "sha256": sha256_file(manifest_path),
        })

    batch_spaces = {item["batch"]: item for item in space.get("batches", [])}
    for batch in batches:
        available = {item["id"] for item in batch_spaces[batch].get(family, [])}
        unknown = sorted(selected[batch] - available)
        if unknown:
            raise ValueError(
                f"candidate selection contains IDs outside the current space for "
                f"{family}/B{batch}: {unknown}"
            )
    return {
        "path": str(selection_path.resolve()),
        "sha256": sha256_file(selection_path),
        "selection_metric": payload.get("selection_metric"),
        "source_manifests": source_manifest_hashes,
        "batches": {batch: sorted(selected[batch]) for batch in batches},
    }


def selected_tilelang_keys(
    space: dict, family: str, batches: list[int],
    allowed_ids: set[str], candidate_selection: dict | None,
    planned_entries: dict[int, dict] | None,
) -> set[tuple[int, str]]:
    """Return exact TileLang entries the current invocation will measure."""
    result = set()
    batch_spaces = {item["batch"]: item for item in space["batches"]}
    for batch in batches:
        candidates = batch_spaces[batch][family]
        if planned_entries is not None:
            planned_id = planned_entries[batch]["candidate_id"]
            candidates = [item for item in candidates if item["id"] == planned_id]
        elif candidate_selection is not None:
            selected_ids = set(candidate_selection["batches"][batch])
            candidates = [item for item in candidates if item["id"] in selected_ids]
        elif allowed_ids:
            candidates = [item for item in candidates if item["id"] in allowed_ids]
        result.update(
            (batch, item["id"])
            for item in candidates
            if item.get("implementation", "tilelang") == "tilelang"
        )
    return result


def load_fat_bundle(
    manifest_path: pathlib.Path, family: str, space_path: pathlib.Path,
    space: dict, required_keys: set[tuple[int, str]],
) -> dict:
    """Load a completed fat bundle without mutating or regenerating it."""
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema") != 1 or manifest.get("family") != family:
        raise ValueError(f"unsupported fat manifest for {family}: {manifest_path}")
    if manifest.get("registry_abi") != "sm120-numeric-sm-count-v2":
        raise ValueError(
            "fat manifest uses the obsolete product-name registry ABI; "
            "rerun sm120_prepare_tilelang_fat_scan.py --reuse-existing to "
            f"rewrite it: {manifest_path}"
        )
    legacy_complete = (
        manifest.get("complete") is None
        and manifest.get("finished_utc")
        and isinstance(manifest.get("entries"), list)
        and len(manifest["entries"]) == sum(
            1
            for batch_space in space.get("batches", [])
            for candidate in batch_space.get(family, [])
            if candidate.get("implementation", "tilelang") == "tilelang"
        )
    )
    if not manifest.get("complete") and not legacy_complete:
        raise ValueError(f"fat manifest is incomplete: {manifest_path}")
    expected_space_sha256 = sha256_file(space_path)
    exact_space_match = manifest.get("space_sha256") == expected_space_sha256

    registry_path = pathlib.Path(manifest["registry_source"])
    if not registry_path.is_file():
        raise ValueError(f"fat manifest registry is missing: {registry_path}")
    registry_sha256 = manifest.get("registry_sha256")
    if registry_sha256 and sha256_file(registry_path) != registry_sha256:
        raise ValueError(f"fat manifest registry hash mismatch: {registry_path}")

    batch_spaces = {item["batch"]: item for item in space["batches"]}
    entries = {}
    for entry in manifest.get("entries", []):
        key = (int(entry["batch"]), str(entry["candidate_id"]))
        if key in entries:
            raise ValueError(f"duplicate fat manifest entry: B{key[0]}/{key[1]}")
        batch_space = batch_spaces.get(key[0])
        if batch_space is None:
            raise ValueError(f"fat entry is outside the current space: {key}")
        current_candidate = next(
            (item for item in batch_space.get(family, []) if item["id"] == key[1]),
            None,
        )
        if current_candidate is None or entry.get("candidate") != current_candidate:
            raise ValueError(f"fat entry candidate mismatch for B{key[0]}/{key[1]}")
        token = symbol_token(family, key[0], key[1])
        if entry.get("symbol_token") != token or entry.get("launch_symbol") != launch_symbol(family, token):
            raise ValueError(f"fat entry symbol mismatch for B{key[0]}/{key[1]}")
        source_path = pathlib.Path(entry["source"])
        metadata_path = pathlib.Path(entry["metadata"])
        if not source_path.is_file() or not metadata_path.is_file():
            raise ValueError(f"fat entry artifact is missing for B{key[0]}/{key[1]}")
        if sha256_file(source_path) != entry.get("source_sha256"):
            raise ValueError(f"fat entry source hash mismatch for B{key[0]}/{key[1]}")
        if sha256_file(metadata_path) != entry.get("metadata_sha256"):
            raise ValueError(f"fat entry metadata hash mismatch for B{key[0]}/{key[1]}")
        entries[key] = entry

    missing = sorted(required_keys - set(entries))
    if missing:
        raise ValueError(
            "fat manifest is missing requested exact entries: "
            + ", ".join(f"B{batch}/{candidate_id}" for batch, candidate_id in missing)
        )
    manifest["path"] = str(manifest_path.resolve())
    manifest["sha256"] = sha256_file(manifest_path)
    manifest["loaded_space_compatibility"] = (
        "exact_sha256" if exact_space_match else "exact_candidate_projection"
    )
    manifest["loaded_space_sha256"] = expected_space_sha256
    return manifest


def capture_command(
    command: list[str], timeout: float = 10.0,
    max_output_chars: int | None = 20000,
) -> dict:
    try:
        completed = subprocess.run(
            command, capture_output=True, text=True, timeout=timeout,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"command": command, "error": str(error)}
    stdout = completed.stdout
    stderr = completed.stderr
    if max_output_chars is not None:
        stdout = stdout[-max_output_chars:]
        stderr = stderr[-max_output_chars:]
    return {
        "command": command,
        "returncode": completed.returncode,
        "stdout": stdout,
        "stderr": stderr,
    }


def collect_environment(
    repo: pathlib.Path, config_path: pathlib.Path, model_path: pathlib.Path,
    extra_roots: dict[str, pathlib.Path] | None = None,
    extra_python: str | None = None,
    device_properties: dict | None = None,
) -> dict:
    """Capture reproducibility context without making it a run-time gate."""
    packages = {}
    try:
        from importlib import metadata
        for distribution in (
            "torch", "tilelang", "tvm", "flash-attn", "triton", "cutlass",
            "cudnn-frontend", "nvidia-cudnn-cu12", "nvidia-cudnn-cu13",
        ):
            try:
                packages[distribution] = metadata.version(distribution)
            except metadata.PackageNotFoundError:
                packages[distribution] = None
    except Exception as error:  # pragma: no cover - defensive metadata path
        packages["_error"] = str(error)

    cudnn_from_torch = None
    torch_version = packages.get("torch")
    if torch_version is not None:
        try:
            import torch
            cudnn_from_torch = torch.backends.cudnn.version()
            torch_version = torch.__version__
        except Exception as error:  # pragma: no cover - depends on host runtime
            packages["torch_import_error"] = str(error)
    packages["torch"] = torch_version

    git_status = capture_command(["git", "-C", str(repo), "status", "--short"])
    git_head = capture_command(["git", "-C", str(repo), "rev-parse", "HEAD"])
    untracked = []
    for line in git_status.get("stdout", "").splitlines():
        if not line.startswith("?? "):
            continue
        relative = line[3:]
        path = repo / relative
        if path.is_file():
            untracked.append({
                "path": relative,
                "size": path.stat().st_size,
                "sha256": sha256_file(path),
            })
    try:
        git_diff_completed = subprocess.run(
            ["git", "-C", str(repo), "diff", "--binary", "HEAD"],
            capture_output=True, timeout=10.0,
        )
        git_diff = {
            "command": ["git", "-C", str(repo), "diff", "--binary", "HEAD"],
            "returncode": git_diff_completed.returncode,
            "stdout_tail": git_diff_completed.stdout[-20000:].decode(
                "utf-8", errors="replace"
            ),
            "stderr": git_diff_completed.stderr[-20000:].decode(
                "utf-8", errors="replace"
            ),
        }
        diff_sha256 = hashlib.sha256(git_diff_completed.stdout).hexdigest()
    except (OSError, subprocess.TimeoutExpired) as error:
        git_diff = {"error": str(error)}
        diff_sha256 = None
    third_party = {}
    third_party_root = repo.parent / "third_party"
    third_party_paths = {
        name: third_party_root / name
        for name in (
            "cutlass", "TileLang", "flash-attention", "triton", "cudnn-frontend",
        )
    }
    for name, path in (extra_roots or {}).items():
        third_party_paths[name] = path.resolve()
    for name, path in third_party_paths.items():
        if path.is_dir():
            status = capture_command(["git", "-C", str(path), "status", "--short"])
            revision = capture_command(["git", "-C", str(path), "rev-parse", "HEAD"])
            third_party[name] = {
                "path": str(path),
                "revision": revision.get("stdout", "").strip(),
                "dirty": bool(status.get("stdout", "").strip()),
                "status": status.get("stdout", ""),
                "revision_probe": revision,
            }

    config_text = config_path.read_text()
    commands = {
        "nvcc_version": capture_command(["nvcc", "--version"]),
        "nvidia_smi_query": capture_command([
            "nvidia-smi", "--query-gpu=name,driver_version,memory.total,compute_cap",
            "--format=csv,noheader",
        ]),
        "nvidia_smi": capture_command(["nvidia-smi"]),
        "ldconfig": capture_command(["ldconfig", "-p"]),
        "python_pip_freeze": capture_command([
            sys.executable, "-m", "pip", "freeze",
        ], max_output_chars=200000),
        "cmake_version": capture_command(["cmake", "--version"]),
        "gcc_version": capture_command(["gcc", "--version"]),
        "gxx_version": capture_command(["g++", "--version"]),
    }
    if extra_python and pathlib.Path(extra_python).resolve() != pathlib.Path(sys.executable).resolve():
        commands["extra_python_version"] = capture_command([extra_python, "--version"])
        commands["extra_python_pip_freeze"] = capture_command([
            extra_python, "-m", "pip", "freeze",
        ], max_output_chars=200000)
    return {
        "schema": 1,
        "captured_utc": utc_now(),
        "host": capture_command(["uname", "-a"]),
        "python": {
            "executable": sys.executable,
            "version": sys.version,
        },
        "packages": packages,
        "cudnn_version_from_torch": cudnn_from_torch,
        "cuda_device": device_properties,
        "environment": {
            key: os.environ.get(key)
            for key in (
                "CUDA_HOME", "CUDA_PATH", "PATH", "LD_LIBRARY_PATH",
                "CMAKE_PREFIX_PATH", "CC", "CXX", "TORCH_CUDA_ARCH_LIST",
                "PYTHONPATH", "TILELANG_HOME", "TVM_HOME", "CUTLASS_ROOT",
            )
            if os.environ.get(key) is not None
        },
        "commands": commands,
        "git": {
            "head": git_head.get("stdout", "").strip(),
            "status": git_status.get("stdout", ""),
            "dirty": bool(git_status.get("stdout", "").strip()),
            "diff_sha256": diff_sha256,
            "diff": git_diff,
            "untracked": untracked,
        },
        "third_party": third_party,
        "config": {
            "path": str(config_path),
            "sha256": sha256_file(config_path),
            "text": config_text,
        },
        "model": {
            "path": str(model_path),
            "sha256": sha256_file(model_path),
        },
    }


def reproducibility_identity(snapshot: dict) -> dict:
    """Stable toolchain subset used to keep resumed rows reproducible."""
    command_identity = {}
    for key in ("nvcc_version", "cmake_version", "gcc_version", "gxx_version"):
        value = snapshot.get("commands", {}).get(key)
        if isinstance(value, dict):
            command_identity[key] = {
                field: value.get(field)
                for field in ("returncode", "stdout", "stderr", "error")
                if field in value
            }
    third_party = {
        name: {
            key: value.get(key)
            for key in ("revision", "dirty", "status")
        }
        for name, value in snapshot.get("third_party", {}).items()
        if isinstance(value, dict)
    }
    identity = {
        "python": snapshot.get("python"),
        "packages": snapshot.get("packages"),
        "cudnn_version_from_torch": snapshot.get("cudnn_version_from_torch"),
        "environment": snapshot.get("environment"),
        "commands": command_identity,
        "third_party": third_party,
    }
    return json.loads(json.dumps(identity))


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
    isolate_family: bool = False, tactic_plan: dict | None = None,
    batch: int | None = None,
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
        elif not isolate_family and tactic_plan is not None and batch is not None:
            planned = tactic_plan.get("families", {}).get(other_family, {}).get(
                "batches", {}
            ).get(str(batch))
            if planned is None:
                requested = "auto"
            else:
                requested = (
                    planned["candidate_id"]
                    if planned.get("implementation", "tilelang") != "fallback"
                    else "disabled"
                )
        else:
            requested = "disabled" if isolate_family else "auto"
        values.append(f"{key}={requested}")
    if family == "l2":
        for key, value in candidate_value["config"].items():
            if isinstance(value, bool):
                value = str(value).lower()
            values.append(f"{key}={value}")
    elif not isolate_family and tactic_plan is not None and batch is not None:
        planned = tactic_plan.get("families", {}).get("l2", {}).get(
            "batches", {}
        ).get(str(batch))
        if planned is not None:
            for key, value in planned["candidate"].get("config", {}).items():
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
    elif not isolate_family and tactic_plan is not None and batch is not None:
        planned = tactic_plan.get("families", {}).get("fa4", {}).get(
            "batches", {}
        ).get(str(batch))
        if planned is not None:
            values.extend(plan_candidate_override("fa4", planned["candidate"]))
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
    parser.add_argument(
        "--cutlass-root",
        default="/workspace/third_party/cutlass",
        help="clean pinned CUTLASS source used by the packed-QKV generator",
    )
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--cmake-arg", action="append", default=[])
    parser.add_argument(
        "--sm120-only",
        action="store_true",
        help="compile the scan binary only for SM120 instead of all CUDA architectures",
    )
    parser.add_argument("--runner", default="")
    parser.add_argument("--override-config", default="")
    parser.add_argument("--candidate-ids", default="")
    parser.add_argument(
        "--candidate-selection",
        default="",
        help="schema-1 per-batch S1 retention output from sm120_select_local_candidates.py",
    )
    parser.add_argument(
        "--tactic-plan",
        default="",
        help=(
            "portable schema-1 plan; validate it and measure only its selected "
            "exact-batch tactics"
        ),
    )
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
    parser.add_argument(
        "--fat-manifest",
        default="",
        help=(
            "read a completed fat-scan manifest without regenerating or "
            "rewriting its sources"
        ),
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.fat_scan and args.family not in TILELANG_FAMILIES:
        parser.error("--fat-scan currently supports only FFN, QKV, and linear2")
    if args.fat_manifest and not args.fat_scan:
        parser.error("--fat-manifest requires --fat-scan")
    if args.fat_manifest and args.reuse_fat_generated:
        parser.error("--fat-manifest cannot be combined with --reuse-fat-generated")

    device_properties = query_cuda_device(args.device)

    repo = pathlib.Path(args.repo).resolve()
    build_dir = pathlib.Path(args.build_dir).resolve()
    active_dir = pathlib.Path(args.active_source_dir).resolve()
    output = pathlib.Path(args.output).resolve()
    logs = output.parent / f"{output.stem}-logs"
    generated = output.parent / f"{output.stem}-generated"
    runner = shlex.split(args.runner)
    fat_manifest_argument = (
        pathlib.Path(args.fat_manifest).resolve() if args.fat_manifest else None
    )
    space_path = pathlib.Path(args.space).resolve()
    config_path = pathlib.Path(args.config).resolve()
    model_path = pathlib.Path(args.model).resolve()
    space = json.loads(space_path.read_text())
    if space.get("schema") != 2:
        raise ValueError("--space must be schema 2")
    batches = parse_int_set(args.batches)
    allowed_ids = {item for item in args.candidate_ids.split(",") if item}
    batch_spaces = {item["batch"]: item for item in space["batches"]}
    missing = sorted(set(batches) - set(batch_spaces))
    if missing:
        raise ValueError(f"batches outside materialized space: {missing}")

    candidate_selection = None
    if args.candidate_selection:
        if args.candidate_ids:
            parser.error("--candidate-ids cannot be combined with --candidate-selection")
        if args.tactic_plan:
            parser.error("--candidate-selection cannot be combined with --tactic-plan")
        candidate_selection = load_candidate_selection(
            pathlib.Path(args.candidate_selection).resolve(),
            space_path, space, args.family, batches,
        )

    tactic_plan = None
    tactic_plan_path = None
    planned_entries_by_family = {}
    if args.tactic_plan:
        if args.candidate_ids:
            parser.error("--candidate-ids cannot be combined with --tactic-plan")
        if args.isolate_family:
            parser.error("--isolate-family is incompatible with --tactic-plan")
        tactic_plan_path = pathlib.Path(args.tactic_plan).resolve()
        tactic_plan = load_plan(tactic_plan_path)
        # A scan-bypass plan must pin every tactic family used by the common
        # graph.  Otherwise an omitted family would silently fall back to
        # runtime auto-selection.
        space["_path"] = str(space_path)
        for planned_family in SEARCH_FAMILIES:
            planned_entries_by_family[planned_family] = validate_plan(
                tactic_plan, space, model_path, planned_family, batches,
                args.streams, config_path, device_properties=device_properties,
            )

    common_graph_policy = (
        "isolated diagnostic" if args.isolate_family
        else "preserve config and exact-batch automatic winners"
    )
    cmake_args = list(args.cmake_arg)
    if args.sm120_only:
        cmake_args.append("-DKATAGO_CUDA_ARCHITECTURES=120")
    regime = {
        "config": str(config_path),
        "config_sha256": sha256_file(config_path),
        "model": str(model_path),
        "model_sha256": sha256_file(model_path),
        "cuda_device_ordinal": args.device,
        "cuda_device_properties": device_properties,
        "streams": args.streams,
        "iterations": args.iterations,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "runner": runner,
        "cutlass_root": str(pathlib.Path(args.cutlass_root).resolve()),
        "extra_override_config": args.override_config,
        "common_graph_policy": common_graph_policy,
        "fat_scan": args.fat_scan,
        "cmake_args": list(cmake_args),
        "space_sha256": sha256_file(space_path),
        "tactic_plan": (
            {
                "path": str(tactic_plan_path),
                "sha256": sha256_file(tactic_plan_path),
                "plan_id": tactic_plan["plan_id"],
            }
            if tactic_plan is not None else None
        ),
        "candidate_selection": (
            {
                "path": candidate_selection["path"],
                "sha256": candidate_selection["sha256"],
            }
            if candidate_selection is not None else None
        ),
        "candidate_ids": sorted(allowed_ids),
        "fat_manifest": (
            {
                "path": str(fat_manifest_argument),
                "sha256": sha256_file(fat_manifest_argument),
            }
            if fat_manifest_argument is not None else None
        ),
        # A config/model/space match is insufficient when runtime dispatch or
        # generators changed. Completed rows are resumed only under this exact
        # implementation identity.
        "implementation_identity": implementation_identity(repo),
    }

    environment = collect_environment(
        repo, config_path, model_path,
        {"cutlass": pathlib.Path(args.cutlass_root).resolve()},
        args.fa4_python,
        device_properties,
    )
    regime["reproducibility_identity"] = reproducibility_identity(environment)

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
    payload.setdefault("environment_snapshots", []).append(environment)
    payload["space_sha256"] = sha256_file(space_path)
    if candidate_selection is not None:
        payload["candidate_selection"] = candidate_selection
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
    fat_candidate_ids = args.candidate_ids
    if candidate_selection is not None:
        fat_candidate_ids = ",".join(sorted({
            candidate_id
            for candidate_ids in candidate_selection["batches"].values()
            for candidate_id in candidate_ids
        }))
    if args.fat_scan and tactic_plan is not None:
        planned_for_fat = planned_entries_by_family[args.family]
        unsupported = sorted({
            entry.get("implementation", "tilelang")
            for entry in planned_for_fat.values()
            if entry.get("implementation", "tilelang") != "tilelang"
        })
        if unsupported:
            raise ValueError(
                "--fat-scan with --tactic-plan requires TileLang selections; "
                f"{args.family} plan contains {unsupported}"
            )
        fat_candidate_ids = ",".join(sorted({
            entry["candidate_id"] for entry in planned_for_fat.values()
        }))
    if args.fat_scan:
        required_fat_keys = selected_tilelang_keys(
            space, args.family, batches, allowed_ids, candidate_selection,
            planned_entries_by_family.get(args.family),
        )
        if args.fat_manifest:
            fat_manifest_path = fat_manifest_argument
            fat_manifest = load_fat_bundle(
                fat_manifest_path, args.family, space_path, space,
                required_fat_keys,
            )
        else:
            fat_output_dir = generated / f"fat-{args.family}"
            fat_manifest = prepare_fat_bundle(
                repo, space_path, args.family, args.batches, fat_candidate_ids,
                args.device, fat_output_dir, args.s1_warmup,
                args.s1_iterations, runner, logs / "fat-prepare",
                args.reuse_fat_generated,
            )
            fat_manifest_path = fat_output_dir / "manifest.json"
        fat_entries = {
            (item["batch"], item["candidate_id"]): item
            for item in fat_manifest["entries"]
        }
        fat_cache_values[args.family] = {
            "registry": fat_manifest["registry_source"],
            "sources": ";".join(fat_manifest["sources"]),
        }
        payload["fat_scan_manifest"] = str(
            fat_manifest_path.resolve()
        )
        payload["fat_scan_manifest_sha256"] = sha256_file(fat_manifest_path)
        payload["build_policy"] = (
            "reuse completed fat bundle or generate all selected TUs, configure/link once"
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
        payload["bootstrap_commands"] = [command]
        run_checked(command, logs / "bootstrap-fa4")
        cmake_args.extend([
            f"-DSM120_SEARCH_FA4_SOURCE={fa4_bridge}",
            f"-DSM120_SEARCH_FA4_OBJECT={fa4_object}",
        ])

    generator = repo / "python/sm120_generate_tilelang_aot.py"
    cute_qkv_generator = repo / "python/sm120_generate_cute_qkv_aot.py"
    configure_command = [
        "cmake", "-S", str(repo / "cpp"), "-B", str(build_dir),
        "-DUSE_BACKEND=CUDA", "-DCMAKE_BUILD_TYPE=Release",
    ]
    configure_command.extend(
        f"-D{SLOT_CACHE_KEYS[family]}={path}"
        for family, path in active.items()
    )
    configure_command.extend(cmake_args)
    configure_command.append("-DSM120_SEARCH_QKV_OBJECT=")
    payload["build_commands"] = {
        "configure": configure_command,
        "initial_build": ["cmake", "--build", str(build_dir), f"-j{args.jobs}"],
        "candidate_build": ["cmake", "--build", str(build_dir), f"-j{args.jobs}"],
        "runner_invocation": [sys.executable, *sys.argv],
        "cmake_args": list(cmake_args),
        "jobs": args.jobs,
        "source_scripts": {
            str(path): sha256_file(path)
            for path in (
                generator,
                cute_qkv_generator,
                repo / "python/sm120_prepare_tilelang_fat_scan.py",
            )
            if path.is_file()
        },
    }
    payload["generator_hashes"] = payload["build_commands"]["source_scripts"]
    write_result(output, payload)
    binary = configure_build(
        repo, build_dir, active, args.jobs, logs, cmake_args,
    )
    cache_path = build_dir / "CMakeCache.txt"
    payload["build_artifacts"] = {
        "binary": str(binary),
        "binary_sha256": sha256_file(binary),
        "cmake_cache": str(cache_path),
        "cmake_cache_sha256": sha256_file(cache_path) if cache_path.is_file() else None,
    }
    write_result(output, payload)
    qkv_build_mode = "planar"
    # Keep the CuTe bridge/object paths stable after the first candidate.  If
    # CMake sees a new source/object path for every exact batch, Make treats the
    # whole target as changed and recompiles the large C++ target repeatedly.
    # Replacing the contents at stable paths lets subsequent batches do only
    # the tiny bridge compile and relink.
    qkv_cute_bridge = active_dir / "active-qkv-cute.cu"
    qkv_cute_header = active_dir / "sm120_qkv_cute_active.h"
    qkv_cute_object = active_dir / "active-qkv-cute.o"
    measurement_index = len(payload["rows"])
    override_plan = None
    if tactic_plan is not None:
        # This invocation materializes one family.  The complete plan remains
        # available in the result and in ``apply`` for the final deployment;
        # do not claim that other families are linked into this family-only
        # build unless their own fat bundle is supplied.
        override_plan = {
            "families": {args.family: tactic_plan["families"][args.family]}
        }

    for batch in batches:
        candidates = batch_spaces[batch][args.family]
        if tactic_plan is not None:
            planned = planned_entries_by_family[args.family][batch]
            candidates = [
                item for item in candidates
                if item["id"] == planned["candidate_id"]
            ]
        elif candidate_selection is not None:
            selected_ids = set(candidate_selection["batches"][batch])
            candidates = [item for item in candidates if item["id"] in selected_ids]
        elif allowed_ids:
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
                        row["generator_metadata_path"] = fat_entry["metadata"]
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
                        row.setdefault("commands", {})["generate"] = command
                        run_checked(command, logs / f"{prefix}-generate")
                        metadata_path = (
                            candidate_dir / f"{args.family}-{candidate_id}.json"
                        )
                        row["generator_metadata"] = json.loads(
                            metadata_path.read_text()
                        )
                        row["generator_metadata_path"] = str(metadata_path)
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
                    row.setdefault("commands", {})["generate"] = command
                    run_checked(command, logs / f"{prefix}-generate")
                    metadata_path = candidate_dir / f"ffn-{candidate_id}.json"
                    row["generator_metadata"] = json.loads(metadata_path.read_text())
                    row["generator_metadata_path"] = str(metadata_path)
                    run_checked(
                        ["cmake", "--build", str(build_dir), f"-j{args.jobs}"],
                        logs / f"{prefix}-build",
                    )
                elif implementation == "cute" and args.family == "qkv":
                    candidate_dir = generated / f"b{batch}" / candidate_id
                    generated_bridge_path = candidate_dir / "sm120_qkv_cute_active.cu"
                    max_active_clusters = candidate_value.get("max_active_clusters")
                    command = runner + [
                        sys.executable, str(cute_qkv_generator),
                        "--batch", str(batch),
                        "--output-dir", str(candidate_dir),
                        "--bridge-path", str(generated_bridge_path),
                        "--candidate-id", candidate_id,
                        "--device", str(args.device),
                        "--cutlass-root", args.cutlass_root,
                    ]
                    if max_active_clusters is not None:
                        command.extend([
                            "--max-active-clusters", str(max_active_clusters),
                        ])
                    row.setdefault("commands", {})["generate"] = command
                    row["generator_parameters"] = {
                        "max_active_clusters": (
                            int(max_active_clusters)
                            if max_active_clusters is not None
                            else int(device_properties["multiprocessor_count"])
                        ),
                        "source": (
                            "candidate_space"
                            if max_active_clusters is not None
                            else "cudaDevAttrMultiProcessorCount"
                        ),
                    }
                    run_checked(command, logs / f"{prefix}-generate")
                    metadata_path = candidate_dir / "sm120_qkv_cute_active.json"
                    row["generator_metadata"] = json.loads(metadata_path.read_text())
                    row["generator_metadata_path"] = str(metadata_path)
                    generated_object_path = candidate_dir / "sm120_qkv_cute_active.o"
                    generated_header_path = candidate_dir / "sm120_qkv_cute_active.h"
                    shutil.copyfile(generated_bridge_path, qkv_cute_bridge)
                    shutil.copyfile(generated_header_path, qkv_cute_header)
                    shutil.copyfile(generated_object_path, qkv_cute_object)
                    row["commands"]["materialize"] = {
                        "generated_bridge": str(generated_bridge_path),
                        "generated_header": str(generated_header_path),
                        "generated_object": str(generated_object_path),
                        "stable_bridge": str(qkv_cute_bridge),
                        "stable_header": str(qkv_cute_header),
                        "stable_object": str(qkv_cute_object),
                    }
                    if qkv_build_mode != "packed":
                        configure_qkv_slot(
                            repo, build_dir, active, qkv_cute_bridge,
                            qkv_cute_object, args.jobs, logs, cmake_args, prefix,
                        )
                    else:
                        run_checked(
                            ["cmake", "--build", str(build_dir), f"-j{args.jobs}"],
                            logs / f"{prefix}-build",
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
                    row.setdefault("commands", {})["generate"] = command
                    run_checked(command, logs / f"{prefix}-generate")
                    row["generator_metadata"] = json.loads(
                        (fa4_active_dir / "active-fa4.json").read_text()
                    )
                    row["generator_metadata_path"] = str(fa4_active_dir / "active-fa4.json")
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
                    args.override_config, args.isolate_family, override_plan, batch,
                )
                for repeat in range(args.repeats):
                    command = runner + [
                        str(binary), "benchmarknn",
                        "-config", str(config_path),
                        "-override-config", override,
                        "-model", str(model_path),
                        "-iterations", str(args.iterations),
                        "-warmup", str(args.warmup),
                        "-batch-size", str(batch),
                        "-boardsize", "19", "-json",
                    ]
                    row.setdefault("commands", {}).setdefault("benchmark", []).append(command)
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
                row["benchmark_command_template"] = [
                    *runner, str(binary), "benchmarknn",
                    "-config", str(config_path),
                    "-override-config", override,
                    "-model", str(model_path),
                    "-iterations", str(args.iterations),
                    "-warmup", str(args.warmup),
                    "-batch-size", str(batch),
                    "-boardsize", "19", "-json",
                ]
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
