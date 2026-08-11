#!/usr/bin/env python3
"""Build only the CUDA artifacts required by one certified runtime plan."""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import pathlib
import subprocess
import sys
from typing import Any


MODEL_NAME = "b11c768h12nbt3tflrs-fson-silu.bin.gz"


def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_plan_checksum(path: pathlib.Path) -> None:
    manifest = path.parent / "SHA256SUMS"
    if not manifest.is_file():
        raise RuntimeError(f"plan checksum manifest is missing: {manifest}")
    subprocess.run(
        ["sha256sum", "--check", "--quiet", manifest.name],
        cwd=path.parent, check=True,
    )


def plan_product_name(plan: dict[str, Any]) -> str:
    target = plan.get("target")
    if not isinstance(target, dict):
        raise ValueError("plan has no target")
    devices = target.get("cuda_device_capabilities_at_scan")
    if not isinstance(devices, list) or not devices or not isinstance(devices[0], dict):
        raise ValueError("plan has no producer CUDA device")
    name = devices[0].get("name")
    if not isinstance(name, str) or not name:
        raise ValueError("plan has no producer CUDA product name")
    return name


def select_product_plan(
    plans: list[tuple[pathlib.Path, dict[str, Any]]], product_name: str,
) -> tuple[pathlib.Path, dict[str, Any]]:
    matches = [item for item in plans if plan_product_name(item[1]) == product_name]
    if not matches:
        available = sorted({plan_product_name(plan) for _, plan in plans})
        suffix = f"; bundled products: {', '.join(available)}" if available else ""
        raise RuntimeError(f"no bundled plan for CUDA product {product_name!r}{suffix}")
    if len(matches) != 1:
        paths = ", ".join(str(path) for path, _ in matches)
        raise RuntimeError(
            f"CUDA product {product_name!r} has multiple plan entries: {paths}"
        )
    return matches[0]


def select_plan(
    *, explicit: pathlib.Path | None, roots: list[pathlib.Path],
    model: pathlib.Path, device_properties: dict[str, object],
) -> tuple[pathlib.Path, dict[str, Any]]:
    from cuda_tactic_workflow import load_plan, validate_plan

    paths: list[pathlib.Path] = []
    if explicit is not None:
        paths = [explicit.resolve()]
    else:
        for root in roots:
            if root.is_dir():
                paths.extend(sorted(root.glob("**/best-tactic-plan.json")))
    loaded: list[tuple[pathlib.Path, dict[str, Any]]] = []
    failures: list[str] = []
    seen_paths: set[pathlib.Path] = set()
    seen_plan_hashes: set[str] = set()
    for path in paths:
        path = path.resolve()
        if path in seen_paths:
            continue
        seen_paths.add(path)
        if not path.is_file():
            failures.append(f"missing plan: {path}")
            continue
        try:
            verify_plan_checksum(path)
            plan_file_sha = sha256(path)
            if plan_file_sha in seen_plan_hashes:
                continue
            seen_plan_hashes.add(plan_file_sha)
            plan = load_plan(path)
            plan_product_name(plan)
        except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
            failures.append(f"{path}: {exc}")
            continue
        loaded.append((path, plan))
    try:
        if explicit is not None:
            if len(loaded) != 1:
                raise RuntimeError("explicit plan could not be loaded")
            selected = loaded[0]
        else:
            product_name = device_properties.get("name")
            if not isinstance(product_name, str) or not product_name:
                raise RuntimeError("CUDA Runtime did not return a product name")
            selected = select_product_plan(loaded, product_name)
        result = validate_plan(
            selected[1], model=model, device_properties=device_properties,
        )
        if not result["production_ready"]:
            raise ValueError("plan is not production-ready")
        return selected
    except (ValueError, RuntimeError) as exc:
        detail = "\n".join(failures[-8:])
        raise RuntimeError(
            str(exc) + (f"\n{detail}" if detail else "")
        ) from exc


def valid_existing_build(
    manifest_path: pathlib.Path, plan: dict[str, Any], plan_file_sha: str,
) -> dict[str, Any] | None:
    try:
        manifest = json.loads(manifest_path.read_text())
        if (
            manifest.get("schema") != 1 or
            manifest.get("kind") != "cuda-plan-build" or
            manifest.get("plan_id") != plan.get("plan_id") or
            manifest.get("plan_sha256") != plan.get("plan_sha256") or
            manifest.get("source_plan_file_sha256") != plan_file_sha
        ):
            return None
        binary = (manifest_path.parent / manifest["binary"]).resolve()
        artifact = (manifest_path.parent / manifest["artifact_bundle"]).resolve()
        space = (manifest_path.parent / manifest["space"]).resolve()
        if not binary.is_file() or not artifact.is_file() or not space.is_file():
            return None
        if sha256(binary) != manifest.get("binary_sha256"):
            return None
        if sha256(artifact) != manifest.get("artifact_bundle_sha256"):
            return None
        if sha256(space) != manifest.get("space_sha256"):
            return None
        return manifest
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
        return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", type=pathlib.Path, required=True)
    parser.add_argument("--repo", type=pathlib.Path, required=True)
    parser.add_argument("--autotune", type=pathlib.Path, required=True)
    parser.add_argument("--plans-root", type=pathlib.Path, action="append", default=[])
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--plan", type=pathlib.Path)
    parser.add_argument("--jobs", type=int)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    prefix = args.prefix.resolve()
    repo = args.repo.resolve()
    autotune = args.autotune.resolve()
    python = prefix / "venv/bin/python"
    model = prefix / "assets" / MODEL_NAME
    for path, label in (
        (python, "configured Python"), (model, "model"),
        (repo / "python/cuda_tactic_workflow.py", "workflow"),
        (autotune, "autotune build driver"),
    ):
        if not path.is_file():
            raise RuntimeError(f"{label} is missing: {path}")

    sys.path.insert(0, str(repo / "python"))
    from cuda_tactic_workflow import (
        materialize_space, restrict_space_to_plan, validate_plan, write_json,
    )
    from portable_cuda_device import query_cuda_device

    device_properties = query_cuda_device(args.device)
    plan_path, plan = select_plan(
        explicit=args.plan,
        roots=[path.resolve() for path in args.plans_root],
        model=model,
        device_properties=device_properties,
    )
    batches = plan.get("batches", [])
    if not isinstance(batches, list) or len(batches) != 1:
        raise RuntimeError("build-only requires a single-batch production plan")
    batch = int(batches[0])
    target = plan["target"]
    architecture = str(target["architecture"])
    gpu_class = str(target["gpu_class"])
    streams = int(target["streams"])
    plan_file_sha = sha256(plan_path)
    out = prefix / "results" / f"plan-build-{plan['plan_id']}-gpu{args.device}"
    out.mkdir(parents=True, exist_ok=True)
    build_manifest = out / "plan-build.json"
    if not args.force:
        existing = valid_existing_build(build_manifest, plan, plan_file_sha)
        if existing is not None:
            print(json.dumps({
                "reused": True,
                "plan": str(plan_path),
                "output": str(build_manifest),
                "binary": str((out / existing["binary"]).resolve()),
            }, indent=2, sort_keys=True))
            return 0

    space_path = out / "space.json"
    reuse_space = False
    if space_path.is_file() and not args.force:
        try:
            space = json.loads(space_path.read_text())
            validate_plan(plan, space=space, batches=[batch])
            policy = space.get("candidate_policy", {})
            reuse_space = (
                isinstance(policy, dict) and
                policy.get("build_only_plan_id") == plan.get("plan_id") and
                int(space.get("device_ordinal", -1)) == args.device
            )
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            reuse_space = False
    if not reuse_space:
        space = materialize_space(
            architecture, gpu_class, args.device, [batch], streams,
            device_properties=device_properties,
        )
        space = restrict_space_to_plan(space, plan, batch)
        write_json(space_path, space)

    command = [
        str(python), str(autotune),
        "--prefix", str(prefix), "--repo", str(repo),
        "--output-dir", str(out), "--device", str(args.device),
        "--batches", str(batch), "--streams", str(streams),
        "--phase", "prepare", "--full-batch-scan",
    ]
    if args.jobs is not None:
        if args.jobs < 1:
            raise RuntimeError("--jobs must be positive")
        command.extend(["--jobs", str(args.jobs)])
    if args.force:
        command.append("--force")
    print("[build-for-plan] + " + " ".join(command), flush=True)
    env = dict(os.environ)
    env["PYTHONPATH"] = (
        str(repo / "python") +
        (f":{env['PYTHONPATH']}" if env.get("PYTHONPATH") else "")
    )
    subprocess.run(command, check=True, env=env)

    binary = out / "build/katago"
    artifact_bundle = out / "artifact-bundle.json"
    for path, label in (
        (binary, "plan binary"),
        (artifact_bundle, "artifact bundle"),
        (space_path, "restricted search space"),
    ):
        if not path.is_file():
            raise RuntimeError(f"{label} is missing after build: {path}")
    validate_plan(plan, space=json.loads(space_path.read_text()), batches=[batch])
    manifest = {
        "schema": 1,
        "kind": "cuda-plan-build",
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "plan_id": plan["plan_id"],
        "plan_sha256": plan["plan_sha256"],
        "source_plan_file_sha256": plan_file_sha,
        "architecture": architecture,
        "gpu_class": gpu_class,
        "batch": batch,
        "streams": streams,
        "device_ordinal_at_build": args.device,
        "device": device_properties,
        "binary": str(binary.relative_to(out)),
        "binary_sha256": sha256(binary),
        "artifact_bundle": str(artifact_bundle.relative_to(out)),
        "artifact_bundle_sha256": sha256(artifact_bundle),
        "space": str(space_path.relative_to(out)),
        "space_sha256": sha256(space_path),
    }
    write_json(build_manifest, manifest)
    print(json.dumps({
        "reused": False,
        "plan": str(plan_path),
        "output": str(build_manifest),
        "binary": str(binary),
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError, subprocess.SubprocessError) as exc:
        print(f"build-for-plan: {exc}", file=sys.stderr)
        raise SystemExit(2)
