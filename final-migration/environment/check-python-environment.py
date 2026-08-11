#!/usr/bin/env python3

from __future__ import annotations

import importlib
import importlib.metadata
import subprocess
import sys

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name


BINARY_BASE = {
    "cuda-python",
    "einops",
    "numpy",
    "onnx",
    "onnx2torch",
    "packaging",
    "pillow",
    "psutil",
    "pyyaml",
    "torch",
    "torch-c-dlpack-ext",
    "triton",
}

SOURCE_DISTRIBUTIONS = {
    "apache-tvm-ffi",
    "flash-attn-4",
    "nvidia-cutlass-dsl",
    "quack-kernels",
    "tilelang",
}

SOURCE_IMPORTS = (
    "cutlass",
    "flash_attn.cute",
    "quack",
    "tilelang",
    "tvm_ffi",
)


ALLOWED_SOURCE_CONFLICTS = {
    ("flash-attn-4", "nvidia-cutlass-dsl"),
    ("quack-kernels", "nvidia-cutlass-dsl"),
}


def source_version_conflict(line: str) -> bool:
    if " has requirement " not in line or ", but you have " not in line:
        return False
    subject = canonicalize_name(line.split(" ", 1)[0])
    raw_requirement = line.split(" has requirement ", 1)[1].rsplit(", but you have ", 1)[0]
    try:
        required = canonicalize_name(Requirement(raw_requirement).name)
    except ValueError:
        return False
    return (subject, required) in ALLOWED_SOURCE_CONFLICTS


def main() -> int:
    errors: list[str] = []
    for distribution in sorted(BINARY_BASE):
        try:
            actual = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            errors.append(f"missing binary base {distribution}")
            continue
        print(f"BINARY_DISTRIBUTION {distribution}=={actual}")

    for distribution in sorted(SOURCE_DISTRIBUTIONS):
        try:
            version = importlib.metadata.version(distribution)
            print(f"SOURCE_DISTRIBUTION {distribution}=={version}")
        except importlib.metadata.PackageNotFoundError:
            errors.append(f"missing source-built distribution {distribution}")

    for module_name in SOURCE_IMPORTS:
        try:
            module = importlib.import_module(module_name)
            print(f"SOURCE_IMPORT {module_name} {getattr(module, '__file__', None)}")
        except Exception as exc:
            errors.append(f"source import {module_name}: {type(exc).__name__}: {exc}")

    result = subprocess.run(
        [sys.executable, "-m", "pip", "check"],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    for line in result.stdout.splitlines():
        line = line.strip()
        if not line or line == "No broken requirements found.":
            continue
        if not source_version_conflict(line):
            errors.append(f"unexpected pip conflict: {line}")

    if errors:
        for error in errors:
            print(f"FAIL {error}")
        return 1

    print("PYTHON_ENVIRONMENT_OK")
    if result.returncode != 0:
        print("Allowed metadata version conflicts are limited to source-built components")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
