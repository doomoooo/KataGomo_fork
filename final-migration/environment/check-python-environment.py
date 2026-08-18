#!/usr/bin/env python3

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import pathlib
import platform
import re
import subprocess
import sys
import sysconfig
from dataclasses import dataclass

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name


BINARY_BASE = {
    "apache-tvm-ffi",
    "cuda-python",
    "cuda-tile",
    "cuda-toolkit",
    "einops",
    "flashinfer-python",
    "liger-kernel",
    "mslk",
    "numpy",
    "nvidia-cudnn-frontend",
    "nvidia-cutlass-dsl",
    "nvidia-cutlass-dsl-libs-cu13",
    "nvidia-cuda-nvcc",
    "nvidia-cuda-nvdisasm",
    "nvidia-cudnn-cu13",
    "packaging",
    "psutil",
    "protobuf",
    "pyyaml",
    "quack-kernels",
    "tilelang",
    "torch",
    "torch-c-dlpack-ext",
    "triton",
}

SOURCE_DISTRIBUTIONS = {
    "flash-attn-4",
}

DEFAULT_REQUIREMENT_PATHS = (
    pathlib.Path(__file__).resolve().parents[1]
    / "autotune"
    / "python-build-requirements.txt",
    pathlib.Path(__file__).resolve().parents[1]
    / "autotune"
    / "python-binary-requirements.txt",
)

SOURCE_IMPORTS = (
    "cutlass",
    "flash_attn.cute",
    "quack",
    "tilelang",
    "tvm_ffi",
)

BINARY_IMPORTS = (
    "cudnn",
    "flashinfer",
    "liger_kernel",
    "mslk",
)


@dataclass(frozen=True)
class AllowedConflict:
    category: str
    required_specifier: str | None = None
    installed_version: str | None = None


ALLOWED_CONFLICTS = {
    ("quack-kernels", "nvidia-cutlass-dsl"): AllowedConflict(
        "published-generator", "==4.6.2", "4.7.0"
    ),
    # PyTorch does not publish a cu133 wheel.  The official cu132 wheel is the
    # latest PyTorch binary, while compilation and native inference use the
    # separately locked CUDA 13.3.1 / cuDNN 9.25 stack.
    ("torch", "cuda-toolkit"): AllowedConflict(
        "mixed-wheel-abi-vs-active-runtime", "==13.2.1", "13.3.1"
    ),
    ("torch", "nvidia-cudnn-cu13"): AllowedConflict(
        "mixed-wheel-abi-vs-active-runtime", "==9.20.0.48", "9.25.0.15"
    ),
}


def load_exact_requirements(
    requirement_paths: tuple[pathlib.Path, ...],
) -> tuple[dict[str, tuple[str, pathlib.Path]], list[str]]:
    expected: dict[str, tuple[str, pathlib.Path]] = {}
    errors: list[str] = []
    for path in requirement_paths:
        if not path.is_file():
            errors.append(f"requirements file is missing: {path}")
            continue
        for line_number, raw in enumerate(path.read_text().splitlines(), start=1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            try:
                requirement = Requirement(line)
            except ValueError as exc:
                errors.append(f"invalid requirement {path}:{line_number}: {exc}")
                continue
            specifiers = list(requirement.specifier)
            if (
                len(specifiers) != 1
                or specifiers[0].operator not in {"==", "==="}
                or "*" in specifiers[0].version
                or requirement.url is not None
                or requirement.marker is not None
            ):
                errors.append(
                    f"requirement is not one unconditional exact pin "
                    f"{path}:{line_number}: {line}"
                )
                continue
            name = canonicalize_name(requirement.name)
            version = specifiers[0].version
            previous = expected.get(name)
            if previous is not None and previous[0] != version:
                errors.append(
                    f"conflicting exact pins for {name}: {previous[0]} in "
                    f"{previous[1]} versus {version} in {path}"
                )
                continue
            expected[name] = (version, path)
    return expected, errors


def validate_exact_environment(
    requirement_paths: tuple[pathlib.Path, ...],
) -> list[str]:
    expected, errors = load_exact_requirements(requirement_paths)
    for name, (version, path) in sorted(expected.items()):
        try:
            actual = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            errors.append(f"missing exact pin {name}=={version} from {path}")
            continue
        if actual != version:
            errors.append(
                f"exact pin drift for {name}: expected {version} from {path}, "
                f"installed {actual}"
            )
        else:
            print(f"EXACT_PIN {name}=={actual}")
    return errors


def allowed_version_conflict(line: str) -> AllowedConflict | None:
    if " has requirement " not in line or ", but you have " not in line:
        return None
    subject = canonicalize_name(line.split(" ", 1)[0])
    raw_requirement = line.split(" has requirement ", 1)[1].rsplit(", but you have ", 1)[0]
    installed = line.rsplit(", but you have ", 1)[1].removesuffix(".")
    try:
        installed_name, installed_version = installed.rsplit(" ", 1)
        requirement = Requirement(raw_requirement)
        required = canonicalize_name(requirement.name)
    except ValueError:
        return None
    if canonicalize_name(installed_name) != required:
        return None
    allowed = ALLOWED_CONFLICTS.get((subject, required))
    if allowed is None:
        return None
    if (
        allowed.required_specifier is not None
        and str(requirement.specifier) != allowed.required_specifier
    ):
        return None
    if (
        allowed.installed_version is not None
        and installed_version != allowed.installed_version
    ):
        return None
    return allowed


def main(requirement_paths: tuple[pathlib.Path, ...]) -> int:
    errors = validate_exact_environment(requirement_paths)
    if platform.python_version() != "3.14.7":
        errors.append(
            f"Python runtime drift: expected 3.14.7, found {platform.python_version()}"
        )

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

    for module_name in BINARY_IMPORTS:
        try:
            module = importlib.import_module(module_name)
            print(f"BINARY_IMPORT {module_name} {getattr(module, '__file__', None)}")
        except Exception as exc:
            errors.append(f"binary import {module_name}: {type(exc).__name__}: {exc}")

    for module_name in SOURCE_IMPORTS:
        try:
            module = importlib.import_module(module_name)
            print(f"SOURCE_IMPORT {module_name} {getattr(module, '__file__', None)}")
        except Exception as exc:
            errors.append(f"source import {module_name}: {type(exc).__name__}: {exc}")

    try:
        torch = importlib.import_module("torch")
        if torch.version.cuda != "13.2":
            errors.append(
                f"PyTorch wheel CUDA drift: expected 13.2, found {torch.version.cuda}"
            )
        cudnn_version = torch.backends.cudnn.version()
        if cudnn_version != 92500:
            errors.append(
                f"active cuDNN drift through PyTorch: expected 92500, found {cudnn_version}"
            )
    except Exception as exc:
        errors.append(f"PyTorch ABI/runtime identity: {type(exc).__name__}: {exc}")

    nvcc = pathlib.Path(sysconfig.get_path("purelib")) / "nvidia/cu13/bin/nvcc"
    if not nvcc.is_file():
        errors.append(f"managed nvcc is missing: {nvcc}")
    else:
        nvcc_result = subprocess.run(
            [str(nvcc), "--version"],
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        match = re.search(r"release ([0-9.]+), V([0-9.]+)", nvcc_result.stdout)
        if nvcc_result.returncode != 0 or match is None:
            errors.append("could not parse managed nvcc --version")
        elif match.groups() != ("13.3", "13.3.73"):
            errors.append(
                "managed nvcc drift: expected release 13.3 / V13.3.73, "
                f"found release {match.group(1)} / V{match.group(2)}"
            )
        else:
            print("NATIVE_NVCC release=13.3 version=13.3.73")

    result = subprocess.run(
        [sys.executable, "-m", "pip", "check"],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    allowed_conflicts: list[tuple[AllowedConflict, str]] = []
    for line in result.stdout.splitlines():
        line = line.strip()
        if not line or line == "No broken requirements found.":
            continue
        allowed = allowed_version_conflict(line)
        if allowed is None:
            errors.append(f"unexpected pip conflict: {line}")
        else:
            allowed_conflicts.append((allowed, line))
            print(f"ALLOWED_METADATA_CONFLICT [{allowed.category}] {line}")

    if errors:
        for error in errors:
            print(f"FAIL {error}")
        return 1

    print("PYTHON_ENVIRONMENT_OK")
    if allowed_conflicts:
        print(
            "Allowed metadata conflicts are recorded above; the PyTorch wheel ABI "
            "versus active CUDA/cuDNN runtime waivers matched exact versions"
        )
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--requirements",
        action="append",
        type=pathlib.Path,
        help="exact requirement file to verify; repeat for multiple files",
    )
    arguments = parser.parse_args()
    paths = tuple(arguments.requirements or DEFAULT_REQUIREMENT_PATHS)
    raise SystemExit(main(paths))
