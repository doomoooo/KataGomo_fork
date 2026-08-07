#!/usr/bin/env python3
"""Bind pinned requirement lines to exact carried wheel hashes."""

from __future__ import annotations

import argparse
import email.parser
import hashlib
import pathlib
import re
import zipfile


def canonical(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def wheel_identity(path: pathlib.Path) -> tuple[str, str]:
    with zipfile.ZipFile(path) as archive:
        metadata_name = next(
            item for item in archive.namelist() if item.endswith(".dist-info/METADATA")
        )
        metadata = email.parser.BytesParser().parsebytes(archive.read(metadata_name))
    return canonical(metadata["Name"]), metadata["Version"]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("requirements", type=pathlib.Path)
    parser.add_argument("wheelhouse", type=pathlib.Path)
    parser.add_argument("output", type=pathlib.Path)
    args = parser.parse_args()
    wheels: dict[tuple[str, str], list[pathlib.Path]] = {}
    for path in sorted(args.wheelhouse.glob("*.whl")):
        wheels.setdefault(wheel_identity(path), []).append(path)
    result = []
    for raw in args.requirements.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "==" not in line:
            raise SystemExit(f"requirement is not exactly pinned: {line}")
        name, version = line.split("==", 1)
        matches = wheels.get((canonical(name), version), [])
        if len(matches) != 1:
            raise SystemExit(f"expected one wheel for {line}, found {matches}")
        digest = hashlib.sha256(matches[0].read_bytes()).hexdigest()
        result.append(f"{line} --hash=sha256:{digest}")
    args.output.write_text("\n".join(result) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
