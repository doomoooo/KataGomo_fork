#!/usr/bin/env python3

from __future__ import annotations

import csv
import importlib.metadata
import sys
from pathlib import Path

from packaging.utils import canonicalize_name


def main() -> int:
    manifest = Path(sys.argv[1])
    with manifest.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))

    # Only components for which this run produced a wheel are omitted from the
    # binary wheelhouse query. CUTLASS DSL is deliberately retained here: its
    # source repository does not publish the compiled MLIR payload needed to
    # build that wheel, so it is an explicit upstream-binary exception.
    locally_built = {
        canonicalize_name(row["distribution"])
        for row in rows
        if row["distribution"] != "-" and row["wheel"] != "-"
    }

    resolved: dict[str, tuple[str, str]] = {}
    for distribution in importlib.metadata.distributions():
        raw_name = distribution.metadata.get("Name")
        if not raw_name:
            continue
        canonical_name = canonicalize_name(raw_name)
        if canonical_name in locally_built:
            continue
        resolved[canonical_name] = (raw_name, distribution.version)

    for canonical_name in sorted(resolved):
        raw_name, version = resolved[canonical_name]
        print(f"{raw_name}=={version}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
