#!/usr/bin/env python3

from __future__ import annotations

import csv
import importlib.metadata
import sys
from pathlib import Path

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name


def main() -> int:
    manifest = Path(sys.argv[1])
    with manifest.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))

    source_distributions = {
        canonicalize_name(row["distribution"])
        for row in rows
        if row["distribution"] != "-"
    }
    emitted: set[str] = set()
    for row in rows:
        distribution = row["distribution"]
        if distribution == "-":
            continue
        for raw_requirement in importlib.metadata.requires(distribution) or []:
            requirement = Requirement(raw_requirement)
            if requirement.marker is not None and not requirement.marker.evaluate():
                continue
            if canonicalize_name(requirement.name) in source_distributions:
                continue
            try:
                installed_version = importlib.metadata.version(requirement.name)
            except importlib.metadata.PackageNotFoundError:
                installed_version = None
            if installed_version is not None and requirement.specifier.contains(
                installed_version, prereleases=True
            ):
                continue
            rendered = str(requirement)
            if rendered not in emitted:
                print(rendered)
                emitted.add(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
