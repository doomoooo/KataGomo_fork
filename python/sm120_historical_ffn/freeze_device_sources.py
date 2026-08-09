#!/usr/bin/env python3
"""Maintainer tool: freeze B1-B32 device sources from the historical compiler."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from generate import (
    CANDIDATE_ID,
    generate_device_source,
    load_historical_kernels,
    load_manifest,
    sha256_bytes,
    verify_dependencies,
    write_if_changed,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    historical_manifest = load_manifest()
    dependencies = verify_dependencies(historical_manifest)
    # The installed-tree hash is useful provenance; the maintainer's absolute
    # venv path is neither portable nor part of the frozen source identity.
    dependencies.pop("tilelangPackageRoot", None)
    kernels = load_historical_kernels()
    entries = []
    for batch in range(1, 33):
        source, kernel_name, evidence = generate_device_source(
            kernels, batch, historical_manifest
        )
        path = args.output_dir / f"b{batch}.cu"
        write_if_changed(path, source)
        entries.append({
            "batch": batch,
            "kernelName": kernel_name,
            "path": path.name,
            "sha256": sha256_bytes(source.encode("utf-8")),
            "bytes": len(source.encode("utf-8")),
            "deviceEvidence": evidence,
        })
        print(f"froze B{batch}: {entries[-1]['sha256']}", flush=True)

    manifest = {
        "schema": 1,
        "candidateId": CANDIDATE_ID,
        "batches": entries,
        "generatorDependencies": dependencies,
        "note": (
            "Immutable codegen output; runtime generation verifies and wraps "
            "these sources without invoking the historical compiler."
        ),
    }
    write_if_changed(
        args.output_dir / "manifest.json",
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
    )


if __name__ == "__main__":
    main()
