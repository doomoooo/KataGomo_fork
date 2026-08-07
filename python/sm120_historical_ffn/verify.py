#!/usr/bin/env python3
"""Static verifier for historical B1..B32 tanh-half2 FFN artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re


CANDIDATE_ID = "ffn-m128-n64-k32-s2-mb3-tanh-half2"
SOURCE_NAME = f"ffn-{CANDIDATE_ID}.cu"
METADATA_NAME = f"ffn-{CANDIDATE_ID}.json"


def sha256_path(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_one(root: Path, batch: int) -> dict:
    directory = root / f"b{batch}-ffn"
    source_path = directory / SOURCE_NAME
    metadata_path = directory / METADATA_NAME
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    source = source_path.read_text(encoding="utf-8")
    expected_grid_y = (batch * 361 + 127) // 128
    kernel_name = f"katago_ffn_tilelang_sm120_b{batch}_s361_kernel"
    checks = {
        "batch": metadata["batch"] == batch,
        "tokens": metadata["tokens"] == batch * 361,
        "candidate": metadata["candidate"]["id"] == CANDIDATE_ID,
        "sourceHash": metadata["sourceSha256"] == sha256_path(source_path),
        "grid": metadata["launch"]["grid"] == [18, expected_grid_y, 1],
        "stableBatchSymbol": "sm120_search_ffn_batch" in source,
        "stableIdSymbol": "sm120_search_ffn_id" in source,
        "stableLaunchSymbol": "sm120_search_ffn_launch" in source,
        "uniqueKernel": source.count(kernel_name) == 3,
        "exactRows": (
            f"< {(batch * 361) // math.gcd(batch * 361, 128)})" in source
        ),
        "half2Tanh": "h2tanh_approx(input.value)" in source,
        "half2Fma": "tl::fma2" in source,
        "noExpf": "expf(" not in source,
        "sm120Guard": "__CUDA_ARCH__ >= 1200" in source,
        "launchGrid": f"dim3(18, {expected_grid_y}, 1)" in source,
        "kernelWeightOrder": bool(
            re.search(
                re.escape(kernel_name)
                + r"<<<.*?input\).*?gate_weights\).*?linear_weights\).*?output\)",
                source,
                re.DOTALL,
            )
        ),
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise AssertionError(f"B{batch} failed static checks: {failed}")
    return {
        "batch": batch,
        "sourceSha256": metadata["sourceSha256"],
        "deviceSourceSha256": metadata["deviceSource"]["sourceSha256"],
        "gridY": expected_grid_y,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, required=True)
    args = parser.parse_args()
    rows = [verify_one(args.artifact_root, batch) for batch in range(1, 33)]
    summary_path = args.artifact_root / "static-verification-b1-b32.json"
    summary = {
        "schema": 1,
        "candidateId": CANDIDATE_ID,
        "status": "passed",
        "gpuUsed": False,
        "batches": rows,
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(summary_path)


if __name__ == "__main__":
    main()
