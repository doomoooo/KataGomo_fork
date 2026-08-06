#!/usr/bin/env python3
"""Compress exact-batch local timing results without GPU counters.

The input is one or more manifests emitted by
``sm120_prepare_tilelang_fat_scan.py``. Selection uses only CUDA-event latency,
correctness status, launch geometry, candidate parameters, and known dynamic
shared memory. NCU output is deliberately neither required nor parsed.
"""

from __future__ import annotations

import argparse
import datetime
import json
import pathlib


def utc_now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def resource_signature(metadata: dict) -> dict:
    candidate = metadata["candidate"]
    launch = metadata.get("launch", {})
    return {
        "tile": [candidate.get(key) for key in ("m", "n", "k")],
        "stages": candidate.get("stages"),
        "threads": candidate.get("threads", 128),
        "min_blocks": candidate.get("min_blocks"),
        "dynamic_smem_bytes": metadata.get("dynamic_smem_bytes"),
        "grid": launch.get("grid"),
        "block": launch.get("block"),
        "cta_count": launch.get("cta_count"),
    }


def signature_key(signature: dict) -> str:
    return json.dumps(signature, sort_keys=True, separators=(",", ":"))


def load_manifest(path: pathlib.Path) -> list[dict]:
    manifest = json.loads(path.read_text())
    if manifest.get("schema") != 1:
        raise ValueError(f"unsupported manifest schema in {path}")
    rows = []
    for entry in manifest["entries"]:
        metadata_path = pathlib.Path(entry["metadata"])
        metadata = json.loads(metadata_path.read_text())
        if metadata.get("batch") != entry["batch"]:
            raise ValueError(f"batch mismatch in {metadata_path}")
        if metadata.get("candidate", {}).get("id") != entry["candidate_id"]:
            raise ValueError(f"candidate mismatch in {metadata_path}")
        if "s1_us_median" not in metadata:
            raise ValueError(f"missing local CUDA-event timing in {metadata_path}")
        if "correctness_against_torch" not in metadata:
            raise ValueError(f"missing correctness result in {metadata_path}")
        rows.append({
            "family": manifest["family"],
            "batch": entry["batch"],
            "candidate_id": entry["candidate_id"],
            "s1_us_median": metadata["s1_us_median"],
            "s1_us_samples": metadata.get("s1_us_samples", []),
            "correctness": metadata["correctness_against_torch"],
            "resource_signature": resource_signature(metadata),
            "metadata": str(metadata_path.resolve()),
            "source": entry["source"],
        })
    return rows


def select_group(
    rows: list[dict], top_k: int, near_best_fraction: float,
    max_retained: int,
) -> dict:
    ranked = sorted(rows, key=lambda row: row["s1_us_median"])
    best = ranked[0]["s1_us_median"]
    selected = ranked[:top_k]
    selected_ids = {row["candidate_id"] for row in selected}
    signatures = {signature_key(row["resource_signature"]) for row in selected}
    for row in ranked[top_k:]:
        if len(selected) >= max_retained:
            break
        if row["s1_us_median"] > best * (1.0 + near_best_fraction):
            break
        key = signature_key(row["resource_signature"])
        if key not in signatures:
            selected.append(row)
            selected_ids.add(row["candidate_id"])
            signatures.add(key)
    return {
        "winner": ranked[0]["candidate_id"],
        "winner_s1_us": best,
        "retained": [row["candidate_id"] for row in selected],
        "ranking": [
            {
                **row,
                "relative_to_best_percent":
                    (row["s1_us_median"] / best - 1.0) * 100.0,
                "retained": row["candidate_id"] in selected_ids,
            }
            for row in ranked
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifests", nargs="+")
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--near-best-fraction", type=float, default=0.05)
    parser.add_argument("--max-retained", type=int, default=4)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.top_k < 1 or args.max_retained < args.top_k:
        parser.error("require 1 <= top-k <= max-retained")
    if args.near_best_fraction < 0:
        parser.error("--near-best-fraction must be nonnegative")

    rows = []
    paths = [pathlib.Path(value).resolve() for value in args.manifests]
    for path in paths:
        rows.extend(load_manifest(path))
    groups: dict[tuple[str, int], list[dict]] = {}
    for row in rows:
        groups.setdefault((row["family"], row["batch"]), []).append(row)
    output_groups = []
    for (family, batch), group in sorted(groups.items()):
        output_groups.append({
            "family": family,
            "batch": batch,
            **select_group(
                group, args.top_k, args.near_best_fraction,
                args.max_retained,
            ),
        })

    payload = {
        "schema": 1,
        "generated_utc": utc_now(),
        "fixed_board": [19, 19],
        "selection_metric": "single-stream CUDA-event kernel latency",
        "counter_policy": {
            "gpu_performance_counters_required": False,
            "ncu_output_parsed": False,
            "ncu_role": "optional manual explanation only",
        },
        "top_k": args.top_k,
        "near_best_fraction": args.near_best_fraction,
        "max_retained": args.max_retained,
        "source_manifests": [str(path) for path in paths],
        "groups": output_groups,
    }
    output = pathlib.Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(output)
    print(json.dumps({
        "output": str(output),
        "groups": len(output_groups),
        "retained": sum(len(group["retained"]) for group in output_groups),
    }))


if __name__ == "__main__":
    main()
