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
import hashlib
import json
import pathlib


def utc_now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def resource_signature(metadata: dict, candidate: dict) -> dict:
    launch = metadata.get("launch", {})
    return {
        "tile": (
            [candidate.get(key) for key in ("m", "n", "k")]
            if any(key in candidate for key in ("m", "n", "k"))
            else metadata.get("tile")
        ),
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
        candidate = metadata.get("candidate", entry.get("candidate", {}))
        if candidate.get("id", metadata.get("candidate_id")) != entry["candidate_id"]:
            raise ValueError(f"candidate mismatch in {metadata_path}")
        measurement = metadata.get("local_measurement", metadata)
        if "s1_us_median" not in measurement:
            raise ValueError(f"missing local CUDA-event timing in {metadata_path}")
        correctness = measurement.get(
            "correctness_against_torch",
            measurement.get("correctness_against_cublas"),
        )
        if correctness is None:
            raise ValueError(f"missing correctness result in {metadata_path}")
        rows.append({
            "family": manifest["family"],
            "batch": entry["batch"],
            "candidate_id": entry["candidate_id"],
            "s1_us_median": measurement["s1_us_median"],
            "s1_us_samples": measurement.get("s1_us_samples", []),
            "correctness": correctness,
            "resource_signature": resource_signature(metadata, candidate),
            "metadata": str(metadata_path.resolve()),
            "source": entry["source"],
        })
    return rows


def sha256_file(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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


def select_complement(rows: list[dict], excluded_ids: set[str]) -> dict:
    """Retain every exact candidate not present in an earlier selection."""
    ranked = sorted(rows, key=lambda row: row["s1_us_median"])
    retained = [row for row in ranked if row["candidate_id"] not in excluded_ids]
    best = retained[0]["s1_us_median"] if retained else None
    retained_ids = {row["candidate_id"] for row in retained}
    return {
        "winner": retained[0]["candidate_id"] if retained else None,
        "winner_s1_us": best,
        "retained": [row["candidate_id"] for row in retained],
        "ranking": [
            {
                **row,
                "relative_to_best_percent": (
                    (row["s1_us_median"] / best - 1.0) * 100.0
                    if best is not None else None
                ),
                "retained": row["candidate_id"] in retained_ids,
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
    parser.add_argument(
        "--complement-of",
        default="",
        help="retain every candidate not retained by this earlier selection JSON",
    )
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
    excluded_by_group: dict[tuple[str, int], set[str]] = {}
    complement_path = None
    if args.complement_of:
        complement_path = pathlib.Path(args.complement_of).resolve()
        complement_payload = json.loads(complement_path.read_text())
        if complement_payload.get("schema") != 1:
            raise ValueError(f"unsupported selection JSON: {complement_path}")
        for group in complement_payload.get("groups", []):
            key = (group["family"], int(group["batch"]))
            if key in excluded_by_group:
                raise ValueError(f"duplicate excluded group: {key}")
            excluded_by_group[key] = set(group.get("retained", []))
    groups: dict[tuple[str, int], list[dict]] = {}
    for row in rows:
        groups.setdefault((row["family"], row["batch"]), []).append(row)
    output_groups = []
    for (family, batch), group in sorted(groups.items()):
        key = (family, batch)
        if complement_path is not None:
            if key not in excluded_by_group:
                raise ValueError(f"excluded selection has no group for {family}/B{batch}")
            selection = select_complement(group, excluded_by_group[key])
        else:
            selection = select_group(
                group, args.top_k, args.near_best_fraction,
                args.max_retained,
            )
        output_groups.append({
            "family": family,
            "batch": batch,
            **selection,
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
    if complement_path is not None:
        payload["selection_kind"] = "s1_complement"
        payload["complement_of"] = {
            "path": str(complement_path),
            "sha256": sha256_file(complement_path),
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
