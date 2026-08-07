#!/usr/bin/env python3
"""Build and validate portable SM120 tactic plans.

A tactic plan is the small, distributable output of a completed whole-graph
scan.  It contains one measured tactic per exact batch and family, together
with the hashes needed to reject a plan produced for a different model,
search space, GPU class, or stream topology.  It is intentionally separate
from the scan result files: result files are evidence, while a plan is an
execution input.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import pathlib
from collections import defaultdict
from collections.abc import Iterable


FAMILIES = ("ffn", "qkv", "linear2", "fa4", "l2")
TACTIC_CONFIG_KEYS = {
    "ffn": "cudaFusedFFNAotTacticSm120",
    "qkv": "cudaWideQKVAotTacticSm120",
    "linear2": "cudaLinear2AotTacticSm120",
}
COMPUTE_CAPABILITY = {
    "rtx5080": [12, 0],
    "rtx5090d": [12, 0],
}


def utc_now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def parse_int_set(value: str) -> list[int]:
    result: set[int] = set()
    for token in value.split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            first, last = (int(item) for item in token.split("-", 1))
            if last < first:
                raise ValueError(f"invalid descending batch range: {token}")
            result.update(range(first, last + 1))
        else:
            result.add(int(token))
    values = sorted(result)
    if not values or values[0] < 1:
        raise ValueError("batch set must contain positive integers")
    return values


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def load_plan(path: pathlib.Path) -> dict:
    payload = json.loads(path.read_text())
    if payload.get("schema") != 1 or payload.get("kind") != "sm120-tactic-plan":
        raise ValueError(f"unsupported tactic plan: {path}")
    if not isinstance(payload.get("families"), dict):
        raise ValueError(f"tactic plan has no family map: {path}")
    return payload


def _space_candidates(space: dict, family: str, batch: int) -> dict[str, dict]:
    batch_space = next(
        (item for item in space.get("batches", []) if item.get("batch") == batch),
        None,
    )
    if batch_space is None:
        raise ValueError(f"search space has no batch B{batch}")
    return {item["id"]: item for item in batch_space.get(family, [])}


def plan_entries(plan: dict, family: str, batches: Iterable[int]) -> dict[int, dict]:
    if family not in FAMILIES:
        raise ValueError(f"unsupported tactic family: {family}")
    family_payload = plan.get("families", {}).get(family)
    if not isinstance(family_payload, dict):
        raise ValueError(f"tactic plan has no {family} family")
    entries = family_payload.get("batches", {})
    result: dict[int, dict] = {}
    for batch in sorted(set(batches)):
        entry = entries.get(str(batch))
        if not isinstance(entry, dict) or not entry.get("candidate_id"):
            raise ValueError(f"tactic plan has no {family}/B{batch} entry")
        result[batch] = entry
    return result


def validate_plan(
    plan: dict,
    space: dict,
    model_path: pathlib.Path | None,
    family: str,
    batches: Iterable[int],
    streams: int,
    config_path: pathlib.Path | None = None,
) -> dict[int, dict]:
    target = plan.get("target", {})
    if target.get("gpu_class") != space.get("gpu_class"):
        raise ValueError(
            "tactic plan GPU class does not match search space: "
            f"{target.get('gpu_class')} != {space.get('gpu_class')}"
        )
    expected_compute_capability = COMPUTE_CAPABILITY.get(space.get("gpu_class"))
    if target.get("compute_capability") != expected_compute_capability:
        raise ValueError("tactic plan compute capability does not match the search space")
    if int(target.get("streams", -1)) != streams:
        raise ValueError("tactic plan stream count does not match the requested run")
    if target.get("fixed_board") != [19, 19]:
        raise ValueError("tactic plans are currently fixed to 19x19")
    if not plan.get("ready_for_scan_bypass", False):
        raise ValueError(
            "tactic plan is partial; finish the scan before using it to bypass search"
        )
    expected_model_sha = target.get("model_sha256")
    if expected_model_sha and model_path is not None:
        actual_model_sha = sha256_file(model_path.resolve())
        if actual_model_sha != expected_model_sha:
            raise ValueError("tactic plan model SHA-256 does not match the supplied model")
    expected_config_sha = target.get("config_sha256")
    if expected_config_sha and config_path is not None:
        actual_config_sha = sha256_file(config_path.resolve())
        if actual_config_sha != expected_config_sha:
            raise ValueError(
                "tactic plan config SHA-256 does not match the supplied config"
            )

    space_sha = sha256_file(pathlib.Path(space["_path"]).resolve()) if "_path" in space else None
    family_payload = plan.get("families", {}).get(family, {})
    expected_space_sha = family_payload.get("space_sha256")
    if expected_space_sha and space_sha and expected_space_sha != space_sha:
        raise ValueError(
            f"tactic plan search-space SHA-256 does not match for {family}"
        )

    selected = plan_entries(plan, family, batches)
    for batch, entry in selected.items():
        candidates = _space_candidates(space, family, batch)
        candidate_id = entry["candidate_id"]
        current = candidates.get(candidate_id)
        if current is None:
            raise ValueError(f"plan tactic {family}/B{batch}/{candidate_id} is absent from space")
        planned_candidate = entry.get("candidate")
        if planned_candidate is not None and planned_candidate != current:
            raise ValueError(
                f"plan candidate parameters do not match the current space for "
                f"{family}/B{batch}/{candidate_id}"
            )
    return selected


def candidate_override(family: str, candidate: dict) -> list[str]:
    implementation = candidate.get("implementation", "tilelang")
    if family in TACTIC_CONFIG_KEYS:
        value = candidate["id"] if implementation != "fallback" else "disabled"
        return [f"{TACTIC_CONFIG_KEYS[family]}={value}"]
    if family == "fa4":
        if implementation == "fallback":
            return ["cudaUseFlashAttentionSm120=false"]
        return [
            "cudaUseFlashAttentionSm120=true",
            "cudaFlashAttentionSm120Accum=both16",
            f"cudaFlashAttentionAotTacticSm120={candidate['id']}",
        ]
    if family == "l2":
        values = []
        for key, value in candidate.get("config", {}).items():
            if isinstance(value, bool):
                value = str(value).lower()
            values.append(f"{key}={value}")
        return values
    raise ValueError(f"unsupported tactic family: {family}")


def plan_override_config(plan: dict, batch: int) -> str:
    """Return only tactic-related overrides for one exact batch."""
    values: list[str] = []
    for family in FAMILIES:
        entries = plan.get("families", {}).get(family, {}).get("batches", {})
        entry = entries.get(str(batch))
        if entry is not None:
            values.extend(candidate_override(family, entry["candidate"]))
    return ",".join(values)


def _result_metadata(path: pathlib.Path, payload: dict) -> dict:
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "family": payload.get("family"),
        "gpu_class": payload.get("gpu_class"),
        "streams": payload.get("streams"),
        "rows": len(payload.get("rows", [])),
        "finished_utc": payload.get("finished_utc"),
    }


def _latest_generator_metadata(row: dict) -> dict:
    """Prefer the on-disk metadata, which may contain post-scan replay data."""
    metadata = row.get("generator_metadata")
    if not isinstance(metadata, dict):
        metadata = {}
    metadata_path = row.get("generator_metadata_path")
    if metadata_path:
        path = pathlib.Path(metadata_path)
        if path.is_file():
            try:
                loaded = json.loads(path.read_text())
            except (OSError, json.JSONDecodeError):
                loaded = None
            if isinstance(loaded, dict):
                return loaded
    # Older CuTe scan rows did not record generator_metadata_path.  Their
    # generate command still contains --output-dir, so recover the metadata
    # file after a correctness replay has enriched it in place.
    command = row.get("commands", {}).get("generate")
    if isinstance(command, list) and "--output-dir" in command:
        output_index = command.index("--output-dir") + 1
        if output_index < len(command):
            output_dir = pathlib.Path(command[output_index])
            candidates = (
                output_dir / "sm120_qkv_cute_active.json",
                output_dir / "active-fa4.json",
                output_dir / (
                    f"ffn-{row.get('candidate_id', '')}.json"
                ),
            )
            for path in candidates:
                if path.is_file():
                    try:
                        loaded = json.loads(path.read_text())
                    except (OSError, json.JSONDecodeError):
                        loaded = None
                    if isinstance(loaded, dict):
                        return loaded
    return metadata


def build_plan(
    result_paths: list[pathlib.Path],
    space_path: pathlib.Path,
    families: list[str],
    batches: list[int],
    allow_partial: bool = False,
) -> dict:
    space = json.loads(space_path.read_text())
    if space.get("schema") != 2:
        raise ValueError("tactic plans require a schema-2 search space")
    gpu_class = space.get("gpu_class")
    if gpu_class not in COMPUTE_CAPABILITY:
        raise ValueError(f"unknown SM120 GPU class in search space: {gpu_class}")

    rows_by_key: dict[tuple[str, int, str], dict] = {}
    result_metadata: list[dict] = []
    environment_snapshots: list[dict] = []
    build_command_snapshots: list[dict] = []
    environment_ids: set[str] = set()
    build_command_ids: set[str] = set()
    target_streams: set[int] = set()
    model_shas: set[str] = set()
    config_shas: set[str] = set()
    missing_model_hash_results: list[str] = []
    missing_config_hash_results: list[str] = []
    for path in result_paths:
        payload = json.loads(path.read_text())
        if payload.get("schema") != 1 or "rows" not in payload:
            raise ValueError(f"unsupported tactic result file: {path}")
        if payload.get("gpu_class") != gpu_class:
            raise ValueError(f"result GPU class does not match the search space: {path}")
        family = payload.get("family")
        if family not in families:
            continue
        streams = int(payload.get("streams", payload.get("regime", {}).get("streams", -1)))
        if streams < 1:
            raise ValueError(f"result has no valid stream count: {path}")
        target_streams.add(streams)
        regime = payload.get("regime", {})
        if regime.get("model_sha256"):
            model_shas.add(regime["model_sha256"])
        else:
            missing_model_hash_results.append(str(path))
        if regime.get("config_sha256"):
            config_shas.add(regime["config_sha256"])
        else:
            missing_config_hash_results.append(str(path))
        result_metadata.append(_result_metadata(path, payload))
        snapshots = payload.get("environment_snapshots", [])
        if not snapshots and payload.get("environment"):
            snapshots = [payload["environment"]]
        for snapshot in snapshots:
            snapshot_id = hashlib.sha256(
                canonical_json({
                    key: value for key, value in snapshot.items()
                    if key != "captured_utc"
                }).encode("utf-8")
            ).hexdigest()
            if snapshot_id not in environment_ids:
                environment_ids.add(snapshot_id)
                environment_snapshots.append(snapshot)
        commands = payload.get("build_commands")
        if commands:
            command_id = hashlib.sha256(
                canonical_json(commands).encode("utf-8")
            ).hexdigest()
            if command_id not in build_command_ids:
                build_command_ids.add(command_id)
                build_command_snapshots.append(commands)
        for row in payload["rows"]:
            batch = int(row["batch"])
            candidate_id = row["candidate_id"]
            key = (family, batch, candidate_id)
            previous = rows_by_key.get(key)
            if previous is None or row.get("finished_utc", "") >= previous.get("finished_utc", ""):
                item = dict(row)
                item["_source_result"] = str(path.resolve())
                item["_streams"] = streams
                rows_by_key[key] = item
    if len(target_streams) != 1:
        raise ValueError(f"result files contain mixed stream counts: {sorted(target_streams)}")
    streams = next(iter(target_streams))
    requested_families = list(dict.fromkeys(families))
    selected_families: dict[str, dict] = {}
    missing: list[dict] = []
    coverage: dict[str, dict] = {}

    for family in requested_families:
        expected_by_batch = {
            batch: _space_candidates(space, family, batch) for batch in batches
        }
        family_batches: dict[str, dict] = {}
        family_coverage: dict[str, dict] = {}
        for batch in batches:
            expected = expected_by_batch[batch]
            observed = {
                candidate_id: rows_by_key[(family, batch, candidate_id)]
                for candidate_id in expected
                if (family, batch, candidate_id) in rows_by_key
            }
            measured = [
                row for row in observed.values()
                if row.get("status") == "measured"
                and isinstance(row.get("nn_evals_per_sec_median"), (int, float))
            ]
            measured.sort(key=lambda row: row["nn_evals_per_sec_median"], reverse=True)
            missing_ids = sorted(set(expected) - set(observed))
            not_measured = sorted(
                candidate_id for candidate_id, row in observed.items()
                if row.get("status") != "measured"
            )
            family_coverage[str(batch)] = {
                "expected_count": len(expected),
                "observed_count": len(observed),
                "measured_count": len(measured),
                "missing_candidate_ids": missing_ids,
                "not_measured_candidate_ids": not_measured,
            }
            if missing_ids or not_measured or not measured:
                missing.append({
                    "family": family,
                    "batch": batch,
                    "missing_candidate_ids": missing_ids,
                    "not_measured_candidate_ids": not_measured,
                })
            if not measured:
                continue
            winner = measured[0]
            expected_candidate = expected[winner["candidate_id"]]
            if winner.get("candidate", expected_candidate) != expected_candidate:
                raise ValueError(
                    f"result candidate parameters do not match the search space for "
                    f"{family}/B{batch}/{winner['candidate_id']}"
                )
            metadata = _latest_generator_metadata(winner)
            fat_entry = winner.get("fat_scan_entry", {})
            artifact_sha256 = (
                metadata.get("source_sha256")
                or fat_entry.get("source_sha256")
                or metadata.get("sha256")
            )
            family_batches[str(batch)] = {
                "candidate_id": winner["candidate_id"],
                "candidate": expected_candidate,
                "implementation": winner.get(
                    "implementation", expected_candidate.get("implementation", "tilelang")
                ),
                "nn_evals_per_sec_median": winner["nn_evals_per_sec_median"],
                "nn_evals_per_sec_samples": winner.get("nn_evals_per_sec_samples", []),
                "binary_sha256": winner.get("binary_sha256"),
                "artifact_sha256": artifact_sha256,
                # Keep the generator/fat-bundle evidence in the portable plan
                # itself.  The source-result path is useful to the producer,
                # but it is not sufficient after the plan is copied away.
                "generator_parameters": winner.get("generator_parameters"),
                "generator_metadata": metadata or None,
                "fat_scan_entry": (
                    {
                        key: fat_entry.get(key)
                        for key in (
                            "batch", "candidate_id", "symbol_token",
                            "launch_symbol", "source", "source_sha256",
                            "metadata", "metadata_sha256", "registry_source",
                        )
                        if key in fat_entry
                    }
                    if fat_entry else None
                ),
                "source_result": pathlib.Path(winner["_source_result"]).name,
                "source_result_path_at_scan": winner["_source_result"],
                "override_config": winner.get("override_config"),
            }
        selected_families[family] = {
            # Keep the plan usable after copying it to another machine.  The
            # hash is authoritative; this path is evidence from the producer.
            "space": space_path.name,
            "space_path_at_scan": str(space_path.resolve()),
            "space_sha256": sha256_file(space_path),
            "batches": family_batches,
        }
        coverage[family] = family_coverage

    if not result_metadata:
        raise ValueError("no requested-family result files were supplied")
    if len(model_shas) > 1 or len(config_shas) > 1:
        raise ValueError("result files contain mixed model/config hashes")
    identity_missing = []
    if not model_shas:
        identity_missing.append("model_sha256")
    elif missing_model_hash_results:
        identity_missing.append("model_sha256 in " + ", ".join(
            pathlib.Path(path).name for path in missing_model_hash_results
        ))
    if not config_shas:
        identity_missing.append("config_sha256")
    elif missing_config_hash_results:
        identity_missing.append("config_sha256 in " + ", ".join(
            pathlib.Path(path).name for path in missing_config_hash_results
        ))
    ready = not missing and not identity_missing
    if not ready and not allow_partial:
        missing_preview = ", ".join(
            f"{item['family']}/B{item['batch']}" for item in missing[:8]
        )
        if identity_missing:
            missing_preview = ", ".join(
                [*identity_missing, missing_preview] if missing_preview else identity_missing
            )
        raise ValueError(
            "scan coverage is incomplete; use --allow-partial only to export a "
            f"non-deployable plan (first gaps: {missing_preview})"
        )

    target = {
        "gpu_class": gpu_class,
        "compute_capability": COMPUTE_CAPABILITY[gpu_class],
        "fixed_board": [19, 19],
        "precision": "FP16/NHWC",
        "streams": streams,
        "model_sha256": next(iter(model_shas), None),
        "config_sha256": next(iter(config_shas), None),
    }
    identity = {
        "target": target,
        "batches": batches,
        "families": {
            family: {
                "space_sha256": selected_families[family]["space_sha256"],
                "selected": {
                    batch: selected_families[family]["batches"].get(str(batch), {}).get("candidate_id")
                    for batch in batches
                },
            }
            for family in requested_families
        },
    }
    plan_hash = hashlib.sha256(canonical_json(identity).encode("utf-8")).hexdigest()
    return {
        "schema": 1,
        "kind": "sm120-tactic-plan",
        "plan_id": f"sm120-{plan_hash[:16]}",
        "plan_sha256": plan_hash,
        "generated_utc": utc_now(),
        "status": "complete_short_scan" if ready else "partial_short_scan",
        "ready_for_scan_bypass": ready,
        "selection": {
            "metric": "natural whole-graph S2 median nnEval/s",
            "method": "per-family, per-batch maximum among measured candidates",
            "is_acceptance": False,
            "required_follow_up": "correctness replay plus long ABBA/BAAB before production acceptance",
        },
        "target": target,
        "batches": batches,
        "families": selected_families,
        "coverage": coverage,
        "missing": missing,
        "identity_missing": identity_missing,
        "source_results": result_metadata,
        "reproducibility": {
            "environment_snapshots": environment_snapshots,
            "build_commands": build_command_snapshots,
            "notes": [
                "Environment snapshots are evidence, not a strict compatibility gate.",
                "Re-run correctness and long ABBA/BAAB on the receiver before production use.",
            ],
        },
        "apply": {
            "per_batch_tactic_overrides": {
                str(batch): plan_override_config(
                    {"families": selected_families}, batch
                )
                for batch in batches
            },
            "prefix": f"numNNServerThreadsPerModel={streams},cudaPersistingL2StreamsSm120={streams}",
        },
    }


def write_json(path: pathlib.Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def build_command(args: argparse.Namespace) -> None:
    families = [item.strip() for item in args.families.split(",") if item.strip()]
    unknown = sorted(set(families) - set(FAMILIES))
    if not families or unknown:
        raise ValueError(f"invalid tactic families: {unknown or families}")
    payload = build_plan(
        [pathlib.Path(value).resolve() for value in args.results],
        pathlib.Path(args.space).resolve(), families,
        parse_int_set(args.batches), args.allow_partial,
    )
    write_json(pathlib.Path(args.output).resolve(), payload)
    print(json.dumps({
        "output": str(pathlib.Path(args.output).resolve()),
        "plan_id": payload["plan_id"],
        "ready_for_scan_bypass": payload["ready_for_scan_bypass"],
        "missing": len(payload["missing"]),
    }))


def validate_command(args: argparse.Namespace) -> None:
    plan = load_plan(pathlib.Path(args.plan).resolve())
    space_path = pathlib.Path(args.space).resolve()
    space = json.loads(space_path.read_text())
    space["_path"] = str(space_path)
    selected = validate_plan(
        plan, space,
        pathlib.Path(args.model).resolve() if args.model else None,
        args.family, parse_int_set(args.batches), args.streams,
        pathlib.Path(args.config).resolve() if args.config else None,
    )
    print(json.dumps({
        "valid": True,
        "plan_id": plan["plan_id"],
        "family": args.family,
        "batches": sorted(selected),
    }))


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build", help="build a plan from whole-graph scan results")
    build.add_argument("results", nargs="+")
    build.add_argument("--space", required=True)
    build.add_argument("--families", default=",".join(FAMILIES))
    build.add_argument("--batches", required=True)
    build.add_argument("--output", required=True)
    build.add_argument("--allow-partial", action="store_true")
    build.set_defaults(function=build_command)
    validate = subparsers.add_parser("validate", help="validate a plan for one run")
    validate.add_argument("--plan", required=True)
    validate.add_argument("--space", required=True)
    validate.add_argument("--model")
    validate.add_argument("--config")
    validate.add_argument("--family", choices=FAMILIES, required=True)
    validate.add_argument("--batches", required=True)
    validate.add_argument("--streams", type=int, required=True)
    validate.set_defaults(function=validate_command)
    return parser


def main() -> None:
    args = make_parser().parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
