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
import copy
import datetime
import hashlib
import json
import pathlib
from collections import defaultdict
from collections.abc import Iterable


FAMILIES = ("ffn", "qkv", "linear2", "fa4", "l2")
MIN_LONG_ITERATIONS = 1000
MIN_STABLE_SAMPLES = 2
MAX_STABLE_RELATIVE_SPREAD = 0.10
TACTIC_CONFIG_KEYS = {
    "ffn": "cudaFusedFFNAotTacticSm120",
    "qkv": "cudaWideQKVAotTacticSm120",
    "linear2": "cudaLinear2AotTacticSm120",
}
SUPPORTED_GPU_CLASSES = {"rtx5080", "rtx5090d"}
TACTIC_HARDWARE_ATTRIBUTES = (
    "multiProcessorCount",
    "sharedMemoryPerMultiprocessor",
    "regsPerMultiprocessor",
    "maxThreadsPerMultiprocessor",
    "maxBlocksPerMultiprocessor",
    "maxSharedMemoryPerBlockOptin",
    "l2CacheSize",
    "persistingL2CacheMaxSize",
    "memoryBusWidth",
)


def utc_now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def cuda_tactic_hardware_signature(properties: dict) -> dict:
    """Return CUDA-reported resources that can affect tactic scheduling."""
    attributes = properties.get("attributes", {})
    result = {
        "compute_capability": properties.get("compute_capability"),
        "multiprocessor_count": properties.get(
            "multiprocessor_count", attributes.get("multiProcessorCount")
        ),
    }
    for key in TACTIC_HARDWARE_ATTRIBUTES:
        if isinstance(attributes, dict) and key in attributes:
            result[key] = attributes[key]
    return {
        key: value for key, value in result.items()
        if value is not None
    }


def require_compatible_cuda_hardware(
    expected_devices: Iterable[dict], actual: dict,
) -> None:
    signatures = {
        canonical_json(cuda_tactic_hardware_signature(value))
        for value in expected_devices
        if isinstance(value, dict)
    }
    if not signatures:
        return
    if len(signatures) != 1:
        raise ValueError("tactic plan contains mixed CUDA hardware signatures")
    expected = json.loads(next(iter(signatures)))
    actual_signature = cuda_tactic_hardware_signature(actual)
    mismatched = {
        key: {"expected": value, "actual": actual_signature.get(key)}
        for key, value in expected.items()
        if actual_signature.get(key) != value
    }
    if mismatched:
        raise ValueError(
            "CUDA-reported tactic hardware does not match the plan: "
            + canonical_json(mismatched)
        )


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
    require_scan_bypass: bool = True,
    device_properties: dict | None = None,
) -> dict[int, dict]:
    target = plan.get("target", {})
    if target.get("gpu_class") != space.get("gpu_class"):
        raise ValueError(
            "tactic plan GPU class does not match search space: "
            f"{target.get('gpu_class')} != {space.get('gpu_class')}"
        )
    space_device = space.get("cuda_device_properties_at_space_generation")
    if isinstance(space_device, dict):
        expected_compute_capability = space_device.get("compute_capability")
        if target.get("compute_capability") != expected_compute_capability:
            raise ValueError(
                "tactic plan compute capability does not match the "
                "CUDA-reported search-space device"
            )
    if device_properties is not None:
        actual_compute_capability = device_properties.get("compute_capability")
        if actual_compute_capability != target.get("compute_capability"):
            raise ValueError(
                "CUDA-reported compute capability does not match the tactic plan: "
                f"{actual_compute_capability} != {target.get('compute_capability')}"
            )
        expected_devices = target.get(
            "cuda_device_capabilities_at_coordinate_scan"
        ) or target.get("cuda_device_capabilities_at_scan") or (
            [space_device] if isinstance(space_device, dict) else []
        )
        require_compatible_cuda_hardware(expected_devices, device_properties)
    if int(target.get("streams", -1)) != streams:
        raise ValueError("tactic plan stream count does not match the requested run")
    if target.get("fixed_board") != [19, 19]:
        raise ValueError("tactic plans are currently fixed to 19x19")
    if require_scan_bypass and not plan.get("ready_for_scan_bypass", False):
        raise ValueError(
            "tactic plan has not passed the joint long-stability gate"
        )
    if not require_scan_bypass and not plan.get("ready_for_joint_gate", False):
        raise ValueError("tactic plan has incomplete discovery coverage")
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


def validate_coordinate_coverage(plan: dict, batches: Iterable[int]) -> None:
    """Require a chained, incumbent-preserving decision for every coordinate."""
    coordinate = plan.get("coordinate_search")
    if not isinstance(coordinate, dict):
        raise ValueError(
            "only an accumulated coordinate plan can enter the joint gate; "
            "independent family maxima are a seed"
        )
    decisions = coordinate.get("decisions", [])
    covered = set()
    previous_state: dict[int, dict] = {}
    for item in decisions:
        if not isinstance(item, dict):
            raise ValueError("coordinate plan has a malformed decision")
        batch = int(item.get("batch", -1))
        family = item.get("family")
        state_before = item.get("state_before")
        state_after = item.get("state_after")
        incumbent_id = item.get("incumbent_candidate_id")
        winner_id = item.get("winner_candidate_id")
        incumbent_nn = item.get("incumbent_nn_evals_per_sec_median")
        winner_nn = item.get("winner_nn_evals_per_sec_median")
        min_improvement = item.get("min_improvement_fraction")
        recorded_improvement = item.get("improvement_fraction_vs_incumbent")
        if (
            family not in FAMILIES or
            not isinstance(state_before, dict) or
            not isinstance(state_after, dict) or
            set(state_before) != set(FAMILIES) or
            set(state_after) != set(FAMILIES) or
            not isinstance(incumbent_id, str) or
            not isinstance(winner_id, str) or
            not isinstance(incumbent_nn, (int, float)) or
            not isinstance(winner_nn, (int, float)) or
            not isinstance(min_improvement, (int, float)) or
            not isinstance(recorded_improvement, (int, float)) or
            incumbent_nn <= 0.0 or winner_nn <= 0.0
        ):
            raise ValueError(
                "coordinate decision lacks measured incumbent/no-op evidence: "
                f"B{batch}/{family}"
            )
        if state_before.get(family) != incumbent_id:
            raise ValueError(
                f"coordinate incumbent does not match state_before: B{batch}/{family}"
            )
        expected_after = dict(state_before)
        expected_after[family] = winner_id
        if state_after != expected_after:
            raise ValueError(
                f"coordinate decision changed more than one family: B{batch}/{family}"
            )
        expected_before_sha = hashlib.sha256(
            canonical_json(state_before).encode("utf-8")
        ).hexdigest()
        if item.get("state_before_sha256") != expected_before_sha:
            raise ValueError(
                f"coordinate state hash mismatch: B{batch}/{family}"
            )
        if batch in previous_state and state_before != previous_state[batch]:
            raise ValueError(
                f"coordinate decisions are not accumulated: B{batch}/{family}"
            )
        if winner_nn < incumbent_nn:
            raise ValueError(
                f"coordinate winner regresses measured incumbent: B{batch}/{family}"
            )
        accepted_change = winner_id != incumbent_id
        if item.get("accepted_change") is not accepted_change:
            raise ValueError(
                f"coordinate accepted_change is inconsistent: B{batch}/{family}"
            )
        expected_improvement = winner_nn / incumbent_nn - 1.0
        if abs(recorded_improvement - expected_improvement) > 1e-12:
            raise ValueError(
                f"coordinate improvement evidence is inconsistent: B{batch}/{family}"
            )
        if (
            min_improvement < 0.0 or min_improvement >= 1.0 or
            (
                accepted_change and
                recorded_improvement + 1e-12 < min_improvement
            ) or
            (not accepted_change and recorded_improvement != 0.0)
        ):
            raise ValueError(
                "coordinate winner does not clear the discovery threshold: "
                f"B{batch}/{family}"
            )
        previous_state[batch] = state_after
        covered.add((batch, family))
    missing = [
        (batch, family)
        for batch in sorted(set(batches))
        for family in FAMILIES
        if (batch, family) not in covered
    ]
    if missing:
        preview = ", ".join(
            f"B{batch}/{family}" for batch, family in missing[:8]
        )
        raise ValueError(f"coordinate plan has incomplete family coverage: {preview}")
    for batch in sorted(set(batches)):
        selected = {
            family: plan["families"][family]["batches"][str(batch)][
                "candidate_id"
            ]
            for family in FAMILIES
        }
        if previous_state.get(batch) != selected:
            raise ValueError(
                f"coordinate final state does not match plan selection: B{batch}"
            )


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
    joint_result_paths: list[pathlib.Path] | None = None,
) -> dict:
    space = json.loads(space_path.read_text())
    if space.get("schema") != 2:
        raise ValueError("tactic plans require a schema-2 search space")
    gpu_class = space.get("gpu_class")
    if gpu_class not in SUPPORTED_GPU_CLASSES:
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
    missing_space_hash_results: list[str] = []
    missing_cuda_device_results: list[str] = []
    expected_space_sha = sha256_file(space_path)
    cuda_capabilities_at_scan: dict[str, dict] = {}
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
        result_space_sha = regime.get("space_sha256", payload.get("space_sha256"))
        if result_space_sha is None:
            missing_space_hash_results.append(str(path))
        elif result_space_sha != expected_space_sha:
            raise ValueError(
                f"result search-space hash does not match --space: {path}"
            )
        cuda_device = regime.get("cuda_device_properties")
        if isinstance(cuda_device, dict):
            cuda_capabilities_at_scan[canonical_json(cuda_device)] = cuda_device
        else:
            missing_cuda_device_results.append(str(path))
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
    if missing_space_hash_results:
        identity_missing.append("space_sha256 in " + ", ".join(
            pathlib.Path(path).name for path in missing_space_hash_results
        ))
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
    reported_compute_capabilities = {
        tuple(value.get("compute_capability", ()))
        for value in cuda_capabilities_at_scan.values()
        if isinstance(value.get("compute_capability"), list)
    }
    if len(reported_compute_capabilities) > 1:
        raise ValueError(
            "result files contain mixed CUDA-reported compute capabilities"
        )
    target_compute_capability = (
        list(next(iter(reported_compute_capabilities)))
        if reported_compute_capabilities else None
    )
    if target_compute_capability is None:
        identity_missing.append("CUDA-reported compute_capability")
    elif missing_cuda_device_results:
        identity_missing.append(
            "cuda_device_properties in " + ", ".join(
                pathlib.Path(path).name for path in missing_cuda_device_results
            )
        )
    space_device = space.get("cuda_device_properties_at_space_generation")
    if (
        isinstance(space_device, dict) and
        target_compute_capability is not None and
        space_device.get("compute_capability") != target_compute_capability
    ):
        raise ValueError(
            "result CUDA capability does not match the CUDA-reported "
            "search-space device"
        )
    discovery_ready = not missing and not identity_missing
    if not discovery_ready and not allow_partial:
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

    joint_rows: dict[int, dict] = {}
    joint_sources: list[dict] = []
    for path in joint_result_paths or []:
        payload = json.loads(path.read_text())
        if (
            payload.get("schema") != 1 or
            payload.get("kind") != "sm120-joint-plan-whole-graph"
        ):
            raise ValueError(f"unsupported joint result file: {path}")
        regime = payload.get("regime", {})
        if regime.get("space_sha256") != sha256_file(space_path):
            raise ValueError(f"joint result search-space hash mismatch: {path}")
        if config_shas and regime.get("config_sha256") not in config_shas:
            raise ValueError(f"joint result config hash mismatch: {path}")
        if model_shas and regime.get("model_sha256") not in model_shas:
            raise ValueError(f"joint result model hash mismatch: {path}")
        if int(regime.get("streams", -1)) != streams:
            raise ValueError(f"joint result stream topology mismatch: {path}")
        joint_device = regime.get("cuda_device_properties")
        if (
            not isinstance(joint_device, dict) or
            joint_device.get("compute_capability") != target_compute_capability
        ):
            raise ValueError(
                f"joint result CUDA-reported compute capability mismatch: {path}"
            )
        joint_sources.append({
            "path": str(path.resolve()),
            "sha256": sha256_file(path),
            "finished_utc": payload.get("finished_utc"),
        })
        for row in payload.get("rows", []):
            if not isinstance(row, dict) or row.get("status") != "measured":
                continue
            batch = int(row.get("batch", -1))
            if batch not in batches:
                continue
            previous = joint_rows.get(batch)
            if (
                previous is None or
                row.get("finished_utc", "") >= previous.get("finished_utc", "")
            ):
                joint_rows[batch] = row

    joint_gate: dict[str, dict] = {}
    joint_gate_missing: list[dict] = []
    for batch in batches:
        row = joint_rows.get(batch)
        error = None
        if row is None:
            error = "missing joint whole-graph row"
        elif row.get("measurement_kind") != "long_stable":
            error = "joint row is not long_stable"
        elif int(row.get("measurement_iterations", -1)) < MIN_LONG_ITERATIONS:
            error = "joint row has too few timed iterations"
        elif int(row.get("measurement_sample_count", -1)) < MIN_STABLE_SAMPLES:
            error = "joint row has too few stable samples"
        elif float(row.get("measurement_relative_spread", float("inf"))) > MAX_STABLE_RELATIVE_SPREAD:
            error = "joint row exceeds the stability spread limit"
        elif not isinstance(row.get("stable_long_nn_evals_per_sec"), (int, float)):
            error = "joint row has no stable long throughput"
        else:
            selected = row.get("selected", {})
            for family in requested_families:
                expected_id = selected_families[family]["batches"].get(
                    str(batch), {}
                ).get("candidate_id")
                actual = selected.get(family, {}) if isinstance(selected, dict) else {}
                if not isinstance(actual, dict) or actual.get("candidate_id") != expected_id:
                    error = f"joint row tactic mismatch for {family}"
                    break
        if error is not None:
            joint_gate_missing.append({"batch": batch, "error": error})
            continue
        joint_gate[str(batch)] = {
            "stable_long_nn_evals_per_sec": row["stable_long_nn_evals_per_sec"],
            "nn_evals_per_sec_samples": row.get("nn_evals_per_sec_samples", []),
            "measurement_iterations": row.get("measurement_iterations"),
            "measurement_warmup": row.get("measurement_warmup"),
            "measurement_sample_count": row.get("measurement_sample_count"),
            "measurement_relative_spread": row.get("measurement_relative_spread"),
            "measurement_kind": row.get("measurement_kind"),
            "binary_sha256": row.get("binary_sha256"),
            "selected": row.get("selected"),
        }
    joint_ready = not joint_gate_missing and bool(batches)
    # Independent family maxima are only a coordinate-search seed. Even a
    # stable measurement of their combined graph does not prove that each
    # family was searched against the accepted choices of the other families.
    # Only ``finalize_plan`` may grant scan bypass, and it requires accumulated
    # coordinate-decision evidence.
    ready = False

    target = {
        "gpu_class": gpu_class,
        # Hardware identity is sourced from the CUDA Runtime records in the
        # scan results, never inferred from a marketing name.
        "compute_capability": target_compute_capability,
        "fixed_board": [19, 19],
        "precision": "FP16/NHWC",
        "streams": streams,
        "model_sha256": next(iter(model_shas), None),
        "config_sha256": next(iter(config_shas), None),
        "cuda_device_capabilities_at_scan": [
            cuda_capabilities_at_scan[key]
            for key in sorted(cuda_capabilities_at_scan)
        ],
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
        "joint_gate": joint_gate,
    }
    plan_hash = hashlib.sha256(canonical_json(identity).encode("utf-8")).hexdigest()
    return {
        "schema": 1,
        "kind": "sm120-tactic-plan",
        "plan_id": f"sm120-{plan_hash[:16]}",
        "plan_sha256": plan_hash,
        "generated_utc": utc_now(),
        "status": (
            "complete_independent_discovery_needs_coordinate"
            if discovery_ready else "partial_discovery"
        ),
        "ready_for_joint_gate": discovery_ready,
        "ready_for_scan_bypass": ready,
        "selection": {
            "metric": "natural whole-graph S2 median nnEval/s",
            "method": "per-family, per-batch maximum among measured candidates",
            "is_acceptance": False,
            "required_follow_up": (
                "run accumulated per-batch coordinate search before the "
                "joint long-stability gate"
            ),
        },
        "target": target,
        "batches": batches,
        "families": selected_families,
        "coverage": coverage,
        "missing": missing,
        "identity_missing": identity_missing,
        "joint_gate": joint_gate,
        "joint_gate_missing": joint_gate_missing,
        "joint_result_sources": joint_sources,
        "independent_joint_evidence_complete": joint_ready,
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


def finalize_plan(
    plan_path: pathlib.Path, joint_result_paths: list[pathlib.Path],
    space_path: pathlib.Path, model_path: pathlib.Path, config_path: pathlib.Path,
    batches: list[int], streams: int,
) -> dict:
    """Attach long-stable evidence to an already selected exact-batch plan."""
    plan = load_plan(plan_path)
    validate_coordinate_coverage(plan, batches)
    space = json.loads(space_path.read_text())
    space["_path"] = str(space_path)
    for family in FAMILIES:
        validate_plan(
            plan, space, model_path, family, batches, streams, config_path,
            require_scan_bypass=False,
        )
    if not joint_result_paths:
        raise ValueError("at least one --joint-result is required")

    expected_space_sha = sha256_file(space_path)
    expected_config_sha = sha256_file(config_path)
    expected_model_sha = sha256_file(model_path)
    expected_plan_sha = sha256_file(plan_path)
    rows: dict[int, dict] = {}
    sources = []
    for path in joint_result_paths:
        payload = json.loads(path.read_text())
        if (
            payload.get("schema") != 1 or
            payload.get("kind") != "sm120-joint-plan-whole-graph"
        ):
            raise ValueError(f"unsupported joint result file: {path}")
        regime = payload.get("regime", {})
        expected = {
            "space_sha256": expected_space_sha,
            "config_sha256": expected_config_sha,
            "model_sha256": expected_model_sha,
            "plan_sha256": expected_plan_sha,
        }
        for key, value in expected.items():
            if regime.get(key) != value:
                raise ValueError(f"joint result {key} mismatch: {path}")
        if int(regime.get("streams", -1)) != streams:
            raise ValueError(f"joint result stream topology mismatch: {path}")
        joint_device = regime.get("cuda_device_properties")
        if (
            not isinstance(joint_device, dict) or
            joint_device.get("compute_capability") != plan["target"].get(
                "compute_capability"
            )
        ):
            raise ValueError(
                f"joint result has no matching CUDA-reported capability: {path}"
            )
        sources.append({
            "path": str(path.resolve()),
            "sha256": sha256_file(path),
            "finished_utc": payload.get("finished_utc"),
        })
        for row in payload.get("rows", []):
            if not isinstance(row, dict) or row.get("status") != "measured":
                continue
            batch = int(row.get("batch", -1))
            if batch not in batches:
                continue
            previous = rows.get(batch)
            if previous is None or row.get("finished_utc", "") >= previous.get("finished_utc", ""):
                rows[batch] = row

    joint_gate = {}
    missing = []
    for batch in batches:
        row = rows.get(batch)
        error = None
        if row is None:
            error = "missing joint whole-graph row"
        elif row.get("measurement_kind") != "long_stable":
            error = "joint row is not long_stable"
        elif int(row.get("measurement_iterations", -1)) < MIN_LONG_ITERATIONS:
            error = "joint row has too few timed iterations"
        elif int(row.get("measurement_sample_count", -1)) < MIN_STABLE_SAMPLES:
            error = "joint row has too few stable samples"
        elif float(row.get("measurement_relative_spread", float("inf"))) > MAX_STABLE_RELATIVE_SPREAD:
            error = "joint row exceeds the stability spread limit"
        elif not isinstance(row.get("stable_long_nn_evals_per_sec"), (int, float)):
            error = "joint row has no stable long throughput"
        else:
            selected = row.get("selected", {})
            for family in FAMILIES:
                expected_id = plan["families"][family]["batches"][str(batch)][
                    "candidate_id"
                ]
                actual = selected.get(family, {}) if isinstance(selected, dict) else {}
                if not isinstance(actual, dict) or actual.get("candidate_id") != expected_id:
                    error = f"joint row tactic mismatch for {family}"
                    break
        if error is not None:
            missing.append({"batch": batch, "error": error})
            continue
        joint_gate[str(batch)] = {
            key: row.get(key)
            for key in (
                "stable_long_nn_evals_per_sec", "nn_evals_per_sec_samples",
                "measurement_iterations", "measurement_warmup",
                "measurement_sample_count", "measurement_relative_spread",
                "measurement_kind", "binary_sha256", "selected",
            )
        }
    if missing:
        preview = ", ".join(
            f"B{item['batch']}: {item['error']}" for item in missing[:8]
        )
        raise ValueError(f"joint long-stability gate is incomplete: {preview}")

    result = copy.deepcopy(plan)
    result.update({
        "generated_utc": utc_now(),
        "status": "complete_long_stable",
        "ready_for_joint_gate": True,
        "ready_for_scan_bypass": True,
        # Scan bypass means that the whole-graph tactic search and long gate
        # may be reused. It is deliberately weaker than production approval:
        # the immutable 8192-row FP32 reference is a separate artifact and is
        # not available in the migration bundle.
        "production_ready": False,
        "numerical_certification": {
            "status": "missing",
            "required_reference": "immutable 8192-row FP32 replay",
        },
        "joint_gate": joint_gate,
        "joint_gate_missing": [],
        "joint_result_sources": sources,
    })
    result.setdefault("selection", {})["is_acceptance"] = True
    result["selection"]["required_follow_up"] = None
    identity = {
        "target": result["target"],
        "batches": batches,
        "families": {
            family: {
                str(batch): result["families"][family]["batches"][str(batch)][
                    "candidate_id"
                ]
                for batch in batches
            }
            for family in FAMILIES
        },
        "joint_gate": joint_gate,
    }
    plan_hash = hashlib.sha256(canonical_json(identity).encode("utf-8")).hexdigest()
    result["plan_id"] = f"sm120-{plan_hash[:16]}"
    result["plan_sha256"] = plan_hash
    result.setdefault("provenance", {})["pre_finalized_plan"] = {
        "path": str(plan_path.resolve()),
        "sha256": expected_plan_sha,
        "plan_id": plan.get("plan_id"),
    }
    return result


def build_command(args: argparse.Namespace) -> None:
    families = [item.strip() for item in args.families.split(",") if item.strip()]
    unknown = sorted(set(families) - set(FAMILIES))
    if not families or unknown:
        raise ValueError(f"invalid tactic families: {unknown or families}")
    payload = build_plan(
        [pathlib.Path(value).resolve() for value in args.results],
        pathlib.Path(args.space).resolve(), families,
        parse_int_set(args.batches), args.allow_partial,
        [pathlib.Path(value).resolve() for value in args.joint_result],
    )
    write_json(pathlib.Path(args.output).resolve(), payload)
    print(json.dumps({
        "output": str(pathlib.Path(args.output).resolve()),
        "plan_id": payload["plan_id"],
        "ready_for_scan_bypass": payload["ready_for_scan_bypass"],
        "missing": len(payload["missing"]),
        "joint_gate_missing": len(payload["joint_gate_missing"]),
    }))


def validate_command(args: argparse.Namespace) -> None:
    try:
        from sm120_device import query_cuda_device
    except ModuleNotFoundError:
        from python.sm120_device import query_cuda_device
    plan = load_plan(pathlib.Path(args.plan).resolve())
    space_path = pathlib.Path(args.space).resolve()
    space = json.loads(space_path.read_text())
    space["_path"] = str(space_path)
    selected = validate_plan(
        plan, space,
        pathlib.Path(args.model).resolve() if args.model else None,
        args.family, parse_int_set(args.batches), args.streams,
        pathlib.Path(args.config).resolve() if args.config else None,
        device_properties=query_cuda_device(args.device),
    )
    print(json.dumps({
        "valid": True,
        "plan_id": plan["plan_id"],
        "family": args.family,
        "batches": sorted(selected),
    }))


def finalize_command(args: argparse.Namespace) -> None:
    payload = finalize_plan(
        pathlib.Path(args.plan).resolve(),
        [pathlib.Path(value).resolve() for value in args.joint_result],
        pathlib.Path(args.space).resolve(),
        pathlib.Path(args.model).resolve(),
        pathlib.Path(args.config).resolve(),
        parse_int_set(args.batches), args.streams,
    )
    output = pathlib.Path(args.output).resolve()
    write_json(output, payload)
    print(json.dumps({
        "output": str(output),
        "plan_id": payload["plan_id"],
        "ready_for_scan_bypass": payload["ready_for_scan_bypass"],
        "long_stable_batches": len(payload["joint_gate"]),
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
    build.add_argument(
        "--joint-result", action="append", default=[],
        help="completed long-stable joint whole-graph result; repeatable",
    )
    build.set_defaults(function=build_command)
    validate = subparsers.add_parser("validate", help="validate a plan for one run")
    validate.add_argument("--plan", required=True)
    validate.add_argument("--space", required=True)
    validate.add_argument("--model")
    validate.add_argument("--config")
    validate.add_argument("--family", choices=FAMILIES, required=True)
    validate.add_argument("--batches", required=True)
    validate.add_argument("--streams", type=int, required=True)
    validate.add_argument("--device", type=int, required=True)
    validate.set_defaults(function=validate_command)
    finalize = subparsers.add_parser(
        "finalize",
        help="attach completed long-stable joint evidence to a selected plan",
    )
    finalize.add_argument("--plan", required=True)
    finalize.add_argument("--joint-result", action="append", required=True)
    finalize.add_argument("--space", required=True)
    finalize.add_argument("--model", required=True)
    finalize.add_argument("--config", required=True)
    finalize.add_argument("--batches", required=True)
    finalize.add_argument("--streams", type=int, required=True)
    finalize.add_argument("--output", required=True)
    finalize.set_defaults(function=finalize_command)
    return parser


def main() -> None:
    args = make_parser().parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
