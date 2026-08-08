#!/usr/bin/env python3
"""Run accumulated exact-batch SM120 coordinate search on the whole graph.

The independent family scans are useful for narrowing candidates, but their
maxima cannot simply be combined: FFN, QKV, Linear2, FA4, and persisting-L2
compete for the same GPU.  This runner starts from a complete discovery plan,
measures every candidate for one family while pinning the current choices for
all other families, accepts that family winner, and proceeds to the next
family.  Optional extra passes revisit interactions without a Cartesian
product search.

Output rows are short discovery evidence.  The emitted plan deliberately
remains ineligible for scan bypass until ``sm120_measure_joint_plan.py`` has
measured it with the long-stability regime and ``sm120_tactic_plan.py
finalize`` has attached that evidence.
"""

from __future__ import annotations

import argparse
import copy
import datetime
import hashlib
import json
import os
import pathlib
import shlex
import sys

try:
    from sm120_benchmark_metrics import benchmark_throughput, summarize_throughput
    from sm120_device import query_cuda_device
    from sm120_measure_joint_plan import (
        configure_command,
        override_for,
        prepare_fa4,
        prepare_ffn,
        prepare_linear2,
        prepare_qkv,
        run_command,
    )
    from sm120_run_tactic_search import (
        collect_environment,
        implementation_identity,
        last_json_object,
        parse_int_set,
        reproducibility_identity,
        sha256_file,
    )
    from sm120_prepare_coordinate_fat import (
        FAT_FAMILIES,
        load_coordinate_fat_bundle,
    )
    from sm120_tactic_plan import (
        FAMILIES,
        canonical_json,
        load_plan,
        plan_override_config,
        validate_plan,
    )
except ModuleNotFoundError:  # imported as python.sm120_coordinate_search
    from python.sm120_benchmark_metrics import benchmark_throughput, summarize_throughput
    from python.sm120_device import query_cuda_device
    from python.sm120_measure_joint_plan import (
        configure_command,
        override_for,
        prepare_fa4,
        prepare_ffn,
        prepare_linear2,
        prepare_qkv,
        run_command,
    )
    from python.sm120_run_tactic_search import (
        collect_environment,
        implementation_identity,
        last_json_object,
        parse_int_set,
        reproducibility_identity,
        sha256_file,
    )
    from python.sm120_prepare_coordinate_fat import (
        FAT_FAMILIES,
        load_coordinate_fat_bundle,
    )
    from python.sm120_tactic_plan import (
        FAMILIES,
        canonical_json,
        load_plan,
        plan_override_config,
        validate_plan,
    )


DEFAULT_MIN_DISCOVERY_IMPROVEMENT_FRACTION = 0.001


def utc_now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def write_json(path: pathlib.Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def family_order(value: str) -> list[str]:
    result = [item.strip() for item in value.split(",") if item.strip()]
    unknown = sorted(set(result) - set(FAMILIES))
    if not result or unknown or len(result) != len(set(result)):
        raise ValueError(f"invalid family order: {unknown or result}")
    return result


def batch_space(space: dict, batch: int) -> dict:
    value = next(
        (item for item in space.get("batches", []) if int(item.get("batch", -1)) == batch),
        None,
    )
    if value is None:
        raise ValueError(f"search space has no B{batch}")
    return value


def selected_ids(plan: dict, batch: int) -> dict[str, str]:
    return {
        family: plan["families"][family]["batches"][str(batch)]["candidate_id"]
        for family in FAMILIES
    }


def initial_coordinate_seed(
    space: dict, space_path: pathlib.Path, batches: list[int], streams: int,
    device_properties: dict, model_path: pathlib.Path, config_path: pathlib.Path,
) -> dict:
    """Create a non-deployable deterministic starting state for first scans.

    Coordinate search measures the incumbent/no-op and every challenger, so a
    previous independent-discovery plan is useful but not required for
    correctness. Prefer the explicit fallback where one exists; otherwise use
    search-space order. This seed can never bypass discovery or the joint gate.
    """
    space_sha = sha256_file(space_path)
    families: dict[str, dict] = {}
    selected: dict[str, dict[str, str]] = {}
    for family in FAMILIES:
        entries: dict[str, dict] = {}
        selected[family] = {}
        for batch in batches:
            candidates = batch_space(space, batch).get(family, [])
            if not candidates:
                raise ValueError(f"search space has no {family}/B{batch} candidates")
            candidate = next(
                (
                    item for item in candidates
                    if item.get("implementation") == "fallback"
                ),
                candidates[0],
            )
            entries[str(batch)] = {
                "candidate_id": candidate["id"],
                "candidate": candidate,
                "implementation": candidate.get("implementation", "tilelang"),
            }
            selected[family][str(batch)] = candidate["id"]
        families[family] = {
            "space": space_path.name,
            "space_path_at_scan": str(space_path),
            "space_sha256": space_sha,
            "batches": entries,
        }
    identity = {
        "gpu_class": space.get("gpu_class"),
        "batches": batches,
        "streams": streams,
        "selected": selected,
        "space_sha256": space_sha,
    }
    seed_sha = hashlib.sha256(canonical_json(identity).encode("utf-8")).hexdigest()
    return {
        "schema": 1,
        "kind": "sm120-tactic-plan",
        "plan_id": f"sm120-coordinate-initial-{seed_sha[:16]}",
        "plan_sha256": seed_sha,
        "generated_utc": utc_now(),
        "status": "coordinate_initial_seed_needs_full_scan",
        # validate_plan uses this field as its complete-entry gate when scan
        # bypass is not requested. It does not imply measured acceptance.
        "ready_for_joint_gate": True,
        "ready_for_scan_bypass": False,
        "selection": {
            "metric": None,
            "method": "fallback-first deterministic coordinate seed",
            "is_acceptance": False,
            "required_follow_up": "measure every coordinate candidate",
        },
        "target": {
            "gpu_class": space.get("gpu_class"),
            "compute_capability": device_properties["compute_capability"],
            "fixed_board": [19, 19],
            "precision": "FP16/NHWC",
            "streams": streams,
            "model_sha256": sha256_file(model_path),
            "config_sha256": sha256_file(config_path),
            "cuda_device_capabilities_at_scan": [device_properties],
        },
        "batches": batches,
        "families": families,
        "provenance": {
            "coordinate_initial_seed": {
                "space": str(space_path),
                "space_sha256": space_sha,
                "rule": "explicit fallback, else first candidate in search-space order",
                "measured": False,
            },
        },
    }


def state_sha256(state: dict[str, str]) -> str:
    return hashlib.sha256(canonical_json(state).encode("utf-8")).hexdigest()


def rebase_coordinate_seed_space(
    seed_plan: dict, space: dict, space_path: pathlib.Path,
    batches: list[int],
) -> dict:
    """Rebind a non-deployable seed to an exact-candidate superset space.

    Coordinate search remeasures every candidate, so an older independent
    discovery plan is only an initial state. Rebinding is safe when the old
    space artifact is hash-valid and every selected candidate is byte-for-byte
    identical in both spaces. The resulting coordinate plan remains gated by
    fresh discovery and long joint measurements.
    """
    result = copy.deepcopy(seed_plan)
    new_space_sha = sha256_file(space_path)
    rebased = []
    for family in FAMILIES:
        family_payload = result.get("families", {}).get(family, {})
        old_space_sha = family_payload.get("space_sha256")
        if old_space_sha == new_space_sha:
            continue
        old_space_value = family_payload.get("space_path_at_scan")
        if not isinstance(old_space_value, str):
            raise ValueError(
                f"cannot rebase seed {family}: source space path is missing"
            )
        old_space_path = pathlib.Path(old_space_value).resolve()
        if not old_space_path.is_file() or sha256_file(old_space_path) != old_space_sha:
            raise ValueError(
                f"cannot rebase seed {family}: source space hash is unavailable"
            )
        old_space = json.loads(old_space_path.read_text())
        if (
            old_space.get("schema") != 2 or
            old_space.get("gpu_class") != space.get("gpu_class")
        ):
            raise ValueError(
                f"cannot rebase seed {family}: target class changed"
            )
        for batch in batches:
            entry = family_payload.get("batches", {}).get(str(batch))
            if not isinstance(entry, dict):
                raise ValueError(f"cannot rebase seed {family}/B{batch}: missing entry")
            candidate_id = entry.get("candidate_id")
            planned = entry.get("candidate")
            old_candidates = {
                item["id"]: item for item in batch_space(old_space, batch).get(family, [])
            }
            new_candidates = {
                item["id"]: item for item in batch_space(space, batch).get(family, [])
            }
            if (
                old_candidates.get(candidate_id) != planned or
                new_candidates.get(candidate_id) != planned
            ):
                raise ValueError(
                    "cannot rebase seed because its selected candidate changed: "
                    f"{family}/B{batch}/{candidate_id}"
                )
        family_payload.update({
            "space": space_path.name,
            "space_path_at_scan": str(space_path.resolve()),
            "space_sha256": new_space_sha,
        })
        rebased.append({
            "family": family,
            "from_space": str(old_space_path),
            "from_space_sha256": old_space_sha,
            "to_space": str(space_path.resolve()),
            "to_space_sha256": new_space_sha,
            "rule": "selected candidate exact in hash-valid source and target spaces",
        })
    if rebased:
        result.setdefault("provenance", {})["coordinate_seed_space_rebase"] = rebased
        result["ready_for_scan_bypass"] = False
    return result


def choose_coordinate_winner(
    measured: list[dict], incumbent_candidate_id: str,
    min_improvement_fraction: float = DEFAULT_MIN_DISCOVERY_IMPROVEMENT_FRACTION,
) -> tuple[dict, dict]:
    """Choose a coordinate winner while proving that no-op was measured.

    SM120 candidates are complete tactic choices rather than incremental
    config fragments.  Re-running the currently selected candidate is
    therefore the coordinate-search no-op.  Make that invariant explicit so
    future candidate narrowing cannot silently reintroduce forced regressions.
    A challenger must clear the configured minimum improvement, not merely
    win by short-run noise. Exact throughput ties retain the incumbent.
    """
    incumbents = [
        row for row in measured
        if row.get("candidate_id") == incumbent_candidate_id
    ]
    if len(incumbents) != 1:
        raise ValueError(
            "coordinate candidate set must measure the incumbent exactly once: "
            f"{incumbent_candidate_id}"
        )
    incumbent = incumbents[0]
    winner = max(
        measured,
        key=lambda row: (
            row["nn_evals_per_sec_median"],
            row.get("candidate_id") == incumbent_candidate_id,
        ),
    )
    required = (
        float(incumbent["nn_evals_per_sec_median"])
        * (1.0 + min_improvement_fraction)
    )
    if (
        winner.get("candidate_id") != incumbent_candidate_id and
        float(winner["nn_evals_per_sec_median"]) < required
    ):
        winner = incumbent
    return winner, incumbent


def select_candidate(
    plan: dict, family: str, batch: int, candidate: dict, evidence: dict | None = None,
) -> None:
    previous = plan["families"][family]["batches"].get(str(batch), {})
    entry = {
        key: value for key, value in previous.items()
        if key not in {
            "candidate_id", "candidate", "implementation",
            "nn_evals_per_sec_median", "nn_evals_per_sec_samples",
            "binary_sha256", "override_config", "coordinate_evidence",
            "artifact_sha256", "generator_parameters", "generator_metadata",
            "fat_scan_entry", "source_result", "source_result_path_at_scan",
            "coordinate_artifact",
        }
    }
    entry.update({
        "candidate_id": candidate["id"],
        "candidate": candidate,
        "implementation": candidate.get("implementation", "tilelang"),
    })
    if evidence is not None:
        entry.update({
            "nn_evals_per_sec_median": evidence["nn_evals_per_sec_median"],
            "nn_evals_per_sec_samples": evidence.get("nn_evals_per_sec_samples", []),
            "binary_sha256": evidence.get("binary_sha256"),
            "override_config": evidence.get("override_config"),
            "coordinate_evidence": {
                "pass": evidence["pass"],
                "family": family,
                "state_before_sha256": evidence["state_before_sha256"],
                "finished_utc": evidence.get("finished_utc"),
                "commands": evidence.get("commands"),
            },
        })
        artifact = evidence.get("artifacts", {}).get(family)
        if isinstance(artifact, dict):
            entry["coordinate_artifact"] = artifact
    plan["families"][family]["batches"][str(batch)] = entry


def materialize_and_measure(
    args: argparse.Namespace, repo: pathlib.Path, space_path: pathlib.Path,
    config_path: pathlib.Path, model_path: pathlib.Path, build_dir: pathlib.Path,
    active_dir: pathlib.Path, logs: pathlib.Path, runner: list[str],
    generator_python: str, fa4_python: str, working_plan: dict, batch: int,
    row_prefix: str,
) -> dict:
    row_logs = logs / row_prefix
    history_root = pathlib.Path(args.historical_root).resolve()
    qkv_root = pathlib.Path(args.qkv_generated_root).resolve()
    cutlass_root = pathlib.Path(args.cutlass_root).resolve()
    ffn = prepare_ffn(
        repo, working_plan, batch, active_dir, history_root,
        generator_python, runner, row_logs, space_path, args.device,
    )
    qkv = prepare_qkv(
        repo, working_plan, batch, active_dir, qkv_root, generator_python,
        runner, row_logs, space_path, args.device, cutlass_root,
    )
    linear2 = prepare_linear2(
        repo, working_plan, batch, active_dir, generator_python, runner,
        row_logs, space_path, args.device,
    )
    fa4 = prepare_fa4(
        repo, working_plan, batch, active_dir, fa4_python, runner,
        row_logs, args.device,
    )
    artifacts = {"ffn": ffn, "qkv": qkv, "linear2": linear2, "fa4": fa4}
    configure = configure_command(repo, build_dir, active_dir, fa4, qkv)
    build = ["cmake", "--build", str(build_dir), f"-j{args.jobs}"]
    run_command(configure, row_logs / "configure")
    run_command(build, row_logs / "build")
    binary = build_dir / "katago"
    if not binary.is_file():
        raise RuntimeError(f"build did not produce {binary}")
    override = override_for(working_plan, batch, args.device, args.streams)
    samples = []
    records = []
    benchmark_template = None
    for repeat in range(args.repeats):
        benchmark_template = runner + [
            str(binary), "benchmarknn", "-config", str(config_path),
            "-override-config", override, "-model", str(model_path),
            "-iterations", str(args.iterations), "-warmup", str(args.warmup),
            "-batch-size", str(batch), "-boardsize", "19", "-json",
        ]
        result = last_json_object(
            run_command(
                benchmark_template, row_logs / f"benchmark-r{repeat}"
            ).stdout
        )
        samples.append(benchmark_throughput(result))
        records.append(result)
    return {
        "artifacts": artifacts,
        "commands": {
            "configure": configure,
            "build": build,
            "benchmark_template": benchmark_template,
        },
        "override_config": override,
        "binary": str(binary),
        "binary_sha256": sha256_file(binary),
        "nn_evals_per_sec_samples": samples,
        "benchmark_records": records,
        **summarize_throughput(
            samples, iterations=args.iterations, warmup=args.warmup,
        ),
    }


def measure_prelinked(
    args: argparse.Namespace, config_path: pathlib.Path,
    model_path: pathlib.Path, logs: pathlib.Path, runner: list[str],
    working_plan: dict, batch: int, row_prefix: str, fat_bundle: dict,
) -> dict:
    """Measure one coordinate state without generation, configure, or build."""
    binary = pathlib.Path(fat_bundle["_binary"])
    artifacts = {}
    for family in FAT_FAMILIES:
        value = working_plan["families"][family]["batches"][str(batch)][
            "candidate"
        ]
        if value.get("implementation", "tilelang") == "fallback":
            artifacts[family] = {
                "candidate": value,
                "implementation": "fallback",
                "linked_artifact": None,
            }
            continue
        key = (family, batch, value["id"])
        entry = fat_bundle["_entries"].get(key)
        if entry is None:
            raise ValueError(f"fat bundle has no runtime tactic for {key}")
        artifacts[family] = entry

    override = override_for(working_plan, batch, args.device, args.streams)
    row_logs = logs / row_prefix
    samples = []
    records = []
    benchmark_template = None
    for repeat in range(args.repeats):
        benchmark_template = runner + [
            str(binary), "benchmarknn", "-config", str(config_path),
            "-override-config", override, "-model", str(model_path),
            "-iterations", str(args.iterations), "-warmup", str(args.warmup),
            "-batch-size", str(batch), "-boardsize", "19", "-json",
        ]
        result = last_json_object(
            run_command(
                benchmark_template, row_logs / f"benchmark-r{repeat}"
            ).stdout
        )
        samples.append(benchmark_throughput(result))
        records.append(result)
    return {
        "artifacts": artifacts,
        "commands": {
            "prepare": None,
            "configure": None,
            "build": None,
            "benchmark_template": benchmark_template,
        },
        "fat_bundle": {
            "path": fat_bundle["_path"],
            "sha256": fat_bundle["_sha256"],
        },
        "override_config": override,
        "binary": str(binary),
        "binary_sha256": fat_bundle["binary_sha256"],
        "nn_evals_per_sec_samples": samples,
        "benchmark_records": records,
        **summarize_throughput(
            samples, iterations=args.iterations, warmup=args.warmup,
        ),
    }


def export_plan(
    seed_plan: dict, working_plan: dict, payload: dict, batches: list[int],
    streams: int, result_path: pathlib.Path,
) -> dict:
    plan = copy.deepcopy(working_plan)
    plan["batches"] = batches
    for family in FAMILIES:
        plan["families"][family]["batches"] = {
            str(batch): plan["families"][family]["batches"][str(batch)]
            for batch in batches
        }
    decision_identity = [
        {
            key: decision[key]
            for key in (
                "pass", "batch", "family", "state_before",
                "state_before_sha256", "incumbent_candidate_id",
                "incumbent_nn_evals_per_sec_median", "winner_candidate_id",
                "winner_nn_evals_per_sec_median", "accepted_change",
                "min_improvement_fraction",
                "improvement_fraction_vs_incumbent",
                "state_after",
            )
        }
        for decision in payload["decisions"]
    ]
    identity = {
        "target": plan["target"],
        "batches": batches,
        "families": {
            family: {
                str(batch): plan["families"][family]["batches"][str(batch)][
                    "candidate_id"
                ]
                for batch in batches
            }
            for family in FAMILIES
        },
        "coordinate_decisions": decision_identity,
    }
    plan_hash = hashlib.sha256(canonical_json(identity).encode("utf-8")).hexdigest()
    full_family_pass = set(payload["regime"]["family_order"]) == set(FAMILIES)
    reproducibility = plan.setdefault("reproducibility", {})
    snapshots = reproducibility.setdefault("environment_snapshots", [])
    if isinstance(payload.get("environment"), dict):
        snapshots.append(payload["environment"])
    notes = reproducibility.setdefault("notes", [])
    coordinate_note = (
        "Coordinate environment and commands are reproduction evidence, not "
        "strict receiver version gates."
    )
    if coordinate_note not in notes:
        notes.append(coordinate_note)
    plan.update({
        "plan_id": f"sm120-coordinate-{plan_hash[:16]}",
        "plan_sha256": plan_hash,
        "generated_utc": utc_now(),
        "status": (
            "complete_coordinate_discovery_needs_joint_gate"
            if full_family_pass else "partial_coordinate_diagnostic"
        ),
        "ready_for_joint_gate": full_family_pass,
        "ready_for_scan_bypass": False,
        "selection": {
            "metric": "natural whole-graph S2 median nnEval/s",
            "method": "accepted-seed accumulated per-batch coordinate search",
            "family_order": payload["regime"]["family_order"],
            "passes": payload["regime"]["passes"],
            "min_improvement_fraction": payload["regime"][
                "min_improvement_fraction"
            ],
            "is_acceptance": False,
            "required_follow_up": (
                "run all five family coordinates before the joint gate"
                if not full_family_pass else
                "run the joint long-stability gate and finalize this exact plan"
            ),
        },
        "joint_gate": {},
        "joint_gate_missing": [
            {"batch": batch, "error": "missing joint whole-graph row"}
            for batch in batches
        ],
        "joint_result_sources": [],
        "coordinate_search": {
            "source_seed_plan_id": seed_plan.get("plan_id"),
            "result_path_at_scan": str(result_path),
            "result_sha256": sha256_file(result_path),
            "regime": payload["regime"],
            "decisions": decision_identity,
        },
        "apply": {
            "per_batch_tactic_overrides": {
                str(batch): plan_override_config(plan, batch) for batch in batches
            },
            "prefix": (
                f"numNNServerThreadsPerModel={streams},"
                f"cudaPersistingL2StreamsSm120={streams}"
            ),
        },
    })
    return plan


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seed-plan", default="",
        help=(
            "optional complete independent-discovery plan; when omitted, use "
            "a non-deployable deterministic seed and still scan every candidate"
        ),
    )
    parser.add_argument("--space", required=True)
    parser.add_argument("--repo", default=".")
    parser.add_argument("--build-dir", default="")
    parser.add_argument("--active-dir", default="")
    parser.add_argument(
        "--fat-bundle", default="",
        help=(
            "hash-validated sm120-coordinate-fat-bundle; when provided, "
            "candidate measurement never configures or builds"
        ),
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--plan-output", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--device", type=int, required=True)
    parser.add_argument("--batches", default="4-32")
    parser.add_argument("--streams", type=int, default=2)
    parser.add_argument("--family-order", default=",".join(FAMILIES))
    parser.add_argument(
        "--allow-partial-family-order", action="store_true",
        help="diagnostic only; emitted plan cannot enter the joint gate",
    )
    parser.add_argument("--passes", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument(
        "--min-improvement-fraction", type=float,
        default=DEFAULT_MIN_DISCOVERY_IMPROVEMENT_FRACTION,
        help=(
            "retain the measured incumbent unless a candidate exceeds it by "
            "this fraction (default: 0.001)"
        ),
    )
    parser.add_argument("--jobs", type=int, default=len(os.sched_getaffinity(0)))
    parser.add_argument("--runner", default="")
    parser.add_argument("--generator-python", default="")
    parser.add_argument("--fa4-python", default="")
    parser.add_argument(
        "--historical-root",
        default=(
            "/workspace/results/rebuild/cross-batch-search/"
            "historical-ffn-static-selftest"
        ),
    )
    parser.add_argument(
        "--qkv-generated-root",
        default=(
            "/workspace/results/rebuild/cross-batch-search/"
            "s2-5090d-b4-32-qkv-cute-generated"
        ),
    )
    parser.add_argument("--cutlass-root", default="/workspace/third_party/cutlass")
    args = parser.parse_args()
    if args.passes < 1 or args.iterations < 1 or args.repeats < 1:
        parser.error("--passes, --iterations, and --repeats must be positive")
    if not 0.0 <= args.min_improvement_fraction < 1.0:
        parser.error("--min-improvement-fraction must be in [0, 1)")

    repo = pathlib.Path(args.repo).resolve()
    seed_path = pathlib.Path(args.seed_plan).resolve() if args.seed_plan else None
    space_path = pathlib.Path(args.space).resolve()
    config_path = pathlib.Path(args.config).resolve()
    model_path = pathlib.Path(args.model).resolve()
    fat_bundle_path = pathlib.Path(args.fat_bundle).resolve() if args.fat_bundle else None
    if fat_bundle_path is None and (not args.build_dir or not args.active_dir):
        parser.error("active-slot mode requires --build-dir and --active-dir")
    build_dir = pathlib.Path(args.build_dir).resolve() if args.build_dir else None
    active_dir = pathlib.Path(args.active_dir).resolve() if args.active_dir else None
    output_path = pathlib.Path(args.output).resolve()
    plan_output = pathlib.Path(args.plan_output).resolve()
    logs = output_path.parent / f"{output_path.stem}-logs"
    batches = parse_int_set(args.batches)
    families = family_order(args.family_order)
    if set(families) != set(FAMILIES) and not args.allow_partial_family_order:
        parser.error(
            "--family-order must include all five families; use "
            "--allow-partial-family-order only for a diagnostic smoke run"
        )
    runner = shlex.split(args.runner)
    generator_python = args.generator_python or (
        "/workspace/venv/bin/python3"
        if pathlib.Path("/workspace/venv/bin/python3").is_file()
        else sys.executable
    )
    fa4_python = args.fa4_python or generator_python
    device_properties = query_cuda_device(args.device)
    space = json.loads(space_path.read_text())
    space["_path"] = str(space_path)
    seed_plan = (
        load_plan(seed_path)
        if seed_path is not None else
        initial_coordinate_seed(
            space, space_path, batches, args.streams, device_properties,
            model_path, config_path,
        )
    )
    fat_bundle = (
        load_coordinate_fat_bundle(
            fat_bundle_path, space, space_path, batches,
        )
        if fat_bundle_path is not None else None
    )
    working_plan = (
        rebase_coordinate_seed_space(seed_plan, space, space_path, batches)
        if seed_path is not None else copy.deepcopy(seed_plan)
    )
    for family in FAMILIES:
        validate_plan(
            working_plan, space, model_path, family, batches, args.streams,
            config_path, require_scan_bypass=False,
            device_properties=device_properties,
        )
    working_plan.setdefault("target", {})["compute_capability"] = (
        device_properties["compute_capability"]
    )
    working_plan["target"]["cuda_device_capabilities_at_coordinate_scan"] = [
        device_properties
    ]

    environment = collect_environment(
        repo, config_path, model_path,
        {"cutlass": pathlib.Path(args.cutlass_root).resolve()},
        fa4_python, device_properties,
    )
    regime = {
        "seed_plan": str(seed_path) if seed_path is not None else None,
        "seed_plan_sha256": sha256_file(seed_path) if seed_path is not None else None,
        "seed_plan_id": seed_plan.get("plan_id"),
        "space": str(space_path), "space_sha256": sha256_file(space_path),
        "config": str(config_path), "config_sha256": sha256_file(config_path),
        "model": str(model_path), "model_sha256": sha256_file(model_path),
        "device": args.device, "cuda_device_properties": device_properties,
        "streams": args.streams, "batches": batches,
        "family_order": families, "passes": args.passes,
        "partial_family_diagnostic": args.allow_partial_family_order,
        "iterations": args.iterations, "warmup": args.warmup,
        "repeats": args.repeats, "jobs": args.jobs, "runner": runner,
        "min_improvement_fraction": args.min_improvement_fraction,
        "generator_python": generator_python, "fa4_python": fa4_python,
        "historical_root": str(pathlib.Path(args.historical_root).resolve()),
        "qkv_generated_root": str(pathlib.Path(args.qkv_generated_root).resolve()),
        "cutlass_root": str(pathlib.Path(args.cutlass_root).resolve()),
        "build_dir": str(build_dir) if build_dir is not None else None,
        "active_dir": str(active_dir) if active_dir is not None else None,
        "fat_bundle": (
            {
                "path": fat_bundle["_path"],
                "sha256": fat_bundle["_sha256"],
                "binary": fat_bundle["_binary"],
                "binary_sha256": fat_bundle["binary_sha256"],
            }
            if fat_bundle is not None else None
        ),
        "implementation_identity": implementation_identity(repo),
        "reproducibility_identity": reproducibility_identity(environment),
    }
    if output_path.is_file():
        payload = json.loads(output_path.read_text())
        if payload.get("regime") != regime:
            raise ValueError("existing coordinate result has a different regime")
    else:
        payload = {
            "schema": 1,
            "kind": "sm120-accumulated-coordinate-search",
            "started_utc": utc_now(),
            "regime": regime,
            "rows": [],
            "decisions": [],
            "environment": environment,
        }
        write_json(output_path, payload)

    row_map = {
        (
            int(row["pass"]), int(row["batch"]), row["family"],
            row["state_before_sha256"], row["candidate_id"],
        ): row
        for row in payload.get("rows", [])
        if row.get("status") == "measured"
    }
    decision_map = {
        (
            int(item["pass"]), int(item["batch"]), item["family"],
            item["state_before_sha256"],
        ): item
        for item in payload.get("decisions", [])
    }
    if build_dir is not None:
        build_dir.mkdir(parents=True, exist_ok=True)
    if active_dir is not None:
        active_dir.mkdir(parents=True, exist_ok=True)

    for pass_index in range(1, args.passes + 1):
        for batch in batches:
            candidates_by_family = batch_space(space, batch)
            for family in families:
                state_before = selected_ids(working_plan, batch)
                before_sha = state_sha256(state_before)
                decision_key = (pass_index, batch, family, before_sha)
                prior_decision = decision_map.get(decision_key)
                candidates = candidates_by_family.get(family, [])
                by_id = {candidate["id"]: candidate for candidate in candidates}
                incumbent_id = state_before[family]
                if incumbent_id not in by_id:
                    raise ValueError(
                        "coordinate space omits the current tactic/no-op: "
                        f"P{pass_index}/B{batch}/{family}/{incumbent_id}"
                    )
                if prior_decision is not None:
                    winner_id = prior_decision["winner_candidate_id"]
                    if winner_id not in by_id:
                        raise ValueError(
                            f"resumed winner is absent from space: {family}/B{batch}/{winner_id}"
                        )
                    winner_row = row_map[
                        (pass_index, batch, family, before_sha, winner_id)
                    ]
                    select_candidate(
                        working_plan, family, batch, by_id[winner_id], winner_row,
                    )
                    print(
                        f"P{pass_index} B{batch} {family}: resume {winner_id}",
                        flush=True,
                    )
                    continue

                measured = []
                for candidate in candidates:
                    candidate_id = candidate["id"]
                    row_key = (pass_index, batch, family, before_sha, candidate_id)
                    previous = row_map.get(row_key)
                    if previous is not None:
                        measured.append(previous)
                        continue
                    trial_plan = copy.deepcopy(working_plan)
                    select_candidate(trial_plan, family, batch, candidate)
                    row = {
                        "pass": pass_index,
                        "batch": batch,
                        "family": family,
                        "candidate_id": candidate_id,
                        "candidate": candidate,
                        "is_incumbent": candidate_id == incumbent_id,
                        "state_before": state_before,
                        "state_before_sha256": before_sha,
                        "selected_graph": selected_ids(trial_plan, batch),
                        "started_utc": utc_now(),
                    }
                    prefix = f"p{pass_index}-b{batch}-{family}-{candidate_id}"
                    try:
                        if fat_bundle is not None:
                            row.update(measure_prelinked(
                                args, config_path, model_path, logs, runner,
                                trial_plan, batch, prefix, fat_bundle,
                            ))
                        else:
                            assert build_dir is not None and active_dir is not None
                            row.update(materialize_and_measure(
                                args, repo, space_path, config_path, model_path,
                                build_dir, active_dir, logs, runner,
                                generator_python, fa4_python, trial_plan, batch,
                                prefix,
                            ))
                        row.update({"status": "measured", "finished_utc": utc_now()})
                        row_map[row_key] = row
                        measured.append(row)
                        print(
                            f"P{pass_index} B{batch} {family} {candidate_id}: "
                            f"{row['nn_evals_per_sec_median']:.3f} nnEval/s",
                            flush=True,
                        )
                    except Exception as error:
                        row.update({
                            "status": "failed", "finished_utc": utc_now(),
                            "error": str(error),
                        })
                        payload["rows"].append(row)
                        payload["updated_utc"] = utc_now()
                        write_json(output_path, payload)
                        raise
                    payload["rows"].append(row)
                    payload["updated_utc"] = utc_now()
                    write_json(output_path, payload)
                if len(measured) != len(candidates):
                    raise RuntimeError(f"incomplete coordinate at P{pass_index}/B{batch}/{family}")
                winner, incumbent = choose_coordinate_winner(
                    measured, incumbent_id, args.min_improvement_fraction,
                )
                incumbent_nn = float(
                    incumbent["nn_evals_per_sec_median"]
                )
                winner_nn = float(winner["nn_evals_per_sec_median"])
                select_candidate(
                    working_plan, family, batch, by_id[winner["candidate_id"]], winner,
                )
                decision = {
                    "pass": pass_index,
                    "batch": batch,
                    "family": family,
                    "state_before": state_before,
                    "state_before_sha256": before_sha,
                    "incumbent_candidate_id": incumbent_id,
                    "incumbent_nn_evals_per_sec_median": incumbent_nn,
                    "winner_candidate_id": winner["candidate_id"],
                    "winner_nn_evals_per_sec_median": winner_nn,
                    "accepted_change": winner["candidate_id"] != incumbent_id,
                    "min_improvement_fraction": args.min_improvement_fraction,
                    "improvement_fraction_vs_incumbent": (
                        winner_nn / incumbent_nn - 1.0
                    ),
                    "state_after": selected_ids(working_plan, batch),
                    "finished_utc": utc_now(),
                }
                payload["decisions"].append(decision)
                decision_map[decision_key] = decision
                payload["updated_utc"] = utc_now()
                write_json(output_path, payload)
                print(
                    f"P{pass_index} B{batch} {family}: accept "
                    f"{winner['candidate_id']} ({winner['nn_evals_per_sec_median']:.3f})",
                    flush=True,
                )

    payload["finished_utc"] = utc_now()
    payload["complete"] = True
    write_json(output_path, payload)
    final_plan = export_plan(
        seed_plan, working_plan, payload, batches, args.streams, output_path,
    )
    write_json(plan_output, final_plan)
    print(json.dumps({
        "output": str(output_path),
        "plan_output": str(plan_output),
        "plan_id": final_plan["plan_id"],
        "ready_for_joint_gate": final_plan["ready_for_joint_gate"],
        "ready_for_scan_bypass": False,
    }))


if __name__ == "__main__":
    main()
