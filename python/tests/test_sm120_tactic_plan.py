from __future__ import annotations

import hashlib
import json
import pathlib
import sys
import tempfile
import unittest


PYTHON_DIR = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PYTHON_DIR))

from sm120_tactic_plan import (  # noqa: E402
    build_plan,
    finalize_plan,
    plan_override_config,
    validate_plan,
)
from sm120_run_tactic_search import load_candidate_selection  # noqa: E402


FAMILIES = ("ffn", "qkv", "linear2", "fa4", "l2")


def make_space() -> dict:
    result = {"schema": 2, "gpu_class": "rtx5090d", "batches": []}
    for batch in (1, 2):
        result["batches"].append({
            "batch": batch,
            "ffn": [
                {"id": f"ffn-a-{batch}", "implementation": "tilelang"},
                {"id": f"ffn-b-{batch}", "implementation": "tilelang"},
            ],
            "qkv": [
                {"id": f"qkv-a-{batch}", "implementation": "tilelang"},
                {"id": f"qkv-b-{batch}", "implementation": "tilelang"},
            ],
            "linear2": [
                {"id": f"linear2-a-{batch}", "implementation": "tilelang"},
                {"id": f"linear2-b-{batch}", "implementation": "tilelang"},
            ],
            "fa4": [
                {
                    "id": f"fa4-a-{batch}",
                    "implementation": "fa4_cute",
                    "tile_m": 128,
                    "tile_n": 64,
                    "num_stages": 1,
                },
                {"id": f"fa4-fallback-{batch}", "implementation": "fallback"},
            ],
            "l2": [
                {
                    "id": f"l2-a-{batch}",
                    "implementation": "config",
                    "config": {"cudaPersistingL2Trunk": True},
                },
                {
                    "id": f"l2-b-{batch}",
                    "implementation": "config",
                    "config": {"cudaPersistingL2Trunk": False},
                },
            ],
        })
    return result


class TacticPlanTests(unittest.TestCase):
    def test_per_batch_candidate_selection_is_hash_checked(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_text:
            temporary = pathlib.Path(temporary_text)
            space_path = temporary / "space.json"
            manifest_path = temporary / "manifest.json"
            selection_path = temporary / "selection.json"
            space = make_space()
            space_path.write_text(json.dumps(space))
            space_sha = hashlib.sha256(space_path.read_bytes()).hexdigest()
            manifest_path.write_text(json.dumps({
                "schema": 1,
                "family": "ffn",
                "space_sha256": space_sha,
                "entries": [],
            }))
            selection_path.write_text(json.dumps({
                "schema": 1,
                "selection_metric": "single-stream CUDA-event kernel latency",
                "source_manifests": [str(manifest_path)],
                "groups": [{
                    "family": "ffn",
                    "batch": 1,
                    "retained": ["ffn-a-1"],
                }],
            }))
            selected = load_candidate_selection(
                selection_path, space_path, space, "ffn", [1]
            )
            self.assertEqual(selected["batches"], {1: ["ffn-a-1"]})
            self.assertEqual(len(selected["source_manifests"]), 1)

    def test_build_validate_and_render_plan(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_text:
            temporary = pathlib.Path(temporary_text)
            space_path = temporary / "space.json"
            model_path = temporary / "model.bin"
            config_path = temporary / "config.cfg"
            space = make_space()
            space_path.write_text(json.dumps(space))
            space_sha = hashlib.sha256(space_path.read_bytes()).hexdigest()
            model_path.write_bytes(b"model")
            config_path.write_text("config\n")
            model_sha = hashlib.sha256(model_path.read_bytes()).hexdigest()
            config_sha = hashlib.sha256(config_path.read_bytes()).hexdigest()

            result_paths = []
            for family in FAMILIES:
                rows = []
                for batch_space in space["batches"]:
                    for index, candidate in enumerate(batch_space[family]):
                        rows.append({
                            "batch": batch_space["batch"],
                            "candidate_id": candidate["id"],
                            "candidate": candidate,
                            "implementation": candidate["implementation"],
                            "status": "measured",
                            "finished_utc": f"2026-08-06T00:00:0{index}Z",
                            "nn_evals_per_sec_median": 100.0 + index,
                            "nn_evals_per_sec_samples": [100.0 + index],
                            "binary_sha256": "binary-hash",
                            "generator_metadata": {
                                "source_sha256": f"source-{family}-{batch_space['batch']}",
                            },
                        })
                result = {
                    "schema": 1,
                    "gpu_class": "rtx5090d",
                    "streams": 2,
                    "family": family,
                    "regime": {
                        "space_sha256": space_sha,
                        "model_sha256": model_sha,
                        "config_sha256": config_sha,
                        "cuda_device_properties": {
                            "compute_capability": [12, 0],
                            "multiprocessor_count": 170,
                        },
                    },
                    "environment_snapshots": [{
                        "schema": 1,
                        "captured_utc": "2026-08-06T00:00:00Z",
                        "packages": {"torch": "test"},
                    }],
                    "build_commands": {"configure": ["cmake", "..." ]},
                    "rows": rows,
                }
                result_path = temporary / f"{family}.json"
                result_path.write_text(json.dumps(result))
                result_paths.append(result_path)

            plan = build_plan(
                result_paths, space_path, list(FAMILIES), [1, 2]
            )
            self.assertTrue(plan["ready_for_joint_gate"])
            self.assertFalse(plan["ready_for_scan_bypass"])
            self.assertEqual(len(plan["joint_gate_missing"]), 2)
            self.assertEqual(plan["target"]["model_sha256"], model_sha)
            self.assertEqual(plan["target"]["config_sha256"], config_sha)
            self.assertEqual(
                plan["families"]["ffn"]["batches"]["1"]["candidate_id"],
                "ffn-b-1",
            )
            self.assertEqual(
                plan["reproducibility"]["environment_snapshots"][0]["packages"]["torch"],
                "test",
            )
            self.assertIn(
                "cudaFusedFFNAotTacticSm120=ffn-b-1",
                plan_override_config(plan, 1),
            )
            self.assertIn("cudaPersistingL2Trunk=false", plan_override_config(plan, 1))

            joint_path = temporary / "joint.json"
            joint_path.write_text(json.dumps({
                "schema": 1,
                "kind": "sm120-joint-plan-whole-graph",
                "regime": {
                    "space_sha256": hashlib.sha256(space_path.read_bytes()).hexdigest(),
                    "model_sha256": model_sha,
                    "config_sha256": config_sha,
                    "streams": 2,
                    "cuda_device_properties": {
                        "compute_capability": [12, 0],
                        "multiprocessor_count": 170,
                    },
                },
                "rows": [{
                    "batch": batch,
                    "status": "measured",
                    "finished_utc": f"2026-08-06T01:00:0{batch}Z",
                    "measurement_kind": "long_stable",
                    "measurement_iterations": 1000,
                    "measurement_warmup": 50,
                    "measurement_sample_count": 2,
                    "measurement_relative_spread": 0.002,
                    "stable_long_nn_evals_per_sec": 200.0 + batch,
                    "nn_evals_per_sec_samples": [200.0 + batch, 200.5 + batch],
                    "binary_sha256": f"joint-binary-{batch}",
                    "selected": {
                        family: {
                            "candidate_id": plan["families"][family]["batches"]
                                [str(batch)]["candidate_id"]
                        }
                        for family in FAMILIES
                    },
                } for batch in (1, 2)],
            }))
            plan = build_plan(
                result_paths, space_path, list(FAMILIES), [1, 2],
                joint_result_paths=[joint_path],
            )
            self.assertFalse(plan["ready_for_scan_bypass"])
            self.assertTrue(plan["independent_joint_evidence_complete"])
            self.assertEqual(
                plan["status"],
                "complete_independent_discovery_needs_coordinate",
            )

            space_for_validation = json.loads(space_path.read_text())
            space_for_validation["_path"] = str(space_path)
            with self.assertRaisesRegex(ValueError, "joint long-stability gate"):
                validate_plan(
                    plan, space_for_validation, model_path, "ffn", [1, 2], 2,
                    config_path,
                )
            selected = validate_plan(
                plan, space_for_validation, model_path, "ffn", [1, 2], 2,
                config_path, require_scan_bypass=False,
            )
            self.assertEqual(selected[2]["candidate_id"], "ffn-b-2")
            with self.assertRaisesRegex(ValueError, "CUDA-reported compute capability"):
                validate_plan(
                    plan, space_for_validation, model_path, "ffn", [1], 2,
                    config_path, require_scan_bypass=False,
                    device_properties={"compute_capability": [8, 9]},
                )
            with self.assertRaisesRegex(ValueError, "tactic hardware"):
                validate_plan(
                    plan, space_for_validation, model_path, "ffn", [1], 2,
                    config_path, require_scan_bypass=False,
                    device_properties={
                        "compute_capability": [12, 0],
                        "multiprocessor_count": 84,
                        "attributes": {"multiProcessorCount": 84},
                    },
                )

    def test_partial_plan_is_not_usable_as_scan_bypass(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_text:
            temporary = pathlib.Path(temporary_text)
            space_path = temporary / "space.json"
            result_path = temporary / "ffn.json"
            space = make_space()
            space_path.write_text(json.dumps(space))
            result_path.write_text(json.dumps({
                "schema": 1,
                "gpu_class": "rtx5090d",
                "streams": 2,
                "family": "ffn",
                "rows": [],
            }))
            with self.assertRaisesRegex(ValueError, "scan coverage is incomplete"):
                build_plan([result_path], space_path, ["ffn"], [1], False)
            plan = build_plan(
                [result_path], space_path, ["ffn"], [1], allow_partial=True
            )
            self.assertFalse(plan["ready_for_scan_bypass"])
            self.assertTrue(plan["missing"])

    def test_result_space_hash_must_match(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_text:
            temporary = pathlib.Path(temporary_text)
            space_path = temporary / "space.json"
            result_path = temporary / "result.json"
            space_path.write_text(json.dumps(make_space()))
            result_path.write_text(json.dumps({
                "schema": 1,
                "gpu_class": "rtx5090d",
                "streams": 2,
                "family": "ffn",
                "regime": {"space_sha256": "wrong"},
                "rows": [],
            }))
            with self.assertRaisesRegex(ValueError, "search-space hash"):
                build_plan(
                    [result_path], space_path, ["ffn"], [1],
                    allow_partial=True,
                )

    def test_finalize_requires_matching_long_joint_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_text:
            temporary = pathlib.Path(temporary_text)
            space_path = temporary / "space.json"
            model_path = temporary / "model.bin"
            config_path = temporary / "config.cfg"
            plan_path = temporary / "coordinate-plan.json"
            joint_path = temporary / "joint.json"
            space = make_space()
            space_path.write_text(json.dumps(space))
            model_path.write_bytes(b"model")
            config_path.write_text("config\n")
            model_sha = hashlib.sha256(model_path.read_bytes()).hexdigest()
            config_sha = hashlib.sha256(config_path.read_bytes()).hexdigest()
            space_sha = hashlib.sha256(space_path.read_bytes()).hexdigest()
            plan = {
                "schema": 1,
                "kind": "sm120-tactic-plan",
                "plan_id": "coordinate-test",
                "plan_sha256": "selection-only",
                "target": {
                    "gpu_class": "rtx5090d",
                    "compute_capability": [12, 0],
                    "fixed_board": [19, 19],
                    "streams": 2,
                    "model_sha256": model_sha,
                    "config_sha256": config_sha,
                },
                "batches": [1],
                "families": {
                    family: {
                        "space_sha256": space_sha,
                        "batches": {
                            "1": {
                                "candidate_id": space["batches"][0][family][0]["id"],
                                "candidate": space["batches"][0][family][0],
                            }
                        },
                    }
                    for family in FAMILIES
                },
                "ready_for_joint_gate": True,
                "ready_for_scan_bypass": False,
                "selection": {},
            }
            plan_path.write_text(json.dumps(plan))
            with self.assertRaisesRegex(ValueError, "accumulated coordinate plan"):
                finalize_plan(
                    plan_path, [], space_path, model_path, config_path, [1], 2,
                )
            plan["coordinate_search"] = {
                "decisions": [
                    {
                        "batch": 1,
                        "family": family,
                        "state_before": {
                            item: plan["families"][item]["batches"]["1"][
                                "candidate_id"
                            ]
                            for item in FAMILIES
                        },
                        "state_before_sha256": hashlib.sha256(json.dumps(
                            {
                                item: plan["families"][item]["batches"]["1"][
                                    "candidate_id"
                                ]
                                for item in FAMILIES
                            },
                            sort_keys=True, separators=(",", ":"),
                        ).encode()).hexdigest(),
                        "incumbent_candidate_id": plan["families"][family][
                            "batches"
                        ]["1"]["candidate_id"],
                        "incumbent_nn_evals_per_sec_median": 123.0,
                        "winner_candidate_id": plan["families"][family][
                            "batches"
                        ]["1"]["candidate_id"],
                        "winner_nn_evals_per_sec_median": 123.0,
                        "accepted_change": False,
                        "min_improvement_fraction": 0.001,
                        "improvement_fraction_vs_incumbent": 0.0,
                        "state_after": {
                            item: plan["families"][item]["batches"]["1"][
                                "candidate_id"
                            ]
                            for item in FAMILIES
                        },
                    }
                    for family in FAMILIES
                ]
            }
            plan_path.write_text(json.dumps(plan))
            selected = {
                family: {
                    "candidate_id": plan["families"][family]["batches"]["1"][
                        "candidate_id"
                    ]
                }
                for family in FAMILIES
            }
            joint = {
                "schema": 1,
                "kind": "sm120-joint-plan-whole-graph",
                "regime": {
                    "plan_sha256": hashlib.sha256(plan_path.read_bytes()).hexdigest(),
                    "space_sha256": space_sha,
                    "model_sha256": model_sha,
                    "config_sha256": config_sha,
                    "streams": 2,
                    "cuda_device_properties": {
                        "compute_capability": [12, 0],
                        "multiprocessor_count": 170,
                    },
                },
                "rows": [{
                    "batch": 1,
                    "status": "measured",
                    "measurement_kind": "long_stable",
                    "measurement_iterations": 1000,
                    "measurement_warmup": 50,
                    "measurement_sample_count": 2,
                    "measurement_relative_spread": 0.01,
                    "stable_long_nn_evals_per_sec": 123.0,
                    "nn_evals_per_sec_samples": [122.5, 123.5],
                    "selected": selected,
                }],
            }
            joint_path.write_text(json.dumps(joint))
            final = finalize_plan(
                plan_path, [joint_path], space_path, model_path, config_path,
                [1], 2,
            )
            self.assertTrue(final["ready_for_scan_bypass"])
            self.assertEqual(final["status"], "complete_long_stable")
            self.assertEqual(
                final["joint_gate"]["1"]["stable_long_nn_evals_per_sec"],
                123.0,
            )

            joint["rows"][0]["measurement_relative_spread"] = 0.2
            joint_path.write_text(json.dumps(joint))
            with self.assertRaisesRegex(ValueError, "spread limit"):
                finalize_plan(
                    plan_path, [joint_path], space_path, model_path,
                    config_path, [1], 2,
                )


if __name__ == "__main__":
    unittest.main()
