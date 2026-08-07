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
                        "model_sha256": model_sha,
                        "config_sha256": config_sha,
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
            self.assertTrue(plan["ready_for_scan_bypass"])
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

            space_for_validation = json.loads(space_path.read_text())
            space_for_validation["_path"] = str(space_path)
            selected = validate_plan(
                plan, space_for_validation, model_path, "ffn", [1, 2], 2,
                config_path,
            )
            self.assertEqual(selected[2]["candidate_id"], "ffn-b-2")

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


if __name__ == "__main__":
    unittest.main()
