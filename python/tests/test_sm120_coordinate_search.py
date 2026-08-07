import copy
import hashlib
import json
import pathlib
import sys
import tempfile
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sm120_coordinate_search import (  # noqa: E402
    FAMILIES,
    choose_coordinate_winner,
    family_order,
    export_plan,
    initial_coordinate_seed,
    rebase_coordinate_seed_space,
    select_candidate,
    selected_ids,
    state_sha256,
)
from sm120_run_tactic_search import reproducibility_identity  # noqa: E402


class CoordinateSearchTests(unittest.TestCase):
    def test_initial_coordinate_seed_prefers_fallback_and_is_not_deployable(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            space_path = root / "space.json"
            model_path = root / "model.bin.gz"
            config_path = root / "bench.cfg"
            model_path.write_bytes(b"model")
            config_path.write_text("config")
            batch = {"batch": 4}
            for family in FAMILIES:
                batch[family] = [
                    {"id": f"{family}-optimized"},
                    {"id": f"{family}-fallback", "implementation": "fallback"},
                ]
            space = {"schema": 2, "gpu_class": "rtx5080", "batches": [batch]}
            space_path.write_text(json.dumps(space))
            space["_path"] = str(space_path)

            seed = initial_coordinate_seed(
                space, space_path, [4], 2,
                {"compute_capability": "12.0", "multiprocessor_count": 84},
                model_path, config_path,
            )

            self.assertFalse(seed["ready_for_scan_bypass"])
            self.assertEqual(seed["status"], "coordinate_initial_seed_needs_full_scan")
            for family in FAMILIES:
                self.assertEqual(
                    seed["families"][family]["batches"]["4"]["candidate_id"],
                    f"{family}-fallback",
                )

    def test_coordinate_winner_can_keep_incumbent(self) -> None:
        measured = [
            {"candidate_id": "new", "nn_evals_per_sec_median": 3900.0},
            {"candidate_id": "current", "nn_evals_per_sec_median": 4000.0},
        ]
        winner, incumbent = choose_coordinate_winner(measured, "current")
        self.assertEqual(winner["candidate_id"], "current")
        self.assertIs(winner, incumbent)

    def test_coordinate_winner_requires_measured_incumbent(self) -> None:
        measured = [
            {"candidate_id": "new", "nn_evals_per_sec_median": 4000.0},
        ]
        with self.assertRaisesRegex(ValueError, "measure the incumbent"):
            choose_coordinate_winner(measured, "current")

    def test_coordinate_exact_tie_keeps_incumbent(self) -> None:
        measured = [
            {"candidate_id": "new", "nn_evals_per_sec_median": 4000.0},
            {"candidate_id": "current", "nn_evals_per_sec_median": 4000.0},
        ]
        winner, _ = choose_coordinate_winner(measured, "current")
        self.assertEqual(winner["candidate_id"], "current")

    def test_coordinate_subthreshold_noise_keeps_incumbent(self) -> None:
        measured = [
            {"candidate_id": "new", "nn_evals_per_sec_median": 4003.9},
            {"candidate_id": "current", "nn_evals_per_sec_median": 4000.0},
        ]
        winner, _ = choose_coordinate_winner(measured, "current", 0.001)
        self.assertEqual(winner["candidate_id"], "current")

    def test_coordinate_threshold_gain_accepts_challenger(self) -> None:
        measured = [
            {"candidate_id": "new", "nn_evals_per_sec_median": 4004.0},
            {"candidate_id": "current", "nn_evals_per_sec_median": 4000.0},
        ]
        winner, _ = choose_coordinate_winner(measured, "current", 0.001)
        self.assertEqual(winner["candidate_id"], "new")

    def test_candidate_is_accumulated_into_next_coordinate_state(self) -> None:
        families = ("ffn", "qkv", "linear2", "fa4", "l2")
        plan = {
            "families": {
                family: {
                    "batches": {
                        "13": {
                            "candidate_id": f"{family}-old",
                            "candidate": {"id": f"{family}-old"},
                            "generator_metadata": {"source": "stale-old-source"},
                        }
                    }
                }
                for family in families
            }
        }
        before = selected_ids(plan, 13)
        select_candidate(
            plan, "ffn", 13, {"id": "ffn-new", "implementation": "tilelang"},
            {
                "pass": 1,
                "state_before_sha256": state_sha256(before),
                "nn_evals_per_sec_median": 4000.0,
                "nn_evals_per_sec_samples": [4000.0],
                "artifacts": {
                    "ffn": {"source": "new-source", "source_sha256": "new-hash"}
                },
            },
        )
        after = selected_ids(plan, 13)
        self.assertEqual(after["ffn"], "ffn-new")
        self.assertEqual(after["qkv"], "qkv-old")
        self.assertNotEqual(state_sha256(before), state_sha256(after))
        entry = plan["families"]["ffn"]["batches"]["13"]
        self.assertNotIn("generator_metadata", entry)
        self.assertEqual(entry["coordinate_artifact"]["source"], "new-source")

    def test_family_order_rejects_duplicates(self) -> None:
        self.assertEqual(family_order("ffn,qkv,l2"), ["ffn", "qkv", "l2"])
        with self.assertRaisesRegex(ValueError, "invalid family order"):
            family_order("ffn,ffn")

    def test_coordinate_seed_rebases_only_exact_selected_candidates(self) -> None:
        families = ("ffn", "qkv", "linear2", "fa4", "l2")
        with tempfile.TemporaryDirectory() as temporary_text:
            temporary = pathlib.Path(temporary_text)
            old_path = temporary / "old-space.json"
            new_path = temporary / "new-space.json"
            old_space = {
                "schema": 2,
                "gpu_class": "rtx5090d",
                "batches": [{
                    "batch": 13,
                    **{
                        family: [{"id": f"{family}-selected"}]
                        for family in families
                    },
                }],
            }
            new_space = copy.deepcopy(old_space)
            new_space["batches"][0]["fa4"].append({"id": "fa4-n96"})
            old_path.write_text(json.dumps(old_space))
            new_path.write_text(json.dumps(new_space))
            old_sha = hashlib.sha256(old_path.read_bytes()).hexdigest()
            plan = {
                "families": {
                    family: {
                        "space": old_path.name,
                        "space_path_at_scan": str(old_path),
                        "space_sha256": old_sha,
                        "batches": {
                            "13": {
                                "candidate_id": f"{family}-selected",
                                "candidate": {"id": f"{family}-selected"},
                            }
                        },
                    }
                    for family in families
                },
                "ready_for_scan_bypass": True,
            }
            rebased = rebase_coordinate_seed_space(
                plan, new_space, new_path, [13],
            )
            new_sha = hashlib.sha256(new_path.read_bytes()).hexdigest()
            self.assertTrue(all(
                rebased["families"][family]["space_sha256"] == new_sha
                for family in families
            ))
            self.assertFalse(rebased["ready_for_scan_bypass"])
            self.assertEqual(
                len(rebased["provenance"]["coordinate_seed_space_rebase"]),
                len(families),
            )
            changed = copy.deepcopy(new_space)
            changed["batches"][0]["qkv"][0]["changed"] = True
            changed_path = temporary / "changed-space.json"
            changed_path.write_text(json.dumps(changed))
            with self.assertRaisesRegex(ValueError, "selected candidate changed"):
                rebase_coordinate_seed_space(plan, changed, changed_path, [13])

    def test_exported_coordinate_plan_embeds_reproduction_evidence(self) -> None:
        families = ("ffn", "qkv", "linear2", "fa4", "l2")
        seed = {
            "schema": 1,
            "kind": "sm120-tactic-plan",
            "plan_id": "seed",
            "target": {"streams": 2},
            "families": {
                family: {
                    "batches": {
                        "13": {
                            "candidate_id": f"{family}-winner",
                            "candidate": {"id": f"{family}-winner"},
                        }
                    }
                }
                for family in families
            },
        }
        payload = {
            "regime": {
                "family_order": list(families),
                "passes": 1,
                "min_improvement_fraction": 0.001,
            },
            "decisions": [
                {
                    "pass": 1,
                    "batch": 13,
                    "family": family,
                    "state_before": {
                        item: f"{item}-winner" for item in families
                    },
                    "state_before_sha256": state_sha256({
                        item: f"{item}-winner" for item in families
                    }),
                    "incumbent_candidate_id": f"{family}-winner",
                    "incumbent_nn_evals_per_sec_median": 4000.0,
                    "winner_candidate_id": f"{family}-winner",
                    "winner_nn_evals_per_sec_median": 4000.0,
                    "accepted_change": False,
                    "min_improvement_fraction": 0.001,
                    "improvement_fraction_vs_incumbent": 0.0,
                    "state_after": {
                        item: f"{item}-winner" for item in families
                    },
                }
                for family in families
            ],
            "environment": {"packages": {"tilelang": "test-version"}},
        }
        with tempfile.TemporaryDirectory() as temporary_text:
            result_path = pathlib.Path(temporary_text) / "coordinate.json"
            result_path.write_text("{}\n")
            plan = export_plan(seed, seed, payload, [13], 2, result_path)
        self.assertTrue(plan["ready_for_joint_gate"])
        self.assertFalse(plan["ready_for_scan_bypass"])
        self.assertEqual(
            plan["reproducibility"]["environment_snapshots"][-1]["packages"]
                ["tilelang"],
            "test-version",
        )
        self.assertTrue(plan["coordinate_search"]["result_sha256"])

    def test_resume_identity_uses_versions_but_not_dynamic_smi_output(self) -> None:
        snapshot = {
            "captured_utc": "first",
            "python": {"version": "3.test"},
            "packages": {"tilelang": "1.2.3"},
            "cudnn_version_from_torch": 92000,
            "environment": {"CUDA_HOME": "/cuda"},
            "commands": {
                "nvcc_version": {"returncode": 0, "stdout": "cuda-test"},
                "nvidia_smi": {"stdout": "dynamic-utilization"},
            },
            "third_party": {
                "cutlass": {"revision": "abc", "dirty": False, "status": ""}
            },
        }
        first = reproducibility_identity(snapshot)
        snapshot["captured_utc"] = "second"
        snapshot["commands"]["nvidia_smi"]["stdout"] = "other-utilization"
        self.assertEqual(first, reproducibility_identity(snapshot))
        snapshot["packages"]["tilelang"] = "2.0.0"
        self.assertNotEqual(first, reproducibility_identity(snapshot))


if __name__ == "__main__":
    unittest.main()
