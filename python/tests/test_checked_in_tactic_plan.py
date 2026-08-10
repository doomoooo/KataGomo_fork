import hashlib
import json
import pathlib
import unittest


REPO = pathlib.Path(__file__).resolve().parents[2]
MODEL_SHA256 = "1881600caab9e9d85a3dd6a019e9b8e7d2c237b5f984e13ed49a8645be3077c6"
EXPECTED_PLANS = {}


class CheckedInTacticPlanTests(unittest.TestCase):
    def test_registry_has_exactly_one_current_plan_per_gpu_model(self) -> None:
        discovered: dict[str, pathlib.Path] = {}
        for path in sorted((REPO / "final-migration/plans").rglob(
            "best-tactic-plan.json"
        )):
            plan = json.loads(path.read_text())
            gpu_class = plan["target"]["gpu_class"]
            self.assertNotIn(
                gpu_class, discovered,
                f"multiple current production plans for {gpu_class}",
            )
            discovered[gpu_class] = path.relative_to(REPO)
        self.assertEqual(
            discovered,
            {key: value["path"] for key, value in EXPECTED_PLANS.items()},
        )

    def test_current_plans_are_certified_immutable_files(self) -> None:
        for gpu_class, expected in EXPECTED_PLANS.items():
            with self.subTest(gpu_class=gpu_class):
                path = REPO / expected["path"]
                raw = path.read_bytes()
                self.assertEqual(
                    hashlib.sha256(raw).hexdigest(), expected["file_sha256"]
                )

                plan = json.loads(raw)
                self.assertEqual(plan["schema"], 1)
                self.assertEqual(plan["kind"], "cuda-tactic-plan")
                self.assertEqual(plan["status"], "complete_long_stable")
                self.assertTrue(plan["ready_for_scan_bypass"])
                self.assertTrue(plan["production_ready"])
                self.assertEqual(plan["batches"], [expected["batch"]])

                target = plan["target"]
                self.assertEqual(target["gpu_class"], gpu_class)
                self.assertEqual(target["architecture"], expected["architecture"])
                self.assertEqual(
                    target["compute_capability"], expected["compute_capability"]
                )
                self.assertEqual(target["fixed_board"], [19, 19])
                self.assertEqual(target["precision"], "FP16/NHWC")
                self.assertEqual(target["streams"], 2)
                self.assertEqual(target["model_sha256"], MODEL_SHA256)

                closure = plan["positive_history_closure"]
                self.assertTrue(closure["complete"])
                self.assertEqual(closure["record_count"], expected["records"])
                self.assertEqual(
                    closure["links"],
                    ["backend", "scan_candidate", "activation", "plan_apply"],
                )

                final = plan["final_joint"][str(expected["batch"])]
                self.assertEqual(final["measurement_kind"], "long_stable")
                self.assertGreater(final["stable_long_nn_evals_per_sec"], 0.0)
                self.assertEqual(final["correctness"]["status"], "passed")
                self.assertEqual(
                    final["correctness"]["thresholds"]["minimum_rows"], 8192
                )

                checksum = (path.parent / "SHA256SUMS").read_text().split()[0]
                self.assertEqual(checksum, expected["file_sha256"])


if __name__ == "__main__":
    unittest.main()
