import hashlib
import json
import pathlib
import unittest


REPO = pathlib.Path(__file__).resolve().parents[2]
PLAN = (
    REPO
    / "final-migration"
    / "plans"
    / "sm89"
    / "rtx4090d-b12-s2"
    / "best-tactic-plan.json"
)
PLAN_FILE_SHA256 = "57aba0d9f5ff009f0103fe792766bd3fe065d156c13396cb99bc40b5488f9edb"
MODEL_SHA256 = "1881600caab9e9d85a3dd6a019e9b8e7d2c237b5f984e13ed49a8645be3077c6"


class CheckedInTacticPlanTests(unittest.TestCase):
    def test_sm89_plan_is_the_certified_immutable_file(self) -> None:
        raw = PLAN.read_bytes()
        self.assertEqual(hashlib.sha256(raw).hexdigest(), PLAN_FILE_SHA256)

        plan = json.loads(raw)
        self.assertEqual(plan["schema"], 1)
        self.assertEqual(plan["kind"], "cuda-tactic-plan")
        self.assertEqual(plan["status"], "complete_long_stable")
        self.assertTrue(plan["ready_for_scan_bypass"])
        self.assertTrue(plan["production_ready"])
        self.assertEqual(plan["batches"], [12])

        target = plan["target"]
        self.assertEqual(target["architecture"], "sm89")
        self.assertEqual(target["compute_capability"], [8, 9])
        self.assertEqual(target["fixed_board"], [19, 19])
        self.assertEqual(target["precision"], "FP16/NHWC")
        self.assertEqual(target["streams"], 2)
        self.assertEqual(target["model_sha256"], MODEL_SHA256)

        closure = plan["positive_history_closure"]
        self.assertTrue(closure["complete"])
        self.assertEqual(closure["record_count"], 62)
        self.assertEqual(
            closure["links"],
            ["backend", "scan_candidate", "activation", "plan_apply"],
        )

        final = plan["final_joint"]["12"]
        self.assertEqual(final["measurement_kind"], "long_stable")
        self.assertGreater(final["stable_long_nn_evals_per_sec"], 0.0)
        self.assertEqual(final["correctness"]["status"], "passed")
        self.assertEqual(final["correctness"]["thresholds"]["minimum_rows"], 8192)


if __name__ == "__main__":
    unittest.main()
