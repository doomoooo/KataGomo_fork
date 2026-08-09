import json
import pathlib
import tempfile
import unittest

from python.sm120_report_joint_plan import report


class JointPlanReportTests(unittest.TestCase):
    def test_reports_highest_stable_batch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_text:
            path = pathlib.Path(temporary_text) / "joint.json"
            path.write_text(json.dumps({
                "schema": 1,
                "kind": "sm120-joint-plan-whole-graph",
                "plan_id": "test-plan",
                "rows": [
                    {
                        "batch": 4,
                        "status": "measured",
                        "stable_long_nn_evals_per_sec": 3000.0,
                        "selected": {"fa4": {"candidate_id": "fa4-64"}},
                    },
                    {
                        "batch": 13,
                        "status": "measured",
                        "stable_long_nn_evals_per_sec": 4000.0,
                        "selected": {"fa4": {"candidate_id": "fa4-96"}},
                    },
                ],
            }))
            result = report(path)
            self.assertEqual(result["peak_batch"], 13)
            self.assertEqual(result["peak_stable_long_nn_evals_per_sec"], 4000.0)


if __name__ == "__main__":
    unittest.main()
