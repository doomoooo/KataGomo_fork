import json
import pathlib
import sys
import tempfile
import unittest


PYTHON_DIR = pathlib.Path(__file__).resolve().parents[1]
REPO_ROOT = PYTHON_DIR.parent
sys.path.insert(0, str(PYTHON_DIR))

import sm103_b29_contract as b29  # noqa: E402


def baseline_payload() -> dict:
    return {
        "schema": 1,
        "kind": "official-backend-baseline",
        "backend": "tensorrt",
        "status": "completed",
        "measurement_mode": "long_confirmation",
        "streams": 2,
        "device": {"compute_capability": [10, 3]},
        "identity": {
            "binary_sha256": b29.FIXED_BASELINE_BINARY_SHA256,
            "config_sha256": b29.FIXED_BASELINE_CONFIG_SHA256,
            "model_sha256": b29.EXPECTED_MODEL_SHA256,
        },
        "rows": [{
            "batch": 29,
            "status": "measured",
            "measurement_kind": "long_stable",
            "measurement_iterations": 1000,
            "measurement_relative_spread": 0.004415,
            "nn_evals_per_sec_median": b29.FIXED_BASELINE_NN_EVALS_PER_SEC,
            "nn_evals_per_sec_samples": [6732.9, 6731.4, 6752.9, 6761.1, 6733.7],
        }],
    }


class Sm103B29ContractTests(unittest.TestCase):
    def test_checked_in_fixed_anchor_matches_the_python_contract(self) -> None:
        path = (
            REPO_ROOT
            / "final-migration/plans/sm103/b300-b29-s2/baseline-anchor.json"
        )
        anchor = json.loads(path.read_text())
        self.assertEqual(anchor["status"], "fixed")
        self.assertEqual(anchor["selection"], {
            "batch": 29,
            "streams": 2,
            "provenance": "user_fixed_after_long_confirmation",
        })
        self.assertEqual(
            anchor["baseline"]["binary_sha256"],
            b29.FIXED_BASELINE_BINARY_SHA256,
        )
        self.assertEqual(
            anchor["baseline"]["config_sha256"],
            b29.FIXED_BASELINE_CONFIG_SHA256,
        )
        self.assertEqual(
            anchor["baseline"]["combined_nn_evals_per_sec_median"],
            b29.FIXED_BASELINE_NN_EVALS_PER_SEC,
        )
        self.assertTrue(
            anchor["optimization_contract"]["batch_selection_fixed"]
        )
        self.assertEqual(
            anchor["optimization_contract"][
                "maximum_request_metric_control_multiplier"
            ],
            b29.TRT16_REQUEST_GATE_MULTIPLIER,
        )
        self.assertFalse(anchor["production_tactic_plan_ready"])

        gate_control_path = path.with_name("request-gate-control.json")
        gate_control = json.loads(gate_control_path.read_text())
        self.assertEqual(gate_control["status"], "fixed")
        self.assertEqual(
            gate_control["request_gate_control"],
            b29.TRT16_REQUEST_GATE_CONTROL,
        )
        self.assertEqual(
            gate_control["acceptance"]["maximum_control_multiplier"],
            b29.TRT16_REQUEST_GATE_MULTIPLIER,
        )

    def test_shape_and_target_are_exact(self) -> None:
        manifest = b29.build_b29_development_manifest()
        self.assertEqual(manifest["batch"], 29)
        self.assertEqual(manifest["streams"], 2)
        self.assertEqual(manifest["rows"], 361 * 29)
        self.assertEqual(manifest["accelerated_target"], "sm_103a")
        self.assertFalse(manifest["production_ready"])
        self.assertTrue(manifest["batch_selection_fixed"])
        self.assertEqual(
            manifest["baseline"]["report_status"],
            "fixed_batch_report_not_attached",
        )

    def test_provisional_baseline_is_bound_but_never_certified(self) -> None:
        with tempfile.TemporaryDirectory() as directory_text:
            path = pathlib.Path(directory_text) / "baseline.json"
            path.write_text(json.dumps(baseline_payload()))
            manifest = b29.build_b29_development_manifest(path)
        anchor = manifest["baseline"]
        self.assertEqual(anchor["batch"], 29)
        self.assertEqual(anchor["sample_count"], 5)
        self.assertEqual(
            anchor["nn_evals_per_sec_median"],
            b29.FIXED_BASELINE_NN_EVALS_PER_SEC,
        )
        self.assertTrue(anchor["batch_selection_fixed"])
        self.assertFalse(anchor["production_ready"])
        self.assertEqual(anchor["report_status"], "fixed_batch_long_confirmation")

    def test_wrong_target_stream_or_model_is_rejected(self) -> None:
        mutations = (
            ("device", {"compute_capability": [10, 0]}),
            ("streams", 1),
        )
        for key, value in mutations:
            with self.subTest(key=key), tempfile.TemporaryDirectory() as directory_text:
                payload = baseline_payload()
                payload[key] = value
                path = pathlib.Path(directory_text) / "baseline.json"
                path.write_text(json.dumps(payload))
                with self.assertRaises(b29.B29AnchorError):
                    b29.load_baseline_anchor(path)

        with tempfile.TemporaryDirectory() as directory_text:
            payload = baseline_payload()
            payload["identity"]["model_sha256"] = "3" * 64
            path = pathlib.Path(directory_text) / "baseline.json"
            path.write_text(json.dumps(payload))
            with self.assertRaisesRegex(b29.B29AnchorError, "different model"):
                b29.load_baseline_anchor(path)

    def test_missing_or_duplicate_b29_row_is_rejected(self) -> None:
        for rows in ([], baseline_payload()["rows"] * 2):
            with self.subTest(count=len(rows)), tempfile.TemporaryDirectory() as directory_text:
                payload = baseline_payload()
                payload["rows"] = rows
                path = pathlib.Path(directory_text) / "baseline.json"
                path.write_text(json.dumps(payload))
                with self.assertRaisesRegex(b29.B29AnchorError, "one measured B29"):
                    b29.load_baseline_anchor(path)

    def test_short_scan_or_changed_identity_is_rejected(self) -> None:
        mutations = (
            ("measurement_kind", "short_scan"),
            ("measurement_iterations", 999),
            ("nn_evals_per_sec_median", 6700.0),
        )
        for key, value in mutations:
            with self.subTest(key=key), tempfile.TemporaryDirectory() as directory_text:
                payload = baseline_payload()
                payload["rows"][0][key] = value
                path = pathlib.Path(directory_text) / "baseline.json"
                path.write_text(json.dumps(payload))
                with self.assertRaises(b29.B29AnchorError):
                    b29.load_baseline_anchor(path)

    def test_trt16_request_gate_uses_the_fixed_control_multiplier(self) -> None:
        thresholds = b29.trt16_request_gate_thresholds()
        for head, metrics in b29.TRT16_REQUEST_GATE_CONTROL.items():
            for metric, value in metrics.items():
                with self.subTest(head=head, metric=metric):
                    self.assertEqual(
                        thresholds[head][metric],
                        value * b29.TRT16_REQUEST_GATE_MULTIPLIER,
                    )

    def test_trt16_and_official_fp16_request_regimes_pass(self) -> None:
        trt_report = {
            "requestGate": b29.TRT16_REQUEST_GATE_CONTROL,
        }
        self.assertTrue(b29.evaluate_trt16_request_gate(trt_report)["passed"])

        official_fp16 = {
            "requestGate": {
                "policyProbability": {
                    "maximumAbs": 0.01633983850479126,
                    "maximumRmse": 0.0009161498746834695,
                },
                "valueProbability": {
                    "maximumAbs": 0.04460674524307251,
                    "maximumRmse": 0.03641660884022713,
                },
                "scoreRaw": {
                    "maximumAbs": 0.49659013748168945,
                    "maximumRmse": 0.20691896975040436,
                },
                "ownershipProbability": {
                    "maximumAbs": 0.006266653537750244,
                    "maximumRmse": 0.00252055237069726,
                },
            }
        }
        outcome = b29.evaluate_trt16_request_gate(official_fp16)
        self.assertTrue(outcome["passed"])
        self.assertLess(
            max(
                ratio
                for metrics in outcome["ratios_to_control"].values()
                for ratio in metrics.values()
            ),
            1.65,
        )

    def test_current_fused_ffn_request_regime_is_rejected(self) -> None:
        candidate = {
            "requestGate": {
                "policyProbability": {
                    "maximumAbs": 0.1679590791463852,
                    "maximumRmse": 0.01229176763445139,
                },
                "valueProbability": {
                    "maximumAbs": 0.06489986181259155,
                    "maximumRmse": 0.052990030497312546,
                },
                "scoreRaw": {
                    "maximumAbs": 0.5262035727500916,
                    "maximumRmse": 0.24151873588562012,
                },
                "ownershipProbability": {
                    "maximumAbs": 0.11509224772453308,
                    "maximumRmse": 0.015420998446643353,
                },
            }
        }
        outcome = b29.evaluate_trt16_request_gate(candidate)
        self.assertFalse(outcome["passed"])
        self.assertFalse(outcome["checks"]["policyProbability"]["maximumAbs"])
        self.assertFalse(
            outcome["checks"]["ownershipProbability"]["maximumRmse"]
        )

    def test_fast_roundtrip_ffn_is_in_the_allowed_error_regime(self) -> None:
        candidate = {
            "requestGate": {
                "policyProbability": {
                    "maximumAbs": 0.018788933753967285,
                    "maximumRmse": 0.001336506917141378,
                },
                "valueProbability": {
                    "maximumAbs": 0.03692281246185303,
                    "maximumRmse": 0.030143601819872856,
                },
                "scoreRaw": {
                    "maximumAbs": 0.43856191635131836,
                    "maximumRmse": 0.20585453510284424,
                },
                "ownershipProbability": {
                    "maximumAbs": 0.011559724807739258,
                    "maximumRmse": 0.002982273930683732,
                },
            }
        }
        outcome = b29.evaluate_trt16_request_gate(candidate)
        self.assertTrue(outcome["passed"])
        self.assertGreater(
            outcome["ratios_to_control"]["policyProbability"]["maximumRmse"],
            2.0,
        )
        self.assertLess(
            outcome["ratios_to_control"]["policyProbability"]["maximumRmse"],
            b29.TRT16_REQUEST_GATE_MULTIPLIER,
        )

    def test_request_gate_fails_closed_on_missing_or_invalid_metrics(self) -> None:
        with self.assertRaisesRegex(b29.B29AnchorError, "no requestGate"):
            b29.evaluate_trt16_request_gate({})
        malformed = {
            "requestGate": {
                head: dict(metrics)
                for head, metrics in b29.TRT16_REQUEST_GATE_CONTROL.items()
            }
        }
        malformed["requestGate"]["policyProbability"]["maximumAbs"] = True
        with self.assertRaisesRegex(b29.B29AnchorError, "finite"):
            b29.evaluate_trt16_request_gate(malformed)


if __name__ == "__main__":
    unittest.main()
