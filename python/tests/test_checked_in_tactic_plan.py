import hashlib
import json
import pathlib
import unittest

from python.cuda_tactic_workflow import validate_plan


REPO = pathlib.Path(__file__).resolve().parents[2]
MODEL_SHA256 = "1881600caab9e9d85a3dd6a019e9b8e7d2c237b5f984e13ed49a8645be3077c6"
EXPECTED_PLANS = {
    "rtx4090d": {
        "path": pathlib.Path(
            "final-migration/plans/sm89/rtx4090d-b12-s2/best-tactic-plan.json"
        ),
        "file_sha256": "29559e4ea40d9d117dbf54f82173b03d5833d8f3ebf55999e42dd7d66dd14912",
        "architecture": "sm89",
        "gpu_class": "rtx4090",
        "device_name": "NVIDIA GeForce RTX 4090 D",
        "compute_capability": [8, 9],
        "batch": 12,
        "records": 60,
    },
    "rtx5080": {
        "path": pathlib.Path(
            "final-migration/plans/sm120/rtx5080-b16-s2/best-tactic-plan.json"
        ),
        "file_sha256": "a5f93fbf012baf6170f3d9a65eb44ebf67051b4e77a6fec1c74c5a60fe2385e3",
        "architecture": "sm120",
        "gpu_class": "rtx5080",
        "device_name": "NVIDIA GeForce RTX 5080",
        "compute_capability": [12, 0],
        "batch": 16,
        "records": 63,
    },
}


def receiver_properties(producer: dict[str, object]) -> dict[str, object]:
    return {
        "name": producer["name"],
        "compute_capability": [
            producer["computeCapabilityMajor"],
            producer["computeCapabilityMinor"],
        ],
        "multiProcessorCount": producer["multiProcessorCount"],
        "totalGlobalMem": producer["totalGlobalMem"],
        "attributes": {
            "maxThreadsPerBlock": producer["maxThreadsPerBlock"],
            "maxThreadsPerMultiprocessor":
                producer["maxThreadsPerMultiProcessor"],
            "regsPerMultiprocessor": producer["regsPerMultiprocessor"],
            "maxSharedMemoryPerBlockOptin":
                producer["sharedMemPerBlockOptin"],
            "sharedMemoryPerMultiprocessor":
                producer["sharedMemPerMultiprocessor"],
            "l2CacheSize": producer["l2CacheSize"],
            "memoryBusWidth": producer["memoryBusWidth"],
            "asyncEngineCount": producer["asyncEngineCount"],
            "concurrentKernels": int(producer["concurrentKernels"]),
        },
    }


class CheckedInTacticPlanTests(unittest.TestCase):
    def test_registry_has_exactly_one_current_plan_per_gpu_model(self) -> None:
        discovered: dict[str, pathlib.Path] = {}
        for path in sorted((REPO / "final-migration/plans").rglob(
            "best-tactic-plan.json"
        )):
            plan = json.loads(path.read_text())
            device_name = plan["target"]["cuda_device_capabilities_at_scan"][0][
                "name"
            ]
            self.assertNotIn(
                device_name, discovered,
                f"multiple current production plans for {device_name}",
            )
            discovered[device_name] = path.relative_to(REPO)
        self.assertEqual(
            discovered,
            {
                value["device_name"]: value["path"]
                for value in EXPECTED_PLANS.values()
            },
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
                absolute_paths: list[str] = []

                def collect_absolute_paths(value, path=("$",)) -> None:
                    if isinstance(value, dict):
                        for key, item in value.items():
                            collect_absolute_paths(item, (*path, str(key)))
                    elif isinstance(value, list):
                        for index, item in enumerate(value):
                            collect_absolute_paths(item, (*path, str(index)))
                    elif isinstance(value, str) and value.startswith("/"):
                        absolute_paths.append(".".join(path))

                collect_absolute_paths(plan)
                self.assertEqual(absolute_paths, [])
                self.assertEqual(plan["schema"], 1)
                self.assertEqual(plan["kind"], "cuda-tactic-plan")
                self.assertEqual(plan["status"], "complete_long_stable")
                self.assertTrue(plan["ready_for_scan_bypass"])
                self.assertTrue(plan["production_ready"])
                self.assertEqual(plan["batches"], [expected["batch"]])

                target = plan["target"]
                self.assertEqual(target["gpu_class"], expected["gpu_class"])
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

                producer = target["cuda_device_capabilities_at_scan"][0]
                self.assertEqual(producer["name"], expected["device_name"])
                receiver = receiver_properties(producer)
                validate_plan(plan, device_properties=receiver)

                wrong_name = dict(receiver, name=receiver["name"] + " Other")
                with self.assertRaisesRegex(ValueError, "receiver name"):
                    validate_plan(plan, device_properties=wrong_name)

                wrong_sm = dict(
                    receiver,
                    multiProcessorCount=producer["multiProcessorCount"] + 1,
                )
                with self.assertRaisesRegex(ValueError, "multiProcessorCount"):
                    validate_plan(plan, device_properties=wrong_sm)

                wrong_attributes = dict(receiver["attributes"])
                wrong_attributes["l2CacheSize"] += 1
                with self.assertRaisesRegex(ValueError, "l2CacheSize"):
                    validate_plan(
                        plan,
                        device_properties=dict(
                            receiver, attributes=wrong_attributes,
                        ),
                    )


if __name__ == "__main__":
    unittest.main()
