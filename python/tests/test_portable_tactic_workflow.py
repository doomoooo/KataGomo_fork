import hashlib
import json
import pathlib
import sys
import tempfile
import unittest


PYTHON_DIR = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PYTHON_DIR))

from portable_tactic_workflow import (  # noqa: E402
    ARTIFACT_BUNDLE_KIND,
    FAMILIES,
    RESULT_KIND,
    SM89_RUNTIME_CONFIG_KEYS,
    _compile_metadata,
    build_plan,
    candidate_config,
    candidate_map,
    choose_history_stage_winner,
    make_generation_plan,
    materialize_space,
    scan_command,
    sha256_file,
    stable_metric,
    summarize_samples,
    tactic_overrides,
    validate_artifact_bundle,
    validate_plan,
    write_json,
)


def _make_result(space_path, space, family, batches, model_path, config_path):
    rows = []
    for batch in batches:
        values = list(candidate_map(space, family, batch).values())
        for index, candidate in enumerate(values):
            # Make the final candidate the accumulated-history winner while
            # keeping every row long/stable. This catches accidental
            # first-candidate/anchor selection.
            base = 100.0 + index * 3.0 + batch
            is_winner = index + 1 == len(values)
            incumbent_median = 100.25 + batch
            winner_median = base + 0.25
            rows.append({
                "family": family,
                "batch": batch,
                "candidate_id": candidate["id"],
                "candidate": candidate,
                "implementation": candidate.get("implementation"),
                "status": "measured",
                "finished_utc": "2026-08-07T00:00:00Z",
                "correctness": {"status": "passed"},
                "history_stage_winner": is_winner,
                "history_final_joint": is_winner and family == FAMILIES[-1],
                "history_incumbent_candidate_id": (
                    f"{family}-keep-incumbent" if is_winner else None
                ),
                "history_incumbent_nn_evals_per_sec": (
                    incumbent_median if is_winner else None
                ),
                "history_accepted_change": True if is_winner else None,
                "history_min_improvement_fraction": 0.001 if is_winner else None,
                "history_improvement_fraction_vs_incumbent": (
                    winner_median / incumbent_median - 1.0
                    if is_winner else None
                ),
                "history_base_overrides": "test-base",
                "history_accumulated_overrides": "test-final" if is_winner else None,
                **summarize_samples([base, base + 0.5], iterations=1000, warmup=50),
            })
    return {
        "schema": 1,
        "kind": RESULT_KIND,
        "architecture": space["architecture"],
        "gpu_class": space["gpu_class"],
        "device_ordinal": space["device_ordinal"],
        "streams": space["streams"],
        "identity": {
            "model_sha256": sha256_file(model_path),
            "config_sha256": sha256_file(config_path),
        },
        "cuda_device_capabilities": [{
            "ordinal": space["device_ordinal"],
            "computeCapabilityMajor": 8,
            "computeCapabilityMinor": 9,
            "multiProcessorCount": 128,
        }],
        "provenance": {"versions": {"test-package": "1.0"}},
        "rows": rows,
    }


class PortableTacticWorkflowTests(unittest.TestCase):
    def test_space_materializes_sm89_all_batches(self):
        sm89 = materialize_space(
            "sm89", "rtx4090", 0, [1, 13], 2,
            device_properties={
                "compute_capability": [8, 9],
                "multiProcessorCount": 128,
            },
        )
        self.assertEqual(sm89["compute_capability"], [8, 9])
        self.assertEqual(
            sm89["cuda_device_properties_at_space_generation"]
                ["multiProcessorCount"],
            128,
        )
        self.assertIn(
            "fa4-n96-both16",
            candidate_map(sm89, "fa4", 13),
        )
        self.assertIn(
            "dual_ffn-m128-n64-k32-s2-mb3-exp",
            candidate_map(sm89, "dual_ffn", 13),
        )
        self.assertIn(
            "qkv-rope-gemm-epilogue",
            candidate_map(sm89, "qkv_rope", 13),
        )
        self.assertEqual(sm89["families"], list(FAMILIES))
        for family in FAMILIES:
            incumbent = candidate_map(sm89, family, 13)[
                f"{family}-keep-incumbent"
            ]
            self.assertEqual(candidate_config(family, incumbent), {})

    def test_only_long_stable_values_are_final_metrics(self):
        stable = summarize_samples([100.0, 101.0], iterations=1000, warmup=50)
        short = summarize_samples([100.0, 101.0], iterations=999, warmup=50)
        noisy = summarize_samples([100.0, 130.0], iterations=1000, warmup=50)
        self.assertEqual(stable["measurement_kind"], "long_stable")
        self.assertEqual(stable_metric(stable), 100.5)
        self.assertIsNone(stable_metric(short))
        self.assertIsNone(stable_metric(noisy))

    def test_history_winner_retains_incumbent_for_tie_or_tiny_gain(self):
        rows = [
            {"candidate_id": "family-new", "score": 100.05},
            {"candidate_id": "family-keep-incumbent", "score": 100.0},
        ]
        winner, incumbent = choose_history_stage_winner(
            rows, "family-keep-incumbent", lambda row: row["score"], 0.001,
        )
        self.assertIs(winner, incumbent)
        rows[0]["score"] = 100.2
        winner, _ = choose_history_stage_winner(
            rows, "family-keep-incumbent", lambda row: row["score"], 0.001,
        )
        self.assertEqual(winner["candidate_id"], "family-new")

    def test_plan_selects_per_batch_long_stable_maximum(self):
        with tempfile.TemporaryDirectory() as temporary_text:
            temporary = pathlib.Path(temporary_text)
            space_path = temporary / "space.json"
            model_path = temporary / "model.bin"
            config_path = temporary / "base.cfg"
            model_path.write_bytes(b"model")
            config_path.write_text("nnMaxBatchSize = 13\n")
            space = materialize_space("sm89", "rtx4090", 0, [1, 13], 2)
            write_json(space_path, space)
            result_paths = []
            for family in FAMILIES:
                result_path = temporary / f"{family}.json"
                write_json(
                    result_path,
                    _make_result(space_path, space, family, [1, 13], model_path, config_path),
                )
                result_paths.append(result_path)
            plan = build_plan(result_paths, space_path, FAMILIES, [1, 13])
            self.assertTrue(plan["ready_for_scan_bypass"])
            self.assertTrue(plan["production_ready"])
            self.assertEqual(
                plan["target"]["cuda_device_capabilities_at_scan"][0]
                    ["multiProcessorCount"],
                128,
            )
            self.assertEqual(
                plan["families"]["fa4"]["batches"]["13"]["candidate_id"],
                list(candidate_map(space, "fa4", 13))[-1],
            )
            checked = validate_plan(
                plan,
                space=space,
                space_path=space_path,
                model=model_path,
                config=config_path,
                families=FAMILIES,
            )
            self.assertTrue(checked["valid"])
            self.assertTrue(checked["production_ready"])

    def test_plan_combines_batch_partitions_from_two_sm89_devices(self):
        with tempfile.TemporaryDirectory() as temporary_text:
            temporary = pathlib.Path(temporary_text)
            space_path = temporary / "space.json"
            model_path = temporary / "model.bin"
            config_path = temporary / "base.cfg"
            model_path.write_bytes(b"model")
            config_path.write_text("nnMaxBatchSize = 13\n")
            space = materialize_space("sm89", "rtx4090", 0, [4, 13], 2)
            write_json(space_path, space)
            result_paths = []
            for family in FAMILIES:
                for device, batches in ((0, [4]), (1, [13])):
                    result = _make_result(
                        space_path, space, family, batches, model_path, config_path,
                    )
                    result["device_ordinal"] = device
                    result_path = temporary / f"{family}-gpu{device}.json"
                    write_json(result_path, result)
                    result_paths.append(result_path)
            plan = build_plan(result_paths, space_path, FAMILIES, [4, 13])
            self.assertTrue(plan["ready_for_scan_bypass"])
            self.assertEqual(plan["target"]["device_ordinal_at_scan"], 0)
            self.assertEqual(plan["target"]["device_ordinals_at_scan"], [0, 1])

    def test_incomplete_or_short_scan_cannot_be_a_bypass_plan(self):
        with tempfile.TemporaryDirectory() as temporary_text:
            temporary = pathlib.Path(temporary_text)
            space_path = temporary / "space.json"
            model_path = temporary / "model.bin"
            config_path = temporary / "base.cfg"
            model_path.write_bytes(b"model")
            config_path.write_text("base\n")
            space = materialize_space("sm89", "rtx4090", 0, [1], 2)
            write_json(space_path, space)
            result = _make_result(space_path, space, "fa4", [1], model_path, config_path)
            result["rows"][0]["measurement_kind"] = "short_scan"
            result["rows"][0]["stable_long_nn_evals_per_sec"] = None
            result_path = temporary / "short.json"
            write_json(result_path, result)
            with self.assertRaisesRegex(ValueError, "coverage is incomplete"):
                build_plan([result_path], space_path, ["fa4"], [1])
            partial = build_plan(
                [result_path], space_path, ["fa4"], [1], allow_partial=True
            )
            self.assertFalse(partial["ready_for_scan_bypass"])

    def test_non_sm89_space_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary_text:
            temporary = pathlib.Path(temporary_text)
            with self.assertRaisesRegex(ValueError, "architecture must be"):
                materialize_space("sm90", "unsupported", 2, [1], 2)

    def test_scan_command_contains_external_stream_topology_and_tactic(self):
        space = materialize_space("sm89", "rtx4090", 0, [13], 2)
        value = candidate_map(space, "fa4", 13)[
            "fa4-n96-both16"
        ]
        command, overrides = scan_command(
            space, "sm89", 0, 2, "fa4", 13, value,
            binary="./katago", config="base.cfg", model="model.bin",
            iterations=1000, warmup=50, extra_override="useFP16=true",
            runner=["gpu-lock.sh", "--gpu", "0", "--"],
        )
        self.assertEqual(command[0:4], ["gpu-lock.sh", "--gpu", "0", "--"])
        self.assertTrue(any("numNNServerThreadsPerModel=2" in item for item in command))
        self.assertEqual(
            overrides["cudaUseFlashAttentionBoth16Sm89"], True
        )

    def test_generation_plan_has_no_batch_13_special_case(self):
        with tempfile.TemporaryDirectory() as temporary_text:
            temporary = pathlib.Path(temporary_text)
            space_path = temporary / "space.json"
            space = materialize_space("sm89", "rtx4090", 0, [4, 13, 32], 2)
            write_json(space_path, space)
            generated = make_generation_plan(space_path, phase="full")
            self.assertFalse(generated["batch_13_special_case"])
            self.assertTrue(generated["complete_history_coverage"])
            self.assertTrue(generated["eligible_for_whole_graph_scan"])
            for family in FAMILIES:
                if family == "l2":
                    continue
                counts = generated["coverage"][family]
                self.assertEqual(counts["4"], counts["13"])
                self.assertEqual(counts["13"], counts["32"])

            smoke = make_generation_plan(space_path, phase="seed")
            self.assertFalse(smoke["complete_history_coverage"])
            self.assertFalse(smoke["eligible_for_whole_graph_scan"])

    def test_every_candidate_uses_a_real_sm89_runtime_config(self):
        space = materialize_space("sm89", "rtx4090", 0, [4, 13, 32], 2)
        artifact_families = set()
        for family in FAMILIES:
            for batch in (4, 13, 32):
                for candidate in candidate_map(space, family, batch).values():
                    keys = set(candidate_config(family, candidate))
                    self.assertLessEqual(keys, SM89_RUNTIME_CONFIG_KEYS)
                    if candidate.get("requires_artifact"):
                        artifact_families.add(family)
        self.assertEqual(artifact_families, {"dual_ffn", "linear2"})
        for candidate in candidate_map(space, "linear2", 4).values():
            if candidate["id"] != "linear2-fallback":
                self.assertIs(tactic_overrides("linear2", candidate)["cudaUseFusedResidual"], True)

    def test_provenance_uses_the_actual_binary_build_directory(self):
        with tempfile.TemporaryDirectory() as temporary_text:
            root = pathlib.Path(temporary_text)
            build = root / "build-sm89-search"
            build.mkdir()
            binary = build / "katago"
            binary.write_bytes(b"binary")
            (build / "compile_commands.json").write_text("[]\n")
            (build / "CMakeCache.txt").write_text(
                "CMAKE_CUDA_ARCHITECTURES:STRING=89\n"
                "CUDNN_LIBRARY:FILEPATH=/opt/cudnn/libcudnn.so\n"
                "SM89_FLASH_ATTN_ROOT:PATH=/opt/flash-attention\n"
                "USE_BACKEND:STRING=CUDA\n"
            )
            metadata = _compile_metadata(root, binary)
            self.assertEqual(
                metadata["compile_commands_path"],
                str((build / "compile_commands.json").resolve()),
            )
            self.assertEqual(metadata["cmake_cache"]["CMAKE_CUDA_ARCHITECTURES"], "89")
            self.assertEqual(metadata["cmake_cache"]["CUDNN_LIBRARY"], "/opt/cudnn/libcudnn.so")

    def test_artifact_bundle_must_cover_and_hash_the_linked_binary(self):
        with tempfile.TemporaryDirectory() as temporary_text:
            temporary = pathlib.Path(temporary_text)
            space_path = temporary / "space.json"
            binary = temporary / "katago"
            bundle_path = temporary / "bundle.json"
            space = materialize_space("sm89", "rtx4090", 0, [4], 2)
            write_json(space_path, space)
            binary.write_bytes(b"linked binary")
            value = next(
                item for item in candidate_map(space, "dual_ffn", 4).values()
                if item.get("requires_artifact")
            )
            key = ("dual_ffn", 4, value["id"])
            bundle = {
                "schema": 1,
                "kind": ARTIFACT_BUNDLE_KIND,
                "complete_history_coverage": True,
                "space_sha256": sha256_file(space_path),
                "architecture": "sm89",
                "gpu_class": "rtx4090",
                "linked_binary_sha256": sha256_file(binary),
                "entries": [{
                    "family": key[0], "batch": key[1], "candidate_id": key[2],
                    "status": "linked", "source_sha256": "a" * 64,
                    "correctness": {"status": "passed"},
                }],
            }
            write_json(bundle_path, bundle)
            evidence, metadata = validate_artifact_bundle(
                bundle_path, space_path=space_path, space=space,
                binary=binary, required=[key],
            )
            self.assertEqual(evidence[key]["correctness"]["status"], "passed")
            self.assertEqual(metadata["linked_binary_sha256"], sha256_file(binary))
            with self.assertRaisesRegex(ValueError, "missing 1"):
                validate_artifact_bundle(
                    bundle_path, space_path=space_path, space=space,
                    binary=binary,
                    required=[key, ("linear2", 4, "missing")],
                )


if __name__ == "__main__":
    unittest.main()
