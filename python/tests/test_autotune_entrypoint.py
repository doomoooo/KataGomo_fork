import importlib.util
import json
import pathlib
import tempfile
import unittest


AUTOTUNE_PATH = (
    pathlib.Path(__file__).resolve().parents[2]
    / "final-migration"
    / "autotune"
    / "autotune.py"
)
SPEC = importlib.util.spec_from_file_location("final_migration_autotune", AUTOTUNE_PATH)
assert SPEC is not None and SPEC.loader is not None
AUTOTUNE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUTOTUNE)

BUILD_FOR_PLAN_PATH = AUTOTUNE_PATH.parent / "build_for_plan.py"
BUILD_FOR_PLAN_SPEC = importlib.util.spec_from_file_location(
    "final_migration_build_for_plan", BUILD_FOR_PLAN_PATH,
)
assert BUILD_FOR_PLAN_SPEC is not None and BUILD_FOR_PLAN_SPEC.loader is not None
BUILD_FOR_PLAN = importlib.util.module_from_spec(BUILD_FOR_PLAN_SPEC)
BUILD_FOR_PLAN_SPEC.loader.exec_module(BUILD_FOR_PLAN)


class AutotuneEntrypointTests(unittest.TestCase):
    def test_direct_clone_exposes_working_root_entrypoints(self) -> None:
        repo = AUTOTUNE_PATH.parents[2]
        setup = repo / "setup.sh"
        runner = repo / "run-autotune.sh"
        for path in (setup, runner):
            self.assertTrue(path.is_file(), path)
            self.assertTrue(path.stat().st_mode & 0o111, path)
        self.assertIn("prepare_source_runtime", setup.read_text())
        self.assertIn('payload/SHA256SUMS', setup.read_text())
        self.assertIn('final-migration/autotune/autotune.py', runner.read_text())
        self.assertIn('--prefix "${prefix}" --repo "${SCRIPT_DIR}"', runner.read_text())
        self.assertIn("KATAGO_LOCAL_ARCHIVE", setup.read_text())
        self.assertIn("--refresh-latest", setup.read_text())
        self.assertNotIn("bootstrap-ubuntu.sh", setup.read_text())
        self.assertNotIn("sudo", setup.read_text())
        installer = (
            repo / "final-migration" / "environment" / "install-python.sh"
        ).read_text()
        environment_setup = (
            repo / "final-migration" / "environment" / "setup.sh"
        ).read_text()
        audit = (
            repo / "final-migration" / "environment" /
            "audit-environment.sh"
        ).read_text()
        runtime_lock = (
            repo / "final-migration" / "environment" /
            "python-runtime.lock.sh"
        ).read_text()
        self.assertNotIn("/usr/bin/python", installer)
        self.assertNotIn("KATAGO_SYSTEM_PYTHON", installer)
        self.assertIn("python-runtime.lock.sh", installer)
        self.assertIn('"${python_root}/bin/python3" -m venv --copies', installer)
        self.assertIn("python -m ensurepip --version", installer)
        self.assertNotIn("bootstrap-ubuntu.sh", environment_setup)
        self.assertNotIn("python3-venv", audit)
        self.assertIn('KATAGO_PYTHON_RUNTIME_VERSION="3.12.13"', runtime_lock)
        self.assertIn("KATAGO_PYTHON_RUNTIME_SHA256", runtime_lock)

    def test_both_architectures_use_one_external_workflow(self) -> None:
        source = AUTOTUNE_PATH.read_text()
        self.assertIn("def workflow_discovery(", source)
        self.assertIn("def workflow_gate(", source)
        self.assertNotIn("python/sm120_tactic_search.py", source)
        self.assertNotIn("python/sm120_coordinate_search.py", source)
        self.assertNotIn("python/sm120_tactic_plan.py", source)
        self.assertNotIn("python/sm120_measure_joint_plan.py", source)
        self.assertNotIn("python/portable_tactic_workflow.py", source)

    def test_generated_activate_expands_ambient_paths(self) -> None:
        setup = (AUTOTUNE_PATH.parents[2] / "setup.sh").read_text()
        self.assertIn("':\\\"\\${PATH}\\\"", setup)
        self.assertIn("':\\\"\\${LD_LIBRARY_PATH:-}\\\"", setup)
        self.assertIn("':\\\"\\${CMAKE_PREFIX_PATH:-}\\\"", setup)
        self.assertNotIn("bin:\\${PATH}'", setup)

    def test_package_preserves_latest_corpus_name_and_reference_sidecar(self) -> None:
        package = (AUTOTUNE_PATH.parent / "package-autotune.sh").read_text()
        self.assertIn('$(basename -- "${CORPUS}")', package)
        self.assertIn('$(basename -- "${CORPUS_MANIFEST}")', package)
        self.assertIn("requires its immutable .json sidecar", package)
        self.assertNotIn(
            '"${asset_stage}/2026-08-01-19x19-8192-seed20260803-full19.npz"',
            package,
        )

    def test_package_exposes_versioned_plans_and_bilingual_readmes(self) -> None:
        package = (AUTOTUNE_PATH.parent / "package-autotune.sh").read_text()
        self.assertIn('final-migration/plans/.', package)
        self.assertIn('"${bundle}/plans/"', package)
        self.assertIn('"${bundle}/README.md"', package)
        self.assertIn('"${bundle}/README.zh-CN.md"', package)
        self.assertIn('"${bundle}/RUNTIME.md"', package)
        self.assertIn('"${bundle}/records/"', package)
        self.assertIn("find payload patches metadata plans records", package)
        self.assertIn('"${REPO_ROOT}/setup.sh"', package)
        self.assertIn('"${REPO_ROOT}/run-autotune.sh"', package)
        self.assertNotIn('"${SCRIPT_DIR}/setup.sh"', package)
        self.assertNotIn('"${SCRIPT_DIR}/run-autotune.sh"', package)
        self.assertIn('"${REPO_ROOT}/build-for-plan.sh"', package)
        self.assertIn('"${SCRIPT_DIR}/build_for_plan.py"', package)

    def test_build_only_path_uses_prepare_without_benchmark_phases(self) -> None:
        helper = (AUTOTUNE_PATH.parent / "build_for_plan.py").read_text()
        self.assertIn('"--phase", "prepare", "--full-batch-scan"', helper)
        self.assertIn("restrict_space_to_plan", helper)
        self.assertNotIn('"discovery"', helper)
        self.assertNotIn('"gate"', helper)
        self.assertNotIn('"accuracy"', helper)

    def test_plan_registry_uses_cuda_product_as_unique_key(self) -> None:
        def plan(name: str) -> dict[str, object]:
            return {
                "target": {
                    "cuda_device_capabilities_at_scan": [{"name": name}],
                },
            }

        selected = BUILD_FOR_PLAN.select_product_plan(
            [(pathlib.Path("5080.json"), plan("NVIDIA GeForce RTX 5080"))],
            "NVIDIA GeForce RTX 5080",
        )
        self.assertEqual(selected[0], pathlib.Path("5080.json"))
        with self.assertRaisesRegex(RuntimeError, "no bundled plan"):
            BUILD_FOR_PLAN.select_product_plan(
                [(pathlib.Path("5080.json"), plan("NVIDIA GeForce RTX 5080"))],
                "NVIDIA GeForce RTX 5090 D",
            )
        with self.assertRaisesRegex(RuntimeError, "multiple plan entries"):
            BUILD_FOR_PLAN.select_product_plan(
                [
                    (pathlib.Path("a.json"), plan("NVIDIA GeForce RTX 5080")),
                    (pathlib.Path("b.json"), plan("NVIDIA GeForce RTX 5080")),
                ],
                "NVIDIA GeForce RTX 5080",
            )

    def test_parse_batch_set(self) -> None:
        self.assertEqual(AUTOTUNE.parse_batch_set("4-6,8,6"), [4, 5, 6, 8])
        with self.assertRaises(ValueError):
            AUTOTUNE.parse_batch_set("6-4")

    def test_distributed_workflow_never_mutates_gpu_clocks_or_power(self) -> None:
        source = AUTOTUNE_PATH.read_text()
        self.assertNotIn("gpu-lock", source)
        self.assertNotIn("-lgc", source)
        self.assertNotIn("nvidia-smi", source)
        self.assertNotIn("performance_profile", source)

    def test_accuracy_replay_uses_fixed_batch_and_removes_candidate_dump(self) -> None:
        source = AUTOTUNE_PATH.read_text()
        replay = (
            AUTOTUNE_PATH.parents[2] / "cpp" / "command" / "replaynn.cpp"
        ).read_text()
        self.assertIn('"detect", "prescan", "prepare", "discovery", "gate",', source)
        self.assertIn('"--full-batch-scan", action="store_true"', source)
        self.assertIn('"--top-batches", type=int, default=3', source)
        self.assertIn("candidate.unlink()", source)
        self.assertIn("fastest long-gate plan", source)
        self.assertIn('"--batches", str(best_batch)', source)
        self.assertIn('out / "best-tactic-plan.json"', source)
        self.assertIn("const int batchSize = maxBatchSize;", replay)
        self.assertIn("b % realBatchSize", replay)
        self.assertIn('"fixedBatchTailPadding\\\":true', replay)

    def test_replaynn_argv_uses_katago_single_dash_options(self) -> None:
        command = AUTOTUNE.replaynn_command(
            pathlib.Path("katago"),
            pathlib.Path("bench.cfg"),
            {"nnMaxBatchSize": 19},
            pathlib.Path("model.bin.gz"),
            pathlib.Path("corpus.npz"),
            pathlib.Path("candidate.krnn"),
            19,
        )
        self.assertEqual(command[1], "replaynn")
        self.assertEqual(
            command[2::2],
            [
                "-config", "-override-config", "-model", "-corpus",
                "-output", "-batch-size",
            ],
        )
        self.assertFalse(any(item.startswith("--") for item in command))

    def test_accuracy_selects_peak_stable_throughput_and_lower_batch_on_tie(self) -> None:
        payload = {
            "rows": [
                {"batch": 4, "status": "measured", "history_long_gate": True,
                 "stable_long_nn_evals_per_sec": 100.0},
                {"batch": 5, "status": "measured", "history_long_gate": True,
                 "stable_long_nn_evals_per_sec": 120.0},
                {"batch": 6, "status": "measured", "history_long_gate": True,
                 "stable_long_nn_evals_per_sec": 120.0},
            ]
        }
        batch, row, metric = AUTOTUNE.select_best_long_gate_row(payload, [4, 5, 6])
        self.assertEqual(batch, 5)
        self.assertEqual(row["batch"], 5)
        self.assertEqual(metric, 120.0)

    def test_accuracy_requires_complete_stable_long_gate(self) -> None:
        with self.assertRaises(RuntimeError):
            AUTOTUNE.select_best_long_gate_row({"rows": []}, [4])
        with self.assertRaises(RuntimeError):
            AUTOTUNE.select_best_long_gate_row({
                "rows": [{
                    "batch": 4, "status": "measured", "history_long_gate": True,
                    "stable_long_nn_evals_per_sec": None,
                }]
            }, [4])

    def test_complete_manifest_requires_exact_domain(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "manifest.json"
            path.write_text(json.dumps({"complete": False, "batches": [4, 5]}))
            self.assertFalse(AUTOTUNE.complete_manifest_for_batches(path, "4-5"))

            path.write_text(json.dumps({"complete": True, "batches": [4, 5]}))
            self.assertTrue(AUTOTUNE.complete_manifest_for_batches(path, "4-5"))
            self.assertFalse(AUTOTUNE.complete_manifest_for_batches(path, "4-6"))

            path.write_text("not json")
            self.assertFalse(AUTOTUNE.complete_manifest_for_batches(path, "4-5"))

    def test_tilelang_root_is_read_from_complete_manifests(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory) / "tilelang"
            (root / "src/tl_templates/cuda").mkdir(parents=True)
            (root / "3rdparty/cutlass/include/cutlass").mkdir(parents=True)
            (root / "src/tl_templates/cuda/debug.h").touch()
            (root / "3rdparty/cutlass/include/cutlass/cutlass.h").touch()
            metadata = pathlib.Path(directory) / "metadata.json"
            metadata.write_text(json.dumps({
                "generation_environment": {"tilelang_root": str(root)},
            }))
            manifest = {"complete": True, "entries": [{"metadata": str(metadata)}]}
            self.assertEqual(AUTOTUNE.tilelang_root_from_manifests(manifest), root)

            with self.assertRaises(RuntimeError):
                AUTOTUNE.tilelang_root_from_manifests({**manifest, "complete": False})


if __name__ == "__main__":
    unittest.main()
