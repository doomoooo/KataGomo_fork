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
        self.assertNotIn("--refresh-latest", setup.read_text())
        self.assertIn("corpus.lock.sh", setup.read_text())
        self.assertIn("AUTOTUNE_CORPUS_SHA256", setup.read_text())
        self.assertIn(
            "https://github.com/lightvector/KataGo/releases/download/"
            "v1.17.1/${model_name}",
            setup.read_text(),
        )
        self.assertIn('model_source="${model_asset}"', setup.read_text())
        self.assertIn('--output "${model_source}.partial"', setup.read_text())
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
        self.assertIn("activate_venv", audit)
        build_matrix = (
            repo / "final-migration" / "environment" / "build-matrix.sh"
        ).read_text()
        self.assertIn("activate_venv", build_matrix)
        self.assertIn('cmake --fresh "${cmake_args[@]}"', build_matrix)
        self.assertIn('KATAGO_PYTHON_RUNTIME_VERSION="3.14.7"', runtime_lock)
        self.assertIn('KATAGO_PYTHON_RUNTIME_RELEASE="20260814"', runtime_lock)
        self.assertIn("KATAGO_PYTHON_RUNTIME_SHA256", runtime_lock)

    def test_triton_is_binary_only_and_never_source_built(self) -> None:
        repo = AUTOTUNE_PATH.parents[2]
        setup = (repo / "setup.sh").read_text()
        package = (AUTOTUNE_PATH.parent / "package-autotune.sh").read_text()
        source_lock = (AUTOTUNE_PATH.parent / "source-lock.tsv").read_text()
        source_catalog = (
            repo / "final-migration/environment/third-party.lock.tsv"
        ).read_text()
        binary_requirements = (
            AUTOTUNE_PATH.parent / "python-binary-requirements.txt"
        ).read_text()

        self.assertIn("triton==3.7.1", binary_requirements)
        self.assertNotIn("\ntriton\t", "\n" + source_lock)
        self.assertNotIn("\ntriton\t", "\n" + source_catalog)
        self.assertNotIn("build_source_wheel triton", setup)
        self.assertNotIn("sources/triton", setup)
        self.assertNotIn("toolchains.tar.gz", setup)
        self.assertNotIn("triton-llvm", package)
        self.assertNotIn("triton-json", package)
        self.assertNotIn("toolchains.tar.gz", package)
        self.assertNotIn(
            "cutlass flash-attention triton quack", package,
        )

    def test_sm103_activation_uses_only_the_managed_toolchain(self) -> None:
        repo = AUTOTUNE_PATH.parents[2]
        activation = (
            repo / "final-migration/environment/activate-sm103.sh"
        ).read_text()
        self.assertIn("activate_venv", activation)
        self.assertIn("activate_toolchain", activation)
        self.assertIn('TRITON_PTXAS_PATH="${KATAGO_CUDA_ROOT}/bin/ptxas"', activation)
        self.assertIn(
            'TRITON_PTXAS_BLACKWELL_PATH="${KATAGO_CUDA_ROOT}/bin/ptxas"',
            activation,
        )
        self.assertIn('CUTE_DSL_ARCH="sm_103a"', activation)
        self.assertIn(
            'CUTE_DSL_PTXAS_PATH="${KATAGO_CUDA_ROOT}/bin/ptxas"', activation
        )
        self.assertIn('FLASHINFER_CUDA_ARCH_LIST="10.3a"', activation)
        self.assertIn("FLASHINFER_NO_DOWNLOAD=1", activation)
        self.assertNotIn("/usr/local/cuda", activation)

        flashinfer_smoke = (
            repo / "final-migration/environment/smoke/flashinfer_sm103.py"
        ).read_text()
        self.assertIn("mma_tiler=(128, 128, 32)", flashinfer_smoke)
        self.assertIn("config.can_implement(dtype_width=16)", flashinfer_smoke)
        self.assertNotIn("cute.compile", flashinfer_smoke)

    def test_published_codegen_packages_are_not_cloned_or_source_built(self) -> None:
        repo = AUTOTUNE_PATH.parents[2]
        catalog = (
            repo / "final-migration/environment/third-party.lock.tsv"
        ).read_text()
        requirements = (
            repo / "final-migration/autotune/python-binary-requirements.txt"
        ).read_text()
        setup = (repo / "setup.sh").read_text()
        package = (AUTOTUNE_PATH.parent / "package-autotune.sh").read_text()

        for component in (
            "TileLang", "apache-tvm-ffi", "quack", "flashinfer-python",
            "nvidia-cudnn-frontend", "liger-kernel",
            "mslk",
        ):
            self.assertNotIn(f"\n{component}\tcore\t", "\n" + catalog)
            self.assertNotIn(f'build_source_wheel {component}', setup)
            self.assertNotIn(f'copy_source "{component}"', package)
        for requirement in (
            "tilelang===0.1.13", "apache-tvm-ffi==0.1.12",
            "quack-kernels==0.6.4", "z3-solver==4.15.4.0",
            "flashinfer-python==0.6.17",
            "nvidia-cudnn-frontend==1.27.0",
            "liger-kernel==0.8.1",
            "mslk==1.3.0+cu132",
        ):
            self.assertIn(requirement, requirements)
        self.assertIn("csrc/cutlass", catalog)
        self.assertIn("build_source_wheel flash_attn_4", setup)

    def test_environment_contract_excludes_unused_onnx_tooling(self) -> None:
        repo = AUTOTUNE_PATH.parents[2]
        checker = (
            repo / "final-migration/environment/check-python-environment.py"
        ).read_text()
        bootstrap = (
            repo / "final-migration/environment/python-bootstrap-requirements.txt"
        ).read_text().splitlines()
        audit = (
            repo / "final-migration/environment/audit-environment.sh"
        ).read_text()

        for unused in ("onnx", "onnx2torch", "pillow"):
            self.assertNotIn(f'"{unused}"', checker)
            self.assertNotIn(unused, bootstrap)
        self.assertNotIn('"onnx"', audit)
        self.assertNotIn('"onnx2torch"', audit)

    def test_source_setup_uses_locked_managed_native_and_codegen_stack(self) -> None:
        repo = AUTOTUNE_PATH.parents[2]
        environment = repo / "final-migration/environment"
        environment_setup = (environment / "setup.sh").read_text()
        common = (environment / "lib/common.sh").read_text()
        third_party_build = (environment / "build-third-party.sh").read_text()
        third_party_catalog = (environment / "third-party.lock.tsv").read_text()
        third_party_acquire = (environment / "acquire-third-party.sh").read_text()
        katago_build = (environment / "build-matrix.sh").read_text()
        verify = (environment / "verify-third-party.sh").read_text()
        audit = (environment / "audit-environment.sh").read_text()
        checker = (environment / "check-python-environment.py").read_text()
        deploy = (environment / "deploy-prebuilt.sh").read_text()
        root_setup = (repo / "setup.sh").read_text()
        package = (repo / "final-migration/autotune/package-autotune.sh").read_text()
        binary_requirements = (
            repo / "final-migration/autotune/python-binary-requirements.txt"
        ).read_text()

        self.assertLess(
            environment_setup.index("install-python.sh"),
            environment_setup.index("build-third-party.sh"),
        )
        self.assertNotIn("acquire-nvidia-toolchain", environment_setup)
        self.assertIn('nvidia/cu13', common)
        self.assertIn('nvidia/cudnn', common)
        self.assertIn("https://pypi.nvidia.com", common)
        self.assertIn("https://download.pytorch.org/whl/cu132", common)
        self.assertIn("KATAGO_PIP_CACHE_DIR", common)
        self.assertIn('export CUDA_HOME="${KATAGO_CUDA_ROOT}"', common)
        for source in (third_party_build, katago_build):
            self.assertIn("activate_toolchain", source)
            self.assertNotIn("/usr/local/cuda", source)
        self.assertIn('source "${SCRIPT_DIR}/activate-sm103.sh"', verify)
        self.assertNotIn("/usr/local/cuda", verify)
        self.assertIn('"-DKATAGO_TILELANG_ROOT=${KATAGO_TILELANG_ROOT}"', katago_build)
        self.assertIn('cuda_root="${KATAGO_CUDA_ROOT}"', root_setup)
        self.assertNotIn("payload/cuda-", root_setup)
        self.assertNotIn("payload/cudnn-", root_setup)
        self.assertNotIn("packing the CUDA", package)
        self.assertIn(
            "6c68991985ca8b09594ac6fd43abbfd5830c4140", third_party_catalog
        )
        self.assertIn(
            "0251105a2fb19d2957484b7f023cd8c115286ced", third_party_catalog
        )
        self.assertIn("expected pinned revision", third_party_acquire)
        self.assertIn("--no-build-isolation", third_party_build)
        for requirement in (
            "cuda-toolkit==13.3.1",
            "cuda-python==13.3.1",
            "cuda-core==1.0.1",
            "nvidia-cuda-nvcc==13.3.73",
            "nvidia-cuda-crt==13.3.73",
            "nvidia-nvvm==13.3.73",
            "nvidia-cudnn-cu13==9.25.0.15",
            "torch==2.13.0+cu132",
            "flashinfer-python==0.6.17",
            "nvidia-cudnn-frontend==1.27.0",
            "liger-kernel==0.8.1",
            "mslk==1.3.0+cu132",
        ):
            self.assertIn(requirement, binary_requirements)
        self.assertIn(
            'TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.9;10.3;12.0}"',
            third_party_build,
        )
        self.assertIn(
            'FLASH_ATTN_CUDA_ARCHS="${FLASH_ATTN_CUDA_ARCHS:-89;103;120}"',
            third_party_build,
        )
        self.assertIn("smoke_archs=(89 103 120)", verify)
        self.assertIn("610.43.02", audit)
        for metadata in (
            "native_cuda_toolkit_pypi=13.3.1",
            "native_cudnn_cuda13_pypi=9.25.0.15",
            "torch=2.13.0+cu132",
            "torch_wheel_cuda=13.2",
            "minimum_driver=610.43.02",
        ):
            self.assertIn(metadata, package)
        self.assertIn("katago-sm89-sm103-sm120-autotune-", package)
        self.assertIn(
            '"mixed-wheel-abi-vs-active-runtime", "==13.2.1", "13.3.1"',
            checker,
        )
        self.assertIn(
            '"mixed-wheel-abi-vs-active-runtime", "==9.20.0.48", "9.25.0.15"',
            checker,
        )
        self.assertIn("ALLOWED_METADATA_CONFLICT", checker)
        self.assertIn("load_exact_requirements", checker)
        self.assertIn("Python runtime drift", checker)
        self.assertIn("managed nvcc drift", checker)
        self.assertIn(
            '--requirements "${bundle}/metadata/python-binary-resolved.txt"',
            deploy,
        )
        self.assertNotIn("pypi_index=", audit)
        self.assertNotIn("pypi_extra_indexes=", audit)
        self.assertNotIn("libcudnn9-dev-cuda-13", audit)
        self.assertNotIn("libzip-dev libcudnn", audit)
        self.assertNotIn("nsys", audit)
        self.assertNotIn("ncu", audit)

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

    def test_package_preserves_frozen_corpus_name_and_reference_sidecar(self) -> None:
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
