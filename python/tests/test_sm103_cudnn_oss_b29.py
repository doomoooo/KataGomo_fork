import json
import pathlib
import subprocess
import sys
import tempfile
import unittest


PYTHON_DIR = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PYTHON_DIR))

import sm103_cudnn_oss_b29 as candidate  # noqa: E402


def verified_provider() -> candidate.ProviderEvidence:
    sources = tuple(
        candidate.SourceEvidence(
            relative_path=relative_path,
            expected_sha256=expected,
            installed_path=f"/fixture/{relative_path}",
            actual_sha256=expected,
        )
        for relative_path, expected in candidate.SOURCE_IDENTITIES.items()
    )
    return candidate.ProviderEvidence(
        distribution=candidate.PROVIDER_DISTRIBUTION,
        expected_version=candidate.PROVIDER_VERSION,
        installed_version=candidate.PROVIDER_VERSION,
        sources=sources,
    )


class Sm103CudnnOssB29Tests(unittest.TestCase):
    def test_problem_is_exact_b29_fp16_dense_swiglu(self) -> None:
        problem = candidate.DenseSwiGLUProblem()
        self.assertEqual((problem.batch, problem.streams, problem.m), (29, 2, 10469))
        self.assertEqual((problem.k, problem.n_packed, problem.n_output), (384, 2304, 1152))
        self.assertEqual(problem.mma_tiler_mn, (128, 128))
        self.assertEqual(problem.cluster_shape_mn, (1, 1))
        self.assertEqual(problem.tensor_contract["a"]["stride"], [384, 1, 10469 * 384])
        self.assertEqual(problem.tensor_contract["b"]["stride"], [1, 2304, 2304 * 384])
        self.assertEqual(problem.tensor_contract["ab12"]["bytes"], 10469 * 2304 * 2)

    def test_problem_mutations_fail_closed(self) -> None:
        invalid = (
            {"batch": 28},
            {"streams": 1},
            {"m": 10468},
            {"k": 383},
            {"n_packed": 1152},
            {"dtype": "bfloat16"},
            {"mma_tiler_mn": (128, 64)},
            {"cluster_shape_mn": (2, 1)},
        )
        for kwargs in invalid:
            with self.subTest(kwargs=kwargs), self.assertRaises(
                candidate.CudnnOssCandidateError
            ):
                candidate.DenseSwiGLUProblem(**kwargs)

    def test_weight_pack_matches_cudnn_acc0_times_silu_acc1(self) -> None:
        self.assertEqual(candidate.packed_weight_row("linear_gate", 0), 0)
        self.assertEqual(candidate.packed_weight_row("linear1", 0), 32)
        self.assertEqual(candidate.packed_weight_row("linear_gate", 31), 31)
        self.assertEqual(candidate.packed_weight_row("linear1", 31), 63)
        self.assertEqual(candidate.packed_weight_row("linear_gate", 32), 64)
        self.assertEqual(candidate.packed_weight_row("linear1", 32), 96)
        self.assertEqual(
            candidate.packed_weight_row("linear1", 1151), 2303
        )

    def test_invalid_weight_pack_requests_fail_closed(self) -> None:
        for projection, channel in (
            ("gate", 0),
            ("linear1", -1),
            ("linear_gate", 1152),
            ("linear1", True),
        ):
            with self.subTest(projection=projection, channel=channel), self.assertRaises(
                candidate.CudnnOssCandidateError
            ):
                candidate.packed_weight_row(projection, channel)

    def test_manifest_aligns_central_anchor_and_stays_nonproduction(self) -> None:
        manifest = candidate.build_candidate_manifest(provider=verified_provider())
        checked = candidate.validate_candidate_manifest(manifest)
        self.assertEqual(checked["anchor"]["batch"], 29)
        self.assertEqual(checked["anchor"]["streams"], 2)
        self.assertEqual(checked["anchor"]["rows"], 10469)
        self.assertTrue(checked["anchor"]["batch_selection_fixed"])
        self.assertFalse(checked["anchor"]["production_ready"])
        self.assertEqual(checked["fixed_baseline_control"]["backend"], "tensorrt")
        self.assertEqual(
            checked["fixed_baseline_control"]["binary_sha256"],
            "883024dc8bbc02e7f6b05b0431034652931acc760b76e7fd455dc996af278612",
        )
        self.assertEqual(
            checked["fixed_baseline_control"]["nn_evals_per_sec_median"],
            6733.719141,
        )
        self.assertEqual(checked["fixed_baseline_control"]["sample_count"], 5)
        self.assertEqual(checked["target"]["compile_target"], "sm_103a")
        self.assertEqual(
            checked["operation"]["weight_pair_order"],
            ["linear_gate", "linear1"],
        )
        self.assertTrue(
            checked["static_support"]["eligible_for_isolated_gpu_probe"]
        )
        self.assertFalse(checked["cpp_bridge"]["integration_ready"])
        self.assertFalse(checked["production_ready"])

    def test_provider_source_mismatch_blocks_candidate(self) -> None:
        evidence = verified_provider()
        first = evidence.sources[0]
        bad_sources = (
            candidate.SourceEvidence(
                relative_path=first.relative_path,
                expected_sha256=first.expected_sha256,
                installed_path=first.installed_path,
                actual_sha256="0" * 64,
            ),
            *evidence.sources[1:],
        )
        bad = candidate.ProviderEvidence(
            distribution=evidence.distribution,
            expected_version=evidence.expected_version,
            installed_version=evidence.installed_version,
            sources=bad_sources,
            errors=("provider source identity changed",),
        )
        manifest = candidate.build_candidate_manifest(provider=bad)
        self.assertEqual(manifest["static_support"]["status"], "blocked")
        with self.assertRaisesRegex(
            candidate.CudnnOssCandidateError, "provider source"
        ):
            candidate.validate_candidate_manifest(manifest)

    def test_fixed_baseline_mutation_fails_closed(self) -> None:
        manifest = candidate.build_candidate_manifest(provider=verified_provider())
        manifest["fixed_baseline_control"]["binary_sha256"] = "0" * 64
        with self.assertRaisesRegex(
            candidate.CudnnOssCandidateError, "baseline identity"
        ):
            candidate.validate_candidate_manifest(manifest)

    def test_checked_in_wheel_matches_audited_sources_when_installed(self) -> None:
        evidence = candidate.inspect_installed_provider()
        if evidence.installed_version is None:
            self.skipTest("nvidia-cudnn-frontend is not installed")
        self.assertTrue(evidence.verified, evidence.errors)

    def test_default_cli_is_cpu_only_and_writes_manifest(self) -> None:
        if candidate.inspect_installed_provider().installed_version is None:
            self.skipTest("nvidia-cudnn-frontend is not installed")
        with tempfile.TemporaryDirectory() as temporary:
            output = pathlib.Path(temporary) / "candidate.json"
            probe = (
                "import json, runpy, sys; "
                f"sys.argv=['sm103_cudnn_oss_b29.py','--output',{str(output)!r}]; "
                f"runpy.run_path({str(PYTHON_DIR / 'sm103_cudnn_oss_b29.py')!r}, run_name='__main__')"
            )
            completed = subprocess.run(
                [sys.executable, "-c", probe],
                cwd=PYTHON_DIR,
                check=True,
                text=True,
                capture_output=True,
            )
            payload = json.loads(completed.stdout)
            self.assertEqual(json.loads(output.read_text()), payload)
            self.assertFalse(payload["correctness"]["gpu_execution"])
            self.assertFalse(payload["production_ready"])
            self.assertNotIn("torch", completed.stderr)

    def test_import_does_not_load_gpu_python_stacks(self) -> None:
        probe = (
            "import json, sys; import sm103_cudnn_oss_b29; "
            "roots={'torch','cudnn','cutlass','cuda','triton'}; "
            "print(json.dumps(sorted(name for name in sys.modules "
            "if name.split('.')[0] in roots)))"
        )
        completed = subprocess.run(
            [sys.executable, "-c", probe],
            cwd=PYTHON_DIR,
            check=True,
            text=True,
            capture_output=True,
        )
        self.assertEqual(json.loads(completed.stdout), [])

    def test_gpu_probe_requires_double_opt_in_before_import(self) -> None:
        with self.assertRaisesRegex(
            candidate.CudnnOssCandidateError, "explicit --allow-gpu"
        ):
            candidate.run_gpu_probe(allow_gpu=False, device=0)

    def test_gpu_benchmark_requires_double_opt_in_before_import(self) -> None:
        with self.assertRaisesRegex(
            candidate.CudnnOssCandidateError, "explicit --allow-gpu"
        ):
            candidate.benchmark_gpu_candidate(allow_gpu=False, device=0)


if __name__ == "__main__":
    unittest.main()
