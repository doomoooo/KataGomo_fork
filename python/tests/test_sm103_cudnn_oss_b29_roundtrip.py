import hashlib
import json
import pathlib
import subprocess
import sys
import tempfile
import unittest


PYTHON_DIR = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PYTHON_DIR))

import sm103_cudnn_oss_b29_roundtrip as variant  # noqa: E402


class Sm103CudnnOssB29RoundtripTests(unittest.TestCase):
    def test_manifest_has_exact_distinct_identity_and_one_factor(self) -> None:
        manifest = variant.validate_candidate_manifest(
            variant.build_candidate_manifest()
        )
        self.assertEqual(
            manifest["candidate_id"],
            "cudnn-fe-1_27-oss-dense-gemm-swiglu-proj-fp16-roundtrip-b29",
        )
        semantics = manifest["operation"]["numeric_semantics"]
        self.assertEqual(semantics["changed_factors"], ["projection FP16 round-trip"])
        self.assertIn("round-to-nearest-even", semantics["projection_boundary"])
        self.assertIn("unchanged", semantics["activation"])
        self.assertIn("AB12 is retained", semantics["ab12"])
        self.assertTrue(
            manifest["correctness"]["required_reference"].endswith(
                ")).half().float()"
            )
        )
        self.assertFalse(manifest["production_ready"])

    def test_manifest_revalidates_base_fields_and_full_derivative(self) -> None:
        manifest = variant.build_candidate_manifest()
        manifest["target"]["compile_target"] = "sm_100a"
        with self.assertRaisesRegex(
            variant.CudnnOssRoundtripError, "base field changed: target"
        ):
            variant.validate_candidate_manifest(manifest)

        manifest = variant.build_candidate_manifest()
        manifest["static_support"]["derivative"]["derivative_sha256"] = "0" * 64
        with self.assertRaisesRegex(
            variant.CudnnOssRoundtripError, "provenance changed"
        ):
            variant.validate_candidate_manifest(manifest)

        manifest = variant.build_candidate_manifest()
        manifest["unreviewed_identity_field"] = "drift"
        with self.assertRaisesRegex(
            variant.CudnnOssRoundtripError, "full Variant A manifest identity"
        ):
            variant.validate_candidate_manifest(manifest)

    def test_derivation_is_exact_and_preserves_fast_activation_and_ab12(self) -> None:
        derivative, evidence = variant.inspect_derivative()
        upstream_path = pathlib.Path(evidence.upstream_installed_path)
        upstream = upstream_path.read_bytes()
        self.assertEqual(hashlib.sha256(upstream).hexdigest(), variant.UPSTREAM_KERNEL_SHA256)
        self.assertEqual(hashlib.sha256(derivative).hexdigest(), evidence.derivative_sha256)
        text = derivative.decode("utf-8")
        self.assertEqual(text.count("acc_vec0_ab12 = acc_vec0.to(self.ab12_dtype)"), 1)
        self.assertEqual(text.count("acc_vec1_ab12 = acc_vec1.to(self.ab12_dtype)"), 1)
        self.assertEqual(text.count("cute.math.exp2(-1 * acc_vec1 * LOG2_E, True)"), 1)
        self.assertEqual(text.count("cute.arch.rcp_approx(res[i])"), 1)
        self.assertEqual(text.count("tRS_rAB12.store(acc_vec0)"), 1)
        self.assertFalse(evidence.site_packages_modified)
        self.assertEqual(evidence.upstream_license, "Apache-2.0")

    def test_derivation_rejects_any_upstream_byte_drift(self) -> None:
        _, evidence = variant.inspect_derivative()
        source = bytearray(pathlib.Path(evidence.upstream_installed_path).read_bytes())
        source[-1] ^= 1
        with self.assertRaisesRegex(variant.CudnnOssRoundtripError, "SHA-256 mismatch"):
            variant.derive_variant_a_source(bytes(source))

    def test_materialization_is_idempotent_and_never_mutates_upstream(self) -> None:
        _, evidence = variant.inspect_derivative()
        upstream_path = pathlib.Path(evidence.upstream_installed_path)
        before = hashlib.sha256(upstream_path.read_bytes()).hexdigest()
        with tempfile.TemporaryDirectory() as temporary:
            output = pathlib.Path(temporary)
            source, provenance, first = variant.materialize_derivative(output)
            source2, provenance2, second = variant.materialize_derivative(output)
            self.assertEqual((source, provenance), (source2, provenance2))
            self.assertEqual(first, second)
            self.assertEqual(
                json.loads(provenance.read_text()), first.to_dict()
            )
            source.write_text("drift", encoding="utf-8")
            with self.assertRaisesRegex(
                variant.CudnnOssRoundtripError, "refusing to overwrite"
            ):
                variant.materialize_derivative(output)
        after = hashlib.sha256(upstream_path.read_bytes()).hexdigest()
        self.assertEqual(before, after)

    def test_materialization_rejects_site_packages_descendants(self) -> None:
        _, evidence = variant.inspect_derivative()
        site_packages = pathlib.Path(evidence.upstream_installed_path)
        for _ in pathlib.PurePosixPath(
            variant.UPSTREAM_KERNEL_RELATIVE_PATH
        ).parts:
            site_packages = site_packages.parent
        with self.assertRaisesRegex(
            variant.CudnnOssRoundtripError, "outside the resolved site-packages"
        ):
            variant.materialize_derivative(site_packages / "katago-variant-a-test")

    def test_import_and_default_manifest_do_not_load_gpu_stacks(self) -> None:
        probe = (
            "import json, sys; import sm103_cudnn_oss_b29_roundtrip as v; "
            "v.validate_candidate_manifest(v.build_candidate_manifest()); "
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

    def test_aot_benchmark_requires_double_opt_in_before_gpu_import(self) -> None:
        with self.assertRaisesRegex(
            variant.CudnnOssRoundtripError, "explicit --allow-gpu"
        ):
            variant.benchmark_aot(
                allow_gpu=False,
                device=0,
                library_path=pathlib.Path("missing.so"),
            )

    def test_tight_correctness_gate_accepts_real_signal_and_small_errors(self) -> None:
        summary = {
            "reference_signal": {"max_abs": 0.25, "rms": 0.02},
            "output": {"max_abs_error": 1.0e-5, "rmse": 1.0e-6},
            "ab12_vs_fp32_gemm_half": {
                "max_abs_error": 2.0e-5,
                "rmse": 2.0e-6,
            },
        }
        checked = variant.validate_correctness_summary(summary)
        self.assertTrue(checked["passed"])
        self.assertTrue(all(checked["checks"].values()))

    def test_tight_correctness_gate_rejects_zero_signal(self) -> None:
        summary = {
            "reference_signal": {"max_abs": 0.0, "rms": 0.0},
            "output": {"max_abs_error": 0.0, "rmse": 0.0},
            "ab12_vs_fp32_gemm_half": {
                "max_abs_error": 0.0,
                "rmse": 0.0,
            },
        }
        with self.assertRaisesRegex(
            variant.CudnnOssRoundtripError, "reference_max_abs_signal"
        ):
            variant.validate_correctness_summary(summary)

    def test_tight_correctness_gate_rejects_wrong_output(self) -> None:
        summary = {
            "reference_signal": {"max_abs": 0.25, "rms": 0.02},
            "output": {"max_abs_error": 0.02, "rmse": 0.01},
            "ab12_vs_fp32_gemm_half": {
                "max_abs_error": 2.0e-5,
                "rmse": 2.0e-6,
            },
        }
        with self.assertRaisesRegex(
            variant.CudnnOssRoundtripError, "output_max_abs"
        ):
            variant.validate_correctness_summary(summary)

    def test_aot_authentication_rejects_old_or_mislabeled_library(self) -> None:
        derivative_source, derivative = variant.inspect_derivative()
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            library = root / "libkatago_sm103_b29_cudnn_swiglu.so"
            library.write_bytes(b"fixture-variant-a-library")
            source_path = root / variant.DERIVATIVE_FILENAME
            source_path.write_bytes(derivative_source)
            provenance_path = root / variant.DERIVATIVE_PROVENANCE_FILENAME
            provenance_path.write_text(
                json.dumps(derivative.to_dict()), encoding="utf-8"
            )

            def evidence(path: pathlib.Path) -> dict[str, object]:
                return {
                    "path": str(path.resolve()),
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "bytes": path.stat().st_size,
                }

            candidate_manifest = variant.build_candidate_manifest()
            manifest = {
                "kind": "katago-sm103-b29-cudnn-oss-aot-artifact",
                "candidate_id": variant.CANDIDATE_ID,
                "numeric_semantics_selector": "projection-fp16-roundtrip",
                "numeric_semantics": variant.numeric_semantics(),
                "compile_target": "sm_103a",
                "compile_options": ["--gpu-arch=sm_103a"],
                "kernel_manifest_provider": candidate_manifest["provider"],
                "derivative": {
                    "evidence": derivative.to_dict(),
                    "artifacts": {
                        "source": evidence(source_path),
                        "provenance": evidence(provenance_path),
                    },
                },
                "artifacts": {
                    "bridge_shared_library": evidence(library)
                },
                "launch_validation": {
                    "status": "passed",
                    "tight_correctness": {"passed": True},
                },
                "production_ready": False,
            }
            manifest_path = root / "aot-manifest.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            checked = variant.authenticate_aot_library(library)
            self.assertEqual(checked["library_path"], str(library.resolve()))

            manifest["candidate_id"] = variant.upstream_candidate.CANDIDATE_ID
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(
                variant.CudnnOssRoundtripError, "candidate_id"
            ):
                variant.authenticate_aot_library(library)

            manifest["candidate_id"] = variant.CANDIDATE_ID
            manifest["artifacts"]["bridge_shared_library"]["sha256"] = "0" * 64
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(
                variant.CudnnOssRoundtripError, "shared_library_sha256"
            ):
                variant.authenticate_aot_library(library)

            manifest["artifacts"]["bridge_shared_library"] = evidence(library)
            manifest["timing_record"] = {"path": "local-result.json"}
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(
                variant.CudnnOssRoundtripError, "no_timing_identity"
            ):
                variant.authenticate_aot_library(library)


if __name__ == "__main__":
    unittest.main()
