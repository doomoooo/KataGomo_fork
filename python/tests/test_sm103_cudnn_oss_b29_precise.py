import hashlib
import json
import pathlib
import subprocess
import sys
import tempfile
import unittest


PYTHON_DIR = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PYTHON_DIR))

import sm103_cudnn_oss_b29_precise as variant  # noqa: E402


class Sm103CudnnOssB29PreciseTests(unittest.TestCase):
    def test_manifest_is_distinct_cumulative_and_nonproduction(self) -> None:
        manifest = variant.validate_candidate_manifest(
            variant.build_candidate_manifest()
        )
        self.assertEqual(
            manifest["candidate_id"],
            "cudnn-fe-1_27-oss-dense-gemm-swiglu-proj-fp16-roundtrip-"
            "precise-math-b29",
        )
        semantics = manifest["operation"]["numeric_semantics"]
        self.assertEqual(
            semantics["parent_candidate_id"], variant.variant_a.CANDIDATE_ID
        )
        self.assertIn("non-fast cute.math.exp", semantics["activation"])
        self.assertEqual(
            semantics["changed_factors_from_variant_a"],
            [
                "fast exp2 -> non-fast exp",
                "rcp_approx multiply -> precise FP32 divide",
            ],
        )
        derivative = manifest["static_support"]["derivative"]
        self.assertEqual(
            derivative["parent_derivative_sha256"],
            variant.EXPECTED_VARIANT_A_DERIVATIVE_SHA256,
        )
        self.assertFalse(manifest["production_ready"])

    def test_derivation_changes_only_exact_variant_a_math_context(self) -> None:
        parent_source, parent = variant.variant_a.inspect_derivative()
        derivative, evidence = variant.inspect_derivative()
        self.assertEqual(
            hashlib.sha256(parent_source).hexdigest(),
            variant.EXPECTED_VARIANT_A_DERIVATIVE_SHA256,
        )
        self.assertEqual(parent.derivative_sha256, evidence.parent_derivative_sha256)
        self.assertEqual(
            hashlib.sha256(derivative).hexdigest(), evidence.derivative_sha256
        )
        text = derivative.decode("utf-8")
        self.assertEqual(
            text.count("cute.math.exp(-acc_vec1, fastmath=False)"), 1
        )
        self.assertEqual(
            text.count("gate = (acc_vec1 / gate_denominator).to(self.acc_dtype)"),
            1,
        )
        self.assertNotIn("cute.arch.rcp_approx(res[i])", text)
        self.assertNotIn(
            "cute.math.exp2(-1 * acc_vec1 * LOG2_E, True)", text
        )
        for unchanged in (
            "acc_vec0_ab12 = acc_vec0.to(self.ab12_dtype)",
            "acc_vec1_ab12 = acc_vec1.to(self.ab12_dtype)",
            "acc_vec_c = (acc_vec0 * gate).to(self.c_dtype)",
            "tRS_rAB12.store(acc_vec0)",
            "tRS_rAB12_1.store(acc_vec1)",
            "tRS_rC.store(acc_vec_c)",
        ):
            self.assertEqual(text.count(unchanged), 1)

    def test_derivation_fails_closed_on_parent_drift(self) -> None:
        parent, _ = variant.variant_a.inspect_derivative()
        drift = bytearray(parent)
        drift[-1] ^= 1
        with self.assertRaisesRegex(
            variant.CudnnOssPreciseError, "Variant-A derivative SHA-256 mismatch"
        ):
            variant.derive_variant_b_source(bytes(drift))

    def test_full_manifest_rejects_unknown_identity(self) -> None:
        manifest = variant.build_candidate_manifest()
        manifest["unreviewed"] = "drift"
        with self.assertRaisesRegex(
            variant.CudnnOssPreciseError, "full Variant-B manifest identity"
        ):
            variant.validate_candidate_manifest(manifest)

    def test_materialization_is_idempotent_and_outside_site_packages(self) -> None:
        _, evidence = variant.inspect_derivative()
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            first = variant.materialize_derivative(root)
            second = variant.materialize_derivative(root)
            self.assertEqual(first, second)
            self.assertEqual(
                json.loads(first[1].read_text()), evidence.to_dict()
            )
        site_packages = variant._site_packages_root(evidence)
        with self.assertRaisesRegex(
            variant.CudnnOssPreciseError, "outside site-packages"
        ):
            variant.materialize_derivative(site_packages / "variant-b-test")

    def test_import_and_manifest_are_device_free(self) -> None:
        probe = (
            "import json,sys; import sm103_cudnn_oss_b29_precise as v; "
            "v.validate_candidate_manifest(v.build_candidate_manifest()); "
            "roots={'torch','cudnn','cutlass','cuda','triton'}; "
            "print(json.dumps(sorted(n for n in sys.modules "
            "if n.split('.')[0] in roots)))"
        )
        completed = subprocess.run(
            [sys.executable, "-c", probe],
            cwd=PYTHON_DIR,
            check=True,
            text=True,
            capture_output=True,
        )
        self.assertEqual(json.loads(completed.stdout), [])

    def test_benchmark_requires_explicit_gpu_acknowledgement(self) -> None:
        with self.assertRaisesRegex(
            variant.CudnnOssPreciseError, "explicit --allow-gpu"
        ):
            variant.benchmark_aot(
                allow_gpu=False,
                device=0,
                library_path=pathlib.Path("missing.so"),
            )


if __name__ == "__main__":
    unittest.main()
