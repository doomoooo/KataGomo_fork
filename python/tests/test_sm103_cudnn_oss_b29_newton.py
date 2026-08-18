import hashlib
import json
import pathlib
import subprocess
import sys
import tempfile
import unittest


PYTHON_DIR = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PYTHON_DIR))

import sm103_cudnn_oss_b29_newton as variant  # noqa: E402


class Sm103CudnnOssB29NewtonTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.parent, cls.parent_evidence = variant.variant_a.inspect_derivative()
        cls.derivative, cls.evidence = variant.inspect_derivative()
        cls.text = cls.derivative.decode("utf-8")

    def test_manifest_is_distinct_cumulative_and_nonproduction(self) -> None:
        manifest = variant.validate_candidate_manifest(
            variant.build_candidate_manifest()
        )
        self.assertEqual(
            manifest["candidate_id"],
            "cudnn-fe-1_27-oss-dense-gemm-swiglu-proj-fp16-roundtrip-"
            "newton1-b29",
        )
        semantics = manifest["operation"]["numeric_semantics"]
        self.assertEqual(
            semantics["parent_candidate_id"], variant.variant_a.CANDIDATE_ID
        )
        self.assertIn("fast exp2", semantics["activation"])
        self.assertIn("exactly one FP32 Newton", semantics["activation"])
        self.assertEqual(
            semantics["changed_factors_from_variant_a"],
            ["one FP32 Newton refinement after the unchanged rcp_approx seed"],
        )
        self.assertEqual(
            manifest["static_support"]["derivative"][
                "parent_derivative_sha256"
            ],
            variant.EXPECTED_VARIANT_A_DERIVATIVE_SHA256,
        )
        self.assertFalse(manifest["production_ready"])

    def test_parent_is_exact_audited_variant_a(self) -> None:
        self.assertEqual(
            hashlib.sha256(self.parent).hexdigest(),
            variant.EXPECTED_VARIANT_A_DERIVATIVE_SHA256,
        )
        self.assertEqual(
            self.parent_evidence.derivative_sha256,
            variant.EXPECTED_VARIANT_A_DERIVATIVE_SHA256,
        )
        self.assertEqual(
            self.evidence.parent_derivative_sha256,
            variant.EXPECTED_VARIANT_A_DERIVATIVE_SHA256,
        )

    def test_derivative_keeps_fast_math_seed_and_adds_one_newton_step(self) -> None:
        # Compile-shaping invariants: one native fast exponential and one
        # reciprocal approximation seed remain; only the refinement is new.
        self.assertEqual(
            self.text.count(
                "cute.math.exp2(-1 * acc_vec1 * LOG2_E, True)"
            ),
            1,
        )
        self.assertEqual(
            self.text.count("cute.arch.rcp_approx(denominator)"), 1
        )
        self.assertEqual(self.text.count("denominator = res[i]"), 1)
        self.assertEqual(
            self.text.count(
                "res[i] = reciprocal * (2.0 - denominator * reciprocal)"
            ),
            1,
        )
        self.assertEqual(
            self.text.count("# then apply exactly one FP32 Newton refinement."),
            1,
        )

    def test_derivative_has_no_precise_exp_or_division_path(self) -> None:
        for forbidden in (
            "cute.math.exp(-acc_vec1",
            "fastmath=False",
            "acc_vec1 / gate_denominator",
            "arith.divf",
            "gate = (acc_vec1 /",
        ):
            self.assertNotIn(forbidden, self.text)

    def test_nonmath_kernel_contract_is_unchanged(self) -> None:
        for unchanged in (
            "acc_vec0_ab12 = acc_vec0.to(self.ab12_dtype)",
            "acc_vec1_ab12 = acc_vec1.to(self.ab12_dtype)",
            "acc_vec_c = (acc_vec0 * gate).to(self.c_dtype)",
            "tRS_rAB12.store(acc_vec0)",
            "tRS_rAB12_1.store(acc_vec1)",
            "tRS_rC.store(acc_vec_c)",
        ):
            self.assertEqual(self.text.count(unchanged), 1)

    def test_derivation_is_exact_and_fails_closed_on_parent_drift(self) -> None:
        self.assertEqual(
            hashlib.sha256(self.derivative).hexdigest(),
            self.evidence.derivative_sha256,
        )
        self.assertEqual(
            variant.derive_variant_c_source(self.parent), self.derivative
        )
        drift = bytearray(self.parent)
        drift[-1] ^= 1
        with self.assertRaisesRegex(
            variant.CudnnOssNewtonError,
            "Variant-A derivative SHA-256 mismatch",
        ):
            variant.derive_variant_c_source(bytes(drift))

    def test_materialization_is_idempotent_and_outside_site_packages(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            first = variant.materialize_derivative(root)
            second = variant.materialize_derivative(root)
            self.assertEqual(first, second)
            self.assertEqual(json.loads(first[1].read_text()), self.evidence.to_dict())
        site_packages = variant._site_packages_root(self.evidence)
        with self.assertRaisesRegex(
            variant.CudnnOssNewtonError, "outside site-packages"
        ):
            variant.materialize_derivative(site_packages / "variant-c-test")

    def test_import_and_manifest_are_device_free(self) -> None:
        probe = (
            "import json,sys; import sm103_cudnn_oss_b29_newton as v; "
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


if __name__ == "__main__":
    unittest.main()
