import ast
import hashlib
import json
import pathlib
import subprocess
import sys
import tempfile
import unittest


PYTHON_DIR = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PYTHON_DIR))

import sm103_cudnn_oss_b29_no_ab12 as variant  # noqa: E402
import sm103_cudnn_oss_b29_export as exporter  # noqa: E402


class Sm103CudnnOssB29NoAb12Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.parent, cls.parent_evidence = variant.parent_candidate.inspect_derivative()
        cls.derivative = variant.derive_no_ab12_source(cls.parent)
        cls.parent_text = cls.parent.decode("utf-8")
        cls.derivative_text = cls.derivative.decode("utf-8")
        cls.audit = variant.audit_derivative(cls.parent, cls.derivative)

    def test_parent_is_exact_finalized_roundtrip_fixture(self) -> None:
        self.assertEqual(
            hashlib.sha256(self.parent).hexdigest(),
            variant.PARENT_DERIVATIVE_SHA256,
        )
        self.assertEqual(
            self.parent_evidence.derivative_sha256,
            variant.PARENT_DERIVATIVE_SHA256,
        )
        self.assertEqual(
            self.parent_evidence.numeric_semantics_id,
            variant.NUMERIC_SEMANTICS_ID,
        )

    def test_every_transform_has_a_tight_one_count_context(self) -> None:
        counts = variant.patch_context_counts(self.parent)
        expected_names = {
            patch.name for patch in (*variant.REGION_PATCHES, *variant.EXACT_PATCHES)
        }
        self.assertEqual(set(counts), expected_names)
        self.assertTrue(counts)
        self.assertEqual(set(counts.values()), {1})
        self.assertEqual(len(counts), 24)

    def test_derivative_is_valid_python_and_deterministic(self) -> None:
        ast.parse(self.derivative_text)
        again = variant.derive_no_ab12_source(self.parent)
        self.assertEqual(again, self.derivative)
        self.assertEqual(
            hashlib.sha256(again).hexdigest(),
            variant.EXPECTED_DERIVATIVE_SHA256,
        )

    def test_external_ab12_abi_and_shape_remain_but_kernel_pointer_is_gone(self) -> None:
        tree = ast.parse(self.derivative_text)
        kernel_class = next(
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef)
            and node.name == "PersistentDenseGemmKernel"
        )
        methods = {
            node.name: node
            for node in kernel_class.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        call_args = [arg.arg for arg in methods["__call__"].args.args]
        kernel_args = [arg.arg for arg in methods["kernel"].args.args]
        grid_args = [arg.arg for arg in methods["_compute_grid"].args.args]
        self.assertIn("ab12", call_args)
        self.assertIn("ab12", grid_args)
        self.assertNotIn("ab12", kernel_args)
        self.assertNotIn("mAB12_mnl", kernel_args)
        self.assertEqual(
            self.audit["abi_shape_contract"],
            {
                "ab12_call_argument_count": 1,
                "ab12_grid_shape_use_count": 1,
                "ab12_device_kernel_argument_count": 0,
            },
        )

    def test_ab12_device_data_path_is_structurally_zero(self) -> None:
        self.assertEqual(
            set(self.audit["forbidden_device_data_path_counts"].values()), {0}
        )
        for token in variant.FORBIDDEN_DEVICE_DATA_PATH_TOKENS:
            self.assertNotIn(token, self.derivative_text)
        self.assertNotIn("storage.sAB12", self.derivative_text)
        self.assertNotIn("cpasync.prefetch_descriptor(tma_atom_ab12)", self.derivative_text)
        self.assertNotIn("tRS_rAB12.store", self.derivative_text)
        self.assertNotIn("cute.copy(\n                            tma_atom_ab12", self.derivative_text)
        self.assertEqual(self.derivative_text.count("cpasync.make_tiled_tma_atom("), 1)
        self.assertEqual(
            self.derivative_text.count("cpasync.prefetch_descriptor(tma_atom_c)"),
            1,
        )

    def test_tcgen05_tma_ab_and_mma_mainloop_are_byte_unchanged(self) -> None:
        protected = self.audit["protected_invariant_sha256"]
        self.assertEqual(
            protected["tcgen05_mma_mainloop"],
            "7866f527251a19a9b9e4ceb7d441944058523c09ec99e2d8efe296f3212e127b",
        )
        self.assertEqual(
            protected["tma_a_b_setup"],
            "4becf065068f67ddfe6174325672aecd7bd39a473793a46cf20dec0428d7c537",
        )
        for token, count in self.audit["protected_operation_counts"].items():
            self.assertEqual(self.parent_text.count(token), count)
            self.assertEqual(self.derivative_text.count(token), count)

    def test_roundtrip_and_c_math_are_literal_parent_bytes(self) -> None:
        self.assertEqual(self.parent_text.count(variant.ROUNDTRIP_MATH_BLOCK), 1)
        self.assertEqual(self.derivative_text.count(variant.ROUNDTRIP_MATH_BLOCK), 1)
        self.assertEqual(
            self.audit["roundtrip_math_block_sha256"],
            hashlib.sha256(variant.ROUNDTRIP_MATH_BLOCK.encode()).hexdigest(),
        )
        self.assertEqual(self.derivative_text.count("tRS_rC.store(acc_vec_c)"), 1)
        self.assertEqual(
            self.derivative_text.count(
                "cute.copy(\n                            tma_atom_c,"
            ),
            1,
        )

    def test_c_pipeline_keeps_required_shared_fences_and_barriers(self) -> None:
        # The parent has no AB12-only barrier: its two epilogue barriers protect
        # C as well.  Both source contexts therefore intentionally remain.
        parent_epilogue = variant._bounded_slice(
            self.parent_text,
            "# Specialized epilogue warps",
            "    def epilog_tmem_copy_and_partition(",
        )
        derivative_epilogue = variant._bounded_slice(
            self.derivative_text,
            "# Specialized epilogue warps",
            "    def epilog_tmem_copy_and_partition(",
        )
        barrier = "barrier_id=self.epilog_sync_bar_id"
        fence = 'cute.arch.fence_proxy(\n                        "async.shared"'
        self.assertEqual(parent_epilogue.count(barrier), 3)
        self.assertEqual(derivative_epilogue.count(barrier), 3)
        self.assertEqual(parent_epilogue.count(fence), 1)
        self.assertEqual(derivative_epilogue.count(fence), 1)
        self.assertEqual(
            self.derivative_text.count(
                "num_stages=4,  # preserve the parent's C TMA flight depth"
            ),
            1,
        )

    def test_resource_hypothesis_is_exact_and_explicitly_unvalidated(self) -> None:
        hypothesis = variant.resource_hypothesis()
        self.assertEqual(
            hypothesis["status"], "predicted_requires_ncu_nsys_validation"
        )
        self.assertEqual(
            hypothesis["global_memory"]["removed_output_bytes_per_stream_launch"],
            48_241_152,
        )
        self.assertEqual(
            hypothesis["global_memory"]["removed_output_bytes_per_dual_stream_round"],
            96_482_304,
        )
        self.assertEqual(
            hypothesis["shared_memory"],
            {
                "parent_dynamic_bytes_per_cta_ncu": 214_016,
                "removed_bytes_per_cta": 32_768,
                "expected_dynamic_bytes_per_cta": 181_248,
                "removed_output_stages": 4,
                "preserved_ab_input_stages": 5,
                "preserved_c_output_stages": 2,
            },
        )
        self.assertEqual(
            hypothesis["synchronization"]["removed_output_only_barriers"], 0
        )

    def test_provenance_chains_parent_upstream_and_patch_specs(self) -> None:
        derivative, evidence, audit = variant.inspect_derivative()
        self.assertEqual(derivative, self.derivative)
        self.assertEqual(evidence.parent_candidate_id, variant.PARENT_CANDIDATE_ID)
        self.assertEqual(
            evidence.parent_derivative_sha256,
            variant.PARENT_DERIVATIVE_SHA256,
        )
        self.assertEqual(
            evidence.parent_upstream_sha256,
            self.parent_evidence.upstream_sha256,
        )
        self.assertEqual(
            evidence.parent_patch_spec_sha256,
            self.parent_evidence.patch_spec_sha256,
        )
        self.assertEqual(evidence.removal_patch_spec_sha256, variant.PATCH_SPEC_SHA256)
        self.assertEqual(evidence.derivative_sha256, audit["derivative_sha256"])
        self.assertFalse(evidence.site_packages_modified)

    def test_manifest_and_export_selector_are_cumulative_fast_variant_a(self) -> None:
        manifest = variant.validate_candidate_manifest(
            variant.build_candidate_manifest()
        )
        self.assertEqual(manifest["candidate_id"], variant.CANDIDATE_ID)
        semantics = manifest["operation"]["numeric_semantics"]
        self.assertIn("fast exp2", semantics["activation"])
        self.assertIn("no AB12", semantics["ab12"])
        self.assertEqual(
            manifest["static_support"]["derivative"]["parent_derivative_sha256"],
            variant.PARENT_DERIVATIVE_SHA256,
        )
        plan = exporter.build_export_plan(
            numeric_semantics=exporter.NO_AB12_NUMERIC_SEMANTICS
        )
        self.assertEqual(plan["candidate_id"], variant.CANDIDATE_ID)
        self.assertEqual(
            plan["numeric_semantics_selector"], variant.NUMERIC_SEMANTICS_SELECTOR
        )
        self.assertIn(variant.DERIVATIVE_FILENAME, plan["expected_artifacts"])
        self.assertIn("AB12", plan["c_abi"]["caller_owned_buffers"])
        self.assertFalse(plan["production_ready"])

    def test_output_only_tight_gate_requires_untouched_abi_buffer(self) -> None:
        summary = {
            "reference_signal": {"max_abs": 0.2, "rms": 0.02},
            "output": {
                "max_abs_error": 1.0e-5,
                "rmse": 1.0e-6,
                "max_rel_error": 1.0e-4,
            },
            "ab12_untouched": True,
        }
        self.assertTrue(variant.validate_correctness_summary(summary)["passed"])
        summary["ab12_untouched"] = False
        with self.assertRaisesRegex(
            variant.NoAb12DerivativeError, "ab12_abi_only"
        ):
            variant.validate_correctness_summary(summary)

    def test_aot_benchmark_requires_explicit_gpu_acknowledgement(self) -> None:
        with self.assertRaisesRegex(
            variant.NoAb12DerivativeError, "explicit --allow-gpu"
        ):
            variant.benchmark_aot(
                allow_gpu=False,
                device=0,
                library_path=pathlib.Path("missing.so"),
            )

    def test_derivation_rejects_parent_byte_drift_before_any_patch(self) -> None:
        drifted = bytearray(self.parent)
        drifted[-1] ^= 1
        with self.assertRaisesRegex(
            variant.NoAb12DerivativeError, "parent SHA-256 mismatch"
        ):
            variant.derive_no_ab12_source(bytes(drifted))

    def test_materialization_is_idempotent_and_rejects_drift(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = pathlib.Path(temporary)
            source, provenance, first = variant.materialize_derivative(output)
            source2, provenance2, second = variant.materialize_derivative(output)
            self.assertEqual((source, provenance), (source2, provenance2))
            self.assertEqual(first, second)
            payload = json.loads(provenance.read_text())
            self.assertEqual(payload["evidence"], first.to_dict())
            self.assertEqual(payload["gpu_validation"], "not_run")
            self.assertFalse(payload["production_ready"])
            source.write_text("drift", encoding="utf-8")
            with self.assertRaisesRegex(
                variant.NoAb12DerivativeError, "refusing to overwrite"
            ):
                variant.materialize_derivative(output)

    def test_materialization_refuses_site_packages(self) -> None:
        parent_path = pathlib.Path(self.parent_evidence.upstream_installed_path)
        site_packages = parent_path.resolve()
        for _ in pathlib.PurePosixPath(
            variant.parent_candidate.UPSTREAM_KERNEL_RELATIVE_PATH
        ).parts:
            site_packages = site_packages.parent
        with self.assertRaisesRegex(
            variant.NoAb12DerivativeError, "outside site-packages"
        ):
            variant.materialize_derivative(site_packages / "katago-no-ab12-test")

    def test_import_and_static_hypothesis_do_not_load_gpu_stacks(self) -> None:
        probe = (
            "import json, sys; import sm103_cudnn_oss_b29_no_ab12 as v; "
            "v.stage_hypothesis(); roots={'torch','cudnn','cutlass','cuda','triton'}; "
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


if __name__ == "__main__":
    unittest.main()
