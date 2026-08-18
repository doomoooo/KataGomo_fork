import json
import inspect
import pathlib
import subprocess
import sys
import tempfile
import unittest


PYTHON_DIR = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PYTHON_DIR))

import sm103_cudnn_oss_b29_export as exporter  # noqa: E402


class Sm103CudnnOssB29ExportTests(unittest.TestCase):
    def test_export_plan_is_native_c_abi_and_nonproduction(self) -> None:
        try:
            plan = exporter.build_export_plan()
        except ValueError as error:
            if "provider source" in str(error):
                self.skipTest("exact nvidia-cudnn-frontend wheel is not installed")
            raise
        self.assertEqual(plan["compile_target"], "sm_103a")
        self.assertEqual(plan["compile_options"], ["--gpu-arch=sm_103a"])
        self.assertFalse(plan["tvm_ffi"])
        self.assertFalse(plan["runtime_requires_python"])
        self.assertEqual(
            plan["c_abi"]["launch"], "katagoCudnnOssB29Launch"
        )
        self.assertFalse(plan["production_ready"])

    def test_export_requires_explicit_gpu_acknowledgement(self) -> None:
        with tempfile.TemporaryDirectory() as temporary, self.assertRaisesRegex(
            exporter.CudnnOssExportError, "explicit --allow-gpu"
        ):
            exporter.export_aot(
                allow_gpu=False,
                device=0,
                output_dir=pathlib.Path(temporary),
            )

    def test_variant_a_export_plan_is_distinct_and_fail_closed(self) -> None:
        plan = exporter.build_export_plan(
            numeric_semantics=exporter.VARIANT_A_NUMERIC_SEMANTICS
        )
        self.assertEqual(
            plan["candidate_id"],
            "cudnn-fe-1_27-oss-dense-gemm-swiglu-proj-fp16-roundtrip-b29",
        )
        self.assertEqual(
            plan["numeric_semantics_selector"],
            "projection-fp16-roundtrip",
        )
        self.assertEqual(
            plan["numeric_semantics"]["changed_factors"],
            ["projection FP16 round-trip"],
        )
        self.assertEqual(plan["derivative"]["upstream_license"], "Apache-2.0")
        self.assertIn(
            "dense_gemm_persistent_swiglu_variant_a.py",
            plan["expected_artifacts"],
        )
        self.assertFalse(plan["production_ready"])

    def test_variant_b_export_plan_is_cumulative_and_distinct(self) -> None:
        plan = exporter.build_export_plan(
            numeric_semantics=exporter.VARIANT_B_NUMERIC_SEMANTICS
        )
        self.assertEqual(
            plan["candidate_id"],
            "cudnn-fe-1_27-oss-dense-gemm-swiglu-proj-fp16-roundtrip-"
            "precise-math-b29",
        )
        self.assertEqual(
            plan["numeric_semantics_selector"],
            "projection-fp16-roundtrip-precise-math",
        )
        self.assertEqual(
            plan["derivative"]["parent_derivative_sha256"],
            "99247c64d70a5f0b14ff75c08ba8d28fde31f159248e1e86c934cec6152777bc",
        )
        self.assertIn(
            "dense_gemm_persistent_swiglu_variant_b.py",
            plan["expected_artifacts"],
        )
        self.assertFalse(plan["production_ready"])

    def test_variant_c_export_plan_keeps_fast_math_and_adds_newton1(self) -> None:
        plan = exporter.build_export_plan(
            numeric_semantics=exporter.VARIANT_C_NUMERIC_SEMANTICS
        )
        self.assertEqual(
            plan["candidate_id"],
            "cudnn-fe-1_27-oss-dense-gemm-swiglu-proj-fp16-roundtrip-"
            "newton1-b29",
        )
        self.assertEqual(
            plan["numeric_semantics_selector"],
            "projection-fp16-roundtrip-newton1",
        )
        self.assertIn("fast exp2", plan["numeric_semantics"]["activation"])
        self.assertIn(
            "exactly one FP32 Newton",
            plan["numeric_semantics"]["activation"],
        )
        self.assertEqual(
            plan["derivative"]["parent_derivative_sha256"],
            "99247c64d70a5f0b14ff75c08ba8d28fde31f159248e1e86c934cec6152777bc",
        )
        self.assertIn(
            "dense_gemm_persistent_swiglu_variant_c.py",
            plan["expected_artifacts"],
        )
        self.assertFalse(plan["production_ready"])

    def test_variant_launch_validation_reuses_tight_output_and_ab12_gate(self) -> None:
        source = inspect.getsource(exporter._validate_bridge_launch)
        self.assertIn("strengthen_probe_signal", source)
        self.assertIn("build_gpu_correctness_summary", source)
        self.assertIn("actual_ab12=ab12[:, :, 0]", source)
        self.assertIn("torch.cuda.synchronize(device)", source)
        self.assertLess(
            source.rfind("torch.cuda.synchronize(device)"),
            source.rfind("katagoCudnnOssB29Destroy(context)"),
        )

    def test_unknown_numeric_semantics_fails_before_gpu_import(self) -> None:
        with self.assertRaisesRegex(
            exporter.CudnnOssExportError, "unsupported numeric semantics"
        ):
            exporter.build_export_plan(numeric_semantics="drift")

    def test_existing_artifacts_are_not_overwritten_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            (root / f"{exporter.ARTIFACT_STEM}.o").write_bytes(b"existing")
            with self.assertRaisesRegex(
                exporter.CudnnOssExportError, "refusing to overwrite"
            ):
                exporter._ensure_export_target(root, force=False)

    def test_bridge_is_shape_exact(self) -> None:
        repo_root = PYTHON_DIR.parent
        header = (
            repo_root / "cpp/neuralnet/cudnn_oss_b29_aot_bridge.h"
        ).read_text()
        source = (
            repo_root / "cpp/neuralnet/cudnn_oss_b29_aot_bridge.cpp"
        ).read_text()
        self.assertIn("katagoCudnnOssB29Launch", header)
        self.assertIn("constexpr int B29Rows = 10469", source)
        self.assertIn("constexpr int InputChannels = 384", source)
        self.assertIn("constexpr int PackedChannels = 2304", source)
        self.assertIn("constexpr int OutputChannels = 1152", source)

    def test_import_and_default_plan_do_not_load_gpu_stacks(self) -> None:
        try:
            exporter.build_export_plan()
        except ValueError as error:
            if "provider source" in str(error):
                self.skipTest("exact nvidia-cudnn-frontend wheel is not installed")
            raise
        probe = (
            "import json, sys; import sm103_cudnn_oss_b29_export as e; "
            "e.build_export_plan(); "
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


if __name__ == "__main__":
    unittest.main()
