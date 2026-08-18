import dataclasses
import json
import pathlib
import sys
import unittest


PYTHON_DIR = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PYTHON_DIR))

import sm103_contract as contract  # noqa: E402


class Sm103DeviceContractTests(unittest.TestCase):
    def test_b300_identity_is_accelerated_sm103(self) -> None:
        self.assertEqual(contract.DEVICE.architecture, "sm103")
        self.assertEqual(contract.DEVICE.gpu_class, "b300")
        self.assertEqual(contract.DEVICE.compute_capability, (10, 3))
        self.assertEqual(contract.DEVICE.accelerated_target, "sm_103a")
        self.assertEqual(contract.DEVICE.isa_family, "sm100_tcgen05")
        self.assertIn("tcgen05", contract.DEVICE.required_features)
        self.assertIn("tmem", contract.DEVICE.required_features)

    def test_contract_is_immutable(self) -> None:
        with self.assertRaises(dataclasses.FrozenInstanceError):
            contract.DEVICE.accelerated_target = "sm_120a"

    def test_wrong_or_generic_target_is_rejected(self) -> None:
        for target in ("sm_100", "sm_100a", "sm_103", "sm_120a", None):
            with self.subTest(target=target):
                with self.assertRaises(contract.ContractValidationError):
                    contract.validate_target(target)
                with self.assertRaises(contract.ContractValidationError):
                    contract.build_manifest(11, target=target)


class B11ShapeContractTests(unittest.TestCase):
    def test_rows_are_361_times_fixed_batch(self) -> None:
        self.assertEqual(contract.SPATIAL_TOKENS, 361)
        self.assertEqual(contract.rows_for_batch(1), 361)
        self.assertEqual(contract.rows_for_batch(11), 3971)
        self.assertEqual(contract.rows_for_batch(32), 11552)

    def test_primary_b11_mnk_boundaries(self) -> None:
        self.assertEqual(
            contract.MODEL_NAME, "b11c768h12nbt3tflrs-fson-silu"
        )
        self.assertEqual(contract.TRUNK_CHANNELS, 768)
        self.assertEqual(contract.MODEL_CHANNELS, 384)
        shapes = {
            problem.boundary: problem.mnk
            for problem in contract.gemm_problems(11)
        }
        rows = 361 * 11
        self.assertEqual(
            shapes,
            {
                "wide_qkv": (rows, 1152, 384),
                "dual_ffn": (rows, 2304, 384),
                "linear2": (rows, 384, 1152),
                "outproj": (rows, 384, 384),
                "preconv": (rows, 384, 768),
                "postconv": (rows, 768, 384),
                "wide_head": (rows, 384, 768),
            },
        )

    def test_flash_problem_is_exact_dense_s361_h12_d32(self) -> None:
        problem = contract.flash_problem(11)
        self.assertEqual(
            (problem.sequence_length, problem.heads, problem.head_dim),
            (361, 12, 32),
        )
        self.assertFalse(problem.causal)
        self.assertEqual(problem.mask, "none")

    def test_invalid_batches_are_rejected_strictly(self) -> None:
        for batch in (0, 33, -1, True, 1.0, "11", None):
            with self.subTest(batch=batch):
                with self.assertRaises(contract.ContractValidationError):
                    contract.gemm_problems(batch)


class CandidateContractTests(unittest.TestCase):
    def test_flash_upstream_control_and_search_axes(self) -> None:
        control = contract.FLASH_UPSTREAM_CONTROL
        self.assertEqual(
            (
                control.tile_m,
                control.tile_n,
                control.q_stages,
                control.kv_stages,
                control.cta_count,
                control.cluster_shape,
            ),
            (128, 128, 2, 24, 1, (1, 1, 1)),
        )
        self.assertEqual(control.accumulator, "float32")
        self.assertEqual(control.p_residency, "tmem")
        self.assertFalse(control.use_tcgen05_ld_red)
        self.assertIn(control, contract.FLASH_FIRST_ROUND_CANDIDATES)

        candidates = contract.FLASH_FIRST_ROUND_CANDIDATES
        self.assertEqual({item.tile_m for item in candidates}, {64, 128})
        self.assertEqual({item.tile_n for item in candidates}, {64, 128})
        self.assertEqual({item.q_stages for item in candidates}, {1, 2})
        self.assertEqual(
            {item.kv_stages for item in candidates}, {3, 4, 8, 16, 24}
        )
        self.assertEqual({item.persistent for item in candidates}, {"static", "none"})

    def test_fp16_gemm_first_round_has_1cta_and_2cta(self) -> None:
        candidates = contract.GEMM_FP16_FIRST_ROUND_CANDIDATES
        self.assertEqual({item.cta_count for item in candidates}, {1, 2})
        self.assertEqual(
            {item.accumulator for item in candidates}, {"float16", "float32"}
        )
        self.assertEqual({item.persistent for item in candidates}, {"static", "dynamic"})
        self.assertEqual({item.tma_store for item in candidates}, {False, True})
        for candidate in candidates:
            contract.validate_candidate(candidate)

    def test_illegal_clusters_are_rejected(self) -> None:
        base = dict(
            candidate_id="illegal-cluster",
            tile_m=128,
            tile_n=128,
            tile_k=64,
            stages=3,
            accumulator="float32",
            persistent="static",
            tma_store=True,
        )
        with self.assertRaises(contract.ContractValidationError):
            contract.GemmCandidate(
                **base, cta_count=1, cluster_shape=(2, 1, 1)
            )
        with self.assertRaises(contract.ContractValidationError):
            contract.GemmCandidate(
                **base, cta_count=2, cluster_shape=(2, 2, 1)
            )
        with self.assertRaises(contract.ContractValidationError):
            contract.GemmCandidate(
                **base, cta_count=2, cluster_shape=(1, 1, 1)
            )
        with self.assertRaises(contract.ContractValidationError):
            contract.GemmCandidate(
                **base, cta_count=2, cluster_shape=[2, 1, 1]
            )

    def test_unknown_candidate_id_is_rejected(self) -> None:
        with self.assertRaises(contract.ContractValidationError):
            contract.flash_candidate("fa-not-in-first-round")
        with self.assertRaises(contract.ContractValidationError):
            contract.gemm_candidate("gemm-not-in-first-round")
        with self.assertRaises(contract.ContractValidationError):
            contract.validate_candidate(
                contract.FlashCandidate(
                    "fa-valid-shape-but-unregistered", 128, 128, 2, 24, "static"
                )
            )
        registered = contract.GEMM_FP16_FIRST_ROUND_CANDIDATES[0]
        with self.assertRaises(contract.ContractValidationError):
            contract.validate_candidate(
                dataclasses.replace(registered, accumulator="float16")
            )
        with self.assertRaises(contract.ContractValidationError):
            contract.validate_candidate({"candidate_id": "untyped"})


class Fp4AndManifestTests(unittest.TestCase):
    def test_fp4_ultra_is_separate_opt_in_k96_track(self) -> None:
        track = contract.FP4_ULTRA_TRACK
        self.assertEqual(track.accelerated_target, "sm_103a")
        self.assertEqual(track.instruction_k, 96)
        self.assertFalse(track.enabled_by_default)
        self.assertTrue(track.requires_activation_quantization)
        self.assertTrue(track.requires_precision_recertification)

        problems = contract.fp4_ultra_problems(11)
        self.assertEqual(
            tuple(problem.boundary for problem in problems), contract.BOUNDARY_NAMES
        )
        for problem in problems:
            with self.subTest(boundary=problem.boundary):
                self.assertEqual(problem.k % 96, 0)
                self.assertEqual(problem.source_activation_dtype, "float16")
                self.assertEqual(problem.operand_dtype, "nvfp4")
                self.assertEqual(problem.output_dtype, "float16")
                self.assertEqual(problem.accumulator, "float32")
                self.assertEqual(problem.instruction_k, 96)

    def test_manifest_is_json_serializable_and_keeps_tracks_separate(self) -> None:
        manifest = contract.build_manifest(11)
        encoded = json.dumps(manifest, sort_keys=True)
        decoded = json.loads(encoded)

        self.assertEqual(decoded["fixed_batch"], 11)
        self.assertEqual(decoded["rows"], 361 * 11)
        self.assertEqual(decoded["device"]["compute_capability"], [10, 3])
        self.assertEqual(decoded["device"]["accelerated_target"], "sm_103a")
        self.assertIn("first_round_candidates", decoded["fp16_gemm"])
        self.assertNotIn("fp4_ultra_k96", decoded["fp16_gemm"])
        self.assertIn("fp4_ultra_k96", decoded["experimental_tracks"])
        fp4_problem = decoded["experimental_tracks"]["fp4_ultra_k96"][
            "problems"
        ][0]
        self.assertEqual(fp4_problem["operand_dtype"], "nvfp4")
        self.assertNotIn("input_dtype", fp4_problem)
        self.assertEqual(json.loads(contract.manifest_json(11)), decoded)


if __name__ == "__main__":
    unittest.main()
