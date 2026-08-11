import hashlib
import json
import pathlib
import sys
import tempfile
import unittest
from types import SimpleNamespace


PYTHON_DIR = pathlib.Path(__file__).resolve().parents[1]
REPO = PYTHON_DIR.parent
sys.path.insert(0, str(PYTHON_DIR))

from cuda_tactic_workflow import (  # noqa: E402
    ARTIFACT_BUNDLE_KIND,
    ALL_FAMILIES,
    SM89_FAMILIES,
    SM89_DECISION_GROUPS,
    SM120_FAMILIES,
    SM120_DECISION_GROUPS,
    RESULT_KIND,
    SM89_RUNTIME_CONFIG_KEYS,
    SM120_RUNTIME_CONFIG_KEYS,
    _compile_metadata,
    build_plan,
    build_parser,
    canonical_architecture,
    canonical_refinement_rows,
    candidate_config,
    candidate_map,
    choose_history_stage_winner,
    command_certify,
    effective_candidate_map,
    effective_activation_markers,
    make_generation_plan,
    mark_superseded_refinement_winner,
    materialize_space,
    nvcc_arch_flag,
    official_fallback_overrides,
    positive_history_seed_candidate_ids,
    refinement_sweep_limit_can_resume,
    refinement_top_candidates,
    resolve_candidate_config_state,
    runtime_tactic_baseline,
    scan_command,
    sha256_file,
    stable_metric,
    stable_optimized_prescan_state,
    stable_prescan_candidate_ids,
    summarize_samples,
    tactic_overrides,
    validate_artifact_bundle,
    validate_plan,
    write_json,
)
from cuda_tactic_history import POSITIVE_HISTORY  # noqa: E402
from portable_fat_scan import select_tilelang_requests  # noqa: E402


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
                "history_final_joint": is_winner and family == SM89_FAMILIES[-1],
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


class CudaTacticWorkflowTests(unittest.TestCase):
    def test_refinement_resume_allows_only_monotonic_sweep_extension(self):
        self.assertTrue(refinement_sweep_limit_can_resume(3, 3))
        self.assertTrue(refinement_sweep_limit_can_resume(3, 5))
        self.assertFalse(refinement_sweep_limit_can_resume(5, 3))
        self.assertFalse(refinement_sweep_limit_can_resume(0, 5))
        self.assertFalse(refinement_sweep_limit_can_resume(True, 5))
        self.assertFalse(refinement_sweep_limit_can_resume(None, 5))

    def test_nvcc_arch_flag_uses_the_compiler_spelling(self):
        self.assertEqual(nvcc_arch_flag([8, 9]), "-arch=sm_89")
        self.assertEqual(nvcc_arch_flag([12, 0]), "-arch=sm_120")
        with self.assertRaisesRegex(ValueError, "compute capability"):
            nvcc_arch_flag("sm89")

    def test_architecture_family_sets_have_one_ordered_union(self):
        self.assertEqual(
            ALL_FAMILIES,
            tuple(dict.fromkeys((*SM89_FAMILIES, *SM120_FAMILIES))),
        )
        self.assertNotIn("exact_mask", ALL_FAMILIES)
        self.assertIn("postconv_bn", SM120_FAMILIES)
        self.assertIn("weight_sharing", SM120_FAMILIES)
        self.assertIn("wide_head", SM120_FAMILIES)
        self.assertIn("head_bn", SM120_FAMILIES)
        self.assertNotIn("swiglu", SM120_FAMILIES)
        self.assertNotIn("wide_ffn", SM120_FAMILIES)
        self.assertIn("dual_ffn", SM120_FAMILIES)
        self.assertIn("wide_projection", SM120_FAMILIES)
        self.assertIn("wide_projection", SM89_FAMILIES)
        self.assertNotIn("outer_projection", ALL_FAMILIES)
        self.assertEqual(len(SM89_DECISION_GROUPS), 10)
        self.assertEqual(len(SM120_DECISION_GROUPS), 10)
        self.assertEqual(
            tuple(family for group in SM89_DECISION_GROUPS for family in group),
            SM89_FAMILIES,
        )
        self.assertEqual(
            tuple(family for group in SM120_DECISION_GROUPS for family in group),
            SM120_FAMILIES,
        )

    def test_space_materializes_sm120_native_families_for_every_batch(self):
        space = materialize_space(
            "sm120", "rtx5080", 0, [4, 13, 19, 32], 2,
            device_properties={
                "compute_capability": [12, 0],
                "multiProcessorCount": 84,
            },
        )
        self.assertEqual(space["families"], list(SM120_FAMILIES))
        for batch in (4, 13, 19, 32):
            for family in SM120_FAMILIES:
                values = candidate_map(space, family, batch)
                self.assertIn(f"{family}-keep-incumbent", values)
        self.assertIn(
            "wide_qkv-m128-n128-k64-s2-cute-atom4x2-packed",
            candidate_map(space, "qkv_rope", 4),
        )
        self.assertIn(
            "fa4-b19-s361-h12-d32-tm128-tn64-s1-both16",
            candidate_map(space, "fa4", 19),
        )
        projections = candidate_map(space, "wide_projection", 4)[
            "wide-projections-s1-bundle"
        ]
        self.assertEqual(
            projections["activation_marker_keys"],
            {
                "SM120 backend: strided-batched QKV projection active": [
                    "cudaUseQKVStridedSm120",
                ],
                "SM120 backend: single-wide FFN projection active": [
                    "cudaUseWideFFNSingleGemm",
                ],
            },
        )
        self.assertEqual(
            candidate_map(space, "policy_p1", 19)["policy_p1-off"]
                ["supersedes"],
            ["wide_head"],
        )
        self.assertEqual(
            candidate_map(space, "head_bn", 19)["head_bn-off"]
                ["supersedes"],
            ["wide_head"],
        )
        self.assertNotIn("supersedes", projections)
        self.assertEqual(len(projections["activation_markers"]), 2)
        self.assertIn(
            "preconv-cutlass-warp64x64",
            candidate_map(space, "preconv", 32),
        )
        self.assertIn(
            "postconv-cutlass-bn-silu",
            candidate_map(space, "postconv_bn", 32),
        )
        bundle = candidate_map(space, "postconv_bn", 32)[
            "outer-projection-cutlass-warp64x64-bundle"
        ]
        self.assertNotIn("supersedes", bundle)
        self.assertIn(
            "SM120 backend: C768->C384 outer projection CUTLASS active, tactic=warp64x64",
            bundle["activation_marker_keys"],
        )
        self.assertEqual(len(bundle["activation_markers"]), 2)

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
            "fa4-d32-m64-n96-w4-pack0-both16",
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
        self.assertEqual(sm89["families"], list(SM89_FAMILIES))
        for family in SM89_FAMILIES:
            incumbent = candidate_map(sm89, family, 13)[
                f"{family}-keep-incumbent"
            ]
            self.assertEqual(candidate_config(family, incumbent), {})

    def test_sm89_fat_scan_consumes_only_the_unified_space_schema(self):
        space = materialize_space("sm89", "rtx4090", 0, [4], 2)
        requests = select_tilelang_requests(space, "dual_ffn", [4])
        self.assertIn(
            "dual_ffn-m128-n64-k32-s2-mb3-exp",
            {request["candidate_id"] for request in requests},
        )

        old_space = dict(space, kind="portable-tactic-search-space")
        with self.assertRaisesRegex(ValueError, "cuda-tactic-search-space"):
            select_tilelang_requests(old_space, "dual_ffn", [4])

    def test_positive_history_four_link_closure_covers_b4_through_b32(self):
        for architecture, gpu_class in (
            ("sm89", "rtx4090"), ("sm120", "rtx5080"),
        ):
            space = materialize_space(
                architecture, gpu_class, 0, range(4, 33), 2,
            )
            closure = space["positive_history_closure"]
            self.assertTrue(closure["complete"])
            self.assertEqual(closure["validated_batches"], list(range(4, 33)))
            self.assertEqual(
                closure["record_ids"],
                [entry["history_id"] for entry in POSITIVE_HISTORY[architecture]],
            )
            self.assertEqual(
                closure["links"],
                ["backend", "scan_candidate", "activation", "plan_apply"],
            )

    def test_runtime_baselines_are_complete_and_have_no_auto_winner(self):
        for architecture, keys in (
            ("sm89", SM89_RUNTIME_CONFIG_KEYS),
            ("sm120", SM120_RUNTIME_CONFIG_KEYS),
        ):
            baseline = runtime_tactic_baseline(architecture)
            self.assertEqual(set(baseline), set(keys))
            self.assertNotIn("auto", baseline.values())

        sm89 = materialize_space("sm89", "rtx4090", 0, [4], 2)
        rms4 = candidate_map(sm89, "rmsnorm", 4)["rmsnorm-warps4"]
        self.assertNotIn("cudaRMSNormRowsPerBlockSm89", "\n".join(
            rms4["activation_markers"]
        ))
        rms8 = candidate_map(sm89, "rmsnorm", 4)["rmsnorm-warps8"]
        self.assertTrue(any(marker.endswith("=8") for marker in rms8["activation_markers"]))

        sm120 = materialize_space("sm120", "rtx5080", 0, [4], 2)
        dual = candidate_map(sm120, "dual_ffn", 4)[
            "dual_ffn-cutlass-shared-a-m128-n64-k32-s3-swizzle2"
        ]
        self.assertNotIn("supersedes", dual)
        self.assertFalse(dual["config"]["cudaUseWideFFNSingleGemm"])
        ffn = candidate_map(sm120, "dual_ffn", 4)
        self.assertIn("wide_ffn-single-projection", ffn)
        self.assertIn("swiglu-on", ffn)
        self.assertFalse(ffn["swiglu-on"]["config"]["cudaUseWideFFNSingleGemm"])
        linear2 = candidate_map(sm120, "linear2", 4)[
            "linear2-m128-n128-k32-s3-cutlass"
        ]
        self.assertTrue(linear2["config"]["cudaUseFusedResidualGemmSm120"])

    def test_optimized_prescan_is_self_contained_and_custom_backend_active(self):
        self.assertEqual(canonical_architecture("sm120", "rtx5080"), "sm120")
        self.assertEqual(canonical_architecture(None, "rtx4090"), "sm89")
        for architecture, gpu_class in (
            ("sm89", "rtx4090"), ("sm120", "rtx5080"),
        ):
            overrides, selected, markers = stable_optimized_prescan_state(
                architecture, gpu_class, 1, 2, 19,
            )
            self.assertEqual(selected, stable_prescan_candidate_ids(
                architecture, gpu_class, 19,
            ))
            self.assertTrue(markers)
            self.assertEqual(
                overrides["cudaSm89Backend"], architecture == "sm89",
            )
            self.assertEqual(
                overrides["cudaSm89Forward"], architecture == "sm89",
            )
            self.assertEqual(
                overrides["cudaSm120Backend"], architecture == "sm120",
            )
            self.assertEqual(overrides["nnMaxBatchSize"], 19)
            self.assertEqual(overrides["numNNServerThreadsPerModel"], 2)

        # The official path remains available only as an explicit diagnostic
        # comparator; it is no longer wired into batch selection.
        diagnostic = official_fallback_overrides("sm120", 1, 2, 19)
        self.assertFalse(diagnostic["cudaSm120Backend"])

        parsed = build_parser().parse_args([
            "baseline-prescan", "--architecture", "sm120",
            "--gpu-class", "rtx5080", "--binary", "katago",
            "--config", "bench.cfg", "--model", "model.bin.gz",
            "--raw-dir", "raw", "--output", "prescan.json",
        ])
        self.assertEqual(parsed.top_batches, 3)

    def test_sm89_c384_pointwise_explicitly_reopens_the_fused_boundary(self):
        space = materialize_space("sm89", "rtx4090", 0, [4], 2)
        pointwise = candidate_map(space, "pointwise", 4)
        c384 = pointwise["pointwise-c384-vec8"]
        self.assertFalse(c384["config"]["cudaUsePostConvBNSiluSm89"])
        self.assertFalse(c384["config"]["cudaUseLinear2PostBNSiluSm89"])
        self.assertEqual(c384["supersedes"], ["postconv_bn"])
        self.assertEqual(
            c384["overrides_keys"], ["cudaUseLinear2PostBNSiluSm89"]
        )
        self.assertIn(
            "SM89 backend: runtime tactic active: "
            "cudaUseScaleBiasSiluVec8C384Sm89",
            c384["activation_markers"],
        )
        self.assertNotIn("supersedes", pointwise["pointwise-c768-vec8"])

        postconv = candidate_map(space, "postconv_bn", 4)[
            "postconv-cutlass-m128-n128-k32-w64-n64-s3-sw1-bn-silu"
        ]
        linear2 = candidate_map(space, "linear2", 4)[
            "linear2-cutlass-m128-n128-k32-w64-n64-s4-sw1-postbn"
        ]
        self.assertNotIn("cudaUseExactMask", repr(linear2))
        effective, superseded, applied, overridden = resolve_candidate_config_state({
            "linear2": linear2,
            "postconv_bn": postconv,
            "pointwise": c384,
        })
        self.assertIn("linear2", effective)
        self.assertNotIn("postconv_bn", effective)
        self.assertEqual(superseded["postconv_bn"], "pointwise")
        self.assertEqual(overridden["linear2"], {
            "cudaUseLinear2PostBNSiluSm89": "pointwise",
        })
        self.assertFalse(applied["cudaUsePostConvBNSiluSm89"])
        self.assertFalse(applied["cudaUseLinear2PostBNSiluSm89"])
        final_linear2_markers = effective_activation_markers(
            linear2, overridden["linear2"]
        )
        self.assertFalse(any(
            "cudaUseLinear2PostBNSiluSm89" in marker
            for marker in final_linear2_markers
        ))
        self.assertTrue(any(
            "cudaLinear2CutlassTacticSm89" in marker
            for marker in final_linear2_markers
        ))

    def test_sm89_grouped_rope_backend_covers_every_scanned_batch(self):
        source = (
            PYTHON_DIR.parent / "cpp/neuralnet/cudabackend_sm89_kernels.cu"
        ).read_text()
        for batch in range(2, 33):
            self.assertIn(
                f"KATAGO_SM89_ROPE_BATCH_GROUP_CASE({batch})", source
            )

    def test_sm120_qkv_rope_never_rewrites_the_fa_winner(self):
        space = materialize_space("sm120", "rtx5080", 0, [4], 2)
        qkv = candidate_map(space, "qkv_rope", 4)
        packed = qkv[
            "wide_qkv-m128-n128-k64-s2-cute-atom4x2-packed"
        ]
        self.assertEqual(packed["requires"], {"fa4.supports_packed": True})
        fa_keys = {
            "cudaUseFlashAttentionSm120",
            "cudaFlashAttentionSm120Accum",
            "cudaFlashAttentionAotTacticSm120",
        }
        for candidate in candidate_map(space, "qkv_rope", 4).values():
            self.assertFalse(fa_keys & set(candidate["config"]))

        rope = candidate_map(space, "qkv_rope", 4)[
            "qkv-rope-half2-with-"
            "wide_qkv-m128-n128-k64-s2-tilelang-planar"
        ]
        self.assertNotIn("supersedes", rope)
        self.assertEqual(rope["artifact_dependencies"], [{
            "family": "qkv_rope",
            "candidate_id":
                "wide_qkv-m128-n128-k64-s2-tilelang-planar",
        }])
        effective, superseded = effective_candidate_map({
            "fa4": candidate_map(space, "fa4", 4)[
                "fa4-b4-s361-h12-d32-tm128-tn96-s1-both16"
            ],
            "qkv_rope": rope,
        })
        self.assertEqual(list(effective), ["fa4", "qkv_rope"])
        self.assertEqual(superseded, {})
        _, _, applied, _ = resolve_candidate_config_state({
            "fa4": candidate_map(space, "fa4", 4)[
                "fa4-b4-s361-h12-d32-tm128-tn96-s1-both16"
            ],
            "qkv_rope": rope,
        })
        self.assertEqual(
            applied["cudaFlashAttentionAotTacticSm120"],
            "fa4-b4-s361-h12-d32-tm128-tn96-s1-both16",
        )

        backend = (
            REPO / "cpp/neuralnet/cudabackend_sm120.cpp"
        ).read_text()
        packed_gate = backend.split(
            "const bool packedAttentionReady", 1
        )[1].split("const bool useQKVRopeAot", 1)[0]
        self.assertIn("allowPackedOutput", packed_gate)
        self.assertIn("options.useFlashAttention", packed_gate)
        self.assertNotIn("flashAttentionAccum", packed_gate)

    def test_sm89_partial_key_ownership_and_wide_head_bundle_are_explicit(self):
        space = materialize_space("sm89", "rtx4090", 0, [4], 2)
        wide_projection = candidate_map(space, "wide_projection", 4)[
            "wide-projection-off"
        ]
        qkv_rope = candidate_map(space, "qkv_rope", 4)[
            "qkv-rope-gemm-epilogue"
        ]
        effective, superseded, applied, overridden = (
            resolve_candidate_config_state({
                "wide_projection": wide_projection,
                "qkv_rope": qkv_rope,
            })
        )
        self.assertEqual(list(effective), ["wide_projection", "qkv_rope"])
        self.assertEqual(superseded, {})
        self.assertTrue(applied["cudaUseWideQKV"])
        self.assertEqual(overridden, {
            "wide_projection": {"cudaUseWideQKV": "qkv_rope"},
        })

        policy = candidate_map(space, "policy_p1", 4)[
            "policy-p1-block96x1"
        ]
        wide_head = candidate_map(space, "wide_head", 4)["wide-head-on"]
        policy_off = candidate_map(space, "policy_p1", 4)[
            "policy-p1-disabled"
        ]
        self.assertFalse(policy_off["config"]["cudaUseWideHeadProjection"])
        self.assertIn(
            "cudaUseWideHeadProjection", policy_off["overrides_keys"]
        )
        effective, superseded, applied, overridden = resolve_candidate_config_state({
            "wide_head": wide_head,
            "policy_p1": policy,
        })
        self.assertEqual(list(effective), ["wide_head", "policy_p1"])
        self.assertEqual(superseded, {})
        self.assertEqual(applied["cudaPolicyP1RowsPerBlockSm89"], 1)
        self.assertEqual(overridden, {
            "wide_head": {
                "cudaPolicyP1RowsPerBlockSm89": "policy_p1",
            },
        })

    def test_full_board_invariant_is_not_a_search_component(self):
        for architecture, gpu_class in (
            ("sm89", "rtx4090"), ("sm120", "rtx5080")
        ):
            space = materialize_space(architecture, gpu_class, 0, [4], 2)
            self.assertNotIn("exact_mask", space["families"])
            self.assertNotIn("ExactMask", repr(space))
            self.assertNotIn("exact-mask", repr(space))

    def test_refinement_retains_superseded_catalog_winner_contract(self):
        first = [
            {
                "family": "postconv_bn", "batch": 12,
                "candidate_id": "postconv-a", "history_stage_winner": False,
                "nn_evals_per_sec_median": 100.0,
            },
            {
                "family": "postconv_bn", "batch": 12,
                "candidate_id": "postconv-b", "history_stage_winner": True,
                "nn_evals_per_sec_median": 101.0,
            },
        ]
        refined = [dict(first[1])]
        mark_superseded_refinement_winner(
            first, refined,
            family="postconv_bn", batch=12,
            candidate_id="postconv-b", superseding_family="wide_head",
            min_improvement_fraction=0.001,
        )
        canonical = canonical_refinement_rows(first, refined)
        winners = [row for row in canonical if row["history_stage_winner"]]
        self.assertEqual([row["candidate_id"] for row in winners], ["postconv-b"])
        self.assertEqual(winners[0]["history_superseded_by"], "wide_head")
        self.assertEqual(
            winners[0]["history_incumbent_candidate_id"], "postconv-b"
        )
        self.assertFalse(winners[0]["history_accepted_change"])
        self.assertEqual(
            winners[0]["history_improvement_fraction_vs_incumbent"], 0.0
        )

    def test_only_long_stable_values_are_final_metrics(self):
        stable = summarize_samples([100.0, 101.0], iterations=1000, warmup=50)
        short = summarize_samples([100.0, 101.0], iterations=999, warmup=50)
        noisy = summarize_samples([100.0, 130.0], iterations=1000, warmup=50)
        self.assertEqual(stable["measurement_kind"], "long_stable")
        self.assertEqual(stable_metric(stable), 100.5)
        self.assertIsNone(stable_metric(short))
        self.assertIsNone(stable_metric(noisy))

    def test_certification_binds_exact_batch_binary_overrides_and_reference(self):
        with tempfile.TemporaryDirectory() as directory_text:
            directory = pathlib.Path(directory_text)
            gate = directory / "gate.json"
            report = directory / "report.json"
            output = directory / "certified.json"
            model_sha = "1" * 64
            binary_sha = "2" * 64
            overrides = {"nnMaxBatchSize": 4}
            write_json(gate, {
                "schema": 1,
                "kind": RESULT_KIND,
                "identity": {"model_sha256": model_sha},
                "rows": [{
                    "history_long_gate": True,
                    "batch": 4,
                    "binary_sha256": binary_sha,
                    "overrides": overrides,
                }],
            })
            write_json(report, {
                "referenceSha256": "3" * 64,
                "candidateSha256": "4" * 64,
                "candidateBinarySha256": binary_sha,
                "candidateOverrides": overrides,
                "corpusSha256": "5" * 64,
                "modelSha256": model_sha,
                "exactBatch": 4,
                "candidateMaxBatchSize": 4,
                "candidateFixedBatchTailPadding": True,
                "referenceFixedBatchTailPadding": True,
                "inputAndTargetSectionsByteExact": True,
                "numRows": 8192,
                "policy": {
                    "top1VsReference": 1.0,
                    "p0lossCandidateWeighted": 1.0,
                    "p0lossReferenceWeighted": 1.0,
                    "probabilityRmse": 0.0,
                },
                "value": {"outcomeRmse": 0.0},
                "score": {"meanRmse": 0.0},
                "ownership": {"sigmoidRmse": 0.0},
                "requestGate": {
                    "policyProbability": {"maximumAbs": 0.0, "maximumRmse": 0.0},
                    "valueProbability": {"maximumAbs": 0.0, "maximumRmse": 0.0},
                    "scoreRaw": {"maximumAbs": 0.0, "maximumRmse": 0.0},
                    "ownershipProbability": {"maximumAbs": 0.0, "maximumRmse": 0.0},
                },
            })
            command_certify(SimpleNamespace(
                gate=str(gate), comparison=[f"4={report}"], output=str(output),
            ))
            certified = json.loads(output.read_text())
            self.assertEqual(
                certified["accuracy_certification"]["reference_sha256"],
                "3" * 64,
            )

            payload = json.loads(report.read_text())
            payload["requestGate"]["valueProbability"]["maximumAbs"] = 0.061
            write_json(report, payload)
            with self.assertRaisesRegex(
                ValueError, "request_value_probability_abs"
            ):
                command_certify(SimpleNamespace(
                    gate=str(gate), comparison=[f"4={report}"],
                    output=str(output), batches="4",
                ))
            payload["requestGate"]["valueProbability"]["maximumAbs"] = 0.0
            payload["requestGate"]["valueProbability"]["maximumRmse"] = 0.051
            write_json(report, payload)
            with self.assertRaisesRegex(
                ValueError, "request_value_probability_rmse"
            ):
                command_certify(SimpleNamespace(
                    gate=str(gate), comparison=[f"4={report}"],
                    output=str(output), batches="4",
                ))
            payload["requestGate"]["valueProbability"]["maximumRmse"] = 0.0
            write_json(report, payload)

            gate_payload = json.loads(gate.read_text())
            gate_payload["rows"].append({
                "history_long_gate": True,
                "batch": 5,
                "binary_sha256": binary_sha,
                "overrides": {"nnMaxBatchSize": 5},
            })
            write_json(gate, gate_payload)
            with self.assertRaisesRegex(ValueError, "missing --comparison for gate B5"):
                command_certify(SimpleNamespace(
                    gate=str(gate), comparison=[f"4={report}"],
                    output=str(output), batches=None,
                ))
            command_certify(SimpleNamespace(
                gate=str(gate), comparison=[f"4={report}"],
                output=str(output), batches="4",
            ))
            certified = json.loads(output.read_text())
            self.assertEqual(certified["accuracy_certification"]["batches"], [4])

            payload = json.loads(report.read_text())
            payload["exactBatch"] = 5
            write_json(report, payload)
            with self.assertRaisesRegex(ValueError, "not bound to exact B4"):
                command_certify(SimpleNamespace(
                    gate=str(gate), comparison=[f"4={report}"],
                    output=str(output), batches="4",
                ))

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

    def test_refinement_top_k_always_retests_the_incumbent(self):
        rows = [
            {
                "candidate_id": f"candidate-{index}",
                "status": "measured",
                "nn_evals_per_sec_median": float(100 - index),
            }
            for index in range(12)
        ]
        selected = refinement_top_candidates(rows, "candidate-11", 10)
        self.assertEqual(len(selected), 10)
        self.assertIn("candidate-11", [row["candidate_id"] for row in selected])
        self.assertNotIn("candidate-9", [row["candidate_id"] for row in selected])
        with self.assertRaisesRegex(ValueError, "positive"):
            refinement_top_candidates(rows, "candidate-11", 0)
        with self.assertRaisesRegex(ValueError, "absent"):
            refinement_top_candidates(rows, "missing", 10)

    def test_refinement_does_not_treat_empty_keep_as_a_concrete_alternative(self):
        rows = [
            {
                "candidate_id": "fa4-keep-incumbent",
                "status": "measured",
                "nn_evals_per_sec_median": 200.0,
            },
            {
                "candidate_id": "fa4-concrete",
                "status": "measured",
                "nn_evals_per_sec_median": 100.0,
            },
        ]
        selected = refinement_top_candidates(rows, "fa4-concrete", 2)
        self.assertEqual(
            [row["candidate_id"] for row in selected], ["fa4-concrete"]
        )

    def test_5080_positive_history_seed_is_materialized_for_every_batch(self):
        for batch in range(4, 33):
            space = materialize_space("sm120", "rtx5080", 0, [batch], 2)
            seed = positive_history_seed_candidate_ids(
                "sm120", "rtx5080", batch,
            )
            self.assertEqual(set(seed), set(SM120_FAMILIES))
            for family, candidate_id in seed.items():
                self.assertIn(candidate_id, candidate_map(space, family, batch))

    def test_4090_positive_history_seed_is_materialized_for_every_batch(self):
        for batch in range(4, 33):
            space = materialize_space("sm89", "rtx4090", 0, [batch], 2)
            seed = positive_history_seed_candidate_ids(
                "sm89", "rtx4090", batch,
            )
            self.assertEqual(set(seed), set(SM89_FAMILIES))
            for family, candidate_id in seed.items():
                self.assertIn(candidate_id, candidate_map(space, family, batch))

    def test_refinement_rows_replace_first_pass_without_duplicate_keys(self):
        first = [
            {"family": "fa4", "batch": 9, "candidate_id": "a", "refinement_pass": 1},
            {"family": "fa4", "batch": 9, "candidate_id": "b", "refinement_pass": 1},
        ]
        refined = [
            {"family": "fa4", "batch": 9, "candidate_id": "b", "refinement_pass": 2},
        ]
        rows = canonical_refinement_rows(first, refined)
        self.assertEqual(len(rows), 2)
        self.assertEqual([row["refinement_pass"] for row in rows], [1, 2])

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
            for family in SM89_FAMILIES:
                result_path = temporary / f"{family}.json"
                result = _make_result(
                    space_path, space, family, [1, 13], model_path, config_path,
                )
                if family == "fa4":
                    for row in result["rows"]:
                        if row["history_stage_winner"]:
                            row["history_incumbent_candidate_id"] = row["candidate_id"]
                            row["history_accepted_change"] = False
                            row["history_improvement_fraction_vs_incumbent"] = 0.0
                write_json(
                    result_path,
                    result,
                )
                result_paths.append(result_path)
            plan = build_plan(result_paths, space_path, SM89_FAMILIES, [1, 13])
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
            self.assertEqual(
                plan["families"]["wide_projection"]["batches"]["13"]
                    ["overridden_keys"],
                {
                    "cudaUseWideQKV": "qkv_rope",
                    "cudaUseWideFFN": "dual_ffn",
                },
            )
            self.assertTrue(
                plan["families"]["policy_p1"]["batches"]["13"]
                    ["effective"]
            )
            self.assertIsNone(
                plan["families"]["policy_p1"]["batches"]["13"]
                    ["superseded_by"]
            )
            checked = validate_plan(
                plan,
                space=space,
                space_path=space_path,
                model=model_path,
                config=config_path,
                families=SM89_FAMILIES,
            )
            self.assertTrue(checked["valid"])
            self.assertTrue(checked["production_ready"])
            plan["apply"]["per_batch_tactic_overrides"]["13"] = "tampered=true"
            with self.assertRaisesRegex(ValueError, "apply mapping differs"):
                validate_plan(
                    plan,
                    space=space,
                    space_path=space_path,
                    model=model_path,
                    config=config_path,
                    families=SM89_FAMILIES,
                )

    def test_plan_consumes_multi_family_result_with_empty_top_level_family(self):
        with tempfile.TemporaryDirectory() as temporary_text:
            temporary = pathlib.Path(temporary_text)
            space_path = temporary / "space.json"
            model_path = temporary / "model.bin"
            config_path = temporary / "base.cfg"
            model_path.write_bytes(b"model")
            config_path.write_text("nnMaxBatchSize = 1\n")
            space = materialize_space("sm89", "rtx4090", 0, [1], 2)
            write_json(space_path, space)
            combined = _make_result(
                space_path, space, SM89_FAMILIES[0], [1], model_path, config_path,
            )
            combined["rows"] = []
            for family in SM89_FAMILIES:
                result = _make_result(
                    space_path, space, family, [1], model_path, config_path,
                )
                combined["rows"].extend(result["rows"])
            # The discovery command emits one multi-family file with no
            # family-specific top-level value; row labels carry ownership.
            combined["family"] = ""
            result_path = temporary / "discovery.json"
            write_json(result_path, combined)
            plan = build_plan(
                [result_path], space_path, SM89_FAMILIES, [1],
            )
            self.assertTrue(plan["ready_for_scan_bypass"])

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
            for family in SM89_FAMILIES:
                for device, batches in ((0, [4]), (1, [13])):
                    result = _make_result(
                        space_path, space, family, batches, model_path, config_path,
                    )
                    result["device_ordinal"] = device
                    result_path = temporary / f"{family}-gpu{device}.json"
                    write_json(result_path, result)
                    result_paths.append(result_path)
            plan = build_plan(result_paths, space_path, SM89_FAMILIES, [4, 13])
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
            "fa4-d32-m64-n96-w4-pack0-both16"
        ]
        command, overrides = scan_command(
            space, "sm89", 0, 2, "fa4", 13, value,
            binary="./katago", config="base.cfg", model="model.bin",
            iterations=1000, warmup=50, extra_override="useFP16=true",
            runner=["env", "CUDA_LAUNCH_BLOCKING=0"],
        )
        self.assertEqual(command[0:2], ["env", "CUDA_LAUNCH_BLOCKING=0"])
        self.assertTrue(any("numNNServerThreadsPerModel=2" in item for item in command))
        self.assertEqual(
            overrides["cudaFlashAttentionTacticSm89"],
            "d32-m64-n96-w4-pack0-both16",
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
            for family in SM89_FAMILIES:
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
        for family in SM89_FAMILIES:
            for batch in (4, 13, 32):
                for candidate in candidate_map(space, family, batch).values():
                    keys = set(candidate_config(family, candidate))
                    self.assertLessEqual(keys, SM89_RUNTIME_CONFIG_KEYS)
                    if candidate.get("requires_artifact"):
                        artifact_families.add(family)
        self.assertEqual(artifact_families, {"dual_ffn", "linear2"})
        for candidate in candidate_map(space, "linear2", 4).values():
            if candidate["id"] not in {
                "linear2-fallback", "linear2-keep-incumbent",
            }:
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
                "positive_history_closure": space["positive_history_closure"],
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
