from pathlib import Path
import unittest


REPO = Path(__file__).resolve().parents[2]


class Sm103ProjectionGemmLtContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.header = (
            REPO / "cpp/neuralnet/cudabackend_sm120.h"
        ).read_text()
        cls.source = (
            REPO / "cpp/neuralnet/cudabackend_sm120.cpp"
        ).read_text()
        cls.core = (REPO / "cpp/neuralnet/cudabackend.cpp").read_text()
        begin = cls.source.index("bool Sm120Model::matMulLt(")
        end = cls.source.index("bool Sm120Model::conv1x1(", begin)
        cls.matmul = cls.source[begin:end]

    def test_default_off_and_native_sm120_fallback_are_preserved(self) -> None:
        self.assertIn("bool useProjectionGemmLt = false;", self.header)
        self.assertIn(
            'getBoolOpt(cfg, "cudaUseProjectionGemmLt", false)',
            self.source,
        )
        self.assertIn(
            "options.portableSm103Adapter && options.useProjectionGemmLt",
            self.matmul,
        )
        # Strict errors are conditional. The historical native-SM120 path can
        # still return false to MatMulLayer and use its legacy Hgemm fallback.
        self.assertGreaterEqual(self.matmul.count("strictSm103B29Contract"), 5)
        self.assertEqual(self.matmul.count("return false;"), 4)

    def test_exact_two_shape_contract_is_shared_by_static_and_runtime_checks(self) -> None:
        for token in (
            "constexpr int Sm103B29BatchSize = 29",
            "constexpr int Sm103B29BoardArea = 19 * 19",
            "constexpr int Sm103B29Rows = Sm103B29BatchSize * Sm103B29BoardArea",
            "Sm103B29ProjectionGemmLtExpectedShapeCount = 2",
            "Sm103B29InitialGlobalShape = {768,29,19}",
            "Sm103B29QKVShape = {384,10469,384}",
            "static_assert(Sm103B29Rows == 10469",
            "classifySm103B29ProjectionGemmLtShape(768,29,19)",
            "classifySm103B29ProjectionGemmLtShape(384,10469,384)",
            "classifySm103B29ProjectionGemmLtShape(384,10468,384)",
        ):
            with self.subTest(token=token):
                self.assertIn(token, self.source)

        runtime_call = self.matmul.index(
            "classifySm103B29ProjectionGemmLtShape("
        )
        unexpected = self.matmul.index(
            "SM103 B29 cuBLASLt MatMul received unexpected FP16 shape"
        )
        plan = self.matmul.index("getOrCreatePlan(")
        self.assertLess(runtime_call, unexpected)
        self.assertLess(unexpected, plan)

    def test_plan_records_selection_and_timing_identity(self) -> None:
        for token in (
            "int heuristicRank;",
            "int heuristicCount;",
            "float measuredUs;",
            "float wavesCount;",
            "size_t cublasLtVersion;",
            "size_t workspaceBytes;",
            "const int requestedAlgoCount = 16",
            "plan->heuristicRank = i + 1",
            "plan->heuristicCount = returnedAlgoCount",
            "plan->measuredUs = averageUs",
            "plan->wavesCount = heuristics[i].wavesCount",
            "plan->cublasLtVersion = cublasLtGetVersion()",
        ):
            with self.subTest(token=token):
                self.assertIn(token, self.source)

    def test_explicit_qkv_candidates_pin_the_complete_tuple(self) -> None:
        for token in (
            '"b29-qkv-id70-tile23-stages35-cluster5"',
            '"b29-qkv-id71-tile19-stages35-cluster6-diagnostic"',
            "70,23,1,0,0,0,35,0,5",
            "71,19,1,0,0,0,35,0,6",
            'cfg.contains("cudaProjectionGemmLtTacticSm103")',
            "cudaProjectionGemmLtTacticSm103 requires cudaUseProjectionGemmLt=true",
            "cublasLtMatmulAlgoInit(",
            "cublasLtMatmulAlgoConfigSetAttribute(",
            "cublasLtMatmulAlgoCheck(",
            "algoIdentityMatchesExplicitConfig(",
            "explicit tuple readback mismatch tactic=",
        ):
            with self.subTest(token=token):
                self.assertIn(token, self.source)

        # Every writable component of the full tuple is set explicitly; ID is
        # supplied to AlgoInit and all fields are then verified by readback.
        writable = (
            "CUBLASLT_ALGO_CONFIG_TILE_ID",
            "CUBLASLT_ALGO_CONFIG_SPLITK_NUM",
            "CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME",
            "CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING",
            "CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION",
            "CUBLASLT_ALGO_CONFIG_STAGES_ID",
            "CUBLASLT_ALGO_CONFIG_INNER_SHAPE_ID",
            "CUBLASLT_ALGO_CONFIG_CLUSTER_SHAPE_ID",
        )
        for attribute in writable:
            with self.subTest(attribute=attribute):
                # Once in identity Get and once in explicit Set.
                self.assertEqual(self.source.count(attribute), 2)

    def test_explicit_tactics_are_qkv_only_and_leave_initial_global_legacy(self) -> None:
        self.assertIn(
            "ltMatmulState->qkvTactic != "
            "Sm103B29ProjectionGemmLtAutotuneTactic",
            self.matmul,
        )
        legacy = self.matmul.index(
            "strictShapeKind == "
            "Sm103B29ProjectionGemmLtShapeKind::InitialGlobal"
        )
        plan = self.matmul.index("getOrCreatePlan(")
        self.assertLess(legacy, plan)
        self.assertIn("return false;", self.matmul[legacy:plan])
        self.assertIn('selectionTactic=" << plan->selectionTactic', self.matmul)
        self.assertIn(
            "options.portableSm103Adapter ? "
            "options.projectionGemmLtTacticSm103",
            self.source,
        )

    def test_complete_cuda13_algo_config_and_selected_cap_identity_is_queried(self) -> None:
        config_attributes = (
            "CUBLASLT_ALGO_CONFIG_ID",
            "CUBLASLT_ALGO_CONFIG_TILE_ID",
            "CUBLASLT_ALGO_CONFIG_SPLITK_NUM",
            "CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME",
            "CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING",
            "CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION",
            "CUBLASLT_ALGO_CONFIG_STAGES_ID",
            "CUBLASLT_ALGO_CONFIG_INNER_SHAPE_ID",
            "CUBLASLT_ALGO_CONFIG_CLUSTER_SHAPE_ID",
        )
        cap_attributes = (
            "CUBLASLT_ALGO_CAP_NUMERICAL_IMPL_FLAGS",
            "CUBLASLT_ALGO_CAP_MIN_ALIGNMENT_A_BYTES",
            "CUBLASLT_ALGO_CAP_MIN_ALIGNMENT_B_BYTES",
            "CUBLASLT_ALGO_CAP_MIN_ALIGNMENT_C_BYTES",
            "CUBLASLT_ALGO_CAP_MIN_ALIGNMENT_D_BYTES",
        )
        for attribute in config_attributes + cap_attributes:
            with self.subTest(attribute=attribute):
                expected = 1 if attribute == "CUBLASLT_ALGO_CONFIG_ID" else (
                    2 if attribute in config_attributes else 1
                )
                self.assertEqual(self.source.count(attribute), expected)
        self.assertIn("cublasLtMatmulAlgoConfigGetAttribute(", self.source)
        self.assertIn("cublasLtMatmulAlgoCapGetAttribute(", self.source)
        self.assertIn("writtenBytes == sizeof(T)", self.source)

    def test_strict_contract_fails_closed_for_plan_identity_and_launch(self) -> None:
        creation = self.matmul.index(
            "SM103 B29 cuBLASLt MatMul plan creation failed"
        )
        identity = self.matmul.index(
            "SM103 B29 cuBLASLt MatMul algorithm identity unavailable"
        )
        launch_call = self.matmul.index("const cublasStatus_t status = cublasLtMatmul(")
        launch_failure = self.matmul.index(
            "SM103 B29 cuBLASLt MatMul launch failed"
        )
        self.assertLess(creation, identity)
        self.assertLess(identity, launch_call)
        self.assertLess(launch_call, launch_failure)
        self.assertIn("if(strictSm103B29Contract && !plan->identity.complete)", self.matmul)
        self.assertIn("plan->identity.unavailableReason", self.matmul)

    def test_every_handle_and_shape_logs_first_success_with_full_identity(self) -> None:
        self.assertIn("bool loggedSuccessfulUse;", self.source)
        self.assertIn(
            "if(!plan->loggedSuccessfulUse && logger != NULL)", self.matmul
        )
        self.assertIn("plan->loggedSuccessfulUse = true", self.matmul)
        launch = self.matmul.index("const cublasStatus_t status = cublasLtMatmul(")
        log = self.matmul.index(
            "shape-keyed autotuned cuBLASLt FP16 MatMul plan active"
        )
        self.assertLess(launch, log)
        for token in (
            "contractShape=",
            "handle=0x",
            "stream=0x",
            "heuristicRank=",
            "heuristicCount=",
            "measuredUs=",
            "wavesCount=",
            "workspaceBytes=",
            "cublasLtVersion=",
            "algoConfig={id=",
            "tileId=",
            "splitK=",
            "reductionScheme=",
            "ctaSwizzling=",
            "customOption=",
            "stagesId=",
            "innerShapeId=",
            "clusterShapeId=",
            "algoCap={numericalImplFlags=0x",
            "minAlignmentABytes=",
            "minAlignmentBBytes=",
            "minAlignmentCBytes=",
            "minAlignmentDBytes=",
            "algoIdentity=unavailable reason={",
        ):
            with self.subTest(token=token):
                self.assertIn(token, self.matmul)

    def test_current_graph_routes_only_hot_qkv_and_initial_global_to_hook(self) -> None:
        # Runtime shape rejection is the authoritative contract. These source
        # adjacency checks guard the graph assumptions that justify its two
        # accepted shapes without importing CUDA or loading the 212 MB model.
        self.assertIn("initialMatMul->apply(", self.core)
        self.assertEqual(self.core.count("qProj.apply(cudaHandles"), 2)
        self.assertEqual(self.core.count("kProj.apply(cudaHandles"), 2)
        self.assertEqual(self.core.count("vProj.apply(cudaHandles"), 2)
        self.assertIn("if(!usedFusedResidual)\n      outProj.apply", self.core)
        self.assertIn("if(!usedSm103FusedFFN && !usedWideFFNSingleGemm)", self.core)
        self.assertIn("if(!usedFusedResidual)\n      linear2.apply", self.core)


if __name__ == "__main__":
    unittest.main()
