import pathlib
import unittest


REPO = pathlib.Path(__file__).resolve().parents[2]


class Sm103PortableTacticsAdapterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.core = (REPO / "cpp/neuralnet/cudabackend.cpp").read_text()
        cls.sm103_h = (REPO / "cpp/neuralnet/cudabackend_sm103.h").read_text()
        cls.sm103 = (REPO / "cpp/neuralnet/cudabackend_sm103.cpp").read_text()
        cls.sm120_h = (REPO / "cpp/neuralnet/cudabackend_sm120.h").read_text()
        cls.sm120 = (REPO / "cpp/neuralnet/cudabackend_sm120.cpp").read_text()

    def test_sm103_opt_in_is_explicit_and_requires_master_switch(self) -> None:
        self.assertIn("bool reusePortableTactics = false;", self.sm103_h)
        self.assertIn(
            'getBoolOpt(cfg, "cudaSm103ReusePortableTactics", false)',
            self.sm103,
        )
        self.assertIn(
            "cudaSm103ReusePortableTactics requires cudaSm103Backend=true",
            self.sm103,
        )

    def test_portable_owner_exists_only_behind_sm103_opt_in(self) -> None:
        gate = self.core.index("const bool sm103PortableTacticsActive")
        validate = self.core.index("Sm120Backend::makeSm103PortableOptions", gate)
        owner = self.core.index("if(sm120HookOwnerActive)", validate)
        construct = self.core.index(
            "std::make_unique<Sm120Backend::Sm120Model>", owner
        )
        self.assertLess(gate, validate)
        self.assertLess(validate, owner)
        self.assertLess(owner, construct)
        # SM103 remains the outer adapter; its official traversal reaches the
        # portable operator hooks owned by Sm120Model.
        self.assertLess(
            self.core.index("if(sm103Model != nullptr)"),
            self.core.index("else if(sm120Model != nullptr)"),
        )

    def test_validator_rejects_all_architecture_bound_routes(self) -> None:
        self.assertIn(
            "Options makeSm103PortableOptions(const Options& requested);",
            self.sm120_h,
        )
        rejected = (
            "cudaUseFlashAttentionSm120",
            "cudaFlashAttentionAotTacticSm120",
            "cudaUseWideQKV",
            "cudaUseQKVGemmAot",
            "cudaQKVRopeAotTacticSm120",
            "cudaUseFusedFFN",
            "cudaFusedFFNAotTacticSm120",
            "cudaUseLinear2ResidualAot",
            "cudaUseOutProjectionResidualAot",
            "cudaOutProjectionAotTacticSm120",
            "cudaOuterProjectionDownTacticSm120",
            "cudaOuterProjectionUpTacticSm120",
            "cudaUsePostConvBNSiluSm120",
            "cudaWideQKVAotTacticSm120",
            "cudaLinear2AotTacticSm120",
            "cudaInitialConvFrontendPlanSm120",
            "cudaWideHeadProjectionTacticSm120",
        )
        for key in rejected:
            with self.subTest(key=key):
                self.assertIn(f'reject(requested.', self.sm120)
                self.assertIn(f'"{key}"', self.sm120)
        self.assertIn("portable.portableSm103Adapter = true;", self.sm120)
        self.assertIn("if(!options.portableSm103Adapter)", self.sm120)

    def test_only_portable_hooks_can_be_installed_on_sm103(self) -> None:
        # Attention is the FA-AOT boundary and is native-SM120-only.
        attention_gate = self.core.index("if(nativeSm120BackendActive)")
        attention_hook = self.core.index(
            "cudaHandles->sm120Attention = &Sm120Backend::applyAttention",
            attention_gate,
        )
        ffn_hook = self.core.index(
            "cudaHandles->sm120FFNSingleGemm = &Sm120Backend::applyFFNSingleGemm"
        )
        self.assertLess(attention_gate, attention_hook)
        self.assertLess(attention_hook, ffn_hook)

        portable_hooks = (
            "applyFFNSingleGemm",  # cuBLAS wide projection + plain CUDA SwiGLU
            "applyMatMulLt",
            "applyInitialGlobal",
            "applyQKVStrided",
            "applyFusedResidualGemm",
            "applyRMSNorm",
            "applyFusedQKRoPE",
            "applySwiGLU",
            "applyAffineSilu",
            "applyFusedPolicyP1",
            "applyPersistingL2Window",
        )
        for hook in portable_hooks:
            with self.subTest(hook=hook):
                self.assertIn(f"&Sm120Backend::{hook}", self.core)

    def test_logs_identify_sm103_portable_activation(self) -> None:
        self.assertIn('"SM103 portable backend: "', self.sm120)
        self.assertIn(
            "SM103 backend: official forward adapter active with portable CUDA tactic hooks",
            self.sm103,
        )
        self.assertIn(
            "SM103 portable backend: head BN direct FP32 output active",
            self.core,
        )

    def test_b29_mixed_affine_silu_is_exact_and_fail_closed(self) -> None:
        tactic = "half2-c384-flat-vec8-c768-b29"
        self.assertIn(f'o.affineSiluTactic != "{tactic}"', self.sm120)
        self.assertIn(
            f'options.affineSiluTactic == "{tactic}"',
            self.sm120,
        )
        self.assertIn(
            "!options.portableSm103Adapter || maxBatchSize != 29 || "
            "batchSize != 29",
            self.sm120,
        )
        self.assertIn(
            "(useB29MixedAffineSilu && channels == 768)",
            self.sm120,
        )
        self.assertIn(
            "B29 half2 C384 + flat vec8 C768 affine SiLU active",
            self.sm120,
        )


if __name__ == "__main__":
    unittest.main()
