from pathlib import Path
import re
import sys
import unittest


REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "python"))

import sm103_generate_flash_attn_b29_native_aot as generator  # noqa: E402


class Sm103Fa4NativeHookTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.backend = (REPO / "cpp/neuralnet/cudabackend.cpp").read_text()
        cls.header = (REPO / "cpp/neuralnet/cudabackend_sm103.h").read_text()
        cls.source = (REPO / "cpp/neuralnet/cudabackend_sm103.cpp").read_text()
        cls.bridge = (
            REPO / "cpp/neuralnet/fa4_sm103_b29_aot_bridge.cpp"
        ).read_text()
        cls.stub = (
            REPO / "cpp/neuralnet/fa4_sm103_b29_aot_stub.cpp"
        ).read_text()
        cls.generator = (
            REPO / "python/sm103_generate_flash_attn_b29_native_aot.py"
        ).read_text()
        cls.cmake = (REPO / "cpp/CMakeLists.txt").read_text()

    def test_tactic_is_exact_and_default_off(self) -> None:
        self.assertIn('std::string attentionTactic = "disabled"', self.header)
        self.assertIn('cfg.contains("cudaSm103AttentionTactic")', self.source)
        tactic = re.search(
            r'constexpr const char\* Fa4Sm103B29AttentionTactic\s*=\s*'
            r'"([^"]+)";',
            self.source,
        )
        self.assertIsNotNone(tactic)
        self.assertEqual(tactic.group(1), generator.CANDIDATE_ID)
        self.assertIn(
            "cudaSm103AttentionTactic requires cudaSm103Backend=true",
            self.source,
        )

    def test_constructor_pins_batch_not_flattened_rows(self) -> None:
        attention_ctor = self.source.index(
            "options.attentionTactic == Fa4Sm103B29AttentionTactic"
        )
        destructor = self.source.index("Sm103Model::~Sm103Model", attention_ctor)
        snippet = self.source[attention_ctor:destructor]
        self.assertIn("maxBatchSize != 29", snippet)
        self.assertNotIn("maxBatchSize != B29Rows", snippet)
        self.assertIn("S361/H12/D32 planar FP16", snippet)

    def test_constructor_failure_cleans_both_aot_contexts(self) -> None:
        creation = self.source.index(
            "fa4Sm103B29Context = katagoFa4Sm103B29Create"
        )
        failure_throw = self.source.index(
            "SM103 FA4 native AOT module unavailable", creation
        )
        snippet = self.source[creation:failure_throw]
        self.assertIn("katagoFa4Sm103B29Destroy(fa4Sm103B29Context)", snippet)
        self.assertIn("fa4Sm103B29Context = nullptr", snippet)
        self.assertIn("katagoCudnnOssB29Destroy(cudnnOssB29Context)", snippet)
        self.assertIn("cudnnOssB29Context = nullptr", snippet)

    def test_dispatch_precedes_cudnn_and_sm120(self) -> None:
        sm103 = self.backend.index("cudaHandles->sm103Attention(")
        sm120 = self.backend.index("cudaHandles->sm120Attention(", sm103)
        cudnn = self.backend.index("sdpaCache->getOrBuildPlan", sm120)
        self.assertLess(sm103, sm120)
        self.assertLess(sm120, cudnn)
        self.assertIn("if(!usedSDPA && cudaHandles->sm120Attention", self.backend)

    def test_bridge_is_exact_planar_b29_d32(self) -> None:
        for token in (
            "constexpr int Batch = 29",
            "constexpr int Sequence = 361",
            "constexpr int Heads = 12",
            "constexpr int HeadDim = 32",
            "mask != nullptr",
            "packedQKV != 0",
            "major == 10 && minor == 3",
            "cute_dsl_fa4_sm103a_b29_wrapper",
            "scale,Sequence,stream",
        ):
            self.assertIn(token, self.bridge)

    def test_export_keeps_max_sequence_runtime_abi_slot(self) -> None:
        self.assertIn('parameter.name == "max_seqlen_q"', self.generator)
        self.assertIn("parameter.replace(annotation=Int32)", self.generator)
        self.assertNotIn(
            'isinstance(value, AuxData) or name == "max_seqlen_q"',
            self.generator,
        )
        self.assertIn("scale,Sequence,stream", self.bridge)

    def test_cmake_authenticates_and_default_stub_fails(self) -> None:
        for token in (
            'set(SM103_FA4_B29_AOT_DIR ""',
            "neuralnet/fa4_sm103_b29_aot_stub.cpp",
            "neuralnet/fa4_sm103_b29_aot_bridge.cpp",
            generator.CANDIDATE_ID,
            generator.FLASH_ATTN_COMMIT,
            'file(SHA256 "${SM103_FA4_B29_HEADER}"',
            'file(SHA256 "${SM103_FA4_B29_ARTIFACT_OBJECT}"',
            'file(SHA256 "${SM103_FA4_B29_RUNTIME_ARCHIVE}"',
            "manifest hash authentication failed for header/object/runtime archive",
            "COPYONLY",
        ):
            self.assertIn(token, self.cmake)
        self.assertIn("return nullptr", self.stub)
        self.assertIn("KATAGO_FA4_SM103_B29_MODULE_LOAD_FAILED", self.stub)
        self.assertNotIn("active, tactic=", self.stub)


if __name__ == "__main__":
    unittest.main()
