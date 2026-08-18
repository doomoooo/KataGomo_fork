from pathlib import Path
import re
import sys
import unittest


REPO = Path(__file__).resolve().parents[2]
PYTHON_DIR = REPO / "python"
sys.path.insert(0, str(PYTHON_DIR))

import sm103_cudnn_oss_b29 as upstream_candidate  # noqa: E402
import sm103_cudnn_oss_b29_roundtrip as variant_a  # noqa: E402
import sm103_cudnn_oss_b29_precise as variant_b  # noqa: E402
import sm103_cudnn_oss_b29_newton as variant_c  # noqa: E402


class Sm103CudnnOssFfnHookTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.backend = (REPO / "cpp/neuralnet/cudabackend.cpp").read_text()
        cls.header = (
            REPO / "cpp/neuralnet/cudabackend_sm103.h"
        ).read_text()
        cls.source = (
            REPO / "cpp/neuralnet/cudabackend_sm103.cpp"
        ).read_text()
        cls.stub = (
            REPO / "cpp/neuralnet/cudnn_oss_b29_aot_stub.cpp"
        ).read_text()
        cls.cmake = (REPO / "cpp/CMakeLists.txt").read_text()

    def test_tactic_is_explicit_exact_and_default_off(self) -> None:
        self.assertIn('std::string dualFfnTactic = "disabled"', self.header)
        self.assertIn('cfg.contains("cudaSm103DualFFNTactic")', self.source)
        tactic = re.search(
            r'constexpr const char\* CudnnOssB29VariantAFfnTactic\s*=\s*'
            r'"([^"]+)";',
            self.source,
        )
        self.assertIsNotNone(tactic)
        self.assertEqual(tactic.group(1), variant_a.CANDIDATE_ID)
        self.assertNotIn(upstream_candidate.CANDIDATE_ID, self.source)
        self.assertNotIn(variant_b.CANDIDATE_ID, self.source)
        self.assertNotIn(variant_c.CANDIDATE_ID, self.source)
        self.assertIn(
            "options.dualFfnTactic != CudnnOssB29VariantAFfnTactic",
            self.source,
        )
        self.assertIn(
            "cudaSm103DualFFNTactic requires cudaSm103Backend=true",
            self.source,
        )
        self.assertIn(
            "cudaSm103DualFFNTactic must be disabled or", self.source
        )

    def test_aot_manifest_identity_is_fail_closed(self) -> None:
        self.assertIn(
            '${SM103_CUDNN_OSS_B29_AOT_DIR}/aot-manifest.json', self.cmake
        )
        self.assertIn(
            'NOT EXISTS "${SM103_CUDNN_OSS_B29_MANIFEST}"', self.cmake
        )
        self.assertIn(
            'file(READ "${SM103_CUDNN_OSS_B29_MANIFEST}"', self.cmake
        )
        self.assertIn(variant_a.CANDIDATE_ID, self.cmake)
        self.assertNotIn(variant_b.CANDIDATE_ID, self.cmake)
        self.assertNotIn(upstream_candidate.CANDIDATE_ID, self.cmake)
        self.assertNotIn(variant_c.CANDIDATE_ID, self.cmake)
        self.assertIn("projection-fp16-roundtrip", self.cmake)
        self.assertIn("sm_103a", self.cmake)
        self.assertIn(
            "AOT manifest is not the exact Variant A candidate",
            self.cmake,
        )
        for token in (
            'file(SHA256 "${SM103_CUDNN_OSS_B29_HEADER}"',
            'file(SHA256 "${SM103_CUDNN_OSS_B29_ARTIFACT_OBJECT}"',
            'file(SHA256 "${SM103_CUDNN_OSS_B29_RUNTIME_ARCHIVE}"',
            "SM103_CUDNN_OSS_B29_HEADER_RECORD",
            "SM103_CUDNN_OSS_B29_OBJECT_RECORD",
            "SM103_CUDNN_OSS_B29_RUNTIME_RECORD",
            "manifest hash authentication failed for header/object/runtime archive",
            'configure_file(',
            '"${SM103_CUDNN_OSS_B29_ARTIFACT_OBJECT}"',
            '"${SM103_CUDNN_OSS_B29_OBJECT}"',
            "COPYONLY",
        ):
            self.assertIn(token, self.cmake)

    def test_hook_is_exact_cc103_b29_fp16(self) -> None:
        for token in (
            "constexpr int B29Rows = 10469",
            "constexpr int InputChannels = 384",
            "constexpr int PackedChannels = 2304",
            "constexpr int OutputChannels = 1152",
            "maxBatchSize != 29",
            "isSm103Arch(majorComputeCapability,minorComputeCapability)",
            "matBatchSize != B29Rows",
            "!usingFP16",
            "R10469/K384/N2304->1152 FP16",
        ):
            self.assertIn(token, self.source)

    def test_weights_are_packed_once_in_gate_then_linear1_pairs(self) -> None:
        self.assertIn("constexpr int PairChannels = 32", self.source)
        self.assertIn("packedFfnWeights.find(key)", self.source)
        self.assertIn("packedFfnWeights.emplace(key,packedWeights)", self.source)
        gate = self.source.index("Sm103CudnnOssFfnPackGate")
        linear1 = self.source.index("Sm103CudnnOssFfnPackLinear1")
        self.assertLess(gate, linear1)
        self.assertIn(
            "(const char*)linearGateWeights + sourceOffset", self.source
        )
        self.assertIn(
            "(const char*)linear1Weights + sourceOffset", self.source
        )
        self.assertIn("pairOffset + chunkBytes", self.source)
        self.assertIn(
            "stream==0 is CUDA's valid legacy/default stream", self.source
        )
        self.assertNotIn("linearGateWeights == NULL || stream == NULL", self.source)
        self.assertIn("if(hasLaunchedFfn)", self.source)
        self.assertIn("hasLaunchedFfn = true", self.source)

    def test_transformer_ffn_reuses_ab12_and_c_buffers(self) -> None:
        call = self.backend.index("cudaHandles->sm103FusedFFN(")
        swiglu = self.backend.index("// Step 3: SwiGLU", call)
        snippet = self.backend[call:swiglu]
        self.assertIn("linear1.matBuf", snippet)
        self.assertIn("linearGate->matBuf", snippet)
        self.assertIn("wideFFNBuf.buf", snippet)
        self.assertIn("ffnBuf.buf", snippet)
        self.assertIn("matBatchSize == 10469", self.backend)
        self.assertIn(
            "scratch->allocator, (size_t)ffnChannels * 2 * "
            "matBatchSize * bytesPerElt",
            self.backend,
        )

    def test_successful_launch_has_unambiguous_activation_log(self) -> None:
        self.assertIn("katagoCudnnOssB29Launch(", self.source)
        self.assertIn(
            "SM103 backend: cuDNN OSS fused FFN active, tactic=",
            self.source,
        )
        self.assertIn(
            "string(CudnnOssB29VariantAFfnTactic)", self.source
        )
        self.assertIn("loggedCudnnOssFfn = true", self.source)
        self.assertIn(
            "SM103 cuDNN OSS fused FFN launch failed, status=",
            self.source,
        )

    def test_default_cmake_uses_stub_and_optional_artifact_is_fail_closed(self) -> None:
        self.assertIn('set(SM103_CUDNN_OSS_B29_AOT_DIR ""', self.cmake)
        self.assertIn(
            "neuralnet/cudnn_oss_b29_aot_stub.cpp", self.cmake
        )
        self.assertIn(
            "neuralnet/cudnn_oss_b29_aot_bridge.cpp", self.cmake
        )
        self.assertIn(
            "SM103_CUDNN_OSS_B29_RUNTIME_ARCHIVE", self.cmake
        )
        self.assertIn(
            "SM103_CUDNN_OSS_B29_AOT_DIR lacks the generated header/object",
            self.cmake,
        )
        self.assertIn(
            "KATAGO_ENABLE_SM103_CUDNN_OSS_B29_AOT", self.cmake
        )
        self.assertIn(
            "KATAGO_CUDNN_OSS_B29_MODULE_LOAD_FAILED", self.stub
        )
        self.assertNotIn(
            "katago_sm103_b29_cudnn_swiglu.h", self.stub
        )

    def test_default_stub_cannot_claim_activation(self) -> None:
        self.assertIn("return nullptr", self.stub)
        self.assertNotIn("active, tactic=", self.stub)
        self.assertNotIn("cudaSuccess", self.stub)


if __name__ == "__main__":
    unittest.main()
