from pathlib import Path
import unittest


REPO = Path(__file__).resolve().parents[2]


class Sm103BackendScaffoldTests(unittest.TestCase):
    def test_scaffold_is_exact_and_default_off(self) -> None:
        source = (REPO / "cpp/neuralnet/cudabackend_sm103.cpp").read_text()
        self.assertIn(
            "majorComputeCapability == 10 && minorComputeCapability == 3",
            source,
        )
        self.assertIn('getBoolOpt(cfg, "cudaSm103Backend", false)', source)
        backend = (REPO / "cpp/neuralnet/cudabackend.cpp").read_text()
        self.assertIn(
            "cudaSm103Backend requires an exact compute capability 10.3 device",
            backend,
        )

    def test_scaffold_cannot_claim_tactic_activation(self) -> None:
        source = (REPO / "cpp/neuralnet/cudabackend_sm103.cpp").read_text()
        self.assertIn(
            "cudaSm103AllowOfficialForwardScaffold=true is required", source
        )
        self.assertIn(
            "cudaSm103AllowOfficialForwardScaffold requires "
            "cudaSm103Backend=true",
            source,
        )
        self.assertIn("no optimized tactics launched", source)
        self.assertNotIn("runtime tactic active", source)

    def test_scaffold_is_wired_before_official_fallback(self) -> None:
        cmake = (REPO / "cpp/CMakeLists.txt").read_text()
        backend = (REPO / "cpp/neuralnet/cudabackend.cpp").read_text()
        self.assertIn("neuralnet/cudabackend_sm103.cpp", cmake)
        self.assertIn(
            "std::unique_ptr<Sm103Backend::Sm103Model> sm103Model", backend
        )
        self.assertLess(
            backend.index("if(sm103Model != nullptr)"),
            backend.index("else if(sm120Model != nullptr)"),
        )


if __name__ == "__main__":
    unittest.main()
