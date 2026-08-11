import pathlib
import subprocess
import tempfile
import unittest
from unittest import mock

from python.source_provenance import clean_source_revision


REVISION = "dcf215af68a2d08d305076c152a06f201728cd53"


class SourceProvenanceTests(unittest.TestCase):
    def test_cute_qkv_records_but_does_not_pin_source_revision(self) -> None:
        repo = pathlib.Path(__file__).resolve().parents[2]
        generator = (
            repo / "python/sm120_generate_cute_qkv_aot.py"
        ).read_text()
        source_policy = (
            repo / "final-migration/autotune/source-lock.tsv"
        ).read_text()
        self.assertNotIn("CUTLASS_COMMIT", generator)
        self.assertNotIn("DENSE_GEMM_SHA256", generator)
        self.assertIn('"cutlass_commit": actual_commit', generator)
        self.assertIn("cutlass\tcurrent-clean-source", source_policy)

    def test_release_archive_revision_marker(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            (root / ".katago-source-revision").write_text(REVISION + "\n")
            with mock.patch(
                "python.source_provenance.subprocess.run",
                return_value=subprocess.CompletedProcess([], 128, "", "not a git tree"),
            ):
                self.assertEqual(
                    clean_source_revision(root), (REVISION, "archive-marker")
                )

    def test_archive_without_identity_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with mock.patch(
                "python.source_provenance.subprocess.run",
                return_value=subprocess.CompletedProcess([], 128, "", "not a git tree"),
            ):
                with self.assertRaisesRegex(RuntimeError, "identity unavailable"):
                    clean_source_revision(pathlib.Path(directory))

    def test_invalid_marker_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            (root / ".katago-source-revision").write_text("not-a-revision\n")
            with mock.patch(
                "python.source_provenance.subprocess.run",
                return_value=subprocess.CompletedProcess([], 128, "", "not a git tree"),
            ):
                with self.assertRaisesRegex(RuntimeError, "invalid source revision"):
                    clean_source_revision(root)


if __name__ == "__main__":
    unittest.main()
