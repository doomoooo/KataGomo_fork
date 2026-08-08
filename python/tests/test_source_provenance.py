import ast
import pathlib
import subprocess
import tempfile
import unittest
from unittest import mock

from python.source_provenance import clean_source_revision


REVISION = "dcf215af68a2d08d305076c152a06f201728cd53"


class SourceProvenanceTests(unittest.TestCase):
    def test_cute_qkv_revision_matches_release_source_lock(self) -> None:
        repo = pathlib.Path(__file__).resolve().parents[2]
        module = ast.parse(
            (repo / "python/sm120_generate_cute_qkv_aot.py").read_text()
        )
        generator_revision = next(
            node.value.value
            for node in module.body
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "CUTLASS_COMMIT"
                for target in node.targets
            )
            and isinstance(node.value, ast.Constant)
        )
        source_lock = {
            columns[0]: columns[1]
            for line in (
                repo / "final-migration/autotune/source-lock.tsv"
            ).read_text().splitlines()[1:]
            if len(columns := line.split("\t")) >= 2
        }
        self.assertEqual(generator_revision, source_lock["cutlass"])

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
