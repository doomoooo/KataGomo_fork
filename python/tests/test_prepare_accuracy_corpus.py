import importlib.util
import io
import json
import pathlib
import tarfile
import tempfile
import unittest


SCRIPT = (
    pathlib.Path(__file__).resolve().parents[2]
    / "final-migration"
    / "autotune"
    / "prepare_accuracy_corpus.py"
)
SPEC = importlib.util.spec_from_file_location("prepare_accuracy_corpus", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class PrepareAccuracyCorpusTests(unittest.TestCase):
    def test_select_latest_uses_dated_archive_names(self) -> None:
        html = """
        <a href="2026-08-02npzs.tgz">old</a>
        <a href="notes.txt">ignore</a>
        <a href="/kata1/trainingdata/2026-08-04npzs.tgz">new</a>
        """
        self.assertEqual(MODULE.select_latest_archive(html), "2026-08-04npzs.tgz")

    def test_select_latest_rejects_empty_index(self) -> None:
        with self.assertRaises(ValueError):
            MODULE.select_latest_archive("<a href='README.txt'>README</a>")

    def test_safe_extract_rejects_traversal(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            archive = root / "bad.tgz"
            with tarfile.open(archive, "w:gz") as payload:
                info = tarfile.TarInfo("../escape.npz")
                data = b"bad"
                info.size = len(data)
                payload.addfile(info, io.BytesIO(data))
            with self.assertRaises(ValueError):
                MODULE.safe_extract_training_archive(archive, root / "out")

    def test_validate_corpus_checks_hash_and_lock(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            corpus = root / "2026-08-04-19x19-8192-seed20260803-full19.npz"
            corpus.write_bytes(b"fixed corpus")
            manifest = root / "corpus.manifest.json"
            manifest.write_text(json.dumps({
                "num_samples": 8192,
                "seed": 20260803,
                "output_npz": corpus.name,
                "output_npz_sha256": MODULE.sha256_file(corpus),
                "source_archive": "2026-08-04npzs.tgz",
                "source_archive_sha256": "1" * 64,
                "source_archive_url": "https://katagoarchive.org/kata1/trainingdata/2026-08-04npzs.tgz",
            }))
            payload = MODULE.validate_corpus(
                corpus,
                manifest,
                expected_archive="2026-08-04npzs.tgz",
                expected_url="https://katagoarchive.org/kata1/trainingdata/2026-08-04npzs.tgz",
            )
            self.assertEqual(payload["num_samples"], 8192)
            corpus.write_bytes(b"changed")
            with self.assertRaises(ValueError):
                MODULE.validate_corpus(corpus, manifest)

    def test_release_setup_invokes_corpus_validator(self) -> None:
        setup = (SCRIPT.parent / "setup.sh").read_text()
        package = (SCRIPT.parent / "package-autotune.sh").read_text()
        self.assertIn('"${SCRIPT_DIR}/prepare_accuracy_corpus.py"', setup)
        self.assertIn("--refresh-latest", package)
        self.assertIn('"${SCRIPT_DIR}/prepare_accuracy_corpus.py"', package)
        self.assertIn('--corpus "${CORPUS}" --manifest "${CORPUS_MANIFEST}"', package)
        self.assertIn('result["source_url"]', package)


if __name__ == "__main__":
    unittest.main()
