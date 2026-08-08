import importlib.util
import json
import pathlib
import tempfile
import unittest


AUTOTUNE_PATH = (
    pathlib.Path(__file__).resolve().parents[2]
    / "final-migration"
    / "autotune"
    / "autotune.py"
)
SPEC = importlib.util.spec_from_file_location("final_migration_autotune", AUTOTUNE_PATH)
assert SPEC is not None and SPEC.loader is not None
AUTOTUNE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUTOTUNE)


class AutotuneEntrypointTests(unittest.TestCase):
    def test_parse_batch_set(self) -> None:
        self.assertEqual(AUTOTUNE.parse_batch_set("4-6,8,6"), [4, 5, 6, 8])
        with self.assertRaises(ValueError):
            AUTOTUNE.parse_batch_set("6-4")

    def test_complete_manifest_requires_exact_domain(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "manifest.json"
            path.write_text(json.dumps({"complete": False, "batches": [4, 5]}))
            self.assertFalse(AUTOTUNE.complete_manifest_for_batches(path, "4-5"))

            path.write_text(json.dumps({"complete": True, "batches": [4, 5]}))
            self.assertTrue(AUTOTUNE.complete_manifest_for_batches(path, "4-5"))
            self.assertFalse(AUTOTUNE.complete_manifest_for_batches(path, "4-6"))

            path.write_text("not json")
            self.assertFalse(AUTOTUNE.complete_manifest_for_batches(path, "4-5"))

    def test_tilelang_root_is_read_from_complete_manifests(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory) / "tilelang"
            (root / "src/tl_templates/cuda").mkdir(parents=True)
            (root / "3rdparty/cutlass/include/cutlass").mkdir(parents=True)
            (root / "src/tl_templates/cuda/debug.h").touch()
            (root / "3rdparty/cutlass/include/cutlass/cutlass.h").touch()
            metadata = pathlib.Path(directory) / "metadata.json"
            metadata.write_text(json.dumps({
                "generation_environment": {"tilelang_root": str(root)},
            }))
            manifest = {"complete": True, "entries": [{"metadata": str(metadata)}]}
            self.assertEqual(AUTOTUNE.tilelang_root_from_manifests(manifest), root)

            with self.assertRaises(RuntimeError):
                AUTOTUNE.tilelang_root_from_manifests({**manifest, "complete": False})


if __name__ == "__main__":
    unittest.main()
