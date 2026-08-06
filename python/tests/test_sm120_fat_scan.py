#!/usr/bin/env python3

from __future__ import annotations

import pathlib
import json
import subprocess
import sys
import tempfile
import unittest


PYTHON_DIR = pathlib.Path(__file__).resolve().parents[1]
REPO = PYTHON_DIR.parent
sys.path.insert(0, str(PYTHON_DIR))

from sm120_fat_scan import (  # noqa: E402
    isolate_tilelang_debug_symbols,
    render_registry,
    select_tilelang_requests,
    symbol_token,
)


def test_space() -> dict:
    candidates = [
        {"id": "tile-a", "implementation": "tilelang"},
        {"id": "tile-b"},
        {"id": "official", "implementation": "fallback"},
    ]
    return {
        "schema": 2,
        "batches": [
            {"batch": batch, "ffn": candidates}
            for batch in range(1, 33)
        ],
    }


class FatScanTests(unittest.TestCase):
    def test_all_explicit_batches_are_materialized(self) -> None:
        requests = select_tilelang_requests(
            test_space(), "ffn", range(1, 33), ("tile-a", "tile-b", "official")
        )
        self.assertEqual(len(requests), 64)
        self.assertEqual({item["batch"] for item in requests}, set(range(1, 33)))
        self.assertEqual(
            {item["candidate_id"] for item in requests}, {"tile-a", "tile-b"}
        )
        self.assertEqual(len({item["symbol_token"] for item in requests}), 64)

    def test_symbol_token_depends_on_exact_batch(self) -> None:
        self.assertNotEqual(
            symbol_token("qkv", 1, "same-tactic"),
            symbol_token("qkv", 32, "same-tactic"),
        )

    def test_registry_contains_exact_batch_and_id_entries(self) -> None:
        requests = select_tilelang_requests(
            test_space(), "ffn", (1, 32), ("tile-a",)
        )
        source = render_registry("ffn", requests)
        self.assertIn('{1, SM120_GPU_OTHER, 0, "tile-a", false,', source)
        self.assertIn('{32, SM120_GPU_OTHER, 0, "tile-a", false,', source)
        for request in requests:
            self.assertEqual(source.count(request["launch_symbol"]), 2)

    def test_debug_header_symbols_are_unique_per_tu(self) -> None:
        original = """#include <tl_templates/cuda/debug.h>
__global__ void kernel() { debug_print_msg("x"); }
"""
        first = isolate_tilelang_debug_symbols(original, "ffn_b1_deadbeef")
        second = isolate_tilelang_debug_symbols(original, "ffn_b2_deadbeef")
        self.assertIn(
            "#define debug_print_msg sm120_tl_debug_print_msg_ffn_b1_deadbeef",
            first,
        )
        self.assertIn(
            "#define debug_print_msg sm120_tl_debug_print_msg_ffn_b2_deadbeef",
            second,
        )
        self.assertNotEqual(first, second)
        self.assertTrue(first.rstrip().endswith("#undef PrintTraits"))

    def test_repository_wires_fat_and_legacy_slots_together(self) -> None:
        cmake = (REPO / "cpp/CMakeLists.txt").read_text()
        registry = (
            REPO / "cpp/neuralnet/cudabackend_sm120_aot_registry.cu"
        ).read_text()
        self.assertIn("SM120_SEARCH_FFN_SOURCE", cmake)
        self.assertIn("SM120_SEARCH_FFN_FAT_SOURCES", cmake)
        self.assertIn("searchFfnTactic", registry)
        self.assertIn("getSm120SearchFfnFatTactics", registry)
        self.assertLess(
            registry.index("getSm120SearchFfnFatTactics", registry.index("findFusedFFNAotTactic")),
            registry.index("ffnTactics", registry.index("findFusedFFNAotTactic")),
        )

    def test_cpu_only_preparer_materializes_b1_through_b32(self) -> None:
        fake_generator = r'''#!/usr/bin/env python3
import argparse, hashlib, json, pathlib
p = argparse.ArgumentParser()
p.add_argument("--space")
p.add_argument("--family")
p.add_argument("--candidate-id")
p.add_argument("--batch", type=int)
p.add_argument("--device")
p.add_argument("--output-dir")
p.add_argument("--source-path")
p.add_argument("--fat-symbol-token")
p.add_argument("--s1-warmup")
p.add_argument("--s1-iterations")
a = p.parse_args()
source = f"// {a.family} B{a.batch} {a.fat_symbol_token}\n"
source_path = pathlib.Path(a.source_path)
source_path.parent.mkdir(parents=True, exist_ok=True)
source_path.write_text(source)
metadata_dir = pathlib.Path(a.output_dir)
metadata_dir.mkdir(parents=True, exist_ok=True)
metadata = {
  "batch": a.batch,
  "candidate": {"id": a.candidate_id},
  "fat_symbol_token": a.fat_symbol_token,
  "launch_symbol": f"sm120_search_{a.family}_fat_launch_{a.fat_symbol_token}",
  "source_sha256": hashlib.sha256(source.encode("ascii")).hexdigest(),
}
(metadata_dir / f"{a.family}-{a.candidate_id}.json").write_text(json.dumps(metadata))
'''
        with tempfile.TemporaryDirectory() as temporary_text:
            temporary = pathlib.Path(temporary_text)
            space_path = temporary / "space.json"
            generator_path = temporary / "fake_generator.py"
            output_dir = temporary / "bundle"
            space_path.write_text(json.dumps(test_space()))
            generator_path.write_text(fake_generator)
            subprocess.run(
                [
                    sys.executable,
                    str(PYTHON_DIR / "sm120_prepare_tilelang_fat_scan.py"),
                    "--space", str(space_path),
                    "--family", "ffn",
                    "--batches", "1-32",
                    "--candidate-ids", "tile-a",
                    "--device", "999",
                    "--generator", str(generator_path),
                    "--output-dir", str(output_dir),
                ],
                check=True,
                text=True,
                capture_output=True,
            )
            manifest = json.loads((output_dir / "manifest.json").read_text())
            self.assertTrue(manifest["complete"])
            self.assertEqual(manifest["requested_entry_count"], 32)
            self.assertEqual(len(manifest["entries"]), 32)
            self.assertEqual(
                {item["batch"] for item in manifest["entries"]}, set(range(1, 33))
            )
            self.assertEqual(len(manifest["sources"]), 32)
            (output_dir / "manifest.json").unlink()
            subprocess.run(
                [
                    sys.executable,
                    str(PYTHON_DIR / "sm120_prepare_tilelang_fat_scan.py"),
                    "--space", str(space_path),
                    "--family", "ffn",
                    "--batches", "1-32",
                    "--candidate-ids", "tile-a",
                    "--device", "999",
                    "--generator", str(generator_path),
                    "--output-dir", str(output_dir),
                    "--reuse-existing",
                ],
                check=True,
                text=True,
                capture_output=True,
            )
            recovered = json.loads((output_dir / "manifest.json").read_text())
            self.assertTrue(recovered["complete"])
            self.assertTrue(all(
                item["recovered_without_prior_manifest"]
                for item in recovered["entries"]
            ))


if __name__ == "__main__":
    unittest.main()
