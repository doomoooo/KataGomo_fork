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
    launch_symbol,
    render_registry,
    select_tilelang_requests,
    symbol_token,
)
from sm120_generate_cute_qkv_aot import render_bridge as render_cute_qkv_bridge  # noqa: E402
from sm120_generate_tilelang_aot import append_wrapper  # noqa: E402


def test_space() -> dict:
    candidates = [
        {"id": "tile-a", "implementation": "tilelang"},
        {"id": "tile-b"},
        {"id": "official", "implementation": "fallback"},
    ]
    return {
        "schema": 1,
        "kind": "cuda-tactic-search-space",
        "architecture": "sm120",
        "batches": [
            {"batch": batch, "dual_ffn": candidates}
            for batch in range(1, 33)
        ],
    }


class FatScanTests(unittest.TestCase):
    def test_all_explicit_batches_are_materialized(self) -> None:
        requests = select_tilelang_requests(
            test_space(), "dual_ffn", range(1, 33), ("tile-a", "tile-b", "official")
        )
        self.assertEqual(len(requests), 64)
        self.assertEqual({item["batch"] for item in requests}, set(range(1, 33)))
        self.assertEqual(
            {item["candidate_id"] for item in requests}, {"tile-a", "tile-b"}
        )
        self.assertEqual(len({item["symbol_token"] for item in requests}), 64)

    def test_symbol_token_depends_on_exact_batch(self) -> None:
        self.assertNotEqual(
            symbol_token("wide_qkv", 1, "same-tactic"),
            symbol_token("wide_qkv", 32, "same-tactic"),
        )

    def test_merged_qkv_coordinate_keeps_the_wide_qkv_artifact_abi(self) -> None:
        space = {
            "schema": 1,
            "kind": "cuda-tactic-search-space",
            "architecture": "sm120",
            "batches": [{
                "batch": 19,
                "qkv_rope": [{
                    "id": "qkv-tile",
                    "implementation": "tilelang",
                    "artifact_family": "wide_qkv",
                }],
            }],
        }
        requests = select_tilelang_requests(space, "wide_qkv", [19])
        self.assertEqual(len(requests), 1)
        self.assertEqual(requests[0]["family"], "qkv_rope")
        self.assertEqual(requests[0]["artifact_family"], "wide_qkv")
        self.assertEqual(
            requests[0]["launch_symbol"],
            launch_symbol("wide_qkv", requests[0]["symbol_token"]),
        )

    def test_registry_contains_exact_batch_and_id_entries(self) -> None:
        requests = select_tilelang_requests(
            test_space(), "dual_ffn", (1, 32), ("tile-a",)
        )
        source = render_registry("dual_ffn", requests)
        self.assertIn('{1, "tile-a", false,', source)
        self.assertIn('{32, "tile-a", false,', source)
        for request in requests:
            self.assertEqual(source.count(request["launch_symbol"]), 2)

    def test_empty_registry_is_valid_for_plan_restricted_build(self) -> None:
        source = render_registry("linear2", [])
        self.assertIn("getSm120SearchLinear2FatTactics", source)
        self.assertIn("count = 0;", source)
        self.assertIn("return nullptr;", source)
        self.assertNotIn("fatTactics[]", source)

    def test_qkv_registry_preserves_packed_output_abi(self) -> None:
        token = symbol_token("wide_qkv", 7, "cute-packed")
        source = render_registry("wide_qkv", [{
            "batch": 7,
            "candidate_id": "cute-packed",
            "candidate": {"id": "cute-packed", "output": "packed"},
            "symbol_token": token,
            "launch_symbol": launch_symbol("wide_qkv", token),
        }])
        self.assertIn(
            '{7, "cute-packed", true,', source,
        )

    def test_fa4_registry_uses_exact_batch_id_getter(self) -> None:
        token = symbol_token("fa4", 13, "fa4-n96")
        source = render_registry("fa4", [{
            "batch": 13,
            "candidate_id": "fa4-n96",
            "candidate": {"id": "fa4-n96"},
            "symbol_token": token,
            "launch_symbol": launch_symbol("fa4", token),
        }])
        self.assertIn('{13, "fa4-n96",', source)
        self.assertIn("getSm120SearchFA4FatTactics", source)
        self.assertIn("float, bool, cudaStream_t", source)

    def test_fa4_bridge_carries_packed_qkv_stride(self) -> None:
        generator = (
            REPO / "cpp/neuralnet/fa4_aot/build_aot.py"
        ).read_text()
        self.assertIn("float scale, bool packedQKV", generator)
        self.assertIn("(packedQKV ? 3 : 1) * heads * dim", generator)

    def test_cute_qkv_fat_bridge_has_no_single_slot_exports(self) -> None:
        token = symbol_token("wide_qkv", 8, "cute-packed")
        source = render_cute_qkv_bridge(
            token, 8, "cute-packed", token,
        )
        self.assertIn(launch_symbol("wide_qkv", token), source)
        self.assertNotIn("sm120_search_qkv_batch()", source)
        self.assertNotIn("sm120_search_qkv_id()", source)

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

    def test_repository_wires_fat_slots_without_implicit_b13_winners(self) -> None:
        cmake = (REPO / "cpp/CMakeLists.txt").read_text()
        registry = (
            REPO / "cpp/neuralnet/cudabackend_sm120_aot_registry.cu"
        ).read_text()
        self.assertIn("SM120_SEARCH_FFN_SOURCE", cmake)
        self.assertIn("SM120_SEARCH_FFN_FAT_SOURCES", cmake)
        self.assertIn("SM120_SEARCH_QKV_FAT_OBJECTS", cmake)
        self.assertIn("SM120_SEARCH_FA4_FAT_OBJECTS", cmake)
        self.assertIn("searchFfnTactic", registry)
        self.assertIn("getSm120SearchFfnFatTactics", registry)
        self.assertIn("getSm120SearchFA4FatTactics", registry)
        self.assertIn('std::strncmp(requestedId, "fa4-b", 5)', registry)
        self.assertIn('std::string("fa4-b") + std::to_string(batchSize)', registry)
        self.assertIn("outproj-m128-n128-k32-s3-cutlass", registry)
        self.assertLess(
            registry.index("getSm120SearchFfnFatTactics", registry.index("findFusedFFNAotTactic")),
            registry.index("searchFfnTactic", registry.index("findFusedFFNAotTactic")),
        )
        self.assertNotIn("ffnTactics", registry)
        self.assertNotIn("qkvTactics", registry)
        self.assertNotIn('strcmp(requestedId, "auto")', registry)

    def test_planar_qkv_single_slot_exports_packed_abi_bit(self) -> None:
        source = append_wrapper(
            "extern __global__ void wide_qkv_kernel() {}\n",
            "wide_qkv",
            {"id": "qkv-planar", "m": 128, "n": 128, "k": 64, "stages": 2,
             "output": "planar"},
            16, 65536,
        )
        self.assertIn("sm120_search_qkv_packed()", source)
        self.assertIn("{ return 0; }", source)

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
prefix = {"dual_ffn": "ffn", "wide_qkv": "qkv"}.get(a.family, a.family)
metadata = {
  "batch": a.batch,
  "candidate": {"id": a.candidate_id},
  "fat_symbol_token": a.fat_symbol_token,
  "launch_symbol": f"sm120_search_{prefix}_fat_launch_{a.fat_symbol_token}",
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
                    "--family", "dual_ffn",
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
            migrated_space = test_space()
            migrated_space["cuda_device_properties_at_space_generation"] = {
                "compute_capability": [12, 0],
                "multiprocessor_count": 170,
            }
            space_path.write_text(json.dumps(migrated_space))
            subprocess.run(
                [
                    sys.executable,
                    str(PYTHON_DIR / "sm120_prepare_tilelang_fat_scan.py"),
                    "--space", str(space_path),
                    "--family", "dual_ffn",
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
            migrated = json.loads((output_dir / "manifest.json").read_text())
            self.assertTrue(all(item["reused"] for item in migrated["entries"]))
            self.assertTrue(all(
                item["reused_from_space_sha256"]
                for item in migrated["entries"]
            ))
            (output_dir / "manifest.json").unlink()
            subprocess.run(
                [
                    sys.executable,
                    str(PYTHON_DIR / "sm120_prepare_tilelang_fat_scan.py"),
                    "--space", str(space_path),
                    "--family", "dual_ffn",
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
