import importlib.util
import pathlib
import unittest


MODULE_PATH = (
    pathlib.Path(__file__).resolve().parents[1]
    / "sm120_select_local_candidates.py"
)
SPEC = importlib.util.spec_from_file_location("sm120_select_local_candidates", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def row(candidate_id, latency, tile, smem):
    return {
        "candidate_id": candidate_id,
        "s1_us_median": latency,
        "resource_signature": {
            "tile": tile,
            "dynamic_smem_bytes": smem,
        },
    }


class SelectLocalCandidatesTest(unittest.TestCase):
    def test_top_k_and_near_best_structural_neighbor(self):
        result = MODULE.select_group(
            [
                row("a", 10.0, [128, 64, 32], 32768),
                row("b", 10.1, [128, 64, 32], 32768),
                row("c", 10.3, [64, 64, 32], 24576),
                row("d", 11.0, [64, 64, 64], 49152),
            ],
            top_k=2,
            near_best_fraction=0.05,
            max_retained=4,
        )
        self.assertEqual(result["winner"], "a")
        self.assertEqual(result["retained"], ["a", "b", "c"])

    def test_retention_cap(self):
        result = MODULE.select_group(
            [
                row("a", 10.0, [128, 64, 32], 32768),
                row("b", 10.1, [64, 64, 32], 24576),
                row("c", 10.2, [128, 64, 64], 49152),
            ],
            top_k=1,
            near_best_fraction=0.05,
            max_retained=2,
        )
        self.assertEqual(result["retained"], ["a", "b"])

    def test_complement_retains_every_pruned_candidate(self):
        result = MODULE.select_complement(
            [
                row("a", 10.0, [128, 64, 32], 32768),
                row("b", 10.1, [64, 64, 32], 24576),
                row("c", 11.0, [128, 64, 64], 49152),
            ],
            {"a"},
        )
        self.assertEqual(result["retained"], ["b", "c"])
        self.assertEqual(result["winner"], "b")

    def test_empty_complement_is_a_valid_group(self):
        result = MODULE.select_complement(
            [row("a", 10.0, [128, 64, 32], 32768)], {"a"}
        )
        self.assertEqual(result["retained"], [])
        self.assertIsNone(result["winner"])


if __name__ == "__main__":
    unittest.main()
