import unittest

from python.sm120_tactic_search import candidate_space


class TacticSearchSpaceTests(unittest.TestCase):
    def test_fa4_n96_is_global_candidate(self) -> None:
        for gpu_class in ("rtx5080", "rtx5090d"):
            for batch in (1, 13, 32):
                candidates = candidate_space(batch, gpu_class)["fa4"]
                tile_ns = [candidate.get("tile_n") for candidate in candidates]
                self.assertEqual(tile_ns[:3], [64, 96, 128])


if __name__ == "__main__":
    unittest.main()
