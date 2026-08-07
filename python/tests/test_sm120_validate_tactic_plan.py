import unittest

from python.sm120_validate_tactic_plan import median_for_arm, sequence_for


class ValidateTacticPlanTests(unittest.TestCase):
    def test_interleaved_orders(self):
        self.assertEqual(sequence_for("abba"), ["A", "B", "B", "A"])
        self.assertEqual(sequence_for("baab"), ["B", "A", "A", "B"])
        self.assertEqual(
            sequence_for("both"), ["A", "B", "B", "A", "B", "A", "A", "B"]
        )

    def test_arm_medians(self):
        values = [(10.0, "A"), (8.0, "B"), (12.0, "B"), (14.0, "A")]
        self.assertEqual(median_for_arm(values, "A"), 12.0)
        self.assertEqual(median_for_arm(values, "B"), 10.0)


if __name__ == "__main__":
    unittest.main()
