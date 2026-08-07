import unittest

from python.sm120_benchmark_metrics import (
    require_stable_throughput,
    summarize_throughput,
)


class BenchmarkMetricsTests(unittest.TestCase):
    def test_short_measurement_is_not_a_final_metric(self) -> None:
        summary = summarize_throughput([10.0, 12.0], iterations=80, warmup=15)
        self.assertEqual(summary["measurement_kind"], "short_scan")
        with self.assertRaises(ValueError):
            require_stable_throughput(summary)

    def test_long_measurement_exposes_stable_median(self) -> None:
        summary = summarize_throughput([10.0, 12.0, 11.0], iterations=1000, warmup=30)
        self.assertEqual(summary["measurement_kind"], "long_stable")
        self.assertEqual(summary["stable_long_nn_evals_per_sec"], 11.0)
        self.assertEqual(require_stable_throughput(summary), 11.0)


if __name__ == "__main__":
    unittest.main()
