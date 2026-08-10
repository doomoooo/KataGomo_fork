import unittest
from unittest import mock

from python import build_parallelism


class BuildParallelismTests(unittest.TestCase):
    def test_memory_limit_caps_large_cpu_count(self) -> None:
        with (
            mock.patch.object(build_parallelism.os, "sched_getaffinity", return_value=set(range(128))),
            mock.patch.object(build_parallelism, "available_memory_bytes", return_value=64 * 1024**3),
        ):
            self.assertEqual(build_parallelism.conservative_build_jobs(), 8)

    def test_cpu_count_caps_large_memory_host(self) -> None:
        with (
            mock.patch.object(build_parallelism.os, "sched_getaffinity", return_value=set(range(16))),
            mock.patch.object(build_parallelism, "available_memory_bytes", return_value=512 * 1024**3),
        ):
            self.assertEqual(build_parallelism.conservative_build_jobs(), 8)

    def test_tiny_memory_still_allows_one_job(self) -> None:
        with (
            mock.patch.object(build_parallelism.os, "sched_getaffinity", return_value=set(range(8))),
            mock.patch.object(build_parallelism, "available_memory_bytes", return_value=1024**3),
        ):
            self.assertEqual(build_parallelism.conservative_build_jobs(), 1)


if __name__ == "__main__":
    unittest.main()
