import hashlib
import json
import pathlib
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch


PYTHON_DIR = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PYTHON_DIR))

from official_backend_baseline import (  # noqa: E402
    benchmark_command,
    build_parser,
    official_backend_overrides,
    run_baseline,
)


class OfficialBackendBaselineTests(unittest.TestCase):
    def _inputs(self, directory: pathlib.Path) -> tuple[pathlib.Path, ...]:
        binary = directory / "katago"
        config = directory / "benchmark.cfg"
        model = directory / "model.bin.gz"
        binary.write_bytes(b"binary")
        binary.chmod(0o755)
        config.write_text("config\n", encoding="utf-8")
        model.write_bytes(b"model")
        return binary, config, model

    def _args(
        self,
        directory: pathlib.Path,
        *,
        backend: str = "cuda",
        batches: str = "4,8",
        iterations: int = 100,
        repeats: int = 2,
        max_attempts: int = 2,
    ):
        binary, config, model = self._inputs(directory)
        return build_parser().parse_args([
            "--backend", backend,
            "--binary", str(binary),
            "--config", str(config),
            "--model", str(model),
            "--device", "3",
            "--streams", "2",
            "--batches", batches,
            "--iterations", str(iterations),
            "--warmup", "7",
            "--repeats", str(repeats),
            "--output", str(directory / "result.json"),
            "--raw-dir", str(directory / "raw"),
            "--home-data-dir", str(directory / "katago-home"),
            "--timeout", "12.5",
            "--max-attempts", str(max_attempts),
        ])

    def test_cuda_overrides_are_an_explicit_official_control(self):
        values = official_backend_overrides("cuda", 3, 2, 13)
        self.assertEqual(values, {
            "nnMaxBatchSize": 13,
            "numNNServerThreadsPerModel": 2,
            "requireMaxBoardSize": True,
            "useFP16": True,
            "cudaSm89Backend": False,
            "cudaSm89Forward": False,
            "cudaSm120Backend": False,
            "cudaUseNHWC": True,
            "cudaWarmupOnlyMaxBatchSize": True,
            "cudaDeviceToUseThread0": 3,
            "cudaDeviceToUseThread1": 3,
        })

    def test_tensorrt_overrides_force_onnx_nhwc_and_thread_devices(self):
        values = official_backend_overrides("tensorrt", 1, 3, 32)
        self.assertEqual(values, {
            "nnMaxBatchSize": 32,
            "numNNServerThreadsPerModel": 3,
            "requireMaxBoardSize": True,
            "useFP16": True,
            "trtDisableOnnx": False,
            "trtTransformerNHWC": True,
            "trtDeviceToUseThread0": 1,
            "trtDeviceToUseThread1": 1,
            "trtDeviceToUseThread2": 1,
        })
        self.assertFalse(any(key.startswith("cuda") for key in values))

    def test_command_uses_whole_graph_json_benchmark(self):
        command = benchmark_command(
            binary=pathlib.Path("/bin/katago"),
            config=pathlib.Path("/cfg/bench.cfg"),
            model=pathlib.Path("/models/b11.bin.gz"),
            overrides={"useFP16": True, "nnMaxBatchSize": 19},
            batch=19,
            iterations=1000,
            warmup=50,
        )
        self.assertEqual(command[:4], [
            "/bin/katago", "benchmarknn", "-config", "/cfg/bench.cfg",
        ])
        self.assertEqual(
            command[command.index("-override-config") + 1],
            "nnMaxBatchSize=19,useFP16=true",
        )
        self.assertEqual(
            command[command.index("-batch-size") + 1], "19",
        )
        self.assertEqual(command[-3:], ["-boardsize", "19", "-json"])

    def test_serial_scan_records_identity_raw_runs_and_ranking(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            args = self._args(root)
            calls: list[tuple[int, int]] = []
            sample_index = {4: 0, 8: 0}
            values = {4: (100.0, 102.0), 8: (120.0, 118.0)}

            def benchmark(command, *, device, timeout):
                batch = int(command[command.index("-batch-size") + 1])
                index = sample_index[batch]
                sample_index[batch] += 1
                calls.append((batch, device))
                stdout = json.dumps({
                    "combinedNNEvalsPerSec": values[batch][index],
                }) + "\n"
                return (
                    subprocess.CompletedProcess(command, 0, stdout, "log\n"),
                    False,
                    {
                        "samples": 3,
                        "foreign_active_sm_pids": [],
                        "error": None,
                        "sm_conflict_retries": [],
                    },
                )

            device = {
                "ordinal": 3,
                "name": "NVIDIA B300",
                "compute_capability": [10, 3],
            }
            with (
                patch(
                    "official_backend_baseline._query_cuda_device",
                    return_value=device,
                ),
                patch(
                    "official_backend_baseline._run_benchmark_with_occupancy",
                    side_effect=benchmark,
                ),
            ):
                payload = run_baseline(args)

            self.assertEqual(calls, [(4, 3), (4, 3), (8, 3), (8, 3)])
            self.assertEqual(payload["status"], "completed")
            self.assertEqual(
                payload["inputs"]["home_data_dir"],
                str((root / "katago-home").resolve()),
            )
            self.assertEqual(
                payload["rows"][0]["overrides"]["homeDataDir"],
                str((root / "katago-home").resolve()),
            )
            self.assertEqual(payload["measurement_mode"], "short_scan")
            self.assertEqual(payload["device"], device)
            self.assertEqual(
                [item["batch"] for item in payload["ranking"]], [8, 4],
            )
            self.assertEqual([row["rank"] for row in payload["rows"]], [2, 1])
            self.assertEqual(
                payload["identity"]["binary_sha256"],
                hashlib.sha256(b"binary").hexdigest(),
            )
            self.assertEqual(
                payload["identity"]["config_sha256"],
                hashlib.sha256(b"config\n").hexdigest(),
            )
            self.assertEqual(
                payload["identity"]["model_sha256"],
                hashlib.sha256(b"model").hexdigest(),
            )
            self.assertEqual(len(payload["rows"][0]["runs"]), 2)
            self.assertEqual(
                len(list((root / "raw").glob("*.out"))), 4,
            )
            self.assertEqual(
                json.loads((root / "result.json").read_text()), payload,
            )

    def test_external_sm_conflict_is_discarded_and_retried(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            args = self._args(
                root, batches="4", repeats=1, max_attempts=2,
            )
            command_result = subprocess.CompletedProcess
            results = [
                (
                    command_result([], 0, '{"combinedNNEvalsPerSec":999}\n', ""),
                    False,
                    {
                        "samples": 1,
                        "foreign_active_sm_pids": [917],
                        "error": None,
                    },
                ),
                (
                    command_result([], 0, '{"combinedNNEvalsPerSec":100}\n', ""),
                    False,
                    {
                        "samples": 2,
                        "foreign_active_sm_pids": [],
                        "error": None,
                    },
                ),
            ]
            with (
                patch(
                    "official_backend_baseline._query_cuda_device",
                    return_value={"ordinal": 3, "name": "B300"},
                ),
                patch(
                    "official_backend_baseline._run_benchmark_with_occupancy",
                    side_effect=results,
                ),
            ):
                payload = run_baseline(args)

            run = payload["rows"][0]["runs"][0]
            self.assertEqual(run["throughput"], 100.0)
            self.assertEqual(len(run["attempts"]), 2)
            self.assertTrue(run["attempts"][0]["discarded"])
            self.assertIn(
                "external SM work", run["attempts"][0]["discard_reason"],
            )
            self.assertTrue(pathlib.Path(run["attempts"][0]["stdout"]).is_file())

    def test_missing_metric_fails_closed_and_persists_diagnostics(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            args = self._args(
                root, batches="4", repeats=1, max_attempts=2,
            )
            result = (
                subprocess.CompletedProcess([], 0, "no metric\n", "details\n"),
                False,
                {
                    "samples": 1,
                    "foreign_active_sm_pids": [],
                    "error": None,
                },
            )
            with (
                patch(
                    "official_backend_baseline._query_cuda_device",
                    return_value={"ordinal": 3, "name": "B300"},
                ),
                patch(
                    "official_backend_baseline._run_benchmark_with_occupancy",
                    side_effect=[result, result],
                ),
            ):
                with self.assertRaisesRegex(RuntimeError, "failed for B4"):
                    run_baseline(args)

            payload = json.loads((root / "result.json").read_text())
            self.assertEqual(payload["status"], "failed")
            self.assertEqual(payload["rows"][0]["status"], "failed")
            attempts = payload["rows"][0]["runs"][0]["attempts"]
            self.assertEqual(len(attempts), 2)
            self.assertTrue(all(attempt["discarded"] for attempt in attempts))
            self.assertEqual(len(list((root / "raw").glob("*.err"))), 2)

    def test_long_confirmation_requires_stable_repeated_samples(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            args = self._args(
                root, batches="4", iterations=1000, repeats=2,
            )
            results = [
                (
                    subprocess.CompletedProcess(
                        [], 0, '{"combinedNNEvalsPerSec":100}\n', "",
                    ),
                    False,
                    {"foreign_active_sm_pids": [], "error": None},
                ),
                (
                    subprocess.CompletedProcess(
                        [], 0, '{"combinedNNEvalsPerSec":130}\n', "",
                    ),
                    False,
                    {"foreign_active_sm_pids": [], "error": None},
                ),
            ]
            with (
                patch(
                    "official_backend_baseline._query_cuda_device",
                    return_value={"ordinal": 3, "name": "B300"},
                ),
                patch(
                    "official_backend_baseline._run_benchmark_with_occupancy",
                    side_effect=results,
                ),
            ):
                with self.assertRaisesRegex(RuntimeError, "unstable for B4"):
                    run_baseline(args)

            payload = json.loads((root / "result.json").read_text())
            self.assertEqual(payload["measurement_mode"], "long_confirmation")
            self.assertEqual(payload["status"], "failed")
            self.assertEqual(
                payload["rows"][0]["measurement_kind"], "long_unstable",
            )


if __name__ == "__main__":
    unittest.main()
