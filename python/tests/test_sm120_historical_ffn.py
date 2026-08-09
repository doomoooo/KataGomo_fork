import hashlib
import json
import pathlib
import unittest

from python.sm120_historical_ffn import generate


class HistoricalFfnFrozenSourceTests(unittest.TestCase):
    def test_frozen_manifest_covers_and_hashes_b1_through_b32(self) -> None:
        payload = json.loads(generate.FROZEN_DEVICE_MANIFEST_PATH.read_text())
        entries = {int(item["batch"]): item for item in payload["batches"]}
        self.assertEqual(sorted(entries), list(range(1, 33)))
        for batch, item in entries.items():
            path = generate.FROZEN_DEVICE_ROOT / item["path"]
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), item["sha256"])
            self.assertEqual(item["kernelName"], f"katago_ffn_tilelang_sm120_b{batch}_s361_kernel")

    def test_b19_retains_historical_golden(self) -> None:
        historical = generate.load_manifest()
        source, kernel_name, evidence = generate.load_frozen_device_source(19, historical)
        self.assertEqual(evidence["sourceSha256"], historical["golden"]["b19RenamedDeviceSourceSha256"])
        self.assertIn(kernel_name, source)
        self.assertNotIn("expf(", source)


if __name__ == "__main__":
    unittest.main()
