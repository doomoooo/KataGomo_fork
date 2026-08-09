import json
import pathlib
import struct
import subprocess
import sys
import tempfile
import unittest

import numpy as np


REPO = pathlib.Path(__file__).resolve().parents[2]
COMPARE = REPO / "python/katago/train/compare_replay_krnn.py"
SECTION_DIMS = (722, 2, 3, 6, 361, 724, 80, 842, 1805, 1086, 7942, 19)


def write_krnn(path: pathlib.Path, batch: int, *, corrupt_input: bool = False) -> None:
    sections = [np.zeros((1, dim), dtype=np.float32) for dim in SECTION_DIMS]
    sections[5][0, 0] = 1.0
    sections[6][0, 25] = 1.0
    if corrupt_input:
        sections[10][0, 0] = 1.0
    metadata = {
        "numRows": 1,
        "posLen": 19,
        "maxBatchSize": batch,
        "fixedBatchTailPadding": True,
        "sections": [
            {"dim": dim, "bytes": dim * 4} for dim in SECTION_DIMS
        ],
    }
    encoded = json.dumps(metadata, separators=(",", ":")).encode()
    with path.open("wb") as out:
        out.write(b"KRNN")
        out.write(struct.pack("<I", len(encoded)))
        out.write(encoded)
        for section in sections:
            out.write(section.tobytes())


class CompareReplayKrnnTests(unittest.TestCase):
    def test_exact_batch_and_input_identity_are_proven_by_comparator(self) -> None:
        with tempfile.TemporaryDirectory() as directory_text:
            directory = pathlib.Path(directory_text)
            reference = directory / "reference.krnn"
            candidate = directory / "candidate.krnn"
            report = directory / "report.json"
            write_krnn(reference, 13)
            write_krnn(candidate, 4)
            command = [
                sys.executable, str(COMPARE),
                "--reference", str(reference),
                "--candidate", str(candidate),
                "--expected-candidate-batch", "4",
                "--output", str(report),
            ]
            completed = subprocess.run(command, text=True, capture_output=True)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            result = json.loads(report.read_text())
            self.assertEqual(result["exactBatch"], 4)
            self.assertEqual(result["candidateMaxBatchSize"], 4)
            self.assertTrue(result["inputAndTargetSectionsByteExact"])

            write_krnn(candidate, 4, corrupt_input=True)
            completed = subprocess.run(command, text=True, capture_output=True)
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("target or input section 10 differs", completed.stderr)

            write_krnn(candidate, 5)
            completed = subprocess.run(command, text=True, capture_output=True)
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("maxBatchSize", completed.stderr)


if __name__ == "__main__":
    unittest.main()
