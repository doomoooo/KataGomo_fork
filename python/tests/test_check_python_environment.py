import importlib.util
import pathlib
import sys
import tempfile
import unittest
from unittest import mock


REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
CHECKER_PATH = (
    REPO_ROOT / "final-migration/environment/check-python-environment.py"
)
SPEC = importlib.util.spec_from_file_location("check_python_environment", CHECKER_PATH)
CHECKER = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = CHECKER
SPEC.loader.exec_module(CHECKER)


class CheckPythonEnvironmentTests(unittest.TestCase):
    def test_checked_in_requirements_are_all_unconditional_exact_pins(self):
        expected, errors = CHECKER.load_exact_requirements(
            CHECKER.DEFAULT_REQUIREMENT_PATHS
        )
        self.assertEqual(errors, [])
        self.assertEqual(len(expected), 80)
        self.assertEqual(expected["torch"][0], "2.13.0+cu132")
        self.assertEqual(expected["cuda-toolkit"][0], "13.3.1")
        self.assertEqual(expected["nvidia-cudnn-cu13"][0], "9.25.0.15")

    def test_range_or_marker_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            requirements = pathlib.Path(directory) / "requirements.txt"
            requirements.write_text("alpha>=1\nbeta==2; python_version > '3'\n")
            expected, errors = CHECKER.load_exact_requirements((requirements,))
        self.assertEqual(expected, {})
        self.assertEqual(len(errors), 2)
        self.assertTrue(all("not one unconditional exact pin" in error for error in errors))

    def test_installed_version_drift_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            requirements = pathlib.Path(directory) / "requirements.txt"
            requirements.write_text("alpha==1.2.3\n")
            with mock.patch.object(
                CHECKER.importlib.metadata, "version", return_value="1.2.4"
            ):
                errors = CHECKER.validate_exact_environment((requirements,))
        self.assertEqual(len(errors), 1)
        self.assertIn("exact pin drift for alpha", errors[0])


if __name__ == "__main__":
    unittest.main()
