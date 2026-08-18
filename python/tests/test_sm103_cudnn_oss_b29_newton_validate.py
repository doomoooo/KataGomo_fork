import json
import pathlib
import subprocess
import sys
import unittest


PYTHON_DIR = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PYTHON_DIR))

import sm103_cudnn_oss_b29_newton_validate as validator  # noqa: E402


class Sm103CudnnOssB29NewtonValidateTests(unittest.TestCase):
    def test_sass_counter_counts_instructions_only(self) -> None:
        sass = """
Function : kernel
        /*0010*/ MUFU.EX2 R1, R2 ;
        /*0020*/ MUFU.RCP R3, R4 ;
        /*0030*/ FFMA R5, R6, R7, R8 ;
        /*0040*/ CALL.REL.NOINC 0x80 ;
symbol_that_mentions_CALL_and_DIV
"""
        self.assertEqual(
            validator._sass_counts(sass),
            {"call": 1, "div": 0, "mufu_ex2": 1, "mufu_rcp": 1, "fma": 1},
        )

    def test_benchmark_requires_explicit_gpu_acknowledgement(self) -> None:
        with self.assertRaisesRegex(
            validator.NewtonValidationError, "explicit --allow-gpu"
        ):
            validator.benchmark(
                allow_gpu=False,
                library_path=pathlib.Path("missing.so"),
                object_path=pathlib.Path("missing.o"),
            )

    def test_default_cli_is_device_free(self) -> None:
        probe = (
            "import json,sys; import sm103_cudnn_oss_b29_newton_validate; "
            "roots={'torch','cudnn','cutlass','cuda','triton'}; "
            "print(json.dumps(sorted(n for n in sys.modules "
            "if n.split('.')[0] in roots)))"
        )
        completed = subprocess.run(
            [sys.executable, "-c", probe],
            cwd=PYTHON_DIR,
            check=True,
            text=True,
            capture_output=True,
        )
        self.assertEqual(json.loads(completed.stdout), [])

    def test_parser_exposes_stable_timing_defaults(self) -> None:
        args = validator.build_parser().parse_args([])
        self.assertEqual(args.warmup, 20000)
        self.assertEqual(args.iterations, 1000)
        self.assertEqual(args.repeats, 5)


if __name__ == "__main__":
    unittest.main()
