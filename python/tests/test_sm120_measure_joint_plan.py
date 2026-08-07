#!/usr/bin/env python3
import pathlib
import sys
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sm120_measure_joint_plan import configure_command  # noqa: E402


class JointPlanBuildWiringTest(unittest.TestCase):
    def test_cute_qkv_compiles_bridge_and_links_object(self) -> None:
        command = configure_command(
            pathlib.Path("/repo"),
            pathlib.Path("/build"),
            pathlib.Path("/active"),
            {"source": "/fa4/active.cpp", "object": "/fa4/active.o"},
            {
                "bridge_source": "/active/active-qkv-cute.cu",
                "object": "/generated/qkv.o",
            },
        )
        self.assertIn(
            "-DSM120_SEARCH_QKV_SOURCE=/active/active-qkv-cute.cu",
            command,
        )
        self.assertIn(
            "-DSM120_SEARCH_QKV_OBJECT=/active/active-qkv-cute.o",
            command,
        )

    def test_planar_qkv_keeps_ordinary_slot_source(self) -> None:
        command = configure_command(
            pathlib.Path("/repo"),
            pathlib.Path("/build"),
            pathlib.Path("/active"),
            {"source": "/fa4/active.cpp", "object": None},
            {"source": "/generated/qkv.cu", "object": None},
        )
        self.assertIn(
            "-DSM120_SEARCH_QKV_SOURCE=/active/active-qkv.cu",
            command,
        )
        self.assertIn("-DSM120_SEARCH_QKV_OBJECT=", command)


if __name__ == "__main__":
    unittest.main()
