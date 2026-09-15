"""Regression coverage for the launcher's explicit provider override.

The public ``run.py --execution-provider`` switch is parsed once by run.py and
again by ``roop.core``.  Keep both halves covered so a CPU acceptance run cannot
silently fall back to the provider saved in config.yaml.
"""

import argparse
import os
import sys
import unittest
from unittest.mock import patch

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.dirname(HERE)
if APP not in sys.path:
    sys.path.insert(0, APP)

import roop.core as core  # noqa: E402
import roop.globals as globals_  # noqa: E402


class LauncherProviderOverrideTests(unittest.TestCase):
    def setUp(self):
        self._argv = sys.argv[:]
        self._startup_args = globals_.startup_args

    def tearDown(self):
        sys.argv[:] = self._argv
        globals_.startup_args = self._startup_args

    def test_core_parser_accepts_the_launcher_provider_switch(self):
        with patch.object(sys, "argv", ["run.py", "--execution-provider", "cpu"]):
            core.parse_args()
        self.assertEqual(globals_.startup_args.execution_provider, "cpu")

    def test_explicit_override_wins_in_memory_without_rewriting_config(self):
        cfg = argparse.Namespace(provider="tensorrt")
        globals_.startup_args = argparse.Namespace(execution_provider="cpu")

        result = core.apply_provider_override(cfg)

        self.assertIs(result, cfg)
        self.assertEqual(cfg.provider, "cpu")

    def test_missing_override_preserves_saved_provider(self):
        cfg = argparse.Namespace(provider="tensorrt")
        globals_.startup_args = argparse.Namespace(execution_provider=None)

        core.apply_provider_override(cfg)

        self.assertEqual(cfg.provider, "tensorrt")


if __name__ == "__main__":
    unittest.main()
