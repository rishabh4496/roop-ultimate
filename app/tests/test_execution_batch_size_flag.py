"""--execution-batch-size must reach ProcessMgr's batch ceiling and stay there."""

import argparse
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import settings  # noqa: E402


class BatchSizeArgTest(unittest.TestCase):

    def test_accepts_positive_integers(self):
        self.assertEqual(settings.batch_size_arg("4"), 4)
        self.assertEqual(settings.batch_size_arg(" 8 "), 8)
        self.assertEqual(settings.batch_size_arg("1"), 1)

    def test_rejects_zero_negative_and_junk(self):
        for bad in ("0", "-2", "four", "", "2.5"):
            with self.subTest(bad=bad):
                with self.assertRaises(argparse.ArgumentTypeError):
                    settings.batch_size_arg(bad)

    def test_argparse_reports_a_clean_error(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--execution-batch-size", type=settings.batch_size_arg)
        with self.assertRaises(SystemExit):
            parser.parse_args(["--execution-batch-size", "0"])


class ApplyTest(unittest.TestCase):

    def test_none_is_a_no_op(self):
        env = {}
        self.assertIsNone(settings.apply_execution_batch_size(None, env))
        self.assertEqual(env, {})

    def test_sets_the_variable_processmgr_reads(self):
        env = {}
        self.assertEqual(settings.apply_execution_batch_size(4, env), 4)
        self.assertEqual(env["ROOP_BATCH_SWAP_MAX"], "4")

    def test_beats_a_value_config_already_exported(self):
        cfg = {"perf_batch_max": "2"}
        env = {}
        settings.apply_env(cfg, env)
        self.assertEqual(env.get("ROOP_BATCH_SWAP_MAX"), "2")
        settings.apply_execution_batch_size(8, env)
        self.assertEqual(env["ROOP_BATCH_SWAP_MAX"], "8")

    def test_survives_a_later_ui_save(self):
        """apply_live_env re-derives settings-OWNED variables; the flag must not be one."""
        env = os.environ
        saved = env.get("ROOP_BATCH_SWAP_MAX")
        try:
            env.pop("ROOP_BATCH_SWAP_MAX", None)
            settings.apply_env({"perf_batch_max": "2"}, env)       # owned by settings
            self.assertIn("ROOP_BATCH_SWAP_MAX", settings._SETTINGS_OWNED_VARS)
            settings.apply_execution_batch_size(8)                  # the CLI flag
            self.assertNotIn("ROOP_BATCH_SWAP_MAX", settings._SETTINGS_OWNED_VARS)
            settings.apply_live_env({"perf_batch_max": "2"})        # a UI save
            self.assertEqual(env["ROOP_BATCH_SWAP_MAX"], "8")
        finally:
            settings._SETTINGS_OWNED_VARS.discard("ROOP_BATCH_SWAP_MAX")
            if saved is None:
                env.pop("ROOP_BATCH_SWAP_MAX", None)
            else:
                env["ROOP_BATCH_SWAP_MAX"] = saved


class BothParsersAcceptTheFlagTest(unittest.TestCase):
    """core.run() re-parses run.py's argv; a flag only one of them knows is an error."""

    def test_both_declare_it(self):
        here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        for rel in ("run.py", os.path.join("roop", "core.py")):
            with self.subTest(file=rel):
                src = open(os.path.join(here, rel), encoding="utf-8").read()
                self.assertIn("--execution-batch-size", src)


if __name__ == "__main__":
    unittest.main()
