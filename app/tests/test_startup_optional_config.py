"""Exercise the actual early startup function without importing GPU/server code.

All config reads happen in a temporary working directory. No live settings,
models, dependencies, frontend build or running process are touched.
"""
import ast
import contextlib
import io
import os
from pathlib import Path
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from roop import degrade


class OptionalStartupConfig(unittest.TestCase):
    def setUp(self):
        source = Path(__file__).resolve().parents[1] / "run.py"
        tree = ast.parse(source.read_text(encoding="utf-8"))
        function = next(node for node in tree.body
                        if isinstance(node, ast.FunctionDef)
                        and node.name == "_apply_perf_env")
        namespace = {"os": os, "_swallowed": degrade.swallowed}
        exec(compile(ast.Module(body=[function], type_ignores=[]),
                     str(source), "exec"), namespace)
        self.apply = namespace["_apply_perf_env"]
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        previous_cwd = os.getcwd()
        os.chdir(self.temp.name)
        self.addCleanup(os.chdir, previous_cwd)
        self.env = patch.dict(os.environ, {}, clear=True)
        self.env.start()
        self.addCleanup(self.env.stop)
        degrade.reset()
        self.addCleanup(degrade.reset)
        self.optimizer = Mock()
        self.optimizer.startup_profile.return_value = SimpleNamespace(
            hardware=SimpleNamespace(gpu_name="test", vram_total_gb=6.0),
            tuning=SimpleNamespace(worker_count=1, trt_context_count=0,
                                   queue_depth=1))
        self.optimizer.apply_environment.return_value = {}
        module = ModuleType("roop.runtime_optimizer")
        module.RuntimeOptimizer = Mock(return_value=self.optimizer)
        self.factory = module.RuntimeOptimizer
        modules = patch.dict("sys.modules", {"roop.runtime_optimizer": module})
        modules.start()
        self.addCleanup(modules.stop)

    def run_startup(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):
            self.apply()
        return output.getvalue()

    def test_missing_config_is_normal_not_a_pinokio_error(self):
        os.environ["ROOP_TRT_POOL"] = "0"
        before = dict(os.environ)
        self.assertEqual(self.run_startup(), "")
        self.assertEqual(degrade.report(), [])
        self.assertEqual(dict(os.environ), before)
        self.factory.assert_not_called()
        self.assertFalse(Path("config.yaml").exists())

    def test_missing_config_is_normal_even_in_strict_mode(self):
        os.environ["ROOP_STRICT_FALLBACK"] = "1"
        self.assertEqual(self.run_startup(), "")
        self.assertEqual(degrade.report(), [])

    def test_existing_device_pools_and_look_preferences_are_preserved(self):
        for pool in (0, 2):
            with self.subTest(pool=pool), patch.dict(os.environ, {}, clear=True):
                content = (f"perf_trt_pool: {pool}\nperf_detmask_pool: {pool}\n"
                           f"perf_detector_pool: {pool}\nblend_ratio: 0.85\n"
                           "face_mask_blend: 25\nmerger_sharpen: 0.55\n"
                           "stabilize_enhancer_strength: 0.6\n")
                Path("config.yaml").write_text(content, encoding="utf-8")
                before = Path("config.yaml").read_bytes()
                self.run_startup()
                for key in ("ROOP_TRT_POOL", "ROOP_DETMASK_POOL", "ROOP_DETECTOR_POOL"):
                    self.assertEqual(os.environ[key], str(pool))
                self.assertEqual(Path("config.yaml").read_bytes(), before)
                self.assertEqual(degrade.report(), [])

    def test_explicit_environment_still_wins(self):
        Path("config.yaml").write_text("perf_trt_pool: 2\n", encoding="utf-8")
        os.environ["ROOP_TRT_POOL"] = "0"
        self.run_startup()
        self.assertEqual(os.environ["ROOP_TRT_POOL"], "0")

    def test_empty_config_retains_existing_default_path(self):
        Path("config.yaml").write_text("", encoding="utf-8")
        self.run_startup()
        self.factory.assert_called_once_with(settings={})
        self.assertEqual(degrade.report(), [])

    def test_malformed_config_is_still_reported(self):
        Path("config.yaml").write_text("perf_trt_pool: [\n", encoding="utf-8")
        self.assertIn("[Fallback]", self.run_startup())
        self.assertEqual(len(degrade.report()), 1)
        self.factory.assert_not_called()

    def test_permission_failure_is_not_mistaken_for_absent_config(self):
        with patch("builtins.open", side_effect=PermissionError(13, "denied")):
            self.assertIn("PermissionError", self.run_startup())
        self.assertEqual(len(degrade.report()), 1)
        self.factory.assert_not_called()


if __name__ == "__main__":
    unittest.main()
