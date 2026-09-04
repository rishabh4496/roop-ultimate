"""ORT thread overrides run EXACTLY as set -- they are advised, never clamped.

`get_onnx_session_options` used to fold an explicit override through
`min(upper, value)`, so `ROOP_ORT_INTRA_THREADS=6` silently built sessions at 4.
That is the same defect the TensorRT pool guards carried until 0382a70 removed
them: a control that accepts a value and then quietly does something else looks
completely wired from the UI, from the config file and from the environment.

These tests pin the contract in both directions -- an explicit value is honoured
at any size, the advisory fires once and only above the measured-safe number,
and `auto`/junk still fall back to the serial default.
"""
import os
import sys
import unittest
from unittest.mock import patch

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.dirname(HERE)
if APP not in sys.path:
    sys.path.insert(0, APP)

import roop.utilities as utilities


class OrtThreadOverrideTest(unittest.TestCase):

    def setUp(self):
        utilities._ORT_THREAD_ADVISED.clear()

    def _opts(self, **env):
        with patch.dict(os.environ, env, clear=False):
            return utilities.get_onnx_session_options()

    def test_explicit_value_above_the_old_ceiling_is_honoured(self):
        """The exact regression: 6 used to arrive as 4."""
        opts = self._opts(ROOP_ORT_INTRA_THREADS="6",
                          ROOP_ORT_INTER_THREADS="2")
        self.assertIsNotNone(opts)
        self.assertEqual(opts.intra_op_num_threads, 6)
        self.assertEqual(opts.inter_op_num_threads, 2)

    def test_no_ceiling_remains_at_any_size(self):
        for value in (1, 4, 6, 12, 32):
            opts = self._opts(ROOP_ORT_INTRA_THREADS=str(value))
            self.assertEqual(opts.intra_op_num_threads, value,
                             "intra=%d was not honoured" % value)

    def test_inter_op_override_is_not_clamped_to_two(self):
        opts = self._opts(ROOP_ORT_INTER_THREADS="8")
        self.assertEqual(opts.inter_op_num_threads, 8)

    def test_defaults_are_serial_when_unset(self):
        env = {k: "" for k in ("ROOP_ORT_INTRA_THREADS",
                               "ROOP_ORT_INTER_THREADS",
                               "ROOP_RUNTIME_ORT_INTRA_THREADS",
                               "ROOP_RUNTIME_ORT_INTER_THREADS")}
        opts = self._opts(**env)
        self.assertEqual(opts.intra_op_num_threads, 1)
        self.assertEqual(opts.inter_op_num_threads, 1)

    def test_auto_and_junk_fall_back_to_the_default(self):
        for raw in ("auto", "default", "not-a-number", ""):
            opts = self._opts(ROOP_ORT_INTRA_THREADS=raw,
                              ROOP_RUNTIME_ORT_INTRA_THREADS="")
            self.assertEqual(opts.intra_op_num_threads, 1,
                             "%r should fall back, not raise" % raw)

    def test_zero_and_negative_floor_at_one(self):
        for raw in ("0", "-4"):
            opts = self._opts(ROOP_ORT_INTRA_THREADS=raw)
            self.assertEqual(opts.intra_op_num_threads, 1)

    def test_runtime_hint_is_used_only_when_the_explicit_key_is_auto(self):
        opts = self._opts(ROOP_ORT_INTRA_THREADS="auto",
                          ROOP_RUNTIME_ORT_INTRA_THREADS="5")
        self.assertEqual(opts.intra_op_num_threads, 5)
        utilities._ORT_THREAD_ADVISED.clear()
        opts = self._opts(ROOP_ORT_INTRA_THREADS="3",
                          ROOP_RUNTIME_ORT_INTRA_THREADS="5")
        self.assertEqual(opts.intra_op_num_threads, 3)

    def test_advisory_fires_once_per_knob_and_only_above_the_threshold(self):
        printed = []
        with patch("builtins.print", lambda *a, **k: printed.append(a[0])):
            self._opts(ROOP_ORT_INTRA_THREADS="4")     # at the threshold
            self.assertEqual(printed, [], "4 is measured-safe, must not warn")
            self._opts(ROOP_ORT_INTRA_THREADS="6")     # above it
            self._opts(ROOP_ORT_INTRA_THREADS="6")     # again, same knob
        self.assertEqual(len(printed), 1,
                         "the advisory must be once per knob, not once per "
                         "session -- there are 50+ construction sites")
        self.assertIn("ROOP_ORT_INTRA_THREADS", printed[0])
        self.assertIn("Honoured exactly", printed[0])

    def test_sequential_execution_mode_is_still_forced(self):
        import onnxruntime
        opts = self._opts(ROOP_ORT_INTRA_THREADS="6")
        self.assertEqual(opts.execution_mode,
                         onnxruntime.ExecutionMode.ORT_SEQUENTIAL)


class OrtGraphOptimizationTest(unittest.TestCase):
    """ORT_ENABLE_ALL is the default, and two models deliberately opt out.

    Forcing ALL globally would be a regression, not an optimization: the
    CodeFormer fp16 export trips SimplifiedLayerNormFusion at ALL and the
    session then fails to BUILD on the CPU provider. UltraMax wraps the same
    weights. Both pass ORT_ENABLE_EXTENDED explicitly for that reason.
    """

    def test_default_level_is_enable_all(self):
        import onnxruntime
        opts = utilities.get_onnx_session_options()
        self.assertEqual(opts.graph_optimization_level,
                         onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL)

    def test_extended_is_respected_when_a_model_asks_for_it(self):
        import onnxruntime
        opts = utilities.get_onnx_session_options(
            optimization_level=onnxruntime.GraphOptimizationLevel.ORT_ENABLE_EXTENDED)
        self.assertEqual(opts.graph_optimization_level,
                         onnxruntime.GraphOptimizationLevel.ORT_ENABLE_EXTENDED)

    def test_codeformer_and_ultramax_still_request_extended(self):
        """Guard the reason, not just the value.

        If someone 'optimizes' these two call sites up to ALL, the sessions
        stop building on the CPU provider -- which looks like an enhancer that
        works right up until a fallback happens.
        """
        for rel in ("roop/processors/Enhance_UltraMax.py",
                    "roop/processors/Enhance_CodeFormer.py"):
            with open(os.path.join(APP, rel), "r", encoding="utf-8") as fh:
                body = fh.read()
            self.assertIn("ORT_ENABLE_EXTENDED", body,
                          "%s must keep its EXTENDED opt-out" % rel)


if __name__ == "__main__":
    unittest.main()
