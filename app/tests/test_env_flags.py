"""Behaviour of the shared ROOP_* environment reader.

These pin the semantics the old per-module copies disagreed about: what a
malformed value does, whether a typo can switch a feature on, and whether a
NaN can reach a blend weight.
"""

import os
import sys
import unittest
from pathlib import Path

APP = Path(__file__).resolve().parents[1]
if str(APP) not in sys.path:
    sys.path.insert(0, str(APP))

from roop.env import env_bool, env_float, env_int, env_raw, env_str


class EnvVarTestCase(unittest.TestCase):
    """Set/restore real environment variables around each test."""

    FLAG = "ROOP_UNIT_TEST_FLAG"

    def setUp(self):
        self._saved = os.environ.get(self.FLAG)
        os.environ.pop(self.FLAG, None)

    def tearDown(self):
        if self._saved is None:
            os.environ.pop(self.FLAG, None)
        else:
            os.environ[self.FLAG] = self._saved

    def set(self, value):
        os.environ[self.FLAG] = value


class TestEnvBool(EnvVarTestCase):
    def test_unset_returns_the_default(self):
        self.assertTrue(env_bool(self.FLAG, True))
        self.assertFalse(env_bool(self.FLAG, False))

    def test_the_shipped_zero_one_vocabulary(self):
        self.set("0")
        self.assertFalse(env_bool(self.FLAG, True))
        self.set("1")
        self.assertTrue(env_bool(self.FLAG, False))

    def test_word_forms_and_case_and_padding(self):
        for raw in ("true", "TRUE", " Yes ", "on"):
            self.set(raw)
            self.assertTrue(env_bool(self.FLAG, False), raw)
        for raw in ("false", "FALSE", " No ", "off"):
            self.set(raw)
            self.assertFalse(env_bool(self.FLAG, True), raw)

    def test_empty_string_is_false_not_a_missing_value(self):
        """`ROOP_X=` on a command line reads as an explicit off."""
        self.set("")
        self.assertFalse(env_bool(self.FLAG, True))

    def test_a_typo_falls_back_and_cannot_silently_enable(self):
        self.set("ture")
        self.assertFalse(env_bool(self.FLAG, False))
        self.assertTrue(env_bool(self.FLAG, True))


class TestEnvInt(EnvVarTestCase):
    def test_parses_and_strips(self):
        self.set(" 12 ")
        self.assertEqual(env_int(self.FLAG, 3), 12)

    def test_malformed_falls_back_to_default(self):
        self.set("eight")
        self.assertEqual(env_int(self.FLAG, 8), 8)

    def test_clamps_to_bounds(self):
        self.set("99")
        self.assertEqual(env_int(self.FLAG, 1, lo=0, hi=16), 16)
        self.set("-99")
        self.assertEqual(env_int(self.FLAG, 1, lo=0, hi=16), 0)

    def test_bounds_also_apply_to_the_default(self):
        self.assertEqual(env_int(self.FLAG, -5, lo=1), 1)


class TestEnvFloat(EnvVarTestCase):
    def test_parses_and_clamps(self):
        self.set("0.75")
        self.assertAlmostEqual(env_float(self.FLAG, 0.0), 0.75)
        self.set("5")
        self.assertAlmostEqual(env_float(self.FLAG, 0.0, lo=0.0, hi=1.0), 1.0)

    def test_malformed_falls_back(self):
        self.set("abc")
        self.assertAlmostEqual(env_float(self.FLAG, 0.25), 0.25)

    def test_nan_and_inf_cannot_reach_a_caller(self):
        for raw in ("nan", "inf", "-inf"):
            self.set(raw)
            self.assertAlmostEqual(env_float(self.FLAG, 0.5), 0.5, msg=raw)

    def test_a_nonfinite_default_degrades_to_zero(self):
        self.set("nan")
        self.assertEqual(env_float(self.FLAG, float("nan")), 0.0)


class TestEnvStr(EnvVarTestCase):
    def test_default_when_unset_or_blank(self):
        self.assertEqual(env_str(self.FLAG, "p5"), "p5")
        self.set("   ")
        self.assertEqual(env_str(self.FLAG, "p5"), "p5")

    def test_lowercases_by_default_and_can_preserve_case(self):
        self.set(" P7 ")
        self.assertEqual(env_str(self.FLAG, "p5"), "p7")
        self.set(" MyPath ")
        self.assertEqual(env_str(self.FLAG, "x", lower=False), "MyPath")


class TestEnvRaw(EnvVarTestCase):
    def test_distinguishes_unset_from_empty(self):
        self.assertIsNone(env_raw(self.FLAG))
        self.set("")
        self.assertEqual(env_raw(self.FLAG), "")


class TestSharedDefaultsAreDeclaredNotInlined(unittest.TestCase):
    """The defect this module exists to prevent.

    A flag read in more than one module used to carry a different default at
    each site, inlined as a string literal, so the same unset flag meant
    different things depending on which entry point read it.
    """

    def test_nvenc_preset_default_is_declared_once(self):
        """This one WAS a bug: p5 in two writers, p4 in a third."""
        from roop import ffmpeg_path, ffmpeg_writer, util_ffmpeg
        from roop import vectorized_pipeline
        canonical = ffmpeg_path.NVENC_PRESET_DEFAULT
        self.assertEqual(canonical, "p5")
        for module in (ffmpeg_writer, util_ffmpeg, vectorized_pipeline):
            self.assertIs(module.NVENC_PRESET_DEFAULT, canonical,
                          f"{module.__name__} must reuse the shared default, "
                          f"not redeclare it")

    def test_strict_trt_defaults_differ_on_purpose_and_are_named(self):
        """This one is NOT a bug, and must not be "fixed" into agreement.

        vectorized_pipeline is the strict-only route: strict mode raises
        rather than falling back, which is the guarantee it exists to provide.
        optimized_prepass is the general path and must still run on a
        batch-one detector. Naming both keeps the divergence reviewable.
        """
        from roop import optimized_prepass, vectorized_pipeline
        self.assertTrue(vectorized_pipeline.STRICT_TRT_DEFAULT)
        self.assertFalse(optimized_prepass.STRICT_TRT_DEFAULT)

    def test_no_module_reinlines_the_nvenc_preset_literal(self):
        """Guard the fix itself: the literals must not creep back."""
        import re
        from pathlib import Path
        roop_dir = Path(__file__).resolve().parents[1] / "roop"
        offenders = []
        pat = re.compile(r"""ROOP_NVENC_PRESET['"]\s*,\s*['"]p\d['"]""")
        for path in roop_dir.rglob("*.py"):
            if pat.search(path.read_text(encoding="utf-8", errors="replace")):
                offenders.append(path.name)
        self.assertEqual(offenders, [],
                         "these modules inline a ROOP_NVENC_PRESET default "
                         "instead of importing NVENC_PRESET_DEFAULT")


if __name__ == "__main__":
    unittest.main()
