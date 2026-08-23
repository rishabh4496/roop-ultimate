"""An explicit pool size must be honoured exactly.

These knobs used to be CLAMPED. `_resolve` silently reduced any
ROOP_TRT_POOL / ROOP_DETMASK_POOL / ROOP_DETECTOR_POOL above `auto * 2` to that
ceiling, so on a 12 GB card (auto 2, ceiling 4) an operator who selected 8 in the
UI got 4 and was told so only in a console line they were unlikely to be reading.
The UI offered a value the backend refused to use — the same class of defect as a
control bound to something nothing reads.

The clamp is gone. The measurement that motivated it is not: an oversubscribed
pool really does collapse throughput (pool=8 measured 2-2.5 fps against 45.3 fps
at pool=2 for the same stage on an RTX 4070), and because TensorRT allocates
context memory on the FIRST INFERENCE the collapse looks like a hang rather than
an out-of-memory error. So the number survives as an advisory WARNING, printed
once per knob, and the value is passed through untouched.

These tests pin both halves: the value is obeyed, and the warning still fires.
"""

import io
import os
import sys
import unittest
from contextlib import redirect_stdout

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.dirname(HERE)
if APP not in sys.path:
    sys.path.insert(0, APP)

from roop import session_pool as sp

POOL_ENVS = ('ROOP_TRT_POOL', 'ROOP_DETMASK_POOL', 'ROOP_DETECTOR_POOL',
             'ROOP_EXPR_POOL')


class PoolTestCase(unittest.TestCase):
    def setUp(self):
        self._saved = {k: os.environ.get(k) for k in POOL_ENVS}
        self._real_vram = sp._detect_vram_gb
        sp._detect_vram_gb = lambda: 12.0        # the card this was measured on
        sp._pool_cache.clear()
        sp._warned.clear()

    def tearDown(self):
        for k, v in self._saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        sp._detect_vram_gb = self._real_vram
        sp._pool_cache.clear()
        sp._warned.clear()

    def _set(self, **kw):
        for k, v in kw.items():
            os.environ[k] = str(v)
        sp._pool_cache.clear()
        sp._warned.clear()


class TestExplicitValuesAreHonoured(PoolTestCase):
    def test_trt_and_detmask_pass_through_untouched(self):
        for want in (1, 2, 3, 4, 6, 8, 12, 16):
            with self.subTest(requested=want):
                self._set(ROOP_TRT_POOL=want, ROOP_DETMASK_POOL=want)
                with redirect_stdout(io.StringIO()):
                    self.assertEqual(sp.pool_size(), want)
                    self.assertEqual(sp.detmask_pool_size(), want)

    def test_detector_pool_passes_through_untouched(self):
        for want in (1, 4, 8, 16):
            with self.subTest(requested=want):
                self._set(ROOP_DETECTOR_POOL=want)
                with redirect_stdout(io.StringIO()):
                    self.assertEqual(sp.detector_pool_size(), want)

    def test_a_value_far_above_the_advisory_is_still_honoured(self):
        """The point of the change: no silent reduction, at any size."""
        self._set(ROOP_TRT_POOL=64)
        with redirect_stdout(io.StringIO()):
            self.assertEqual(sp.pool_size(), 64)

    def test_unset_falls_back_to_the_vram_tiered_default(self):
        for k in POOL_ENVS:
            os.environ.pop(k, None)
        sp._pool_cache.clear()
        with redirect_stdout(io.StringIO()):
            auto_trt, auto_detmask = sp._auto_pool_defaults()
            self.assertEqual(sp.pool_size(), auto_trt)
            self.assertEqual(sp.detmask_pool_size(), auto_detmask)

    def test_a_junk_value_falls_back_rather_than_crashing(self):
        self._set(ROOP_TRT_POOL='banana')
        with redirect_stdout(io.StringIO()):
            self.assertEqual(sp.pool_size(), sp._auto_pool_defaults()[0])


class TestAdvisoryWarning(PoolTestCase):
    def test_warns_once_above_the_measured_safe_size(self):
        self._set(ROOP_TRT_POOL=8)
        buf = io.StringIO()
        with redirect_stdout(buf):
            sp.pool_size()
            sp._pool_cache.clear()
            sp.pool_size()          # asked twice
        out = buf.getvalue()
        self.assertIn('ROOP_TRT_POOL=8', out)
        self.assertIn('Honouring it', out)
        self.assertEqual(out.count('is above the measured-safe'), 1,
                         "the advisory must print once per knob, not per query")

    def test_silent_at_or_below_the_advisory(self):
        self._set(ROOP_TRT_POOL=4)          # 12GB: auto 2, advisory 4
        buf = io.StringIO()
        with redirect_stdout(buf):
            sp.pool_size()
        self.assertNotIn('above the measured-safe', buf.getvalue())

    def test_the_warning_names_the_failure_mode_not_just_a_number(self):
        """Someone who sets 8, sees 0.2 fps and thinks the app hung is the
        entire reason this survived the clamp's removal."""
        self._set(ROOP_DETMASK_POOL=16)
        buf = io.StringIO()
        with redirect_stdout(buf):
            sp.detmask_pool_size()
        out = buf.getvalue()
        self.assertIn('thrashing', out)
        self.assertIn('not a hang', out)


class TestExpressionPoolIsCallable(PoolTestCase):
    """`expression_pool_size` passed 2 args to a 3-arg `_resolve` for as long as
    it existed, so every call raised TypeError. Nothing caught it, and it stayed
    invisible because the stage only initialises when the user turns Expression
    Restore on — at which point the render crashed."""

    def test_it_returns_an_int_instead_of_raising(self):
        for k in POOL_ENVS:
            os.environ.pop(k, None)
        sp._pool_cache.clear()
        with redirect_stdout(io.StringIO()):
            self.assertIsInstance(sp.expression_pool_size(), int)
            self.assertIsInstance(sp.expression_pooling_enabled(), bool)

    def test_it_honours_an_explicit_value(self):
        self._set(ROOP_EXPR_POOL=6)
        with redirect_stdout(io.StringIO()):
            self.assertEqual(sp.expression_pool_size(), 6)


class TestUiOffersWhatTheBackendAccepts(unittest.TestCase):
    def test_dropdown_is_not_the_new_limiter(self):
        """With the clamp gone, the UI list is the only remaining cap, so it has
        to reach past the largest auto default (8, on a 16GB+ card)."""
        import re
        with open(os.path.join(APP, 'api.py'), encoding='utf-8') as f:
            src = f.read()
        m = re.search(r'"pool_sizes":\s*\[(.*?)\]', src, re.S)
        self.assertIsNotNone(m, 'pool_sizes list not found in api.py')
        vals = [int(v) for v in re.findall(r'"(\d+)"', m.group(1))]
        self.assertIn('auto', m.group(1))
        self.assertGreater(max(vals), 8,
                           'the dropdown must offer more than the largest auto '
                           'default, or it silently caps the operator again')


if __name__ == '__main__':
    unittest.main()
