"""A preview must not shrink the FaceAnalysis pool a render built.

`session_pool.detmask_pooling_enabled()` returns False whenever
`roop.globals.is_preview` is set, so `_ensure_face_analyser` used to compute a
target width of 1 for every preview. That width is also its cache key, so the
pool flipped 2 -> 1 on entering a preview and 1 -> 2 on leaving it, and each
flip is a full teardown plus rebuild of every FaceAnalysis instance --
det_10g + landmark_2d_106 + w600k_r50 apiece, with their TensorRT engines.

`core.live_swap` runs on every /api/preview, so the ordinary
tweak-a-setting-then-render loop paid that round trip continuously. A render
log covering five pipeline initialisations showed the pool rebuilt four times;
the retinaface pool, which caches and never re-sizes, was built exactly once in
the same log.

A wider pool is harmless for a preview -- it leases one instance and returns
it -- so the fix is to accept whatever is already resident.
"""

import os
import sys
import unittest
from unittest import mock

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.dirname(HERE)
if APP not in sys.path:
    sys.path.insert(0, APP)

import roop.globals                                              # noqa: E402
from roop import face_util                                       # noqa: E402


class _FakeAnalyser:
    """Stands in for a FaceAnalysis instance; identity is what we assert on."""


class PreviewReusesAResidentPool(unittest.TestCase):

    def setUp(self):
        self._saved = {
            'pool': face_util.FACE_ANALYSER_POOL,
            'primary': face_util.FACE_ANALYSER,
            'q': face_util._ANALYSER_Q,
            'det_size': face_util._ANALYSER_DET_SIZE,
            'det_thresh': face_util._ANALYSER_DET_THRESH,
            'engine': face_util._ANALYSER_ENGINE,
            'lm68': face_util._ANALYSER_LM68_LAZY,
            'current': roop.globals.g_current_face_analysis,
            'desired': roop.globals.g_desired_face_analysis,
            'preview': getattr(roop.globals, 'is_preview', False),
        }
        self.built = []

    def tearDown(self):
        face_util.FACE_ANALYSER_POOL = self._saved['pool']
        face_util.FACE_ANALYSER = self._saved['primary']
        face_util._ANALYSER_Q = self._saved['q']
        face_util._ANALYSER_DET_SIZE = self._saved['det_size']
        face_util._ANALYSER_DET_THRESH = self._saved['det_thresh']
        face_util._ANALYSER_ENGINE = self._saved['engine']
        face_util._ANALYSER_LM68_LAZY = self._saved['lm68']
        roop.globals.g_current_face_analysis = self._saved['current']
        roop.globals.g_desired_face_analysis = self._saved['desired']
        roop.globals.is_preview = self._saved['preview']

    def _seat_a_rendered_pool(self, width):
        """Put the module in the state a finished render leaves it in."""
        modules = ['landmark_2d_106', 'detection', 'recognition']
        roop.globals.g_desired_face_analysis = modules
        roop.globals.g_current_face_analysis = modules
        face_util.FACE_ANALYSER_POOL = [_FakeAnalyser() for _ in range(width)]
        face_util.FACE_ANALYSER = face_util.FACE_ANALYSER_POOL[0]
        face_util._ANALYSER_DET_SIZE = face_util._desired_det_size()
        face_util._ANALYSER_DET_THRESH = getattr(
            roop.globals, 'face_detector_threshold', 0.60)
        face_util._ANALYSER_ENGINE = face_util._current_engine()
        face_util._ANALYSER_LM68_LAZY = bool(
            getattr(roop.globals, 'lm68_lazy', False))

    def _run(self, is_preview):
        """Drive _ensure_face_analyser with pooling reporting preview state."""
        roop.globals.is_preview = is_preview

        def _build():
            a = _FakeAnalyser()
            self.built.append(a)
            return a

        with mock.patch.object(face_util.session_pool,
                               'detmask_pooling_enabled',
                               return_value=not is_preview), \
             mock.patch.object(face_util.session_pool, 'detmask_pool_size',
                               return_value=2), \
             mock.patch.object(face_util, '_build_face_analyser',
                               side_effect=_build):
            return face_util._ensure_face_analyser()

    def test_preview_reuses_the_wide_pool_instead_of_rebuilding_it(self):
        self._seat_a_rendered_pool(2)
        before = list(face_util.FACE_ANALYSER_POOL)
        self._run(is_preview=True)
        self.assertEqual(self.built, [], 'preview rebuilt the pool')
        self.assertEqual(list(face_util.FACE_ANALYSER_POOL), before)

    def test_the_render_after_a_preview_finds_its_pool_intact(self):
        """The other half of the round trip: no rebuild on the way back."""
        self._seat_a_rendered_pool(2)
        self._run(is_preview=True)
        before = list(face_util.FACE_ANALYSER_POOL)
        self._run(is_preview=False)
        self.assertEqual(self.built, [], 'render rebuilt the pool after a preview')
        self.assertEqual(list(face_util.FACE_ANALYSER_POOL), before)

    def test_a_cold_preview_still_builds_the_narrow_pool(self):
        """Reuse must not become "always build wide" -- with nothing resident a
        preview is still a single instance."""
        roop.globals.g_desired_face_analysis = [
            'landmark_2d_106', 'detection', 'recognition']
        roop.globals.g_current_face_analysis = None
        face_util.FACE_ANALYSER_POOL = []
        face_util.FACE_ANALYSER = None
        self._run(is_preview=True)
        self.assertEqual(len(self.built), 1)
        self.assertEqual(len(face_util.FACE_ANALYSER_POOL), 1)

    def test_a_render_still_widens_a_pool_a_cold_preview_left_behind(self):
        """The one rebuild that IS necessary must still happen."""
        self._seat_a_rendered_pool(1)
        self._run(is_preview=False)
        self.assertEqual(len(face_util.FACE_ANALYSER_POOL), 2)

    def test_a_genuine_setting_change_still_rebuilds_during_a_preview(self):
        """Reuse is keyed on WIDTH only. A different module set, det_size,
        threshold or engine must still invalidate the pool, or a preview would
        silently render with the previous run's detector."""
        self._seat_a_rendered_pool(2)
        roop.globals.g_desired_face_analysis = [
            'landmark_3d_68', 'landmark_2d_106', 'detection', 'recognition']
        self._run(is_preview=True)
        self.assertTrue(self.built, 'a changed module set did not rebuild')


if __name__ == '__main__':
    unittest.main()
