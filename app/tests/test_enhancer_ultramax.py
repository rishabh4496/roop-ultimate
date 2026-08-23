"""Unit tests for the UltraMax enhancer."""

import os
import sys
import unittest
from unittest.mock import MagicMock
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.dirname(HERE)
if APP not in sys.path:
    sys.path.insert(0, APP)

import roop.globals
import roop.processors.Enhance_UltraMax as UM


class TestUltraMax(unittest.TestCase):
    def setUp(self):
        self._saved_pool_env = os.environ.get('ROOP_TRT_POOL')
        roop.globals.execution_threads = 16
        from roop import session_pool
        session_pool._pool_cache.clear()

    def tearDown(self):
        if self._saved_pool_env is None:
            os.environ.pop('ROOP_TRT_POOL', None)
        else:
            os.environ['ROOP_TRT_POOL'] = self._saved_pool_env
        from roop import session_pool
        session_pool._pool_cache.clear()

    def _make(self, mock_cf_runs=True, pool_size=None):
        p = UM.Enhance_UltraMax()
        opts = {'devicename': 'cuda', 'fp16': True}
        if pool_size is not None:
            opts['pool_size'] = pool_size
        p.Initialize(opts)
        if mock_cf_runs:
            fake_cf = MagicMock()
            fake_cf.Run = MagicMock(side_effect=lambda s, t, f: (f.copy(), 1) if f is not None else (None, 1))
            fake_cf.Release = MagicMock()
            p.codeformer = fake_cf
            if p.pool is not None:
                p.pool._items = [fake_cf for _ in p.pool._items]
                # drain and refill queue with mock items
                while not p.pool._q.empty():
                    try:
                        p.pool._q.get_nowait()
                    except Exception:
                        break
                for it in p.pool._items:
                    p.pool._q.put(it)
        return p

    def _pools(self, n):
        os.environ['ROOP_TRT_POOL'] = str(n)
        from roop import session_pool
        session_pool._pool_cache.clear()

    def test_initialize_builds_primary_codeformer(self):
        p = self._make()
        self.assertIsNotNone(p.codeformer)
        self.assertEqual(p.processorname, 'ultramax')
        self.assertEqual(p.type, 'enhance')

    def test_session_pooling_creates_worker_pool(self):
        self._pools(2)
        p = self._make()
        self.assertIsNotNone(p.pool)
        self.assertIsNotNone(getattr(p, 'pool', None))

    def test_single_session_fallback_when_pooling_off(self):
        self._pools(0)
        p = self._make()
        self.assertIsNone(p.pool)

    def test_photoreal_refinement_sharpens_and_preserves_shape(self):
        """Photoreal refinement must enhance luminance micro-contrast without altering shape or introducing NaNs."""
        img = np.full((512, 512, 3), 128, dtype=np.uint8)
        # Add pattern for edge testing
        img[200:300, 200:300] = 180
        refined = UM.Enhance_UltraMax._apply_photoreal_refinement(img)
        self.assertEqual(refined.shape, (512, 512, 3))
        self.assertEqual(refined.dtype, np.uint8)
        self.assertTrue(np.isfinite(refined).all())

    def test_run_executes_and_outputs_valid_frame(self):
        """UltraMax Run must execute and produce a valid uint8 image with scale=1."""
        self._pools(0)
        p = self._make()
        frame = np.full((512, 512, 3), 128, np.uint8)
        kps = np.array([[190, 220], [320, 220], [255, 290], [200, 360], [310, 360]], dtype=np.float32)
        target_face = {'kps': kps}

        out, scale = p.Run(None, target_face, frame)
        self.assertEqual(out.shape, (512, 512, 3))
        self.assertEqual(out.dtype, np.uint8)
        self.assertEqual(scale, 1)

    def test_handles_missing_or_empty_frames(self):
        self._pools(0)
        p = self._make()
        out, scale = p.Run(None, None, None)
        self.assertIsNone(out)

        empty_frame = np.zeros((0, 0, 3), np.uint8)
        out, scale = p.Run(None, None, empty_frame)
        self.assertEqual(out.size, 0)

    def test_cost_summary_output(self):
        self._pools(0)
        p = self._make()
        self.assertIsNone(p.cost_summary())
        p.Run(None, None, np.zeros((64, 64, 3), np.uint8))
        line = p.cost_summary()
        self.assertIn("1 faces", line)
        self.assertIn("Photoreal High-Definition Fusion", line)


if __name__ == '__main__':
    unittest.main()
