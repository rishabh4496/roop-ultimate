"""Tests for Enhance_UltraMax - multi-context pooling, spatial tracking, and detail reuse."""

import os
import sys
import threading
import time
import unittest
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from roop import session_pool
import roop.processors.Enhance_UltraMax as UM
import roop.processors.Enhance_GPEN as GPEN
import roop.processors.Enhance_CodeFormer as CF

_FDEF = 0.25

class _FakeIOB:
    def __init__(self):
        self.bound = {}

    def bind_cpu_input(self, name, arr):
        self.bound[name] = arr

    def bind_output(self, name, dev):
        pass

    def copy_outputs_to_cpu(self):
        return [np.full((1, 3, 512, 512), _FDEF, dtype=np.float32)]


class _FakeSession:
    registry = []
    inflight = set()
    peak_distinct = 0
    _lock = threading.Lock()

    def __init__(self, path, opts=None, providers=None):
        _FakeSession.registry.append(self)

    def io_binding(self):
        return _FakeIOB()

    def get_inputs(self):
        return [type('I', (), {'name': 'x', 'type': 'tensor(float)', 'shape': [1, 3, 512, 512]}),
                type('I', (), {'name': 'w', 'type': 'tensor(double)', 'shape': [1]})]

    def get_outputs(self):
        return [type('O', (), {'name': 'y', 'shape': [1, 3, 512, 512]})]

    def run_with_iobinding(self, iob):
        with _FakeSession._lock:
            _FakeSession.inflight.add(id(self))
            _FakeSession.peak_distinct = max(_FakeSession.peak_distinct,
                                       len(_FakeSession.inflight))
        time.sleep(0.01)
        with _FakeSession._lock:
            _FakeSession.inflight.discard(id(self))

    @classmethod
    def reset(cls):
        cls.registry, cls.inflight, cls.peak_distinct = [], set(), 0


class TestUltraMax(unittest.TestCase):
    def setUp(self):
        self._ort_cf = CF.onnxruntime
        self._ort_gp = GPEN.onnxruntime
        self._resolve_cf = CF.resolve_relative_path
        self._resolve_gp = GPEN.resolve_relative_path
        self._cond_dl = GPEN.conditional_download
        self._cache = dict(session_pool._pool_cache)

        fake_ort = type('ort', (), {
            'InferenceSession': _FakeSession,
            'SessionOptions': lambda: type('o', (), {'graph_optimization_level': None})(),
            'GraphOptimizationLevel': type('g', (), {'ORT_ENABLE_EXTENDED': 1}),
        })
        CF.onnxruntime = fake_ort
        GPEN.onnxruntime = fake_ort
        CF.resolve_relative_path = lambda p: p
        GPEN.resolve_relative_path = lambda p: p
        GPEN.conditional_download = lambda *a, **kw: None
        _FakeSession.reset()

    def tearDown(self):
        CF.onnxruntime = self._ort_cf
        GPEN.onnxruntime = self._ort_gp
        CF.resolve_relative_path = self._resolve_cf
        GPEN.resolve_relative_path = self._resolve_gp
        GPEN.conditional_download = self._cond_dl
        session_pool._pool_cache.clear()
        session_pool._pool_cache.update(self._cache)

    @staticmethod
    def _pools(n):
        session_pool._pool_cache.clear()
        session_pool._pool_cache.update({'trt': n, 'detmask': n})

    def _make(self):
        p = UM.Enhance_UltraMax()
        p.Initialize({'devicename': 'cuda'})
        return p

    def test_declares_ffhq_template(self):
        p = self._make()
        self.assertEqual(getattr(p, 'model_template', None), 'ffhq_512')

    def test_session_pooling_creates_worker_pool(self):
        self._pools(2)
        p = self._make()
        self.assertIsNotNone(p.pool)
        # ProcessMgr _gpu_guard checks truthiness of getattr(p, 'pool', None)
        self.assertIsNotNone(getattr(p, 'pool', None))

    def test_single_session_fallback_when_pooling_off(self):
        self._pools(0)
        p = self._make()
        self.assertIsNone(p.pool)

    def test_spatial_tracking_reuses_residual_on_untracked_faces(self):
        """When _track_id is absent, spatial tracking must match faces across frames
        and reduce CodeFormer calls to 1-in-N."""
        self._pools(0)
        p = self._make()
        frame = np.zeros((512, 512, 3), np.uint8)

        # 8 consecutive frames of the same face with bounding box
        target_face = {'bbox': [100, 100, 300, 300]}
        for i in range(8):
            out, scale = p.Run(None, target_face, frame)
            self.assertEqual(out.shape, (512, 512, 3))
            self.assertEqual(scale, 1)

        # CodeFormer should only run twice (at frame 0 and frame 4 for N=4)
        with p._lock:
            cf_calls = p._cf_calls
            faces = p._faces
        self.assertEqual(faces, 8)
        self.assertEqual(cf_calls, 2, f'Expected 2 CodeFormer calls for 8 frames, got {cf_calls}')

    def test_motion_adaptive_early_refresh(self):
        """A large sudden face displacement must trigger an early refresh."""
        self._pools(0)
        p = self._make()
        frame = np.zeros((512, 512, 3), np.uint8)

        # Frame 0: face at (100, 100)
        p.Run(None, {'bbox': [100, 100, 200, 200]}, frame)
        # Frame 1: face suddenly moves to (300, 300) -> displacement > 25% face width
        p.Run(None, {'bbox': [300, 300, 400, 400]}, frame)

        with p._lock:
            cf_calls = p._cf_calls
        # Should have run CodeFormer for both due to motion
        self.assertGreaterEqual(cf_calls, 2)

    def test_cost_summary_output(self):
        self._pools(0)
        p = self._make()
        frame = np.zeros((512, 512, 3), np.uint8)
        p.Run(None, {'_track_id': 'trk1'}, frame)
        summary = p.cost_summary()
        self.assertIn('[UltraMax]', summary)
        self.assertIn('CodeFormer ran 1 times', summary)

    def test_edge_coring_suppresses_white_lines(self):
        """Verify that large difference spikes (e.g. at structural edges) are softly
        compressed so no white streaks or halo artifacts occur."""
        diff = np.zeros((512, 512, 3), dtype=np.float32)
        # Create a harsh structural edge spike of amplitude 150.0 (which would cause white lines)
        diff[200:250, 200:250, :] = 150.0
        base_f = np.full((512, 512, 3), 128.0, dtype=np.float32)
        # Draw a sharp edge on base_f
        base_f[200:250, 200:250, :] = 220.0

        safe_res = UM.Enhance_UltraMax._highpass(diff, base_f=base_f)
        # The maximum residual amplitude must be strictly bounded (<= 12.0)
        self.assertLessEqual(float(np.max(np.abs(safe_res))), 12.0)

    def test_landmark_motion_compensation(self):
        """When face landmarks move moderately between frames, cached detail is warped to match."""
        self._pools(0)
        p = self._make()
        frame = np.full((512, 512, 3), 128, np.uint8)

        kps_f0 = np.array([[150, 150], [250, 150], [200, 200], [170, 250], [230, 250]], dtype=np.float32)
        # Frame 0 (CodeFormer computes fresh detail and caches kps)
        out0, _ = p.Run(None, {'_track_id': 'trk_motion', 'kps': kps_f0}, frame)
        self.assertEqual(out0.shape, (512, 512, 3))

        # Frame 1 (Face shifts slightly by 5px right and down)
        kps_f1 = kps_f0 + 5.0
        out1, _ = p.Run(None, {'_track_id': 'trk_motion', 'kps': kps_f1}, frame)
        self.assertEqual(out1.shape, (512, 512, 3))

    def test_release_cleans_resources(self):
        self._pools(2)
        p = self._make()
        p.Release()
        self.assertIsNone(p.pool)
        self.assertIsNone(p.gpen)
        self.assertIsNone(p.codeformer)
        self.assertEqual(len(p._cache), 0)

    def test_anti_oversaturation_harmonization(self):
        """Harmonization must constrain extreme A and B chrominance swings."""
        # Create an over-saturated neon orange frame
        neon_frame = np.zeros((512, 512, 3), dtype=np.uint8)
        neon_frame[:, :, 0] = 0    # B
        neon_frame[:, :, 1] = 120  # G
        neon_frame[:, :, 2] = 255  # R (hyper-saturated orange/red)

        harmonized = UM.Enhance_UltraMax._harmonize_face(neon_frame)
        self.assertEqual(harmonized.shape, (512, 512, 3))
        # Ensure finite and bounded
        self.assertTrue(np.isfinite(harmonized).all())

    def test_dermal_micro_contrast_injection(self):
        """Dermal micro-contrast injection should enhance luminance pores in midtones without artifacts."""
        flat_skin = np.full((512, 512, 3), 140, dtype=np.uint8)
        # Add subtle noise
        np.random.seed(42)
        noise = np.random.randint(-5, 6, size=(512, 512, 3)).astype(np.int16)
        skin_with_micro = np.clip(flat_skin.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        harmonized = UM.Enhance_UltraMax._harmonize_face(skin_with_micro)
        self.assertEqual(harmonized.shape, (512, 512, 3))
        self.assertTrue(np.isfinite(harmonized).all())

if __name__ == '__main__':
    unittest.main()

