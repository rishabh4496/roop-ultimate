"""Self-verification for Adaptive LOD model routing and LivePortrait neural gaze retargeting."""

import os
import sys
from pathlib import Path
import unittest

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
APP_DIR = REPO_ROOT / 'app'
for p in (str(REPO_ROOT), str(APP_DIR)):
    if p in sys.path:
        sys.path.remove(p)
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(APP_DIR))

from roop.core import (
    calculate_face_diagonal,
    dispatch_adaptive_lod,
    AdaptiveLODDispatcher,
    AdaptiveLODDecision,
    get_processing_plugins
)


class AdaptiveLODDispatcherTest(unittest.TestCase):
    """Test suite asserting Adaptive Level-of-Detail (LOD) model routing."""

    def test_diagonal_measurement_formula(self):
        """Diagonal must strictly compute D = sqrt(w^2 + h^2)."""
        bbox = [10.0, 20.0, 70.0, 100.0]  # w = 60, h = 80 -> D = 100
        d = calculate_face_diagonal(bbox)
        self.assertAlmostEqual(d, 100.0, places=4)

        # Object with .bbox attribute
        class MockFace:
            def __init__(self, box):
                self.bbox = np.array(box, dtype=np.float32)

        face = MockFace([0.0, 0.0, 30.0, 40.0])  # w = 30, h = 40 -> D = 50
        self.assertAlmostEqual(calculate_face_diagonal(face), 50.0, places=4)

        # Dictionary with 'bbox'
        self.assertAlmostEqual(calculate_face_diagonal({'bbox': [0, 0, 90, 120]}), 150.0, places=4)

    def test_small_bounding_box_selects_lod_0(self):
        """Small bounding box (D < 120px) must select LOD 0, lightweight 128px swapper, and bypass GPEN."""
        # w = 60, h = 70 -> D = sqrt(3600 + 4900) = sqrt(8500) ≈ 92.20px < 120px
        bbox = [10.0, 10.0, 70.0, 80.0]
        decision = dispatch_adaptive_lod(bbox)

        self.assertIsInstance(decision, AdaptiveLODDecision)
        self.assertEqual(decision.lod, 0)
        self.assertEqual(decision['lod'], 0)
        self.assertIn("LOD 0", decision.level)
        self.assertLess(decision.diagonal, 120.0)

        # Swapper model routing: 128px lightweight model
        self.assertEqual(decision.swap_model, 'inswapper')
        self.assertEqual(decision.swap_size, 128)

        # Restoration: GPEN bypassed completely
        self.assertTrue(decision.bypass_gpen)
        self.assertIsNone(decision.enhancer)
        self.assertIsNone(decision.gpen_size)
        self.assertFalse(decision.dermal_injection)

        # Plugins dict contains only faceswap (no GPEN)
        self.assertIn('faceswap', decision.plugins)
        self.assertEqual(decision.plugins['faceswap']['swap_model'], 'inswapper')
        self.assertNotIn('gpen', decision.plugins)

    def test_midground_bounding_box_selects_lod_1(self):
        """Mid-ground bounding box (120px <= D <= 350px) must select LOD 1, RealSwap 256px + GPEN-256."""
        # w = 150, h = 200 -> D = 250px
        bbox = [20.0, 20.0, 170.0, 220.0]
        decision = dispatch_adaptive_lod(bbox, masking_engine='RealityUX')

        self.assertEqual(decision.lod, 1)
        self.assertIn("LOD 1", decision.level)
        self.assertGreaterEqual(decision.diagonal, 120.0)
        self.assertLessEqual(decision.diagonal, 350.0)

        # Swapper model routing: 256px model (RealSwap/RealityUX)
        self.assertEqual(decision.swap_model, 'realswap')
        self.assertEqual(decision.swap_size, 256)

        # Restoration: GPEN-256
        self.assertFalse(decision.bypass_gpen)
        self.assertEqual(decision.gpen_size, 256)
        self.assertEqual(decision.enhancer, 'GPEN 256')
        self.assertEqual(decision.mask_engine, 'RealityUX')
        self.assertFalse(decision.dermal_injection)

        # Plugins check
        self.assertEqual(decision.plugins['faceswap']['swap_model'], 'realswap')
        self.assertEqual(decision.plugins['gpen']['size'], 256)
        self.assertIn('mask_realityux', decision.plugins)

    def test_large_bounding_box_selects_lod_2(self):
        """Large bounding box (D > 350px) must select LOD 2, 512px model + full GPEN-512 + dermal injection."""
        # w = 300, h = 400 -> D = 500px > 350px
        bbox = [50.0, 50.0, 350.0, 450.0]
        decision = AdaptiveLODDispatcher.dispatch(bbox, masking_engine='RealityUX')

        self.assertEqual(decision.lod, 2)
        self.assertIn("LOD 2", decision.level)
        self.assertGreater(decision.diagonal, 350.0)

        # Swapper model routing: 512px model
        self.assertEqual(decision.swap_model, 'simswap_512')
        self.assertEqual(decision.swap_size, 512)

        # Restoration: full GPEN-512 + high-frequency dermal injection
        self.assertFalse(decision.bypass_gpen)
        self.assertEqual(decision.gpen_size, 512)
        self.assertEqual(decision.enhancer, 'GPEN')
        self.assertTrue(decision.dermal_injection)

        # Plugins check
        self.assertEqual(decision.plugins['faceswap']['swap_model'], 'simswap_512')
        self.assertTrue(decision.plugins['faceswap']['dermal_injection'])
        self.assertEqual(decision.plugins['gpen']['size'], 512)
        self.assertIn('mask_realityux', decision.plugins)

    def test_get_processing_plugins_adaptive_routing(self):
        """get_processing_plugins must delegate to Adaptive LOD when enabled."""
        small_face = {'bbox': [0, 0, 50, 50]}  # D ≈ 70.7px -> LOD 0
        plugins_lod0 = get_processing_plugins('RealityUX', target_face=small_face, enable_adaptive_lod=True)
        self.assertEqual(plugins_lod0['faceswap']['swap_model'], 'inswapper')
        self.assertNotIn('gpen', plugins_lod0)

        large_face = {'bbox': [0, 0, 300, 300]}  # D ≈ 424.2px -> LOD 2
        plugins_lod2 = get_processing_plugins('RealityUX', target_face=large_face, enable_adaptive_lod=True)
        self.assertEqual(plugins_lod2['faceswap']['swap_model'], 'simswap_512')
        self.assertTrue(plugins_lod2['faceswap']['dermal_injection'])
        self.assertEqual(plugins_lod2['gpen']['size'], 512)


# NeuralGazeRetargeterTest was removed 2026-09-25 with the code it tested: the
# "LivePortrait gaze retargeter" in frame/face_swapper.py was an identity-matrix
# Gemm on a dead path. Eye handling is Expression_LivePortrait's gaze follow /
# blink sync now; see tests/expression_eye_bench.py and test_expression_gaze_blink.py.


if __name__ == '__main__':
    unittest.main()
