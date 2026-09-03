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
from roop.processors.frame import face_swapper


def create_mock_68_landmarks(
    left_eye_center=(80.0, 95.0),
    right_eye_center=(176.0, 95.0),
    eye_width=30.0,
    eye_height=16.0
) -> np.ndarray:
    """Generate canonical 68-point facial landmark array in 256x256 coordinate space."""
    pts = np.zeros((68, 2), dtype=np.float32)

    # Jaw (0..16)
    for i in range(17):
        pts[i] = [40.0 + i * 10.0, 100.0 + abs(i - 8) * 10.0]

    # Eyebrows (17..26)
    for i in range(5):
        pts[17 + i] = [55.0 + i * 12.0, 70.0]
        pts[22 + i] = [145.0 + i * 12.0, 70.0]

    # Nose (27..35)
    for i in range(9):
        pts[27 + i] = [128.0, 85.0 + i * 7.0]

    # Left eye (36..41)
    lx, ly = float(left_eye_center[0]), float(left_eye_center[1])
    hw, hh = eye_width / 2.0, eye_height / 2.0
    pts[36] = [lx - hw, ly]
    pts[37] = [lx - hw * 0.33, ly - hh]
    pts[38] = [lx + hw * 0.33, ly - hh]
    pts[39] = [lx + hw, ly]
    pts[40] = [lx + hw * 0.33, ly + hh]
    pts[41] = [lx - hw * 0.33, ly + hh]

    # Right eye (42..47)
    rx, ry = float(right_eye_center[0]), float(right_eye_center[1])
    pts[42] = [rx - hw, ry]
    pts[43] = [rx - hw * 0.33, ry - hh]
    pts[44] = [rx + hw * 0.33, ry - hh]
    pts[45] = [rx + hw, ry]
    pts[46] = [rx + hw * 0.33, ry + hh]
    pts[47] = [rx - hw * 0.33, ry + hh]

    # Mouth (48..67)
    pts[48] = [95.0, 180.0]
    pts[54] = [161.0, 180.0]
    for i in range(49, 54):
        pts[i] = [95.0 + (i - 48) * 13.0, 175.0]
    for i in range(55, 60):
        pts[i] = [161.0 - (i - 54) * 13.0, 185.0]
    for i in range(60, 68):
        pts[i] = [105.0 + (i - 60) * 6.0, 180.0]

    return pts


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


class NeuralGazeRetargeterTest(unittest.TestCase):
    """Test suite asserting pupil extraction, gaze displacement vectors, and neural retargeting."""

    def setUp(self):
        self.size = 256
        self.landmarks = create_mock_68_landmarks(
            left_eye_center=(80.0, 95.0),
            right_eye_center=(176.0, 95.0)
        )

    def test_pupil_center_extraction_from_landmarks(self):
        """Pupil centers must be accurately extracted from target facial landmarks."""
        l_pupil, r_pupil = face_swapper.extract_pupil_coordinates(self.landmarks)
        self.assertAlmostEqual(float(l_pupil[0]), 80.0, places=1)
        self.assertAlmostEqual(float(l_pupil[1]), 95.0, places=1)
        self.assertAlmostEqual(float(r_pupil[0]), 176.0, places=1)
        self.assertAlmostEqual(float(r_pupil[1]), 95.0, places=1)

    def test_gaze_displacement_vectors_adjust_correctly(self):
        """Gaze displacement vectors must adjust in sign and magnitude with target gaze shifts."""
        # Baseline: swap face with pupils looking straight at (80, 95) and (176, 95)
        swap_left = np.array([80.0, 95.0], dtype=np.float32)
        swap_right = np.array([176.0, 95.0], dtype=np.float32)

        # Case 1: Target looking to the right (+6px X displacement)
        target_rightward_left = np.array([86.0, 95.0], dtype=np.float32)
        target_rightward_right = np.array([182.0, 95.0], dtype=np.float32)
        disp_rightward = face_swapper.compute_gaze_displacement_vector(target_rightward_left, swap_left)
        self.assertGreater(disp_rightward[0], 0.0)
        self.assertAlmostEqual(float(disp_rightward[0]), 6.0, places=2)
        self.assertAlmostEqual(float(disp_rightward[1]), 0.0, places=2)

        # Case 2: Target looking to the left (-6px X displacement)
        target_leftward_left = np.array([74.0, 95.0], dtype=np.float32)
        disp_leftward = face_swapper.compute_gaze_displacement_vector(target_leftward_left, swap_left)
        self.assertLess(disp_leftward[0], 0.0)
        self.assertAlmostEqual(float(disp_leftward[0]), -6.0, places=2)

        # Case 3: Target looking up (-4px Y displacement)
        target_upward_left = np.array([80.0, 91.0], dtype=np.float32)
        disp_upward = face_swapper.compute_gaze_displacement_vector(target_upward_left, swap_left)
        self.assertLess(disp_upward[1], 0.0)
        self.assertAlmostEqual(float(disp_upward[1]), -4.0, places=2)

    def test_pupil_projection_onto_swapped_face_reduces_disparity(self):
        """Projecting pupil position onto swapped face must align pupil and eliminate gaze mismatch."""
        # Create mock target crop with eyes shifted right (gaze = right)
        target_crop = np.full((self.size, self.size, 3), 180, dtype=np.uint8)
        cv2.circle(target_crop, (86, 95), 5, (25, 25, 25), -1)  # shifted pupil (+6px)
        cv2.circle(target_crop, (182, 95), 5, (25, 25, 25), -1)

        # Create mock swap crop with eyes looking straight (gaze = center)
        swap_crop = np.full((self.size, self.size, 3), 180, dtype=np.uint8)
        cv2.circle(swap_crop, (80, 95), 5, (25, 25, 25), -1)    # center pupil
        cv2.circle(swap_crop, (176, 95), 5, (25, 25, 25), -1)

        # Apply neural gaze retargeting
        retargeted_crop, meta = face_swapper.retarget_eye_gaze(
            swap_crop, target_crop, self.landmarks, strength=1.0)

        self.assertTrue(meta['applied'])
        # Target displacement was +6.0 on left eye
        self.assertAlmostEqual(float(meta['displacement_left'][0]), 6.0, delta=1.0)

        # Post-retargeting pupil center must have shifted towards target (86, 95)
        ret_left = meta['retargeted_left_pupil']
        ret_right = meta['retargeted_right_pupil']
        self.assertGreater(ret_left[0], 83.0)
        self.assertGreater(ret_right[0], 179.0)

    def test_fp16_onnx_gaze_session_executes(self):
        """The lightweight FP16 ONNX gaze-retargeting session must load and execute valid FP16 inference."""
        session = face_swapper.get_gaze_retargeter()
        self.assertIsNotNone(session)

        # Test FP16 inference
        input_name = session.get_inputs()[0].name
        raw_delta = np.array([[5.0, -2.0, 5.0, -2.0]], dtype=np.float16)
        outputs = session.run(None, {input_name: raw_delta})
        self.assertEqual(len(outputs), 1)
        out = outputs[0]
        self.assertEqual(out.dtype, np.float16)
        self.assertEqual(out.shape, (1, 4))
        self.assertFalse(np.isnan(out).any())
        self.assertFalse(np.isinf(out).any())


if __name__ == '__main__':
    unittest.main()
