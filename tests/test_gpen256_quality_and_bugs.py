import os
import sys
import unittest
import numpy as np
import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
APP = os.path.join(ROOT, 'app')
if APP not in sys.path:
    sys.path.insert(0, APP)

from roop.one_euro import EnhancerStabilizer, MaskStabilizer
from roop.face_overlap import build_regions, _same_face
from roop.ProcessMgr import ProcessMgr
from roop.temporal_smoother import HighFrequencyFlowStabilizer
from roop.processors.Enhance_GPEN256Pro import Enhance_GPEN256Pro


class DummyFace:
    def __init__(self, bbox, kps, track_id=None):
        self.bbox = np.asarray(bbox, dtype=np.float32)
        self.kps = np.asarray(kps, dtype=np.float32)
        self.det_score = 0.95
        if track_id is not None:
            self._track_id = track_id


class TestGPEN256QualityAndBugs(unittest.TestCase):
    def test_bug1_two_target_faces_no_smudge_enhancer_stabilizer(self):
        """When 2 target faces are selected/swapped, EnhancerStabilizer must NOT
        cross-contaminate Face B with Face A's texture even when bounding boxes touch.
        """
        es = EnhancerStabilizer(strength=0.6)
        
        # Face A (blueish)
        crop_a = np.full((256, 256, 3), 100, dtype=np.uint8)
        crop_a[:, :, 0] = 250
        kps_a = np.array([[100, 100], [150, 100], [125, 125], [110, 150], [140, 150]], dtype=np.float32)

        # Face B (reddish), nearby/touching
        crop_b = np.full((256, 256, 3), 100, dtype=np.uint8)
        crop_b[:, :, 2] = 250
        kps_b = kps_a + np.array([20.0, 10.0], dtype=np.float32)

        # Apply on frame 0
        out_a0 = es.apply(crop_a, kps_a, t=0, track_id=0)
        out_b0 = es.apply(crop_b, kps_b, t=0, track_id=1)
        self.assertEqual(len(es.tracks), 2, "Tracks must maintain separate entries per track_id")

        # Over 20 frames, Face B must not be smudged with Face A's blue channel
        for t in range(1, 21):
            out_a = es.apply(crop_a, kps_a, t=t, track_id=0)
            out_b = es.apply(crop_b, kps_b, t=t, track_id=1)
            # Face B blue channel should stay 100, not drift towards Face A's 250
            self.assertEqual(int(out_b[128, 128, 0]), 100,
                             f"Face B blue channel contaminated on frame {t}!")
            # Face A red channel should stay 100, not drift towards Face B's 250
            self.assertEqual(int(out_a[128, 128, 2]), 100,
                             f"Face A red channel contaminated on frame {t}!")

    def test_bug1_two_target_faces_mask_stabilizer_isolation(self):
        """MaskStabilizer must also isolate tracks by track_id."""
        ms = MaskStabilizer(strength=0.5)
        mask_a = np.ones((256, 256), dtype=np.float32) * 0.9
        mask_b = np.ones((256, 256), dtype=np.float32) * 0.2
        kps_a = np.array([[100, 100], [150, 100], [125, 125], [110, 150], [140, 150]], dtype=np.float32)
        kps_b = kps_a + np.array([15.0, 5.0], dtype=np.float32)

        out_a0 = ms.apply(mask_a, kps_a, t=0, track_id=0)
        out_b0 = ms.apply(mask_b, kps_b, t=0, track_id=1)
        self.assertEqual(len(ms.tracks), 2)

        for t in range(1, 10):
            out_a = ms.apply(mask_a, kps_a, t=t, track_id=0)
            out_b = ms.apply(mask_b, kps_b, t=t, track_id=1)
            self.assertAlmostEqual(float(out_a.mean()), 0.9, places=2)
            self.assertAlmostEqual(float(out_b.mean()), 0.2, places=2)

    def test_bug2_long_run_quality_retention_in_temporal_smoother(self):
        """Verify HighFrequencyFlowStabilizer does NOT compound blur on long runs."""
        hf_stab = HighFrequencyFlowStabilizer(weight=0.15, enabled=True)
        # Create a textured image
        rng = np.random.default_rng(42)
        base = rng.normal(128, 20, (256, 256, 3)).clip(0, 255).astype(np.uint8)
        
        # Initial Laplacian variance / high-frequency energy
        initial_lap = cv2.Laplacian(base, cv2.CV_64F).var()
        
        cur = base.copy()
        for frame_idx in range(60):
            # Slight subpixel motion/jitter
            jitter = rng.normal(0, 0.5, (256, 256, 3))
            frame = np.clip(base + jitter, 0, 255).astype(np.uint8)
            cur = hf_stab.stabilize(frame, track_id=1, frame_index=frame_idx)

        final_lap = cv2.Laplacian(cur, cv2.CV_64F).var()
        # Sharpness should be retained (>= 75% of initial variance, not decaying to 20%)
        ratio = final_lap / initial_lap
        self.assertGreater(ratio, 0.75, f"High frequency energy collapsed on frame 60: ratio={ratio:.3f}")

    def test_bug3_half_face_swap_duplicate_detection_suppressed(self):
        """Verify concentric duplicate detections do NOT slice a face in half in build_regions."""
        # Face 1: normal detection
        kps1 = np.array([[100, 100], [140, 100], [120, 120], [105, 140], [135, 140]], dtype=np.float32)
        f1 = DummyFace(bbox=[80, 80, 160, 160], kps=kps1)
        
        # Face 2: concentric duplicate detection of the exact same head (slightly offset box)
        kps2 = kps1 + np.array([2.0, -1.0], dtype=np.float32)
        f2 = DummyFace(bbox=[82, 78, 158, 162], kps=kps2)

        faces = [f1, f2]
        frame_shape = (480, 640, 3)
        order = [0, 1]  # Both were claimed/painted

        regions = build_regions(faces, frame_shape, order=order)
        # One of the duplicate claims must be suppressed, meaning there are NOT 2 competing regions
        # carving the head down the middle.
        if regions is not None:
            # If regions is not None, at most 1 face should have a region, or regions should not carve f1
            self.assertTrue(0 not in regions or 1 not in regions,
                            f"Both duplicate faces received carved regions: {list(regions.keys())}")

    def test_procmgr_temporal_track_id_resolution(self):
        """Verify _temporal_track_id correctly extracts track_id from both dicts and objects."""
        d_face = {'_track_id': 7}
        self.assertEqual(ProcessMgr._temporal_track_id(d_face), 7)

        d_face2 = {'track_id': 12}
        self.assertEqual(ProcessMgr._temporal_track_id(d_face2), 12)

        obj_face = DummyFace(bbox=[0, 0, 10, 10], kps=np.zeros((5, 2)), track_id=42)
        self.assertEqual(ProcessMgr._temporal_track_id(obj_face), 42)

        none_face = DummyFace(bbox=[0, 0, 10, 10], kps=np.zeros((5, 2)), track_id=None)
        self.assertIsNone(ProcessMgr._temporal_track_id(none_face))

    def test_gpen256_pro_sharpness_floor_preservation(self):
        """Verify GPEN 256 Pro sharpness floor retains eye/lip feature contrast on soft frames."""
        cls = Enhance_GPEN256Pro
        # Create a completely smooth/flat face (skin_gate -> 1.0)
        smooth = np.full((256, 256, 3), 140, dtype=np.uint8)
        # Add high-frequency eye/lash edge
        smooth[120:130, 100:150] = 30
        
        res = cls._enhance_textures_and_sharpness_cpu(smooth, smooth, input_size=256)
        self.assertEqual(res.shape, (512, 512, 3))
        # Edge contrast should be preserved
        edge_diff = int(res[250, 250, 0]) - int(res[240, 250, 0])
        self.assertGreater(abs(edge_diff), 10)


if __name__ == '__main__':
    unittest.main()
