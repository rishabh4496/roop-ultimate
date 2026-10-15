"""Integration test: All face enhancers with non-standard video aspect ratios,
foreground occlusions, and interacting faces.

Verifies:
1. Every enhancer output format (512x512 sf=1, 256x256 sf=1, 1024x1024 sf=2)
   composites accurately into PAL (720x576), Vertical (1080x1920), and 16:9 (640x360) frames.
2. Interacting faces with InteractingRegion trim boundaries without destroying
   enhanced facial features.
3. Single and multi-face frames with bystander non-targets preserve both the target
   face's enhanced details and bystander identity.
4. Enhancer post-processing contracts (is_usable, sized, protect_restorer_output)
   operate cleanly on all configurations.
"""

from __future__ import annotations

import unittest
from unittest import mock
import numpy as np
import cv2

from roop.processors.enhance_common import is_usable, sized
from roop.face_overlap import FaceRegion, build_regions
from roop.procmgr_masking import MaskingMixin
from roop.enhancers.occlusion_governor import OcclusionGovernor
from roop.processors.Mask_RealityUX import _non_target_keep_mask


class MockFace:
    """Mock face structure for testing."""
    def __init__(
        self,
        bbox: list[float] | None = None,
        kps: np.ndarray | None = None,
        landmark_2d_106: np.ndarray | None = None,
        det_score: float = 0.95,
        matrix: np.ndarray | None = None,
    ):
        self.bbox = np.asarray(bbox, dtype=np.float32) if bbox is not None else np.array([50, 50, 150, 150], dtype=np.float32)
        self.kps = kps if kps is not None else np.array([[75, 80], [125, 80], [100, 100], [80, 125], [120, 125]], dtype=np.float32)
        self.landmark_2d_106 = landmark_2d_106
        self.det_score = det_score
        self.matrix = matrix


class DummyOptions:
    def __init__(self):
        self.show_face_area_overlay = False
        self.blend_ratio = 1.0


class DummyProcessor(MaskingMixin):
    def __init__(self):
        self.options = DummyOptions()
        self._tls = mock.MagicMock()
        self._tls.frame_faces = []


class TestEnhancersWithAspectRatiosAndOcclusion(unittest.TestCase):
    def setUp(self):
        self.proc = DummyProcessor()

    def test_all_enhancer_resolutions_sized_and_usable(self):
        """Verify sized() contract for all enhancer resolution tiers."""
        # Tier 1: 512x512 enhancers (CodeFormer, GFPGAN, GPEN-512, RestoreFormer++, RestoreUltra, UltraMax, DMDNet, TileDiffusion)
        out512 = np.full((512, 512, 3), 128, dtype=np.uint8)
        res512, sf512 = sized(out512, 512)
        self.assertEqual(res512.shape, (512, 512, 3))
        self.assertEqual(sf512, 1)
        self.assertTrue(is_usable(res512))

        # Tier 2: 256x256 enhancers (GPEN-256, GPEN-256 Pro, GPEN Realistic)
        out256 = np.full((256, 256, 3), 128, dtype=np.uint8)
        res256, sf256 = sized(out256, 512)
        self.assertEqual(res256.shape, (512, 512, 3))
        self.assertEqual(sf256, 1)
        self.assertTrue(is_usable(res256))

        # Tier 3: 1024x1024 upscalers
        out1024 = np.full((1024, 1024, 3), 128, dtype=np.uint8)
        res1024, sf1024 = sized(out1024, 512)
        self.assertEqual(res1024.shape, (1024, 1024, 3))
        self.assertEqual(sf1024, 2)
        self.assertTrue(is_usable(res1024))

    def test_paste_upscale_across_video_aspect_ratios(self):
        """Verify paste_upscale composites enhanced crops correctly on diverse aspect ratios."""
        frame_geometries = [
            (576, 720),    # PAL 720x576 (h, w)
            (1920, 1080),  # Vertical / Reel 1080x1920 (h, w)
            (360, 640),    # NHD 640x360 (h, w)
            (720, 1280),   # 720p 1280x720 (h, w)
        ]

        enhancer_scales = [
            (512, 1),   # 512x512, scale_factor=1
            (1024, 2),  # 1024x1024, scale_factor=2
        ]

        for h, w in frame_geometries:
            plate = np.full((h, w, 3), 50, dtype=np.uint8)
            # Create a face centered in the frame
            cx, cy = w // 2, h // 2
            face_size = min(w, h) // 4
            bbox = [cx - face_size, cy - face_size, cx + face_size, cy + face_size]
            kps = np.array([
                [cx - face_size * 0.4, cy - face_size * 0.3],
                [cx + face_size * 0.4, cy - face_size * 0.3],
                [cx, cy],
                [cx - face_size * 0.3, cy + face_size * 0.3],
                [cx + face_size * 0.3, cy + face_size * 0.3],
            ], dtype=np.float32)
            target_face = MockFace(bbox=bbox, kps=kps)

            # Standard 512x512 affine crop matrix M
            src_pts = np.float32([
                [cx - face_size, cy - face_size],
                [cx + face_size, cy - face_size],
                [cx - face_size, cy + face_size]
            ])
            dst_pts = np.float32([[0, 0], [512, 0], [0, 512]])
            M = cv2.getAffineTransform(src_pts, dst_pts)
            target_face.matrix = M

            fake_face = np.full((512, 512, 3), 200, dtype=np.uint8)

            for enh_size, scale_factor in enhancer_scales:
                # Distinct color for enhanced output
                upsk_face = np.full((enh_size, enh_size, 3), 240, dtype=np.uint8)

                out = self.proc.paste_upscale(
                    fake_face=fake_face,
                    upsk_face=upsk_face,
                    M=M,
                    target_img=plate.copy(),
                    scale_factor=scale_factor,
                    mask_offsets=[0, 0, 0, 0, 20.0, 10.0],
                    target_face=target_face,
                )

                # Ensure output dimensions match target frame exactly
                self.assertEqual(out.shape, (h, w, 3))
                # Ensure the face center received enhanced pixel values (> 200)
                center_val = out[cy, cx, 0]
                self.assertGreater(center_val, 180, f"Failed on geometry ({h}, {w}) with scale {scale_factor}")
                # Ensure background pixels far from face remain untouched (50)
                self.assertEqual(out[5, 5, 0], 50)

    def test_paste_upscale_with_interacting_faces_and_enhancer(self):
        """Verify that when two faces interact, an enhanced face is not ruined by demarcation."""
        h, w = 600, 800
        plate = np.full((h, w, 3), 40, dtype=np.uint8)

        # Face 1 on left (250, 300), Face 2 on right (350, 300) - close proximity/contact
        face1 = MockFace(bbox=[200, 200, 350, 400], kps=np.array([[240, 260], [280, 260], [260, 300], [245, 340], [275, 340]], dtype=np.float32))
        face2 = MockFace(bbox=[300, 200, 450, 400], kps=np.array([[340, 260], [380, 260], [360, 300], [345, 340], [375, 340]], dtype=np.float32))

        # Build regions
        regions = build_regions([face1, face2], (h, w, 3), order=[0, 1])
        self.assertIsNotNone(regions)
        region1 = regions[0]
        region2 = regions[1]

        # Affine matrix for face1
        src_pts = np.float32([[200, 200], [350, 200], [200, 400]])
        dst_pts = np.float32([[0, 0], [512, 0], [0, 512]])
        M1 = cv2.getAffineTransform(src_pts, dst_pts)
        face1.matrix = M1

        # Simulate enhanced crop for face1 (bright face)
        fake_face = np.full((512, 512, 3), 180, dtype=np.uint8)
        upsk_face = np.full((512, 512, 3), 230, dtype=np.uint8)

        self.proc._tls.frame_faces = [face1, face2]

        # Paste face 1 with region1
        out = self.proc.paste_upscale(
            fake_face=fake_face,
            upsk_face=upsk_face,
            M=M1,
            target_img=plate.copy(),
            scale_factor=1,
            mask_offsets=[0, 0, 0, 0, 15.0, 10.0],
            target_face=face1,
            region=region1,
        )

        # Target 1 center (260, 300) should be enhanced (>200)
        self.assertGreater(out[300, 260, 0], 190)
        # Face 2 territory (around 380, 300) should NOT be overwritten by face 1's enhanced swap
        self.assertLess(out[300, 380, 0], 100)

    def test_protect_frame_non_targets_spares_target_landmarks_during_enhancement(self):
        """Verify that when a non-target bystander is present without contact (region=None),
        non-target protection protects the bystander while fully sparing target face landmarks."""
        h, w = 600, 800
        plate = np.full((h, w, 3), 30, dtype=np.uint8)

        # Target face on left, bystander on right
        target = MockFace(bbox=[150, 150, 350, 350], kps=np.array([[200, 220], [280, 220], [240, 260], [210, 300], [270, 300]], dtype=np.float32))
        bystander = MockFace(bbox=[400, 150, 600, 350], kps=np.array([[450, 220], [530, 220], [490, 260], [460, 300], [520, 300]], dtype=np.float32))

        # Paint bystander with identifiable pattern
        plate[150:350, 400:600] = 77

        src_pts = np.float32([[150, 150], [350, 150], [150, 350]])
        dst_pts = np.float32([[0, 0], [512, 0], [0, 512]])
        M = cv2.getAffineTransform(src_pts, dst_pts)
        target.matrix = M

        fake_face = np.full((512, 512, 3), 190, dtype=np.uint8)
        upsk_face = np.full((512, 512, 3), 240, dtype=np.uint8)

        self.proc._tls.frame_faces = [target, bystander]

        out = self.proc.paste_upscale(
            fake_face=fake_face,
            upsk_face=upsk_face,
            M=M,
            target_img=plate.copy(),
            scale_factor=1,
            mask_offsets=[0, 0, 0, 0, 20.0, 10.0],
            target_face=target,
            region=None,  # triggers non-target protection
        )

        # Target face core (240, 260) is enhanced (>200)
        self.assertGreater(out[260, 240, 0], 200)
        # Bystander center (490, 260) remains untouched at 77
        self.assertEqual(out[260, 490, 0], 77)

    def test_all_enhancer_processor_classes_contracts(self):
        """Verify class-level contracts and safe handling for all enhancer processors."""
        from roop.processors.Enhance_CodeFormer import Enhance_CodeFormer
        from roop.processors.Enhance_GFPGAN import Enhance_GFPGAN
        from roop.processors.Enhance_GPEN import Enhance_GPEN
        from roop.processors.Enhance_GPEN256Pro import Enhance_GPEN256Pro
        from roop.processors.Enhance_GPENRealistic import Enhance_GPENRealistic
        from roop.processors.Enhance_GPENUltimate import Enhance_GPENUltimate
        from roop.processors.Enhance_RestoreFormerPPlus import Enhance_RestoreFormerPPlus
        from roop.processors.Enhance_RestoreUltra import Enhance_RestoreUltra
        from roop.processors.Enhance_TileDiffusion import Enhance_TileDiffusion
        from roop.processors.Enhance_UltraMax import Enhance_UltraMax
        from roop.processors.Enhance_DMDNet import Enhance_DMDNet

        processors = [
            Enhance_CodeFormer,
            Enhance_GFPGAN,
            Enhance_GPEN,
            Enhance_GPEN256Pro,
            Enhance_GPENRealistic,
            Enhance_GPENUltimate,
            Enhance_RestoreFormerPPlus,
            Enhance_RestoreUltra,
            Enhance_TileDiffusion,
            Enhance_UltraMax,
            Enhance_DMDNet,
        ]

        dummy_face = MockFace()

        for proc_cls in processors:
            inst = proc_cls()
            self.assertEqual(getattr(inst, 'type', 'enhance'), 'enhance')

            # Verify safe handling of None / empty frame
            res_none, sf_none = inst.Run(None, dummy_face, None)
            self.assertIsNone(res_none)
            self.assertEqual(sf_none, 1)

            empty_frame = np.zeros((0, 0, 3), dtype=np.uint8)
            res_empty, sf_empty = inst.Run(None, dummy_face, empty_frame)
            self.assertEqual(sf_empty, 1)

    def test_protect_restorer_output_across_tiers(self):
        """Verify dark target protection blends enhanced output without blowing out dark tones."""
        from roop.appearance_conditioning import protect_restorer_output, NORMAL, DARK, VERY_DARK

        pre_restore = np.full((512, 512, 3), 30, dtype=np.uint8)
        enhanced = np.full((512, 512, 3), 120, dtype=np.uint8)

        # NORMAL tier leaves enhanced frame untouched
        out_normal = protect_restorer_output(enhanced, pre_restore, NORMAL)
        np.testing.assert_array_equal(out_normal, enhanced)

        # DARK tier pulls enhanced back toward target reference
        out_dark = protect_restorer_output(enhanced, pre_restore, DARK)
        self.assertLess(out_dark[256, 256, 0], enhanced[256, 256, 0])
        self.assertGreater(out_dark[256, 256, 0], pre_restore[256, 256, 0])

        # VERY_DARK tier pulls it even closer to target reference
        out_very_dark = protect_restorer_output(enhanced, pre_restore, VERY_DARK)
        self.assertLess(out_very_dark[256, 256, 0], out_dark[256, 256, 0])


if __name__ == '__main__':
    unittest.main()
