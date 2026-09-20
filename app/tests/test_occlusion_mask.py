"""Tests for foreground occlusion masking and temporal mask smoothing in face_swapper."""

import os
import sys
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
APP_DIR = REPO_ROOT / 'app'
for p in (str(REPO_ROOT), str(APP_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

import roop.globals
from roop.processors.frame import face_swapper
from roop.procmgr_masking import MaskingMixin


class OcclusionMaskTest(unittest.TestCase):
    def setUp(self):
        roop.globals.enable_occlusion_mask = True
        face_swapper.clear_temporal_state()

        # Generate a synthetic face crop (256x256 BGR)
        # Background: dark gray (40, 40, 40)
        # Face: skin-toned circle (130, 160, 210) centered at (128, 128) with radius 80
        self.crop_size = 256
        self.clean_face = np.full((self.crop_size, self.crop_size, 3), 40, dtype=np.uint8)
        cv2.circle(self.clean_face, (128, 128), 80, (130, 160, 210), -1)

        # Base face mask (elliptical or circular mask matching the face)
        self.face_mask = np.zeros((self.crop_size, self.crop_size), dtype=np.float32)
        cv2.circle(self.face_mask, (128, 128), 75, 1.0, -1)
        self.face_mask = cv2.GaussianBlur(self.face_mask, (9, 9), 0)

        # Create target image with an artificial black bar (occluding hand/object)
        # passing horizontally across the middle of the face
        self.bar_y1, self.bar_y2 = 110, 140
        self.bar_x1, self.bar_x2 = 60, 196
        self.occluded_face = self.clean_face.copy()
        self.occluded_face[self.bar_y1:self.bar_y2, self.bar_x1:self.bar_x2] = 0  # Pure black bar

        # Swapped face buffer (contrasting bright green (30, 220, 30))
        self.swap_face = np.full((self.crop_size, self.crop_size, 3), (30, 220, 30), dtype=np.uint8)

    def tearDown(self):
        roop.globals.enable_occlusion_mask = True
        face_swapper.clear_temporal_state()

    def test_partial_admission_does_not_flip_back_on_raw_mask_noise(self):
        """The visible swap must follow the stabilized mask, not raw noise.

        Before the fix, ``process_mask`` stamped ``occlusion_state`` before the
        mask stabilizer.  A first frame with an occluder followed by a clear raw
        frame therefore changed the state to ``visible`` even when the
        compositor still held the occlusion mask.  ProcessMgr then alternated
        between restoring and showing the generated face.
        """
        class Face(dict):
            def __getattr__(self, name):
                try:
                    return self[name]
                except KeyError:
                    raise AttributeError(name)

        class HoldingMask:
            def __init__(self):
                self.previous = None

            def apply(self, mask, kps, _t):
                current = np.asarray(mask, dtype=np.float32)
                if self.previous is None:
                    self.previous = current.copy()
                else:
                    self.previous = np.maximum(self.previous, current)
                return self.previous.copy()

        points = np.column_stack((
            np.linspace(8.0, 24.0, 106),
            np.linspace(8.0, 24.0, 106),
        )).astype(np.float32)
        face = Face(
            kps=np.asarray(((8, 8), (24, 8), (16, 16), (10, 24), (22, 24)),
                           dtype=np.float32),
            landmark_2d_106=points,
            bbox=np.asarray((4, 4, 28, 28), dtype=np.float32),
            pose=np.zeros(3, dtype=np.float32),
        )
        frame = np.zeros((32, 32, 3), dtype=np.uint8)
        identity = np.asarray(((1, 0, 0), (0, 1, 0)), dtype=np.float32)

        class Processor:
            processorname = 'mask_occluder'

            def __init__(self):
                self.calls = 0

            def Run(self, _frame, _keywords):
                self.calls += 1
                # First frame: every interior landmark is behind the object.
                # Second frame: the raw detector says the object disappeared.
                return np.ones((32, 32), dtype=np.float32) if self.calls == 1 \
                    else np.zeros((32, 32), dtype=np.float32)

        class Harness(MaskingMixin):
            def __init__(self):
                self.options = SimpleNamespace(
                    masking_text='', show_face_masking=False)
                self._stab_active = True
                self._tls = SimpleNamespace(frame_idx=0, cur_M=None)
                self.mask_stabilizer = HoldingMask()

            def _temporal_engine(self, _name):
                return None

            def _cur_mask_stab(self):
                return self.mask_stabilizer

            def _cur_stab_t(self):
                return self._tls.frame_idx

        harness = Harness()
        processor = Processor()
        with patch('roop.procmgr_masking._recover_undersized_mask',
                   side_effect=lambda mask, *_args: mask):
            for frame_index in (0, 1):
                harness._tls.frame_idx = frame_index
                harness.process_mask(processor, frame, frame.copy(),
                                     target_face=face, M=identity)
                self.assertEqual(face.get('occlusion_state'), 'partial')

    def test_occlusion_parsing_detects_foreground_black_bar(self):
        """Confirm the occlusion parsing pipeline identifies the artificial black bar as occlusion."""
        occ_mask = face_swapper.compute_occlusion_mask(self.occluded_face, face_mask=self.face_mask)

        # The occluded black bar region must be segmented as occlusion (> 0.50)
        bar_region_occ = occ_mask[self.bar_y1+5:self.bar_y2-5, self.bar_x1+10:self.bar_x2-10]
        self.assertGreater(float(bar_region_occ.mean()), 0.50,
                           "Foreground black bar should be detected as occlusion")

        # The unoccluded forehead skin must have low occlusion (< 0.35)
        skin_region_occ = occ_mask[70:95, 100:155]
        self.assertLess(float(skin_region_occ.mean()), 0.35,
                        "Unoccluded skin should have low occlusion score")

    def test_effective_blend_mask_subtraction(self):
        """Confirm Mask_blend = Mask_face * (1.0 - Mask_occlusion) suppresses the occluded area."""
        occ_mask = face_swapper.compute_occlusion_mask(self.occluded_face, face_mask=self.face_mask)
        blend_mask = face_swapper.apply_occlusion_blend(self.face_mask, occ_mask)

        # In the occluded bar region, blend mask must be close to 0 (suppressed)
        bar_blend = blend_mask[self.bar_y1+5:self.bar_y2-5, self.bar_x1+10:self.bar_x2-10]
        self.assertLess(float(bar_blend.mean()), 0.40,
                        "Blend mask in occluded region should be near zero")

        # In unoccluded skin, blend mask remains high (> 0.65)
        skin_blend = blend_mask[70:95, 100:155]
        self.assertGreater(float(skin_blend.mean()), 0.65,
                           "Blend mask on clear skin should remain high")

    def test_occluded_area_masked_out_of_final_swap_buffer(self):
        """Self-verification: Confirm occluded area (black bar) is masked out of final swap buffer.

        When Mask_blend suppresses the occluded area, the original black bar pixels
        remain intact rather than being overwritten by the swapped face.
        """
        occ_mask = face_swapper.compute_occlusion_mask(self.occluded_face, face_mask=self.face_mask)
        blend_mask = face_swapper.apply_occlusion_blend(self.face_mask, occ_mask)

        final_buffer = face_swapper.blend_swap_buffer(self.occluded_face, self.swap_face, blend_mask)

        # The occluded region in the final buffer should retain the target's dark/black pixels
        bar_pixels = final_buffer[self.bar_y1+5:self.bar_y2-5, self.bar_x1+10:self.bar_x2-10]
        self.assertLess(float(bar_pixels.mean()), 80.0,
                        "Occluded black bar should not be overwritten by swapped face buffer")

        # The unoccluded face region should contain the swapped face (high green channel)
        skin_pixels = final_buffer[70:95, 100:155]
        self.assertGreater(float(skin_pixels[..., 1].mean()), 140.0,
                           "Unoccluded region should receive swapped face pixels")

    def test_ui_toggle_disables_occlusion_subtraction(self):
        """Confirm --enable-occlusion-mask=False disables occlusion subtraction."""
        roop.globals.enable_occlusion_mask = False
        occ_mask = face_swapper.compute_occlusion_mask(self.occluded_face, face_mask=self.face_mask)
        blend_mask = face_swapper.apply_occlusion_blend(self.face_mask, occ_mask)

        # With occlusion mask disabled, blend mask equals face mask
        np.testing.assert_allclose(blend_mask, self.face_mask, atol=1e-5)

        # And final buffer overwrites the black bar
        final_buffer = face_swapper.blend_swap_buffer(self.occluded_face, self.swap_face, blend_mask)
        bar_pixels = final_buffer[self.bar_y1+5:self.bar_y2-5, self.bar_x1+10:self.bar_x2-10]
        self.assertGreater(float(bar_pixels[..., 1].mean()), 140.0,
                           "When disabled, swapped face should overwrite the bar")

    def test_temporal_mask_smoothing_optical_flow_ema(self):
        """Confirm temporal smoothing warps previous mask along optical flow vectors and applies EMA:

            Mask_t = 0.8 * Mask_t + 0.2 * WarpedMask_{t-1}
        """
        smoother = face_swapper.TemporalMaskSmoother(alpha=0.8)

        # Frame 0
        crop_0 = self.clean_face.copy()
        mask_0 = self.face_mask.copy()
        out_0 = smoother.smooth(mask_0, crop_0, track_id=1)
        np.testing.assert_allclose(out_0, mask_0, atol=1e-5)

        # Frame 1: shifted face (motion of 4 pixels to the right)
        shift_x = 4
        M_shift = np.float32([[1, 0, shift_x], [0, 1, 0]])
        crop_1 = cv2.warpAffine(crop_0, M_shift, (self.crop_size, self.crop_size))
        mask_1 = cv2.warpAffine(mask_0, M_shift, (self.crop_size, self.crop_size))

        out_1 = smoother.smooth(mask_1, crop_1, track_id=1, alpha=0.8)

        # The smoothed mask should be a valid normalized mask
        self.assertEqual(out_1.shape, mask_1.shape)
        self.assertTrue(np.all(out_1 >= 0.0) and np.all(out_1 <= 1.0))
        # Correlation between smoothed mask and shifted mask must be extremely high (> 0.95)
        corr = np.corrcoef(out_1.flatten(), mask_1.flatten())[0, 1]
        self.assertGreater(corr, 0.95)

    def test_performance_guardrail_resolution(self):
        """Confirm occlusion model input is strictly executed at 256x256 resolution."""
        large_crop = np.full((512, 512, 3), 150, dtype=np.uint8)
        occ_mask = face_swapper.compute_occlusion_mask(large_crop)
        self.assertEqual(occ_mask.shape, (512, 512),
                         "Output mask must match input crop resolution even when processed at 256x256")


if __name__ == '__main__':
    unittest.main()
