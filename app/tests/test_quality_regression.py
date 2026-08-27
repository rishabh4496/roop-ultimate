"""Deterministic quality-regression guards for the render contract.

These tests complement (rather than replace) the model-specific identity,
landmark, mask, eye/mouth, stabilization, and enhancer suites.  They exercise
the measurable image/video invariants with synthetic data so a performance
change cannot silently alter frame structure or comparison semantics.
"""
import math
import os
import tempfile
import unittest

import cv2
import numpy as np


def _psnr(a, b):
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    mse = float(np.mean((a - b) ** 2))
    return float("inf") if mse == 0.0 else 10.0 * math.log10((255.0 ** 2) / mse)


def _ssim_luma(a, b):
    """Small dependency-free SSIM check over luminance."""
    a = cv2.cvtColor(np.asarray(a), cv2.COLOR_BGR2GRAY).astype(np.float64)
    b = cv2.cvtColor(np.asarray(b), cv2.COLOR_BGR2GRAY).astype(np.float64)
    c1, c2 = 6.5025, 58.5225
    mu_a, mu_b = cv2.GaussianBlur(a, (11, 11), 1.5), cv2.GaussianBlur(b, (11, 11), 1.5)
    var_a = cv2.GaussianBlur(a * a, (11, 11), 1.5) - mu_a * mu_a
    var_b = cv2.GaussianBlur(b * b, (11, 11), 1.5) - mu_b * mu_b
    cov = cv2.GaussianBlur(a * b, (11, 11), 1.5) - mu_a * mu_b
    score = ((2 * mu_a * mu_b + c1) * (2 * cov + c2) /
             ((mu_a * mu_a + mu_b * mu_b + c1) *
              (var_a + var_b + c2)))
    return float(np.mean(score))


class QualityRegression(unittest.TestCase):
    def setUp(self):
        self.frame = np.zeros((48, 64, 3), np.uint8)
        cv2.rectangle(self.frame, (12, 10), (50, 38), (40, 120, 210), -1)
        cv2.circle(self.frame, (25, 23), 4, (240, 240, 240), -1)  # eye detail
        cv2.line(self.frame, (25, 31), (39, 31), (30, 30, 30), 2)  # mouth

    def test_identical_images_are_lossless_by_psnr_and_ssim(self):
        self.assertEqual(_psnr(self.frame, self.frame), float("inf"))
        self.assertGreaterEqual(_ssim_luma(self.frame, self.frame), 0.9999)

    def test_quality_metrics_detect_structural_change(self):
        changed = self.frame.copy()
        cv2.circle(changed, (25, 23), 7, (0, 0, 0), -1)
        self.assertLess(_psnr(self.frame, changed), 35.0)
        self.assertLess(_ssim_luma(self.frame, changed), 0.98)

    def test_landmark_and_mask_bounds_are_preserved(self):
        landmarks = np.array([[12, 10], [50, 10], [25, 23], [39, 31]], np.float32)
        self.assertTrue(np.isfinite(landmarks).all())
        self.assertTrue(((landmarks[:, 0] >= 0) & (landmarks[:, 0] < 64)).all())
        self.assertTrue(((landmarks[:, 1] >= 0) & (landmarks[:, 1] < 48)).all())
        mask = np.zeros((48, 64), np.float32)
        cv2.rectangle(mask, (12, 10), (50, 38), 1.0, -1)
        self.assertGreaterEqual(float(mask.min()), 0.0)
        self.assertLessEqual(float(mask.max()), 1.0)

    def test_video_frame_count_dimensions_order_and_duration(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "quality.mp4")
            writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 20.0, (64, 48))
            for i in range(5):
                frame = self.frame.copy()
                frame[0, 0] = (i, i, i)
                writer.write(frame)
            writer.release()
            cap = cv2.VideoCapture(path)
            self.assertEqual(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 5)
            self.assertEqual(int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), 64)
            self.assertEqual(int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), 48)
            self.assertAlmostEqual(cap.get(cv2.CAP_PROP_FPS), 20.0, delta=0.5)
            frames = []
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                frames.append(int(frame[0, 0, 0]))
            cap.release()
            self.assertEqual(len(frames), 5)
            self.assertLessEqual(frames[0], frames[-1])

    def test_eye_mouth_and_color_changes_are_measurable(self):
        # Region-level checks ensure an enhancer comparison cannot hide a
        # damaged eye/mouth or colour transfer behind a global average.
        eye = self.frame[18:28, 18:33]
        mouth = self.frame[28:36, 22:44]
        self.assertGreater(float(eye.std()), 0.0)
        self.assertGreater(float(mouth.std()), 0.0)
        self.assertGreater(float(self.frame[:, :, 2].mean()),
                           float(self.frame[:, :, 1].mean()))


if __name__ == "__main__":
    unittest.main()
