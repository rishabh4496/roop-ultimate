"""create_landmark_mask / _mask_crop_box — the paste-time face hull.

The 106-point landmark model stops at the eyebrows, so the mask extends a
forehead on top. That extension used to be projected straight up in IMAGE
space, which is only correct for an upright head: on a tilted one the forehead
polygon lands off-axis, so the hull swallows background on one side while
clipping real forehead on the other.

Properties asserted:
  * upright faces are BIT-IDENTICAL to the original implementation, so the
    frontal path (the overwhelming majority of frames) provably did not move;
  * the mask is rotationally EQUIVARIANT — rolling the head rolls the mask,
    rather than deforming it;
  * the forehead extension tracks the head's own up-axis under YAW and PITCH,
    not just roll (TestTurnedAndTilted).

That last one needs a head with real depth. Everything above it rotates a flat
2-D face in-plane, which is all roll requires — but a flat face collapses to a
line under yaw, so those fixtures cannot say anything about a turned head.
facegeom.head_106 exists for this.
"""

import os
import sys
import unittest

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from roop.ProcessMgr import ProcessMgr                  # noqa: E402
from tests.facegeom import (head_106, project_kps,      # noqa: E402
                            project_points)

SHAPE = (600, 600, 3)
CENTER = (300.0, 300.0)

create_landmark_mask = ProcessMgr.create_landmark_mask


def original_implementation(landmarks_2d, frame_shape, blend_amount):
    """The pre-change code, kept verbatim as the oracle for upright faces."""
    mask = np.zeros(frame_shape[:2], dtype=np.uint8)
    pts = landmarks_2d.astype(np.int32)
    top_brow_y = int(np.min(pts[33:53, 1]))
    chin_y = int(np.max(pts[:, 1]))
    face_h = max(1, chin_y - top_brow_y)
    forehead_y = max(0, top_brow_y - int(face_h * 0.6))
    top_zone = pts[pts[:, 1] < top_brow_y + int(face_h * 0.15)]
    if len(top_zone) >= 2:
        left_x, right_x = int(np.min(top_zone[:, 0])), int(np.max(top_zone[:, 0]))
    else:
        left_x, right_x = int(np.min(pts[:, 0])), int(np.max(pts[:, 0]))
    forehead = np.array([[left_x, forehead_y],
                         [(left_x + right_x) // 2, forehead_y],
                         [right_x, forehead_y]], dtype=np.int32)
    cv2.fillConvexPoly(mask, cv2.convexHull(np.vstack([pts, forehead])), 255)
    if blend_amount > 0:
        face_w = max(1, right_x - left_x)
        expand = max(1, int(np.sqrt(face_h * face_w) * blend_amount / 400))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (expand * 2 + 1,) * 2)
        mask = cv2.dilate(mask, kernel, iterations=1)
    return mask


def synthetic_face(roll_deg, scale=120.0, center=CENTER, seed=3):
    """A deterministic 106-point face plus its 5 keypoints, rolled by roll_deg."""
    rng = np.random.default_rng(seed)
    base = rng.uniform(-1, 1, size=(106, 2)) * np.array([0.55, 0.85])
    base[33:53, 1] = np.linspace(0.45, 0.60, 20)     # eyebrow band
    base[0, 1] = -0.95                                # chin
    theta = np.radians(roll_deg)
    rot = np.array([[np.cos(theta), -np.sin(theta)],
                    [np.sin(theta), np.cos(theta)]])
    pts = (base * scale) @ rot.T
    landmarks = np.column_stack([center[0] + pts[:, 0],
                                 center[1] - pts[:, 1]]).astype(np.float32)
    kp = np.array([[-30, 45], [30, 45], [0, 5], [-25, -45], [25, -45]], float) @ rot.T
    kps = np.column_stack([center[0] + kp[:, 0],
                           center[1] - kp[:, 1]]).astype(np.float32)
    return landmarks, kps


def rotate_image(img, deg):
    m = cv2.getRotationMatrix2D(CENTER, deg, 1.0)
    return cv2.warpAffine(img, m, (img.shape[1], img.shape[0]),
                          flags=cv2.INTER_NEAREST)


def iou(a, b):
    a, b = a > 127, b > 127
    union = (a | b).sum()
    return float((a & b).sum() / union) if union else 1.0


class TestUprightUnchanged(unittest.TestCase):
    """The frontal path must not have moved by even one pixel."""

    def test_bit_identical_without_kps(self):
        for seed in range(40):
            landmarks, _ = synthetic_face(0.0, seed=seed)
            for blend in (0.0, 10.0, 25.0):
                np.testing.assert_array_equal(
                    create_landmark_mask(None, landmarks, SHAPE, blend, kps=None),
                    original_implementation(landmarks, SHAPE, blend),
                    f"seed={seed} blend={blend}")

    def test_bit_identical_with_image_up_kps(self):
        """Supplying keypoints whose axis is already image-up must reduce to the
        no-kps case exactly."""
        up = np.array([[280, 250], [320, 250], [300, 280],
                       [285, 320], [315, 320]], dtype=np.float32)
        for seed in range(20):
            landmarks, _ = synthetic_face(0.0, seed=seed)
            np.testing.assert_array_equal(
                create_landmark_mask(None, landmarks, SHAPE, 15.0, kps=up),
                original_implementation(landmarks, SHAPE, 15.0), f"seed={seed}")


class TestRotationalEquivariance(unittest.TestCase):
    def test_mask_follows_a_rolled_head(self):
        landmarks0, kps0 = synthetic_face(0.0)
        baseline = create_landmark_mask(None, landmarks0, SHAPE, 15.0, kps=kps0)
        for roll in (5, 10, 20, 30, 45):
            landmarks, kps = synthetic_face(roll)
            got = create_landmark_mask(None, landmarks, SHAPE, 15.0, kps=kps)
            score = iou(got, rotate_image(baseline, roll))
            self.assertGreater(score, 0.95, f"roll={roll} deg: IoU {score:.4f} — "
                                            f"mask no longer tracks the head axis")

    def test_beats_the_image_up_projection_it_replaced(self):
        """Documents the size of the defect: the old projection drifts badly as
        the head tilts, which is what made profile masks miss the forehead."""
        landmarks0, _ = synthetic_face(0.0)
        old_baseline = original_implementation(landmarks0, SHAPE, 15.0)
        for roll in (30, 45):
            landmarks, kps = synthetic_face(roll)
            expected = rotate_image(old_baseline, roll)
            old_score = iou(original_implementation(landmarks, SHAPE, 15.0), expected)
            new_score = iou(create_landmark_mask(None, landmarks, SHAPE, 15.0, kps=kps),
                            rotate_image(create_landmark_mask(
                                None, landmarks0, SHAPE, 15.0,
                                kps=synthetic_face(0.0)[1]), roll))
            self.assertLess(old_score, 0.80, f"roll={roll}: old code unexpectedly good")
            self.assertGreater(new_score, old_score + 0.15, f"roll={roll}")


class TestRobustness(unittest.TestCase):
    def test_face_near_the_frame_edge_does_not_crash(self):
        """The forehead can project off-frame; vertices are left off-frame and
        clipped by fillConvexPoly rather than clamped to y=0, which used to drag
        the hull's top edge inward and eat real forehead."""
        for center in ((60.0, 40.0), (560.0, 30.0), (20.0, 300.0), (300.0, 20.0)):
            for roll in (0.0, 35.0):
                landmarks, kps = synthetic_face(roll, scale=150.0, center=center)
                mask = create_landmark_mask(None, landmarks, SHAPE, 15.0, kps=kps)
                self.assertEqual(mask.shape, SHAPE[:2])
                self.assertEqual(mask.dtype, np.uint8)

    def test_degenerate_kps_fall_back_to_image_up(self):
        landmarks, _ = synthetic_face(0.0)
        expected = original_implementation(landmarks, SHAPE, 15.0)
        for bad in (np.zeros((5, 2), np.float32),
                    np.full((5, 2), 5.0, np.float32),
                    np.zeros((3, 2), np.float32)):
            np.testing.assert_array_equal(
                create_landmark_mask(None, landmarks, SHAPE, 15.0, kps=bad), expected)

    def test_mask_is_non_empty_and_binary_ranged(self):
        for roll in (0, 25, 60, 90):
            landmarks, kps = synthetic_face(roll)
            mask = create_landmark_mask(None, landmarks, SHAPE, 15.0, kps=kps)
            self.assertGreater((mask > 0).sum(), 0)
            self.assertLessEqual(mask.max(), 255)


class TestMaskCropBox(unittest.TestCase):
    """_mask_crop_box feeds the non-frontal masking path; returning a bad
    rectangle there used to abort the whole swap inside cv2.resize."""

    class _Face:
        def __init__(self, bbox):
            self.bbox = bbox

    def test_returns_none_when_face_is_off_frame(self):
        frame = np.zeros((480, 640, 3), np.uint8)
        for bbox in ([-500, -500, -400, -400], [900, 700, 1000, 800],
                     [-200, 100, -150, 200]):
            self.assertIsNone(
                ProcessMgr._mask_crop_box(self._Face(bbox), frame), f"bbox={bbox}")

    def test_box_is_clamped_inside_the_frame(self):
        frame = np.zeros((480, 640, 3), np.uint8)
        for bbox in ([100, 100, 200, 220], [0, 0, 60, 60], [600, 440, 700, 540]):
            box = ProcessMgr._mask_crop_box(self._Face(bbox), frame)
            if box is None:
                continue
            _, _, _, _, cx0, cy0, cx1, cy1 = box
            self.assertGreaterEqual(cx0, 0)
            self.assertGreaterEqual(cy0, 0)
            self.assertLessEqual(cx1, 640)
            self.assertLessEqual(cy1, 480)
            self.assertGreater(cx1 - cx0, 0)
            self.assertGreater(cy1 - cy0, 0)

    def test_box_covers_the_face_with_padding(self):
        frame = np.zeros((1080, 1920, 3), np.uint8)
        box = ProcessMgr._mask_crop_box(self._Face([800, 400, 900, 520]), frame)
        x0, y0, x1, y1 = box[:4]
        self.assertLessEqual(x0, 800)
        self.assertLessEqual(y0, 400)
        self.assertGreaterEqual(x1, 900)
        self.assertGreaterEqual(y1, 520)


class TestTurnedAndTilted(unittest.TestCase):
    """Yaw and pitch, which the roll fixtures above structurally cannot reach.

    The hull itself is just the convex hull of the landmarks, so it follows the
    face wherever the landmarks go. The part that is NOT derived from landmarks —
    and so the part that can be wrong at a pose nobody tested — is the forehead
    extension, which is projected along an axis taken from the 5 keypoints.
    """

    SHAPE = (512, 512, 3)

    @staticmethod
    def _mask(yaw, pitch, roll, blend=15.0):
        lm = project_points(head_106(), yaw, pitch, roll)
        kps = project_kps(yaw, pitch, roll)
        return create_landmark_mask(None, lm, TestTurnedAndTilted.SHAPE,
                                    blend, kps=kps), lm, kps

    def test_every_keypoint_stays_inside_the_mask(self):
        checked = 0
        for yaw in range(-90, 91, 10):
            for pitch in (-40, -20, 0, 20, 40):
                for roll in (-45, 0, 45):
                    mask, _, kps = self._mask(yaw, pitch, roll)
                    for name, p in zip(("Leye", "Reye", "nose", "Lmouth",
                                        "Rmouth"), kps):
                        x, y = int(round(p[0])), int(round(p[1]))
                        if 0 <= y < 512 and 0 <= x < 512:
                            self.assertGreater(
                                mask[y, x], 0,
                                f"{name} outside the mask at yaw={yaw} "
                                f"pitch={pitch} roll={roll}")
                    checked += 1
        self.assertGreater(checked, 250)

    def test_forehead_extends_along_the_head_axis_at_every_pose(self):
        """~0.6 of the brow-to-chin distance, measured along the head's own up
        axis. If the extension were still projected in image space this would
        collapse as soon as the head turned or tilted."""
        for yaw in (-60, -30, 0, 30, 60):
            for pitch in (-40, 0, 40):
                mask, lm, kps = self._mask(yaw, pitch, 0, blend=0.0)
                ys, xs = np.nonzero(mask)
                self.assertGreater(len(xs), 0)
                k = np.asarray(kps, np.float64)
                u = ((k[0] + k[1]) / 2.0) - ((k[3] + k[4]) / 2.0)
                u = u / max(np.linalg.norm(u), 1e-9)
                brow = float(np.max(lm[33:53] @ u))
                face_h = brow - float(np.min(lm @ u))
                top = float(np.max(np.column_stack([xs, ys]).astype(np.float64) @ u))
                ratio = (top - brow) / max(face_h, 1e-9)
                self.assertGreater(ratio, 0.35,
                                   f"forehead only {ratio:.2f} of face height at "
                                   f"yaw={yaw} pitch={pitch}")
                self.assertLess(ratio, 0.90,
                                f"forehead over-extended to {ratio:.2f} at "
                                f"yaw={yaw} pitch={pitch}")

    def test_mask_area_is_sane_across_the_sphere(self):
        for yaw in range(-90, 91, 15):
            for pitch in (-40, 0, 40):
                mask, _, _ = self._mask(yaw, pitch, 0)
                cov = float((mask > 0).mean())
                self.assertGreater(cov, 0.004,
                                   f"mask nearly empty at yaw={yaw} pitch={pitch}")
                self.assertLess(cov, 0.60,
                                f"mask swallowed the frame at yaw={yaw} "
                                f"pitch={pitch}")


if __name__ == "__main__":
    unittest.main()
