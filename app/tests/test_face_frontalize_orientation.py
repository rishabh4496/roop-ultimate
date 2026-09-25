"""face_frontalize must never hand the swapper a mirrored face.

Until 2026-09-25 the frontal reference was re-projected at rvec = 0. _REF3D_68 is
y-up with the nose toward +z and an OpenCV camera is y-down looking along +z,
so that reference was the head upside down and facing away, and the affine fit
to it was a vertical mirror: every frontalized face was swapped upside down (id
to the source 0.52 -> 0.02 at 45-75 deg yaw on tests/frontalize_yaw_bench.py).
These tests project the reference head at known poses and check the geometry.
"""
import os
import sys
import unittest

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from roop.face_3d_recon import _REF3D_68, _DIST, _build_camera  # noqa: E402
from roop.face_frontalize import (  # noqa: E402
    frontalize_crop, get_frontal_landmarks_from_pose)

SIZE = 256
FACING = np.array([np.pi, 0.0, 0.0])     # the reference head facing the camera


def _project(yaw_deg):
    """68 crop-space points of the reference head turned by `yaw_deg`."""
    r_face, _ = cv2.Rodrigues(FACING.reshape(3, 1))
    r_yaw, _ = cv2.Rodrigues(np.array([[0.0], [np.radians(yaw_deg)], [0.0]]))
    rvec, _ = cv2.Rodrigues(r_yaw @ r_face)
    pts3d = _REF3D_68 * 60.0
    tvec = np.array([[0.0], [0.0], [SIZE * 1.2 * 1.0]])
    pts, _ = cv2.projectPoints(pts3d, rvec, tvec, _build_camera(SIZE), _DIST)
    return pts.reshape(-1, 2).astype(np.float32)


class FrontalReferenceOrientation(unittest.TestCase):

    def test_the_projected_head_is_upright(self):
        # Guards the fixture itself: brows (17-26) above the chin (8).
        lm = _project(0)
        self.assertLess(lm[17:27, 1].mean(), lm[8, 1])

    def test_a_frontal_face_maps_to_itself(self):
        lm = _project(0)
        front = get_frontal_landmarks_from_pose(lm, SIZE)
        self.assertIsNotNone(front)
        # Same orientation: brows stay above the chin, left stays left.
        self.assertLess(front[17:27, 1].mean(), front[8, 1])
        self.assertLess(front[36, 0], front[45, 0])
        err = np.linalg.norm(front - lm, axis=1).mean() / np.ptp(lm[:, 0])
        self.assertLess(err, 0.05)

    def test_turned_faces_are_never_mirrored(self):
        img = np.zeros((SIZE, SIZE, 3), np.uint8)
        for yaw in (-60, -45, -30, 30, 45, 60):
            with self.subTest(yaw=yaw):
                lm = _project(yaw)
                _out, M = frontalize_crop(img, lm)
                self.assertIsNotNone(M)
                self.assertGreater(np.linalg.det(M[:, :2]), 0.0)
                self.assertGreater(M[1, 1], 0.0)   # the old bug: M[1,1] ~ -1.19

    def test_a_mirroring_fit_is_refused(self):
        img = np.full((SIZE, SIZE, 3), 7, np.uint8)
        lm = _project(0)
        flipped = lm.copy()
        flipped[:, 1] = SIZE - flipped[:, 1]
        out, M = frontalize_crop(img, lm, frontal_lm68=flipped)
        self.assertIsNone(M)
        self.assertIs(out, img)


if __name__ == "__main__":
    unittest.main()
