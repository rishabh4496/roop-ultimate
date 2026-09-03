"""Occlusion-aware landmark symmetry inpainting, and every case it refuses.

The refusals matter more than the repairs. Inpainting replaces a MEASUREMENT
with an ESTIMATE, so a repair applied where it was not warranted is strictly
worse than no repair at all: it is a wrong landmark that looks like a right one,
and the alignment is a similarity fit over exactly these points.
"""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from roop.tracker import (STATE_COASTED, STATE_PARTIAL, STATE_VISIBLE,
                          derive_mirror_map, landmark_visibility,
                          occlusion_state_for, symmetry_inpaint_landmarks,
                          symmetry_axis)


def _synthetic_face(roll_deg=0.0, cx=128.0, cy=128.0, scale=1.0):
    """A bilaterally symmetric 20-point face, optionally rolled in plane.

    Points are laid out in mirror pairs about x=0 plus three midline points, so
    the ground-truth pairing is known exactly and the derived map can be checked
    against it rather than against itself.
    """
    half = np.array([
        [-30.0, -40.0], [-18.0, -42.0],          # brow
        [-20.0, -20.0], [-12.0, -18.0],          # eye
        [-28.0,   6.0],                          # cheek
        [-10.0,  34.0], [-22.0,  24.0],          # mouth corner, jaw
        [-34.0, -10.0],                          # temple
    ], dtype=np.float64)
    mirrored = half.copy()
    mirrored[:, 0] *= -1.0
    midline = np.array([[0.0, -46.0], [0.0, 4.0], [0.0, 44.0]], dtype=np.float64)

    points = np.concatenate((half, mirrored, midline), axis=0) * scale
    theta = np.deg2rad(roll_deg)
    rot = np.array([[np.cos(theta), -np.sin(theta)],
                    [np.sin(theta), np.cos(theta)]], dtype=np.float64)
    return points @ rot.T + np.array([cx, cy])


def _truth_map(count_half=8, count_mid=3):
    """Ground-truth pairing for `_synthetic_face`."""
    n = count_half * 2 + count_mid
    m = np.arange(n, dtype=np.int32)
    for i in range(count_half):
        m[i] = i + count_half
        m[i + count_half] = i
    return m


def _kps_for(points):
    """Five detector keypoints consistent with `_synthetic_face`'s layout."""
    # eye pair (indices 3 and 11), a midline nose (index 17), mouth pair (5, 13)
    return np.stack((points[3], points[11], points[17],
                     points[5], points[13])).astype(np.float32)


class MirrorMapTest(unittest.TestCase):

    def test_the_pairing_is_derived_not_assumed(self):
        points = _synthetic_face()
        derived = derive_mirror_map(points, _kps_for(points))
        self.assertIsNotNone(derived)
        np.testing.assert_array_equal(derived, _truth_map())

    def test_the_pairing_survives_in_plane_roll(self):
        """The axis is derived from the landmarks, so a rolled face is the same face."""
        for roll in (-40.0, -15.0, 25.0, 60.0):
            points = _synthetic_face(roll_deg=roll)
            derived = derive_mirror_map(points, _kps_for(points))
            self.assertIsNotNone(derived, f'roll {roll}')
            np.testing.assert_array_equal(derived, _truth_map())

    def test_an_asymmetric_point_set_is_refused(self):
        """Refusal is the feature: a wrong permutation mirrors chin onto brow."""
        rng = np.random.default_rng(0)
        noise = rng.normal(0.0, 60.0, size=(19, 2))
        self.assertIsNone(derive_mirror_map(noise))

    def test_the_axis_prefers_the_five_keypoints(self):
        points = _synthetic_face()
        frame = symmetry_axis(points, _kps_for(points))
        self.assertIsNotNone(frame)
        _, axis = frame
        # Eyes -> mouth on an upright face points straight down the image.
        self.assertAlmostEqual(float(axis[0]), 0.0, places=5)
        self.assertGreater(float(axis[1]), 0.9)


class VisibilityTest(unittest.TestCase):

    def test_no_mask_means_everything_is_visible(self):
        """Never invent an occlusion: that would replace good data with a guess."""
        points = _synthetic_face()
        visible = landmark_visibility(points, None)
        self.assertTrue(visible.all())

    def test_points_under_the_occluder_read_as_hidden(self):
        points = _synthetic_face()
        mask = np.zeros((256, 256), dtype=np.float32)
        mask[:, :128] = 1.0                        # a hand over the left half
        visible = landmark_visibility(points, mask)
        left = points[:, 0] < 128.0
        np.testing.assert_array_equal(visible, ~left)

    def test_points_outside_the_mask_are_not_called_hidden(self):
        points = _synthetic_face(cx=-500.0)
        mask = np.ones((256, 256), dtype=np.float32)
        self.assertTrue(landmark_visibility(points, mask).all())


class InpaintTest(unittest.TestCase):

    def setUp(self):
        self.points = _synthetic_face()
        self.kps = _kps_for(self.points)
        self.truth = _truth_map()

    def test_a_hidden_half_is_recovered_from_the_visible_one(self):
        visible = np.ones(len(self.points), dtype=bool)
        visible[:8] = False                        # the whole left half hidden
        repaired, filled = symmetry_inpaint_landmarks(
            self.points, visible, kps=self.kps, mirror_map=self.truth)

        self.assertTrue(filled[:8].all())
        self.assertFalse(filled[8:].any())
        # The synthetic face is exactly symmetric, so the repair must land on
        # the true positions, not merely near them.
        np.testing.assert_allclose(repaired[:8], self.points[:8], atol=1e-3)

    def test_an_unoccluded_face_is_returned_untouched(self):
        """The common frame has to be bit-identical, or this is a global change."""
        visible = np.ones(len(self.points), dtype=bool)
        repaired, filled = symmetry_inpaint_landmarks(
            self.points, visible, kps=self.kps)
        self.assertFalse(filled.any())
        np.testing.assert_array_equal(repaired, self.points.astype(np.float32))

    def test_a_midline_point_is_never_repaired(self):
        """It is its own mirror; reflecting it recovers nothing."""
        visible = np.ones(len(self.points), dtype=bool)
        visible[16:] = False                       # the three midline points
        _, filled = symmetry_inpaint_landmarks(
            self.points, visible, kps=self.kps, mirror_map=self.truth)
        self.assertFalse(filled.any())

    def test_both_halves_hidden_repairs_nothing(self):
        visible = np.zeros(len(self.points), dtype=bool)
        visible[16:] = True
        _, filled = symmetry_inpaint_landmarks(
            self.points, visible, kps=self.kps, mirror_map=self.truth)
        self.assertFalse(filled.any())

    def test_strong_yaw_is_refused(self):
        """Past ~55 degrees the far half is foreshortened, not mirrored."""
        visible = np.ones(len(self.points), dtype=bool)
        visible[:8] = False
        _, filled = symmetry_inpaint_landmarks(
            self.points, visible, kps=self.kps, mirror_map=self.truth,
            pose=(0.0, 75.0, 0.0))
        self.assertFalse(filled.any())

        _, filled_ok = symmetry_inpaint_landmarks(
            self.points, visible, kps=self.kps, mirror_map=self.truth,
            pose=(0.0, 20.0, 0.0))
        self.assertTrue(filled_ok[:8].all())

    def test_a_rolled_face_is_repaired_in_its_own_frame(self):
        """The repair must follow the head, not the image axes."""
        points = _synthetic_face(roll_deg=35.0)
        visible = np.ones(len(points), dtype=bool)
        visible[:8] = False
        repaired, filled = symmetry_inpaint_landmarks(
            points, visible, kps=_kps_for(points), mirror_map=self.truth)
        self.assertTrue(filled[:8].all())
        np.testing.assert_allclose(repaired[:8], points[:8], atol=1e-2)

    def test_a_refused_mirror_map_repairs_nothing(self):
        rng = np.random.default_rng(1)
        noise = rng.normal(0.0, 60.0, size=(19, 2))
        visible = np.ones(19, dtype=bool)
        visible[:4] = False
        _, filled = symmetry_inpaint_landmarks(noise, visible)
        self.assertFalse(filled.any())


class OcclusionStateTest(unittest.TestCase):

    def test_a_clear_face_is_visible(self):
        self.assertEqual(occlusion_state_for(np.ones(20, dtype=bool)),
                         STATE_VISIBLE)

    def test_one_grazing_landmark_is_not_an_occlusion(self):
        """A flag that is on for every frame carries no information."""
        visible = np.ones(20, dtype=bool)
        visible[0] = False                          # 5%, below the 8% floor
        self.assertEqual(occlusion_state_for(visible), STATE_VISIBLE)

    def test_a_covered_face_is_partial(self):
        visible = np.ones(20, dtype=bool)
        visible[:8] = False
        self.assertEqual(occlusion_state_for(visible), STATE_PARTIAL)

    def test_coasted_wins_over_everything(self):
        self.assertEqual(occlusion_state_for(np.ones(20, dtype=bool), coasted=True),
                         STATE_COASTED)


if __name__ == '__main__':
    unittest.main()
