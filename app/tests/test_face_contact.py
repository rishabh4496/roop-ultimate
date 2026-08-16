"""face_contact — the detector and the recogniser when two faces are touching.

Two rules, both narrow, both easy to get wrong in the direction that costs a
real face rather than a phantom. Everything here is geometry, so the cases the
sample clip does not happen to contain (three people in a row, a small face
framed between two near ones) are built by hand rather than hoped for.

  * A JUNCTION IS NOT A FACE. Where two faces meet the detector fires a third
    box made of half of each, at up to 0.99 confidence, on a third of the
    frames of the measured clip. It must go, and nothing else may go with it.

  * A SHARED CROP IS NOT AN IDENTITY. ArcFace fits five keypoints to a frontal
    template, which for a near-profile face resolves to a crop far wider than
    the detection box. When another face is standing in that extra area, both
    people's crops converge onto the same picture and the embeddings with them.
    The measurement here is purely how much of the crop belongs to somebody
    else; what the pipeline does about it is asserted at the bottom, because a
    number nobody reads fixes nothing.

The crop quad is checked against insightface's own `estimate_norm` rather than
against remembered numbers: face_contact reimplements the similarity fit so it
can stay importable without skimage, and a reimplementation that has drifted
from the real aligner would measure the wrong rectangle while looking healthy.
"""

import ast
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from roop import face_contact                                   # noqa: E402
from roop.face_contact import (crop_contamination, merged_indices,   # noqa: E402
                               suppress_merged, unreliable, _crop_quad)

APP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def box(cx, cy, w, h):
    return (cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0)


class _Face(dict):
    """Only the two attributes face_contact reads, on a dict like the real one."""

    def __init__(self, bbox, kps=None):
        super().__init__()
        self.bbox = np.asarray(bbox, dtype=np.float32)
        self.kps = None if kps is None else np.asarray(kps, dtype=np.float32)

    def __getattr__(self, k):
        try:
            return self[k]
        except KeyError:
            raise AttributeError(k)

    def __setattr__(self, k, v):
        self[k] = v


def profile_kps(cx, cy, w, h, facing=1):
    """Five keypoints of a head seen in profile.

    The eyes are a few pixels apart and the nose sits well outside them on the
    side the head faces — the configuration that makes the ArcFace fit resolve
    to a crop much larger than the box, which is the whole mechanism under
    test. Proportions taken from real detections on the sample clip
    (interocular ~6% of box width).
    """
    return [[cx + facing * 0.22 * w, cy - 0.10 * h],
            [cx + facing * 0.28 * w, cy - 0.13 * h],
            [cx + facing * 0.49 * w, cy + 0.02 * h],
            [cx + facing * 0.24 * w, cy + 0.25 * h],
            [cx + facing * 0.27 * w, cy + 0.22 * h]]


class MergedDetections(unittest.TestCase):

    def test_junction_between_two_touching_profiles_is_dropped(self):
        """The reported case: two heads meeting, a third box across the join."""
        left = box(100, 100, 120, 190)
        right = box(215, 100, 115, 175)
        junction = box(160, 100, 135, 190)
        self.assertEqual(merged_indices([left, right, junction]), [2])

    def test_only_the_junction_is_dropped(self):
        """Neither parent may be taken with it.

        Worth its own test: each parent is also heavily overlapped once the
        junction box exists, and a rule written on overlap alone eats one of
        them — which is a face vanishing from half the frames of a clip.
        """
        boxes = [box(100, 100, 120, 190), box(215, 100, 115, 175),
                 box(160, 100, 135, 190)]
        self.assertNotIn(0, merged_indices(boxes))
        self.assertNotIn(1, merged_indices(boxes))

    def test_three_real_faces_in_a_row_survive(self):
        """Neighbours on the SAME side, so the middle face is not between two
        of them in the sense the rule means, and most of it is uncovered."""
        self.assertEqual(merged_indices([box(100, 100, 110, 180),
                                         box(200, 100, 110, 180),
                                         box(300, 100, 110, 180)]), [])

    def test_three_overlapping_faces_in_a_row_survive(self):
        self.assertEqual(merged_indices([box(100, 100, 140, 200),
                                         box(200, 100, 140, 200),
                                         box(300, 100, 140, 200)]), [])

    def test_small_distant_face_between_two_near_ones_survives(self):
        """The case the size floor exists for. A head further away, framed
        between two nearer ones, IS between them and IS covered by their
        union — everything except its size says junction."""
        self.assertEqual(merged_indices([box(100, 100, 220, 300),
                                         box(420, 100, 220, 300),
                                         box(260, 100, 60, 90)]), [])

    def test_junction_bridging_a_small_real_gap_is_dropped(self):
        """A profile-view kiss where the two real face boxes come close but
        do not quite overlap (measured on baseline_double/d2.mp4, frame 1214:
        a 17.8px gap on ~54px-radius boxes). The phantom still bridges the
        gap and is still a junction; only the coverage math needed help
        seeing it, not the between/each/size checks."""
        woman = (478.8, 162.0, 550.8, 269.1)
        phantom = (518.9, 152.0, 582.7, 264.7)
        man = (568.6, 139.2, 632.5, 247.1)
        self.assertEqual(merged_indices([woman, phantom, man]), [1])

    def test_a_duplicate_box_is_not_treated_as_a_junction(self):
        """Two boxes on one face plus a second face. The duplicate pair is
        concentric, so it cannot play the role of the two parents."""
        self.assertEqual(merged_indices([box(100, 100, 120, 190),
                                         box(106, 103, 126, 196),
                                         box(240, 100, 115, 175)]), [])

    def test_fewer_than_three_boxes_is_never_touched(self):
        self.assertEqual(merged_indices([box(100, 100, 120, 190),
                                         box(160, 100, 120, 190)]), [])

    def test_suppress_merged_keeps_the_objects_and_their_order(self):
        faces = [_Face(box(100, 100, 120, 190)),
                 _Face(box(215, 100, 115, 175)),
                 _Face(box(160, 100, 135, 190))]
        kept, dropped = suppress_merged(faces)
        self.assertEqual(dropped, 1)
        self.assertIs(kept[0], faces[0])
        self.assertIs(kept[1], faces[1])
        self.assertEqual(len(kept), 2)

    def test_the_switch_turns_it_off(self):
        boxes = [box(100, 100, 120, 190), box(215, 100, 115, 175),
                 box(160, 100, 135, 190)]
        old = face_contact.MERGE_SUPPRESS
        try:
            face_contact.MERGE_SUPPRESS = False
            self.assertEqual(merged_indices(boxes), [])
        finally:
            face_contact.MERGE_SUPPRESS = old


class CropQuad(unittest.TestCase):

    def test_matches_insightface_estimate_norm(self):
        """Same rectangle the recogniser actually crops, to under a pixel."""
        import cv2
        from insightface.utils import face_align

        rng = np.random.RandomState(0)
        worst = 0.0
        for _ in range(200):
            kps = rng.uniform(0, 500, (5, 2)).astype(np.float32)
            m = face_align.estimate_norm(kps, 112)
            inv = cv2.invertAffineTransform(m)
            corners = np.array([[0, 0], [112, 0], [112, 112], [0, 112]],
                               np.float32).reshape(-1, 1, 2)
            ref = cv2.transform(corners, inv).reshape(-1, 2)
            worst = max(worst, float(np.abs(ref - _crop_quad(kps)).max()))
        self.assertLess(worst, 0.1)

    def test_a_profile_crop_is_far_wider_than_the_box(self):
        """The mechanism itself. If this ever stops being true the whole
        contamination story is about something else."""
        w, h = 120.0, 190.0
        quad = _crop_quad(profile_kps(0, 0, w, h))
        span = float(quad[:, 0].max() - quad[:, 0].min())
        self.assertGreater(span, 1.4 * w)


class CropContamination(unittest.TestCase):

    def test_separated_faces_are_clean(self):
        w, h = 120.0, 190.0
        a = _Face(box(100, 100, w, h), profile_kps(100, 100, w, h, +1))
        b = _Face(box(400, 100, w, h), profile_kps(400, 100, w, h, -1))
        self.assertLess(max(crop_contamination([a, b])), 0.1)

    def test_touching_profiles_are_contaminated(self):
        """Facing each other and adjacent — each crop reaches over the join.

        The spacing is the measured one, not a spacing chosen to pass: on the
        sample clip the two boxes sit 0.7 face-widths apart centre to centre
        and overlap by about 33px of a 115px box, and this reproduces the
        0.4-0.6 contamination the real frames read there.
        """
        w, h = 120.0, 190.0
        a = _Face(box(100, 100, w, h), profile_kps(100, 100, w, h, +1))
        b = _Face(box(184, 100, w, h), profile_kps(184, 100, w, h, -1))
        c = crop_contamination([a, b])
        self.assertGreater(min(c), face_contact.CONTAM_MAX)
        self.assertLess(max(c), 0.6)

    def test_a_lone_face_is_never_contaminated(self):
        w, h = 120.0, 190.0
        a = _Face(box(100, 100, w, h), profile_kps(100, 100, w, h, +1))
        self.assertEqual(crop_contamination([a]), [0.0])

    def test_measurement_survives_the_switch_being_off(self):
        """ROOP_EMB_CONTAM=0 stops the pipeline ACTING on contamination; it
        must not stop it being measurable, or a baseline run and the bench that
        grades it both go blind at the same moment."""
        w, h = 120.0, 190.0
        a = _Face(box(100, 100, w, h), profile_kps(100, 100, w, h, +1))
        b = _Face(box(184, 100, w, h), profile_kps(184, 100, w, h, -1))
        old = face_contact.CONTAM_MAX
        try:
            face_contact.CONTAM_MAX = 0.0
            self.assertGreater(max(crop_contamination([a, b])), 0.1)
            self.assertFalse(unreliable(_Face(box(0, 0, 10, 10))))
        finally:
            face_contact.CONTAM_MAX = old

    def test_annotate_tags_every_survivor(self):
        w, h = 120.0, 190.0
        faces = [_Face(box(100, 100, w, h), profile_kps(100, 100, w, h, +1)),
                 _Face(box(184, 100, w, h), profile_kps(184, 100, w, h, -1)),
                 _Face(box(142, 100, 135, h), profile_kps(142, 100, 135, h))]
        kept, dropped = face_contact.annotate(faces)
        self.assertEqual(dropped, 1)
        for f in kept:
            self.assertIn('_emb_contam', f)
        self.assertTrue(any(unreliable(f) for f in kept))


class Wiring(unittest.TestCase):
    """The verdict has to be read where identity is decided.

    Every one of these is a place that, left alone, decides who somebody is
    from an embedding of two people blended together. A measurement nobody
    consults is worse than none, because the run then reports contamination in
    its summary while still acting on it.
    """

    @staticmethod
    def _source(rel):
        with open(os.path.join(APP, rel), encoding='utf-8') as fh:
            return fh.read()

    @staticmethod
    def _function_body(src, marker):
        tree = ast.parse(src)
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) \
                    and node.name == marker:
                return ast.get_source_segment(src, node) or ''
        return ''

    def test_detection_annotates_every_frame(self):
        # Lives in _enrich_detected_faces, shared by _detect_faces AND
        # get_all_faces_hires (the higher-resolution retry), not duplicated.
        body = self._function_body(self._source('roop/face_util.py'),
                                   '_enrich_detected_faces')
        self.assertIn('face_contact.annotate', body)

    def test_the_tracking_scan_disbelieves_a_shared_crop(self):
        body = self._function_body(self._source('roop/procmgr_tracking.py'),
                                   '_consume')
        self.assertIn('face_contact.unreliable', body)
        # ...and does not let it into the track's mean, which is the number the
        # source assignment is decided from.
        self.assertIn('emb_dirty', body)

    def test_the_swap_loop_disbelieves_a_shared_crop(self):
        body = self._function_body(self._source('roop/ProcessMgr.py'),
                                   'swap_faces')
        self.assertGreaterEqual(body.count('face_contact.unreliable'), 2,
                                'both the identity-lock branch and the '
                                'per-frame `selected` branch decide identity '
                                'from an embedding and both must check')

    def test_the_outcome_guard_picks_the_face_it_is_checking(self):
        """swap_moved_the_face re-detects inside a padded window, which on two
        touching people holds the neighbour and the junction as well. Taking
        the biggest measured how far the NEIGHBOUR's keypoints were from this
        face's, and discarded a good swap for it."""
        body = self._function_body(self._source('roop/face_util.py'),
                                   'swap_moved_the_face')
        self.assertIn('suppress_merged', body)
        self.assertIn('_bbox_iou_1d', body)
        self.assertNotIn('key=lambda f: (f.bbox[2] - f.bbox[0])', body)


if __name__ == '__main__':
    unittest.main()
