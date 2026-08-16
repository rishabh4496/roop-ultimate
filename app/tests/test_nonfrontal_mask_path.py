"""The unwarped-crop masking path, and the premise it was built on.

`process_mask` could route a face down one of two very different derivations.
The normal one runs the mask model on the aligned crop. The other crops an
unwarped box out of the original frame, masks that, and warps the result back —
and it was selected by pose, on the stated grounds that "the standard affine
aligned crop is too distorted for a frontal-trained mask model to label
correctly" once a head turns or tilts.

That premise is false. `estimate_norm` forces a SimilarityTransform, and both
`yaw_align` variants return similarities too. A similarity is a rotation, a
uniform scale and a translation — it has no shear or anisotropy to give. The
first test here measures that to machine precision over the whole pose sphere,
because the entire path rests on it.

With no benefit to weigh, its costs are what is left, and they are real: it
shows the model a face at roughly half the linear size, it undoes the in-plane
rotation that makes a face upright, and switching between two differently
derived masks part-way through a clip is itself a flicker source (the router's
hysteresis exists solely to damp the chatter that switching causes). All three
land on turned and tilted faces specifically — which is where "the original
shows through the middle of the swap" gets reported.

So it is off by default now, `ROOP_NONFRONTAL_MASK=auto` restores the routing,
and the box it uses has been re-sized so the path is defensible if switched on.
"""

import math
import os
import re
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import roop.globals                                              # noqa: E402
from roop.face_util import estimate_norm, _project_reference     # noqa: E402
from roop.procmgr_masking import MaskingMixin                    # noqa: E402

APP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MASKING = os.path.join(APP, 'roop', 'procmgr_masking.py')

CROP = 256
FRAME = np.zeros((1080, 1920, 3), dtype=np.uint8)


def kps_at(yaw, pitch, roll=0.0, face_px=300.0, cx=960.0, cy=540.0):
    ref = _project_reference(0.0, 0.0)
    scale = (face_px * 0.36) / float(np.linalg.norm(ref[1] - ref[0]))
    p = _project_reference(yaw, pitch)
    c, s = math.cos(math.radians(roll)), math.sin(math.radians(roll))
    p = p @ np.array([[c, -s], [s, c]]).T
    p = p - p.mean(axis=0)
    return (p * scale + np.array([cx, cy])).astype(np.float64)


class _Face:
    """Only the geometry _mask_crop_box reads."""

    def __init__(self, kps):
        x0, y0 = kps.min(axis=0)
        x1, y1 = kps.max(axis=0)
        w, h = (x1 - x0) * 1.9, (y1 - y0) * 2.4
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        self.bbox = np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])


POSES = [(0, 0), (20, 0), (40, 0), (60, 0), (80, 0), (0, 30), (60, 30), (75, -30)]


class TestTheAlignedCropDoesNotDistort(unittest.TestCase):
    """The load-bearing claim. If this ever fails, the routing decision above it
    has to be revisited — so it is measured, not asserted in a comment."""





class TestTheBoxMatchesTheCrop(unittest.TestCase):
    """When the path IS used, what it shows the model should differ from the
    aligned crop only by the warp — not by framing or by scale."""



    def _ratio(self, yaw, pitch, **kw):
        """Face scale in the aligned crop over face scale in the mask box.

        Both terms are (a fixed distance on the face) / (the crop it sits in),
        so the face's own foreshortening cancels and what is left is purely the
        ratio of the two crop sizes.
        """
        k = kps_at(yaw, pitch)
        M = estimate_norm(k, CROP)
        kc = np.hstack([k, np.ones((5, 1))]) @ M.T
        aligned = float(np.linalg.norm(kc[1] - kc[0])) / CROP
        box = MaskingMixin._mask_crop_box(_Face(k), FRAME, **kw)
        self.assertIsNotNone(box)
        span = max(box[2] - box[0], box[3] - box[1])
        boxed = float(np.linalg.norm(k[1] - k[0])) / span
        return aligned / boxed

    def test_the_new_box_shows_a_bigger_face_than_the_old_one(self):
        for yaw, pitch in POSES:
            old = self._ratio(yaw, pitch)
            new = self._ratio(yaw, pitch, M=estimate_norm(kps_at(yaw, pitch), CROP),
                              crop_shape=(CROP, CROP))
            self.assertLess(new, old,
                            f'yaw={yaw} pitch={pitch}: {new:.2f}x is no better '
                            f'than the old {old:.2f}x')

    def test_the_worst_pose_improves_most(self):
        """yaw 60 with 30 degrees of pitch measured 2.04x — a turned AND tilted
        head, which is exactly the reported case."""
        M = estimate_norm(kps_at(60, 30), CROP)
        self.assertLess(self._ratio(60, 30, M=M, crop_shape=(CROP, CROP)), 1.75)
        self.assertGreater(self._ratio(60, 30), 1.9)

    def test_the_box_covers_the_whole_aligned_crop(self):
        """Anything the box misses defaults to 1.0 — "restore original" — which
        would cut the swap off along a straight box edge."""
        import cv2
        for yaw, pitch in POSES:
            k = kps_at(yaw, pitch)
            M = estimate_norm(k, CROP)
            box = MaskingMixin._mask_crop_box(_Face(k), FRAME, M=M,
                                              crop_shape=(CROP, CROP))
            probe = np.zeros(FRAME.shape[:2], dtype=np.float32)
            probe[box[5]:box[7], box[4]:box[6]] = 1.0
            cov = cv2.warpAffine(probe, M, (CROP, CROP), flags=cv2.INTER_NEAREST,
                                 borderMode=cv2.BORDER_CONSTANT,
                                 borderValue=0.0).mean()
            self.assertGreater(cov, 0.999,
                               f'box covers {cov*100:.1f}% of the crop at '
                               f'yaw={yaw} pitch={pitch}')

    def test_it_still_works_without_a_matrix(self):
        """The fallback keeps the old bbox multiple, so a caller that has no M
        gets the previous behaviour rather than nothing."""
        box = MaskingMixin._mask_crop_box(_Face(kps_at(0, 0)), FRAME)
        self.assertIsNotNone(box)
        self.assertEqual(len(box), 8)

    def test_a_face_off_the_frame_still_returns_none(self):
        k = kps_at(0, 0, cx=-4000.0, cy=-4000.0)
        M = estimate_norm(k, CROP)
        self.assertIsNone(MaskingMixin._mask_crop_box(_Face(k), FRAME, M=M,
                                                      crop_shape=(CROP, CROP)))

    def test_a_degenerate_matrix_falls_back_rather_than_raising(self):
        k = kps_at(0, 0)
        for M in (np.zeros((2, 3), np.float32), 'nonsense'):
            box = MaskingMixin._mask_crop_box(_Face(k), FRAME, M=M,
                                              crop_shape=(CROP, CROP))
            self.assertIsNotNone(box)


class TestTheDefaultRouting(unittest.TestCase):
    """Read out of the source: standing process_mask up needs a live model."""

    def setUp(self):
        with open(MASKING, encoding='utf-8') as fh:
            src = fh.read()
        self.code = re.sub(r'#[^\n]*', '', src)

    def test_the_path_is_off_unless_asked_for(self):
        m = re.search(r"os\.environ\.get\(\s*'ROOP_NONFRONTAL_MASK'\s*,\s*'([^']*)'",
                      self.code)
        self.assertIsNotNone(m, 'the switch is gone')
        self.assertEqual(m.group(1), '0',
                         'the unwarped mask path is on by default again')

    def test_auto_still_restores_the_pose_routing(self):
        self.assertIn("_nf_mode == 'auto'", self.code,
                      'no way left to re-enable the routing for comparison')
        self.assertIn('nonfrontal_score', self.code)

    def test_the_router_is_not_consulted_when_the_path_is_off(self):
        """It costs a pose solve per face per mask processor, and its answer
        cannot be used — so it must sit inside the `auto` branch."""
        block = self.code.split("_nf_mode = ")[1].split('crop_box = None')[0]
        auto = block.split("_nf_mode == 'auto'")[1]
        self.assertIn('router.verdict', auto,
                      'the router verdict escaped the auto branch')

    def test_the_publisher_asks_before_taking_the_shared_lock(self):
        """The other half, and the more expensive one.

        ProcessMgr.process_face publishes every face into the router as soon as
        the keypoints are known, so a worker asking for a verdict later finds
        the event already logged. observe() scores the face and then takes a
        SHARED lock to record it — per face, per frame, across every worker. With
        the routing off that is contention on the hot path producing something
        nothing will ever read, so the publish has to be gated too. Gating only
        the reader would leave the cost exactly where it was.
        """
        with open(os.path.join(APP, 'roop', 'ProcessMgr.py'), encoding='utf-8') as fh:
            mgr = re.sub(r'#[^\n]*', '', fh.read())
        idx = mgr.find('_nonfrontal_router.observe')
        self.assertNotEqual(idx, -1, 'the router publish is gone entirely')
        before = mgr[max(0, idx - 400):idx]
        self.assertIn('nonfrontal_routing_enabled()', before,
                      'observe() is called unconditionally again')

    def test_the_two_gates_read_the_same_switch(self):
        """Reader and publisher disagreeing would be worse than either bug: the
        router would log events nobody reads, or answer from a log with holes
        in it."""
        self.assertIn('def nonfrontal_routing_enabled', self.code)
        # To the next top-level def, not a fixed slice: these bodies carry long
        # docstrings and a fixed window cuts the code off before it starts.
        fn = re.split(r'\ndef ', self.code.split('def nonfrontal_routing_enabled')[1])[0]
        self.assertIn('nonfrontal_mask_mode()', fn,
                      'the publisher gate reads the env var separately')
        self.assertIn("== 'auto'", fn)


class TestFrontalizationGate(unittest.TestCase):
    """Frontalization thresholds on the same angles the mask router used to get.

    It needs landmark_3d_68, which means the old EPnP path was always taken for
    it — and that reported roughly 180 - true_yaw, so abs() cleared any sane
    threshold on every face. A dead-frontal head was warped toward frontal and
    warped back: a resample for nothing. With true angles the threshold does
    what its name and its UI control say.
    """

    def setUp(self):
        with open(os.path.join(APP, 'roop', 'ProcessMgr.py'), encoding='utf-8') as fh:
            self.code = re.sub(r'#[^\n]*', '', fh.read())

    def test_the_gate_reads_the_corrected_angles(self):
        m = re.search(r"frontalization_threshold[^\n]*\n\s*if\s+abs\((\w+)\)[^\n]*abs\((\w+)\)",
                      self.code)
        self.assertIsNotNone(m, 'the frontalization gate moved or changed shape')
        self.assertEqual({m.group(1), m.group(2)}, {'tgt_yaw_deg', 'tgt_pitch_deg'})

    def test_a_frontal_face_is_below_any_sane_threshold_now(self):
        from roop.face_util import solve_pose_5pt
        from tests.test_face_pose_source import project_68, kps_from_68
        yaw, pitch, _ = solve_pose_5pt(kps_from_68(project_68(0, 0)))
        self.assertLess(max(abs(yaw), abs(pitch)), 25.0)
        # ...and what the old path produced never was.
        self.assertGreater(abs(178.5), 25.0)


class TheAlignmentIsAPureSimilarity(unittest.TestCase):
    def test_the_alignment_produces_a_pure_similarity(self):
        """A similarity is a rotation, a uniform scale and a translation. Shear
        or anisotropy here would distort the face the swapper is handed."""
        worst_aniso, worst_shear = 0.0, 0.0
        for yaw in range(-90, 91, 15):
            for pitch in (-40, 0, 40):
                for roll in (-30, 0, 30):
                    for size in (112, 256, 512):
                        A = np.asarray(estimate_norm(
                            kps_at(yaw, pitch, roll), size))[:, :2]
                        sv = np.linalg.svd(A, compute_uv=False)
                        worst_aniso = max(worst_aniso,
                                          abs(sv[0] / max(sv[1], 1e-12) - 1.0))
                        c0, c1 = A[:, 0], A[:, 1]
                        cos = float(c0 @ c1 / (np.linalg.norm(c0)
                                               * np.linalg.norm(c1) + 1e-12))
                        worst_shear = max(worst_shear, abs(cos))
        self.assertLess(worst_aniso, 1e-9, 'the fit is not a uniform scale')
        self.assertLess(worst_shear, 1e-9, 'the fit shears')


if __name__ == '__main__':
    unittest.main()
