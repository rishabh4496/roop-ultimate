"""What /api/target/auto_angles is allowed to put in a person's angle bank.

The bank is the reference every identity gate in the swap measures against, and
each of them takes the MINIMUM over a person's angles — so one wrong angle makes
a stranger's whole track measure ~0 to that person and inherit their source. A
polluted bank silently defeats gates that are otherwise working.

Two of auto-capture's four identity guards are relative (is anyone else in this
frame a better match?), which is why they can be strict without punishing a hard
pose — but both need a second face to compare against. On a frame holding one
face, and by definition on every frame where the target has walked off, neither
can run and only absolute distances remain. A typical stranger fails those; a
DEGRADED frame is what crosses them, because a blurred or tiny crop produces an
embedding that sits near everybody.

Hence an intake gate on image quality. The trap it must avoid is obvious once
stated: this feature exists to harvest PROFILES, so a gate that judges pose
would reject its whole purpose while happily admitting a crisp frontal frame of
the wrong person. These pin that separation, and the blur rule that follows from
it.
"""

import os
import sys
import unittest

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from roop.face_quality import (image_quality, score_face, blur_outlier,   # noqa: E402
                               BLUR_FRAC, BLUR_WARMUP)


def _face_image(size=200):
    """A face is mostly SMOOTH. Random noise is the worst possible stand-in:
    it keeps high-frequency detail through a blur, so a sharpness gate measured
    against it would look far more forgiving than it really is."""
    img = np.zeros((size, size, 3), np.uint8)
    s = size / 200.0
    cv2.ellipse(img, (int(100 * s), int(110 * s)),
                (int(70 * s), int(95 * s)), 0, 0, 360, (180, 150, 130), -1)
    cv2.circle(img, (int(75 * s), int(90 * s)), max(1, int(9 * s)), (40, 40, 40), -1)
    cv2.circle(img, (int(125 * s), int(90 * s)), max(1, int(9 * s)), (40, 40, 40), -1)
    cv2.ellipse(img, (int(100 * s), int(150 * s)),
                (int(28 * s), int(12 * s)), 0, 0, 180, (90, 60, 60), -1)
    return img


class _Face:
    def __init__(self, px=200, det_score=0.9, yaw=0.0):
        self.bbox = np.array([0, 0, px, px], np.float32)
        self.det_score = det_score
        self.embedding = np.ones(512, np.float32) * 0.9
        # Both pose sources agree, so neither can quietly carry the result.
        self.pose = np.array([0.0, yaw, 0.0], np.float32)
        off = 0.5 * min(yaw, 80.0) / 80.0
        self.kps = np.array([[0.3 * px, 0.4 * px], [0.7 * px, 0.4 * px],
                             [(0.5 + off) * px, 0.55 * px],
                             [0.35 * px, 0.7 * px], [0.65 * px, 0.7 * px]], np.float32)


class PoseIndependenceTest(unittest.TestCase):
    """The property the whole gate rests on."""

    def test_a_profile_is_not_penalised(self):
        """Same crop, same size, same confidence — only the pose differs. The
        intake score must not move; if it does, the gate is rejecting the
        angles the feature was built to collect."""
        crop = _face_image()
        frontal, _ = image_quality(_Face(yaw=0.0), crop)
        profile, _ = image_quality(_Face(yaw=80.0), crop)
        self.assertAlmostEqual(frontal, profile, places=6)

    def test_the_composite_score_does_penalise_it(self):
        """Contrast, and the reason a separate function exists: score_face is
        right for ranking crops to BLEND (a profile blends badly) and wrong as
        an intake gate. Reusing it here would have looked reasonable."""
        crop = _face_image()
        frontal, _ = score_face(_Face(yaw=0.0), crop)
        profile, _ = score_face(_Face(yaw=80.0), crop)
        self.assertGreater(frontal - profile, 0.05)

    def test_frontality_is_absent_from_the_breakdown(self):
        _q, bits = image_quality(_Face(), _face_image())
        self.assertNotIn('frontality', bits)
        self.assertIn('sharpness', bits)


class DegradationIsDetectedTest(unittest.TestCase):

    def _sharp(self, face, crop):
        return image_quality(face, crop)[1]['sharpness']

    def test_blur_collapses_the_sharpness_axis(self):
        clean = _face_image()
        mild = cv2.GaussianBlur(clean, (9, 9), 3)
        heavy = cv2.GaussianBlur(clean, (31, 31), 12)
        f = _Face()
        self.assertGreater(self._sharp(f, clean), 0.5)
        self.assertLess(self._sharp(f, mild), 0.2)
        self.assertLess(self._sharp(f, heavy), self._sharp(f, mild) + 1e-6)

    def test_face_px_is_reported_for_the_size_floor(self):
        _q, bits = image_quality(_Face(px=48), _face_image(48))
        self.assertEqual(bits['face_px'], 48)

    def test_the_composite_cannot_be_the_blur_gate(self):
        """Why a dedicated blur rule exists: a heavily blurred but LARGE,
        confidently-detected face still totals mid-range, because size and
        confidence prop up exactly the frame whose embedding is worthless."""
        heavy = cv2.GaussianBlur(_face_image(), (31, 31), 12)
        q, _ = image_quality(_Face(px=200, det_score=0.7), heavy)
        self.assertGreater(q, 0.35, 'a weighted average hides this — hence blur_outlier')


class RelativeBlurRuleTest(unittest.TestCase):
    """An absolute sharpness cutoff cannot serve both a crisp and a soft clip,
    so the bar is the median of what this clip actually offers."""

    def test_rejects_a_candidate_far_below_the_clips_median(self):
        samples = [0.9] * 10
        self.assertTrue(blur_outlier(0.05, samples))

    def test_keeps_an_ordinary_candidate(self):
        samples = [0.9] * 10
        self.assertFalse(blur_outlier(0.8, samples))

    def test_a_uniformly_soft_clip_is_not_wiped_out(self):
        """The failure mode of a fixed threshold: every face on a soft or
        heavily compressed clip scores low, and an absolute gate would refuse
        all of them. Relative to its own median, an ordinary frame passes."""
        samples = [0.12, 0.14, 0.11, 0.13, 0.15, 0.10, 0.13, 0.12]
        self.assertFalse(blur_outlier(0.11, samples))
        self.assertTrue(blur_outlier(0.02, samples), 'the worst is still refused')

    def test_nothing_is_refused_before_the_median_is_trustworthy(self):
        self.assertFalse(blur_outlier(0.01, [0.9, 0.9, 0.9]))
        self.assertFalse(blur_outlier(0.01, []))

    def test_disabled_by_zero(self):
        self.assertFalse(blur_outlier(0.0, [0.9] * 20, frac=0))

    def test_the_shipped_default_is_on(self):
        """Every other assertion here passes `frac` explicitly, so all of them
        would still pass with the gate disabled by env — which is exactly how a
        gate ends up dead with a green suite. This one calls it the way api.py
        does, with no override, so it fails if the default is ever turned off.
        """
        self.assertGreater(BLUR_FRAC, 0)
        self.assertGreaterEqual(BLUR_WARMUP, 1)
        self.assertTrue(blur_outlier(0.02, [0.9] * BLUR_WARMUP))

    def test_the_call_site_passes_no_second_default(self):
        """api.py must forward the request override or None, never re-read the
        env — two owners of one threshold is how a gate ends up on in one place
        and off in another."""
        src = open(os.path.join(os.path.dirname(__file__), '..', 'api.py'),
                   encoding='utf-8').read()
        self.assertNotIn("os.environ.get('ROOP_ANGLE_BLUR_FRAC'", src)
        self.assertIn('BLUR_FRAC = payload.get("blur_frac")', src)

    def test_median_not_mean(self):
        """One pristine frame among blurred ones must not drag the bar up: the
        median ignores it, a mean would not."""
        samples = [0.10] * 9 + [1.0]
        self.assertFalse(blur_outlier(0.09, samples))


class EndpointWiringTest(unittest.TestCase):
    """Importing api.py works (surface_snapshot.py does it) but pulls in the
    whole model stack and costs seconds against a suite that runs in ten, so the
    wiring is asserted at the source instead. Each of these is a gate that
    silently does nothing if its call site is dropped — and the route binding
    itself is covered by surface_snapshot, which is the only thing that catches
    a helper inserted under an @app.post decorator rebinding the route."""

    @classmethod
    def setUpClass(cls):
        cls.src = open(os.path.join(os.path.dirname(__file__), '..', 'api.py'),
                       encoding='utf-8').read()

    def test_intake_uses_the_pose_free_score(self):
        self.assertIn('from roop.face_quality import image_quality, blur_outlier', self.src)
        self.assertIn('qual, qbits = image_quality(best_f, crop)', self.src)
        self.assertNotIn('score_face(best_f', self.src)

    def test_every_candidate_feeds_the_blur_median(self):
        """Recording only ACCEPTED faces would make the median describe what got
        through rather than what was available, and the gate would drift shut."""
        i_append = self.src.index('sharp_samples.append(sharp)')
        i_gate = self.src.index('blur_outlier(sharp, sharp_samples')
        self.assertLess(i_append, i_gate, 'sample must be recorded before the gate')

    def test_the_blur_median_is_sampled_before_the_pose_budget(self):
        """The gate was DEAD when first written, and a green suite said nothing.

        Sampling sat after the novelty and per-bin-cap returns, which skip most
        frames on a clip without much pose variety — so the median collected a
        handful of values, the warm-up count was never reached, blur_outlier
        returned False every time and blurred frames were banked. Only driving
        the endpoint showed it. These two orderings are what keep the gate fed.
        """
        i_append = self.src.index('sharp_samples.append(sharp)')
        i_novelty = self.src.index('if best_d < NOVELTY and bin_counts.get')
        i_cap = self.src.index('if bin_counts.get(pose_bin, 0) >= PER_BIN_CAP')
        self.assertLess(i_append, i_novelty,
                        'novelty skip must not starve the blur median')
        self.assertLess(i_append, i_cap,
                        'the per-bin cap must not starve the blur median')

    def test_the_unguarded_frame_gate_is_present(self):
        self.assertIn("if len(scored) == 1 and not other_embeddings and best_d > LONE_ACCEPT",
                      self.src)

    def test_the_review_report_is_returned(self):
        self.assertIn('"review": review', self.src)
        self.assertIn('"rejected": dict(rejected)', self.src)

    def test_the_ui_reads_the_fields_the_backend_sends(self):
        """New UI code has shipped reading field names the backend never sent.
        Both halves are checked here rather than assumed."""
        ui = open(os.path.join(os.path.dirname(__file__), '..', '..', 'react-ui',
                               'src', 'components', 'PersonGroups.jsx'),
                  encoding='utf-8').read()
        self.assertIn('res.review', ui)
        self.assertIn('res.rejected', ui)


if __name__ == '__main__':
    unittest.main()
