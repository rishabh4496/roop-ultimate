"""Semantic-parsing protection for hair/bangs and eyeglass frames.

The measurements these assertions encode were taken with
`tests/probe_parser_protection.py` and `tests/probe_glasses_rim.py` over 13
clips of the roster (~21.7M parsed hair pixels, ~800k glasses pixels on three
separate subjects):

    hair/bangs   painted by XSeg alone 0.8%   after the fusion 0.3%
    glasses      painted by XSeg alone 71%    after the fusion 71%  (unchanged)

Hair is therefore already protected, and is asserted here as a REGRESSION
guard rather than as new behaviour. Glasses were not protected at all, and
that is what `glasses_frame_mask` addresses.

No model or GPU is used: the parser's class-id map is the unit under test, so
these build label maps directly and stub the two ONNX calls.
"""
import os
import sys
import unittest

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.dirname(HERE)
if APP not in sys.path:
    sys.path.insert(0, APP)

import roop.globals                                            # noqa: E402
from roop.processors.Mask_FaceParser import (                  # noqa: E402
    GLASSES_CLASS, PARSER_REGIONS, PARSER_DEFAULT_ON, _eye_socket,
    _region_mask, glasses_frame_mask,
)

CROP = 512
HAIR, SKIN, L_EYE = 17, 1, 4


def eye_centres(crop=CROP, size=512):
    from roop.face_util import swap_template_points
    dst = swap_template_points(crop, 'arcface') * (float(size) / float(crop))
    return dst[0], dst[1]


def blank(size=512, fill=SKIN):
    return np.full((size, size), fill, dtype=np.int64)


def spectacles(size=512, crop=CROP, rim=8, lens_r=44):
    """A plausible class-6 layout: two filled lenses joined by a bridge, with
    temple arms running out to the edge. BiSeNet labels lens AND frame as 6,
    which is the entire difficulty."""
    labels = blank(size)
    left, right = eye_centres(crop, size)
    m = np.zeros((size, size), np.uint8)
    for c in (left, right):
        cv2.circle(m, (int(c[0]), int(c[1])), lens_r, 1, -1)
    cv2.line(m, (int(left[0]), int(left[1])), (int(right[0]), int(right[1])),
             1, rim)                                            # bridge
    cv2.line(m, (int(left[0] - lens_r), int(left[1])), (0, int(left[1]) - 20),
             1, rim)                                            # temple arm L
    cv2.line(m, (int(right[0] + lens_r), int(right[1])),
             (size - 1, int(right[1]) - 20), 1, rim)            # temple arm R
    labels[m > 0] = GLASSES_CLASS
    return labels


class GlassesFrameProtection(unittest.TestCase):
    def setUp(self):
        self._saved = getattr(roop.globals, 'glasses_frame_protect', True)
        roop.globals.glasses_frame_protect = True

    def tearDown(self):
        roop.globals.glasses_frame_protect = self._saved

    def test_the_frame_is_protected(self):
        """The bridge is frame, never lens, and must be kept from the swap."""
        m = glasses_frame_mask(spectacles(), CROP)
        self.assertIsNotNone(m)
        left, right = eye_centres()
        mid = ((left + right) / 2.0).astype(int)
        self.assertGreater(float(m[mid[1], mid[0]]), 0.5,
                           'the bridge of the glasses is being painted over')

    def test_the_lens_over_the_eye_stays_swappable(self):
        """The requirement `_NONFACE_OPAQUE` records: the swapped eyes must be
        the faceset's own, never the original's, even under a lens. That is the
        exact reason class 6 could not simply be added to that set."""
        m = glasses_frame_mask(spectacles(), CROP)
        for c in eye_centres():
            self.assertEqual(float(m[int(c[1]), int(c[0])]), 0.0,
                             'the lens over the pupil is being kept from the '
                             'swap, which shows the original eye')

    def test_dilation_never_pushes_protection_back_over_the_eye(self):
        """The 2px grow covering the anti-aliased label boundary has to be
        applied BEFORE the socket is removed. Applied after, it walks the
        protected region back over the pupil -- silently, and only on frames
        where the frame sits close to the eye."""
        m = glasses_frame_mask(spectacles(lens_r=60), CROP)
        self.assertIsNotNone(m)
        for c in eye_centres():
            self.assertEqual(float(m[int(c[1]), int(c[0])]), 0.0)

    def test_no_glasses_returns_none_rather_than_a_zero_array(self):
        self.assertIsNone(glasses_frame_mask(blank(), CROP))

    def test_a_few_stray_pixels_are_label_noise_not_spectacles(self):
        labels = blank()
        labels[10:16, 10:16] = GLASSES_CLASS          # 36 px
        self.assertIsNone(glasses_frame_mask(labels, CROP))

    def test_an_implausibly_large_class_6_is_refused(self):
        """Protecting half a crop would cut the face in half -- the same
        failure background(0) was removed from the fusion for."""
        labels = blank()
        labels[:, :] = GLASSES_CLASS
        self.assertIsNone(glasses_frame_mask(labels, CROP))

    def test_the_setting_turns_it_off(self):
        roop.globals.glasses_frame_protect = False
        self.assertIsNone(glasses_frame_mask(spectacles(), CROP))

    def test_the_mask_is_feathered_not_binary(self):
        """A hard cutout along the frame reads as a jagged sticker."""
        m = glasses_frame_mask(spectacles(), CROP)
        soft = ((m > 0.01) & (m < 0.99)).sum()
        self.assertGreater(soft, 0, 'mask has no soft boundary at all')

    def test_the_socket_comes_from_the_template_not_a_guess(self):
        """`arcface_dst * size/112` is the obvious reconstruction and is wrong
        by 53px at 512; a since-removed visibility polygon was built on it."""
        from roop.face_util import arcface_dst
        socket = _eye_socket((512, 512), CROP)
        for c in (arcface_dst * (512.0 / 112.0))[:2]:
            self.assertEqual(socket[int(c[1]), int(c[0])], 0,
                             'socket sits where the naive template guess puts '
                             'the eyes, not where swap_template_points does')
        for c in eye_centres():
            self.assertEqual(socket[int(c[1]), int(c[0])], 1)

    def test_a_visible_eye_is_believed_as_well_as_the_geometry(self):
        """Thin or rimless frames leave the eye parsed normally; that evidence
        should be used on top of the socket, not instead of it."""
        labels = spectacles(lens_r=70)
        off = np.array(eye_centres()[0]) + np.array([0, 46])
        # cv2 will not draw into an int64 array; stamp through a uint8 scratch.
        scratch = np.zeros(labels.shape, np.uint8)
        cv2.circle(scratch, (int(off[0]), int(off[1])), 14, 1, -1)
        labels[scratch > 0] = L_EYE
        m = glasses_frame_mask(labels, CROP)
        # Not exactly 0: this point sits ~14px from protected frame, and the
        # mask is deliberately feathered, so a tail of ~1e-5 reaches it. What
        # matters is that it carries no usable blend weight.
        self.assertLess(float(m[int(off[1]), int(off[0])]), 0.01,
                        'a parsed eye outside the geometric socket is being '
                        'kept from the swap')


class HairProtectionRegression(unittest.TestCase):
    """Hair measured 0.3% painted across 13 clips. These pin the mechanism that
    delivers it, so it cannot be lost while tuning something else."""

    def setUp(self):
        self._regions = getattr(roop.globals, 'parser_regions', None)
        roop.globals.parser_regions = None

    def tearDown(self):
        roop.globals.parser_regions = self._regions

    def test_hair_is_outside_the_default_swap_region(self):
        self.assertNotIn('hair', PARSER_DEFAULT_ON)
        labels = blank()
        labels[:200, :] = HAIR                       # fringe over the forehead
        region = _region_mask(labels)
        self.assertEqual(float(region[:200, :].max()), 0.0,
                         'the default swap region includes hair')

    def test_a_heavy_forehead_fringe_is_excluded_where_it_overlaps_skin(self):
        """The reported failure: a fringe hanging over the forehead gets
        painted. Built with an irregular boundary rather than a straight edge,
        because a straight edge would pass even if only whole rows worked."""
        labels = blank()
        rng = np.random.default_rng(0)
        for x in range(512):
            top = int(150 + 60 * np.sin(x / 40.0) + rng.integers(0, 12))
            labels[:top, x] = HAIR
        region = _region_mask(labels)
        hair = labels == HAIR
        self.assertEqual(float(region[hair].max()), 0.0)
        self.assertGreater(float(region[~hair].mean()), 0.9,
                           'excluding the fringe also ate the face below it')

    def test_hair_is_in_the_realityux_subtraction_set(self):
        from roop.processors.Mask_RealityUX import _NONFACE_OPAQUE
        self.assertIn(PARSER_REGIONS['hair'][0], _NONFACE_OPAQUE)
        self.assertNotIn(0, _NONFACE_OPAQUE,
                         'background back in the set halves angled faces')
        self.assertNotIn(GLASSES_CLASS, _NONFACE_OPAQUE,
                         'class 6 wholesale keeps the original eye under the '
                         'lens; it has its own geometric path')


class RealityUXComposite(unittest.TestCase):
    """Exercises the real Run(), with both ONNX calls stubbed."""

    def setUp(self):
        from roop.processors.Mask_RealityUX import Mask_RealityUX
        self._saved = getattr(roop.globals, 'glasses_frame_protect', True)
        roop.globals.glasses_frame_protect = True
        self.eng = Mask_RealityUX()
        self.eng._parser_enabled = True
        self.labels = spectacles()
        # XSeg wants to swap the glasses: measured p50 0.002-0.004 there.
        self.xseg = np.zeros((512, 512), np.float32)
        self.eng._parser.RunLabels = lambda img: self.labels
        self.eng._xseg.Run = lambda img, kw: self.xseg

    def tearDown(self):
        roop.globals.glasses_frame_protect = self._saved

    def _run(self):
        return self.eng.Run(np.zeros((CROP, CROP, 3), np.uint8), '')

    def test_the_frame_survives_an_xseg_that_wants_to_swap_it(self):
        """The whole defect: `accessory_allowed` scales with xseg, so at
        xseg 0 the gated path contributes exactly zero. The frame path must
        not be subject to it."""
        out = self._run()
        left, right = eye_centres()
        mid = ((left + right) / 2.0).astype(int)
        self.assertGreater(float(out[mid[1], mid[0]]), 0.5)

    def test_the_eye_is_still_swapped(self):
        out = self._run()
        for c in eye_centres():
            self.assertLess(float(out[int(c[1]), int(c[0])]), 0.5)

    def test_hair_still_obeys_the_accessory_gate(self):
        """Hair measures fine and is deliberately NOT moved off the gate. If
        someone ungates it later this fails, and the 0.3% measurement is the
        argument for leaving it alone."""
        self.labels = blank()
        self.labels[:200, :] = HAIR
        out = self._run()
        self.assertLess(float(out[:200, :].max()), 0.5,
                        'hair is now bypassing accessory_allowed')

    def test_output_stays_in_range_and_shape(self):
        out = self._run()
        self.assertEqual(out.shape, (512, 512))
        self.assertEqual(out.dtype, np.float32)
        self.assertGreaterEqual(float(out.min()), 0.0)
        self.assertLessEqual(float(out.max()), 1.0)

    def test_disabling_the_setting_restores_the_old_mask(self):
        before = self._run().copy()
        roop.globals.glasses_frame_protect = False
        after = self._run()
        self.assertGreater(float(before.max()), 0.5)
        self.assertEqual(float(after.max()), 0.0,
                         'off should reproduce the pre-change behaviour '
                         'exactly: xseg 0 everywhere, nothing added')


class SettingIsWired(unittest.TestCase):
    def test_registered_in_settings_and_the_react_defaults(self):
        with open(os.path.join(APP, 'settings.py'), encoding='utf-8') as fh:
            self.assertIn('glasses_frame_protect', fh.read())
        d = os.path.join(os.path.dirname(APP), 'react-ui', 'src',
                         'components', 'faceswap', 'defaults.js')
        if os.path.exists(d):
            with open(d, encoding='utf-8') as fh:
                self.assertIn('glasses_frame_protect', fh.read())

    def test_a_control_that_nothing_reads_is_not_wired(self):
        """A control bound to a value nothing consumes looks completely wired.
        Assert the engine actually reads the setting."""
        import inspect
        from roop.processors import Mask_FaceParser as mod
        self.assertIn('glasses_frame_protect',
                      inspect.getsource(mod.glasses_protection_enabled))


if __name__ == '__main__':
    unittest.main(verbosity=2)
