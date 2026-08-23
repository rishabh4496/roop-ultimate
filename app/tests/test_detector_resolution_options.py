"""The detection-resolution dropdown must offer the size that measures best.

`detect` is the largest stage in a render — 42.4% of one measured 60,460-frame
job. Measured on an RTX 4070, retinaface_r50, 240 frames of angled footage, with
the module set a real render builds (landmark_2d_106 + recognition; genderage is
already excluded for non-gender swap modes and landmark_3d_68 is already lazy):

    640px   14.27 ms/frame    98.5% recall
    512px   10.95 ms/frame    99.4% recall

512 is 1.30x faster AND slightly more accurate, with landmarks shifting only
0.24-0.72 px (p95 1.54 px on hard poses). It was not selectable: the dropdown
offered 320/640/960/1280 only, so the best setting on 720p and 1080p sources was
unreachable from the UI.

The reason it wins is geometry rather than the model: a 16:9 frame letterboxed
into a square canvas leaves ~44% of that canvas black, so most of 640's extra
pixels are padding.

These are source-level assertions. The alternative is a browser, and the failure
being guarded is simply "the option is missing from the list".
"""

import os
import re
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.dirname(HERE)
ROOT = os.path.dirname(APP)
PANEL = os.path.join(ROOT, 'react-ui', 'src', 'components', 'FaceSwap.jsx')


def panel_src():
    with open(PANEL, encoding='utf-8') as f:
        return f.read()


class TestResolutionOptions(unittest.TestCase):
    def setUp(self):
        if not os.path.exists(PANEL):
            self.skipTest('React panel not present')
        self.src = panel_src()
        m = re.search(r"label=\"Face detection resolution\".*?options=\{\[(.*?)\]\}",
                      self.src, re.S)
        self.assertIsNotNone(m, 'resolution Select not found — the guard is dead')
        self.options = re.findall(r"'(\d+)'", m.group(1))

    def test_512_is_offered(self):
        self.assertIn('512', self.options,
                      "512 measured 1.30x faster AND more accurate than 640; "
                      "leaving it out makes the best setting unreachable")

    def test_the_standard_sizes_are_still_there(self):
        for s in ('320', '640', '960', '1280'):
            self.assertIn(s, self.options)

    def test_options_are_sorted_so_the_menu_reads_in_order(self):
        vals = [int(x) for x in self.options]
        self.assertEqual(vals, sorted(vals))

    def test_every_option_is_a_multiple_of_32(self):
        """The detector's strides are 8/16/32, so a size that is not a multiple
        of 32 does not tile the feature maps cleanly."""
        for s in self.options:
            with self.subTest(size=s):
                self.assertEqual(int(s) % 32, 0)

    def test_the_help_text_carries_the_measurement(self):
        """A number the operator can check beats 'higher is better'."""
        m = re.search(r"label=\"Face detection resolution\"\s*\n\s*info=\"(.*?)\"",
                      self.src, re.S)
        self.assertIsNotNone(m)
        info = m.group(1)
        self.assertIn('512', info)
        self.assertIn('recall', info.lower())

    def test_the_help_text_says_which_engines_ignore_it(self):
        """yoloface_8n and det_10g are fixed 640x640 exports. Silently ignoring
        the setting is confusing; yoloface used to CRASH on a mismatch."""
        m = re.search(r"label=\"Face detection resolution\"\s*\n\s*info=\"(.*?)\"",
                      self.src, re.S)
        info = m.group(1).lower()
        self.assertIn('yoloface', info)
        self.assertIn('scrfd', info)


class TestGenderageAlreadyConditional(unittest.TestCase):
    """Not a change — a guard on something that is already right.

    `genderage` is only appended to the analyser's module list for the
    all_female / all_male swap modes, so it is not loaded in a normal render.
    Measured, it costs nothing anyway (10.95 ms/frame without it, 10.82 with),
    but a regression that loaded it unconditionally would be invisible.
    """

    def test_genderage_is_requested_only_for_gender_swap_modes(self):
        with open(os.path.join(APP, 'roop', 'ProcessMgr.py'), encoding='utf-8') as f:
            src = f.read()
        m = re.search(r'if options\.swap_mode == "all_female" or '
                      r'options\.swap_mode == "all_male":\s*\n\s*'
                      r'roop\.globals\.g_desired_face_analysis\.append\("genderage"\)', src)
        self.assertIsNotNone(
            m, "genderage must stay conditional on the gender swap modes")

    def test_the_base_module_list_excludes_genderage(self):
        with open(os.path.join(APP, 'roop', 'ProcessMgr.py'), encoding='utf-8') as f:
            src = f.read()
        m = re.search(r'modules = \[([^\]]*)\]', src)
        self.assertIsNotNone(m)
        self.assertNotIn('genderage', m.group(1))


if __name__ == '__main__':
    unittest.main()
