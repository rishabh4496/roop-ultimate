"""Mask_FaceParser region selection.

The Face Parser produced all 19 CelebAMask-HQ classes and threw 10 of them
away against a hardcoded array, so hair, glasses, ears and neck were
permanently outside the swap region with no way to say otherwise. They are
selectable now, with a per-region grow.

What must hold:

  * THE DEFAULT IS THE OLD BEHAVIOUR, BIT FOR BIT. This engine is a default in
    a pipeline people have already tuned against; a mask that moved by a pixel
    on an untouched install would change every render subtly and be blamed on
    something else entirely. The old expression is kept here as the oracle.

  * GROW IS PER REGION. Dilating the union instead of each part would be
    cheaper and wrong: asking for a little more mouth would also push the
    outer boundary of the whole face outward and swallow a ring of background.
    That distinction is invisible in a mask you glance at and obvious in the
    seam it leaves.

  * THE UI AND THE ENGINE AGREE ON THE REGION NAMES. They are matched by
    string across a JSON boundary, so a rename on either side silently drops a
    region — the panel keeps offering it and the engine keeps ignoring it.
"""

import os
import re
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import roop.globals                                       # noqa: E402
from roop.processors.Mask_FaceParser import (             # noqa: E402
    PARSER_DEFAULT_ON, PARSER_REGIONS, _FACE_CLASSES, _region_mask,
)

APP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO = os.path.dirname(APP)
PANEL = os.path.join(REPO, 'react-ui', 'src', 'components', 'faceswap', 'ParserRegions.jsx')


def labels(seed=0, size=64):
    """A synthetic parse map holding every class, in blobs."""
    rng = np.random.default_rng(seed)
    out = np.zeros((size, size), dtype=np.int64)
    for cls in range(19):
        y, x = rng.integers(0, size - 8, 2)
        out[y:y + 8, x:x + 8] = cls
    return out


class DefaultsAreUnchanged(unittest.TestCase):
    def setUp(self):
        roop.globals.parser_regions = None
        roop.globals.parser_region_grow = None

    def tearDown(self):
        roop.globals.parser_regions = None
        roop.globals.parser_region_grow = None

    def test_untouched_settings_reproduce_the_old_mask(self):
        for seed in range(6):
            lab = labels(seed)
            oracle = np.isin(lab, _FACE_CLASSES).astype(np.float32)
            np.testing.assert_array_equal(_region_mask(lab), oracle)

    def test_the_default_group_set_is_the_old_class_array(self):
        ids = sorted({c for g in PARSER_DEFAULT_ON for c in PARSER_REGIONS[g]})
        self.assertEqual(ids, sorted(_FACE_CLASSES.tolist()),
                         'the default groups must expand to exactly the class '
                         'list this engine has always used')

    def test_an_empty_or_junk_selection_falls_back_rather_than_blanking(self):
        """An empty mask means a completely un-swapped face, which reads as a
        broken app rather than a setting."""
        lab = labels(1)
        oracle = np.isin(lab, _FACE_CLASSES).astype(np.float32)
        for bad in ([], None, 'skin', ['nonsense'], 42):
            roop.globals.parser_regions = bad
            np.testing.assert_array_equal(_region_mask(lab), oracle,
                                          f'{bad!r} should fall back to the default set')


class RegionSelection(unittest.TestCase):
    def setUp(self):
        roop.globals.parser_region_grow = None

    def tearDown(self):
        roop.globals.parser_regions = None
        roop.globals.parser_region_grow = None

    def test_including_hair_adds_exactly_the_hair_class(self):
        lab = labels(2)
        roop.globals.parser_regions = list(PARSER_DEFAULT_ON)
        without = _region_mask(lab)
        roop.globals.parser_regions = list(PARSER_DEFAULT_ON) + ['hair']
        with_hair = _region_mask(lab)
        added = (with_hair > 0) & ~(without > 0)
        self.assertTrue(added.any(), 'adding hair changed nothing')
        self.assertTrue((lab[added] == 17).all(),
                        'adding hair pulled in classes other than hair')

    def test_excluding_a_region_removes_only_that_region(self):
        lab = labels(3)
        roop.globals.parser_regions = list(PARSER_DEFAULT_ON)
        full = _region_mask(lab)
        roop.globals.parser_regions = [g for g in PARSER_DEFAULT_ON if g != 'nose']
        without = _region_mask(lab)
        lost = (full > 0) & ~(without > 0)
        self.assertTrue(lost.any())
        self.assertTrue((lab[lost] == 10).all(),
                        'dropping the nose dropped something else too')


class GrowIsPerRegion(unittest.TestCase):
    def tearDown(self):
        roop.globals.parser_regions = None
        roop.globals.parser_region_grow = None

    def test_zero_grow_matches_the_fast_path(self):
        lab = labels(4)
        roop.globals.parser_regions = list(PARSER_DEFAULT_ON)
        roop.globals.parser_region_grow = {g: 0 for g in PARSER_DEFAULT_ON}
        grown = _region_mask(lab)
        roop.globals.parser_region_grow = None
        np.testing.assert_array_equal(grown, _region_mask(lab))

    def test_growing_a_region_only_expands_that_region(self):
        """The property that separates per-region growth from growing the
        union: the extra pixels must touch the grown part, not the outer edge
        of some other one."""
        lab = np.zeros((80, 80), dtype=np.int64)
        lab[10:20, 10:20] = 1        # skin, far left
        lab[10:20, 60:70] = 11       # mouth, far right
        roop.globals.parser_regions = ['skin', 'mouth']
        roop.globals.parser_region_grow = None
        base = _region_mask(lab)
        roop.globals.parser_region_grow = {'mouth': 5}
        grown = _region_mask(lab)

        added = (grown > 0) & ~(base > 0)
        self.assertTrue(added.any(), 'grow did nothing')
        # Everything new is on the mouth side of the frame.
        xs = np.nonzero(added)[1]
        self.assertGreater(xs.min(), 40,
                           'growing the mouth expanded the skin region too — '
                           'the dilation is being applied to the union')

    def test_grow_is_monotonic(self):
        lab = labels(5)
        roop.globals.parser_regions = list(PARSER_DEFAULT_ON)
        areas = []
        for px in (0, 2, 6):
            roop.globals.parser_region_grow = {'skin': px}
            areas.append(int((_region_mask(lab) > 0).sum()))
        self.assertEqual(areas, sorted(areas))
        self.assertLess(areas[0], areas[2], 'grow never expanded the mask')


class TheUiAgrees(unittest.TestCase):
    def test_the_panel_offers_exactly_the_engine_s_regions(self):
        src = open(PANEL, encoding='utf-8').read()
        block = src.split('const REGION_LABELS', 1)[1].split('];', 1)[0]
        offered = set(re.findall(r"\['([a-z]+)',", block))
        self.assertEqual(
            offered, set(PARSER_REGIONS),
            'ParserRegions.jsx and Mask_FaceParser disagree on the region '
            'names; they are matched by string across JSON, so a mismatch '
            'means the panel offers a control the engine ignores')

    def test_the_panel_defaults_match_the_engine_defaults(self):
        src = open(PANEL, encoding='utf-8').read()
        line = re.search(r'const DEFAULT_ON = \[([^\]]*)\]', src).group(1)
        self.assertEqual(set(re.findall(r"'([a-z]+)'", line)), set(PARSER_DEFAULT_ON),
                         "the panel's reset would produce a different mask than "
                         'a fresh install')


if __name__ == '__main__':
    unittest.main()
