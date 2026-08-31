"""DMDNet must not die when a face carries no 106-point landmarks.

`insightface`'s `Face.__getattr__` returns None for a key the analyser never
wrote, so `face.landmark_2d_106` is None -- not a KeyError -- whenever the
106-point model did not run on that face. `Enhance_DMDNet.enhance_face` indexed
that value directly, and `landmarks106_to_68` does `pt106[index]`, which is
`TypeError: 'NoneType' object is not subscriptable`. That is the exact error
DMDNet raised mid-render on the RTX 3060 while every other enhancer ran.

Every other consumer of this attribute in the app already guards it --
`procmgr_masking.restore_original_mouth`, `face_util:694`, `face_overlap:212`,
`ProcessMgr:5086` -- and `apply_eyes_area`'s docstring states outright that
lm106 "is optional (it is absent unless the 106 model ran)". DMDNet was the
only consumer that assumed otherwise.

The second half of this guard matters as much as the first: a fallback that
silently skips every face still produces a valid video, and both the swap audit
and the output-integrity sweep PASS it -- that is precisely how four enhancers
came to be reported at "100% success" on 2026-08-30 while failing on 60 of 60
frames. So the skip is counted, and the counters are asserted here.
"""

import ast
import os
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.dirname(HERE)
SRC = os.path.join(APP, 'roop', 'processors', 'Enhance_DMDNet.py')


def _source():
    with open(SRC, encoding='utf-8') as f:
        return f.read()


class TestDMDNetLandmarkGuard(unittest.TestCase):

    def test_no_unguarded_landmark_attribute_read(self):
        """`face.landmark_2d_106` must never be read as a bare attribute.

        The bare form returns None instead of raising, so the failure surfaces
        later at the subscript. Reads go through `getattr(..., None)` and are
        checked before use.
        """
        src = _source()
        self.assertNotIn(
            '= face.landmark_2d_106', src,
            "bare attribute read reintroduced: Face.__getattr__ yields None "
            "for an absent key, which crashes at the subscript in "
            "landmarks106_to_68")
        self.assertNotIn('= ref_face.landmark_2d_106', src)

    def test_both_call_sites_are_guarded(self):
        """Both the target face and each reference face are checked."""
        src = _source()
        self.assertIn("lm106 = getattr(face, 'landmark_2d_106', None)", src)
        self.assertIn(
            "ref_lm106 = getattr(ref_face, 'landmark_2d_106', None)", src)
        self.assertIn("if lm106 is None:", src)
        self.assertIn("if ref_lm106 is None or i >= len(ref_images):", src)

    def test_reference_loop_does_not_shadow_the_target_face(self):
        """The reference loop variable must not be named `face`.

        `enhance_face(self, ref_faceset, temp_frame, face)` takes the TARGET
        face as `face`. The loop used to rebind that name, so `face.matrix`
        inside the loop read a reference face while reading as the target's.
        """
        tree = ast.parse(_source())
        fn = next(n for n in ast.walk(tree)
                  if isinstance(n, ast.FunctionDef) and n.name == 'enhance_face')
        for node in ast.walk(fn):
            if isinstance(node, ast.For) and isinstance(node.target, ast.Tuple):
                names = [e.id for e in node.target.elts if isinstance(e, ast.Name)]
                self.assertNotIn(
                    'face', names,
                    "the reference loop rebinds the target-face parameter")

    def test_empty_specific_dictionary_falls_back_to_generic(self):
        """Skipping references must not leave torch.cat() an empty list.

        `torch.cat([])` raises. When fewer than two references survive the
        landmark check there is no specific dictionary to build, and DMDNet's
        generic path -- which needs no reference landmarks at all -- is the
        correct outcome rather than an exception.
        """
        src = _source()
        self.assertIn('if len(SpecificImgs) < 2:', src)
        self.assertIn('SpMem256Para = SpMem128Para = SpMem64Para = None', src)

    def test_skips_are_counted_and_reported(self):
        """A silent fallback is the worse failure; the tally is mandatory."""
        src = _source()
        self.assertIn("_LM_MISS = {'target': 0, 'ref': 0, 'ok': 0}", src)
        self.assertIn("_LM_MISS['target'] += 1", src)
        self.assertIn("_LM_MISS['ref'] += 1", src)
        self.assertIn("_LM_MISS['ok'] += 1", src)
        self.assertIn('faces enhanced', src)

    def test_landmarks106_to_68_still_maps_all_68_points(self):
        """The guard must not have disturbed the mapping table itself."""
        import re
        src = _source()
        m = re.search(r'map106to68=\[(.*?)\]', src, re.S)
        self.assertIsNotNone(m)
        idx = [int(x) for x in re.findall(r'\d+', m.group(1))]
        self.assertEqual(len(idx), 68)
        self.assertLess(max(idx), 106)


class TestDMDNetGuardExecutes(unittest.TestCase):
    """Exercise the guard against a stand-in Face, no model and no GPU."""

    def test_missing_target_landmarks_returns_input_unchanged(self):
        import numpy as np
        try:
            from roop.processors.Enhance_DMDNet import Enhance_DMDNet, _LM_MISS
        except Exception as e:                       # pragma: no cover
            self.skipTest(f'torch stack unavailable: {e}')

        class FakeFace(dict):
            # insightface's Face returns None for an absent key rather than
            # raising -- the behaviour this whole guard exists for.
            def __getattr__(self, k):
                return self.get(k)

        frame = np.zeros((256, 256, 3), np.uint8)
        face = FakeFace(bbox=np.array([0, 0, 64, 64], np.float32))
        self.assertIsNone(face.landmark_2d_106)

        before = _LM_MISS['target']
        out = Enhance_DMDNet().enhance_face(None, frame, face)
        self.assertIs(out, frame, 'must hand back the un-enhanced crop')
        self.assertEqual(_LM_MISS['target'], before + 1,
                         'the skip must be counted, not swallowed')


if __name__ == '__main__':
    unittest.main()
