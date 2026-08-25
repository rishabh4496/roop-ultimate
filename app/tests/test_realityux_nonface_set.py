"""The BiSeNet class set RealityUX subtracts, and its documentation, must agree.

RealityUX lets the Face Parser subtract pixels from XSeg's swap region, but only
for classes it is unambiguously right about. WHICH classes was recorded in two
places that disagreed: a module-level `_NONFACE_STRICT` listing seven classes
INCLUDING background(0), referenced by name from both docstrings, and a
hardcoded six-class list inside `Run()` that omitted background and was what
actually executed. The constant was dead code, so the disagreement could not
show up as a bug — only as a reader believing something false about the exact
class the surrounding comment is about.

Background is the one that matters. BiSeNet's frontal priors label the outer
part of an angled or lying-down face as background; subtracting it cuts real
faces in half, which is why it was taken out of the applied list. A future
reader following the docs back to the constant would have put it back.
"""

import os
import re
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

MODULE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      'roop', 'processors', 'Mask_RealityUX.py')


class TheSubtractedClassSet(unittest.TestCase):

    def setUp(self):
        from roop.processors import Mask_RealityUX
        self.mod = Mask_RealityUX

    def test_background_is_not_subtractable(self):
        """The whole reason the two lists differed."""
        self.assertNotIn(0, self.mod._NONFACE_OPAQUE,
                         'subtracting BiSeNet background(0) halves angled faces')

    def test_glasses_and_neck_are_not_subtractable(self):
        """Both measured to cause real harm — see the module comment."""
        for cls in (6, 14, 15):
            self.assertNotIn(cls, self.mod._NONFACE_OPAQUE)

    def test_the_opaque_non_face_classes_are_all_present(self):
        """ears, cloth, hair, hat."""
        self.assertEqual(sorted(self.mod._NONFACE_OPAQUE), [7, 8, 9, 16, 17, 18])

    def test_the_set_is_applied_by_name_not_re_listed(self):
        """A second literal list is how the two drifted apart in the first
        place, so the constant has to be the only spelling of it."""
        src = open(MODULE, encoding='utf-8').read()
        body = src.split('_NONFACE_OPAQUE = ', 1)[1].split('\n', 1)[1]
        self.assertNotRegex(
            body, r'np\.isin\(\s*labels\s*,\s*\[',
            'Run() must apply _NONFACE_OPAQUE, not a fresh literal list')
        self.assertIn('np.isin(labels, _NONFACE_OPAQUE)', body)

    def test_the_dead_constant_survives_only_as_history(self):
        """The old name is worth keeping greppable — the session notes that
        reported this refer to it — but only in prose. As live code it is the
        second spelling all over again."""
        src = open(MODULE, encoding='utf-8').read()
        for n, line in enumerate(src.splitlines(), 1):
            if '_NONFACE_STRICT' not in line:
                continue
            self.assertTrue(line.lstrip().startswith('#'),
                            f'line {n}: _NONFACE_STRICT is back as code, not history')


class TheDocsDoNotContradictTheCode(unittest.TestCase):
    """Both docstrings named background as something BiSeNet is 'allowed to
    subtract'. That is the sentence this test exists to keep false."""

    def setUp(self):
        self.src = open(MODULE, encoding='utf-8').read()
        # Everything above Run()'s body: the module comment and the class
        # docstring, i.e. what a reader meets before any code.
        self.prose = self.src.split('def Run', 1)[0]

    def test_the_prose_does_not_list_background_as_subtracted(self):
        allowed = re.search(r'Classes BiSeNet is allowed to subtract.*?--(.*?)\.',
                            self.prose, re.S)
        self.assertIsNotNone(allowed, 'the class list comment moved or was removed')
        self.assertNotIn('background', allowed.group(1).lower())

    def test_the_prose_says_background_is_excluded_on_purpose(self):
        """Removing the wrong claim is not enough; the reason has to survive or
        someone re-adds class 0 as an obvious omission."""
        self.assertRegex(self.prose, r'background\(0\)')
        self.assertIn('deliberately not any more', self.prose)


if __name__ == '__main__':
    unittest.main()
