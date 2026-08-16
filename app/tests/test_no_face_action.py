"""The "no face" policy must be the policy that was chosen.

`no_face_action` is a five-way choice about what to do with a frame the swap did
not land on. Four of the five are non-retry policies; only "Retry rotated" asks
for a second detection pass. The dispatch at the end of `process_frame` is an
if/elif chain with a bare `else` on the end, which makes it silently wrong in
one specific way: any value the chain has no branch for lands in the `else` and
gets Retry-rotated behaviour.

That is what "Skip Frame if no similar face" did. Its only branch sat under
`num_swapped > 0` (the partial case — some sources matched, some did not), so a
frame where NOTHING was swapped — the case the setting exists to decide — fell
past every branch into the retry. The user asked to drop the frame and got an
extra full detection pass and the frame kept.

These read the source rather than calling `process_frame`: importing ProcessMgr
costs ~9s and initialises CUDA, which does not belong in a suite that runs in
five seconds. What is asserted is the shape of the dispatch, which is where the
defect lived.
"""

import os
import re
import sys
import unittest
from pathlib import Path

APP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, APP)

SRC = Path(APP, 'roop', 'ProcessMgr.py').read_text(encoding='utf-8')

# The body of process_frame, up to the start of the next method.
BODY = re.search(r'\n    def process_frame\(self.*?\n    def ', SRC, re.S).group(0)
# Everything after the `num_swapped > 0` early-return block: the zero-swap
# dispatch, which is the part under test. Comments are stripped — the fix for
# this bug mentions the constant in a comment, and a test that a comment
# satisfies is not a test.
def _code_only(text):
    return '\n'.join(re.sub(r'#.*$', '', ln) for ln in text.splitlines())


ZERO_SWAP = _code_only(
    BODY.split('if num_swapped > 0:', 1)[1].split('return temp_frame', 1)[1])


class NoFaceActionDispatch(unittest.TestCase):

    def test_the_body_was_actually_found(self):
        """Guard the parsing — an empty body would make every test below pass."""
        self.assertIn('no_face_action', BODY)
        self.assertIn('retry_rotated', ZERO_SWAP)
        self.assertGreater(len(ZERO_SWAP), 200)

    def test_skip_if_dissimilar_is_handled_when_nothing_was_swapped(self):
        """The regression: no branch for it, so it fell into the retry."""
        self.assertIn('SKIP_FRAME_IF_DISSIMILAR', ZERO_SWAP,
                      '"Skip Frame if no similar face" has no branch in the '
                      'zero-swap dispatch, so it silently retries rotated instead')

    def test_every_action_the_ui_offers_is_dispatched(self):
        """A new choice added to the UI list must reach a branch here.

        The five names in api.py's no_face_choices are index-mapped onto
        eNoFaceAction, so the two lists have to stay the same length — an extra
        UI entry with no enum member maps to an int nothing compares equal to,
        and lands in the retry `else` exactly as this bug did.
        """
        api = Path(APP, 'api.py').read_text(encoding='utf-8')
        choices = re.findall(r'"([^"]+)"', re.search(
            r'no_face_choices = \[(.*?)\]', api, re.S).group(1))
        members = re.findall(r'^    ([A-Z_]+) = (\d+)$',
                             SRC.split('class eNoFaceAction')[1].split('\n\n')[0], re.M)
        self.assertEqual(len(choices), len(members),
                         f'{len(choices)} UI choices vs {len(members)} enum members — '
                         f'the index mapping is broken')
        self.assertEqual([int(v) for _, v in members], list(range(len(members))),
                         'eNoFaceAction values must be 0..N-1 to index-map onto '
                         'no_face_choices')
        # Every member is named somewhere in process_frame's dispatch.
        for name, _ in members:
            if name == 'RETRY_ROTATED':
                continue          # it IS the fallthrough; it has no named branch
            with self.subTest(action=name):
                self.assertIn(name, BODY,
                              f'{name} is offered in the UI but never compared '
                              f'against in process_frame — it silently retries')


if __name__ == '__main__':
    unittest.main()
