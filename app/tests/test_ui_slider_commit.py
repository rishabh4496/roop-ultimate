"""A slider drag must not re-render the whole app sixty times a second.

`<Slider>` is used at 53 call sites and its `onChange` is, at nearly all of
them, `set(key, value)` -- which writes App's `settings` state. `settings` is a
prop of every screen, so calling `onChange` from the range input's `onChange`
(which fires on EVERY pointer move, 60+ a second) re-rendered the entire tree
for the whole duration of a drag: FaceSwap at ~3,300 lines of JSX plus
InteractivePreview, Timeline and SliderTrackerBar beneath it, none memoised.

Nothing downstream ever wanted those intermediate values. The preview refresh
is debounced 350 ms in FaceSwap and the settings POST is debounced in App, so
the only thing the per-move commits bought was the re-renders themselves.

So the thumb is local state and the commit upward is debounced, with an
immediate flush on release. These tests pin that shape:

  * the range input's onChange must not call the `onChange` prop directly;
  * a release path must exist, or a drag would only commit after the debounce;
  * the prop must still be followed when it changes for an external reason
    (preset load, Reset), or those controls would stop moving the thumb.
"""

import os
import re
import unittest

APP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UI = os.path.join(os.path.dirname(APP), 'react-ui', 'src', 'components')


def _slider_source():
    with open(os.path.join(UI, 'ui.jsx'), encoding='utf-8') as fh:
        src = fh.read()
    start = src.index('export const Slider =')
    end = src.index('export const Toggle =')
    return src[start:end]


class SliderCommitTest(unittest.TestCase):
    def setUp(self):
        self.src = _slider_source()

    def test_input_does_not_commit_directly(self):
        # The exact shape that regressed: onChange={(e) => onChange(...)} on the
        # range input, i.e. a global state write per pointer move.
        self.assertNotRegex(
            self.src,
            r'onChange=\{\(e\)\s*=>\s*onChange\(',
            'the range input commits to global state on every pointer move; '
            'drive the thumb from local state and commit on a debounce instead')

    def test_thumb_is_driven_by_local_state(self):
        self.assertRegex(self.src, r'useState\(',
                         'the thumb needs local state to stay responsive while '
                         'the commit upward is deferred')
        self.assertRegex(
            self.src, r'value=\{local\}',
            'the input must render the local value, not the prop, or the thumb '
            'will snap back to the last committed value mid-drag')

    def test_a_release_path_flushes_immediately(self):
        # Without this a drag's final value only lands after the debounce, which
        # is felt as lag on exactly the interaction the fix is meant to improve.
        self.assertTrue(
            any(handler in self.src
                for handler in ('onPointerUp', 'onMouseUp', 'onPointerCancel')),
            'no release handler: the committed value would always trail the '
            'drag by the debounce interval')

    def test_keyboard_and_blur_are_covered(self):
        # Arrow keys on a focused range fire no pointer event at all, so the
        # release path alone would leave keyboard edits to the debounce only,
        # and a blur mid-edit could drop the value entirely.
        for handler in ('onKeyUp', 'onBlur'):
            self.assertIn(handler, self.src,
                          f'{handler} missing: keyboard-driven slider edits '
                          'have no release event to flush on')

    def test_external_value_changes_still_move_the_thumb(self):
        # Presets, Reset and profile loads all change the prop without any
        # interaction here. If the component ignored the prop, those controls
        # would silently stop working -- a control that looks wired and is not.
        self.assertRegex(
            self.src, r'useEffect\(\s*\(\)\s*=>\s*\{[^}]*setLocal',
            'the component never follows its own `value` prop, so a preset or '
            'Reset would not move the thumb')
        self.assertIn(
            '}, [value]);', self.src,
            'the follow effect must be keyed on `value`')

    def test_debounce_interval_is_short_enough_to_feel_immediate(self):
        match = re.search(r'SLIDER_COMMIT_MS\s*=\s*(\d+)', self.src) \
            or re.search(r'SLIDER_COMMIT_MS\s*=\s*(\d+)', _whole_file())
        self.assertIsNotNone(match, 'SLIDER_COMMIT_MS is not defined')
        self.assertLessEqual(int(match.group(1)), 150,
                             'the commit debounce is long enough to be felt as '
                             'lag on keyboard-driven edits')


def _whole_file():
    with open(os.path.join(UI, 'ui.jsx'), encoding='utf-8') as fh:
        return fh.read()


if __name__ == '__main__':
    unittest.main()
