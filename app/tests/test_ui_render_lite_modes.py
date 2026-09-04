"""Lite UI must be reachable, and must survive a tab switch.

`data-render-lite` strips the 59 backdrop-filter panels and the three keyframe
animations that repaint rather than composite. index.css documents that as the
dominant cost this window imposes on the GPU -- and it is paid while IDLE too,
because the blurs are re-composited on every state change, not just during a
run.

Two things were wrong with how it was wired:

  * it only ever applied WHILE A RENDER WAS RUNNING, and its only control lived
    in the Processing tab's dock -- reachable only during a run, which is when
    you are least likely to go looking for a UI-performance setting;
  * the effect removed the attribute on cleanup unconditionally, while three
    components now mount the hook (App, Processing, Face Swap). Leaving the Face
    Swap tab mid-render ran its cleanup and stripped the attribute out from under
    the others, whose effects do not re-run -- so the setting silently stopped
    applying at the first tab change.

The default appearance is deliberately unchanged: 'auto' is the default and
behaves exactly as the old boolean did.
"""

import os
import re
import unittest

APP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UI = os.path.join(os.path.dirname(APP), 'react-ui', 'src')


def _read(*parts):
    with open(os.path.join(UI, *parts), encoding='utf-8') as fh:
        return fh.read()


class RenderLiteModesTest(unittest.TestCase):
    def setUp(self):
        self.hook = _read('components', 'faceswap', 'useRenderLite.js')

    def test_three_modes_exist(self):
        match = re.search(r"RENDER_LITE_MODES\s*=\s*\[([^\]]*)\]", self.hook)
        self.assertIsNotNone(match, 'RENDER_LITE_MODES is not defined')
        modes = re.findall(r"'([a-z]+)'", match.group(1))
        self.assertEqual(modes, ['auto', 'always', 'off'],
                         'auto must be first so it is the default')

    def test_default_is_auto_so_the_idle_look_is_unchanged(self):
        # The whole point of adding 'always' rather than just flipping the
        # switch on: nobody's default appearance changes.
        self.assertIn("return 'auto';", self.hook)
        self.assertRegex(self.hook, r"raw === null\) return 'auto'",
                         'a fresh install must land on auto')

    def test_legacy_boolean_preference_is_honoured(self):
        # The key used to hold '0'/'1'. Reading those as an unknown value would
        # silently reset everyone who had turned it off.
        self.assertRegex(self.hook, r"raw === '0'\) return 'off'")
        self.assertRegex(self.hook, r"raw === '1'")

    def test_always_mode_applies_without_a_run(self):
        self.assertRegex(
            self.hook, r"current === 'always' \|\| \(current === 'auto' && processing\)",
            "'always' must not be gated on `processing`, or it is just 'auto'")

    def test_cleanup_is_refcounted_not_unconditional(self):
        # The tab-switch defect: three components mount this hook and share one
        # attribute, so an unmounting one must not be able to strip it.
        self.assertIn('liteWanters', self.hook,
                      'no refcount: one unmounting mounter can strip the '
                      'attribute out from under the others')
        self.assertNotRegex(
            self.hook,
            r"return \(\) => \{?\s*(?:if [^\n]*\n\s*)?el\.removeAttribute\('data-render-lite'\)",
            'cleanup removes the shared attribute unconditionally')

    def test_state_is_shared_not_per_component(self):
        # Two surfaces read it (the dock button and App's command palette) and
        # they are in different trees. Per-component state would leave one of
        # them reporting a mode the other is not in.
        self.assertIn('useSyncExternalStore', self.hook,
                      'the preference must be a shared store, not local state')

    def test_it_is_reachable_from_anywhere(self):
        app = _read('App.jsx')
        self.assertIn('cycleRenderLiteMode', app,
                      'Lite UI has no command-palette entry, so it is still '
                      'only reachable from the Processing dock during a run')
        self.assertIn('useRenderLite', app,
                      'App must mount the hook, or "always" would stop '
                      'applying on tabs that do not mount it themselves')


if __name__ == '__main__':
    unittest.main()
