"""The launcher must open the React client, and start.js must resolve onto it.

WHY THIS EXISTS.  React UI 1.0 is the sole production client: `start.js`
re-exports `start_react.js` and the Pinokio menu's default action starts it.
React UI 2.0 was removed after its seven unique capabilities were migrated
into V1 and verified -- see docs/development/UI_V1_V2_MIGRATION_AUDIT.md, and
`test_default_client_capability.py` for the guard that it stays removed.

The assertions that survived the two-client era are the ones that were never
about which client won:

  * the Pinokio menu's default action must actually start something, and it
    must be the client `start.js` re-exports -- these drifted apart once
    already, and the menu then offered a Terminal for the wrong process;
  * `pinokio.js` must claim a running `start.js`, or the Terminal button
    launches a SECOND copy of the whole stack instead of showing the running
    one;
  * the URL-capture pattern (regex event -> `local.set`) is this project's
    Pinokio contract and is what surfaces the "Open" action at all.

These are text-level assertions because Pinokio scripts are configuration:
there is no Python surface to exercise, and the failure mode is a missing
entry rather than a wrong return value.
"""

import os
import re
import unittest

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _read(name):
    with open(os.path.join(_ROOT, name), encoding="utf-8") as handle:
        return handle.read()


class LauncherActivationTests(unittest.TestCase):
    def test_start_js_launches_the_production_client(self):
        self.assertIn("require('./start_react.js')", _read("start.js"))

    def test_the_launch_scripts_still_exist(self):
        for name in ("start_react.js", "start_legacy.js"):
            self.assertTrue(os.path.isfile(os.path.join(_ROOT, name)), name)

    def test_the_launcher_targets_the_react_client(self):
        self.assertIn('path: "react-ui"', _read("start_react.js"))

    @staticmethod
    def _idle_menu():
        """The installed-and-nothing-running branch.

        Anchored on the idle menu's own 'Start Legacy UI' LABEL, which is
        unique to that branch -- the running-legacy branch says 'Open Legacy
        UI' and reaches start_legacy.js by href only.  A
        naive last-`} else {` slice lands on the NOT-installed branch (whose
        only action is Install) and would pass or fail for the wrong reason.
        """
        menu = _read("pinokio.js")
        anchor = menu.index('text: "Start Legacy UI"')
        start = menu.rindex("return [{", 0, anchor)
        return menu[start:menu.index("]", anchor)]

    def test_the_default_menu_action_starts_the_production_client(self):
        idle = self._idle_menu()
        first = idle.index("href:")
        self.assertIn("start_react.js", idle[first:first + 60],
                      "the default idle action must start the production client")
        self.assertIn("default: true", idle[:first])

    def test_pinokio_resolves_a_running_start_js_as_the_client_it_launches(self):
        """The menu's branch detection must agree with what start.js re-exports.

        `start.js` is a thin re-export, so a running `start.js` has to be
        resolved onto the client it launches for both `info.local()` and the
        Terminal href. That mapping lives in `pinokio.js` and is easy to leave
        behind: while two clients existed, rolling the default back to V1
        without updating this line meant a running V1 process was still
        resolved as V2 and the menu offered a "Terminal - React UI 2.0" for it.
        There is one client now, so the assertion is that the resolver claims a
        running start.js at all -- without it, the Terminal button starts a
        SECOND copy of the whole stack instead of showing the running one.
        """
        self.assertIn("require('./start_react.js')", _read("start.js"))
        self.assertIsNotNone(
            re.search(r'start_react_script\s*=.*?info\.running\("start\.js"\)',
                      _read("pinokio.js"), re.S),
            "pinokio.js must resolve a running start.js onto the React client")

    def test_install_and_reset_cover_the_client(self):
        self.assertIn("react-ui", _read("install.js"))
        self.assertIn("react-ui", _read("reset.js"))

    def test_the_url_capture_pattern_is_intact(self):
        """The project's Pinokio contract: capture the URL, set it via local.set."""
        text = _read("start_react.js")
        self.assertTrue(re.search(r'"event":\s*"/\(http', text))
        self.assertIn('url: "{{input.event[1]}}"', text)
        self.assertIn('method: "local.set"', text)


if __name__ == "__main__":
    unittest.main()
