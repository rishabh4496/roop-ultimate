"""The launcher must open V2 by default and must never lose the V1 action.

WHY THIS EXISTS.  React UI 2.0 is now the production client: `start.js`
re-exports `start_react_v2.js` and the Pinokio menu's default action starts V2.
The explicit, standing requirement alongside that is that React UI 1.0 stays
present and launchable -- it still owns feature surfaces V2 does not provide.

"V1 is still on disk" is not the same as "a user can start V1".  A menu edit
that dropped the V1 entry would leave the tree intact and the fallback
unreachable, and nothing else in this suite would notice.  These are text-level
assertions on the launcher because Pinokio scripts are configuration: there is
no Python surface to exercise, and the failure mode is a missing entry rather
than a wrong return value.
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
        self.assertIn("require('./start_react_v2.js')", _read("start.js"))

    def test_both_launch_scripts_still_exist(self):
        for name in ("start_react.js", "start_react_v2.js", "start_legacy.js"):
            self.assertTrue(os.path.isfile(os.path.join(_ROOT, name)), name)

    def test_v1_launcher_still_targets_the_v1_client(self):
        v1 = _read("start_react.js")
        self.assertIn('path: "react-ui"', v1)
        self.assertNotIn("react-ui-v2", v1)

    def test_v2_launcher_targets_the_v2_client(self):
        self.assertIn('path: "react-ui-v2"', _read("start_react_v2.js"))

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
        self.assertIn("start_react_v2.js", idle[first:first + 60],
                      "the default idle action must start the production client")
        self.assertIn("default: true", idle[:first])

    def test_the_idle_menu_still_offers_the_other_client(self):
        """Both clients must be reachable from the menu, not just present."""
        self.assertIn('href: "start_react.js"', self._idle_menu())

    def test_every_launcher_branch_keeps_a_route_back_to_v1(self):
        """While V2 runs, the menu must still offer the V1 action."""
        menu = _read("pinokio.js")
        self.assertIn("start_v1_item", menu)
        # Both V2 branches (with and without a captured URL) spread it in.
        self.assertGreaterEqual(menu.count("...start_v1_item"), 2)

    def test_pinokio_resolves_a_running_start_js_as_the_client_it_launches(self):
        """The menu's branch detection must agree with what start.js re-exports.

        `start.js` is a thin re-export, so a running `start.js` has to be
        resolved onto one of the two clients for both `info.local()` and the
        Terminal href. That mapping lives in `pinokio.js` and is easy to leave
        behind: when the default was rolled back to V1 and this line was not,
        a running V1 process was still resolved as V2 and the menu offered a
        "Terminal - React UI 2.0" for it.
        """
        start = _read("start.js")
        menu = _read("pinokio.js")
        v1_is_default = "require('./start_react.js')" in start
        # The variable that ALSO accepts "start.js" is the one start.js feeds.
        v1_takes_start_js = re.search(
            r'start_react_script\s*=.*?info\.running\("start\.js"\)',
            menu, re.S) is not None
        v2_takes_start_js = re.search(
            r'start_react_v2_script\s*=.*?info\.running\("start\.js"\)',
            menu, re.S) is not None
        self.assertNotEqual(v1_takes_start_js, v2_takes_start_js,
                            "exactly one client may claim a running start.js")
        self.assertEqual(v1_is_default, v1_takes_start_js,
                         "pinokio.js resolves a running start.js onto a "
                         "different client than start.js actually launches")

    def test_install_and_reset_cover_both_clients(self):
        install, reset = _read("install.js"), _read("reset.js")
        for name in ("react-ui", "react-ui-v2"):
            self.assertIn(name, install)
            self.assertIn(name, reset)

    def test_the_url_capture_pattern_is_intact_on_both_launchers(self):
        """The project's Pinokio contract: capture the URL, set it via local.set."""
        for name in ("start_react.js", "start_react_v2.js"):
            text = _read(name)
            self.assertTrue(re.search(r'"event":\s*"/\(http', text), name)
            self.assertIn('url: "{{input.event[1]}}"', text, name)
            self.assertIn('method: "local.set"', text, name)


if __name__ == "__main__":
    unittest.main()
