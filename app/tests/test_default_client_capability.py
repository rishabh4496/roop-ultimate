"""The default client must be able to START A JOB, not merely render.

WHY THIS EXISTS.  React UI 2.0 was promoted to the production default and
passed every acceptance row that was run against it: 22 of 22 real-browser
checks, 7 themes, 44 controls with no unlabelled control and no page error,
zero horizontal overflow, and a 29-of-29 end-to-end runtime lifecycle.

It could not capture a face, add a source, add a target, open the faceset
library or scrub a timeline.

Both instruments missed it for the same reason, and it is the reason this file
is here:

  * the browser acceptance graded that controls RENDER and carry accessible
    names -- a client with no capture control at all has no unlabelled capture
    control, so it scores perfectly;
  * the end-to-end lifecycle drove the FastAPI boundary DIRECTLY, loading the
    faceset and target itself before asking for a render, so it proved the
    backend worked and never touched the client under test.

A UI is not validated by whether its widgets mount. It is validated by whether
a user can complete the workflow IN IT. This asserts the intake half of that
mechanically: the client `start.js` promotes must reference the routes without
which a job cannot be created from a cold start.

This is deliberately a floor, not a parity check. It says nothing about whether
V2 should eventually be the default -- only that a client which cannot take in
media or pick a face must not be what launches by default.
"""

import os
import re
import unittest

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Without these a user cannot get from a cold start to a render: no media in,
# no face chosen, no frame range. Each was absent from V2 when it shipped as
# the default.
REQUIRED_ROUTES = {
    "/api/source/add": "add a source image",
    "/api/target/add_path": "add target media",
    "/api/target/use_face": "choose which detected face to swap",
    "/api/target/set_frame": "bound the frame range / scrub",
}


def _read(name):
    with open(os.path.join(_ROOT, name), encoding="utf-8") as handle:
        return handle.read()


def _client_dir_of_default():
    """Which client `start.js` actually promotes, read from the launcher."""
    start = _read("start.js")
    match = re.search(r"require\('\./(start_react(?:_v2)?)\.js'\)", start)
    assert match, "start.js must re-export one of the React launchers"
    launcher = _read(match.group(1) + ".js")
    path = re.search(r'path:\s*"(react-ui(?:-v2)?)"', launcher)
    assert path, "the promoted launcher must name its client directory"
    return path.group(1)


def _routes_referenced(client_dir):
    root = os.path.join(_ROOT, client_dir, "src")
    found = set()
    for base, _dirs, files in os.walk(root):
        for name in files:
            if not name.endswith((".js", ".jsx")):
                continue
            with open(os.path.join(base, name), encoding="utf-8") as handle:
                found.update(re.findall(r"/api/[a-zA-Z0-9_/-]+", handle.read()))
    return found


class DefaultClientCapabilityTests(unittest.TestCase):
    def test_the_default_client_can_take_in_media_and_pick_a_face(self):
        client = _client_dir_of_default()
        referenced = _routes_referenced(client)
        self.assertTrue(referenced, f"{client} referenced no API routes at all")
        missing = {r: why for r, why in REQUIRED_ROUTES.items()
                   if r not in referenced}
        self.assertEqual(
            {}, missing,
            f"{client} is the default client but cannot: "
            + "; ".join(f"{why} ({route})" for route, why in sorted(missing.items()))
            + ". A client that cannot create a job from a cold start must not be "
              "what start.js promotes.")

    def test_the_guard_reads_the_launcher_rather_than_a_hardcoded_name(self):
        """It must follow start.js, or flipping the default silently skips it.

        The regex in `_client_dir_of_default` still accepts a `_v2` launcher on
        purpose: the point of this guard is that it FOLLOWS start.js instead of
        naming a directory, and it has to keep working if a second client is
        ever introduced again.
        """
        self.assertTrue(os.path.isdir(
            os.path.join(_ROOT, _client_dir_of_default(), "src")))

    # ── React UI 2.0, second edition (2026-11-28) ─────────────────────────────
    # The first React UI 2.0 was a rewrite and was removed (docs/development/
    # UI_V1_V2_MIGRATION_AUDIT.md); this file used to fail if any of it came
    # back. It came back as something different, recorded in
    # docs/UI_V2_VISUAL_AUDIT.md: 1.0's client COPIED, plus a visual layer
    # (react-ui-v2/src/v2fx/), served at /v2/ beside 1.0. These replace the
    # "it is gone" assertion with the three things that make the new one safe.

    def test_1_0_is_still_what_start_js_promotes(self):
        """2.0 is opt-in at /v2/. The default client is unchanged."""
        self.assertEqual(_client_dir_of_default(), "react-ui")
        self.assertFalse(os.path.exists(os.path.join(_ROOT, "start_react_v2.js")))
        self.assertNotIn("react-ui-v2", _read("start.js"))

    def test_2_0_can_take_in_media_and_pick_a_face(self):
        """The floor the first 2.0 failed: the routes without which no job can be made."""
        if not os.path.isdir(os.path.join(_ROOT, "react-ui-v2", "src")):
            self.skipTest("react-ui-v2 is not present")
        referenced = _routes_referenced("react-ui-v2")
        missing = {r: why for r, why in REQUIRED_ROUTES.items() if r not in referenced}
        self.assertEqual({}, missing, "react-ui-v2 cannot: " + "; ".join(
            f"{why} ({route})" for route, why in sorted(missing.items())))

    def test_2_0_is_1_0_plus_a_visual_layer(self):
        """Every 1.0 file is in 2.0 byte for byte, except the short OWNED list
        2.0 edits to mount its visual layer - and those are flagged the moment
        1.0 changes them. A 1.0 change that is not carried into 2.0 fails here:
        run `python scripts/sync_react_ui_v2.py`."""
        if not os.path.isdir(os.path.join(_ROOT, "react-ui-v2", "src")):
            self.skipTest("react-ui-v2 is not present")
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "sync_react_ui_v2", os.path.join(_ROOT, "scripts", "sync_react_ui_v2.py"))
        sync = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(sync)
        report = sync.audit()
        drift = {k: v for k, v in report.items() if v}
        self.assertEqual({}, drift,
                         "React UI 2.0 has drifted from 1.0 + its visual layer. Run "
                         "`python scripts/sync_react_ui_v2.py` (copies 1.0 changes, lists "
                         "owned files to merge by hand): " + repr(drift))


if __name__ == "__main__":
    unittest.main()
