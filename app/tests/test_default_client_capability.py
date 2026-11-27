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


# React UI 2.0's must-haves beyond job intake (2026-11-28 brief): each is a
# screen or control the user asked for by name.
V2_MUST_HAVE = {
    "/api/preview": "show the swap preview",
    "/api/target/preview": "draw timeline frames",
    "/api/target/timeline_tracks": "draw the timeline's presence lanes",
    "/api/swap": "start a render",
    "/api/pause": "pause a render",
    "/api/resume": "resume a render",
    "/api/stop": "stop a render",
    "/api/queue/add": "queue a batch job",
    "/api/queue/start": "run the batch queue",
    "/api/faceset/library": "open the faceset library",
    "/api/faceset/library/load": "load a faceset into the cast",
    "/api/models/integrity": "show the models on disk",
}

_IMPORT_RE = re.compile(r"""(?:import|export)\s[^'"]*?from\s*['"]([^'"]+)['"]|import\s*\(\s*['"]([^'"]+)['"]\s*\)|import\s+['"]([^'"]+)['"]""")


def _resolve_v2_import(spec, here):
    """A react-ui-v2 import specifier as a file, following its vite aliases."""
    if spec.startswith("@engine/"):
        base = os.path.join(_ROOT, "react-ui", "src", spec[len("@engine/"):])
    elif spec.startswith("."):
        base = os.path.normpath(os.path.join(os.path.dirname(here), spec))
    else:
        return None  # a package, or @schema (data)
    for cand in (base, base + ".js", base + ".jsx", base + ".ts", base + ".tsx",
                 os.path.join(base, "index.js")):
        if os.path.isfile(cand):
            return cand
    return None


def _v2_reachable_files():
    """Every source file reachable from react-ui-v2/src/main.jsx."""
    start = os.path.join(_ROOT, "react-ui-v2", "src", "main.jsx")
    seen, todo = set(), [start]
    while todo:
        path = todo.pop()
        if path in seen or not path.endswith((".js", ".jsx", ".ts", ".tsx")):
            seen.add(path)
            continue
        seen.add(path)
        with open(path, encoding="utf-8") as handle:
            for match in _IMPORT_RE.finditer(handle.read()):
                spec = next(g for g in match.groups() if g)
                target = _resolve_v2_import(spec, path)
                if target and target not in seen:
                    todo.append(target)
    return sorted(seen)


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

    # ── React UI 2.0 (2026-11-28) ──────────────────────────────────────────
    # The first React UI 2.0 was removed (docs/development/UI_V1_V2_MIGRATION_AUDIT.md):
    # it re-derived the engine contract, passed every acceptance row, and could
    # not create a job - its "parity" was adapter code no screen imported. The
    # current one (docs/UI_V2_VISUAL_AUDIT.md) has its own screens and NO 1.0
    # visuals, and talks to the engine only through 1.0's LOGIC modules
    # (imported via the `@engine` alias, not copied). These hold it to that, and
    # read routes only from modules REACHABLE from its entry point, so code that
    # nothing imports cannot satisfy them.

    def test_1_0_is_still_what_start_js_promotes(self):
        """2.0 is opt-in at /v2/. The default client is unchanged."""
        self.assertEqual(_client_dir_of_default(), "react-ui")
        self.assertFalse(os.path.exists(os.path.join(_ROOT, "start_react_v2.js")))
        self.assertNotIn("react-ui-v2", _read("start.js"))

    def _v2_reachable(self):
        if not os.path.isdir(os.path.join(_ROOT, "react-ui-v2", "src")):
            self.skipTest("react-ui-v2 is not present")
        return _v2_reachable_files()

    def test_2_0_reaches_every_route_a_job_needs(self):
        """The floor the first 2.0 failed, plus the workstation's must-haves
        (preview, timeline, render controls, batch, faceset library, models)."""
        files = self._v2_reachable()
        found = set()
        for path in files:
            with open(path, encoding="utf-8") as handle:
                found.update(re.findall(r"/api/[a-zA-Z0-9_/-]+", handle.read()))
        missing = {r: why for r, why in {**REQUIRED_ROUTES, **V2_MUST_HAVE}.items() if r not in found}
        self.assertEqual({}, missing, "react-ui-v2 cannot: " + "; ".join(
            f"{why} ({route})" for route, why in sorted(missing.items())))

    def test_2_0_imports_no_1_0_visuals(self):
        """Only 1.0's logic crosses over: no 1.0 component (.jsx/.tsx) and no 1.0 stylesheet."""
        engine = os.path.normcase(os.path.join(_ROOT, "react-ui", "src"))
        crossed = [p for p in self._v2_reachable() if os.path.normcase(p).startswith(engine)]
        self.assertTrue(crossed, "2.0 should reach the engine through 1.0's logic modules")
        visual = [os.path.relpath(p, _ROOT) for p in crossed if p.endswith((".jsx", ".tsx", ".css"))]
        self.assertEqual([], visual, "react-ui-v2 imports 1.0 visuals")

if __name__ == "__main__":
    unittest.main()
