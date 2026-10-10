"""Every long-running API call in the React UI opts out of the 15 s deadline, explicitly.

react-ui/src/api.js gives each getJSON / postJSON / postFile(s) call 15 seconds unless it
passes `timeout: 0`. The default used to be NO deadline (opt in per call), so most calls had
none and a backend that stalled left the UI waiting in silence. Inverting it makes the
opposite mistake possible: a 4 GB upload, a cold TensorRT preview or an ffmpeg join cut off
at 15 s, reported as a failure while the server is still working.

Which endpoints may outrun the default is decided once, with a reason each, in
react-ui/src/longRunning.js. `react-ui/scripts/api-callsites.mjs` (`npm run
lint:api-timeouts`, part of `npm run check`) enumerates EVERY call site and fails on:

    missing-opt-out     a listed long-running endpoint called without `timeout: 0`
    unlisted-opt-out    `timeout: 0` on an endpoint that is not listed
    dynamic-no-timeout  a computed path with no `timeoutFor(...)` / `timeout:` (unclassifiable)
    stale-entry         a registry entry no call site references

This runs that scan from the Python suite, checks the registry against the backend's real
routes (a typo'd entry would silently never match), and runs the scanner against a fixture
tree, because a scanner that reports nothing is worthless unless it is known to fire.
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import test_ui_api_surface as surface  # noqa: E402  (its route parser is the one source of backend routes)

ROOT = Path(__file__).resolve().parents[2]
UI = ROOT / "react-ui"
SCRIPT = UI / "scripts" / "api-callsites.mjs"
REGISTRY = UI / "src" / "longRunning.js"


def _node():
    found = shutil.which("node")
    if found:
        return found
    # <PINOKIO_HOME>/api/roop-ultimate is this repo; Pinokio's own node lives under bin/.
    home = ROOT.parents[1]
    for cand in (home / "bin" / "miniconda" / "node.exe", home / "bin" / "miniforge" / "node.exe"):
        if cand.exists():
            return str(cand)
    return None


def _scan(script):
    proc = subprocess.run([_node(), str(script), "--json"], capture_output=True, text=True, timeout=60)
    if proc.returncode not in (0, 1):
        raise AssertionError(f"scan crashed ({proc.returncode}): {proc.stderr}")
    return proc.returncode, json.loads(proc.stdout)


@unittest.skipUnless(_node(), "node is not available")
class ApiTimeouts(unittest.TestCase):
    def test_every_call_site_states_its_deadline(self):
        code, report = _scan(SCRIPT)
        self.assertGreater(len(report["sites"]), 120, "the scan found almost no call sites; it has drifted from the source")
        lines = [f"{v['rule']}: {v.get('file')}:{v.get('line')} {v.get('path') or v.get('key') or ''}"
                 for v in report["violations"]]
        self.assertEqual(report["violations"], [],
                         "API calls whose timeout policy is wrong (see react-ui/src/longRunning.js):\n  " + "\n  ".join(lines))
        self.assertEqual(code, 0)

    def test_the_scan_sees_the_calls_that_matter(self):
        """If these were not found, a pass would mean nothing."""
        _code, report = _scan(SCRIPT)
        seen = {(s["method"], s["path"]) for s in report["sites"] if s["kind"] == "literal"}
        for key in [("POST", "/api/swap"), ("POST", "/api/target/auto_capture"), ("POST", "/api/preview_upscale"),
                    ("POST", "/api/source/add"), ("POST", "/api/target/add"), ("GET", "/api/state")]:
            self.assertIn(key, seen, f"{key} is not among the enumerated call sites")
        opt_out = {(s["method"], s["path"]) for s in report["sites"] if s.get("optOut")}
        for key in [("POST", "/api/swap"), ("POST", "/api/target/auto_capture"), ("POST", "/api/preview_upscale"),
                    ("POST", "/api/source/add"), ("POST", "/api/target/add")]:
            self.assertIn(key, opt_out, f"{key} must carry an explicit `timeout: 0`")

    def test_every_registry_entry_is_a_real_backend_route(self):
        node = _node()
        out = subprocess.run(
            [node, "-e", f"import('{REGISTRY.as_uri()}').then(m => console.log(JSON.stringify(Object.keys(m.LONG_RUNNING))))"],
            capture_output=True, text=True, timeout=30)
        self.assertEqual(out.returncode, 0, out.stderr)
        keys = json.loads(out.stdout)
        self.assertGreater(len(keys), 20)
        routes = surface.backend_routes()
        missing = [k for k in keys if not surface._matches(surface._norm(k.split(" ", 1)[1]), routes)]
        self.assertEqual(missing, [], "registry entries that name no backend route (a typo never matches anything): " + ", ".join(missing))

    def test_the_scanner_fires_on_each_kind_of_violation(self):
        """Run the real script against a tree it can be wrong about."""
        with tempfile.TemporaryDirectory() as tmp:
            pkg = Path(tmp) / "react-ui"
            (pkg / "scripts").mkdir(parents=True)
            (pkg / "src").mkdir(parents=True)
            shutil.copy(SCRIPT, pkg / "scripts" / "api-callsites.mjs")
            (pkg / "src" / "longRunning.js").write_text(
                "export const LONG_RUNNING = {\n"
                "  'POST /api/slow': 'does a lot of work',\n"
                "  'POST /api/upload': 'an upload',\n"
                "  'POST /api/ghost': 'nothing calls this',\n"
                "  'POST /api/viawrapper': 'only reached through a wrapper',\n"
                "};\n"
                "const re = (p) => new RegExp('^' + p.replace(/\\{\\}/g, '[^/]+') + '$');\n"
                "const pats = Object.keys(LONG_RUNNING).map((k) => ({ k, m: k.split(' ')[0], re: re(k.split(' ')[1]) }));\n"
                "export const longRunningKey = (m, p) => (pats.find((x) => x.m === m && x.re.test(String(p).split('?')[0])) || {}).k || null;\n",
                encoding="utf-8")
            (pkg / "src" / "good.js").write_text(
                "import { getJSON, postJSON, postFile, timeoutFor } from './api';\n"
                "export const a = () => postJSON('/api/slow', {}, { timeout: 0 });\n"
                "export const b = () => postFile('/api/upload', f, undefined, { timeout: 0, signal });\n"
                "export const c = () => getJSON('/api/quick?x=1');\n"
                "export const d = () => getJSON('/api/poll', { timeout: 5000 });\n"
                "export const e = (path) => postJSON(path, {}, { timeout: timeoutFor('POST', path) });\n"
                "export const f2 = () => postJSON(`/api/update/check${q}`, {});\n"
                "const wrapped = (p) => postJSON(p, {}, { timeout: timeoutFor('POST', p) });\n"
                "wrapped('/api/viawrapper');\n"
                "// postJSON('/api/slow', {}) <- a call inside a comment is not a call\n"
                "const note = 'postJSON(\"/api/slow\")';\n", encoding="utf-8")
            code, report = _scan(pkg / "scripts" / "api-callsites.mjs")
            self.assertEqual([(v["rule"], v.get("key") or v.get("path")) for v in report["violations"]],
                             [("stale-entry", "POST /api/ghost")],
                             "the good file must pass, and only the unused registry entry is reported")
            self.assertEqual(code, 1)

            (pkg / "src" / "bad.js").write_text(
                "import { getJSON, postJSON, postFiles } from './api';\n"
                "export const a = () => postJSON('/api/slow', { go: 1 });\n"                        # missing opt-out
                "export const b = () => postFiles('/api/upload', files);\n"                          # missing opt-out (upload, no opts at all)
                "export const c = () => postJSON('/api/slow', {}, { signal });\n"                    # opts present, still no timeout:0
                "export const d = () => postJSON('/api/slow', {}, { timeout: 30000 });\n"             # a finite deadline is not an opt-out
                "export const e = () => getJSON('/api/quick', { timeout: 0 });\n"                    # unlisted opt-out
                "export const f = (path) => postJSON(path, {});\n"                                   # computed, silent
                "export const g = (id) => postJSON(`/api/things/${id}/go`, {});\n"                   # templated segment, silent
                "export const h = () => postJSON('/api/quick', {}, { timeout: 0 });\n",              # unlisted opt-out
                encoding="utf-8")
            code, report = _scan(pkg / "scripts" / "api-callsites.mjs")
            found = sorted((v["rule"], v["line"]) for v in report["violations"] if v["file"] == "bad.js")
            self.assertEqual(found, [
                ("dynamic-no-timeout", 7), ("dynamic-no-timeout", 8),
                ("missing-opt-out", 2), ("missing-opt-out", 3), ("missing-opt-out", 4), ("missing-opt-out", 5),
                ("unlisted-opt-out", 6), ("unlisted-opt-out", 9)])
            self.assertEqual(code, 1)


if __name__ == "__main__":
    unittest.main()
